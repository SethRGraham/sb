"""Falsification tests for chain-induced hard-cost mollification."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from schrodinger_bridge.solvers.mollified_gradient import (
    GaussianHalfspaceCost,
    MollifiedGradientBackend,
    MollifiedGradientConfig,
    MollifiedGradientEstimator,
    mollified_gradient_backend_info,
)


def _dtype():
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


def _halfspace_problem():
    dtype = _dtype()
    means = jnp.asarray([[0.1, -0.2], [0.45, 0.15]], dtype=dtype)
    # Deliberately nonsymmetric: solve(L.T, xi), not solve(L, xi), is required.
    factor = jnp.asarray([[0.8, 0.35], [-0.2, 0.55]], dtype=dtype)
    cost = GaussianHalfspaceCost(
        normal=jnp.asarray([0.7, -1.1], dtype=dtype),
        threshold=0.2,
        high_value=1.7,
        low_value=-0.3,
    )
    return means, factor, cost


def test_analytic_halfspace_matches_closed_form_and_has_no_oracle_queries():
    means, factor, cost = _halfspace_problem()
    estimator = MollifiedGradientEstimator(MollifiedGradientConfig(backend="analytic"))
    result = estimator.estimate(
        jax.random.PRNGKey(0),
        cost,
        means,
        factor,
        time=jnp.asarray([0.2, 0.7], dtype=means.dtype),
        context=None,
    )

    normal = np.asarray(cost.normal)
    projected_scale = np.linalg.norm(np.asarray(factor).T @ normal)
    z = (np.asarray(means) @ normal - cost.threshold) / projected_scale
    expected = (
        (cost.high_value - cost.low_value)
        * np.exp(-0.5 * z**2)[:, None]
        / np.sqrt(2.0 * np.pi)
        / projected_scale
        * normal[None, :]
    )
    np.testing.assert_allclose(np.asarray(result.gradient), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(np.asarray(result.standard_error), np.zeros_like(expected))
    assert int(result.physical_query_count) == 0
    assert bool(jnp.all(result.finite))
    assert int(result.backend_code) == MollifiedGradientBackend.ANALYTIC
    assert bool(result.exact)


@pytest.mark.parametrize(
    ("backend", "num_samples", "seed", "rtol", "atol"),
    [
        ("antithetic_stein", 131_072, 17, 0.035, 0.008),
        ("iid_stein", 131_072, 31, 0.045, 0.010),
    ],
)
def test_stein_estimators_match_nonsymmetric_gaussian_halfspace_truth(
    backend, num_samples, seed, rtol, atol
):
    means, factor, cost = _halfspace_problem()
    truth = MollifiedGradientEstimator(MollifiedGradientConfig(backend="analytic")).estimate(
        jax.random.PRNGKey(0), cost, means, factor, 0.4, None
    )
    result = MollifiedGradientEstimator(
        MollifiedGradientConfig(backend=backend, num_samples=num_samples)
    ).estimate(jax.random.PRNGKey(seed), cost, means, factor, 0.4, None)

    np.testing.assert_allclose(
        np.asarray(result.gradient),
        np.asarray(truth.gradient),
        rtol=rtol,
        atol=atol,
    )
    assert bool(jnp.all(result.finite))
    assert bool(result.standard_error_available)
    expected_queries = means.shape[0] * num_samples * (2 if backend == "antithetic_stein" else 1)
    assert int(result.physical_query_count) == expected_queries


def test_iid_centering_corrects_exact_m_minus_one_over_m_bias_factor():
    """Same-sample mean centering without the Bessel factor is biased."""
    dtype = _dtype()
    batch_size = 32_768
    num_samples = 4
    key = jax.random.PRNGKey(91)
    means = jnp.zeros((batch_size, 1), dtype=dtype)

    def linear_cost(states, times, context):
        del times, context
        return states[:, 0]

    result = MollifiedGradientEstimator(
        MollifiedGradientConfig(backend="iid_stein", num_samples=num_samples)
    ).estimate(key, linear_cost, means, 1.0, 0.0, None)

    # Recreate the same draws and the commonly used, but biased, estimator.
    xi = jax.random.normal(key, (batch_size, num_samples, 1), dtype=dtype)
    values = xi[..., 0]
    naive = jnp.mean((values - jnp.mean(values, axis=1, keepdims=True))[..., None] * xi, axis=1)

    np.testing.assert_allclose(
        np.asarray(result.gradient),
        np.asarray(naive * num_samples / (num_samples - 1)),
        rtol=2e-6,
        atol=2e-6,
    )
    assert abs(float(jnp.mean(result.gradient)) - 1.0) < 0.015
    assert abs(float(jnp.mean(naive)) - (num_samples - 1) / num_samples) < 0.015


@pytest.mark.parametrize(
    "factor",
    [
        0.7,
        jnp.asarray([0.7, 0.4]),
        jnp.asarray([[0.7, 0.1], [-0.2, 0.4]]),
        jnp.asarray([[[0.7, 0.1], [-0.2, 0.4]], [[0.5, -0.1], [0.2, 0.8]]]),
    ],
)
def test_scalar_diagonal_full_and_batched_factors_have_batched_output(factor):
    dtype = _dtype()
    means = jnp.asarray([[0.1, 0.2], [-0.2, 0.3]], dtype=dtype)
    cost = GaussianHalfspaceCost(jnp.asarray([1.0, -0.4], dtype=dtype), threshold=0.1)
    result = MollifiedGradientEstimator(
        MollifiedGradientConfig(backend="antithetic_stein", num_samples=8)
    ).estimate(jax.random.PRNGKey(4), cost, means, jnp.asarray(factor, dtype=dtype), 0.3, None)
    assert result.gradient.shape == means.shape
    assert result.standard_error.shape == means.shape
    assert result.finite.shape == (means.shape[0],)
    assert bool(jnp.all(result.finite))


def test_estimate_is_jittable_deterministic_and_stops_oracle_values():
    dtype = _dtype()
    estimator = MollifiedGradientEstimator(
        MollifiedGradientConfig(backend="antithetic_stein", num_samples=16)
    )
    key = jax.random.PRNGKey(101)
    mean = jnp.asarray([[0.2], [-0.3]], dtype=dtype)

    def run(run_key, run_mean):
        def cost(states, times, context):
            del times, context
            return (states[:, 0] >= 0.0).astype(states.dtype)

        return estimator.estimate(run_key, cost, run_mean, 0.5, 0.2, None)

    eager = run(key, mean)
    compiled = jax.jit(run)(key, mean)
    np.testing.assert_array_equal(np.asarray(compiled.gradient), np.asarray(eager.gradient))
    np.testing.assert_array_equal(np.asarray(compiled.finite), np.asarray(eager.finite))

    def label_sum(scale):
        def scaled_cost(states, times, context):
            del times, context
            return scale * states[:, 0]

        return jnp.sum(estimator.estimate(key, scaled_cost, mean, 0.5, 0.2, None).gradient)

    assert float(jax.grad(label_sum)(jnp.asarray(1.0, dtype=dtype))) == 0.0


def test_nonfinite_oracle_rows_fail_closed_without_contaminating_other_rows():
    dtype = _dtype()
    means = jnp.asarray([[0.0], [1.0]], dtype=dtype)

    def bad_cost(states, times, context):
        del times, context
        return jnp.where(states[:, 0] < 0.5, jnp.inf, states[:, 0])

    result = MollifiedGradientEstimator(
        MollifiedGradientConfig(backend="antithetic_stein", num_samples=4)
    ).estimate(jax.random.PRNGKey(0), bad_cost, means, 0.01, 0.0, None)
    assert not bool(result.finite[0])
    assert bool(result.finite[1])
    np.testing.assert_array_equal(np.asarray(result.gradient[0]), np.zeros((1,)))
    assert np.isinf(np.asarray(result.standard_error[0])).all()


@pytest.mark.parametrize("backend", ["antithetic_stein", "analytic"])
def test_nonfinite_mean_time_or_context_rows_fail_closed(backend):
    dtype = _dtype()
    means = jnp.asarray([[jnp.nan], [0.0], [0.0]], dtype=dtype)
    times = jnp.asarray([0.2, jnp.nan, 0.4], dtype=dtype)
    context = jnp.asarray([[0.0], [0.0], [jnp.nan]], dtype=dtype)
    cost = GaussianHalfspaceCost(jnp.ones((1,), dtype=dtype))
    result = MollifiedGradientEstimator(
        MollifiedGradientConfig(backend=backend, num_samples=4)
    ).estimate(
        jax.random.PRNGKey(17),
        cost,
        means,
        jnp.asarray([[0.5]], dtype=dtype),
        times,
        context,
    )
    np.testing.assert_array_equal(np.asarray(result.finite), [False, False, False])
    np.testing.assert_array_equal(np.asarray(result.gradient), np.zeros((3, 1)))


def test_invalid_and_unimplemented_backends_fail_loudly():
    with pytest.raises(ValueError, match="backend"):
        MollifiedGradientConfig(backend="mystery")
    with pytest.raises(TypeError, match="backend"):
        MollifiedGradientConfig(backend=1)  # type: ignore[arg-type]
    for invalid_samples in (True, 1.5):
        with pytest.raises(TypeError, match="num_samples"):
            MollifiedGradientConfig(num_samples=invalid_samples)  # type: ignore[arg-type]
    for invalid_floor in (True, float("nan"), float("inf"), 0.0):
        error = TypeError if isinstance(invalid_floor, bool) else ValueError
        with pytest.raises(error, match="min_singular_value"):
            MollifiedGradientConfig(min_singular_value=invalid_floor)
    with pytest.raises(ValueError, match="at least two"):
        MollifiedGradientConfig(backend="iid_stein", num_samples=1)

    means = jnp.zeros((1, 1), dtype=_dtype())
    cost = GaussianHalfspaceCost(jnp.ones((1,), dtype=_dtype()))
    with pytest.raises(ValueError, match="full-rank"):
        MollifiedGradientEstimator().estimate(jax.random.PRNGKey(0), cost, means, 0.0, 0.0, None)

    with pytest.raises(TypeError, match="normal must be real"):
        GaussianHalfspaceCost(jnp.asarray([1.0 + 0.0j], dtype=jnp.complex64))

    complex_mean = means.astype(jnp.complex64)
    with pytest.raises(TypeError, match="mean must be real"):
        MollifiedGradientEstimator().estimate(
            jax.random.PRNGKey(0), cost, complex_mean, 1.0, 0.0, None
        )
    with pytest.raises(TypeError, match="noise_factor must be real"):
        MollifiedGradientEstimator().estimate(
            jax.random.PRNGKey(0),
            cost,
            means,
            jnp.asarray([[1.0 + 0.0j]], dtype=jnp.complex64),
            0.0,
            None,
        )
    with pytest.raises(TypeError, match="time must be real"):
        MollifiedGradientEstimator().estimate(
            jax.random.PRNGKey(0), cost, means, 1.0, 0.0 + 0.0j, None
        )
    with pytest.raises(TypeError, match="context must be real"):
        MollifiedGradientEstimator().estimate(
            jax.random.PRNGKey(0),
            cost,
            means,
            1.0,
            0.0,
            jnp.asarray([[0.0 + 0.0j]], dtype=jnp.complex64),
        )

    class UnregisteredAnalyticCost:
        def analytic_mollified_gradient(self, mean, noise_factor, time, context):
            del noise_factor, time, context
            return jnp.zeros_like(mean)

    with pytest.raises(TypeError, match="audited registered formulas"):
        MollifiedGradientEstimator(MollifiedGradientConfig(backend="analytic")).estimate(
            jax.random.PRNGKey(0),
            UnregisteredAnalyticCost(),
            means,
            1.0,
            0.0,
            None,
        )
    for backend in ("sdf", "learned"):
        info = mollified_gradient_backend_info(backend)
        assert not info.implemented
        assert not info.exact_in_expectation
        with pytest.raises(NotImplementedError, match="declared approximate"):
            MollifiedGradientEstimator(MollifiedGradientConfig(backend=backend)).estimate(
                jax.random.PRNGKey(0), cost, means, 1.0, 0.0, None
            )
