"""Calibrations for the finite pinned path-integral kill baseline."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from schrodinger_bridge.solvers.mam_path_integral import (
    PathIntegralConfig,
    StaticKernelSinkhornConfig,
    estimate_path_integral_control_from_samples,
    estimate_pinned_path_integral_control,
    estimate_static_feynman_kac_kernel,
    scale_static_feynman_kac_kernel,
    simulate_pinned_reference_suffix,
)


def test_one_step_linear_cost_matches_exponential_tilt():
    """For ell(mean + gamma xi)=a x, E_w[xi]=-dt*a*gamma exactly."""
    key = jax.random.PRNGKey(11)
    sample_count = 200_000
    dt = 0.25
    gamma = 0.4
    slope = 0.8
    mean = -0.2
    innovations = jax.random.normal(key, (1, sample_count, 1, 1))
    arrivals = mean + gamma * innovations[..., 0]
    running_values = slope * arrivals
    controls = jnp.zeros_like(innovations)
    result = estimate_path_integral_control_from_samples(
        innovations,
        controls,
        running_values,
        dt,
        minimum_ess_fraction=1e-4,
    )
    expected = -jnp.sqrt(dt) * slope * gamma
    np.testing.assert_allclose(result.raw_control_target[0, 0], expected, atol=7e-3)
    assert bool(result.usable[0])
    assert bool(result.exact_tilted_path_moment_in_population)
    assert not bool(result.exact_finite_gaussian_shift_optimum)
    # Deprecated compatibility alias for the tilted-moment claim only.
    assert bool(result.exact_in_population)
    assert bool(result.finite_sample_ratio_biased)


def test_tilted_path_moment_is_not_general_finite_gaussian_shift_optimum():
    """A hard half-space cost separates the two exact population objects."""
    mean = 0.2
    hard_cost = 3.0
    standard_density_at_mean = math.exp(-0.5 * mean**2) / math.sqrt(2.0 * math.pi)
    positive_probability = 0.5 * (1.0 + math.erf(mean / math.sqrt(2.0)))
    tilted_moment = (
        -(1.0 - math.exp(-hard_cost))
        * standard_density_at_mean
        / (1.0 - positive_probability + math.exp(-hard_cost) * positive_probability)
    )

    # The restricted one-step family is X = mean + u + Z, Z ~ N(0,1), with
    # objective 0.5*u**2 + hard_cost*P(X >= 0).  Its first-order condition is
    # u + hard_cost*phi(mean + u) = 0.  Bisection brackets its strict local and
    # global minimum for these declared constants.
    lower, upper = -1.5, -0.5
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        derivative = midpoint + hard_cost * math.exp(-0.5 * (mean + midpoint) ** 2) / math.sqrt(
            2.0 * math.pi
        )
        if derivative < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    restricted_optimum = 0.5 * (lower + upper)

    def restricted_objective(control: float) -> float:
        event_probability = 0.5 * (1.0 + math.erf((mean + control) / math.sqrt(2.0)))
        return 0.5 * control**2 + hard_cost * event_probability

    tilted_stationarity_residual = tilted_moment + hard_cost * math.exp(
        -0.5 * (mean + tilted_moment) ** 2
    ) / math.sqrt(2.0 * math.pi)
    assert abs(tilted_stationarity_residual) > 0.1
    assert restricted_objective(restricted_optimum) + 5e-3 < restricted_objective(tilted_moment)


def test_reference_likelihood_ratio_recovers_zero_control_for_zero_cost():
    """The Girsanov signs are fixed by this nonzero-reference calibration."""
    key = jax.random.PRNGKey(12)
    sample_count = 200_000
    dt = 0.2
    reference = 0.7
    innovations = jax.random.normal(key, (1, sample_count, 1, 1))
    controls = jnp.full_like(innovations, reference)
    running_values = jnp.zeros((1, sample_count, 1))
    result = estimate_path_integral_control_from_samples(
        innovations,
        controls,
        running_values,
        dt,
        minimum_ess_fraction=1e-4,
    )
    np.testing.assert_allclose(result.raw_control_target[0, 0], 0.0, atol=9e-3)
    assert bool(result.usable[0])


def test_hard_cost_antithetic_suffix_is_pinned_jittable_and_counted():
    def hard_arrival(states, times, context):
        del times, context
        return (states[:, 0] >= 0.1).astype(states.dtype)

    config = PathIntegralConfig(
        num_samples=128,
        antithetic=True,
        minimum_ess_fraction=0.01,
    )
    start = jnp.array([[0.0]])
    endpoint = jnp.array([[1.0]])
    times = jnp.array([0.0, 0.5, 1.0])

    run = jax.jit(
        lambda key: estimate_pinned_path_integral_control(
            key,
            start,
            endpoint,
            times,
            1.0,
            hard_arrival,
            config=config,
        )
    )
    samples, estimate = run(jax.random.PRNGKey(3))
    expected_endpoint = jnp.broadcast_to(endpoint[:, None, :], (1, 256, 1))
    np.testing.assert_array_equal(samples.states[:, :, -1, :], expected_endpoint)
    assert samples.states.shape == (1, 256, 3, 1)
    assert int(samples.physical_query_count) == 256
    assert int(estimate.physical_query_count) == 256
    assert int(estimate.independent_base_draw_count) == 128
    assert bool(samples.antithetic)
    assert bool(estimate.finite[0])
    assert bool(estimate.usable[0])
    assert jnp.all(jnp.isfinite(estimate.control_target))


def test_short_horizon_nonuniform_grid_fails_eager_and_jit():
    def zero_cost(states, times, context):
        del times, context
        return jnp.zeros((states.shape[0],), dtype=states.dtype)

    start = jnp.zeros((1, 1), dtype=jnp.float32)
    endpoint = jnp.ones((1, 1), dtype=jnp.float32)
    bad_times = jnp.asarray([0.0, 1.0e-8, 1.0e-7], dtype=jnp.float32)
    good_times = jnp.linspace(0.0, 1.0e-7, 3, dtype=jnp.float32)
    config = PathIntegralConfig(num_samples=2, antithetic=False)

    def run(times):
        return simulate_pinned_reference_suffix(
            jax.random.PRNGKey(103),
            start,
            endpoint,
            times,
            1.0,
            zero_cost,
            config=config,
        )

    with pytest.raises(ValueError, match="uniform"):
        run(bad_times)
    invalid = jax.jit(run)(bad_times)
    assert not bool(jnp.all(invalid.finite))
    valid = jax.jit(run)(good_times)
    assert bool(jnp.all(valid.finite))


def test_path_integral_callbacks_are_evaluated_as_independent_rows():
    # These callbacks intentionally broadcast their first input row.  A
    # flattened-batch call would couple all Monte Carlo paths; singleton vmap
    # semantics instead recover the declared rowwise Markov functions u(x)=x
    # and ell(x)=x_0 for every path independently.
    def first_row_control(states, time, context):
        del time, context
        return jnp.broadcast_to(states[:1], states.shape)

    def first_row_cost(states, times, context):
        del times, context
        return jnp.broadcast_to(states[:1, 0], (states.shape[0],))

    samples = simulate_pinned_reference_suffix(
        jax.random.PRNGKey(104),
        jnp.asarray([[0.0], [0.7]], dtype=jnp.float32),
        jnp.asarray([[1.0], [1.4]], dtype=jnp.float32),
        jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32),
        0.5,
        first_row_cost,
        first_row_control,
        config=PathIntegralConfig(num_samples=2, antithetic=False),
    )
    np.testing.assert_allclose(
        samples.reference_controls,
        samples.states[:, :, : samples.reference_controls.shape[2], :],
    )
    np.testing.assert_allclose(
        samples.running_values,
        samples.states[:, :, 1 : samples.running_values.shape[2] + 1, 0],
    )
    assert bool(jnp.all(samples.finite))


def test_nonfinite_cost_fails_closed():
    innovations = jnp.array([[[[0.0]], [[1.0]], [[-1.0]]]])
    controls = jnp.zeros_like(innovations)
    running = jnp.array([[[0.0], [jnp.nan], [1.0]]])
    result = jax.jit(estimate_path_integral_control_from_samples)(
        innovations, controls, running, 0.5
    )
    assert not bool(result.finite[0])
    assert not bool(result.usable[0])
    assert bool(result.degenerate[0])
    np.testing.assert_array_equal(result.control_target, jnp.zeros((1, 1)))
    np.testing.assert_array_equal(result.raw_control_target, jnp.zeros((1, 1)))


def test_low_ess_returns_raw_diagnostic_but_no_authorized_update():
    innovations = jnp.array([[[[-2.0]], [[-1.0]], [[0.0]], [[1.0]], [[2.0]]]])
    controls = jnp.full_like(innovations, 0.25)
    # One finite trajectory dominates without using infinities or clipping.
    running = jnp.array([[[0.0], [50.0], [100.0], [150.0], [200.0]]])
    result = estimate_path_integral_control_from_samples(
        innovations,
        controls,
        running,
        1.0,
        minimum_ess_fraction=0.5,
    )
    assert bool(result.finite[0])
    assert bool(result.degenerate[0])
    assert not bool(result.usable[0])
    np.testing.assert_allclose(result.control_target, jnp.array([[0.25]]))
    assert not np.isclose(float(result.raw_control_target[0, 0]), 0.25)


def test_static_zero_potential_kernel_matches_brownian_density():
    innovations = jnp.array([[[[-1.0]], [[0.0]], [[1.0]]]])
    controls = jnp.zeros_like(innovations)
    running = jnp.zeros((1, 3, 1))
    path = estimate_path_integral_control_from_samples(
        innovations,
        controls,
        running,
        0.5,
        minimum_ess_fraction=0.1,
    )
    start = jnp.array([[0.0]])
    endpoint = jnp.array([[1.0]])
    sigma = 2.0
    duration = 1.0
    kernel = estimate_static_feynman_kac_kernel(start, endpoint, duration, sigma, path)
    expected = np.exp(-0.5 * (1.0 / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
    np.testing.assert_allclose(kernel.kernel[0], expected, rtol=1e-6)
    np.testing.assert_allclose(kernel.log_bridge_desirability[0], 0.0, atol=1e-7)
    assert bool(kernel.log_domain_finite[0])
    assert bool(kernel.normal_scale_representable[0])
    assert bool(kernel.finite[0])
    assert bool(kernel.usable[0])
    assert bool(kernel.unbiased_in_exact_arithmetic)
    assert bool(kernel.finite_sample_log_biased)


@pytest.mark.parametrize(
    ("dimension", "endpoint_value", "duration", "expected_log_sign"),
    [
        (64, 0.0, 1e-12, 1.0),  # exp(log K) overflows in float32 and float64.
        (1, 100.0, 1.0, -1.0),  # exp(log K) underflows in float32 and float64.
    ],
)
def test_static_kernel_preserves_valid_log_value_when_normal_scale_is_unrepresentable(
    dimension,
    endpoint_value,
    duration,
    expected_log_sign,
):
    innovations = jnp.zeros((1, 2, 1, dimension))
    path = estimate_path_integral_control_from_samples(
        innovations,
        jnp.zeros_like(innovations),
        jnp.zeros((1, 2, 1)),
        0.5,
        minimum_ess_fraction=0.1,
    )
    start = jnp.zeros((1, dimension))
    endpoint = jnp.full((1, dimension), endpoint_value)
    kernel = jax.jit(
        lambda x, y: estimate_static_feynman_kac_kernel(
            x,
            y,
            duration,
            jnp.eye(dimension),
            path,
        )
    )(start, endpoint)

    assert bool(kernel.log_domain_finite[0])
    assert bool(kernel.finite[0])  # Compatibility alias is log-domain validity.
    assert bool(kernel.usable[0])
    assert not bool(kernel.normal_scale_representable[0])
    assert math.isfinite(float(kernel.log_kernel[0]))
    assert expected_log_sign * float(kernel.log_kernel[0]) > 0.0
    assert float(kernel.kernel[0]) == 0.0


def test_static_feynman_kac_sinkhorn_scales_nonuniform_endpoint_masses_under_jit():
    log_kernel = jnp.log(
        jnp.asarray(
            [
                [1.0, 0.2, 0.5],
                [0.3, 2.0, 0.4],
            ]
        )
    )
    source = jnp.asarray([0.3, 0.7])
    target = jnp.asarray([0.2, 0.5, 0.3])
    config = StaticKernelSinkhornConfig(iterations=300, tolerance=1e-6)
    eager = scale_static_feynman_kac_kernel(
        log_kernel,
        source,
        target,
        config=config,
    )
    compiled = jax.jit(
        lambda matrix: scale_static_feynman_kac_kernel(
            matrix,
            source,
            target,
            config=config,
        )
    )(log_kernel)
    assert bool(eager.finite)
    assert bool(eager.converged)
    assert bool(eager.usable)
    np.testing.assert_allclose(eager.source_marginal, source, atol=1e-6)
    np.testing.assert_allclose(eager.target_marginal, target, atol=1e-6)
    np.testing.assert_allclose(compiled.coupling, eager.coupling, rtol=1e-6, atol=1e-6)


def test_static_feynman_kac_sinkhorn_fails_closed_on_invalid_kernel_and_config():
    with pytest.raises(ValueError, match="finite"):
        scale_static_feynman_kac_kernel(jnp.asarray([[0.0, jnp.nan]]))
    with pytest.raises(TypeError, match="iterations"):
        StaticKernelSinkhornConfig(iterations=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tolerance"):
        StaticKernelSinkhornConfig(tolerance=float("nan"))

    dynamic = jax.jit(scale_static_feynman_kac_kernel)(jnp.asarray([[0.0, jnp.nan], [0.0, 0.0]]))
    assert not bool(dynamic.finite)
    assert not bool(dynamic.converged)
    assert not bool(dynamic.usable)
    np.testing.assert_array_equal(dynamic.coupling, jnp.zeros((2, 2)))

    dynamic_mass = jax.jit(
        lambda mass: scale_static_feynman_kac_kernel(
            jnp.zeros((2, 2)),
            mass,
            None,
        )
    )(jnp.asarray([1.0, jnp.nan]))
    assert not bool(dynamic_mass.finite)
    assert not bool(dynamic_mass.usable)


def test_invalid_initial_reference_controls_fail_closed():
    innovations = jnp.zeros((1, 3, 1, 1))
    controls = jnp.array([[[[0.0]], [[1.0]], [[0.0]]]])
    running = jnp.zeros((1, 3, 1))
    result = estimate_path_integral_control_from_samples(innovations, controls, running, 0.5)
    assert not bool(result.finite[0])
    assert not bool(result.usable[0])


def test_eager_and_jit_agree_and_config_rejects_invalid_values():
    innovations = jnp.array([[[[-1.0]], [[0.5]], [[1.0]]]])
    controls = jnp.zeros_like(innovations)
    running = (innovations[..., 0] > 0.0).astype(jnp.float32)
    eager = estimate_path_integral_control_from_samples(innovations, controls, running, 0.25)
    compiled = jax.jit(estimate_path_integral_control_from_samples)(
        innovations, controls, running, 0.25
    )
    for eager_value, compiled_value in zip(eager, compiled, strict=True):
        np.testing.assert_allclose(eager_value, compiled_value, rtol=1e-6, atol=1e-6)

    with pytest.raises(ValueError, match="num_samples"):
        PathIntegralConfig(num_samples=0)
    for invalid_samples in (True, 1.5):
        with pytest.raises(TypeError, match="num_samples"):
            PathIntegralConfig(num_samples=invalid_samples)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="antithetic"):
        PathIntegralConfig(antithetic=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="minimum_ess_fraction"):
        PathIntegralConfig(minimum_ess_fraction=0.0)
    for field_name in (
        "minimum_ess_fraction",
        "diffusion_rcond",
        "initial_control_tolerance",
    ):
        with pytest.raises(ValueError, match=field_name):
            PathIntegralConfig(**{field_name: float("nan")})


def test_singular_diffusion_fails_eagerly():
    def zero_cost(states, times, context):
        del times, context
        return jnp.zeros((states.shape[0],), dtype=states.dtype)

    with pytest.raises(ValueError, match="full-rank diffusion"):
        estimate_pinned_path_integral_control(
            jax.random.PRNGKey(0),
            jnp.zeros((1, 2)),
            jnp.ones((1, 2)),
            jnp.array([0.0, 0.5, 1.0]),
            jnp.array([[1.0, 0.0], [0.0, 0.0]]),
            zero_cost,
            config=PathIntegralConfig(num_samples=2),
        )
