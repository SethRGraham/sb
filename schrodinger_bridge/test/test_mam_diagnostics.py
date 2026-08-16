"""Tests for null-calibrated, bidirectional MAM endpoint diagnostics."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from schrodinger_bridge.solvers.mam_diagnostics import (
    DirectionEndpointSamples,
    EndpointAuditConfig,
    EndpointThresholdFloors,
    EndpointThresholds,
    audit_bidirectional_endpoints,
    audit_endpoint,
    calibrate_endpoint_thresholds,
    compute_endpoint_metrics,
    entropic_sinkhorn_divergence,
    rbf_mmd2,
    sliced_wasserstein_distance,
)


def _config(**overrides) -> EndpointAuditConfig:
    config = EndpointAuditConfig(
        num_projections=8,
        sinkhorn_epsilon=0.5,
        sinkhorn_iterations=60,
        sinkhorn_tolerance=5e-3,
        sinkhorn_max_samples=16,
        null_replicates=3,
        null_quantile=0.8,
        null_threshold_scale=1.0,
    )
    return replace(config, **overrides)


def _normal_sampler(location: float):
    def sample(key, size):
        return location + jax.random.normal(key, (size, 2))

    return sample


def test_metric_kernels_are_jittable_and_zero_on_identical_clouds():
    x = jax.random.normal(jax.random.PRNGKey(1), (12, 2))
    mmd = jax.jit(lambda a: rbf_mmd2(a, a))(x)
    sw = jax.jit(
        lambda a: sliced_wasserstein_distance(jax.random.PRNGKey(2), a, a, num_projections=8)
    )(x)
    sinkhorn = jax.jit(
        lambda a: entropic_sinkhorn_divergence(a, a, epsilon=0.5, iterations=100, tolerance=2e-3)
    )(x)
    np.testing.assert_allclose(np.asarray(mmd), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(sw), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(sinkhorn.value), 0.0, atol=2e-6)
    assert bool(sinkhorn.finite)
    assert bool(sinkhorn.converged)


def test_sinkhorn_log_domain_is_translation_stable():
    key_x, key_y = jax.random.split(jax.random.PRNGKey(3))
    x = 1.0e4 + 0.1 * jax.random.normal(key_x, (16, 2))
    y = 1.0e4 + 0.1 * jax.random.normal(key_y, (16, 2))
    result = entropic_sinkhorn_divergence(x, y, epsilon=0.1, iterations=100, tolerance=2e-3)
    assert bool(result.finite)
    assert bool(result.converged)
    assert float(result.value) >= 0.0


def test_null_calibration_is_seeded_and_uses_reference_sample_sizes():
    config = _config(reference_size=12)
    sampler = _normal_sampler(0.0)
    first = calibrate_endpoint_thresholds(
        jax.random.PRNGKey(4), sampler, generated_size=10, config=config
    )
    second = calibrate_endpoint_thresholds(
        jax.random.PRNGKey(4), sampler, generated_size=10, config=config
    )
    assert first.thresholds == second.thresholds
    assert first.thresholds.generated_size == 10
    assert first.thresholds.reference_size == 12
    assert first.thresholds.valid
    assert first.status == "NULL_CALIBRATED"
    assert len(first.null_metrics) == config.null_replicates


def test_shifted_endpoint_fails_null_calibrated_gate():
    config = _config(null_threshold_scale=1.5)
    sampler = _normal_sampler(0.0)
    calibration = calibrate_endpoint_thresholds(
        jax.random.PRNGKey(5), sampler, generated_size=16, config=config
    )
    reference = sampler(jax.random.PRNGKey(6), 16)
    shifted = 6.0 + sampler(jax.random.PRNGKey(7), 16)
    audit = audit_endpoint(
        jax.random.PRNGKey(8),
        shifted,
        reference,
        calibration.thresholds,
        config,
    )
    assert audit.finite
    assert not audit.passed
    assert not audit.metric_pass["mean_error"]
    assert not audit.metric_pass["sliced_wasserstein"]
    assert not audit.metric_pass["sinkhorn_divergence"]


def test_mode_labels_report_proportions_and_detect_imbalance():
    config = _config()
    reference = jnp.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    imbalanced = jnp.asarray([[1.0], [1.5], [2.0], [2.5]])

    def labels(samples):
        return (samples[:, 0] >= 0.0).astype(jnp.int32)

    metrics = compute_endpoint_metrics(
        jax.random.PRNGKey(9),
        imbalanced,
        reference,
        config,
        mode_label_fn=labels,
        num_modes=2,
    )
    assert metrics.finite
    assert metrics.sample_mode_proportions == (0.0, 1.0)
    assert metrics.reference_mode_proportions == (0.5, 0.5)
    assert metrics.mode_proportion_l1 == 1.0


@pytest.mark.parametrize("invalid", [True, 2.0, 2.5, np.float64(2.0)])
def test_mode_count_requires_a_strict_integer(invalid):
    samples = jnp.asarray([[-1.0], [1.0]])

    def labels(values):
        return (values[:, 0] >= 0.0).astype(jnp.int32)

    with pytest.raises(TypeError, match="num_modes must be an integer"):
        compute_endpoint_metrics(
            jax.random.PRNGKey(91),
            samples,
            samples,
            _config(),
            mode_label_fn=labels,
            num_modes=invalid,
        )


def test_mode_count_rejects_nonpositive_and_ignored_values():
    samples = jnp.asarray([[-1.0], [1.0]])

    def labels(values):
        return (values[:, 0] >= 0.0).astype(jnp.int32)

    with pytest.raises(ValueError, match="num_modes must be positive"):
        compute_endpoint_metrics(
            jax.random.PRNGKey(92),
            samples,
            samples,
            _config(),
            mode_label_fn=labels,
            num_modes=0,
        )
    with pytest.raises(ValueError, match="only with a mode-label"):
        compute_endpoint_metrics(
            jax.random.PRNGKey(93),
            samples,
            samples,
            _config(),
            num_modes=2,
        )

    metrics = compute_endpoint_metrics(
        jax.random.PRNGKey(94),
        samples,
        samples,
        _config(),
        mode_label_fn=labels,
        num_modes=np.int32(2),
    )
    assert metrics.sample_mode_proportions == (0.5, 0.5)


def test_nonfinite_endpoint_fails_closed():
    config = _config()
    samples = jnp.asarray([[0.0], [jnp.nan], [1.0]])
    reference = jnp.asarray([[0.0], [0.5], [1.0]])
    thresholds = EndpointThresholds(
        mmd2=100.0,
        sliced_wasserstein=100.0,
        sinkhorn_divergence=100.0,
        mean_error=100.0,
        covariance_error=100.0,
        mode_proportion_l1=None,
        generated_size=3,
        reference_size=3,
        null_replicates=2,
        null_quantile=0.95,
        valid=True,
    )
    result = audit_endpoint(jax.random.PRNGKey(10), samples, reference, thresholds, config)
    assert not result.finite
    assert not result.passed
    assert result.status == "FAILED_ENDPOINT_GATE"


def test_endpoint_audit_configuration_rejects_ambiguous_numeric_types():
    for field_name in (
        "num_projections",
        "sinkhorn_iterations",
        "null_replicates",
        "sinkhorn_max_samples",
        "reference_size",
    ):
        for invalid in (True, 2.5):
            with pytest.raises(TypeError, match=field_name):
                EndpointAuditConfig(**{field_name: invalid})
    for field_name in (
        "mmd_bandwidth",
        "sinkhorn_epsilon",
        "sinkhorn_tolerance",
        "null_quantile",
        "null_threshold_scale",
    ):
        with pytest.raises((TypeError, ValueError), match=field_name):
            EndpointAuditConfig(**{field_name: True})
        with pytest.raises(ValueError, match=field_name):
            EndpointAuditConfig(**{field_name: float("nan")})
    with pytest.raises(ValueError, match="threshold floors"):
        EndpointThresholdFloors(mmd2=True)
    with pytest.raises(TypeError, match="floors"):
        EndpointAuditConfig(floors={})  # type: ignore[arg-type]


def test_bidirectional_gate_requires_all_four_marginals():
    source_sampler = _normal_sampler(-1.0)
    target_sampler = _normal_sampler(1.0)
    keys = jax.random.split(jax.random.PRNGKey(11), 4)
    forward = DirectionEndpointSamples(
        source=source_sampler(keys[0], 16),
        target=target_sampler(keys[1], 16),
    )
    backward = DirectionEndpointSamples(
        source=source_sampler(keys[2], 16),
        target=target_sampler(keys[3], 16),
    )
    permissive = _config(
        null_threshold_scale=8.0,
        floors=EndpointThresholdFloors(
            mmd2=0.2,
            sliced_wasserstein=1.0,
            sinkhorn_divergence=2.0,
            mean_error=1.0,
            covariance_error=2.0,
        ),
    )
    passing = audit_bidirectional_endpoints(
        jax.random.PRNGKey(12),
        forward,
        backward,
        source_sampler,
        target_sampler,
        permissive,
    )
    assert passing.finite
    assert passing.forward.passed
    assert passing.backward.passed
    assert passing.passed
    broken_backward = DirectionEndpointSamples(
        source=backward.source,
        target=backward.target + 10.0,
    )
    failing = audit_bidirectional_endpoints(
        jax.random.PRNGKey(12),
        forward,
        broken_backward,
        source_sampler,
        target_sampler,
        permissive,
    )
    assert failing.forward.passed
    assert not failing.backward.target.passed
    assert not failing.backward.passed
    assert not failing.passed
    assert failing.status == "FAILED_BIDIRECTIONAL_ENDPOINT_GATE"
