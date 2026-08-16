"""Held-out endpoint diagnostics for generalized bridge solvers.

The routines in this module are deliberately independent of the MAM training
loop.  Exact pinning of a conditional path does *not* imply that an
endpoint-free Markov projection has the requested two marginals, so the latter
must be tested on fresh samples in both time directions.

Metric kernels return JAX scalars and can be JIT compiled.  Calibration and
pass/fail decisions are host-side operations: they call marginal samplers,
optional mode callbacks, and convert scalar diagnostics to Python values for
logging.  Null thresholds are empirical quantiles of independent
reference-versus-reference comparisons with the same sample sizes used by the
held-out audit.  They are statistical diagnostics, not theorem-level endpoint
guarantees.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..core.types import Array, PRNGKey

MarginalSampler = Callable[[PRNGKey, int], Array]
ModeLabelFn = Callable[[Array], Array]
ModeProportionFn = Callable[[Array], Array]


class SinkhornDivergenceResult(NamedTuple):
    """JAX-compatible result for a debiased entropic OT calculation."""

    value: Array
    cross_cost: Array
    self_x_cost: Array
    self_y_cost: Array
    marginal_error: Array
    converged: Array
    finite: Array


@dataclass(frozen=True)
class EndpointThresholdFloors:
    """Optional predeclared lower bounds for null-calibrated thresholds."""

    mmd2: float = 0.0
    sliced_wasserstein: float = 0.0
    sinkhorn_divergence: float = 0.0
    mean_error: float = 0.0
    covariance_error: float = 0.0
    mode_proportion_l1: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.mmd2,
            self.sliced_wasserstein,
            self.sinkhorn_divergence,
            self.mean_error,
            self.covariance_error,
            self.mode_proportion_l1,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or value < 0.0
            for value in values
        ):
            raise ValueError("endpoint threshold floors must be finite and nonnegative")


@dataclass(frozen=True)
class EndpointAuditConfig:
    """Numerical and statistical settings for a held-out endpoint audit."""

    num_projections: int = 64
    mmd_bandwidth: float | None = None
    sinkhorn_epsilon: float = 0.1
    sinkhorn_iterations: int = 200
    sinkhorn_tolerance: float = 1e-4
    sinkhorn_max_samples: int | None = 256
    reference_size: int | None = None
    null_replicates: int = 16
    null_quantile: float = 0.95
    null_threshold_scale: float = 1.0
    floors: EndpointThresholdFloors = field(default_factory=EndpointThresholdFloors)

    def __post_init__(self) -> None:
        integer_fields: dict[str, int] = {
            "num_projections": self.num_projections,
            "sinkhorn_iterations": self.sinkhorn_iterations,
            "null_replicates": self.null_replicates,
        }
        optional_integer_fields: dict[str, int | None] = {
            "sinkhorn_max_samples": self.sinkhorn_max_samples,
            "reference_size": self.reference_size,
        }
        for name, integer_value in integer_fields.items():
            if isinstance(integer_value, bool) or not isinstance(integer_value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
        for name, optional_integer_value in optional_integer_fields.items():
            if optional_integer_value is not None and (
                isinstance(optional_integer_value, bool)
                or not isinstance(optional_integer_value, (int, np.integer))
            ):
                raise TypeError(f"{name} must be an integer or None")
        scalar_fields: dict[str, float] = {
            "sinkhorn_epsilon": self.sinkhorn_epsilon,
            "sinkhorn_tolerance": self.sinkhorn_tolerance,
            "null_quantile": self.null_quantile,
            "null_threshold_scale": self.null_threshold_scale,
        }
        if self.mmd_bandwidth is not None:
            scalar_fields["mmd_bandwidth"] = self.mmd_bandwidth
        for name, scalar_value in scalar_fields.items():
            if isinstance(scalar_value, bool) or not isinstance(
                scalar_value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{name} must be a real scalar")
        if not isinstance(self.floors, EndpointThresholdFloors):
            raise TypeError("floors must be EndpointThresholdFloors")
        if self.num_projections < 1:
            raise ValueError("num_projections must be positive")
        if self.mmd_bandwidth is not None and (
            not np.isfinite(self.mmd_bandwidth) or self.mmd_bandwidth <= 0.0
        ):
            raise ValueError("mmd_bandwidth must be finite and positive")
        if not np.isfinite(self.sinkhorn_epsilon) or self.sinkhorn_epsilon <= 0.0:
            raise ValueError("sinkhorn_epsilon must be finite and positive")
        if self.sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be positive")
        if not np.isfinite(self.sinkhorn_tolerance) or self.sinkhorn_tolerance <= 0.0:
            raise ValueError("sinkhorn_tolerance must be finite and positive")
        if self.sinkhorn_max_samples is not None and self.sinkhorn_max_samples < 2:
            raise ValueError("sinkhorn_max_samples must be at least two")
        if self.reference_size is not None and self.reference_size < 2:
            raise ValueError("reference_size must be at least two")
        if self.null_replicates < 2:
            raise ValueError("null_replicates must be at least two")
        if not 0.0 < self.null_quantile < 1.0:
            raise ValueError("null_quantile must lie strictly between zero and one")
        if not np.isfinite(self.null_threshold_scale) or self.null_threshold_scale <= 0.0:
            raise ValueError("null_threshold_scale must be finite and positive")


@dataclass(frozen=True)
class EndpointMetrics:
    """Host-readable endpoint discrepancy metrics for one marginal."""

    mmd2: float
    sliced_wasserstein: float
    sinkhorn_divergence: float
    mean_error: float
    covariance_error: float
    mode_proportion_l1: float | None
    sample_mode_proportions: tuple[float, ...] | None
    reference_mode_proportions: tuple[float, ...] | None
    sinkhorn_marginal_error: float
    sinkhorn_converged: bool
    finite: bool


@dataclass(frozen=True)
class EndpointThresholds:
    """Empirical null thresholds for one marginal and fixed sample sizes."""

    mmd2: float
    sliced_wasserstein: float
    sinkhorn_divergence: float
    mean_error: float
    covariance_error: float
    mode_proportion_l1: float | None
    generated_size: int
    reference_size: int
    null_replicates: int
    null_quantile: float
    valid: bool


@dataclass(frozen=True)
class NullCalibrationResult:
    """Thresholds plus calibration health and the raw null diagnostics."""

    thresholds: EndpointThresholds
    null_metrics: tuple[EndpointMetrics, ...]
    finite: bool
    sinkhorn_converged: bool
    status: str


@dataclass(frozen=True)
class EndpointAuditResult:
    """Pass/fail result for one generated marginal against fresh reference data."""

    metrics: EndpointMetrics
    thresholds: EndpointThresholds
    metric_pass: dict[str, bool]
    passed: bool
    finite: bool
    status: str


class DirectionEndpointSamples(NamedTuple):
    """Chronological source and target samples produced by one direction."""

    source: Array
    target: Array


@dataclass(frozen=True)
class DirectionEndpointAudit:
    """Source and target audits for one forward or backward sampler."""

    source: EndpointAuditResult
    target: EndpointAuditResult
    passed: bool
    finite: bool
    direction: str


@dataclass(frozen=True)
class BidirectionalEndpointAudit:
    """Explicit both-direction endpoint gate.

    ``passed`` is true only if all four held-out marginal comparisons pass.
    This result is empirical and must not be presented as an exact bridge
    theorem.
    """

    forward: DirectionEndpointAudit
    backward: DirectionEndpointAudit
    source_calibration: NullCalibrationResult
    target_calibration: NullCalibrationResult
    passed: bool
    finite: bool
    status: str


def _validate_point_clouds(x: Array, y: Array) -> tuple[Array, Array]:
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("endpoint samples must have shape [sample, dimension]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("endpoint samples and references must have the same dimension")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("endpoint metrics require at least two samples per distribution")
    if not jnp.issubdtype(x.dtype, jnp.floating) or not jnp.issubdtype(y.dtype, jnp.floating):
        raise TypeError("endpoint samples must have floating-point dtype")
    dtype = jnp.result_type(x.dtype, y.dtype)
    return x.astype(dtype), y.astype(dtype)


def rbf_mmd2(x: Array, y: Array, bandwidth: float | Array | None = None) -> Array:
    """Unbiased RBF MMD squared, clipped at zero for audit reporting.

    If ``bandwidth`` is omitted, the statistic includes the pooled median
    heuristic.  Null calibration calls this exact same statistic, including
    that data-dependent bandwidth choice.
    """

    x, y = _validate_point_clouds(x, y)
    pooled = jnp.concatenate([x, y], axis=0)
    pooled_sq = jnp.sum((pooled[:, None, :] - pooled[None, :, :]) ** 2, axis=-1)
    if bandwidth is None:
        bandwidth2 = jnp.maximum(jnp.median(pooled_sq), jnp.finfo(x.dtype).eps)
    else:
        bandwidth_array = jnp.asarray(bandwidth, dtype=x.dtype)
        bandwidth2 = bandwidth_array**2
    xx = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    yy = jnp.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    xy = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    k_xx = jnp.exp(-xx / (2.0 * bandwidth2))
    k_yy = jnp.exp(-yy / (2.0 * bandwidth2))
    k_xy = jnp.exp(-xy / (2.0 * bandwidth2))
    n = x.shape[0]
    m = y.shape[0]
    value = (
        (jnp.sum(k_xx) - jnp.trace(k_xx)) / (n * (n - 1))
        + (jnp.sum(k_yy) - jnp.trace(k_yy)) / (m * (m - 1))
        - 2.0 * jnp.mean(k_xy)
    )
    return jnp.maximum(value, jnp.asarray(0.0, dtype=value.dtype))


def sliced_wasserstein_distance(
    key: PRNGKey,
    x: Array,
    y: Array,
    *,
    num_projections: int = 64,
) -> Array:
    """Monte Carlo sliced Wasserstein-1 distance with deterministic keying."""

    x, y = _validate_point_clouds(x, y)
    if num_projections < 1:
        raise ValueError("num_projections must be positive")
    directions = jax.random.normal(key, (num_projections, x.shape[1]), dtype=x.dtype)
    norms = jnp.linalg.norm(directions, axis=-1, keepdims=True)
    directions = directions / jnp.maximum(norms, jnp.finfo(x.dtype).tiny)
    x_sorted = jnp.sort(x @ directions.T, axis=0)
    y_sorted = jnp.sort(y @ directions.T, axis=0)
    quantiles = jnp.linspace(0.0, 1.0, max(x.shape[0], y.shape[0]), dtype=x.dtype)
    x_grid = jnp.linspace(0.0, 1.0, x.shape[0], dtype=x.dtype)
    y_grid = jnp.linspace(0.0, 1.0, y.shape[0], dtype=x.dtype)

    def interpolate_projection(values: Array, grid: Array) -> Array:
        return jax.vmap(lambda column: jnp.interp(quantiles, grid, column), in_axes=1)(values).T

    x_quantiles = interpolate_projection(x_sorted, x_grid)
    y_quantiles = interpolate_projection(y_sorted, y_grid)
    return jnp.mean(jnp.abs(x_quantiles - y_quantiles))


def _entropic_ot(
    x: Array,
    y: Array,
    *,
    epsilon: float,
    iterations: int,
) -> tuple[Array, Array, Array]:
    """Log-domain entropic OT with uniform empirical weights.

    The regularized objective is

    ``<P,C> + epsilon * KL(P || a tensor b)``

    with squared Euclidean ground cost and uniform ``a,b``.  The returned
    marginal error is the maximum absolute row/column residual.
    """

    cost = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    eps = jnp.asarray(epsilon, dtype=x.dtype)
    log_kernel = -cost / eps
    log_a = jnp.full((x.shape[0],), -jnp.log(jnp.asarray(x.shape[0], dtype=x.dtype)))
    log_b = jnp.full((y.shape[0],), -jnp.log(jnp.asarray(y.shape[0], dtype=x.dtype)))

    def update(_: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        log_u, log_v = state
        log_u = log_a - jax.scipy.special.logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - jax.scipy.special.logsumexp(log_kernel.T + log_u[None, :], axis=1)
        return log_u, log_v

    zeros_u = jnp.zeros_like(log_a)
    zeros_v = jnp.zeros_like(log_b)
    log_u, log_v = jax.lax.fori_loop(0, iterations, update, (zeros_u, zeros_v))
    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    plan = jnp.exp(log_plan)
    row_error = jnp.max(jnp.abs(jnp.sum(plan, axis=1) - jnp.exp(log_a)))
    column_error = jnp.max(jnp.abs(jnp.sum(plan, axis=0) - jnp.exp(log_b)))
    marginal_error = jnp.maximum(row_error, column_error)
    reference_log_mass = log_a[:, None] + log_b[None, :]
    kl = jnp.sum(jnp.where(plan > 0.0, plan * (log_plan - reference_log_mass), 0.0))
    objective = jnp.sum(plan * cost) + eps * kl
    finite = jnp.all(jnp.isfinite(log_u)) & jnp.all(jnp.isfinite(log_v))
    finite &= jnp.isfinite(objective) & jnp.isfinite(marginal_error)
    return objective, marginal_error, finite


def entropic_sinkhorn_divergence(
    x: Array,
    y: Array,
    *,
    epsilon: float = 0.1,
    iterations: int = 200,
    tolerance: float = 1e-4,
) -> SinkhornDivergenceResult:
    """Stable debiased Sinkhorn divergence without optional dependencies."""

    x, y = _validate_point_clouds(x, y)
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    cross, error_xy, finite_xy = _entropic_ot(x, y, epsilon=epsilon, iterations=iterations)
    self_x, error_xx, finite_xx = _entropic_ot(x, x, epsilon=epsilon, iterations=iterations)
    self_y, error_yy, finite_yy = _entropic_ot(y, y, epsilon=epsilon, iterations=iterations)
    raw_value = cross - 0.5 * (self_x + self_y)
    value = jnp.maximum(raw_value, jnp.asarray(0.0, dtype=raw_value.dtype))
    marginal_error = jnp.maximum(error_xy, jnp.maximum(error_xx, error_yy))
    finite = finite_xy & finite_xx & finite_yy & jnp.isfinite(value)
    converged = finite & (marginal_error <= tolerance)
    return SinkhornDivergenceResult(
        value=value,
        cross_cost=cross,
        self_x_cost=self_x,
        self_y_cost=self_y,
        marginal_error=marginal_error,
        converged=converged,
        finite=finite,
    )


def _random_subset(key: PRNGKey, samples: Array, maximum: int | None) -> Array:
    if maximum is None or samples.shape[0] <= maximum:
        return samples
    indices = jax.random.choice(key, samples.shape[0], shape=(maximum,), replace=False)
    return samples[indices]


def _mode_proportions(
    samples: Array,
    *,
    mode_label_fn: ModeLabelFn | None,
    num_modes: int | None,
    mode_proportion_fn: ModeProportionFn | None,
) -> Array | None:
    if mode_label_fn is not None and mode_proportion_fn is not None:
        raise ValueError("provide a mode-label callback or a proportion callback, not both")
    if mode_label_fn is None and num_modes is not None:
        raise ValueError("num_modes is meaningful only with a mode-label callback")
    if mode_label_fn is not None:
        if num_modes is None:
            raise ValueError("num_modes is required with a mode-label callback")
        if isinstance(num_modes, bool) or not isinstance(num_modes, (int, np.integer)):
            raise TypeError("num_modes must be an integer")
        if num_modes < 1:
            raise ValueError("num_modes must be positive")
        mode_count = int(num_modes)
        labels = jnp.asarray(mode_label_fn(samples))
        if labels.shape != (samples.shape[0],):
            return jnp.full((mode_count,), jnp.nan, dtype=samples.dtype)
        labels_as_float = labels.astype(samples.dtype)
        valid = jnp.all(jnp.isfinite(labels_as_float))
        valid &= jnp.all(labels_as_float == jnp.floor(labels_as_float))
        valid &= jnp.all((labels_as_float >= 0) & (labels_as_float < mode_count))
        safe_labels = jnp.clip(labels_as_float, 0, mode_count - 1).astype(jnp.int32)
        proportions = jnp.bincount(safe_labels, length=mode_count).astype(samples.dtype)
        proportions /= samples.shape[0]
        return jnp.where(valid, proportions, jnp.full_like(proportions, jnp.nan))
    if mode_proportion_fn is None:
        return None
    proportions = jnp.asarray(mode_proportion_fn(samples), dtype=samples.dtype)
    if proportions.ndim != 1 or proportions.shape[0] < 1:
        return jnp.asarray([jnp.nan], dtype=samples.dtype)
    valid = jnp.all(jnp.isfinite(proportions)) & jnp.all(proportions >= 0.0)
    valid &= jnp.isclose(jnp.sum(proportions), 1.0, rtol=1e-5, atol=1e-6)
    return jnp.where(valid, proportions, jnp.full_like(proportions, jnp.nan))


def compute_endpoint_metrics(
    key: PRNGKey,
    samples: Array,
    reference: Array,
    config: EndpointAuditConfig,
    *,
    mode_label_fn: ModeLabelFn | None = None,
    num_modes: int | None = None,
    mode_proportion_fn: ModeProportionFn | None = None,
) -> EndpointMetrics:
    """Compute all endpoint discrepancies for one held-out comparison.

    Optional mode callbacks are audit-only host interfaces.  A label callback
    returns one integer in ``[0, num_modes)`` per sample.  A proportion callback
    directly returns a finite nonnegative vector summing to one.
    """

    samples, reference = _validate_point_clouds(samples, reference)
    if not bool(jnp.all(jnp.isfinite(samples)) & jnp.all(jnp.isfinite(reference))):
        return EndpointMetrics(
            mmd2=float("nan"),
            sliced_wasserstein=float("nan"),
            sinkhorn_divergence=float("nan"),
            mean_error=float("nan"),
            covariance_error=float("nan"),
            mode_proportion_l1=None,
            sample_mode_proportions=None,
            reference_mode_proportions=None,
            sinkhorn_marginal_error=float("nan"),
            sinkhorn_converged=False,
            finite=False,
        )
    sw_key, sample_subset_key, reference_subset_key = jax.random.split(key, 3)
    mmd2 = rbf_mmd2(samples, reference, bandwidth=config.mmd_bandwidth)
    sw = sliced_wasserstein_distance(
        sw_key,
        samples,
        reference,
        num_projections=config.num_projections,
    )
    sinkhorn_samples = _random_subset(sample_subset_key, samples, config.sinkhorn_max_samples)
    sinkhorn_reference = _random_subset(
        reference_subset_key, reference, config.sinkhorn_max_samples
    )
    sinkhorn = entropic_sinkhorn_divergence(
        sinkhorn_samples,
        sinkhorn_reference,
        epsilon=config.sinkhorn_epsilon,
        iterations=config.sinkhorn_iterations,
        tolerance=config.sinkhorn_tolerance,
    )
    sample_mean = jnp.mean(samples, axis=0)
    reference_mean = jnp.mean(reference, axis=0)
    sample_centered = samples - sample_mean
    reference_centered = reference - reference_mean
    sample_covariance = sample_centered.T @ sample_centered / (samples.shape[0] - 1)
    reference_covariance = reference_centered.T @ reference_centered / (reference.shape[0] - 1)
    mean_error = jnp.linalg.norm(sample_mean - reference_mean)
    covariance_error = jnp.linalg.norm(sample_covariance - reference_covariance)
    sample_mode = _mode_proportions(
        samples,
        mode_label_fn=mode_label_fn,
        num_modes=num_modes,
        mode_proportion_fn=mode_proportion_fn,
    )
    reference_mode = _mode_proportions(
        reference,
        mode_label_fn=mode_label_fn,
        num_modes=num_modes,
        mode_proportion_fn=mode_proportion_fn,
    )
    if (sample_mode is None) != (reference_mode is None):  # pragma: no cover
        raise RuntimeError("internal mode callback mismatch")
    mode_error_array = None
    if sample_mode is not None and reference_mode is not None:
        if sample_mode.shape != reference_mode.shape:
            mode_error_array = jnp.asarray(jnp.nan, dtype=samples.dtype)
        else:
            mode_error_array = jnp.sum(jnp.abs(sample_mode - reference_mode))
    scalar_arrays = [
        mmd2,
        sw,
        sinkhorn.value,
        mean_error,
        covariance_error,
        sinkhorn.marginal_error,
    ]
    if mode_error_array is not None:
        scalar_arrays.append(mode_error_array)
    finite = bool(jnp.all(jnp.stack([jnp.isfinite(value) for value in scalar_arrays])))
    finite = finite and bool(sinkhorn.finite)
    sample_mode_tuple = (
        None if sample_mode is None else tuple(float(value) for value in np.asarray(sample_mode))
    )
    reference_mode_tuple = (
        None
        if reference_mode is None
        else tuple(float(value) for value in np.asarray(reference_mode))
    )
    return EndpointMetrics(
        mmd2=float(mmd2),
        sliced_wasserstein=float(sw),
        sinkhorn_divergence=float(sinkhorn.value),
        mean_error=float(mean_error),
        covariance_error=float(covariance_error),
        mode_proportion_l1=(None if mode_error_array is None else float(mode_error_array)),
        sample_mode_proportions=sample_mode_tuple,
        reference_mode_proportions=reference_mode_tuple,
        sinkhorn_marginal_error=float(sinkhorn.marginal_error),
        sinkhorn_converged=bool(sinkhorn.converged),
        finite=finite,
    )


def _higher_quantile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values), quantile, method="higher"))


def calibrate_endpoint_thresholds(
    key: PRNGKey,
    sampler: MarginalSampler,
    generated_size: int,
    config: EndpointAuditConfig,
    *,
    mode_label_fn: ModeLabelFn | None = None,
    num_modes: int | None = None,
    mode_proportion_fn: ModeProportionFn | None = None,
) -> NullCalibrationResult:
    """Calibrate one endpoint gate from independent null comparisons.

    Each replicate draws a fresh sample of ``generated_size`` and an
    independent reference sample.  No generated bridge samples are reused in
    calibration.
    """

    if generated_size < 2:
        raise ValueError("generated_size must be at least two")
    reference_size = config.reference_size or generated_size
    keys = jax.random.split(key, 3 * config.null_replicates)
    metrics: list[EndpointMetrics] = []
    for replicate in range(config.null_replicates):
        first = jnp.asarray(sampler(keys[3 * replicate], generated_size))
        second = jnp.asarray(sampler(keys[3 * replicate + 1], reference_size))
        metrics.append(
            compute_endpoint_metrics(
                keys[3 * replicate + 2],
                first,
                second,
                config,
                mode_label_fn=mode_label_fn,
                num_modes=num_modes,
                mode_proportion_fn=mode_proportion_fn,
            )
        )
    finite = all(metric.finite for metric in metrics)
    converged = all(metric.sinkhorn_converged for metric in metrics)

    def threshold(name: str, floor: float) -> float:
        values = [float(getattr(metric, name)) for metric in metrics]
        if not all(np.isfinite(value) for value in values):
            return float("nan")
        return max(
            floor, config.null_threshold_scale * _higher_quantile(values, config.null_quantile)
        )

    has_mode = metrics[0].mode_proportion_l1 is not None
    mode_threshold = None
    if has_mode:
        mode_values = [metric.mode_proportion_l1 for metric in metrics]
        if any(value is None for value in mode_values):  # pragma: no cover
            finite = False
            mode_threshold = float("nan")
        else:
            cast_values = [float(value) for value in mode_values if value is not None]
            mode_threshold = max(
                config.floors.mode_proportion_l1,
                config.null_threshold_scale * _higher_quantile(cast_values, config.null_quantile),
            )
    thresholds = EndpointThresholds(
        mmd2=threshold("mmd2", config.floors.mmd2),
        sliced_wasserstein=threshold("sliced_wasserstein", config.floors.sliced_wasserstein),
        sinkhorn_divergence=threshold("sinkhorn_divergence", config.floors.sinkhorn_divergence),
        mean_error=threshold("mean_error", config.floors.mean_error),
        covariance_error=threshold("covariance_error", config.floors.covariance_error),
        mode_proportion_l1=mode_threshold,
        generated_size=generated_size,
        reference_size=reference_size,
        null_replicates=config.null_replicates,
        null_quantile=config.null_quantile,
        valid=finite and converged,
    )
    status = "NULL_CALIBRATED" if thresholds.valid else "INVALID_NULL_CALIBRATION_FAIL_CLOSED"
    return NullCalibrationResult(
        thresholds=thresholds,
        null_metrics=tuple(metrics),
        finite=finite,
        sinkhorn_converged=converged,
        status=status,
    )


def audit_endpoint(
    key: PRNGKey,
    samples: Array,
    reference: Array,
    thresholds: EndpointThresholds,
    config: EndpointAuditConfig,
    *,
    mode_label_fn: ModeLabelFn | None = None,
    num_modes: int | None = None,
    mode_proportion_fn: ModeProportionFn | None = None,
) -> EndpointAuditResult:
    """Apply pre-calibrated thresholds to a fresh endpoint comparison."""

    if samples.shape[0] != thresholds.generated_size:
        raise ValueError("audit sample size does not match null calibration")
    if reference.shape[0] != thresholds.reference_size:
        raise ValueError("reference sample size does not match null calibration")
    metrics = compute_endpoint_metrics(
        key,
        samples,
        reference,
        config,
        mode_label_fn=mode_label_fn,
        num_modes=num_modes,
        mode_proportion_fn=mode_proportion_fn,
    )
    metric_pass = {
        "mmd2": metrics.mmd2 <= thresholds.mmd2,
        "sliced_wasserstein": (metrics.sliced_wasserstein <= thresholds.sliced_wasserstein),
        "sinkhorn_divergence": (metrics.sinkhorn_divergence <= thresholds.sinkhorn_divergence),
        "mean_error": metrics.mean_error <= thresholds.mean_error,
        "covariance_error": metrics.covariance_error <= thresholds.covariance_error,
    }
    if thresholds.mode_proportion_l1 is not None:
        metric_pass["mode_proportion_l1"] = (
            metrics.mode_proportion_l1 is not None
            and metrics.mode_proportion_l1 <= thresholds.mode_proportion_l1
        )
    finite = metrics.finite and thresholds.valid
    passed = finite and metrics.sinkhorn_converged and all(metric_pass.values())
    return EndpointAuditResult(
        metrics=metrics,
        thresholds=thresholds,
        metric_pass=metric_pass,
        passed=passed,
        finite=finite,
        status="PASSED_EMPIRICAL_ENDPOINT_GATE" if passed else "FAILED_ENDPOINT_GATE",
    )


def audit_bidirectional_endpoints(
    key: PRNGKey,
    forward: DirectionEndpointSamples,
    backward: DirectionEndpointSamples,
    source_sampler: MarginalSampler,
    target_sampler: MarginalSampler,
    config: EndpointAuditConfig,
    *,
    source_mode_label_fn: ModeLabelFn | None = None,
    source_num_modes: int | None = None,
    source_mode_proportion_fn: ModeProportionFn | None = None,
    target_mode_label_fn: ModeLabelFn | None = None,
    target_num_modes: int | None = None,
    target_mode_proportion_fn: ModeProportionFn | None = None,
) -> BidirectionalEndpointAudit:
    """Calibrate and audit source/target marginals in both directions.

    Forward and backward sample counts must agree for each chronological
    marginal so they can share a null-calibrated threshold.  Calibration draws
    and all four held-out reference draws use disjoint deterministic PRNG
    subkeys.
    """

    if forward.source.shape[0] != backward.source.shape[0]:
        raise ValueError("forward/backward source audit sizes must agree")
    if forward.target.shape[0] != backward.target.shape[0]:
        raise ValueError("forward/backward target audit sizes must agree")
    keys = jax.random.split(key, 10)
    source_calibration = calibrate_endpoint_thresholds(
        keys[0],
        source_sampler,
        forward.source.shape[0],
        config,
        mode_label_fn=source_mode_label_fn,
        num_modes=source_num_modes,
        mode_proportion_fn=source_mode_proportion_fn,
    )
    target_calibration = calibrate_endpoint_thresholds(
        keys[1],
        target_sampler,
        forward.target.shape[0],
        config,
        mode_label_fn=target_mode_label_fn,
        num_modes=target_num_modes,
        mode_proportion_fn=target_mode_proportion_fn,
    )
    source_reference_size = source_calibration.thresholds.reference_size
    target_reference_size = target_calibration.thresholds.reference_size
    forward_source_reference = source_sampler(keys[2], source_reference_size)
    forward_target_reference = target_sampler(keys[3], target_reference_size)
    backward_source_reference = source_sampler(keys[4], source_reference_size)
    backward_target_reference = target_sampler(keys[5], target_reference_size)
    forward_source = audit_endpoint(
        keys[6],
        forward.source,
        forward_source_reference,
        source_calibration.thresholds,
        config,
        mode_label_fn=source_mode_label_fn,
        num_modes=source_num_modes,
        mode_proportion_fn=source_mode_proportion_fn,
    )
    forward_target = audit_endpoint(
        keys[7],
        forward.target,
        forward_target_reference,
        target_calibration.thresholds,
        config,
        mode_label_fn=target_mode_label_fn,
        num_modes=target_num_modes,
        mode_proportion_fn=target_mode_proportion_fn,
    )
    backward_source = audit_endpoint(
        keys[8],
        backward.source,
        backward_source_reference,
        source_calibration.thresholds,
        config,
        mode_label_fn=source_mode_label_fn,
        num_modes=source_num_modes,
        mode_proportion_fn=source_mode_proportion_fn,
    )
    backward_target = audit_endpoint(
        keys[9],
        backward.target,
        backward_target_reference,
        target_calibration.thresholds,
        config,
        mode_label_fn=target_mode_label_fn,
        num_modes=target_num_modes,
        mode_proportion_fn=target_mode_proportion_fn,
    )
    forward_audit = DirectionEndpointAudit(
        source=forward_source,
        target=forward_target,
        passed=forward_source.passed and forward_target.passed,
        finite=forward_source.finite and forward_target.finite,
        direction="forward",
    )
    backward_audit = DirectionEndpointAudit(
        source=backward_source,
        target=backward_target,
        passed=backward_source.passed and backward_target.passed,
        finite=backward_source.finite and backward_target.finite,
        direction="backward",
    )
    finite = (
        source_calibration.finite
        and target_calibration.finite
        and forward_audit.finite
        and backward_audit.finite
    )
    passed = finite and forward_audit.passed and backward_audit.passed
    return BidirectionalEndpointAudit(
        forward=forward_audit,
        backward=backward_audit,
        source_calibration=source_calibration,
        target_calibration=target_calibration,
        passed=passed,
        finite=finite,
        status=(
            "PASSED_BOTH_DIRECTIONS_EMPIRICALLY" if passed else "FAILED_BIDIRECTIONAL_ENDPOINT_GATE"
        ),
    )


__all__ = [
    "BidirectionalEndpointAudit",
    "DirectionEndpointAudit",
    "DirectionEndpointSamples",
    "EndpointAuditConfig",
    "EndpointAuditResult",
    "EndpointMetrics",
    "EndpointThresholdFloors",
    "EndpointThresholds",
    "MarginalSampler",
    "ModeLabelFn",
    "ModeProportionFn",
    "NullCalibrationResult",
    "SinkhornDivergenceResult",
    "audit_bidirectional_endpoints",
    "audit_endpoint",
    "calibrate_endpoint_thresholds",
    "compute_endpoint_metrics",
    "entropic_sinkhorn_divergence",
    "rbf_mmd2",
    "sliced_wasserstein_distance",
]
