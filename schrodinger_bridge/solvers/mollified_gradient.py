"""Value-only gradients of Gaussian-mollified running costs.

The estimand in this module is deliberately local.  For a value-only cost
``ell`` and a Gaussian transition

    X = mean + noise_factor @ xi,       xi ~ Normal(0, I),

we estimate ``grad_mean E[ell(X, time, context)]``.  ``noise_factor`` is a
factor of the transition covariance, not the covariance itself.  The Stein
identity therefore uses ``noise_factor^{-T} xi``.  This orientation matters
for nonsymmetric factors.

Both Monte Carlo backends are unbiased for the gradient of the *same*
chain-induced Gaussian convolution.  Cost values are stopped before label
assembly, making the result suitable as a regression target rather than an
autodiff path through a hard oracle.  The module is rollout-free, but this
local fact does not make a complete Malliavin-adjoint bridge solver
simulation-free.

V1 requires a square, full-rank factor.  Scalar, shared diagonal ``[d]``,
shared full ``[d,d]``, and batched full ``[B,d,d]`` factors are accepted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, NamedTuple, Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from ..core.types import Array, PRNGKey, Scalar

ValueOnlyRunningCost = Callable[[Array, Array, Array], Array]


class MollifiedGradientBackend(IntEnum):
    """Stable, JAX-compatible codes returned in ``MollifiedGradientBatch``."""

    ANTITHETIC_STEIN = 0
    IID_STEIN = 1
    ANALYTIC = 2
    SDF = 3
    LEARNED = 4


_BACKEND_CODES = {
    "antithetic_stein": MollifiedGradientBackend.ANTITHETIC_STEIN,
    "iid_stein": MollifiedGradientBackend.IID_STEIN,
    "analytic": MollifiedGradientBackend.ANALYTIC,
    "sdf": MollifiedGradientBackend.SDF,
    "learned": MollifiedGradientBackend.LEARNED,
}
_IMPLEMENTED_BACKENDS = frozenset({"antithetic_stein", "iid_stein", "analytic"})
_EXACT_BACKENDS = frozenset(_IMPLEMENTED_BACKENDS)


@dataclass(frozen=True)
class MollifiedGradientBackendInfo:
    """Human-readable backend metadata kept outside JIT result pytrees."""

    name: str
    code: MollifiedGradientBackend
    implemented: bool
    exact_in_expectation: bool
    description: str


def mollified_gradient_backend_info(name: str) -> MollifiedGradientBackendInfo:
    """Return declared implementation and approximation status for ``name``."""
    if name not in _BACKEND_CODES:
        raise ValueError(f"unknown mollified-gradient backend {name!r}")
    descriptions = {
        "antithetic_stein": "unbiased antithetic Gaussian Stein estimator",
        "iid_stein": "unbiased leave-one-out Gaussian Stein estimator",
        "analytic": "cost-provided exact Gaussian-convolution gradient",
        "sdf": "approximate local signed-distance expansion (not implemented in V1)",
        "learned": "approximate learned conservative mollifier (not implemented in V1)",
    }
    return MollifiedGradientBackendInfo(
        name=name,
        code=_BACKEND_CODES[name],
        implemented=name in _IMPLEMENTED_BACKENDS,
        exact_in_expectation=name in _EXACT_BACKENDS,
        description=descriptions[name],
    )


@runtime_checkable
class AnalyticMollifiableCost(Protocol):
    """Protocol for an exact gradient of the chain-induced convolution."""

    def analytic_mollified_gradient(
        self,
        mean: Array,
        noise_factor: Array,
        time: Array,
        context: Array,
    ) -> Array:
        """Return ``grad_mean E[ell(mean + L xi)]`` with shape ``[B,d]``."""


@dataclass(frozen=True)
class GaussianHalfspaceCost:
    """Binary half-space oracle with an exact Gaussian mollification.

    The high value is returned on ``normal @ x >= threshold``.  For
    ``X ~ Normal(mean, L L^T)``, the exact gradient is

    ``(high-low) * normal * phi(z) / ||L^T normal||``, where
    ``z = (normal @ mean - threshold) / ||L^T normal||``.
    """

    normal: Array
    threshold: float = 0.0
    high_value: float = 1.0
    low_value: float = 0.0

    def __post_init__(self) -> None:
        normal = jnp.asarray(self.normal)
        if normal.ndim != 1 or normal.shape[0] < 1:
            raise ValueError("normal must have shape [d] with d >= 1")
        if np.issubdtype(normal.dtype, np.complexfloating):
            raise TypeError("normal must be real-valued")
        if not np.issubdtype(normal.dtype, np.floating):
            normal = normal.astype(jnp.float32)
        if not isinstance(normal, jax.core.Tracer):
            concrete = np.asarray(jax.device_get(normal))
            if not np.all(np.isfinite(concrete)) or np.linalg.norm(concrete) == 0.0:
                raise ValueError("normal must be finite and nonzero")
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if not np.isfinite(self.high_value) or not np.isfinite(self.low_value):
            raise ValueError("half-space values must be finite")
        object.__setattr__(self, "normal", normal)

    def __call__(self, states: Array, times: Array, context: Array) -> Array:
        del times, context
        normal = jnp.asarray(self.normal, dtype=states.dtype)
        projection = states @ normal
        return jnp.where(
            projection >= jnp.asarray(self.threshold, dtype=states.dtype),
            jnp.asarray(self.high_value, dtype=states.dtype),
            jnp.asarray(self.low_value, dtype=states.dtype),
        )

    def analytic_mollified_gradient(
        self,
        mean: Array,
        noise_factor: Array,
        time: Array,
        context: Array,
    ) -> Array:
        del time, context
        normal = jnp.asarray(self.normal, dtype=mean.dtype)
        projected_noise = jnp.einsum("bij,j->bi", jnp.swapaxes(noise_factor, -1, -2), normal)
        scale = jnp.linalg.norm(projected_noise, axis=-1)
        z = (mean @ normal - jnp.asarray(self.threshold, dtype=mean.dtype)) / scale
        normal_pdf = jnp.exp(-0.5 * z**2) / jnp.sqrt(jnp.asarray(2.0 * np.pi, dtype=mean.dtype))
        value_jump = jnp.asarray(self.high_value - self.low_value, dtype=mean.dtype)
        return jnp.asarray(value_jump * (normal_pdf / scale)[:, None] * normal[None, :])


# Analytic status is an audited allow-list, not structural duck typing.  A
# callable merely exposing a method with the right name is not evidence that
# its formula targets the declared Gaussian convolution.
_AUDITED_ANALYTIC_COSTS: dict[type[Any], str] = {
    GaussianHalfspaceCost: "gaussian_halfspace_v1",
}


def registered_analytic_mollifier(cost: Any) -> str | None:
    """Return the audited formula identifier for an exact analytic backend."""
    return _AUDITED_ANALYTIC_COSTS.get(type(cost))


@dataclass(frozen=True)
class MollifiedGradientConfig:
    """Configuration for a value-only Gaussian-mollified gradient.

    ``num_samples`` counts antithetic pairs for ``antithetic_stein`` and
    ordinary Gaussian draws for ``iid_stein``.  IID leave-one-out centering
    requires at least two draws.  The default is one antithetic pair, matching
    the local actor-correction default in MAM-GSBM.
    """

    backend: str = "antithetic_stein"
    num_samples: int = 1
    min_singular_value: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str):
            raise TypeError("backend must be a string")
        if self.backend not in _BACKEND_CODES:
            raise ValueError(
                f"backend must be one of {tuple(_BACKEND_CODES)}, got {self.backend!r}"
            )
        if isinstance(self.num_samples, bool) or not isinstance(
            self.num_samples, (int, np.integer)
        ):
            raise TypeError("num_samples must be an integer")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.backend == "iid_stein" and self.num_samples < 2:
            raise ValueError("iid_stein requires at least two samples for unbiased centering")
        if isinstance(self.min_singular_value, bool) or not isinstance(
            self.min_singular_value, (int, float, np.integer, np.floating)
        ):
            raise TypeError("min_singular_value must be a real scalar")
        if self.min_singular_value <= 0 or not np.isfinite(self.min_singular_value):
            raise ValueError("min_singular_value must be finite and positive")


class MollifiedGradientBatch(NamedTuple):
    """Batched gradient labels and diagnostics.

    Shapes:
        gradient: ``[B,d]``
        standard_error: ``[B,d]``; ``inf`` when one draw cannot estimate it
        physical_query_count: scalar number of evaluated state/cost pairs
        finite: ``[B]`` validity mask; invalid rows have zero gradients
        backend_code: scalar integer ``MollifiedGradientBackend`` code
        exact: scalar bool; true means exact analytically or in expectation
        standard_error_available: scalar bool

    IID standard errors use the delete-one jackknife.  Antithetic standard
    errors use independent-pair sample variance.
    """

    gradient: Array
    standard_error: Array
    physical_query_count: Array
    finite: Array
    backend_code: Array
    exact: Array
    standard_error_available: Array


def _require_concrete(predicate: Array, message: str) -> None:
    """Raise eagerly while preserving a fail-closed path under ``jax.jit``."""
    if isinstance(predicate, jax.core.Tracer):
        return
    if not bool(np.all(np.asarray(jax.device_get(predicate)))):
        raise ValueError(message)


def _normalize_mean(mean: Array) -> Array:
    mean = jnp.asarray(mean)
    if mean.ndim != 2 or mean.shape[-1] < 1:
        raise ValueError(f"mean must have shape [B,d], got {mean.shape}")
    if np.issubdtype(mean.dtype, np.complexfloating):
        raise TypeError("mean must be real-valued")
    if not np.issubdtype(mean.dtype, np.floating):
        mean = mean.astype(jnp.float32)
    return mean


def _normalize_time(time: Array | Scalar, batch_size: int, dtype: Any) -> Array:
    if jnp.issubdtype(jnp.asarray(time).dtype, jnp.complexfloating):
        raise TypeError("time must be real-valued")
    time = jnp.asarray(time, dtype=dtype)
    if time.ndim == 0:
        return jnp.full((batch_size,), time, dtype=dtype)
    if time.shape != (batch_size,):
        raise ValueError(f"time must be scalar or have shape {(batch_size,)}, got {time.shape}")
    return time


def _normalize_context(context: Array | None, batch_size: int, dtype: Any) -> Array:
    if context is None:
        return jnp.zeros((batch_size, 0), dtype=dtype)
    if jnp.issubdtype(jnp.asarray(context).dtype, jnp.complexfloating):
        raise TypeError("context must be real-valued")
    context = jnp.asarray(context, dtype=dtype)
    if context.ndim == 1:
        context = context[None, :]
    if context.ndim != 2:
        raise ValueError("context must have shape [c], [1,c], or [B,c]")
    if context.shape[0] == 1 and batch_size != 1:
        context = jnp.broadcast_to(context, (batch_size, context.shape[-1]))
    if context.shape[0] != batch_size:
        raise ValueError("context batch size must be one or match mean")
    return context


def _normalize_noise_factor(
    noise_factor: Array | Scalar,
    batch_size: int,
    dim: int,
    dtype: Any,
) -> tuple[Array, bool]:
    if jnp.issubdtype(jnp.asarray(noise_factor).dtype, jnp.complexfloating):
        raise TypeError("noise_factor must be real-valued")
    factor = jnp.asarray(noise_factor, dtype=dtype)
    eye = jnp.eye(dim, dtype=dtype)
    if factor.ndim == 0:
        factor = factor * eye
        return jnp.broadcast_to(factor, (batch_size, dim, dim)), True
    if factor.ndim == 1:
        if factor.shape != (dim,):
            raise ValueError(f"diagonal noise_factor must have shape {(dim,)}, got {factor.shape}")
        factor = jnp.diag(factor)
        return jnp.broadcast_to(factor, (batch_size, dim, dim)), True
    if factor.ndim == 2:
        if factor.shape != (dim, dim):
            raise ValueError(f"full noise_factor must have shape {(dim, dim)}, got {factor.shape}")
        return jnp.broadcast_to(factor, (batch_size, dim, dim)), True
    if factor.ndim == 3 and factor.shape == (batch_size, dim, dim):
        return factor, False
    raise ValueError("noise_factor must be scalar, diagonal [d], full [d,d], or batched [B,d,d]")


def _factor_validity(
    factor: Array,
    min_singular_value: float,
    factor_is_shared: bool,
) -> Array:
    if factor_is_shared:
        finite = jnp.all(jnp.isfinite(factor[0]))
        singular_values = jnp.linalg.svd(factor[0], compute_uv=False)
        full_rank = jnp.min(singular_values) >= min_singular_value
        return jnp.broadcast_to(finite & full_rank, (factor.shape[0],))
    finite = jnp.all(jnp.isfinite(factor), axis=(-2, -1))
    singular_values = jnp.linalg.svd(factor, compute_uv=False)
    full_rank = jnp.min(singular_values, axis=-1) >= min_singular_value
    return finite & full_rank


def _broadcast_query_inputs(
    samples: Array,
    time: Array,
    context: Array,
) -> tuple[Array, Array, Array]:
    batch_size, num_samples, dim = samples.shape
    flat_samples = samples.reshape((batch_size * num_samples, dim))
    flat_time = jnp.broadcast_to(time[:, None], (batch_size, num_samples)).reshape(-1)
    flat_context = jnp.broadcast_to(
        context[:, None, :],
        (batch_size, num_samples, context.shape[-1]),
    ).reshape((batch_size * num_samples, context.shape[-1]))
    return flat_samples, flat_time, flat_context


def _evaluate_cost(
    cost: ValueOnlyRunningCost,
    samples: Array,
    time: Array,
    context: Array,
) -> Array:
    flat_samples, flat_time, flat_context = _broadcast_query_inputs(samples, time, context)
    values = jnp.asarray(cost(flat_samples, flat_time, flat_context), dtype=samples.dtype)
    expected = (flat_samples.shape[0],)
    if values.shape != expected:
        raise ValueError(f"cost must return shape {expected}, got {values.shape}")
    values = values.reshape(samples.shape[:2])
    return jax.lax.stop_gradient(values)


def _stein_weights(factor: Array, xi: Array, factor_is_shared: bool) -> Array:
    """Return ``L^{-T} xi`` for factors ``[B,d,d]`` and noise ``[B,M,d]``."""
    transposed = jnp.swapaxes(factor, -1, -2)
    if factor_is_shared:
        batch_size, num_samples, dim = xi.shape
        right_hand_sides = xi.reshape((batch_size * num_samples, dim)).T
        solved = jnp.linalg.solve(transposed[0], right_hand_sides).T
        return jnp.asarray(solved.reshape((batch_size, num_samples, dim)))
    # One factorization per batch member, with all Monte Carlo draws as RHSs.
    right_hand_sides = jnp.swapaxes(xi, 1, 2)
    return jnp.swapaxes(jnp.linalg.solve(transposed, right_hand_sides), 1, 2)


def _antithetic_standard_error(contributions: Array) -> tuple[Array, bool]:
    num_pairs = contributions.shape[1]
    if num_pairs == 1:
        return jnp.full(contributions.shape[::2], jnp.inf, dtype=contributions.dtype), False
    centered = contributions - jnp.mean(contributions, axis=1, keepdims=True)
    variance = jnp.sum(centered**2, axis=1) / (num_pairs - 1)
    return jnp.sqrt(jnp.maximum(variance, 0.0) / num_pairs), True


def _iid_cross_covariance(values: Array, weights: Array) -> Array:
    """Unbiased sample covariance between scalar values and score weights."""
    num_samples = values.shape[1]
    sum_values = jnp.sum(values, axis=1)
    sum_weights = jnp.sum(weights, axis=1)
    sum_products = jnp.sum(values[..., None] * weights, axis=1)
    return (sum_products - sum_values[:, None] * sum_weights / num_samples) / (num_samples - 1)


def _iid_jackknife_standard_error(values: Array, weights: Array) -> tuple[Array, bool]:
    """Delete-one jackknife standard error for the unbiased covariance label."""
    num_samples = values.shape[1]
    if num_samples < 3:
        return jnp.full(weights.shape[::2], jnp.inf, dtype=weights.dtype), False

    sum_values = jnp.sum(values, axis=1, keepdims=True)
    sum_weights = jnp.sum(weights, axis=1, keepdims=True)
    products = values[..., None] * weights
    sum_products = jnp.sum(products, axis=1, keepdims=True)
    retained_count = num_samples - 1
    delete_one = (
        sum_products
        - products
        - (sum_values - values)[..., None] * (sum_weights - weights) / retained_count
    ) / (retained_count - 1)
    jackknife_mean = jnp.mean(delete_one, axis=1, keepdims=True)
    variance = (
        (num_samples - 1)
        / num_samples
        * jnp.sum(
            (delete_one - jackknife_mean) ** 2,
            axis=1,
        )
    )
    return jnp.sqrt(jnp.maximum(variance, 0.0)), True


def _sanitize_batch(
    gradient: Array,
    standard_error: Array,
    oracle_finite: Array,
    factor_valid: Array,
    standard_error_available: bool,
) -> tuple[Array, Array, Array]:
    finite = oracle_finite & factor_valid & jnp.all(jnp.isfinite(gradient), axis=-1)
    if standard_error_available:
        finite = finite & jnp.all(jnp.isfinite(standard_error), axis=-1)
    safe_gradient = jnp.where(finite[:, None], gradient, jnp.zeros_like(gradient))
    safe_standard_error = jnp.where(
        finite[:, None],
        standard_error,
        jnp.full_like(standard_error, jnp.inf),
    )
    return (
        jax.lax.stop_gradient(safe_gradient),
        jax.lax.stop_gradient(safe_standard_error),
        jax.lax.stop_gradient(finite),
    )


class MollifiedGradientEstimator:
    """Estimate local chain-induced Gaussian-convolution gradients."""

    def __init__(self, config: MollifiedGradientConfig | None = None):
        self.config = config or MollifiedGradientConfig()

    def estimate(
        self,
        key: PRNGKey,
        cost: ValueOnlyRunningCost | AnalyticMollifiableCost,
        mean: Array,
        noise_factor: Array | Scalar,
        time: Array | Scalar,
        context: Array | None,
    ) -> MollifiedGradientBatch:
        """Estimate ``grad_mean E[cost(mean + noise_factor @ xi)]``.

        All state queries are flattened to ``[Q,d]`` before calling ``cost``.
        Thus an oracle sees the same signature for shared or batched factors.
        Invalid dynamic rows are returned with a zero gradient, infinite
        standard error, and ``finite=False``.  Concrete singular factors fail
        eagerly instead of silently using a pseudoinverse.
        """
        info = mollified_gradient_backend_info(self.config.backend)
        if not info.implemented:
            raise NotImplementedError(
                f"backend {info.name!r} is declared approximate but is not implemented in V1"
            )

        mean = _normalize_mean(mean)
        batch_size, dim = mean.shape
        time = _normalize_time(time, batch_size, mean.dtype)
        context = _normalize_context(context, batch_size, mean.dtype)
        factor, factor_is_shared = _normalize_noise_factor(
            noise_factor,
            batch_size=batch_size,
            dim=dim,
            dtype=mean.dtype,
        )
        factor_valid = _factor_validity(
            factor,
            self.config.min_singular_value,
            factor_is_shared=factor_is_shared,
        )
        _require_concrete(
            jnp.all(factor_valid),
            "noise_factor must be finite and full-rank above min_singular_value",
        )
        input_finite = (
            jnp.all(jnp.isfinite(mean), axis=-1)
            & jnp.isfinite(time)
            & jnp.all(jnp.isfinite(context), axis=-1)
        )

        if self.config.backend == "analytic":
            if not isinstance(cost, AnalyticMollifiableCost):
                raise TypeError(
                    "analytic backend requires an AnalyticMollifiableCost implementing "
                    "analytic_mollified_gradient"
                )
            if registered_analytic_mollifier(cost) is None:
                raise TypeError(
                    "analytic backend accepts only audited registered formulas; "
                    "use a Stein backend for an unregistered cost"
                )
            gradient = jnp.asarray(
                cost.analytic_mollified_gradient(mean, factor, time, context),
                dtype=mean.dtype,
            )
            if gradient.shape != mean.shape:
                raise ValueError(
                    f"analytic gradient must have shape {mean.shape}, got {gradient.shape}"
                )
            standard_error = jnp.zeros_like(gradient)
            gradient, standard_error, finite = _sanitize_batch(
                gradient,
                standard_error,
                oracle_finite=input_finite,
                factor_valid=factor_valid,
                standard_error_available=True,
            )
            return MollifiedGradientBatch(
                gradient=gradient,
                standard_error=standard_error,
                physical_query_count=jnp.asarray(0, dtype=jnp.int32),
                finite=finite,
                backend_code=jnp.asarray(int(info.code), dtype=jnp.int32),
                exact=jnp.asarray(info.exact_in_expectation),
                standard_error_available=jnp.asarray(True),
            )

        xi = jax.random.normal(
            key,
            shape=(batch_size, self.config.num_samples, dim),
            dtype=mean.dtype,
        )
        perturbation = jnp.einsum("bij,bmj->bmi", factor, xi)
        weights = _stein_weights(factor, xi, factor_is_shared=factor_is_shared)
        value_cost = cast(ValueOnlyRunningCost, cost)

        if self.config.backend == "antithetic_stein":
            positive_values = _evaluate_cost(
                value_cost, mean[:, None, :] + perturbation, time, context
            )
            negative_values = _evaluate_cost(
                value_cost, mean[:, None, :] - perturbation, time, context
            )
            contributions = 0.5 * (positive_values - negative_values)[..., None] * weights
            gradient = jnp.mean(contributions, axis=1)
            standard_error, standard_error_available = _antithetic_standard_error(contributions)
            oracle_finite = input_finite & jnp.all(
                jnp.isfinite(positive_values) & jnp.isfinite(negative_values), axis=1
            )
            query_count = 2 * batch_size * self.config.num_samples
        else:
            values = _evaluate_cost(value_cost, mean[:, None, :] + perturbation, time, context)
            gradient = _iid_cross_covariance(values, weights)
            standard_error, standard_error_available = _iid_jackknife_standard_error(
                values, weights
            )
            oracle_finite = input_finite & jnp.all(jnp.isfinite(values), axis=1)
            query_count = batch_size * self.config.num_samples

        gradient, standard_error, finite = _sanitize_batch(
            gradient,
            standard_error,
            oracle_finite=oracle_finite,
            factor_valid=factor_valid,
            standard_error_available=standard_error_available,
        )
        return MollifiedGradientBatch(
            gradient=gradient,
            standard_error=standard_error,
            physical_query_count=jnp.asarray(query_count, dtype=jnp.int32),
            finite=finite,
            backend_code=jnp.asarray(int(info.code), dtype=jnp.int32),
            exact=jnp.asarray(info.exact_in_expectation),
            standard_error_available=jnp.asarray(standard_error_available),
        )


__all__ = [
    "AnalyticMollifiableCost",
    "GaussianHalfspaceCost",
    "MollifiedGradientBackend",
    "MollifiedGradientBackendInfo",
    "MollifiedGradientBatch",
    "MollifiedGradientConfig",
    "MollifiedGradientEstimator",
    "registered_analytic_mollifier",
    "ValueOnlyRunningCost",
    "mollified_gradient_backend_info",
]
