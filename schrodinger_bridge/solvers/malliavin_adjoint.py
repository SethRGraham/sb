"""Value-only Malliavin costates for conditional bridge control.

This module implements the first, deliberately narrow, Malliavin Adjoint
Matching (MAM) component.  It is a conditional stochastic-control *inner*
solver, not a complete generalized Schrodinger bridge solver.  In particular,
a Hamiltonian control proposal does not by itself preserve a population target
marginal; a reciprocal/Markov projection outer loop is still required.

The theorem-facing implementation uses cost minimization, constant square
full-rank diffusion, additive Markov running values, and adapted BEL/Itô
weights.  Cost values are always stopped before label assembly.  The generic
EM reference kernel supports terminal values; in an endpoint-pinned inner
problem an endpoint-only terminal cost is constant and belongs to the future
outer coupling update, not the conditional costate label.

For Euler--Maruyama, the exact declared discrete weight is

    H[k,m] = 1 / ((m-k) dt)
             sum_{j=k}^{m-1} (Sigma^+ J[j+1,k])^T dW[j].

The arrival-flow Jacobian ``J[j+1,k]`` and the transpose solve are important:
using ``J[j,k]`` or ``solve(Sigma, dW)`` is generally biased at finite step
size.  Brownian/scalar tests cannot expose either error, so the test suite also
contains linear-drift and nonsymmetric-matrix calibrations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from numbers import Integral, Real
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..core.problem import BrownianMotion, SBProblem
from ..core.types import Array, Params, PRNGKey, Scalar
from ..network_factory import MLPFactory, NetworkFactory, sanity_check
from ..networks import AdamState, adam_update, init_adam

RunningCostFn = Callable[[Array, Array, Array], Array]
TerminalCostFn = Callable[[Array, Array], Array]
DriftFnWithContext = Callable[[Array, Scalar, Array], Array]
NoiseControlFn = Callable[[Array, Scalar, Array], Array]
_RHO_CONSISTENCY_EPS_MULTIPLIER = 32.0
ANTITHETIC_PINNED_ARRIVAL_ESTIMATOR = 0
ANTITHETIC_DIRECT_RETURN_SCORE_ESTIMATOR = 1


@dataclass(frozen=True)
class ValueOnlyCost:
    """Value-only additive cost oracle.

    ``running_cost(x, t, context)`` receives batched states ``[B,d]``, times
    ``[B]``, and context ``[B,c]`` and returns ``[B]``.  ``terminal_cost``
    receives ``(x_terminal, context)``.  The functions may be nonsmooth, but
    the current JAX implementation requires them to be JAX-evaluable.  A host
    simulator can instead supply already evaluated arrays to the pure label
    functions.  ``terminal_cost`` is supported by the generic EM reference
    kernel, but the endpoint-pinned inner solver rejects it because the fixed
    endpoint makes its interior-state derivative zero.
    """

    running_cost: RunningCostFn | None = None
    terminal_cost: TerminalCostFn | None = None
    identifier: str = "zero_cost"

    def __post_init__(self) -> None:
        if self.running_cost is not None and not callable(self.running_cost):
            raise TypeError("running_cost must be callable or None")
        if self.terminal_cost is not None and not callable(self.terminal_cost):
            raise TypeError("terminal_cost must be callable or None")
        if not isinstance(self.identifier, str) or not self.identifier:
            raise ValueError("identifier must be a nonempty string")

    def running_values(self, states: Array, times: Array, context: Array) -> Array:
        """Evaluate stopped running values on ``[B,N+1,d]`` states."""
        batch_size, num_times, dim = states.shape
        del dim
        context = _broadcast_context(context, batch_size)
        if self.running_cost is None:
            return jnp.zeros((batch_size, num_times), dtype=states.dtype)
        flat_states = states.reshape((-1, states.shape[-1]))
        flat_times = jnp.broadcast_to(times[None, :], (batch_size, num_times)).reshape(-1)
        flat_context = jnp.broadcast_to(
            context[:, None, :],
            (batch_size, num_times, context.shape[-1]),
        ).reshape((-1, context.shape[-1]))
        running_cost = self.running_cost

        def one(state: Array, time: Array, row_context: Array) -> Array:
            value = jnp.asarray(
                running_cost(
                    state[None, :],
                    jnp.reshape(time, (1,)),
                    row_context[None, :],
                ),
                dtype=states.dtype,
            )
            if value.shape != (1,):
                raise ValueError(
                    f"running_cost must return shape {(1,)} for one state row, got {value.shape}"
                )
            return value[0]

        values = jax.vmap(one)(flat_states, flat_times, flat_context).reshape(
            (batch_size, num_times)
        )
        return jax.lax.stop_gradient(values)

    def terminal_values(self, states: Array, context: Array) -> Array:
        """Evaluate stopped terminal values on terminal states ``[B,d]``."""
        states = jnp.atleast_2d(states)
        context = _broadcast_context(context, states.shape[0])
        if self.terminal_cost is None:
            return jnp.zeros((states.shape[0],), dtype=states.dtype)
        terminal_cost = self.terminal_cost

        def one(state: Array, row_context: Array) -> Array:
            value = jnp.asarray(
                terminal_cost(state[None, :], row_context[None, :]),
                dtype=states.dtype,
            )
            if value.shape != (1,):
                raise ValueError(
                    f"terminal_cost must return shape {(1,)} for one state row, got {value.shape}"
                )
            return value[0]

        values = jax.vmap(one)(states, context)
        return jax.lax.stop_gradient(values)


@dataclass
class MalliavinAdjointConfig:
    """Configuration for the conditional MAM inner solver."""

    hidden_dims: tuple[int, ...] = (64, 64)
    time_embed_dim: int = 32
    learning_rate: float = 1e-3
    training_steps: int = 1_000
    batch_size: int = 256
    minimum_remaining_steps: int = 2
    anchor_sampling: str = "stratified"
    ema_decay: float = 0.999
    trust_region: float = 0.1
    max_control_norm: float | None = None
    include_control_energy: bool = True
    matrix_free_labels: bool = True
    center_running_values: bool = True
    diffusion_rcond: float = 1e-8
    network_factory: NetworkFactory | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hidden_dims, tuple)
            or not self.hidden_dims
            or any(not _is_strict_integer(width) or width < 1 for width in self.hidden_dims)
        ):
            raise ValueError("hidden_dims must contain positive widths")
        if not _is_strict_integer(self.time_embed_dim) or self.time_embed_dim < 2:
            raise ValueError("time_embed_dim must be at least two")
        if not _is_finite_real(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if (
            not _is_strict_integer(self.training_steps)
            or not _is_strict_integer(self.batch_size)
            or self.training_steps < 1
            or self.batch_size < 1
        ):
            raise ValueError("training_steps and batch_size must be positive")
        if not _is_strict_integer(self.minimum_remaining_steps) or self.minimum_remaining_steps < 1:
            raise ValueError("minimum_remaining_steps must be positive")
        if not isinstance(self.anchor_sampling, str) or self.anchor_sampling not in {
            "stratified",
            "iid_uniform",
        }:
            raise ValueError("anchor_sampling must be 'stratified' or 'iid_uniform'")
        if not _is_finite_real(self.ema_decay) or not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not _is_finite_real(self.trust_region) or not 0.0 <= self.trust_region <= 1.0:
            raise ValueError("trust_region must be in [0, 1]")
        if self.max_control_norm is not None and (
            not _is_finite_real(self.max_control_norm) or self.max_control_norm <= 0
        ):
            raise ValueError("max_control_norm must be positive when supplied")
        for name in ("include_control_energy", "matrix_free_labels", "center_running_values"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if not _is_finite_real(self.diffusion_rcond) or self.diffusion_rcond <= 0:
            raise ValueError("diffusion_rcond must be positive")


class AdjointRollout(NamedTuple):
    """Fixed-innovation rollout data used by the pure BEL label kernels.

    Shapes:
        states: ``[B,N+1,d]``
        innovations: ``[B,S,d]``; ``dW`` for EM and standard normal for pinned
        local_jacobians: ``[B,S,d,d]``
        noise_matrices: ``[S,d,d]`` mapping innovations to state increments
        controls: ``[B,S,d]`` in noise coordinates
        times: ``[N+1]``
        context: ``[B,c]``; pinned endpoints use ``c=d``
    """

    states: Array
    innovations: Array
    local_jacobians: Array
    noise_matrices: Array
    controls: Array
    times: Array
    context: Array


class MatrixFreeAdjointRollout(NamedTuple):
    """Pinned rollout without materialized state-transition Jacobians.

    Shapes match :class:`AdjointRollout` except that no ``[B,N,d,d]`` local
    Jacobian tensor is stored.  Transition VJPs are recomputed during label
    assembly, optionally under rematerialization.  This is the production
    representation for state dimensions where dense tangent storage is
    prohibitive.
    """

    states: Array
    innovations: Array
    noise_matrices: Array
    controls: Array
    times: Array
    context: Array


class BELCostateBatch(NamedTuple):
    """One stopped BEL costate label per trajectory."""

    anchor_index: Array
    anchor_time: Array
    anchor_state: Array
    context: Array
    label: Array
    terminal_component: Array
    running_component: Array
    direct_component: Array
    terminal_weight: Array
    tangent_condition_number: Array
    finite: Array


class ActionTargetBatch(NamedTuple):
    """Stopped arrival-aware targets for one pinned control transition.

    ``target`` is in Brownian/noise-control coordinates.  The continuation
    and arrival components sum to it.  ``innovation`` stores the base
    antithetic innovations with shape ``[B,M,d]``; the negative branch is
    implicit.  A target is exact for the declared current-policy residual
    only when ``next_costate`` is the exact continuation costate.
    """

    target: Array
    continuation_component: Array
    arrival_component: Array
    mean_state: Array
    innovation: Array
    finite: Array
    physical_oracle_queries: Array
    estimator: Array


class DirectActionScoreBatch(NamedTuple):
    """Tangent-free full-return action-score baseline.

    ``positive_return`` and ``negative_return`` must be independently valid
    suffix-return evaluations whose first transition innovations are ``xi``
    and ``-xi`` respectively.  They may share all later random numbers as a
    variance-reducing common-random-number coupling.
    """

    target: Array
    finite: Array
    physical_return_queries: Array
    estimator: Array


class ControlProposal(NamedTuple):
    """A conservative Hamiltonian control proposal.

    This proposal is not an endpoint projection.  ``endpoint_preserved`` is
    intentionally always false unless a future outer bridge algorithm proves
    and validates that property.
    """

    current_control: Array
    target_control: Array
    proposed_control: Array
    costate: Array
    step_size: Array
    endpoint_preserved: Array
    convention: str
    coordinates: str
    update_semantics: str
    outer_projection_required: bool


class MalliavinAdjointResult(NamedTuple):
    """Training result for the conditional costate regressor."""

    params: Params
    ema_params: Params
    loss_history: Array
    final_metrics: dict[str, Array]


def _is_strict_integer(value: Any) -> bool:
    """Return whether ``value`` is an integer, excluding booleans."""
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _is_finite_real(value: Any) -> bool:
    """Return whether ``value`` is a finite real scalar, excluding booleans."""
    return (
        isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(float(value)))
    )


def _validate_positive_finite_real(value: Any, name: str) -> float:
    """Validate a static positive scalar used in numerical preconditions."""
    if not _is_finite_real(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return float(value)


def _broadcast_context(context: Array | None, batch_size: int) -> Array:
    if context is None:
        return jnp.zeros((batch_size, 0))
    context = jnp.asarray(context)
    if context.ndim == 1:
        context = context[None, :]
    if context.shape[0] == 1 and batch_size != 1:
        context = jnp.broadcast_to(context, (batch_size, context.shape[-1]))
    if context.shape[0] != batch_size:
        raise ValueError("context batch size must be one or match the states")
    return context


def _as_batch_vector(value: Array | Scalar, batch_size: int, name: str, dtype: Any) -> Array:
    """Normalize a scalar or ``[B]`` value without accepting ambiguous shapes."""
    array = jnp.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return jnp.full((batch_size,), array, dtype=dtype)
    if array.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar or have shape {(batch_size,)}, got {array.shape}")
    return array


def _require_concrete(predicate: Array, message: str) -> None:
    """Raise eagerly when a concrete numerical precondition is false."""
    if isinstance(predicate, jax.core.Tracer):
        return
    if not bool(np.all(np.asarray(jax.device_get(predicate)))):
        raise ValueError(message)


def _tree_all_finite(tree: Any) -> Array:
    """Return one JAX boolean for all numerical leaves in a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    checks = [jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves]
    return jnp.all(jnp.stack(checks))


def _constant_diffusion_matrix(diffusion: Array | Scalar, dim: int, dtype: Any) -> Array:
    sigma = jnp.asarray(diffusion, dtype=dtype)
    if sigma.ndim == 0:
        return sigma * jnp.eye(dim, dtype=dtype)
    if sigma.ndim == 1:
        if sigma.shape != (dim,):
            raise ValueError(f"diagonal diffusion must have shape {(dim,)}, got {sigma.shape}")
        return jnp.diag(sigma)
    if sigma.ndim == 2 and sigma.shape == (dim, dim):
        return sigma
    raise ValueError("MAM v1 requires scalar, diagonal [d], or constant square [d,d] diffusion")


def _validate_full_rank(sigma: Array, rcond: float = 1e-8) -> None:
    """Fail eagerly for concrete singular/ill-conditioned V1 diffusion."""
    rcond = _validate_positive_finite_real(rcond, "diffusion_rcond")
    if isinstance(sigma, jax.core.Tracer):
        return
    singular_values = np.linalg.svd(np.asarray(jax.device_get(sigma)), compute_uv=False)
    if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
        raise ValueError("diffusion singular values must be finite")
    if singular_values[-1] <= rcond * singular_values[0]:
        raise ValueError("MAM v1 requires a full-rank, well-conditioned diffusion matrix")


def _full_rank_validity(sigma: Array, rcond: float) -> Array:
    """JAX-native full-rank/finiteness predicate for a square matrix."""
    singular_values = jnp.linalg.svd(sigma, compute_uv=False)
    return (
        jnp.all(jnp.isfinite(sigma))
        & jnp.all(jnp.isfinite(singular_values))
        & (jnp.min(singular_values) > rcond * jnp.max(singular_values))
    )


def _uniform_grid_validity(times: Array) -> tuple[Scalar, Array, Array, Array]:
    """Return ``(dt, valid, increasing, uniform)`` using only JAX operations.

    Shape and dtype checks are static, so they also fail while tracing.  The
    numerical predicates remain available to callers under ``jax.jit`` and
    are used to poison rollout outputs or mark label rows invalid.
    """
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("times must be one-dimensional with at least two points")
    if not jnp.issubdtype(times.dtype, jnp.floating):
        raise ValueError("times must have a floating dtype")
    increments = jnp.diff(times)
    finite = jnp.all(jnp.isfinite(times)) & jnp.all(jnp.isfinite(increments))
    increasing = finite & jnp.all(increments > 0.0)
    interval_count = jnp.asarray(times.shape[0] - 1, dtype=times.dtype)
    dt = (times[-1] - times[0]) / interval_count
    # ``linspace`` rounds adjacent float32 intervals slightly differently.
    # Scale the tolerance to the represented interval size.  A unit-scale
    # absolute floor would accept arbitrarily nonuniform short-horizon grids
    # in float32 (for example, increments 1e-8 and 9e-8).
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(dt), jnp.max(jnp.abs(increments))),
        jnp.asarray(jnp.finfo(times.dtype).tiny, dtype=times.dtype),
    )
    tolerance = jnp.asarray(64.0 * jnp.finfo(times.dtype).eps, dtype=times.dtype) * scale
    uniform = finite & jnp.isfinite(dt) & jnp.all(jnp.abs(increments - dt) <= tolerance)
    return dt, increasing & uniform, increasing, uniform


def _validate_uniform_times(times: Array) -> tuple[Scalar, Array]:
    """Validate a time grid eagerly and retain its validity under tracing."""
    dt, valid, increasing, uniform = _uniform_grid_validity(times)
    _require_concrete(jnp.all(jnp.isfinite(times)), "times must be finite")
    _require_concrete(increasing, "times must be strictly increasing")
    _require_concrete(uniform, "MAM v1 requires a uniform time grid")
    return dt, valid


def _normalize_anchors(
    anchors: Array,
    *,
    batch_size: int,
    maximum: int,
) -> tuple[Array, Array, Array]:
    """Validate integer anchors and provide safe indices plus row validity."""
    raw = jnp.asarray(anchors)
    if raw.ndim != 1:
        raise ValueError("anchors must have shape [batch]")
    if raw.shape != (batch_size,):
        raise ValueError(f"anchors must have shape {(batch_size,)}, got {raw.shape}")
    if jnp.issubdtype(raw.dtype, jnp.bool_) or not jnp.issubdtype(raw.dtype, jnp.integer):
        raise ValueError("anchors must have an integer dtype (boolean and floating anchors fail)")
    row_valid = (raw >= 0) & (raw <= maximum)
    _require_concrete(jnp.all(row_valid), f"anchors must lie in [0, {maximum}]")
    safe = jnp.clip(raw, 0, maximum)
    return raw, safe, row_valid


def _poison_float(array: Array, valid: Array) -> Array:
    """Replace a floating result with NaNs when a dynamic precondition fails."""
    return jnp.where(valid, array, jnp.full_like(array, jnp.nan))


def _single_drift_value(
    drift_fn: DriftFnWithContext,
    x: Array,
    t: Scalar,
    context: Array,
) -> Array:
    """Evaluate one row under the callback contract used for differentiation."""
    value = jnp.asarray(drift_fn(x[None, :], t, context[None, :]), dtype=x.dtype)
    expected = (1, x.shape[0])
    if value.shape != expected:
        raise ValueError(f"drift_fn must return shape {expected}, got {value.shape}")
    return value[0]


def _batch_drift_values(
    drift_fn: DriftFnWithContext,
    x: Array,
    t: Scalar,
    context: Array,
) -> Array:
    """Evaluate a drift independently row by row to match its local Jacobian."""
    return jax.vmap(lambda x_i, context_i: _single_drift_value(drift_fn, x_i, t, context_i))(
        x, context
    )


def _batch_drift_jacobian(
    drift_fn: DriftFnWithContext,
    x: Array,
    t: Scalar,
    context: Array,
) -> Array:
    def one(x_i: Array, context_i: Array) -> Array:
        return jnp.asarray(
            jax.jacfwd(lambda value: _single_drift_value(drift_fn, value, t, context_i))(x_i)
        )

    return jax.vmap(one)(x, context)


def _single_control_value(
    control_fn: NoiseControlFn,
    x: Array,
    t: Scalar,
    endpoint: Array,
) -> Array:
    """Evaluate one control row with an exact state-dimensional output."""
    value = jnp.asarray(control_fn(x[None, :], t, endpoint[None, :]), dtype=x.dtype)
    expected = (1, x.shape[0])
    if value.shape != expected:
        raise ValueError(f"control_fn must return shape {expected}, got {value.shape}")
    return value[0]


def _batch_control_values(
    control_fn: NoiseControlFn,
    x: Array,
    t: Scalar,
    endpoint: Array,
) -> Array:
    """Evaluate controls independently row by row, matching the VJP path."""
    return jax.vmap(lambda x_i, endpoint_i: _single_control_value(control_fn, x_i, t, endpoint_i))(
        x, endpoint
    )


def _batch_control_jacobian(
    control_fn: NoiseControlFn,
    x: Array,
    t: Scalar,
    endpoint: Array,
) -> Array:
    def one(x_i: Array, endpoint_i: Array) -> Array:
        return jnp.asarray(
            jax.jacfwd(lambda value: _single_control_value(control_fn, value, t, endpoint_i))(x_i)
        )

    return jax.vmap(one)(x, endpoint)


def simulate_additive_em_rollout(
    key: PRNGKey,
    x0: Array,
    times: Array,
    drift_fn: DriftFnWithContext,
    diffusion: Array | Scalar,
    context: Array | None = None,
    *,
    diffusion_rcond: float = 1e-8,
) -> AdjointRollout:
    """Simulate additive Euler--Maruyama and record exact local tangents.

    ``innovations`` stores Brownian increments with covariance ``dt I``.
    The local Jacobian includes every state dependence inside ``drift_fn``;
    callers should therefore close a differentiable feedback policy into that
    drift instead of stopping the policy-state derivative.
    """
    diffusion_rcond = _validate_positive_finite_real(diffusion_rcond, "diffusion_rcond")
    x0 = jnp.atleast_2d(x0)
    if x0.ndim != 2 or not jnp.issubdtype(x0.dtype, jnp.floating):
        raise ValueError("x0 must have shape [batch, state_dim] and a floating dtype")
    times = jnp.asarray(times, dtype=x0.dtype)
    dt, grid_valid = _validate_uniform_times(times)
    safe_dt = jnp.where(grid_valid, dt, jnp.asarray(1.0, dtype=x0.dtype))
    batch_size, dim = x0.shape
    normalized_context = _broadcast_context(context, batch_size)
    sigma = _constant_diffusion_matrix(diffusion, dim, x0.dtype)
    _validate_full_rank(sigma, diffusion_rcond)
    diffusion_valid = _full_rank_validity(sigma, diffusion_rcond)
    row_input_valid = (
        jnp.all(jnp.isfinite(x0), axis=-1)
        & jnp.all(jnp.isfinite(normalized_context), axis=-1)
        & grid_valid
        & diffusion_valid
    )
    _require_concrete(jnp.all(row_input_valid), "nonfinite or invalid EM rollout input")

    num_steps = times.shape[0] - 1
    d_w = jnp.sqrt(safe_dt) * jax.random.normal(
        key,
        (batch_size, num_steps, dim),
        dtype=x0.dtype,
    )
    scan_d_w = jnp.swapaxes(d_w, 0, 1)
    eye = jnp.eye(dim, dtype=x0.dtype)

    def step(x: Array, inputs: tuple[Scalar, Array]):
        t, d_w_step = inputs
        drift = _batch_drift_values(drift_fn, x, t, normalized_context)
        jacobian = _batch_drift_jacobian(drift_fn, x, t, normalized_context)
        local = eye[None, :, :] + safe_dt * jacobian
        x_next = x + safe_dt * drift + d_w_step @ sigma.T
        return x_next, (x_next, local)

    _, (states_after, local_time_major) = jax.lax.scan(
        step,
        x0,
        (times[:-1], scan_d_w),
    )
    states = jnp.concatenate(
        [x0[:, None, :], jnp.swapaxes(states_after, 0, 1)],
        axis=1,
    )
    local = jnp.swapaxes(local_time_major, 0, 1)
    noise_matrices = jnp.broadcast_to(sigma[None, :, :], (num_steps, dim, dim))
    controls = jnp.zeros((batch_size, num_steps, dim), dtype=x0.dtype)
    row_valid = (
        row_input_valid
        & jnp.all(jnp.isfinite(states), axis=(1, 2))
        & jnp.all(jnp.isfinite(d_w), axis=(1, 2))
        & jnp.all(jnp.isfinite(local), axis=(1, 2, 3))
    )
    _require_concrete(jnp.all(row_valid), "nonfinite EM rollout result")
    states = _poison_float(states, row_valid[:, None, None])
    d_w = _poison_float(d_w, row_valid[:, None, None])
    local = _poison_float(local, row_valid[:, None, None, None])
    controls = _poison_float(controls, row_valid[:, None, None])
    normalized_context = _poison_float(normalized_context, row_valid[:, None])
    noise_matrices = _poison_float(noise_matrices, grid_valid & diffusion_valid)
    return AdjointRollout(
        states,
        d_w,
        local,
        noise_matrices,
        controls,
        times,
        normalized_context,
    )


def _em_label_one(
    local_jacobians: Array,
    d_w: Array,
    sigma_steps: Array,
    running_values: Array,
    terminal_value: Array,
    anchor: Array,
    dt: Scalar,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    dim = d_w.shape[-1]
    eye = jnp.eye(dim, dtype=d_w.dtype)
    indices = jnp.arange(d_w.shape[0])

    def step(carry: tuple[Array, Array, Array, Array], inputs: tuple[Array, ...]):
        tangent, integral, running, maximum_condition = carry
        index, local, increment, sigma = inputs
        active = index >= anchor
        tangent_candidate = local @ tangent
        # (Sigma^{-1} J)^T dW == J^T Sigma^{-T} dW.
        sigma_transpose_solve = jnp.linalg.solve(sigma.T, increment)
        integral_candidate = integral + tangent_candidate.T @ sigma_transpose_solve
        count = index + 1 - anchor
        denominator = jnp.maximum(count, 1) * dt
        weight = integral_candidate / denominator
        running_candidate = running + dt * running_values[index + 1] * weight
        condition_candidate = jnp.nan_to_num(
            jnp.linalg.cond(tangent_candidate),
            nan=jnp.inf,
            posinf=jnp.inf,
        )
        tangent = jnp.where(active, tangent_candidate, tangent)
        integral = jnp.where(active, integral_candidate, integral)
        running = jnp.where(active, running_candidate, running)
        maximum_condition = jnp.where(
            active,
            jnp.maximum(maximum_condition, condition_candidate),
            maximum_condition,
        )
        return (tangent, integral, running, maximum_condition), weight

    initial = (
        eye,
        jnp.zeros((dim,), dtype=d_w.dtype),
        jnp.zeros((dim,), dtype=d_w.dtype),
        jnp.asarray(1.0, dtype=d_w.dtype),
    )
    (tangent, integral, running, max_condition), _ = jax.lax.scan(
        step,
        initial,
        (indices, local_jacobians, d_w, sigma_steps),
    )
    del tangent
    horizon_steps = d_w.shape[0] - anchor
    terminal_weight = integral / (jnp.maximum(horizon_steps, 1) * dt)
    terminal = terminal_value * terminal_weight
    label = terminal + running
    # A singular state tangent can yield a perfectly finite (often zero) BEL
    # label.  Only the diffusion matrix must be invertible in V1; tangent
    # conditioning is reported as a diagnostic, not used as a validity veto.
    finite = jnp.all(jnp.isfinite(label))
    return label, terminal, running, terminal_weight, max_condition, finite


def assemble_bel_costate_labels(
    rollout: AdjointRollout,
    anchors: Array,
    running_values: Array,
    terminal_values: Array,
    immediate_gradients: Array | None = None,
) -> BELCostateBatch:
    """Assemble exact-discretization EM BEL labels from evaluated values.

    ``immediate_gradients`` is an optional, already quadrature-weighted
    derivative of a known differentiable cost at the anchor.  It is separate
    from the stopped value-only oracle.  This is needed, for example, for the
    current control-energy term, which has no future transition noise through
    which a BEL integration-by-parts weight could represent its derivative.
    """
    batch_size, num_times, dim = rollout.states.shape
    num_steps = num_times - 1
    anchors, safe_anchors, anchor_valid = _normalize_anchors(
        anchors,
        batch_size=batch_size,
        maximum=num_steps - 1,
    )
    if rollout.innovations.shape != (batch_size, num_steps, dim):
        raise ValueError("rollout innovations must have shape [batch, steps, state_dim]")
    if rollout.local_jacobians.shape != (batch_size, num_steps, dim, dim):
        raise ValueError("rollout local_jacobians must have shape [batch, steps, dim, dim]")
    if rollout.noise_matrices.shape != (num_steps, dim, dim):
        raise ValueError("rollout noise_matrices must have shape [steps, dim, dim]")
    if rollout.controls.shape != (batch_size, num_steps, dim):
        raise ValueError("rollout controls must have shape [batch, steps, state_dim]")
    if rollout.context.ndim != 2 or rollout.context.shape[0] != batch_size:
        raise ValueError("rollout context must have shape [batch, context_dim]")
    running_values = jnp.asarray(running_values, dtype=rollout.states.dtype)
    terminal_values = jnp.asarray(terminal_values, dtype=rollout.states.dtype)
    if running_values.shape != (batch_size, num_times):
        raise ValueError(
            f"running_values must have shape {(batch_size, num_times)}, got {running_values.shape}"
        )
    if terminal_values.shape != (batch_size,):
        raise ValueError(
            f"terminal_values must have shape {(batch_size,)}, got {terminal_values.shape}"
        )
    if immediate_gradients is None:
        immediate_gradients = jnp.zeros(
            (batch_size, rollout.states.shape[-1]), dtype=rollout.states.dtype
        )
    immediate_gradients = jnp.asarray(immediate_gradients, dtype=rollout.states.dtype)
    if immediate_gradients.shape != (batch_size, rollout.states.shape[-1]):
        raise ValueError("immediate_gradients must have shape [batch, state_dim]")

    dt, grid_valid = _validate_uniform_times(rollout.times)
    dt = jnp.asarray(dt, dtype=rollout.states.dtype)
    stopped_running = jax.lax.stop_gradient(running_values)
    stopped_terminal = jax.lax.stop_gradient(terminal_values)
    outputs = jax.vmap(_em_label_one, in_axes=(0, 0, None, 0, 0, 0, None))(
        rollout.local_jacobians,
        rollout.innovations,
        rollout.noise_matrices,
        stopped_running,
        stopped_terminal,
        safe_anchors,
        dt,
    )
    label, terminal, running, terminal_weight, condition, finite = outputs
    direct = jax.lax.stop_gradient(immediate_gradients)
    label = label + direct
    row_input_valid = (
        jnp.all(jnp.isfinite(rollout.states), axis=(1, 2))
        & jnp.all(jnp.isfinite(rollout.innovations), axis=(1, 2))
        & jnp.all(jnp.isfinite(rollout.local_jacobians), axis=(1, 2, 3))
        & jnp.all(jnp.isfinite(rollout.controls), axis=(1, 2))
        & jnp.all(jnp.isfinite(rollout.context), axis=-1)
        & jnp.all(jnp.isfinite(running_values), axis=-1)
        & jnp.isfinite(terminal_values)
        & jnp.all(jnp.isfinite(direct), axis=-1)
        & anchor_valid
        & grid_valid
        & jnp.all(jnp.isfinite(rollout.noise_matrices))
    )
    finite = finite & row_input_valid
    batch_indices = jnp.arange(batch_size)
    anchor_states = rollout.states[batch_indices, safe_anchors]
    anchor_times = rollout.times[safe_anchors]
    return BELCostateBatch(
        anchors,
        anchor_times,
        anchor_states,
        rollout.context,
        jax.lax.stop_gradient(label),
        jax.lax.stop_gradient(terminal),
        jax.lax.stop_gradient(running),
        direct,
        jax.lax.stop_gradient(terminal_weight),
        jax.lax.stop_gradient(condition),
        finite,
    )


def simulate_pinned_brownian_rollout(
    key: PRNGKey,
    x0: Array,
    endpoint: Array,
    times: Array,
    diffusion: Array | Scalar,
    control_fn: NoiseControlFn | None = None,
    *,
    diffusion_rcond: float = 1e-8,
) -> AdjointRollout:
    """Sample an exactly endpoint-pinned discrete Brownian bridge.

    For each stochastic transition, standard innovations ``xi`` are mapped by
    ``Gamma_n = sqrt(dt * rho_n) Sigma``.  The controlled mean shift is
    ``Gamma_n sqrt(dt) u_n``; consequently the transition KL increment is
    ``dt/2 ||u_n||^2``.  The final transition is deterministic and is omitted
    from the stored innovation/tangent arrays.
    """
    diffusion_rcond = _validate_positive_finite_real(diffusion_rcond, "diffusion_rcond")
    x0 = jnp.atleast_2d(x0)
    endpoint = jnp.atleast_2d(endpoint)
    if x0.ndim != 2 or not jnp.issubdtype(x0.dtype, jnp.floating):
        raise ValueError("x0 must have shape [batch, state_dim] and a floating dtype")
    if endpoint.shape != x0.shape:
        raise ValueError(f"endpoint shape {endpoint.shape} must match x0 {x0.shape}")
    endpoint = jnp.asarray(endpoint, dtype=x0.dtype)
    times = jnp.asarray(times, dtype=x0.dtype)
    dt, grid_valid = _validate_uniform_times(times)
    safe_dt = jnp.where(grid_valid, dt, jnp.asarray(1.0, dtype=x0.dtype))
    batch_size, dim = x0.shape
    sigma = _constant_diffusion_matrix(diffusion, dim, x0.dtype)
    _validate_full_rank(sigma, diffusion_rcond)
    diffusion_valid = _full_rank_validity(sigma, diffusion_rcond)
    num_steps = times.shape[0] - 1
    if num_steps < 2:
        raise ValueError("pinned bridge requires at least two time steps")
    stochastic_steps = num_steps - 1
    raw_rho = (times[-1] - times[1 : stochastic_steps + 1]) / (times[-1] - times[:stochastic_steps])
    rho_valid = jnp.all(jnp.isfinite(raw_rho) & (raw_rho > 0.0) & (raw_rho < 1.0))
    chain_valid = grid_valid & rho_valid
    safe_rho = jnp.where(chain_valid, raw_rho, jnp.full_like(raw_rho, 0.5))
    row_input_valid = (
        jnp.all(jnp.isfinite(x0), axis=-1)
        & jnp.all(jnp.isfinite(endpoint), axis=-1)
        & chain_valid
        & diffusion_valid
    )
    _require_concrete(jnp.all(row_input_valid), "nonfinite or invalid pinned rollout input")
    innovations = jax.random.normal(
        key,
        (batch_size, stochastic_steps, dim),
        dtype=x0.dtype,
    )
    scan_innovations = jnp.swapaxes(innovations, 0, 1)
    eye = jnp.eye(dim, dtype=x0.dtype)

    def zero_control(x: Array, t: Scalar, context: Array) -> Array:
        del t, context
        return jnp.zeros_like(x)

    active_control = control_fn or zero_control

    def step(x: Array, inputs: tuple[Scalar, Scalar, Array]):
        t, rho, innovation = inputs
        gamma = jnp.sqrt(safe_dt * rho) * sigma
        control = _batch_control_values(active_control, x, t, endpoint)
        control_jacobian = _batch_control_jacobian(active_control, x, t, endpoint)
        local = rho * eye[None, :, :] + jnp.sqrt(safe_dt) * jnp.einsum(
            "ij,bjk->bik", gamma, control_jacobian
        )
        mean = rho * x + (1.0 - rho) * endpoint + jnp.sqrt(safe_dt) * (control @ gamma.T)
        x_next = mean + innovation @ gamma.T
        return x_next, (x_next, local, control, gamma)

    _, (interior_time_major, local_time_major, control_time_major, gamma_time_major) = jax.lax.scan(
        step,
        x0,
        (
            times[:stochastic_steps],
            safe_rho,
            scan_innovations,
        ),
    )
    interior = jnp.swapaxes(interior_time_major, 0, 1)
    states = jnp.concatenate([x0[:, None, :], interior, endpoint[:, None, :]], axis=1)
    local = jnp.swapaxes(local_time_major, 0, 1)
    controls = jnp.swapaxes(control_time_major, 0, 1)
    row_valid = (
        row_input_valid
        & jnp.all(jnp.isfinite(states), axis=(1, 2))
        & jnp.all(jnp.isfinite(innovations), axis=(1, 2))
        & jnp.all(jnp.isfinite(local), axis=(1, 2, 3))
        & jnp.all(jnp.isfinite(controls), axis=(1, 2))
        & jnp.all(jnp.isfinite(gamma_time_major))
    )
    _require_concrete(jnp.all(row_valid), "nonfinite pinned rollout result")
    states = _poison_float(states, row_valid[:, None, None])
    innovations = _poison_float(innovations, row_valid[:, None, None])
    local = _poison_float(local, row_valid[:, None, None, None])
    controls = _poison_float(controls, row_valid[:, None, None])
    endpoint = _poison_float(endpoint, row_valid[:, None])
    gamma_time_major = _poison_float(gamma_time_major, chain_valid & diffusion_valid)
    return AdjointRollout(
        states,
        innovations,
        local,
        gamma_time_major,
        controls,
        times,
        endpoint,
    )


def simulate_pinned_brownian_rollout_matrix_free(
    key: PRNGKey,
    x0: Array,
    endpoint: Array,
    times: Array,
    diffusion: Array | Scalar,
    control_fn: NoiseControlFn | None = None,
    *,
    diffusion_rcond: float = 1e-8,
) -> MatrixFreeAdjointRollout:
    """Sample the pinned chain without forming dense control Jacobians.

    This simulates exactly the same finite Gaussian chain as
    :func:`simulate_pinned_brownian_rollout`.  It records only states,
    innovations, transition noise factors, and controls.  The corresponding
    label assembler applies transition transposes with ``jax.vjp``.
    """
    diffusion_rcond = _validate_positive_finite_real(diffusion_rcond, "diffusion_rcond")
    x0 = jnp.atleast_2d(x0)
    endpoint = jnp.atleast_2d(endpoint)
    if x0.ndim != 2 or not jnp.issubdtype(x0.dtype, jnp.floating):
        raise ValueError("x0 must have shape [batch, state_dim] and a floating dtype")
    if endpoint.shape != x0.shape:
        raise ValueError(f"endpoint shape {endpoint.shape} must match x0 {x0.shape}")
    endpoint = jnp.asarray(endpoint, dtype=x0.dtype)
    times = jnp.asarray(times, dtype=x0.dtype)
    dt, grid_valid = _validate_uniform_times(times)
    safe_dt = jnp.where(grid_valid, dt, jnp.asarray(1.0, dtype=x0.dtype))
    batch_size, dim = x0.shape
    sigma = _constant_diffusion_matrix(diffusion, dim, x0.dtype)
    _validate_full_rank(sigma, diffusion_rcond)
    diffusion_valid = _full_rank_validity(sigma, diffusion_rcond)
    num_steps = times.shape[0] - 1
    if num_steps < 2:
        raise ValueError("pinned bridge requires at least two time steps")
    stochastic_steps = num_steps - 1
    raw_rho = (times[-1] - times[1 : stochastic_steps + 1]) / (times[-1] - times[:stochastic_steps])
    rho_valid = jnp.all(jnp.isfinite(raw_rho) & (raw_rho > 0.0) & (raw_rho < 1.0))
    chain_valid = grid_valid & rho_valid
    safe_rho = jnp.where(chain_valid, raw_rho, jnp.full_like(raw_rho, 0.5))
    row_input_valid = (
        jnp.all(jnp.isfinite(x0), axis=-1)
        & jnp.all(jnp.isfinite(endpoint), axis=-1)
        & chain_valid
        & diffusion_valid
    )
    _require_concrete(jnp.all(row_input_valid), "nonfinite or invalid pinned rollout input")
    innovations = jax.random.normal(
        key,
        (batch_size, stochastic_steps, dim),
        dtype=x0.dtype,
    )
    scan_innovations = jnp.swapaxes(innovations, 0, 1)

    def zero_control(x: Array, t: Scalar, context: Array) -> Array:
        del t, context
        return jnp.zeros_like(x)

    active_control = control_fn or zero_control

    def step(x: Array, inputs: tuple[Scalar, Scalar, Array]):
        time, rho, innovation = inputs
        gamma = jnp.sqrt(safe_dt * rho) * sigma
        control = _batch_control_values(active_control, x, time, endpoint)
        mean = rho * x + (1.0 - rho) * endpoint + jnp.sqrt(safe_dt) * (control @ gamma.T)
        x_next = mean + innovation @ gamma.T
        return x_next, (x_next, control, gamma)

    _, (interior_tm, controls_tm, gamma_tm) = jax.lax.scan(
        step,
        x0,
        (
            times[:stochastic_steps],
            safe_rho,
            scan_innovations,
        ),
    )
    states = jnp.concatenate(
        [x0[:, None, :], jnp.swapaxes(interior_tm, 0, 1), endpoint[:, None, :]],
        axis=1,
    )
    controls = jnp.swapaxes(controls_tm, 0, 1)
    row_valid = (
        row_input_valid
        & jnp.all(jnp.isfinite(states), axis=(1, 2))
        & jnp.all(jnp.isfinite(innovations), axis=(1, 2))
        & jnp.all(jnp.isfinite(controls), axis=(1, 2))
        & jnp.all(jnp.isfinite(gamma_tm))
    )
    _require_concrete(jnp.all(row_valid), "nonfinite pinned rollout result")
    states = _poison_float(states, row_valid[:, None, None])
    innovations = _poison_float(innovations, row_valid[:, None, None])
    controls = _poison_float(controls, row_valid[:, None, None])
    endpoint = _poison_float(endpoint, row_valid[:, None])
    gamma_tm = _poison_float(gamma_tm, chain_valid & diffusion_valid)
    return MatrixFreeAdjointRollout(
        states=states,
        innovations=innovations,
        noise_matrices=gamma_tm,
        controls=controls,
        times=times,
        context=endpoint,
    )


def _pinned_label_one(
    local_jacobians: Array,
    innovations: Array,
    gamma_steps: Array,
    running_values: Array,
    anchor: Array,
    dt: Scalar,
) -> tuple[Array, Array, Array]:
    dim = innovations.shape[-1]
    eye = jnp.eye(dim, dtype=innovations.dtype)
    indices = jnp.arange(innovations.shape[0])

    def step(carry: tuple[Array, Array, Array, Array], inputs: tuple[Array, ...]):
        tangent, integral, running, maximum_condition = carry
        index, local, innovation, gamma = inputs
        active = index >= anchor
        tangent_candidate = local @ tangent
        gamma_transpose_solve = jnp.linalg.solve(gamma.T, innovation)
        integral_candidate = integral + tangent_candidate.T @ gamma_transpose_solve
        count = jnp.maximum(index + 1 - anchor, 1)
        weight = integral_candidate / count
        running_candidate = running + dt * running_values[index + 1] * weight
        condition_candidate = jnp.nan_to_num(
            jnp.linalg.cond(tangent_candidate),
            nan=jnp.inf,
            posinf=jnp.inf,
        )
        tangent = jnp.where(active, tangent_candidate, tangent)
        integral = jnp.where(active, integral_candidate, integral)
        running = jnp.where(active, running_candidate, running)
        maximum_condition = jnp.where(
            active,
            jnp.maximum(maximum_condition, condition_candidate),
            maximum_condition,
        )
        return (tangent, integral, running, maximum_condition), weight

    initial = (
        eye,
        jnp.zeros((dim,), dtype=innovations.dtype),
        jnp.zeros((dim,), dtype=innovations.dtype),
        jnp.asarray(1.0, dtype=innovations.dtype),
    )
    (_, _, running, max_condition), _ = jax.lax.scan(
        step,
        initial,
        (indices, local_jacobians, innovations, gamma_steps),
    )
    finite = jnp.all(jnp.isfinite(running))
    return running, max_condition, finite


def assemble_pinned_brownian_labels(
    rollout: AdjointRollout,
    anchors: Array,
    running_values: Array,
    terminal_values: Array | None = None,
    immediate_gradients: Array | None = None,
) -> BELCostateBatch:
    """Assemble value-only labels for endpoint-pinned conditional paths.

    Arrival ``N`` is deterministic and is not assigned a BEL weight.  A
    terminal value depending only on the fixed endpoint is therefore constant
    with respect to an interior anchor and its costate component is exactly
    zero.  ``terminal_values`` is accepted for API/provenance symmetry and is
    stopped, but cannot affect the returned label.
    """
    batch_size, num_times, dim = rollout.states.shape
    stochastic_steps = rollout.innovations.shape[1]
    if num_times != stochastic_steps + 2:
        raise ValueError("pinned rollout must have one deterministic final transition")
    anchors, safe_anchors, anchor_valid = _normalize_anchors(
        anchors,
        batch_size=batch_size,
        maximum=stochastic_steps - 1,
    )
    if rollout.innovations.shape != (batch_size, stochastic_steps, dim):
        raise ValueError("rollout innovations must have shape [batch, steps, state_dim]")
    if rollout.local_jacobians.shape != (batch_size, stochastic_steps, dim, dim):
        raise ValueError("rollout local_jacobians must have shape [batch, steps, dim, dim]")
    if rollout.noise_matrices.shape != (stochastic_steps, dim, dim):
        raise ValueError("rollout noise_matrices must have shape [steps, dim, dim]")
    if rollout.controls.shape != (batch_size, stochastic_steps, dim):
        raise ValueError("rollout controls must have shape [batch, steps, state_dim]")
    if rollout.context.ndim != 2 or rollout.context.shape[0] != batch_size:
        raise ValueError("rollout context must have shape [batch, context_dim]")
    running_values = jnp.asarray(running_values, dtype=rollout.states.dtype)
    if running_values.shape != (batch_size, num_times):
        raise ValueError(
            f"running_values must have shape {(batch_size, num_times)}, got {running_values.shape}"
        )
    if terminal_values is not None:
        terminal_values = jnp.asarray(terminal_values, dtype=rollout.states.dtype)
        if terminal_values.shape != (batch_size,):
            raise ValueError("terminal_values must have shape [batch]")
        terminal_row_valid = jnp.isfinite(terminal_values)
        terminal_values = jax.lax.stop_gradient(terminal_values)
    else:
        terminal_row_valid = jnp.ones((batch_size,), dtype=bool)
    if immediate_gradients is None:
        immediate_gradients = jnp.zeros((batch_size, dim), dtype=rollout.states.dtype)
    immediate_gradients = jnp.asarray(immediate_gradients, dtype=rollout.states.dtype)
    if immediate_gradients.shape != (batch_size, dim):
        raise ValueError("immediate_gradients must have shape [batch, state_dim]")

    dt, grid_valid = _validate_uniform_times(rollout.times)
    safe_dt = jnp.where(grid_valid, dt, jnp.asarray(1.0, dtype=rollout.states.dtype))
    stopped_running = jax.lax.stop_gradient(running_values)
    running, condition, finite = jax.vmap(
        _pinned_label_one,
        in_axes=(0, 0, None, 0, 0, None),
    )(
        rollout.local_jacobians,
        rollout.innovations,
        rollout.noise_matrices,
        stopped_running,
        safe_anchors,
        safe_dt,
    )
    zeros = jnp.zeros((batch_size, dim), dtype=rollout.states.dtype)
    direct = jax.lax.stop_gradient(immediate_gradients)
    label = running + direct
    row_input_valid = (
        jnp.all(jnp.isfinite(rollout.states), axis=(1, 2))
        & jnp.all(jnp.isfinite(rollout.innovations), axis=(1, 2))
        & jnp.all(jnp.isfinite(rollout.local_jacobians), axis=(1, 2, 3))
        & jnp.all(jnp.isfinite(rollout.controls), axis=(1, 2))
        & jnp.all(jnp.isfinite(rollout.context), axis=-1)
        & jnp.all(jnp.isfinite(running_values), axis=-1)
        & terminal_row_valid
        & jnp.all(jnp.isfinite(direct), axis=-1)
        & anchor_valid
        & grid_valid
        & jnp.all(jnp.isfinite(rollout.noise_matrices))
    )
    finite = finite & row_input_valid
    batch_indices = jnp.arange(batch_size)
    anchor_states = rollout.states[batch_indices, safe_anchors]
    anchor_times = rollout.times[safe_anchors]
    return BELCostateBatch(
        anchors,
        anchor_times,
        anchor_states,
        rollout.context,
        jax.lax.stop_gradient(label),
        zeros,
        jax.lax.stop_gradient(running),
        direct,
        zeros,
        jax.lax.stop_gradient(condition),
        finite,
    )


def assemble_pinned_brownian_labels_matrix_free(
    rollout: MatrixFreeAdjointRollout | AdjointRollout,
    anchors: Array,
    hard_running_values: Array,
    control_fn: NoiseControlFn | None = None,
    *,
    include_control_energy: bool = True,
    center_running: bool = True,
    running_baseline: Array | None = None,
    immediate_gradients: Array | None = None,
    diffusion_rcond: float = 1e-8,
) -> BELCostateBatch:
    """Assemble pinned costates with reverse transition VJPs.

    ``control_fn`` always defines the transition whose VJP is propagated.
    ``include_control_energy`` independently decides whether its known
    quadratic energy belongs to the objective.  Conflating these choices
    would drop the policy Jacobian and bias hard-cost labels whenever a
    state-feedback policy is present.  The hard running potential is handled
    by the complete correlated BEL sum.  When requested, control energy is
    differentiated pathwise, including its direct anchor derivative.  No
    dense state-transition Jacobian is read or materialized.  For one sampled
    anchor per path, auxiliary storage is linear in
    ``batch * time * state_dim``.

    Centering subtracts either ``running_baseline[B]`` or, by default, the
    stopped anchor running value from every future running value.  This is
    mean preserving because it is measurable before the suffix innovations.
    A learned baseline is valid here only when trained on an independent split
    (or otherwise cross-fitted) and stopped by the caller.

    This is a low-level algebraic assembler.  Its sampled-value replay checks
    can reject a callback whose controls or transitions differ on the recorded
    path, but no finite replay can prove equality of the callback's derivative
    law away from those states.  The caller must therefore supply the same
    pure differentiable ``control_fn`` (including the same closed-over
    parameters) that generated ``rollout``.  The production
    :meth:`MalliavinAdjointInnerSolver.make_label_batch` path enforces that
    provenance by fusing simulation and assembly around one stored callback;
    ``finite`` from this standalone function is not a general policy-identity
    certificate.
    """
    diffusion_rcond = _validate_positive_finite_real(diffusion_rcond, "diffusion_rcond")
    batch_size, num_times, dim = rollout.states.shape
    stochastic_steps = rollout.innovations.shape[1]
    if num_times != stochastic_steps + 2:
        raise ValueError("pinned rollout must have one deterministic final transition")
    anchors, safe_anchors, anchor_valid = _normalize_anchors(
        anchors,
        batch_size=batch_size,
        maximum=stochastic_steps - 1,
    )
    if rollout.innovations.shape != (batch_size, stochastic_steps, dim):
        raise ValueError("rollout innovations must have shape [batch, steps, state_dim]")
    if rollout.noise_matrices.shape != (stochastic_steps, dim, dim):
        raise ValueError("rollout noise_matrices must have shape [steps, dim, dim]")
    if rollout.controls.shape != (batch_size, stochastic_steps, dim):
        raise ValueError("rollout controls must have shape [batch, steps, state_dim]")
    if rollout.context.ndim != 2 or rollout.context.shape[0] != batch_size:
        raise ValueError("rollout context must have shape [batch, context_dim]")
    hard_running_values = jnp.asarray(hard_running_values, dtype=rollout.states.dtype)
    if hard_running_values.shape != (batch_size, num_times):
        raise ValueError(
            "hard_running_values must have shape "
            f"{(batch_size, num_times)}, got {hard_running_values.shape}"
        )
    if immediate_gradients is None:
        immediate_gradients = jnp.zeros((batch_size, dim), dtype=rollout.states.dtype)
    immediate_gradients = jnp.asarray(immediate_gradients, dtype=rollout.states.dtype)
    if immediate_gradients.shape != (batch_size, dim):
        raise ValueError("immediate_gradients must have shape [batch, state_dim]")
    stopped_running = jax.lax.stop_gradient(hard_running_values)
    stopped_immediate = jax.lax.stop_gradient(immediate_gradients)
    batch_indices = jnp.arange(batch_size)
    if running_baseline is None:
        running_baseline = stopped_running[batch_indices, safe_anchors]
    running_baseline = jnp.asarray(running_baseline, dtype=rollout.states.dtype)
    if running_baseline.shape != (batch_size,):
        raise ValueError("running_baseline must have shape [batch]")
    stopped_baseline = jax.lax.stop_gradient(running_baseline)
    dt, grid_valid = _validate_uniform_times(rollout.times)
    dt = jnp.asarray(dt, dtype=rollout.states.dtype)
    safe_dt = jnp.where(grid_valid, dt, jnp.asarray(1.0, dtype=rollout.states.dtype))
    terminal_time = rollout.times[-1]
    departures = rollout.times[:stochastic_steps]
    arrivals = rollout.times[1 : stochastic_steps + 1]
    rho_steps = (terminal_time - arrivals) / (terminal_time - departures)
    rho_valid = jnp.all(jnp.isfinite(rho_steps) & (rho_steps > 0.0) & (rho_steps < 1.0))
    chain_valid = grid_valid & rho_valid
    safe_rho_steps = jnp.where(chain_valid, rho_steps, jnp.full_like(rho_steps, 0.5))
    noise_scales = jnp.sqrt(safe_dt * safe_rho_steps)
    base_sigma = rollout.noise_matrices[0] / noise_scales[0]
    expected_noise_matrices = noise_scales[:, None, None] * base_sigma[None, :, :]
    scale = jnp.maximum(
        jnp.maximum(
            jnp.max(jnp.abs(expected_noise_matrices)),
            jnp.max(jnp.abs(rollout.noise_matrices)),
        ),
        jnp.asarray(jnp.finfo(rollout.states.dtype).tiny, dtype=rollout.states.dtype),
    )
    chain_tolerance = 64.0 * jnp.finfo(rollout.states.dtype).eps * scale
    noise_chain_valid = jnp.all(
        jnp.abs(rollout.noise_matrices - expected_noise_matrices) <= chain_tolerance
    )
    base_sigma_values = jnp.linalg.svd(base_sigma, compute_uv=False)
    base_sigma_valid = (
        jnp.all(jnp.isfinite(base_sigma))
        & jnp.all(jnp.isfinite(base_sigma_values))
        & (jnp.min(base_sigma_values) > diffusion_rcond * jnp.max(base_sigma_values))
    )
    flattened_noise = rollout.innovations.reshape((-1, dim)).T
    base_scores = jnp.linalg.solve(base_sigma.T, flattened_noise).T.reshape(
        (batch_size, stochastic_steps, dim)
    )
    score_innovations = base_scores / noise_scales[None, :, None]

    def zero_control(x: Array, t: Scalar, context: Array) -> Array:
        del t, context
        return jnp.zeros_like(x)

    active_control = control_fn or zero_control

    departure_states_tm = jnp.swapaxes(rollout.states[:, :stochastic_steps], 0, 1)
    recomputed_controls_tm = jax.vmap(
        lambda state_batch, time: _batch_control_values(
            active_control,
            state_batch,
            time,
            rollout.context,
        )
    )(departure_states_tm, rollout.times[:stochastic_steps])
    recomputed_controls = jnp.swapaxes(recomputed_controls_tm, 0, 1)
    control_scale = jnp.maximum(
        jnp.maximum(
            jnp.max(jnp.abs(recomputed_controls), axis=(1, 2)),
            jnp.max(jnp.abs(rollout.controls), axis=(1, 2)),
        ),
        jnp.asarray(jnp.finfo(rollout.states.dtype).tiny, dtype=rollout.states.dtype),
    )
    control_tolerance = 64.0 * jnp.finfo(rollout.states.dtype).eps * control_scale
    control_consistent = jnp.all(
        jnp.abs(recomputed_controls - rollout.controls) <= control_tolerance[:, None, None],
        axis=(1, 2),
    )

    expected_arrivals = (
        safe_rho_steps[None, :, None] * rollout.states[:, :stochastic_steps]
        + (1.0 - safe_rho_steps)[None, :, None] * rollout.context[:, None, :]
        + jnp.sqrt(safe_dt) * jnp.einsum("tij,btj->bti", rollout.noise_matrices, rollout.controls)
        + jnp.einsum("tij,btj->bti", rollout.noise_matrices, rollout.innovations)
    )
    observed_arrivals = rollout.states[:, 1 : stochastic_steps + 1]
    state_scale = jnp.maximum(
        jnp.maximum(
            jnp.max(jnp.abs(expected_arrivals), axis=(1, 2)),
            jnp.max(jnp.abs(observed_arrivals), axis=(1, 2)),
        ),
        jnp.asarray(jnp.finfo(rollout.states.dtype).tiny, dtype=rollout.states.dtype),
    )
    state_tolerance = 128.0 * jnp.finfo(rollout.states.dtype).eps * state_scale
    state_consistent = jnp.all(
        jnp.abs(expected_arrivals - observed_arrivals) <= state_tolerance[:, None, None],
        axis=(1, 2),
    )
    endpoint_scale = jnp.maximum(
        jnp.maximum(
            jnp.max(jnp.abs(rollout.states[:, -1]), axis=-1),
            jnp.max(jnp.abs(rollout.context), axis=-1),
        ),
        jnp.asarray(jnp.finfo(rollout.states.dtype).tiny, dtype=rollout.states.dtype),
    )
    endpoint_consistent = jnp.all(
        jnp.abs(rollout.states[:, -1] - rollout.context)
        <= (64.0 * jnp.finfo(rollout.states.dtype).eps * endpoint_scale)[:, None],
        axis=-1,
    )

    def one_path(
        states: Array,
        innovations: Array,
        innovation_scores: Array,
        controls: Array,
        endpoint: Array,
        running_values: Array,
        baseline_value: Array,
        anchor: Array,
        extra_direct: Array,
    ) -> tuple[Array, Array, Array]:
        indices = jnp.arange(stochastic_steps)
        arrival_indices = jnp.arange(num_times)
        baseline = jnp.where(center_running, baseline_value, 0.0)
        denominator = jnp.maximum(arrival_indices - anchor, 1)
        eligible = (arrival_indices > anchor) & (arrival_indices <= stochastic_steps)
        weighted_cost = jnp.where(
            eligible,
            (running_values - baseline) / denominator,
            0.0,
        )
        suffix = jnp.cumsum(weighted_cost[::-1])[::-1]
        hard_coefficients = safe_dt * suffix[indices + 1]
        hard_sources = innovation_scores * hard_coefficients[:, None]

        def energy_gradient(state: Array, time: Scalar) -> Array:
            def energy(value: Array) -> Array:
                control = _single_control_value(active_control, value, time, endpoint)
                return 0.5 * jnp.sum(control**2)

            return jnp.asarray(jax.grad(energy)(state))

        departure_energy_gradients = jax.lax.cond(
            include_control_energy,
            lambda _: jax.vmap(energy_gradient)(
                states[:stochastic_steps],
                rollout.times[:stochastic_steps],
            ),
            lambda _: jnp.zeros((stochastic_steps, dim), dtype=states.dtype),
            operand=None,
        )
        future_energy_sources = jnp.concatenate(
            [
                safe_dt * departure_energy_gradients[1:],
                jnp.zeros((1, dim), dtype=states.dtype),
            ],
            axis=0,
        )
        sources = hard_sources + future_energy_sources

        def reverse_step(carry: Array, inputs: tuple[Array, ...]):
            state, time, rho, gamma, source = inputs

            @jax.checkpoint
            def transition_mean(value: Array) -> Array:
                control = _single_control_value(active_control, value, time, endpoint)
                return rho * value + (1.0 - rho) * endpoint + jnp.sqrt(safe_dt) * (gamma @ control)

            _, pullback = jax.vjp(transition_mean, state)
            value = pullback(source + carry)[0]
            return value, value

        _, reverse_values = jax.lax.scan(
            reverse_step,
            jnp.zeros((dim,), dtype=states.dtype),
            (
                states[:stochastic_steps],
                rollout.times[:stochastic_steps],
                safe_rho_steps,
                rollout.noise_matrices,
                sources,
            ),
            reverse=True,
        )
        direct = safe_dt * departure_energy_gradients[anchor] + extra_direct
        label = reverse_values[anchor] + direct
        finite = (
            jnp.all(jnp.isfinite(label))
            & jnp.all(jnp.isfinite(states))
            & jnp.all(jnp.isfinite(innovations))
            & jnp.all(jnp.isfinite(controls))
            & jnp.all(jnp.isfinite(running_values))
            & jnp.isfinite(baseline_value)
            & jnp.all(jnp.isfinite(extra_direct))
        )
        return label, direct, finite

    labels, direct, finite = jax.vmap(one_path)(
        rollout.states,
        rollout.innovations,
        score_innovations,
        rollout.controls,
        rollout.context,
        stopped_running,
        stopped_baseline,
        safe_anchors,
        stopped_immediate,
    )
    row_input_valid = (
        jnp.all(jnp.isfinite(rollout.states), axis=(1, 2))
        & jnp.all(jnp.isfinite(rollout.innovations), axis=(1, 2))
        & jnp.all(jnp.isfinite(rollout.controls), axis=(1, 2))
        & jnp.all(jnp.isfinite(rollout.context), axis=-1)
        & jnp.all(jnp.isfinite(hard_running_values), axis=-1)
        & jnp.isfinite(running_baseline)
        & jnp.all(jnp.isfinite(immediate_gradients), axis=-1)
        & anchor_valid
        & chain_valid
        & jnp.all(jnp.isfinite(rollout.noise_matrices))
        & noise_chain_valid
        & base_sigma_valid
        & control_consistent
        & state_consistent
        & endpoint_consistent
    )
    finite = finite & row_input_valid
    anchor_states = rollout.states[batch_indices, safe_anchors]
    anchor_times = rollout.times[safe_anchors]
    zeros = jnp.zeros((batch_size, dim), dtype=rollout.states.dtype)
    # Dense tangent condition numbers are intentionally unavailable on the
    # matrix-free path.  The negative sentinel is explicit and finite.
    condition_unavailable = -jnp.ones((batch_size,), dtype=rollout.states.dtype)
    return BELCostateBatch(
        anchor_index=anchors,
        anchor_time=anchor_times,
        anchor_state=anchor_states,
        context=rollout.context,
        label=jax.lax.stop_gradient(labels),
        terminal_component=zeros,
        running_component=jax.lax.stop_gradient(labels - direct),
        direct_component=jax.lax.stop_gradient(direct),
        terminal_weight=zeros,
        tangent_condition_number=condition_unavailable,
        finite=finite,
    )


def assemble_pinned_actor_targets(
    key: PRNGKey,
    state: Array,
    endpoint: Array,
    time: Array | Scalar,
    next_time: Array | Scalar,
    terminal_time: Array | Scalar,
    diffusion: Array | Scalar,
    current_control: Array,
    running_cost: RunningCostFn,
    next_costate: Callable[[Array, Array, Array], Array] | None = None,
    *,
    num_antithetic: int = 1,
    diffusion_rcond: float = 1e-8,
) -> ActionTargetBatch:
    """Construct the arrival-aware current-policy actor target.

    For the declared pinned transition, this estimates

    ``-sqrt(rho) Sigma.T E[p_next] - sqrt(dt) E[(ell_next-c) xi]``.

    Antithetic differences remove the scalar baseline from the arrival term.
    ``next_costate=None`` represents a zero continuation, which is required at
    the final stochastic arrival.  Cost and costate values are stopped before
    the target is returned.
    """
    if not _is_strict_integer(num_antithetic) or num_antithetic < 1:
        raise ValueError("num_antithetic must be a positive integer (boolean is invalid)")
    diffusion_rcond = _validate_positive_finite_real(diffusion_rcond, "diffusion_rcond")
    state = jnp.atleast_2d(state)
    if state.ndim != 2 or not jnp.issubdtype(state.dtype, jnp.floating):
        raise ValueError("state must have shape [batch, state_dim] and a floating dtype")
    endpoint = _broadcast_context(endpoint, state.shape[0])
    if endpoint.shape != state.shape:
        raise ValueError("endpoint must match state shape [batch, state_dim]")
    endpoint = jnp.asarray(endpoint, dtype=state.dtype)
    batch_size, dim = state.shape
    time_array = _as_batch_vector(time, batch_size, "time", state.dtype)
    next_time_array = _as_batch_vector(next_time, batch_size, "next_time", state.dtype)
    terminal = jnp.asarray(terminal_time, dtype=state.dtype)
    if terminal.ndim != 0:
        raise ValueError("terminal_time must be scalar")
    dt = next_time_array - time_array
    remaining = terminal - time_array
    rho = (terminal - next_time_array) / remaining
    _require_concrete(
        jnp.all(jnp.isfinite(time_array))
        & jnp.all(jnp.isfinite(next_time_array))
        & jnp.isfinite(terminal),
        "time values must be finite",
    )
    _require_concrete(jnp.all(dt > 0.0), "next_time must be greater than time")
    _require_concrete(
        jnp.all((rho > 0.0) & (rho < 1.0)),
        "pinned actor target requires 0 < rho < 1",
    )
    time_valid = (
        jnp.isfinite(time_array)
        & jnp.isfinite(next_time_array)
        & jnp.isfinite(terminal)
        & jnp.isfinite(dt)
        & jnp.isfinite(remaining)
        & jnp.isfinite(rho)
        & (dt > 0.0)
        & (remaining > 0.0)
        & (next_time_array < terminal)
        & (rho > 0.0)
        & (rho < 1.0)
    )
    safe_dt = jnp.where(time_valid, dt, jnp.ones_like(dt))
    safe_rho = jnp.where(time_valid, rho, jnp.full_like(rho, 0.5))
    sigma = _constant_diffusion_matrix(diffusion, dim, state.dtype)
    _validate_full_rank(sigma, diffusion_rcond)
    diffusion_valid = _full_rank_validity(sigma, diffusion_rcond)
    control = jnp.asarray(current_control, dtype=state.dtype)
    if control.ndim == 1:
        control = control[None, :]
    try:
        control = jnp.broadcast_to(control, state.shape)
    except ValueError as error:
        raise ValueError("current_control must broadcast to state shape") from error
    input_valid = (
        time_valid
        & jnp.all(jnp.isfinite(state), axis=-1)
        & jnp.all(jnp.isfinite(endpoint), axis=-1)
        & jnp.all(jnp.isfinite(control), axis=-1)
        & diffusion_valid
    )
    _require_concrete(jnp.all(input_valid), "nonfinite or invalid pinned actor input")
    gamma = jnp.sqrt(safe_dt * safe_rho)[:, None, None] * sigma[None, :, :]
    mean_state = (
        safe_rho[:, None] * state
        + (1.0 - safe_rho)[:, None] * endpoint
        + jnp.sqrt(safe_dt)[:, None] * jnp.einsum("bij,bj->bi", gamma, control)
    )
    innovations = jax.random.normal(
        key,
        (batch_size, num_antithetic, dim),
        dtype=state.dtype,
    )
    perturbation = jnp.einsum("bij,bmj->bmi", gamma, innovations)
    plus = mean_state[:, None, :] + perturbation
    minus = mean_state[:, None, :] - perturbation
    flat_plus = plus.reshape((-1, dim))
    flat_minus = minus.reshape((-1, dim))
    repeated_time = jnp.repeat(next_time_array, num_antithetic)
    repeated_context = jnp.repeat(endpoint, num_antithetic, axis=0)
    plus_cost = jnp.asarray(
        running_cost(flat_plus, repeated_time, repeated_context),
        dtype=state.dtype,
    ).reshape((batch_size, num_antithetic))
    minus_cost = jnp.asarray(
        running_cost(flat_minus, repeated_time, repeated_context),
        dtype=state.dtype,
    ).reshape((batch_size, num_antithetic))
    plus_cost = jax.lax.stop_gradient(plus_cost)
    minus_cost = jax.lax.stop_gradient(minus_cost)
    arrival = -jnp.sqrt(safe_dt)[:, None] * jnp.mean(
        0.5 * (plus_cost - minus_cost)[:, :, None] * innovations,
        axis=1,
    )

    if next_costate is None:
        continuation_mean = jnp.zeros_like(state)
        costate_valid = jnp.ones((batch_size,), dtype=bool)
    else:
        plus_costate = jnp.asarray(
            next_costate(flat_plus, repeated_time, repeated_context),
            dtype=state.dtype,
        ).reshape((batch_size, num_antithetic, dim))
        minus_costate = jnp.asarray(
            next_costate(flat_minus, repeated_time, repeated_context),
            dtype=state.dtype,
        ).reshape((batch_size, num_antithetic, dim))
        costate_valid = jnp.all(
            jnp.isfinite(plus_costate) & jnp.isfinite(minus_costate),
            axis=(1, 2),
        )
        continuation_mean = jnp.mean(0.5 * (plus_costate + minus_costate), axis=1)
        continuation_mean = jax.lax.stop_gradient(continuation_mean)
    continuation = -jnp.sqrt(safe_rho)[:, None] * (continuation_mean @ sigma)
    target = jax.lax.stop_gradient(continuation + arrival)
    finite = (
        input_valid
        & jnp.all(jnp.isfinite(target), axis=-1)
        & jnp.all(jnp.isfinite(state), axis=-1)
        & jnp.all(jnp.isfinite(endpoint), axis=-1)
        & jnp.all(jnp.isfinite(control), axis=-1)
        & jnp.all(jnp.isfinite(mean_state), axis=-1)
        & jnp.all(jnp.isfinite(innovations), axis=(1, 2))
        & jnp.all(jnp.isfinite(plus_cost) & jnp.isfinite(minus_cost), axis=-1)
        & costate_valid
        & jnp.all(jnp.isfinite(gamma), axis=(1, 2))
        & diffusion_valid
    )
    _require_concrete(jnp.all(finite), "nonfinite pinned actor target")
    return ActionTargetBatch(
        target=target,
        continuation_component=jax.lax.stop_gradient(continuation),
        arrival_component=jax.lax.stop_gradient(arrival),
        mean_state=jax.lax.stop_gradient(mean_state),
        innovation=jax.lax.stop_gradient(innovations),
        finite=finite,
        physical_oracle_queries=jnp.asarray(
            2 * batch_size * num_antithetic,
            dtype=jnp.int32,
        ),
        estimator=jnp.asarray(ANTITHETIC_PINNED_ARRIVAL_ESTIMATOR, dtype=jnp.int32),
    )


def assemble_antithetic_direct_action_score(
    positive_return: Array,
    negative_return: Array,
    innovations: Array,
    dt: Array | Scalar,
) -> DirectActionScoreBatch:
    """Assemble the mandatory tangent-free full-suffix score baseline.

    For an antithetic first-step pair, the current-policy action target is

    ``-E[(G_plus-G_minus) xi / (2 sqrt(dt)) | state, context]``.

    This function only assembles already evaluated returns; it does not hide
    suffix simulation or oracle calls.  Returns have shape ``[B,M]``, base
    innovations ``[B,M,d]``, and ``dt`` is scalar or ``[B]``.
    """
    positive = jnp.asarray(positive_return)
    negative = jnp.asarray(negative_return, dtype=positive.dtype)
    noise = jnp.asarray(innovations, dtype=positive.dtype)
    if positive.ndim != 2 or negative.shape != positive.shape:
        raise ValueError("positive_return and negative_return must share shape [B,M]")
    if noise.ndim != 3 or noise.shape[:2] != positive.shape:
        raise ValueError("innovations must have shape [B,M,d]")
    batch_size, num_pairs = positive.shape
    step = _as_batch_vector(dt, batch_size, "dt", positive.dtype)
    valid_step = jnp.isfinite(step) & (step > 0.0)
    stopped_difference = jax.lax.stop_gradient(positive - negative)
    target = (
        -jnp.mean(
            0.5 * stopped_difference[:, :, None] * jax.lax.stop_gradient(noise),
            axis=1,
        )
        / jnp.sqrt(step)[:, None]
    )
    finite = (
        valid_step
        & jnp.all(jnp.isfinite(positive) & jnp.isfinite(negative), axis=-1)
        & jnp.all(jnp.isfinite(noise), axis=(1, 2))
        & jnp.all(jnp.isfinite(target), axis=-1)
    )
    _require_concrete(jnp.all(finite), "nonfinite direct action-score target")
    return DirectActionScoreBatch(
        target=jax.lax.stop_gradient(target),
        finite=finite,
        physical_return_queries=jnp.asarray(2 * batch_size * num_pairs, dtype=jnp.int32),
        estimator=jnp.asarray(ANTITHETIC_DIRECT_RETURN_SCORE_ESTIMATOR, dtype=jnp.int32),
    )


def summarize_costate_labels(batch: BELCostateBatch) -> dict[str, Any]:
    """Return host-side finite/tail diagnostics without hiding nonfinite rows."""
    labels = np.asarray(jax.device_get(batch.label), dtype=float)
    declared_finite = np.asarray(jax.device_get(batch.finite), dtype=bool)
    finite_rows = np.all(np.isfinite(labels), axis=-1) & declared_finite
    norms = np.linalg.norm(labels[finite_rows], axis=-1)
    result: dict[str, Any] = {
        "count": int(labels.shape[0]),
        "finite_count": int(finite_rows.sum()),
        "finite_fraction": float(finite_rows.mean()) if labels.shape[0] else 0.0,
        "independent_unit": "trajectory",
        "theorem_facing_clipping": False,
    }
    if norms.size == 0:
        result.update(
            {
                "mean": None,
                "standard_error": None,
                "norm_p50": None,
                "norm_p95": None,
                "norm_p99": None,
                "norm_p999": None,
                "norm_max": None,
                "top_1_percent_centered_energy_share": None,
                "top_0_1_percent_centered_energy_share": None,
            }
        )
        return result

    finite_labels = labels[finite_rows]
    mean = finite_labels.mean(axis=0)
    if finite_labels.shape[0] > 1:
        standard_error = finite_labels.std(axis=0, ddof=1) / np.sqrt(finite_labels.shape[0])
    else:
        standard_error = np.full_like(mean, np.nan)
    centered_energy = np.sum((finite_labels - mean) ** 2, axis=-1)
    total_energy = centered_energy.sum()

    def top_share(fraction: float) -> float:
        count = max(1, int(np.ceil(fraction * centered_energy.size)))
        if total_energy <= 0:
            return 0.0
        return float(np.sort(centered_energy)[-count:].sum() / total_energy)

    result.update(
        {
            "mean": mean.tolist(),
            "standard_error": standard_error.tolist(),
            "norm_p50": float(np.quantile(norms, 0.50)),
            "norm_p95": float(np.quantile(norms, 0.95)),
            "norm_p99": float(np.quantile(norms, 0.99)),
            "norm_p999": float(np.quantile(norms, 0.999)),
            "norm_max": float(norms.max()),
            "top_1_percent_centered_energy_share": top_share(0.01),
            "top_0_1_percent_centered_energy_share": top_share(0.001),
        }
    )
    return result


class MalliavinAdjointInnerSolver:
    """Learn conditional value costates on endpoint-pinned Brownian paths.

    This class intentionally does not inherit :class:`SBSolver`.  It supplies
    the conditional-SOC inner component needed by a future generalized bridge
    algorithm, but it does not perform reciprocal/Markov projection and must
    not be represented as a solved global bridge.
    """

    status = "CONDITIONAL_MAM_FOUNDATION"
    endpoint_constrained_global_bridge = False

    def __init__(
        self,
        problem: SBProblem,
        value_cost: ValueOnlyCost,
        mam_config: MalliavinAdjointConfig | None = None,
        control_fn: NoiseControlFn | None = None,
    ):
        self.problem = problem
        self.value_cost = value_cost
        self.mam_config = mam_config or MalliavinAdjointConfig()
        self.control_fn = control_fn
        self._factory = self.mam_config.network_factory or MLPFactory(
            hidden_dims=self.mam_config.hidden_dims,
            time_embed_dim=self.mam_config.time_embed_dim,
        )
        self._ema_params: Params | None = None
        self._params: Params | None = None
        self._validate_problem()

    def _validate_problem(self) -> None:
        # The V1 contract freezes an actually constant Brownian reference.
        # Accepting subclasses after a few numerical probes is fail-open: a
        # subclass can agree at the probes while remaining state/time dependent
        # elsewhere, after which ``self._diffusion`` would silently freeze the
        # wrong matrix.  A future declared constant-diffusion interface can
        # broaden this boundary without relying on probing.
        if type(self.problem.reference) is not BrownianMotion:
            raise ValueError(
                "pinned MAM v1 requires an explicit BrownianMotion reference; "
                "generic/state-dependent reference dynamics are not yet supported"
            )
        if self.value_cost.terminal_cost is not None:
            raise ValueError(
                "A terminal cost is constant on an endpoint-pinned conditional path. "
                "Handle it in the future endpoint-coupling outer loop; the v1 inner "
                "solver accepts additive running costs only."
            )
        dim = self.problem.dim
        probe0 = jnp.zeros((2, dim), dtype=jnp.float32)
        probe1 = jnp.ones((2, dim), dtype=jnp.float32)
        t0 = self.problem.time_grid.t0
        t1 = self.problem.time_grid.t1
        sigma0 = _constant_diffusion_matrix(
            self.problem.reference.diffusion(probe0, t0), dim, probe0.dtype
        )
        sigma1 = _constant_diffusion_matrix(
            self.problem.reference.diffusion(probe1, t1), dim, probe0.dtype
        )
        _validate_full_rank(sigma0, self.mam_config.diffusion_rcond)
        if not np.allclose(np.asarray(sigma0), np.asarray(sigma1), rtol=1e-7, atol=1e-8):
            raise ValueError("MAM v1 requires constant diffusion")
        drift0 = np.asarray(self.problem.reference.drift(probe0, t0))
        drift1 = np.asarray(self.problem.reference.drift(probe1, t1))
        if not (np.allclose(drift0, 0.0) and np.allclose(drift1, 0.0)):
            raise ValueError("pinned MAM v1 requires a Brownian reference drift")
        if self.mam_config.minimum_remaining_steps > self.problem.time_grid.num_steps - 1:
            raise ValueError("minimum_remaining_steps leaves no stochastic pinned transition")
        self._diffusion = sigma0

    def init_params(self, key: PRNGKey) -> Params:
        input_dim = 2 * self.problem.dim
        params = cast(Params, self._factory.init(key, input_dim, self.problem.dim))
        sanity_check(self._factory, key, input_dim, self.problem.dim)
        return params

    def _network_input(self, state: Array, endpoint: Array) -> Array:
        state = jnp.atleast_2d(state)
        endpoint = _broadcast_context(endpoint, state.shape[0])
        if endpoint.shape[-1] != self.problem.dim:
            raise ValueError("endpoint context must have state dimension")
        return jnp.concatenate([state, endpoint], axis=-1)

    def _evaluate_costs(self, rollout: AdjointRollout) -> tuple[Array, Array]:
        running = self.value_cost.running_values(
            rollout.states,
            rollout.times,
            rollout.context,
        )
        if self.mam_config.include_control_energy:
            energy = 0.5 * jnp.sum(rollout.controls**2, axis=-1)
            # Control ``n`` is evaluated at departure state X_n.  Future
            # departure costs (n > anchor) can use H[k,n], while the current
            # n == anchor term is differentiated explicitly below.  Shifting
            # these values to n+1 would silently attach the wrong BEL weight.
            running = running.at[:, : energy.shape[1]].add(energy)
        terminal = self.value_cost.terminal_values(
            rollout.states[:, -1, :],
            rollout.context,
        )
        return jax.lax.stop_gradient(running), jax.lax.stop_gradient(terminal)

    def _immediate_control_energy_gradient(
        self,
        rollout: AdjointRollout,
        anchors: Array,
    ) -> Array:
        """Differentiate only the known current quadratic control cost.

        Value-only potential costs remain stopped.  The immediate control
        energy is different: at the anchor it is measurable before the next
        innovation, so a future-noise BEL weight cannot recover its state
        derivative.  The returned quantity already includes ``dt``.
        """
        batch_size = rollout.states.shape[0]
        control_fn = self.control_fn
        if not self.mam_config.include_control_energy or control_fn is None:
            return jnp.zeros((batch_size, self.problem.dim), dtype=rollout.states.dtype)
        batch_indices = jnp.arange(batch_size)
        states = rollout.states[batch_indices, anchors]
        times = rollout.times[anchors]
        endpoints = rollout.context

        def one(state: Array, time: Scalar, endpoint: Array) -> Array:
            def energy(value: Array) -> Array:
                control = jnp.asarray(
                    control_fn(value[None, :], time, endpoint[None, :]),
                    dtype=value.dtype,
                )[0]
                return 0.5 * jnp.sum(control**2)

            return jnp.asarray(jax.grad(energy)(state))

        dt = rollout.times[1] - rollout.times[0]
        return jax.lax.stop_gradient(dt * jax.vmap(one)(states, times, endpoints))

    def make_label_batch(
        self,
        key: PRNGKey,
        x0: Array,
        endpoint: Array,
        *,
        running_baseline_fn: Callable[[Array, Array, Array], Array] | None = None,
    ) -> BELCostateBatch:
        """Generate labels for caller-supplied endpoint pairs.

        This seam lets a future reciprocal/GSBM outer loop supply its current
        coupling instead of silently falling back to the independent product
        coupling used by :meth:`sample_label_batch`.
        """
        x0 = jnp.atleast_2d(x0)
        endpoint = jnp.atleast_2d(endpoint)
        if x0.shape != endpoint.shape or x0.shape[-1] != self.problem.dim:
            raise ValueError("x0 and endpoint must have matching shape [batch, state_dim]")
        rollout_key, anchor_key = jax.random.split(key)
        rollout: MatrixFreeAdjointRollout | AdjointRollout
        if self.mam_config.matrix_free_labels:
            rollout = simulate_pinned_brownian_rollout_matrix_free(
                rollout_key,
                x0,
                endpoint,
                self.problem.time_grid.times,
                self._diffusion,
                self.control_fn,
                diffusion_rcond=self.mam_config.diffusion_rcond,
            )
        else:
            rollout = simulate_pinned_brownian_rollout(
                rollout_key,
                x0,
                endpoint,
                self.problem.time_grid.times,
                self._diffusion,
                self.control_fn,
                diffusion_rcond=self.mam_config.diffusion_rcond,
            )
        stochastic_steps = rollout.innovations.shape[1]
        maximum_anchor = stochastic_steps - self.mam_config.minimum_remaining_steps
        if self.mam_config.anchor_sampling == "stratified":
            offset_key, permutation_key = jax.random.split(anchor_key)
            anchor_count = maximum_anchor + 1
            offset = jax.random.randint(offset_key, (), 0, anchor_count)
            anchors = (jnp.arange(x0.shape[0], dtype=jnp.int32) + offset) % anchor_count
            anchors = jax.random.permutation(permutation_key, anchors)
        else:
            anchors = jax.random.randint(
                anchor_key,
                (x0.shape[0],),
                minval=0,
                maxval=maximum_anchor + 1,
            )
        if self.mam_config.matrix_free_labels:
            matrix_free_rollout = cast(MatrixFreeAdjointRollout, rollout)
            hard_running = self.value_cost.running_values(
                matrix_free_rollout.states,
                matrix_free_rollout.times,
                matrix_free_rollout.context,
            )
            running_baseline = None
            if running_baseline_fn is not None:
                rows = jnp.arange(x0.shape[0])
                anchor_states = matrix_free_rollout.states[rows, anchors]
                anchor_times = matrix_free_rollout.times[anchors]
                running_baseline = jax.lax.stop_gradient(
                    jnp.asarray(
                        running_baseline_fn(
                            anchor_states,
                            anchor_times,
                            matrix_free_rollout.context,
                        ),
                        dtype=matrix_free_rollout.states.dtype,
                    ).reshape((x0.shape[0],))
                )
            return assemble_pinned_brownian_labels_matrix_free(
                matrix_free_rollout,
                anchors,
                hard_running,
                self.control_fn,
                include_control_energy=self.mam_config.include_control_energy,
                center_running=self.mam_config.center_running_values,
                running_baseline=running_baseline,
                diffusion_rcond=self.mam_config.diffusion_rcond,
            )
        dense_rollout = cast(AdjointRollout, rollout)
        running, terminal = self._evaluate_costs(dense_rollout)
        if self.mam_config.center_running_values:
            hard_running = self.value_cost.running_values(
                dense_rollout.states,
                dense_rollout.times,
                dense_rollout.context,
            )
            baseline = hard_running[jnp.arange(x0.shape[0]), anchors]
            running = running - baseline[:, None]
        immediate = self._immediate_control_energy_gradient(dense_rollout, anchors)
        return assemble_pinned_brownian_labels(
            dense_rollout,
            anchors,
            running,
            terminal,
            immediate,
        )

    def sample_label_batch(self, key: PRNGKey, batch_size: int) -> BELCostateBatch:
        """Generate labels under an explicitly independent endpoint coupling."""
        if not _is_strict_integer(batch_size) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        pair_key, batch_key = jax.random.split(key)
        x0, endpoint = self.problem.sample_pair(pair_key, batch_size)
        return self.make_label_batch(batch_key, x0, endpoint)

    def loss(self, params: Params, batch: BELCostateBatch) -> tuple[Array, dict[str, Array]]:
        prediction = jnp.asarray(
            self._factory.forward(
                params,
                self._network_input(batch.anchor_state, batch.context),
                batch.anchor_time,
            ),
            dtype=batch.anchor_state.dtype,
        )
        if prediction.shape != batch.label.shape:
            raise ValueError(
                f"costate factory output must have shape {batch.label.shape}, "
                f"got {prediction.shape}"
            )
        # Invalid *labels* may be masked for diagnostic loss evaluation, but a
        # nonfinite model prediction must never be reclassified as an excluded
        # row.  On a valid training batch it therefore propagates into the loss
        # and the explicit prediction-finiteness diagnostic below.
        row_finite = batch.finite
        safe_label = jnp.where(row_finite[:, None], batch.label, 0.0)
        difference = jnp.where(row_finite[:, None], prediction - safe_label, 0.0)
        squared_error = jnp.sum(difference**2, axis=-1)
        finite_weight = row_finite.astype(prediction.dtype)
        denominator = jnp.maximum(jnp.sum(finite_weight), 1.0)
        loss = jnp.sum(finite_weight * squared_error) / denominator
        metrics = {
            "loss": loss,
            "loss_finite": jnp.isfinite(loss),
            "finite_fraction": jnp.mean(finite_weight),
            "label_norm": jnp.sum(finite_weight * jnp.linalg.norm(safe_label, axis=-1))
            / denominator,
            "prediction_norm": jnp.mean(jnp.linalg.norm(prediction, axis=-1)),
            "prediction_finite": jnp.all(jnp.isfinite(prediction)),
            "tangent_condition_max": jnp.max(batch.tangent_condition_number),
        }
        return loss, metrics

    def update_from_batch(
        self,
        params: Params,
        opt_state: AdamState,
        batch: BELCostateBatch,
    ) -> tuple[Params, AdamState, dict[str, Array]]:
        """Update from an external batch, failing closed on numerical errors."""
        if not bool(jax.device_get(_tree_all_finite(params))):
            raise FloatingPointError("nonfinite costate parameters; refusing training update")
        if not bool(jax.device_get(_tree_all_finite(opt_state))):
            raise FloatingPointError("nonfinite optimizer state; refusing training update")
        labels_finite = jnp.all(batch.finite) & jnp.all(jnp.isfinite(batch.label))
        if not bool(jax.device_get(labels_finite)):
            raise FloatingPointError("nonfinite Malliavin adjoint label; refusing training update")
        new_params, new_opt_state, metrics = self._update_jit(params, opt_state, batch)
        checks = (
            ("prediction_finite", "model prediction"),
            ("loss_finite", "training loss"),
            ("gradient_finite", "gradient"),
            ("updated_parameters_finite", "updated costate parameters"),
            ("updated_optimizer_state_finite", "updated optimizer state"),
        )
        for key, description in checks:
            if not bool(jax.device_get(metrics[key])):
                raise FloatingPointError(f"nonfinite {description}; refusing training update")
        return new_params, new_opt_state, metrics

    @partial(jax.jit, static_argnums=0)
    def _update_jit(
        self,
        params: Params,
        opt_state: AdamState,
        batch: BELCostateBatch,
    ) -> tuple[Params, AdamState, dict[str, Array]]:
        (_, metrics), grads = jax.value_and_grad(self.loss, has_aux=True)(params, batch)
        new_params, new_opt_state = adam_update(
            opt_state,
            grads,
            params,
            lr=self.mam_config.learning_rate,
        )
        metrics = {
            **metrics,
            "gradient_finite": _tree_all_finite(grads),
            "updated_parameters_finite": _tree_all_finite(new_params),
            "updated_optimizer_state_finite": _tree_all_finite(new_opt_state),
        }
        return new_params, new_opt_state, metrics

    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: AdamState,
        batch_size: int | None = None,
    ) -> tuple[Params, AdamState, dict[str, Array]]:
        selected_batch_size = self.mam_config.batch_size if batch_size is None else batch_size
        if not _is_strict_integer(selected_batch_size) or selected_batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        size = int(selected_batch_size)
        batch = self.sample_label_batch(key, size)
        new_params, new_opt_state, metrics = self.update_from_batch(params, opt_state, batch)
        if self._ema_params is None:
            self._ema_params = new_params
        else:
            decay = self.mam_config.ema_decay
            self._ema_params = jax.tree_util.tree_map(
                lambda old, new: decay * old + (1.0 - decay) * new,
                self._ema_params,
                new_params,
            )
        return new_params, new_opt_state, metrics

    def train(
        self,
        key: PRNGKey,
        training_steps: int | None = None,
        batch_size: int | None = None,
    ) -> MalliavinAdjointResult:
        selected_steps = (
            self.mam_config.training_steps if training_steps is None else training_steps
        )
        selected_batch_size = self.mam_config.batch_size if batch_size is None else batch_size
        if not _is_strict_integer(selected_steps) or selected_steps < 1:
            raise ValueError("training_steps must be a positive integer")
        if not _is_strict_integer(selected_batch_size) or selected_batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        steps = int(selected_steps)
        size = int(selected_batch_size)
        key, init_key = jax.random.split(key)
        params = self.init_params(init_key)
        opt_state = init_adam(params)
        self._ema_params = params
        losses = []
        final_metrics: dict[str, Array] = {}
        for _ in range(steps):
            key, step_key = jax.random.split(key)
            params, opt_state, final_metrics = self.train_step(step_key, params, opt_state, size)
            losses.append(final_metrics["loss"])
        self._params = params
        assert self._ema_params is not None
        return MalliavinAdjointResult(
            params,
            self._ema_params,
            jnp.asarray(losses),
            final_metrics,
        )

    def extract_costate(
        self,
        params: Params | None = None,
        *,
        use_ema: bool = True,
    ) -> Callable[[Array, Scalar, Array], Array]:
        active = params
        if active is None and use_ema:
            active = self._ema_params
        if active is None:
            active = self._params
        if active is None:
            raise ValueError("No costate parameters available; train or provide params")

        def costate(state: Array, time: Scalar, endpoint: Array) -> Array:
            state_batch = jnp.atleast_2d(state)
            endpoint_batch = _broadcast_context(endpoint, state_batch.shape[0])
            time_array = jnp.asarray(time, dtype=state_batch.dtype)
            if time_array.ndim == 0:
                time_array = jnp.full((state_batch.shape[0],), time_array)
            value = jnp.asarray(
                self._factory.forward(
                    active,
                    self._network_input(state_batch, endpoint_batch),
                    time_array,
                ),
                dtype=state_batch.dtype,
            )
            expected_shape = (state_batch.shape[0], self.problem.dim)
            if value.shape != expected_shape:
                raise ValueError(
                    f"costate factory output must have shape {expected_shape}, got {value.shape}"
                )
            return value

        return costate

    def make_action_target_batch(
        self,
        key: PRNGKey,
        state: Array,
        time: Array | Scalar,
        endpoint: Array,
        *,
        next_time: Array | Scalar,
        params: Params | None = None,
        current_control: Array | None = None,
        num_antithetic: int = 1,
    ) -> ActionTargetBatch:
        """Generate corrected stopped targets for a separate actor regressor."""
        state = jnp.atleast_2d(state)
        endpoint = _broadcast_context(endpoint, state.shape[0])
        time_array = _as_batch_vector(time, state.shape[0], "time", state.dtype)
        next_time_array = _as_batch_vector(
            next_time,
            state.shape[0],
            "next_time",
            state.dtype,
        )
        if current_control is None:
            if self.control_fn is None:
                current = jnp.zeros_like(state)
            else:
                current = jnp.asarray(
                    self.control_fn(state, time_array, endpoint),
                    dtype=state.dtype,
                )
        else:
            current = jnp.asarray(current_control, dtype=state.dtype)

        running_cost_fn: RunningCostFn
        if self.value_cost.running_cost is None:

            def zero_running_cost(x: Array, t: Array, context: Array) -> Array:
                del t, context
                return jnp.zeros((x.shape[0],), dtype=x.dtype)

            running_cost_fn = zero_running_cost
        else:
            running_cost_fn = self.value_cost.running_cost

        extracted = self.extract_costate(params)
        declared_grid = jnp.asarray(self.problem.time_grid.times, dtype=state.dtype)
        declared_departures = declared_grid[:-2]
        declared_arrivals = declared_grid[1:-1]
        adjacent_grid_pair = jnp.any(
            (time_array[:, None] == declared_departures[None, :])
            & (next_time_array[:, None] == declared_arrivals[None, :]),
            axis=1,
        )
        _require_concrete(
            jnp.all(adjacent_grid_pair),
            "time and next_time must be an adjacent stochastic pair on the declared grid",
        )
        penultimate_time = jnp.asarray(
            self.problem.time_grid.times[-2],
            dtype=state.dtype,
        )

        def continuation_costate(x: Array, t: Array, context: Array) -> Array:
            value = extracted(x, t, context)
            # The pair was validated against the represented grid above, so
            # exact equality identifies only the final stochastic arrival.
            final_arrival = t == penultimate_time
            return jnp.where(final_arrival[:, None], 0.0, value)

        result = assemble_pinned_actor_targets(
            key,
            state,
            endpoint,
            time_array,
            next_time_array,
            self.problem.time_grid.t1,
            self._diffusion,
            current,
            running_cost_fn,
            continuation_costate,
            num_antithetic=num_antithetic,
            diffusion_rcond=self.mam_config.diffusion_rcond,
        )
        finite = result.finite & adjacent_grid_pair
        return result._replace(
            target=_poison_float(result.target, finite[:, None]),
            continuation_component=_poison_float(
                result.continuation_component,
                finite[:, None],
            ),
            arrival_component=_poison_float(result.arrival_component, finite[:, None]),
            mean_state=_poison_float(result.mean_state, finite[:, None]),
            innovation=_poison_float(result.innovation, finite[:, None, None]),
            finite=finite,
        )

    def propose_control(
        self,
        state: Array,
        time: Scalar,
        endpoint: Array,
        params: Params | None = None,
        current_control: Array | None = None,
        *,
        rho: Array | Scalar | None = None,
        next_state: Array | None = None,
        next_time: Array | Scalar | None = None,
        step_size: float | None = None,
    ) -> ControlProposal:
        """Return a stopped conservative proposal, not an endpoint projection.

        Without ``rho``, this returns the continuous-time target
        ``-Sigma.T p(t,x)``.  With ``rho``, callers must also supply a sampled
        ``next_state`` and ``next_time``; the returned one-sample target is
        ``-sqrt(rho) Sigma.T p(t_next,x_next)``.  That legacy branch is exact
        only when the arrival running cost is zero.  For a nonzero running
        potential callers must use :meth:`make_action_target_batch`, which
        adds the chain-induced arrival correction.
        The supplied ``rho`` must equal ``(T-next_time)/(T-time)`` up to
        ``atol=rtol=32*eps(state.dtype)``.
        """
        state = jnp.atleast_2d(state)
        endpoint = _broadcast_context(endpoint, state.shape[0])
        time_array = _as_batch_vector(time, state.shape[0], "time", state.dtype)
        costate_state = state
        costate_time = time_array
        rho_array = None
        update_semantics = "continuous_hamiltonian"
        if rho is not None:
            if self.value_cost.running_cost is not None:
                raise ValueError(
                    "the legacy discrete proposal omits the arrival running cost; "
                    "use make_action_target_batch"
                )
            if next_state is None or next_time is None:
                raise ValueError("rho requires next_state and next_time for the discrete target")
            rho_array = _as_batch_vector(rho, state.shape[0], "rho", state.dtype)
            _require_concrete(jnp.all(jnp.isfinite(rho_array)), "rho must be finite")
            _require_concrete(
                jnp.all((rho_array > 0.0) & (rho_array < 1.0)),
                "rho must satisfy 0 < rho < 1",
            )
            costate_state = jnp.atleast_2d(next_state)
            if costate_state.shape != state.shape:
                raise ValueError("next_state must match current state shape")
            costate_time = _as_batch_vector(
                next_time,
                state.shape[0],
                "next_time",
                state.dtype,
            )
            _require_concrete(
                jnp.all(jnp.isfinite(time_array)) & jnp.all(jnp.isfinite(costate_time)),
                "time and next_time must be finite",
            )
            initial_time = jnp.asarray(self.problem.time_grid.t0, dtype=state.dtype)
            terminal_time = jnp.asarray(self.problem.time_grid.t1, dtype=state.dtype)
            _require_concrete(
                jnp.all((time_array >= initial_time) & (time_array < terminal_time)),
                "time must lie in the bridge interval [t0, t1)",
            )
            _require_concrete(
                jnp.all(costate_time > time_array),
                "next_time must be strictly greater than time",
            )
            _require_concrete(
                jnp.all(costate_time < terminal_time),
                "next_time must be strictly before the terminal time",
            )
            remaining_time = terminal_time - time_array
            expected_rho = (terminal_time - costate_time) / remaining_time
            tolerance = _RHO_CONSISTENCY_EPS_MULTIPLIER * jnp.finfo(state.dtype).eps
            _require_concrete(
                jnp.all(
                    jnp.isclose(
                        rho_array,
                        expected_rho,
                        rtol=tolerance,
                        atol=tolerance,
                    )
                ),
                "rho must agree with (t1 - next_time) / (t1 - time) within "
                "atol=rtol=32*eps(state.dtype)",
            )
            update_semantics = "one_sample_pinned_discrete"
        costate = jax.lax.stop_gradient(
            self.extract_costate(params)(costate_state, costate_time, endpoint)
        )
        target = -(costate @ self._diffusion)
        if rho_array is not None:
            target = jnp.sqrt(rho_array)[:, None] * target
        if self.mam_config.max_control_norm is not None:
            norm = jnp.linalg.norm(target, axis=-1, keepdims=True)
            scale = jnp.minimum(
                1.0,
                self.mam_config.max_control_norm / jnp.maximum(norm, 1e-12),
            )
            target = target * scale
        if current_control is None:
            if self.control_fn is None:
                current = jnp.zeros_like(target)
            else:
                current = jnp.asarray(
                    self.control_fn(state, time_array, endpoint), dtype=state.dtype
                )
        else:
            current = jnp.asarray(current_control, dtype=state.dtype)
            if current.ndim == 1:
                current = current[None, :]
            current = jnp.broadcast_to(current, target.shape)
        eta = float(self.mam_config.trust_region if step_size is None else step_size)
        if not 0.0 <= eta <= 1.0:
            raise ValueError("step_size must be in [0, 1]")
        proposed = (1.0 - eta) * current + eta * target
        return ControlProposal(
            current,
            target,
            proposed,
            costate,
            jnp.asarray(eta, dtype=state.dtype),
            jnp.asarray(False),
            "cost_minimization",
            "brownian_noise_control",
            update_semantics,
            True,
        )


__all__ = [
    "ActionTargetBatch",
    "DirectActionScoreBatch",
    "AdjointRollout",
    "BELCostateBatch",
    "ControlProposal",
    "MatrixFreeAdjointRollout",
    "MalliavinAdjointConfig",
    "MalliavinAdjointInnerSolver",
    "MalliavinAdjointResult",
    "ValueOnlyCost",
    "assemble_bel_costate_labels",
    "assemble_pinned_actor_targets",
    "assemble_antithetic_direct_action_score",
    "assemble_pinned_brownian_labels",
    "assemble_pinned_brownian_labels_matrix_free",
    "simulate_additive_em_rollout",
    "simulate_pinned_brownian_rollout",
    "simulate_pinned_brownian_rollout_matrix_free",
    "summarize_costate_labels",
]
