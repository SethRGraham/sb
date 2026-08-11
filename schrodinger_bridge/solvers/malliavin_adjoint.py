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
from typing import Any, NamedTuple

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
        values = jnp.asarray(
            self.running_cost(flat_states, flat_times, flat_context),
            dtype=states.dtype,
        ).reshape((batch_size, num_times))
        return jax.lax.stop_gradient(values)

    def terminal_values(self, states: Array, context: Array) -> Array:
        """Evaluate stopped terminal values on terminal states ``[B,d]``."""
        states = jnp.atleast_2d(states)
        context = _broadcast_context(context, states.shape[0])
        if self.terminal_cost is None:
            return jnp.zeros((states.shape[0],), dtype=states.dtype)
        values = jnp.asarray(self.terminal_cost(states, context), dtype=states.dtype)
        values = values.reshape((states.shape[0],))
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
    ema_decay: float = 0.999
    trust_region: float = 0.1
    max_control_norm: float | None = None
    include_control_energy: bool = True
    diffusion_rcond: float = 1e-8
    network_factory: NetworkFactory | None = None

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.time_embed_dim < 2:
            raise ValueError("time_embed_dim must be at least two")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.training_steps < 1 or self.batch_size < 1:
            raise ValueError("training_steps and batch_size must be positive")
        if self.minimum_remaining_steps < 1:
            raise ValueError("minimum_remaining_steps must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0.0 <= self.trust_region <= 1.0:
            raise ValueError("trust_region must be in [0, 1]")
        if self.max_control_norm is not None and self.max_control_norm <= 0:
            raise ValueError("max_control_norm must be positive when supplied")
        if self.diffusion_rcond <= 0:
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
    if isinstance(sigma, jax.core.Tracer):
        return
    singular_values = np.linalg.svd(np.asarray(jax.device_get(sigma)), compute_uv=False)
    if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
        raise ValueError("diffusion singular values must be finite")
    if singular_values[-1] <= rcond * singular_values[0]:
        raise ValueError("MAM v1 requires a full-rank, well-conditioned diffusion matrix")


def _validate_uniform_times(times: Array) -> Scalar:
    if isinstance(times, jax.core.Tracer):
        return times[1] - times[0]
    raw_times = np.asarray(jax.device_get(times))
    times_host = raw_times.astype(float, copy=False)
    if times_host.ndim != 1 or times_host.size < 2:
        raise ValueError("times must be one-dimensional with at least two points")
    increments = np.diff(times_host)
    if not np.all(increments > 0):
        raise ValueError("times must be strictly increasing")
    # ``linspace`` rounds adjacent float32 intervals slightly differently.
    # The validation tolerance follows the represented grid dtype while
    # remaining strict enough to reject a genuinely nonuniform grid.
    dtype = raw_times.dtype if np.issubdtype(raw_times.dtype, np.floating) else np.dtype(float)
    tolerance = max(1e-12, 64.0 * np.finfo(dtype).eps)
    if not np.allclose(increments, increments[0], rtol=tolerance, atol=tolerance):
        raise ValueError("MAM v1 requires a uniform time grid")
    return float(increments[0])


def _validate_anchors(anchors: Array, *, maximum: int) -> None:
    if isinstance(anchors, jax.core.Tracer):
        return
    values = np.asarray(jax.device_get(anchors))
    if values.ndim != 1:
        raise ValueError("anchors must have shape [batch]")
    if np.any(values < 0) or np.any(values > maximum):
        raise ValueError(f"anchors must lie in [0, {maximum}]")


def _batch_drift_jacobian(
    drift_fn: DriftFnWithContext,
    x: Array,
    t: Scalar,
    context: Array,
) -> Array:
    def one(x_i: Array, context_i: Array) -> Array:
        return jax.jacfwd(
            lambda value: jnp.asarray(drift_fn(value[None, :], t, context_i[None, :]))[0]
        )(x_i)

    return jax.vmap(one)(x, context)


def _batch_control_jacobian(
    control_fn: NoiseControlFn,
    x: Array,
    t: Scalar,
    endpoint: Array,
) -> Array:
    def one(x_i: Array, endpoint_i: Array) -> Array:
        return jax.jacfwd(
            lambda value: jnp.asarray(control_fn(value[None, :], t, endpoint_i[None, :]))[0]
        )(x_i)

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
    x0 = jnp.atleast_2d(x0)
    times = jnp.asarray(times, dtype=x0.dtype)
    dt = _validate_uniform_times(times)
    batch_size, dim = x0.shape
    context = _broadcast_context(context, batch_size)
    sigma = _constant_diffusion_matrix(diffusion, dim, x0.dtype)
    _validate_full_rank(sigma, diffusion_rcond)

    num_steps = times.shape[0] - 1
    d_w = jnp.sqrt(dt) * jax.random.normal(key, (batch_size, num_steps, dim), dtype=x0.dtype)
    scan_d_w = jnp.swapaxes(d_w, 0, 1)
    eye = jnp.eye(dim, dtype=x0.dtype)

    def step(x: Array, inputs: tuple[Scalar, Array]):
        t, d_w_step = inputs
        drift = jnp.asarray(drift_fn(x, t, context), dtype=x.dtype)
        jacobian = _batch_drift_jacobian(drift_fn, x, t, context)
        local = eye[None, :, :] + dt * jacobian
        x_next = x + dt * drift + d_w_step @ sigma.T
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
    return AdjointRollout(states, d_w, local, noise_matrices, controls, times, context)


def _em_label_one(
    local_jacobians: Array,
    d_w: Array,
    sigma_steps: Array,
    running_values: Array,
    terminal_value: Array,
    anchor: Array,
    dt: float,
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
    batch_size, num_times, _ = rollout.states.shape
    num_steps = num_times - 1
    anchors = jnp.asarray(anchors, dtype=jnp.int32)
    _validate_anchors(anchors, maximum=num_steps - 1)
    if anchors.shape != (batch_size,):
        raise ValueError(f"anchors must have shape {(batch_size,)}, got {anchors.shape}")
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

    dt = rollout.times[1] - rollout.times[0]
    stopped_running = jax.lax.stop_gradient(running_values)
    stopped_terminal = jax.lax.stop_gradient(terminal_values)
    outputs = jax.vmap(_em_label_one, in_axes=(0, 0, None, 0, 0, 0, None))(
        rollout.local_jacobians,
        rollout.innovations,
        rollout.noise_matrices,
        stopped_running,
        stopped_terminal,
        anchors,
        dt,
    )
    label, terminal, running, terminal_weight, condition, finite = outputs
    direct = jax.lax.stop_gradient(immediate_gradients)
    label = label + direct
    finite = finite & jnp.all(jnp.isfinite(direct), axis=-1)
    batch_indices = jnp.arange(batch_size)
    anchor_states = rollout.states[batch_indices, anchors]
    anchor_times = rollout.times[anchors]
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
    x0 = jnp.atleast_2d(x0)
    endpoint = jnp.atleast_2d(endpoint)
    if endpoint.shape != x0.shape:
        raise ValueError(f"endpoint shape {endpoint.shape} must match x0 {x0.shape}")
    times = jnp.asarray(times, dtype=x0.dtype)
    dt = _validate_uniform_times(times)
    batch_size, dim = x0.shape
    sigma = _constant_diffusion_matrix(diffusion, dim, x0.dtype)
    _validate_full_rank(sigma, diffusion_rcond)
    num_steps = times.shape[0] - 1
    if num_steps < 2:
        raise ValueError("pinned bridge requires at least two time steps")
    stochastic_steps = num_steps - 1
    innovations = jax.random.normal(
        key,
        (batch_size, stochastic_steps, dim),
        dtype=x0.dtype,
    )
    scan_innovations = jnp.swapaxes(innovations, 0, 1)
    t_terminal = times[-1]
    eye = jnp.eye(dim, dtype=x0.dtype)

    def zero_control(x: Array, t: Scalar, context: Array) -> Array:
        del t, context
        return jnp.zeros_like(x)

    active_control = control_fn or zero_control

    def step(x: Array, inputs: tuple[Array, Scalar, Scalar, Array]):
        index, t, t_next, innovation = inputs
        del index
        rho = (t_terminal - t_next) / (t_terminal - t)
        gamma = jnp.sqrt(dt * rho) * sigma
        control = jnp.asarray(active_control(x, t, endpoint), dtype=x.dtype)
        control_jacobian = _batch_control_jacobian(active_control, x, t, endpoint)
        local = rho * eye[None, :, :] + jnp.sqrt(dt) * jnp.einsum(
            "ij,bjk->bik", gamma, control_jacobian
        )
        mean = rho * x + (1.0 - rho) * endpoint + jnp.sqrt(dt) * (control @ gamma.T)
        x_next = mean + innovation @ gamma.T
        return x_next, (x_next, local, control, gamma)

    indices = jnp.arange(stochastic_steps)
    _, (interior_time_major, local_time_major, control_time_major, gamma_time_major) = jax.lax.scan(
        step,
        x0,
        (
            indices,
            times[:stochastic_steps],
            times[1 : stochastic_steps + 1],
            scan_innovations,
        ),
    )
    interior = jnp.swapaxes(interior_time_major, 0, 1)
    states = jnp.concatenate([x0[:, None, :], interior, endpoint[:, None, :]], axis=1)
    local = jnp.swapaxes(local_time_major, 0, 1)
    controls = jnp.swapaxes(control_time_major, 0, 1)
    return AdjointRollout(
        states,
        innovations,
        local,
        gamma_time_major,
        controls,
        times,
        endpoint,
    )


def _pinned_label_one(
    local_jacobians: Array,
    innovations: Array,
    gamma_steps: Array,
    running_values: Array,
    anchor: Array,
    dt: float,
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
    anchors = jnp.asarray(anchors, dtype=jnp.int32)
    _validate_anchors(anchors, maximum=stochastic_steps - 1)
    if anchors.shape != (batch_size,):
        raise ValueError(f"anchors must have shape {(batch_size,)}, got {anchors.shape}")
    running_values = jnp.asarray(running_values, dtype=rollout.states.dtype)
    if running_values.shape != (batch_size, num_times):
        raise ValueError(
            f"running_values must have shape {(batch_size, num_times)}, got {running_values.shape}"
        )
    if terminal_values is not None:
        terminal_values = jnp.asarray(terminal_values, dtype=rollout.states.dtype)
        if terminal_values.shape != (batch_size,):
            raise ValueError("terminal_values must have shape [batch]")
        jax.lax.stop_gradient(terminal_values)
    if immediate_gradients is None:
        immediate_gradients = jnp.zeros((batch_size, dim), dtype=rollout.states.dtype)
    immediate_gradients = jnp.asarray(immediate_gradients, dtype=rollout.states.dtype)
    if immediate_gradients.shape != (batch_size, dim):
        raise ValueError("immediate_gradients must have shape [batch, state_dim]")

    dt = rollout.times[1] - rollout.times[0]
    stopped_running = jax.lax.stop_gradient(running_values)
    running, condition, finite = jax.vmap(
        _pinned_label_one,
        in_axes=(0, 0, None, 0, 0, None),
    )(
        rollout.local_jacobians,
        rollout.innovations,
        rollout.noise_matrices,
        stopped_running,
        anchors,
        dt,
    )
    zeros = jnp.zeros((batch_size, dim), dtype=rollout.states.dtype)
    direct = jax.lax.stop_gradient(immediate_gradients)
    label = running + direct
    finite = finite & jnp.all(jnp.isfinite(direct), axis=-1)
    batch_indices = jnp.arange(batch_size)
    anchor_states = rollout.states[batch_indices, anchors]
    anchor_times = rollout.times[anchors]
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
        if not isinstance(self.problem.reference, BrownianMotion):
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
        params = self._factory.init(key, input_dim, self.problem.dim)
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
        if not self.mam_config.include_control_energy or self.control_fn is None:
            return jnp.zeros((batch_size, self.problem.dim), dtype=rollout.states.dtype)
        batch_indices = jnp.arange(batch_size)
        states = rollout.states[batch_indices, anchors]
        times = rollout.times[anchors]
        endpoints = rollout.context

        def one(state: Array, time: Scalar, endpoint: Array) -> Array:
            def energy(value: Array) -> Array:
                control = jnp.asarray(
                    self.control_fn(value[None, :], time, endpoint[None, :]),
                    dtype=value.dtype,
                )[0]
                return 0.5 * jnp.sum(control**2)

            return jax.grad(energy)(state)

        dt = rollout.times[1] - rollout.times[0]
        return jax.lax.stop_gradient(dt * jax.vmap(one)(states, times, endpoints))

    def make_label_batch(
        self,
        key: PRNGKey,
        x0: Array,
        endpoint: Array,
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
        anchors = jax.random.randint(
            anchor_key,
            (x0.shape[0],),
            minval=0,
            maxval=maximum_anchor + 1,
        )
        running, terminal = self._evaluate_costs(rollout)
        immediate = self._immediate_control_energy_gradient(rollout, anchors)
        return assemble_pinned_brownian_labels(
            rollout,
            anchors,
            running,
            terminal,
            immediate,
        )

    def sample_label_batch(self, key: PRNGKey, batch_size: int) -> BELCostateBatch:
        """Generate labels under an explicitly independent endpoint coupling."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        pair_key, batch_key = jax.random.split(key)
        x0, endpoint = self.problem.sample_pair(pair_key, batch_size)
        return self.make_label_batch(batch_key, x0, endpoint)

    def loss(self, params: Params, batch: BELCostateBatch) -> tuple[Array, dict[str, Array]]:
        prediction = self._factory.forward(
            params,
            self._network_input(batch.anchor_state, batch.context),
            batch.anchor_time,
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
        size = int(batch_size or self.mam_config.batch_size)
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
        steps = int(training_steps or self.mam_config.training_steps)
        size = int(batch_size or self.mam_config.batch_size)
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
            return self._factory.forward(
                active,
                self._network_input(state_batch, endpoint_batch),
                time_array,
            )

        return costate

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
        ``-sqrt(rho) Sigma.T p(t_next,x_next)``.  Regressing/averaging these
        samples conditional on the current state estimates the exact discrete
        target.  A single proposal is not itself that conditional expectation.
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
    "AdjointRollout",
    "BELCostateBatch",
    "ControlProposal",
    "MalliavinAdjointConfig",
    "MalliavinAdjointInnerSolver",
    "MalliavinAdjointResult",
    "ValueOnlyCost",
    "assemble_bel_costate_labels",
    "assemble_pinned_brownian_labels",
    "simulate_additive_em_rollout",
    "simulate_pinned_brownian_rollout",
    "summarize_costate_labels",
]
