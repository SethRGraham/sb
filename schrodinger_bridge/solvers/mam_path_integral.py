r"""Path-integral reference controls for endpoint-pinned Brownian chains.

This module is a deliberately small *kill baseline* for MAM.  It targets the
same finite, endpoint-pinned chain and the same noise/control geometry as the
conditional MAM solver.  From an anchor state ``x`` and fixed endpoint ``y``,

.. math::

    X_{j+1}=\rho_j X_j+(1-\rho_j)y
              +\Gamma_j(\sqrt{\Delta t}\,u_j^{\rm ref}+\xi_j),
    \qquad \xi_j\sim N(0,I).

The last transition is deterministic and is not sampled.  Let

.. math::

    C=\Delta t\sum_j \ell_{j+1}(X_{j+1}).

Because the Gaussian noise and control channels are matched, the zero-control
pinned law ``P0`` and an adapted reference-control law ``Qref`` obey

.. math::

    \log {dP_0\over dQ_{\rm ref}}
      =-\sum_j\left(\sqrt{\Delta t}\,u_j^{\rm ref}\!\cdot\xi_j
                    +{\Delta t\over2}\|u_j^{\rm ref}\|^2\right).

Consequently, with ``w = exp(-C) dP0/dQref``, define the exponentially tilted
path law

.. math::

    {dQ^*\over dP_0}={e^{-C}\over E_{P_0}[e^{-C}]}.

When the normalizer is finite and nonzero and the displayed expectations exist,
this is the population minimizer of the unrestricted Gibbs/KL objective
``E_Q[C] + KL(Q || P0)``.  Its first zero-control innovation moment, expressed
using samples from ``Qref``, is

.. math::

    u_{\rm tilt}(x) = u_0^{\rm ref}(x)
      + {1\over\sqrt{\Delta t}}
        {E_{Q_{\rm ref}}[w\xi_0]\over E_{Q_{\rm ref}}[w]}.

This moment identity is exact in population.  It is **not**, in general, the
minimizer of the finite-step objective restricted to Gaussian transition laws
with fixed covariance and a mean-shift control.  Exponential tilting does not
preserve that restricted family for a general, and especially discontinuous,
cost.  The continuous-time linearly-solvable control interpretation therefore
requires its own limiting assumptions; this finite-chain routine reports only
the exact tilted-law moment that it computes.

Its self-normalized Monte Carlo estimate is a finite-sample ratio estimator and
is generally biased.  The returned diagnostics say so explicitly and reject
low-ESS updates rather than hiding importance-weight collapse.  No
differentiability of ``ell`` is used: cost values are stopped before the
weights are assembled.

The companion Feynman--Kac routine estimates the static endpoint kernel

.. math::

    K_\ell(x,y)=p_0(y\mid x) E_{P_0^{x,y}}[e^{-C}].

The kernel estimate in normal scale is unbiased in exact arithmetic; taking
its logarithm introduces the usual finite-sample log bias.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from ..core.types import Array, PRNGKey, Scalar

ValueOnlyRunningCost = Callable[[Array, Array, Array], Array]
NoiseControlFn = Callable[[Array, Scalar, Array], Array]


@dataclass(frozen=True)
class PathIntegralConfig:
    """Sampling and fail-closed thresholds for the path-integral baseline.

    ``num_samples`` is the number of independent Gaussian suffixes when
    ``antithetic`` is false and the number of independent *pairs* otherwise.
    The conventional importance ESS is reported over the resulting
    trajectories.  Under antithetic sampling those trajectories are dependent,
    so the ESS remains a collapse diagnostic rather than an independent-sample
    count.
    """

    num_samples: int = 256
    antithetic: bool = True
    minimum_ess_fraction: float = 0.01
    diffusion_rcond: float = 1e-8
    initial_control_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if isinstance(self.num_samples, bool) or not isinstance(
            self.num_samples, (int, np.integer)
        ):
            raise TypeError("num_samples must be an integer")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if not isinstance(self.antithetic, (bool, np.bool_)):
            raise TypeError("antithetic must be boolean")
        for name, value in (
            ("minimum_ess_fraction", self.minimum_ess_fraction),
            ("diffusion_rcond", self.diffusion_rcond),
            ("initial_control_tolerance", self.initial_control_tolerance),
        ):
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{name} must be a real scalar")
        if not 0.0 < self.minimum_ess_fraction <= 1.0:
            raise ValueError("minimum_ess_fraction must lie in (0, 1]")
        if self.diffusion_rcond <= 0.0 or not np.isfinite(self.diffusion_rcond):
            raise ValueError("diffusion_rcond must be finite and positive")
        if self.initial_control_tolerance < 0.0 or not np.isfinite(self.initial_control_tolerance):
            raise ValueError("initial_control_tolerance must be finite and nonnegative")


class PinnedPathIntegralSamples(NamedTuple):
    """Reference suffix paths used by the pure path-integral assembler.

    Shapes:
        states: ``[B,M,S+2,d]`` (including start and deterministic endpoint)
        innovations: ``[B,M,S,d]``
        reference_controls: ``[B,M,S,d]``
        running_values: ``[B,M,S]`` at stochastic arrivals
        times: ``[S+2]``
        start, endpoint: ``[B,d]``

    ``M`` is the total trajectory count and ``S`` is the number of stochastic
    transitions.  ``physical_query_count`` counts scalar cost-oracle values,
    not vectorized Python/JAX invocations.
    """

    states: Array
    innovations: Array
    reference_controls: Array
    running_values: Array
    times: Array
    start: Array
    endpoint: Array
    finite: Array
    physical_query_count: Array
    trajectory_count: Array
    independent_base_draw_count: Array
    antithetic: Array


class PathIntegralControlEstimate(NamedTuple):
    """Self-normalized tilted-path first-noise moment and diagnostics.

    ``raw_control_target`` is the finite ratio estimate even when its ESS is
    too small.  ``control_target`` fails closed to the reference control unless
    ``usable`` is true.  This makes accidental use of a collapsed update less
    likely without clipping or silently changing the theorem-facing estimate.

    ``exact_tilted_path_moment_in_population`` records the exact scope of the
    population identity under the declared likelihood and exponential-moment
    conditions: the target is the first zero-control innovation mean under the
    unrestricted exponentially tilted path law.  The false
    ``exact_finite_gaussian_shift_optimum`` flag records that no general
    finite-step optimality claim is made for the fixed-covariance Gaussian
    mean-shift family; special costs can still make the two targets coincide.
    ``exact_in_population`` is retained as a deprecated compatibility alias for
    ``exact_tilted_path_moment_in_population`` and must not be read more broadly.
    """

    control_target: Array
    raw_control_target: Array
    reference_control: Array
    correction: Array
    log_desirability: Array
    effective_sample_size: Array
    ess_fraction: Array
    maximum_normalized_weight: Array
    finite: Array
    degenerate: Array
    usable: Array
    physical_query_count: Array
    trajectory_count: Array
    independent_base_draw_count: Array
    exact_tilted_path_moment_in_population: Array
    exact_finite_gaussian_shift_optimum: Array
    exact_in_population: Array
    finite_sample_ratio_biased: Array


class FeynmanKacKernelEstimate(NamedTuple):
    """Static killed Brownian endpoint-kernel estimate.

    ``kernel`` is the normal-scale Monte Carlo estimate.  It is unbiased in
    exact arithmetic when the supplied path weights use the declared
    reference-law correction.  ``log_kernel`` is numerically preferable but
    is a biased finite-sample transform.  A finite ``log_kernel`` remains
    usable by log-domain consumers even when exponentiation overflows or
    underflows.  ``normal_scale_representable`` says whether ``kernel`` is a
    finite, strictly positive floating-point value.  ``finite`` is retained as
    a compatibility alias for ``log_domain_finite``.
    """

    kernel: Array
    log_kernel: Array
    log_reference_endpoint_density: Array
    log_bridge_desirability: Array
    effective_sample_size: Array
    ess_fraction: Array
    log_domain_finite: Array
    normal_scale_representable: Array
    finite: Array
    degenerate: Array
    usable: Array
    unbiased_in_exact_arithmetic: Array
    finite_sample_log_biased: Array


@dataclass(frozen=True)
class StaticKernelSinkhornConfig:
    """Fixed-iteration scaling settings for a static Feynman--Kac kernel.

    This is a finite discrete coupling baseline, not an assertion that Monte
    Carlo kernel estimation or finitely stopped Sinkhorn exactly preserves the
    population endpoint marginals.
    """

    iterations: int = 500
    tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, (int, np.integer)):
            raise TypeError("iterations must be an integer")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if isinstance(self.tolerance, bool) or not isinstance(
            self.tolerance, (int, float, np.integer, np.floating)
        ):
            raise TypeError("tolerance must be a real scalar")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")


class StaticKernelSinkhornResult(NamedTuple):
    """Log-domain scaling of one positive static endpoint kernel.

    ``coupling`` has shape ``[num_source,num_target]``.  ``converged`` refers
    only to the supplied finite empirical kernel and tolerance.
    """

    coupling: Array
    source_marginal: Array
    target_marginal: Array
    marginal_error: Array
    finite: Array
    converged: Array
    usable: Array
    iterations: Array


def _require_concrete(predicate: Array, message: str) -> None:
    """Raise eagerly while retaining a validity mask under ``jax.jit``."""
    if isinstance(predicate, jax.core.Tracer):
        return
    if not bool(np.all(np.asarray(jax.device_get(predicate)))):
        raise ValueError(message)


def _normalize_states(start: Array, endpoint: Array) -> tuple[Array, Array]:
    start = jnp.asarray(start)
    if not np.issubdtype(start.dtype, np.inexact):
        start = start.astype(jnp.float32)
    start = jnp.atleast_2d(start)
    endpoint = jnp.asarray(endpoint, dtype=start.dtype)
    endpoint = jnp.atleast_2d(endpoint)
    if start.ndim != 2 or start.shape[-1] < 1:
        raise ValueError("start must have shape [B,d] with d >= 1")
    if endpoint.shape != start.shape:
        raise ValueError(f"endpoint shape {endpoint.shape} must match start {start.shape}")
    return start, endpoint


def _constant_diffusion_matrix(diffusion: Array | Scalar, dim: int, dtype: jnp.dtype) -> Array:
    value = jnp.asarray(diffusion, dtype=dtype)
    if value.ndim == 0:
        return value * jnp.eye(dim, dtype=dtype)
    if value.ndim == 1 and value.shape == (dim,):
        return jnp.diag(value)
    if value.ndim == 2 and value.shape == (dim, dim):
        return value
    raise ValueError("diffusion must be scalar, diagonal [d], or a constant square [d,d] matrix")


def _diffusion_valid(sigma: Array, rcond: float) -> Array:
    singular_values = jnp.linalg.svd(sigma, compute_uv=False)
    valid = (
        jnp.all(jnp.isfinite(sigma))
        & jnp.all(jnp.isfinite(singular_values))
        & (singular_values[-1] > jnp.asarray(rcond, dtype=sigma.dtype) * singular_values[0])
    )
    _require_concrete(valid, "pinned path-integral V1 requires finite full-rank diffusion")
    return valid


def _uniform_time_data(times: Array, dtype: jnp.dtype) -> tuple[Array, Array, Array]:
    times = jnp.asarray(times, dtype=dtype)
    if times.ndim != 1 or times.shape[0] < 3:
        raise ValueError("times must be one-dimensional with at least three points")
    increments = jnp.diff(times)
    interval_count = jnp.asarray(times.shape[0] - 1, dtype=dtype)
    dt = (times[-1] - times[0]) / interval_count
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(dt), jnp.max(jnp.abs(increments))),
        jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
    )
    tolerance = 64.0 * jnp.finfo(dtype).eps * scale
    valid = (
        jnp.all(jnp.isfinite(times))
        & jnp.all(jnp.isfinite(increments))
        & (dt > 0.0)
        & jnp.all(increments > 0.0)
        & jnp.all(jnp.abs(increments - dt) <= tolerance)
    )
    _require_concrete(valid, "times must be finite, strictly increasing, and uniform")
    return times, dt, valid


def _single_control_value(
    control: NoiseControlFn,
    state: Array,
    time: Scalar,
    endpoint: Array,
) -> Array:
    """Evaluate one independent path row under the declared Markov control."""
    value = jnp.asarray(
        control(state[None, :], time, endpoint[None, :]),
        dtype=state.dtype,
    )
    expected = (1, state.shape[0])
    if value.shape != expected:
        raise ValueError(f"reference_control must return shape {expected}, got {value.shape}")
    return value[0]


def _batch_control_values(
    control: NoiseControlFn,
    state: Array,
    time: Scalar,
    endpoint: Array,
) -> Array:
    return jax.vmap(
        lambda state_i, endpoint_i: _single_control_value(
            control,
            state_i,
            time,
            endpoint_i,
        )
    )(state, endpoint)


def _single_running_value(
    running_cost: ValueOnlyRunningCost,
    state: Array,
    time: Scalar,
    endpoint: Array,
) -> Array:
    """Evaluate one independent path-row cost with an exact scalar output."""
    time_row = jnp.reshape(jnp.asarray(time, dtype=state.dtype), (1,))
    value = jnp.asarray(
        running_cost(state[None, :], time_row, endpoint[None, :]),
        dtype=state.dtype,
    )
    if value.shape != (1,):
        raise ValueError(f"running_cost must return shape {(1,)}, got {value.shape}")
    return value[0]


def _batch_running_values(
    running_cost: ValueOnlyRunningCost,
    state: Array,
    time: Array,
    endpoint: Array,
) -> Array:
    return jax.vmap(
        lambda state_i, time_i, endpoint_i: _single_running_value(
            running_cost,
            state_i,
            time_i,
            endpoint_i,
        )
    )(state, time, endpoint)


def _validate_sample_arrays(
    innovations: Array,
    reference_controls: Array,
    running_values: Array,
) -> tuple[Array, Array, Array]:
    innovations = jnp.asarray(innovations)
    if not np.issubdtype(innovations.dtype, np.inexact):
        innovations = innovations.astype(jnp.float32)
    reference_controls = jnp.asarray(reference_controls, dtype=innovations.dtype)
    running_values = jnp.asarray(running_values, dtype=innovations.dtype)
    if innovations.ndim != 4 or innovations.shape[-1] < 1:
        raise ValueError("innovations must have shape [B,M,S,d]")
    if reference_controls.shape != innovations.shape:
        raise ValueError("reference_controls must have the same shape as innovations")
    if running_values.shape != innovations.shape[:3]:
        raise ValueError("running_values must have shape [B,M,S]")
    if innovations.shape[1] < 1 or innovations.shape[2] < 1:
        raise ValueError("at least one trajectory and stochastic transition are required")
    return innovations, reference_controls, running_values


def simulate_pinned_reference_suffix(
    key: PRNGKey,
    start: Array,
    endpoint: Array,
    times: Array,
    diffusion: Array | Scalar,
    running_cost: ValueOnlyRunningCost,
    reference_control: NoiseControlFn | None = None,
    *,
    config: PathIntegralConfig | None = None,
) -> PinnedPathIntegralSamples:
    """Simulate value-only suffixes from an adapted reference control.

    The running oracle is evaluated only at stochastic arrivals, matching
    ``sum_{m=1}^{N-1} dt * ell_m(X_m)``.  The fixed endpoint is not queried.
    The complete path simulation is not "simulation free."
    """
    config = config or PathIntegralConfig()
    start, endpoint = _normalize_states(start, endpoint)
    times, dt, time_valid = _uniform_time_data(times, start.dtype)
    batch_size, dim = start.shape
    sigma = _constant_diffusion_matrix(diffusion, dim, start.dtype)
    diffusion_valid = _diffusion_valid(sigma, config.diffusion_rcond)
    stochastic_steps = times.shape[0] - 2
    base_innovations = jax.random.normal(
        key,
        (batch_size, config.num_samples, stochastic_steps, dim),
        dtype=start.dtype,
    )
    innovations = (
        jnp.concatenate([base_innovations, -base_innovations], axis=1)
        if config.antithetic
        else base_innovations
    )
    trajectory_count = innovations.shape[1]
    flat_count = batch_size * trajectory_count
    flat_innovations = innovations.reshape((flat_count, stochastic_steps, dim))
    scan_innovations = jnp.swapaxes(flat_innovations, 0, 1)
    flat_endpoint = jnp.broadcast_to(
        endpoint[:, None, :], (batch_size, trajectory_count, dim)
    ).reshape((flat_count, dim))
    flat_start = jnp.broadcast_to(start[:, None, :], (batch_size, trajectory_count, dim)).reshape(
        (flat_count, dim)
    )
    terminal_time = times[-1]

    def zero_control(state: Array, time: Scalar, context: Array) -> Array:
        del time, context
        return jnp.zeros_like(state)

    active_control = reference_control or zero_control

    def step(state: Array, inputs: tuple[Array, Array, Array]):
        time, next_time, innovation = inputs
        rho = (terminal_time - next_time) / (terminal_time - time)
        gamma = jnp.sqrt(dt * rho) * sigma
        control = _batch_control_values(active_control, state, time, flat_endpoint)
        mean = rho * state + (1.0 - rho) * flat_endpoint
        next_state = mean + (jnp.sqrt(dt) * control + innovation) @ gamma.T
        cost_time = jnp.broadcast_to(next_time, (flat_count,))
        cost = _batch_running_values(running_cost, next_state, cost_time, flat_endpoint)
        return next_state, (next_state, control, jax.lax.stop_gradient(cost))

    _, (interior_tm, controls_tm, running_tm) = jax.lax.scan(
        step,
        flat_start,
        (
            times[:stochastic_steps],
            times[1 : stochastic_steps + 1],
            scan_innovations,
        ),
    )
    interior = jnp.swapaxes(interior_tm, 0, 1).reshape(
        (batch_size, trajectory_count, stochastic_steps, dim)
    )
    controls = jnp.swapaxes(controls_tm, 0, 1).reshape(
        (batch_size, trajectory_count, stochastic_steps, dim)
    )
    running = jnp.swapaxes(running_tm, 0, 1).reshape(
        (batch_size, trajectory_count, stochastic_steps)
    )
    start_states = jnp.broadcast_to(start[:, None, None, :], (batch_size, trajectory_count, 1, dim))
    endpoint_states = jnp.broadcast_to(
        endpoint[:, None, None, :], (batch_size, trajectory_count, 1, dim)
    )
    states = jnp.concatenate([start_states, interior, endpoint_states], axis=2)
    row_finite = (
        jnp.all(jnp.isfinite(states), axis=(1, 2, 3))
        & jnp.all(jnp.isfinite(innovations), axis=(1, 2, 3))
        & jnp.all(jnp.isfinite(controls), axis=(1, 2, 3))
        & jnp.all(jnp.isfinite(running), axis=(1, 2))
        & time_valid
        & diffusion_valid
    )
    physical_query_count = batch_size * trajectory_count * stochastic_steps
    return PinnedPathIntegralSamples(
        states=states,
        innovations=innovations,
        reference_controls=controls,
        running_values=jax.lax.stop_gradient(running),
        times=times,
        start=start,
        endpoint=endpoint,
        finite=row_finite,
        physical_query_count=jnp.asarray(physical_query_count, dtype=jnp.int32),
        trajectory_count=jnp.asarray(trajectory_count, dtype=jnp.int32),
        independent_base_draw_count=jnp.asarray(config.num_samples, dtype=jnp.int32),
        antithetic=jnp.asarray(config.antithetic),
    )


def estimate_path_integral_control_from_samples(
    innovations: Array,
    reference_controls: Array,
    running_values: Array,
    dt: Array | Scalar,
    *,
    minimum_ess_fraction: float = 0.01,
    initial_control_tolerance: float = 1e-5,
    physical_query_count: Array | int | None = None,
    independent_base_draw_count: Array | int | None = None,
    external_finite: Array | None = None,
) -> PathIntegralControlEstimate:
    """Assemble the self-normalized tilted-law moment from sampled suffixes.

    This pure function is the reference algebraic kernel.  Cost values are
    stopped; differentiating the output cannot differentiate through a hard
    oracle.  All trajectories for a batch row must start from the same
    conditioning state, so their first reference controls must agree within
    ``initial_control_tolerance``.  A mismatch fails the row closed.  The
    returned moment is population-exact for the unrestricted exponentially
    tilted path law, not generally optimal in the finite fixed-covariance
    Gaussian mean-shift control class.
    """
    if not 0.0 < minimum_ess_fraction <= 1.0:
        raise ValueError("minimum_ess_fraction must lie in (0, 1]")
    if initial_control_tolerance < 0.0 or not np.isfinite(initial_control_tolerance):
        raise ValueError("initial_control_tolerance must be finite and nonnegative")
    innovations, reference_controls, running_values = _validate_sample_arrays(
        innovations, reference_controls, running_values
    )
    batch_size, trajectory_count, stochastic_steps, dim = innovations.shape
    dt = jnp.asarray(dt, dtype=innovations.dtype)
    if dt.ndim != 0:
        raise ValueError("dt must be scalar")
    dt_valid = jnp.isfinite(dt) & (dt > 0.0)
    _require_concrete(dt_valid, "dt must be finite and positive")
    safe_dt = jnp.where(dt_valid, dt, jnp.asarray(1.0, dtype=innovations.dtype))
    stopped_cost = jax.lax.stop_gradient(running_values)
    stopped_controls = jax.lax.stop_gradient(reference_controls)
    stopped_innovations = jax.lax.stop_gradient(innovations)

    running_action = safe_dt * jnp.sum(stopped_cost, axis=-1)
    stochastic_integral = jnp.sqrt(safe_dt) * jnp.sum(
        stopped_controls * stopped_innovations,
        axis=(-1, -2),
    )
    control_energy = 0.5 * safe_dt * jnp.sum(stopped_controls**2, axis=(-1, -2))
    log_weights = -running_action - stochastic_integral - control_energy

    reference = stopped_controls[:, 0, 0, :]
    first_controls = stopped_controls[:, :, 0, :]
    control_scale = jnp.maximum(jnp.max(jnp.abs(reference), axis=-1), 1.0)
    initial_consistent = (
        jnp.max(jnp.abs(first_controls - reference[:, None, :]), axis=(1, 2))
        <= jnp.asarray(initial_control_tolerance, dtype=innovations.dtype) * control_scale
    )
    finite = (
        jnp.all(jnp.isfinite(stopped_innovations), axis=(1, 2, 3))
        & jnp.all(jnp.isfinite(stopped_controls), axis=(1, 2, 3))
        & jnp.all(jnp.isfinite(stopped_cost), axis=(1, 2))
        & jnp.all(jnp.isfinite(log_weights), axis=1)
        & initial_consistent
        & dt_valid
    )
    if external_finite is not None:
        external_finite = jnp.asarray(external_finite, dtype=bool)
        if external_finite.shape != (batch_size,):
            raise ValueError("external_finite must have shape [B]")
        finite = finite & external_finite

    # Invalid rows are sanitized solely to keep JIT outputs finite.  Their
    # ``finite`` and ``usable`` flags remain false.
    safe_log_weights = jnp.where(finite[:, None], log_weights, 0.0)
    maximum = jnp.max(safe_log_weights, axis=1, keepdims=True)
    shifted = jnp.exp(safe_log_weights - maximum)
    denominator = jnp.sum(shifted, axis=1, keepdims=True)
    normalized = shifted / denominator
    first_noise_mean = jnp.sum(normalized[:, :, None] * stopped_innovations[:, :, 0, :], axis=1)
    correction = first_noise_mean / jnp.sqrt(safe_dt)
    raw_target = reference + correction
    log_desirability = (
        maximum[:, 0]
        + jnp.log(denominator[:, 0])
        - jnp.log(jnp.asarray(trajectory_count, dtype=innovations.dtype))
    )
    ess = 1.0 / jnp.sum(normalized**2, axis=1)
    ess_fraction = ess / jnp.asarray(trajectory_count, dtype=innovations.dtype)
    maximum_normalized_weight = jnp.max(normalized, axis=1)
    output_finite = (
        jnp.all(jnp.isfinite(raw_target), axis=-1)
        & jnp.isfinite(log_desirability)
        & jnp.isfinite(ess)
        & jnp.isfinite(maximum_normalized_weight)
    )
    finite = finite & output_finite
    degenerate = (~finite) | (
        ess_fraction < jnp.asarray(minimum_ess_fraction, dtype=innovations.dtype)
    )
    usable = finite & ~degenerate
    safe_reference = jnp.where(jnp.isfinite(reference), reference, 0.0)
    control_target = jnp.where(usable[:, None], raw_target, safe_reference)
    raw_target = jnp.where(finite[:, None], raw_target, 0.0)
    correction = jnp.where(finite[:, None], correction, 0.0)
    log_desirability = jnp.where(finite, log_desirability, 0.0)
    ess = jnp.where(finite, ess, 0.0)
    ess_fraction = jnp.where(finite, ess_fraction, 0.0)
    maximum_normalized_weight = jnp.where(finite, maximum_normalized_weight, 0.0)

    if physical_query_count is None:
        physical_query_count = batch_size * trajectory_count * stochastic_steps
    if independent_base_draw_count is None:
        independent_base_draw_count = trajectory_count
    return PathIntegralControlEstimate(
        control_target=jax.lax.stop_gradient(control_target),
        raw_control_target=jax.lax.stop_gradient(raw_target),
        reference_control=jax.lax.stop_gradient(safe_reference),
        correction=jax.lax.stop_gradient(correction),
        log_desirability=jax.lax.stop_gradient(log_desirability),
        effective_sample_size=jax.lax.stop_gradient(ess),
        ess_fraction=jax.lax.stop_gradient(ess_fraction),
        maximum_normalized_weight=jax.lax.stop_gradient(maximum_normalized_weight),
        finite=finite,
        degenerate=degenerate,
        usable=usable,
        physical_query_count=jnp.asarray(physical_query_count, dtype=jnp.int32),
        trajectory_count=jnp.asarray(trajectory_count, dtype=jnp.int32),
        independent_base_draw_count=jnp.asarray(independent_base_draw_count, dtype=jnp.int32),
        exact_tilted_path_moment_in_population=jnp.asarray(True),
        exact_finite_gaussian_shift_optimum=jnp.asarray(False),
        # Deprecated compatibility alias.  It means exact *tilted-law moment*,
        # not exact finite Gaussian mean-shift optimal control.
        exact_in_population=jnp.asarray(True),
        finite_sample_ratio_biased=jnp.asarray(True),
    )


def estimate_pinned_path_integral_control(
    key: PRNGKey,
    start: Array,
    endpoint: Array,
    times: Array,
    diffusion: Array | Scalar,
    running_cost: ValueOnlyRunningCost,
    reference_control: NoiseControlFn | None = None,
    *,
    config: PathIntegralConfig | None = None,
) -> tuple[PinnedPathIntegralSamples, PathIntegralControlEstimate]:
    """Simulate pinned suffixes and estimate the tilted-law first-noise moment."""
    config = config or PathIntegralConfig()
    samples = simulate_pinned_reference_suffix(
        key,
        start,
        endpoint,
        times,
        diffusion,
        running_cost,
        reference_control,
        config=config,
    )
    estimate = estimate_path_integral_control_from_samples(
        samples.innovations,
        samples.reference_controls,
        samples.running_values,
        samples.times[1] - samples.times[0],
        minimum_ess_fraction=config.minimum_ess_fraction,
        initial_control_tolerance=config.initial_control_tolerance,
        physical_query_count=samples.physical_query_count,
        independent_base_draw_count=samples.independent_base_draw_count,
        external_finite=samples.finite,
    )
    return samples, estimate


def estimate_static_feynman_kac_kernel(
    start: Array,
    endpoint: Array,
    duration: Array | Scalar,
    diffusion: Array | Scalar,
    path_estimate: PathIntegralControlEstimate,
    *,
    diffusion_rcond: float = 1e-8,
) -> FeynmanKacKernelEstimate:
    """Attach the Brownian endpoint density to a pinned desirability estimate.

    The reference is driftless Brownian motion with constant square diffusion
    ``Sigma`` and covariance ``duration * Sigma @ Sigma.T``.  This routine is
    therefore not valid for state-dependent/rank-deficient diffusion or an
    unaccounted reference drift.
    """
    if isinstance(diffusion_rcond, bool) or not isinstance(
        diffusion_rcond,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("diffusion_rcond must be a real scalar")
    if diffusion_rcond <= 0.0 or not np.isfinite(diffusion_rcond):
        raise ValueError("diffusion_rcond must be finite and positive")
    start, endpoint = _normalize_states(start, endpoint)
    batch_size, dim = start.shape
    if path_estimate.log_desirability.shape != (batch_size,):
        raise ValueError("path_estimate batch size must match start and endpoint")
    sigma = _constant_diffusion_matrix(diffusion, dim, start.dtype)
    diffusion_valid = _diffusion_valid(sigma, diffusion_rcond)
    duration = jnp.asarray(duration, dtype=start.dtype)
    if duration.ndim != 0:
        raise ValueError("duration must be scalar")
    duration_valid = jnp.isfinite(duration) & (duration > 0.0)
    _require_concrete(duration_valid, "duration must be finite and positive")
    safe_duration = jnp.where(duration_valid, duration, 1.0)
    covariance = safe_duration * (sigma @ sigma.T)
    sign, log_determinant = jnp.linalg.slogdet(covariance)
    displacement = endpoint - start
    quadratic = jnp.sum(displacement * jnp.linalg.solve(covariance, displacement.T).T, axis=-1)
    log_density = -0.5 * (
        jnp.asarray(dim, dtype=start.dtype) * jnp.log(2.0 * jnp.pi) + log_determinant + quadratic
    )
    log_kernel = log_density + path_estimate.log_desirability
    normal_kernel = jnp.exp(log_kernel)
    log_domain_finite = (
        path_estimate.finite
        & diffusion_valid
        & duration_valid
        & (sign > 0.0)
        & jnp.isfinite(log_determinant)
        & jnp.all(jnp.isfinite(displacement), axis=-1)
        & jnp.isfinite(log_density)
        & jnp.isfinite(log_kernel)
    )
    normal_scale_representable = (
        log_domain_finite & jnp.isfinite(normal_kernel) & (normal_kernel > 0.0)
    )
    usable = log_domain_finite & ~path_estimate.degenerate
    return FeynmanKacKernelEstimate(
        kernel=jnp.where(normal_scale_representable, normal_kernel, 0.0),
        log_kernel=jnp.where(log_domain_finite, log_kernel, 0.0),
        log_reference_endpoint_density=jnp.where(log_domain_finite, log_density, 0.0),
        log_bridge_desirability=jnp.where(log_domain_finite, path_estimate.log_desirability, 0.0),
        effective_sample_size=path_estimate.effective_sample_size,
        ess_fraction=path_estimate.ess_fraction,
        log_domain_finite=log_domain_finite,
        normal_scale_representable=normal_scale_representable,
        # Compatibility alias: validity is now defined in log space.
        finite=log_domain_finite,
        degenerate=(~log_domain_finite) | path_estimate.degenerate,
        usable=usable,
        unbiased_in_exact_arithmetic=jnp.asarray(True),
        finite_sample_log_biased=jnp.asarray(True),
    )


def scale_static_feynman_kac_kernel(
    log_kernel: Array,
    source_weights: Array | None = None,
    target_weights: Array | None = None,
    *,
    config: StaticKernelSinkhornConfig | None = None,
) -> StaticKernelSinkhornResult:
    """Scale an empirical positive Feynman--Kac endpoint kernel.

    Given ``K_ij = exp(log_kernel_ij)`` and positive discrete endpoint masses
    ``a,b``, this computes the classical Sinkhorn form

    ``coupling = diag(u) K diag(v)``.

    The calculation is a required static kill baseline for the global MAM
    loop.  It uses no cost derivatives, but it does require an estimated
    kernel for every retained endpoint pair.  Kernel Monte Carlo error,
    finite-iteration marginal error, and rare-event degeneration remain
    separate errors.
    """
    config = config or StaticKernelSinkhornConfig()
    log_kernel = jnp.asarray(log_kernel)
    if log_kernel.ndim != 2 or min(log_kernel.shape) < 1:
        raise ValueError("log_kernel must have shape [num_source,num_target]")
    if not jnp.issubdtype(log_kernel.dtype, jnp.inexact):
        log_kernel = log_kernel.astype(jnp.float32)
    num_source, num_target = log_kernel.shape

    def normalize_mass(value: Array | None, size: int, name: str) -> tuple[Array, Array]:
        if value is None:
            return (
                jnp.full((size,), 1.0 / size, dtype=log_kernel.dtype),
                jnp.asarray(True),
            )
        mass = jnp.asarray(value, dtype=log_kernel.dtype)
        if mass.shape != (size,):
            raise ValueError(f"{name} must have shape {(size,)}")
        valid = jnp.all(jnp.isfinite(mass)) & jnp.all(mass > 0.0) & (jnp.sum(mass) > 0.0)
        _require_concrete(valid, f"{name} must be finite and strictly positive")
        safe = jnp.where(valid, mass, jnp.ones_like(mass))
        return safe / jnp.sum(safe), valid

    source, source_valid = normalize_mass(source_weights, num_source, "source_weights")
    target, target_valid = normalize_mass(target_weights, num_target, "target_weights")
    input_valid = (
        jnp.all(jnp.isfinite(log_kernel))
        & source_valid
        & target_valid
        & jnp.all(jnp.isfinite(source))
        & jnp.all(jnp.isfinite(target))
        & jnp.all(source > 0.0)
        & jnp.all(target > 0.0)
    )
    _require_concrete(input_valid, "static kernel and endpoint masses must be finite and positive")
    safe_log_kernel = jnp.where(input_valid, log_kernel, jnp.zeros_like(log_kernel))
    log_source = jnp.log(source)
    log_target = jnp.log(target)

    def iteration(carry: tuple[Array, Array], unused: None) -> tuple[tuple[Array, Array], None]:
        del unused
        _previous_log_u, log_v = carry
        log_u = log_source - logsumexp(safe_log_kernel + log_v[None, :], axis=1)
        log_v = log_target - logsumexp(safe_log_kernel + log_u[:, None], axis=0)
        return (log_u, log_v), None

    initial = (
        jnp.zeros((num_source,), dtype=log_kernel.dtype),
        jnp.zeros((num_target,), dtype=log_kernel.dtype),
    )
    (log_u, log_v), _ = jax.lax.scan(iteration, initial, xs=None, length=config.iterations)
    log_coupling = log_u[:, None] + safe_log_kernel + log_v[None, :]
    coupling = jnp.exp(log_coupling)
    source_marginal = jnp.sum(coupling, axis=1)
    target_marginal = jnp.sum(coupling, axis=0)
    marginal_error = jnp.maximum(
        jnp.max(jnp.abs(source_marginal - source)),
        jnp.max(jnp.abs(target_marginal - target)),
    )
    output_valid = (
        input_valid
        & jnp.all(jnp.isfinite(coupling))
        & jnp.all(coupling >= 0.0)
        & jnp.isfinite(marginal_error)
    )
    converged = output_valid & (
        marginal_error <= jnp.asarray(config.tolerance, dtype=log_kernel.dtype)
    )
    coupling = jnp.where(output_valid, coupling, jnp.zeros_like(coupling))
    source_marginal = jnp.where(output_valid, source_marginal, jnp.zeros_like(source_marginal))
    target_marginal = jnp.where(output_valid, target_marginal, jnp.zeros_like(target_marginal))
    marginal_error = jnp.where(
        output_valid,
        marginal_error,
        jnp.asarray(jnp.inf, dtype=log_kernel.dtype),
    )
    return StaticKernelSinkhornResult(
        coupling=jax.lax.stop_gradient(coupling),
        source_marginal=jax.lax.stop_gradient(source_marginal),
        target_marginal=jax.lax.stop_gradient(target_marginal),
        marginal_error=jax.lax.stop_gradient(marginal_error),
        finite=output_valid,
        converged=converged,
        usable=converged,
        iterations=jnp.asarray(config.iterations, dtype=jnp.int32),
    )


__all__ = [
    "FeynmanKacKernelEstimate",
    "PathIntegralConfig",
    "PathIntegralControlEstimate",
    "PinnedPathIntegralSamples",
    "StaticKernelSinkhornConfig",
    "StaticKernelSinkhornResult",
    "estimate_path_integral_control_from_samples",
    "estimate_pinned_path_integral_control",
    "estimate_static_feynman_kac_kernel",
    "scale_static_feynman_kac_kernel",
    "simulate_pinned_reference_suffix",
]
