"""Runtime process abstraction for solved Schrödinger Bridges.

This module defines :class:`BridgeProcess`, a light-weight SDE/process object
that exposes the learned bridge coefficients and simulation routines. It sits
between a trained solver solution and downstream usage such as generative
sampling, visualization, and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp

from .core.problem import SBProblem
from .core.types import (
    Array,
    DriftFn,
    PRNGKey,
    RepresentationType,
    Scalar,
    SDECoefficients,
    SolverType,
    TimeGrid,
    TrajectoryBatch,
)
from .integrators import EulerMaruyama, Integrator

try:
    import diffrax
except ImportError:
    diffrax = None


def _apply_diffusion_covariance(
    sigma: Union[Scalar, Array],
    vector: Array,
) -> Array:
    """Apply diffusion covariance a = sigma sigma^T to a batch of vectors."""
    sigma_arr = jnp.asarray(sigma)
    vector = jnp.atleast_2d(vector)

    if sigma_arr.ndim <= 1:
        return (sigma_arr ** 2) * vector

    if sigma_arr.ndim == 2:
        cov = sigma_arr @ sigma_arr.T
        return vector @ cov.T

    if sigma_arr.ndim == 3:
        cov = sigma_arr @ jnp.swapaxes(sigma_arr, -1, -2)
        return jnp.einsum("bij,bj->bi", cov, vector)

    raise ValueError(
        f"Unsupported diffusion shape {sigma_arr.shape}; expected scalar, "
        "matrix, or batched matrix diffusion."
    )


def _drift_single_sample(
    drift_fn: DriftFn,
    y: Array,
    t: Scalar,
) -> Array:
    """Evaluate a batched drift function on a single state sample."""
    y_batch = jnp.atleast_2d(y)
    drift = jnp.asarray(drift_fn(y_batch, t))
    return drift[0] if drift.ndim > 1 else drift


def _diffusion_matrix_single_sample(
    diffusion_fn: Callable[[Array, Scalar], Union[Scalar, Array]],
    y: Array,
    t: Scalar,
) -> Array:
    """Convert a diffusion coefficient into a matrix for Diffrax."""
    sigma = jnp.asarray(diffusion_fn(jnp.atleast_2d(y), t))
    dim = y.shape[-1]
    eye = jnp.eye(dim, dtype=y.dtype)

    if sigma.ndim == 0:
        return sigma * eye

    if sigma.ndim == 1:
        if sigma.shape[0] == dim:
            return jnp.diag(sigma)
        if sigma.shape[0] == 1:
            return sigma.reshape(()) * eye
        raise ValueError(
            f"Unsupported 1D diffusion shape {sigma.shape}; expected length 1 or {dim}."
        )

    if sigma.ndim == 2:
        if sigma.shape[0] == 1 and sigma.shape[1] == dim:
            return jnp.diag(sigma[0])
        return sigma

    if sigma.ndim == 3 and sigma.shape[0] == 1:
        return sigma[0]

    raise ValueError(
        f"Unsupported diffusion shape {sigma.shape}; expected scalar, vector, or matrix."
    )


@dataclass
class BridgeProcess:
    """Runtime stochastic process induced by a solved Schrödinger Bridge."""

    problem: SBProblem
    solver_type: SolverType
    representation_type: RepresentationType
    params: Any
    forward_drift_fn: DriftFn
    backward_drift_fn: Optional[DriftFn] = None
    score_fn: Optional[Callable[[Array, Scalar], Array]] = None
    integrator: Optional[Integrator] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    backend: str = "native"
    initial_sampler: Optional[Callable[[PRNGKey, int], Array]] = None
    terminal_sampler: Optional[Callable[[PRNGKey, int], Array]] = None

    def __post_init__(self):
        if self.initial_sampler is None:
            self.initial_sampler = self.problem.sample_source
        if self.terminal_sampler is None:
            self.terminal_sampler = self.problem.sample_target

    @property
    def dim(self) -> int:
        """State dimension."""
        return self.problem.dim

    @property
    def time_grid(self) -> TimeGrid:
        """Nominal time grid used for simulation."""
        return self.problem.time_grid

    @property
    def times(self) -> Array:
        """Nominal save times."""
        return self.time_grid.times

    def has_reverse(self) -> bool:
        """Whether a backward drift is available."""
        return self.backward_drift_fn is not None

    def has_score(self) -> bool:
        """Whether a score function is available."""
        return self.score_fn is not None

    def drift(
        self,
        x: Array,
        t: Scalar,
        direction: str = "forward",
    ) -> Array:
        """Evaluate the learned drift field."""
        if direction == "forward":
            return self.forward_drift_fn(x, t)
        if direction == "backward":
            if self.backward_drift_fn is None:
                raise ValueError("Backward drift not available for this process.")
            return self.backward_drift_fn(x, t)
        raise ValueError(f"Unknown direction '{direction}'. Expected 'forward' or 'backward'.")

    def diffusion(self, x: Array, t: Scalar) -> Union[Scalar, Array]:
        """Evaluate the reference diffusion coefficient."""
        return self.problem.sigma(x, t)

    def coefficients(self, direction: str = "forward") -> SDECoefficients:
        """Return SDE coefficients for the selected direction."""
        return SDECoefficients(
            drift=lambda x, t: self.drift(x, t, direction=direction),
            diffusion=lambda t: self.problem.reference.diffusion(None, t),
            is_diffusion_scalar=self.problem.reference.is_diffusion_scalar,
        )

    def score(self, x: Array, t: Scalar) -> Array:
        """Evaluate the score function when available."""
        if self.score_fn is None:
            raise ValueError("Score function not available for this process.")
        return self.score_fn(x, t)

    def probability_flow_drift(self, x: Array, t: Scalar) -> Array:
        """Deterministic drift sharing the bridge marginals when a score is known."""
        if self.score_fn is None:
            raise ValueError("Probability flow drift requires a score function.")
        drift = self.forward_drift_fn(x, t)
        sigma = self.diffusion(x, t)
        correction = _apply_diffusion_covariance(sigma, self.score(x, t))
        return drift - 0.5 * correction

    def reverse(self) -> "BridgeProcess":
        """Return the reverse-time process when available."""
        if self.backward_drift_fn is None:
            raise ValueError("Backward drift not available for this process.")

        reverse_metadata = dict(self.metadata)
        reverse_metadata["reversed"] = not reverse_metadata.get("reversed", False)

        return BridgeProcess(
            problem=self.problem,
            solver_type=self.solver_type,
            representation_type=self.representation_type,
            params=self.params,
            forward_drift_fn=self.backward_drift_fn,
            backward_drift_fn=self.forward_drift_fn,
            score_fn=self.score_fn,
            integrator=self.integrator,
            metadata=reverse_metadata,
            backend=self.backend,
            initial_sampler=self.terminal_sampler,
            terminal_sampler=self.initial_sampler,
        )

    def _get_integrator(self, integrator: Optional[Integrator]) -> Integrator:
        if integrator is not None:
            return integrator
        if self.integrator is not None:
            return self.integrator
        return EulerMaruyama()

    def _check_backend(self):
        if self.backend not in {"native", "diffrax"}:
            raise NotImplementedError(
                f"Backend '{self.backend}' is not implemented yet. "
                "Use backend='native' or backend='diffrax'."
            )

    def _require_diffrax(self):
        if diffrax is None:
            raise ImportError(
                "Diffrax backend requested, but `diffrax` is not installed. "
                "Install it with `pip install diffrax` or `pip install -e .[diffrax]`."
            )

    def _resolve_initial_state(
        self,
        key: PRNGKey,
        num_samples: int,
        initial_state: Optional[Array],
        direction: str,
    ) -> Array:
        if initial_state is not None:
            return jnp.atleast_2d(initial_state)

        if direction == "forward":
            return self.initial_sampler(key, num_samples)
        if direction == "backward":
            return self.terminal_sampler(key, num_samples)
        raise ValueError(f"Unknown direction '{direction}'. Expected 'forward' or 'backward'.")

    def _make_subgrid(
        self,
        t0: Optional[float],
        t1: Optional[float],
    ) -> TimeGrid:
        base = self.time_grid
        start = base.t0 if t0 is None else float(t0)
        end = base.t1 if t1 is None else float(t1)

        if abs(start - base.t0) <= 1e-12 and abs(end - base.t1) <= 1e-12:
            return base

        total_span = base.t1 - base.t0
        span = end - start
        steps = max(1, int(round(base.num_steps * span / total_span)))
        return TimeGrid(t0=start, t1=end, num_steps=steps)

    def _diffrax_sde_solver(self):
        """Construct the Diffrax solver used for stochastic sampling."""
        self._require_diffrax()
        solver_name = str(self.metadata.get("diffrax_sde_solver", "euler")).lower()

        if solver_name == "euler":
            return diffrax.Euler()

        raise ValueError(
            f"Unsupported Diffrax SDE solver '{solver_name}'. "
            "Currently supported: 'euler'."
        )

    def _diffrax_ode_solver(self):
        """Construct the Diffrax solver used for probability-flow ODEs."""
        self._require_diffrax()
        solver_name = str(self.metadata.get("diffrax_ode_solver", "tsit5")).lower()

        if solver_name == "heun":
            return diffrax.Heun()
        if solver_name == "tsit5":
            return diffrax.Tsit5()
        if solver_name == "dopri5":
            return diffrax.Dopri5()

        raise ValueError(
            f"Unsupported Diffrax ODE solver '{solver_name}'. "
            "Currently supported: 'heun', 'tsit5', 'dopri5'."
        )

    def _solve_diffrax_paths(
        self,
        key: PRNGKey,
        initial_state: Array,
        time_grid: TimeGrid,
        drift_fn: DriftFn,
        diffusion_fn: Callable[[Array, Scalar], Union[Scalar, Array]],
        return_full: bool,
        direction: str,
    ) -> Union[TrajectoryBatch, Array]:
        """Sample paths using Diffrax over the provided time grid."""
        self._require_diffrax()
        initial_state = jnp.atleast_2d(initial_state)
        batch_size = initial_state.shape[0]
        dim = initial_state.shape[1]

        solve_t0 = float(time_grid.t0) if direction == "forward" else float(time_grid.t1)
        solve_t1 = float(time_grid.t1) if direction == "forward" else float(time_grid.t0)
        dt0 = float(time_grid.dt) if direction == "forward" else -float(time_grid.dt)

        save_times = time_grid.times if direction == "forward" else time_grid.times[::-1]
        brownian_t0 = min(float(time_grid.t0), float(time_grid.t1))
        brownian_t1 = max(float(time_grid.t0), float(time_grid.t1))
        brownian_tol = max(abs(dt0) / 2.0, 1e-4)
        max_steps = max(4096, 4 * time_grid.num_steps + 64)

        solver = self._diffrax_sde_solver()

        def solve_one(sample_key: PRNGKey, y0: Array) -> Array:
            brownian = diffrax.VirtualBrownianTree(
                t0=brownian_t0,
                t1=brownian_t1,
                tol=brownian_tol,
                shape=(dim,),
                key=sample_key,
            )

            drift_term = diffrax.ODETerm(
                lambda t, y, args: _drift_single_sample(drift_fn, y, t)
            )
            diffusion_term = diffrax.ControlTerm(
                lambda t, y, args: _diffusion_matrix_single_sample(diffusion_fn, y, t),
                brownian,
            )
            terms = diffrax.MultiTerm(drift_term, diffusion_term)

            saveat = (
                diffrax.SaveAt(ts=save_times)
                if return_full
                else diffrax.SaveAt(t1=True)
            )

            sol = diffrax.diffeqsolve(
                terms,
                solver,
                t0=solve_t0,
                t1=solve_t1,
                dt0=dt0,
                y0=y0,
                saveat=saveat,
                max_steps=max_steps,
            )

            ys = sol.ys
            if return_full and direction == "backward":
                ys = ys[::-1]
            return ys

        sample_keys = jax.random.split(key, batch_size)
        ys = jax.vmap(solve_one)(sample_keys, initial_state)

        if return_full:
            return TrajectoryBatch(paths=ys, times=time_grid.times)
        return ys

    def _solve_diffrax_flow(
        self,
        key: PRNGKey,
        initial_state: Array,
        time_grid: TimeGrid,
        return_full: bool,
    ) -> Union[TrajectoryBatch, Array]:
        """Sample the deterministic probability-flow ODE with Diffrax."""
        self._require_diffrax()
        initial_state = jnp.atleast_2d(initial_state)
        batch_size = initial_state.shape[0]
        dt0 = float(time_grid.dt)
        max_steps = max(4096, 4 * time_grid.num_steps + 64)
        solver = self._diffrax_ode_solver()

        def solve_one(sample_key: PRNGKey, y0: Array) -> Array:
            del sample_key
            terms = diffrax.ODETerm(
                lambda t, y, args: _drift_single_sample(self.probability_flow_drift, y, t)
            )
            saveat = (
                diffrax.SaveAt(ts=time_grid.times)
                if return_full
                else diffrax.SaveAt(t1=True)
            )

            sol = diffrax.diffeqsolve(
                terms,
                solver,
                t0=float(time_grid.t0),
                t1=float(time_grid.t1),
                dt0=dt0,
                y0=y0,
                saveat=saveat,
                max_steps=max_steps,
            )
            return sol.ys

        dummy_keys = jax.random.split(key, batch_size)
        ys = jax.vmap(solve_one)(dummy_keys, initial_state)

        if return_full:
            return TrajectoryBatch(paths=ys, times=time_grid.times)
        return ys

    def sample_paths(
        self,
        key: PRNGKey,
        num_samples: int,
        x0: Optional[Array] = None,
        direction: str = "forward",
        return_full: bool = True,
        integrator: Optional[Integrator] = None,
    ) -> Union[TrajectoryBatch, Array]:
        """Sample paths from the bridge process."""
        self._check_backend()
        k1, k2 = jax.random.split(key)
        initial_state = self._resolve_initial_state(k1, num_samples, x0, direction)

        if self.backend == "diffrax":
            drift_fn = self.forward_drift_fn
            if direction == "backward":
                if self.backward_drift_fn is None:
                    raise ValueError("Backward drift not available for this process.")
                drift_fn = self.backward_drift_fn
            return self._solve_diffrax_paths(
                k2,
                initial_state,
                self.time_grid,
                drift_fn,
                self.diffusion,
                return_full,
                direction,
            )

        stepper = self._get_integrator(integrator)

        if direction == "forward":
            return stepper.integrate(
                k2,
                initial_state,
                self.time_grid,
                self.forward_drift_fn,
                self.diffusion,
                return_full,
            )

        if direction == "backward":
            if self.backward_drift_fn is None:
                raise ValueError("Backward drift not available for this process.")
            return stepper.integrate_backward(
                k2,
                initial_state,
                self.time_grid,
                self.backward_drift_fn,
                self.diffusion,
                return_full,
            )

        raise ValueError(f"Unknown direction '{direction}'. Expected 'forward' or 'backward'.")

    def rollout_from(
        self,
        key: PRNGKey,
        x0: Array,
        t0: Optional[float] = None,
        t1: Optional[float] = None,
        direction: str = "forward",
        return_full: bool = True,
        integrator: Optional[Integrator] = None,
    ) -> Union[TrajectoryBatch, Array]:
        """Roll out the process from a provided state over a sub-interval."""
        self._check_backend()
        time_grid = self._make_subgrid(t0, t1)
        initial_state = jnp.atleast_2d(x0)

        if self.backend == "diffrax":
            drift_fn = self.forward_drift_fn
            if direction == "backward":
                if self.backward_drift_fn is None:
                    raise ValueError("Backward drift not available for this process.")
                drift_fn = self.backward_drift_fn
            return self._solve_diffrax_paths(
                key,
                initial_state,
                time_grid,
                drift_fn,
                self.diffusion,
                return_full,
                direction,
            )

        stepper = self._get_integrator(integrator)

        if direction == "forward":
            return stepper.integrate(
                key,
                initial_state,
                time_grid,
                self.forward_drift_fn,
                self.diffusion,
                return_full,
            )

        if direction == "backward":
            if self.backward_drift_fn is None:
                raise ValueError("Backward drift not available for this process.")
            return stepper.integrate_backward(
                key,
                initial_state,
                time_grid,
                self.backward_drift_fn,
                self.diffusion,
                return_full,
            )

        raise ValueError(f"Unknown direction '{direction}'. Expected 'forward' or 'backward'.")

    def sample_endpoint(
        self,
        key: PRNGKey,
        num_samples: int,
        x0: Optional[Array] = None,
        direction: str = "forward",
        integrator: Optional[Integrator] = None,
    ) -> Array:
        """Sample the terminal point of a rollout."""
        return self.sample_paths(
            key,
            num_samples,
            x0=x0,
            direction=direction,
            return_full=False,
            integrator=integrator,
        )

    def sample_marginal(
        self,
        key: PRNGKey,
        t: Scalar,
        num_samples: int,
        x0: Optional[Array] = None,
        direction: str = "forward",
        integrator: Optional[Integrator] = None,
    ) -> Array:
        """Sample the process marginal at the closest saved time to ``t``."""
        batch = self.sample_paths(
            key,
            num_samples,
            x0=x0,
            direction=direction,
            return_full=True,
            integrator=integrator,
        )
        t_idx = jnp.argmin(jnp.abs(batch.times - t))
        return batch.at_time(int(t_idx))

    def sample_flow(
        self,
        key: PRNGKey,
        num_samples: int,
        x0: Optional[Array] = None,
        return_full: bool = True,
        integrator: Optional[Integrator] = None,
    ) -> Union[TrajectoryBatch, Array]:
        """Sample the deterministic probability-flow dynamics."""
        self._check_backend()
        if self.score_fn is None:
            raise ValueError("Probability flow sampling requires a score function.")

        k1, k2 = jax.random.split(key)
        initial_state = self._resolve_initial_state(k1, num_samples, x0, "forward")

        if self.backend == "diffrax":
            return self._solve_diffrax_flow(
                k2,
                initial_state,
                self.time_grid,
                return_full,
            )

        stepper = self._get_integrator(integrator)

        def zero_diffusion(x: Array, t: Scalar) -> Scalar:
            del x, t
            return 0.0

        return stepper.integrate(
            k2,
            initial_state,
            self.time_grid,
            self.probability_flow_drift,
            zero_diffusion,
            return_full,
        )


__all__ = ["BridgeProcess"]
