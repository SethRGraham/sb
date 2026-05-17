"""IPF Diffusion Schrodinger Bridge solver.

This module implements the IPF-style Diffusion Schrodinger Bridge (DSB)
procedure from De Bortoli et al. (NeurIPS 2021). It intentionally lives next
to, rather than replacing, ``score_based.py``:

- ``score_based.py`` trains one score network with denoising score matching.
- ``ipf_dsb.py`` trains two Markov transition mean maps with IPF.

The learned networks output the transition mean maps directly:

    F_k(x) approximates E[X_{k+1} | X_k = x]
    B_k(x) approximates E[X_{k-1} | X_k = x]

This avoids the common off-by-gamma bug where the paper's ``F=x+gamma f`` and
``B=x+gamma b`` maps are accidentally treated as raw drifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from ..core.problem import SBProblem
from ..core.types import (
    Array,
    DriftFn,
    Params,
    PRNGKey,
    RepresentationType,
    Scalar,
    SolverConfig,
    SolverResult,
    SolverType,
    TrainingConfig,
    TrajectoryBatch,
)
from ..network_factory import MLPFactory, NetworkFactory, sanity_check
from ..networks import adam_update, init_adam
from .base import SBSolver
from .score_based import _apply_diffusion


@dataclass
class IPFDSBConfig:
    """Configuration for the IPF Diffusion Schrodinger Bridge solver."""

    N: Optional[int] = None
    gamma: Optional[float] = None
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    time_embed_dim: int = 64
    learning_rate: float = 1e-4
    inner_steps: int = 500
    batch_size: int = 128
    ema_decay: float = 0.999
    n_ipf_iterations: int = 5
    cache_trajectories: bool = True
    cache_size: int = 10000
    warmstart_with_ot: bool = False
    ot_regularization: float = 0.1
    diffusion_matrix_is_covariance: bool = False
    use_ema: bool = True
    seed: int = 42
    network_factory: Optional[NetworkFactory] = None

    def __post_init__(self) -> None:
        if self.N is not None and self.N < 1:
            raise ValueError("N must be positive.")
        if self.gamma is not None and self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        if self.inner_steps < 1:
            raise ValueError("inner_steps must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.cache_size < 1:
            raise ValueError("cache_size must be positive.")
        if self.n_ipf_iterations < 1:
            raise ValueError("n_ipf_iterations must be positive.")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1).")


class IPFDSBSolver(SBSolver):
    """IPF solver using the DSB paper's forward/backward mean-map losses."""

    def __init__(
        self,
        problem: SBProblem,
        dsb_config: Optional[IPFDSBConfig] = None,
        solver_config: Optional[SolverConfig] = None,
        **kwargs: Any,
    ):
        if dsb_config is None and "ipf_dsb_config" in kwargs:
            dsb_config = kwargs.pop("ipf_dsb_config")
        self.dsb_config = dsb_config or IPFDSBConfig()
        if self.dsb_config.N is None:
            self.dsb_config.N = problem.time_grid.num_steps
        if self.dsb_config.gamma is None:
            total_time = problem.time_grid.t1 - problem.time_grid.t0
            self.dsb_config.gamma = total_time / self.dsb_config.N

        if solver_config is None and "config" in kwargs:
            solver_config = kwargs.pop("config")

        super().__init__(problem, config=solver_config)

        self._factory = self.dsb_config.network_factory or MLPFactory(
            hidden_dims=self.dsb_config.hidden_dims,
            time_embed_dim=self.dsb_config.time_embed_dim,
        )
        self._ema_params: Optional[Params] = None
        self._F_is_reference: bool = True

    @property
    def solver_type(self) -> SolverType:
        return SolverType.DSB

    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.DRIFT

    @property
    def N(self) -> int:
        return int(self.dsb_config.N)

    @property
    def gamma(self) -> float:
        return float(self.dsb_config.gamma)

    def init_params(self, key: PRNGKey) -> Params:
        """Initialize independent forward and backward mean-map networks."""
        k1, k2, k3 = jax.random.split(key, 3)
        params = {
            "F": self._factory.init(k1, self.problem.dim, self.problem.dim),
            "B": self._factory.init(k2, self.problem.dim, self.problem.dim),
        }
        sanity_check(self._factory, k3, self.problem.dim, self.problem.dim)
        return params

    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: Any,
        batch_size: int,
    ) -> Tuple[Params, Any, Dict[str, Scalar]]:
        """Single-step training is not the natural DSB unit.

        DSB alternates cached forward trajectories, backward map updates,
        cached backward trajectories, and forward map updates. Use ``train``.
        """
        raise NotImplementedError("IPFDSBSolver uses an alternating IPF train loop.")

    def train(
        self,
        key: PRNGKey,
        training_config: Optional[TrainingConfig] = None,
        callback: Optional[Any] = None,
    ) -> SolverResult:
        """Train the DSB solver with alternating IPF half-iterations."""
        train_config = training_config or TrainingConfig(
            batch_size=self.dsb_config.batch_size,
        )
        batch_size = (
            train_config.batch_size
            if training_config is not None
            else self.dsb_config.batch_size
        )
        cache_size = (
            self.dsb_config.cache_size
            if self.dsb_config.cache_trajectories
            else batch_size
        )

        key, init_key = jax.random.split(key)
        params = self.init_params(init_key)
        opt_F = init_adam(params["F"])
        opt_B = init_adam(params["B"])

        self._params = params
        self._ema_params = {"F": params["F"], "B": params["B"]}
        self._F_is_reference = True

        loss_history: List[float] = []
        checkpoint_paths: List[str] = []
        global_step = 0
        last_metrics: Dict[str, Any] = {}

        for ipf_iter in range(self.dsb_config.n_ipf_iterations):
            key, cache_key = jax.random.split(key)
            if self.dsb_config.warmstart_with_ot and ipf_iter == 0:
                forward_cache = self._build_ot_warmstart_cache(cache_key, cache_size)
            else:
                forward_cache = self._build_forward_cache(
                    cache_key,
                    cache_size,
                    params,
                    F_is_reference=self._F_is_reference,
                )

            for inner_step in range(self.dsb_config.inner_steps):
                key, batch_key = jax.random.split(key)
                batch = self._draw_batch(batch_key, forward_cache, batch_size)

                params["B"], opt_B, loss = self._backward_update_jit(
                    params["B"],
                    opt_B,
                    params["F"],
                    batch,
                    F_is_reference=self._F_is_reference,
                )
                if self.dsb_config.use_ema:
                    self._ema_params = {
                        "F": self._ema_params["F"],
                        "B": self._ema_update(self._ema_params["B"], params["B"]),
                    }
                else:
                    self._ema_params = params

                global_step += 1
                metrics = {
                    "loss": loss,
                    "ipf_iteration": ipf_iter,
                    "inner_step": inner_step,
                    "phase": "backward",
                }
                last_metrics = metrics
                loss_history.append(float(loss))
                self._params = params

                if callback is not None:
                    callback(global_step, metrics)
                self._log_step(global_step, metrics, train_config)
                checkpoint = self._maybe_save_checkpoint(
                    train_config,
                    step=global_step,
                    params=params,
                    opt_state={"F": opt_F, "B": opt_B},
                    loss_history=loss_history,
                    metrics=metrics,
                )
                if checkpoint is not None:
                    checkpoint_paths.append(checkpoint)

            key, cache_key = jax.random.split(key)
            backward_cache = self._build_backward_cache(cache_key, cache_size, params)

            for inner_step in range(self.dsb_config.inner_steps):
                key, batch_key = jax.random.split(key)
                batch = self._draw_batch(batch_key, backward_cache, batch_size)

                params["F"], opt_F, loss = self._forward_update_jit(
                    params["F"],
                    opt_F,
                    params["B"],
                    batch,
                )
                if self.dsb_config.use_ema:
                    self._ema_params = {
                        "F": self._ema_update(self._ema_params["F"], params["F"]),
                        "B": self._ema_params["B"],
                    }
                else:
                    self._ema_params = params

                global_step += 1
                metrics = {
                    "loss": loss,
                    "ipf_iteration": ipf_iter,
                    "inner_step": inner_step,
                    "phase": "forward",
                }
                last_metrics = metrics
                loss_history.append(float(loss))
                self._params = params
                self._F_is_reference = False

                if callback is not None:
                    callback(global_step, metrics)
                self._log_step(global_step, metrics, train_config)
                checkpoint = self._maybe_save_checkpoint(
                    train_config,
                    step=global_step,
                    params=params,
                    opt_state={"F": opt_F, "B": opt_B},
                    loss_history=loss_history,
                    metrics=metrics,
                )
                if checkpoint is not None:
                    checkpoint_paths.append(checkpoint)

            self._F_is_reference = False

        self._params = params
        if not self.dsb_config.use_ema:
            self._ema_params = params
        self._is_trained = True

        diagnostics = self._run_diagnostics(key, params)
        metadata = {
            "converged": True,
            "final_step": global_step,
            "solver_type": self.solver_type.name,
            "n_ipf_iterations": self.dsb_config.n_ipf_iterations,
            "inner_steps": self.dsb_config.inner_steps,
            "N": self.N,
            "gamma": self.gamma,
        }
        final_checkpoint = self._maybe_save_checkpoint(
            train_config,
            step=global_step,
            params=params,
            opt_state={"F": opt_F, "B": opt_B},
            loss_history=loss_history,
            metrics=last_metrics,
            final=True,
            metadata=metadata,
        )
        if final_checkpoint is not None:
            checkpoint_paths.append(final_checkpoint)
            metadata["checkpoint_path"] = final_checkpoint
        if checkpoint_paths:
            metadata["checkpoint_paths"] = checkpoint_paths

        return SolverResult(
            params=params,
            loss_history=jnp.asarray(loss_history),
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def _log_step(
        self,
        global_step: int,
        metrics: Dict[str, Any],
        train_config: TrainingConfig,
    ) -> None:
        if self.config.verbose < 1:
            return
        if global_step % train_config.eval_every != 0:
            return
        print(
            f"Step {global_step}: {metrics['phase']} loss = "
            f"{float(metrics['loss']):.6f}"
        )

    def _ema_update(self, old: Params, new: Params) -> Params:
        decay = self.dsb_config.ema_decay
        return jax.tree_util.tree_map(
            lambda old_leaf, new_leaf: decay * old_leaf + (1.0 - decay) * new_leaf,
            old,
            new,
        )

    def _active_params(
        self,
        params: Optional[Params] = None,
        *,
        use_ema: bool = True,
    ) -> Params:
        if params is not None:
            return params
        if not self._is_trained or self._params is None:
            raise ValueError(
                "Solver not trained. Call train() first or provide params."
            )
        if use_ema and self.dsb_config.use_ema and self._ema_params is not None:
            return self._ema_params
        return self._params

    def get_trained_params(self, use_ema: bool = True) -> Params:
        """Return trained parameters, optionally using EMA weights."""
        return self._active_params(None, use_ema=use_ema)

    def _net_time(self, k_index: Scalar) -> Array:
        return jnp.asarray(k_index, dtype=jnp.float32) / float(max(self.N, 1))

    def _physical_time(self, k_index: Scalar) -> Array:
        return self.problem.time_grid.t0 + self.gamma * jnp.asarray(
            k_index,
            dtype=jnp.float32,
        )

    def _index_batch(self, k_index: Scalar, batch_size: int) -> Array:
        k = jnp.asarray(k_index, dtype=jnp.float32)
        if k.ndim == 0:
            return jnp.full((batch_size,), k)
        if k.shape[0] == 1:
            return jnp.full((batch_size,), k.reshape(()))
        return k

    def _network_map(self, net_params: Params, x: Array, k_index: Scalar) -> Array:
        x = jnp.atleast_2d(x)
        k = self._index_batch(k_index, x.shape[0])
        return self._factory.forward(net_params, x, self._net_time(k))

    def _reference_forward_map(self, x: Array, k_index: Scalar) -> Array:
        x = jnp.atleast_2d(x)
        k = self._index_batch(k_index, x.shape[0])
        t = self._physical_time(k)
        return x + self.gamma * self.problem.reference.drift(x, t)

    def _F_map(
        self,
        F_params: Params,
        x: Array,
        k_index: Scalar,
        *,
        F_is_reference: bool = False,
    ) -> Array:
        if F_is_reference:
            return self._reference_forward_map(x, k_index)
        return self._network_map(F_params, x, k_index)

    def _B_map(self, B_params: Params, x: Array, k_index: Scalar) -> Array:
        return self._network_map(B_params, x, k_index)

    def _step_noise(self, key: PRNGKey, x: Array, k_index: Scalar) -> Array:
        k = self._index_batch(k_index, x.shape[0])
        t = self._physical_time(k)
        sigma = self.problem.reference.diffusion(x, t)
        z = jax.random.normal(key, x.shape)
        return jnp.sqrt(self.gamma) * _apply_diffusion(
            sigma,
            z,
            is_scalar_diffusion=self.problem.reference.is_diffusion_scalar,
            matrix_is_covariance=self.dsb_config.diffusion_matrix_is_covariance,
        )

    def _map_over_steps(
        self,
        map_fn: Any,
        x_steps: Array,
        k_indices: Array,
    ) -> Array:
        batch_size, num_steps, dim = x_steps.shape
        x_flat = x_steps.reshape((-1, dim))
        k_flat = jnp.broadcast_to(
            k_indices[None, :],
            (batch_size, num_steps),
        ).reshape((-1,))
        mapped = map_fn(x_flat, k_flat)
        return mapped.reshape((batch_size, num_steps, dim))

    def _backward_loss(
        self,
        B_params: Params,
        F_params: Params,
        trajectories: Array,
        *,
        F_is_reference: bool,
    ) -> Array:
        x_k = trajectories[:, :-1, :]
        x_k1 = trajectories[:, 1:, :]
        k_indices = jnp.arange(self.N, dtype=jnp.float32)

        F_xk = self._map_over_steps(
            lambda x, k: self._F_map(
                F_params,
                x,
                k,
                F_is_reference=F_is_reference,
            ),
            x_k,
            k_indices,
        )
        F_xk1 = self._map_over_steps(
            lambda x, k: self._F_map(
                F_params,
                x,
                k,
                F_is_reference=F_is_reference,
            ),
            x_k1,
            k_indices,
        )
        target = jax.lax.stop_gradient(x_k1 + F_xk - F_xk1)
        pred = self._map_over_steps(
            lambda x, k: self._B_map(B_params, x, k),
            x_k1,
            k_indices + 1.0,
        )
        return jnp.mean((pred - target) ** 2)

    def _forward_loss(
        self,
        F_params: Params,
        B_params: Params,
        trajectories: Array,
    ) -> Array:
        x_k = trajectories[:, :-1, :]
        x_k1 = trajectories[:, 1:, :]
        k_indices = jnp.arange(self.N, dtype=jnp.float32)

        B_xk1 = self._map_over_steps(
            lambda x, k: self._B_map(B_params, x, k),
            x_k1,
            k_indices + 1.0,
        )
        B_xk = self._map_over_steps(
            lambda x, k: self._B_map(B_params, x, k),
            x_k,
            k_indices + 1.0,
        )
        target = jax.lax.stop_gradient(x_k + B_xk1 - B_xk)
        pred = self._map_over_steps(
            lambda x, k: self._F_map(F_params, x, k, F_is_reference=False),
            x_k,
            k_indices,
        )
        return jnp.mean((pred - target) ** 2)

    @partial(jax.jit, static_argnums=0, static_argnames=("F_is_reference",))
    def _backward_update_jit(
        self,
        B_params: Params,
        opt_B: Any,
        F_params: Params,
        batch: Array,
        *,
        F_is_reference: bool,
    ) -> Tuple[Params, Any, Array]:
        loss, grads = jax.value_and_grad(self._backward_loss)(
            B_params,
            F_params,
            batch,
            F_is_reference=F_is_reference,
        )
        B_params, opt_B = adam_update(
            opt_B,
            grads,
            B_params,
            lr=self.dsb_config.learning_rate,
        )
        return B_params, opt_B, loss

    @partial(jax.jit, static_argnums=0)
    def _forward_update_jit(
        self,
        F_params: Params,
        opt_F: Any,
        B_params: Params,
        batch: Array,
    ) -> Tuple[Params, Any, Array]:
        loss, grads = jax.value_and_grad(self._forward_loss)(
            F_params,
            B_params,
            batch,
        )
        F_params, opt_F = adam_update(
            opt_F,
            grads,
            F_params,
            lr=self.dsb_config.learning_rate,
        )
        return F_params, opt_F, loss

    @partial(jax.jit, static_argnums=0, static_argnames=("F_is_reference",))
    def _sample_forward_chain(
        self,
        key: PRNGKey,
        x0: Array,
        params: Params,
        *,
        F_is_reference: bool = False,
    ) -> Array:
        states = [x0]
        x = x0
        for k in range(self.N):
            key, noise_key = jax.random.split(key)
            mean = self._F_map(
                params["F"],
                x,
                jnp.asarray(k, dtype=jnp.float32),
                F_is_reference=F_is_reference,
            )
            x = mean + self._step_noise(noise_key, x, jnp.asarray(k, dtype=jnp.float32))
            states.append(x)
        return jnp.stack(states, axis=1)

    @partial(jax.jit, static_argnums=0)
    def _sample_backward_chain(
        self,
        key: PRNGKey,
        xN: Array,
        params: Params,
    ) -> Array:
        states = [xN]
        x = xN
        for k in range(self.N, 0, -1):
            key, noise_key = jax.random.split(key)
            k_array = jnp.asarray(k, dtype=jnp.float32)
            mean = self._B_map(params["B"], x, k_array)
            x = mean + self._step_noise(noise_key, x, k_array)
            states.append(x)
        return jnp.stack(states[::-1], axis=1)

    def _build_forward_cache(
        self,
        key: PRNGKey,
        cache_size: int,
        params: Params,
        *,
        F_is_reference: bool,
    ) -> Array:
        k1, k2 = jax.random.split(key)
        x0 = self.problem.sample_source(k1, cache_size)
        return self._sample_forward_chain(k2, x0, params, F_is_reference=F_is_reference)

    def _build_backward_cache(
        self,
        key: PRNGKey,
        cache_size: int,
        params: Params,
    ) -> Array:
        k1, k2 = jax.random.split(key)
        xN = self.problem.sample_target(k1, cache_size)
        return self._sample_backward_chain(k2, xN, params)

    def _compute_ot_coupling(
        self,
        x0: Array,
        x1: Array,
        reg: float,
    ) -> Array:
        batch_size = x0.shape[0]
        cost = jnp.sum((x0[:, None, :] - x1[None, :, :]) ** 2, axis=-1)
        kernel = jnp.exp(-cost / reg)
        u = jnp.ones(batch_size)
        v = jnp.ones(batch_size)
        for _ in range(50):
            u = 1.0 / (kernel @ v + 1e-8)
            v = 1.0 / (kernel.T @ u + 1e-8)
        coupling = u[:, None] * kernel * v[None, :]
        return jnp.argmax(coupling, axis=1)

    def _build_ot_warmstart_cache(self, key: PRNGKey, cache_size: int) -> Array:
        k1, k2 = jax.random.split(key)
        x0 = self.problem.sample_source(k1, cache_size)
        xN = self.problem.sample_target(k2, cache_size)
        indices = self._compute_ot_coupling(
            x0,
            xN,
            reg=self.dsb_config.ot_regularization,
        )
        xN = xN[indices]
        s = jnp.linspace(0.0, 1.0, self.N + 1)
        return (
            (1.0 - s[None, :, None]) * x0[:, None, :]
            + s[None, :, None] * xN[:, None, :]
        )

    def _draw_batch(
        self,
        key: PRNGKey,
        trajectories: Array,
        batch_size: int,
    ) -> Array:
        indices = jax.random.randint(key, (batch_size,), 0, trajectories.shape[0])
        return trajectories[indices]

    def sample(
        self,
        key: PRNGKey,
        num_samples: int,
        params: Optional[Params] = None,
        x0: Optional[Array] = None,
        direction: str = "forward",
    ) -> TrajectoryBatch:
        """Sample a DSB Markov chain with the learned mean maps.

        ``direction="forward"`` transports source to target with F maps.
        ``direction="backward"`` transports target to source with B maps, the
        generative direction used in the DSB paper.
        """
        active_params = self._active_params(params)
        k1, k2 = jax.random.split(key)
        if direction == "forward":
            start = self.problem.sample_source(k1, num_samples) if x0 is None else x0
            paths = self._sample_forward_chain(
                k2,
                start,
                active_params,
                F_is_reference=False,
            )
        elif direction == "backward":
            terminal = self.problem.sample_target(k1, num_samples) if x0 is None else x0
            paths = self._sample_backward_chain(k2, terminal, active_params)
        else:
            raise ValueError("direction must be 'forward' or 'backward'.")

        times = self.problem.time_grid.t0 + self.gamma * jnp.arange(self.N + 1)
        return TrajectoryBatch(paths=paths, times=times)

    def sample_backward(
        self,
        key: PRNGKey,
        num_samples: int,
        params: Optional[Params] = None,
        xN: Optional[Array] = None,
    ) -> TrajectoryBatch:
        """Sample the paper's generative target-to-source DSB chain."""
        return self.sample(
            key,
            num_samples,
            params=params,
            x0=xN,
            direction="backward",
        )

    def extract_drift(
        self,
        params: Optional[Params] = None,
        *,
        direction: str = "forward",
        use_ema: bool = True,
    ) -> DriftFn:
        """Approximate a continuous drift from the learned discrete mean map."""
        active_params = params if params is not None else self._active_params(
            None,
            use_ema=use_ema,
        )

        def drift(x: Array, t: Scalar) -> Array:
            x_arr = jnp.asarray(x)
            was_unbatched = x_arr.ndim == 1
            x_batch = jnp.atleast_2d(x_arr)
            k = (
                jnp.asarray(t, dtype=jnp.float32) - self.problem.time_grid.t0
            ) / self.gamma
            if direction == "forward":
                k = jnp.clip(jnp.floor(k), 0, self.N - 1)
                mean = self._F_map(
                    active_params["F"],
                    x_batch,
                    k,
                    F_is_reference=False,
                )
            elif direction == "backward":
                k = jnp.clip(jnp.ceil(k), 1, self.N)
                mean = self._B_map(active_params["B"], x_batch, k)
            else:
                raise ValueError("direction must be 'forward' or 'backward'.")
            value = (mean - x_batch) / self.gamma
            return value[0] if was_unbatched else value

        return drift

    def extract_backward_drift(self, params: Optional[Params] = None) -> DriftFn:
        """Return the backward map as a continuous-time drift approximation."""
        return self.extract_drift(params, direction="backward")

    def _checkpoint_state(self) -> Dict[str, Any]:
        return {
            "ema_params": self._ema_params,
            "F_is_reference": self._F_is_reference,
        }

    def _restore_checkpoint_state(self, state: Dict[str, Any]) -> None:
        self._ema_params = state.get("ema_params")
        self._F_is_reference = bool(state.get("F_is_reference", False))


DSBConfig = IPFDSBConfig
DSBSolver = IPFDSBSolver

__all__ = ["IPFDSBConfig", "IPFDSBSolver", "DSBConfig", "DSBSolver"]
