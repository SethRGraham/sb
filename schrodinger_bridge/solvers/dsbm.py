"""Diffusion Schrodinger Bridge Matching solver.

This solver follows the bridge-matching construction of Shi et al. (2023).
DSBM alternates forward and backward Markovian fitting steps: endpoint pairs are
refreshed from the previous half-iteration, noisy bridge interiors are sampled
between those endpoints, and time-conditioned networks regress analytic bridge
drift targets.

Reference:
    Shi, De Bortoli, Campbell, and Doucet. "Diffusion Schrodinger Bridge
    Matching." NeurIPS 2023.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

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
from ..core.problem import SBProblem
from ..network_factory import MLPFactory, NetworkFactory, sanity_check
from ..networks import adam_update, init_adam
from .base import SBSolver


@dataclass
class DSBMConfig:
    """Configuration for Diffusion Schrodinger Bridge Matching.

    Attributes:
        num_steps: Discrete sampling steps. Defaults to the problem time grid.
        sigma: Scalar bridge diffusion scale used in the DSBM bridge target.
            Defaults to the scalar reference diffusion at ``t0`` when possible.
        sde: ``"ve"`` for variance-exploding Brownian bridge matching or
            ``"vp"`` for the Ornstein-Uhlenbeck variant used in the reference.
        epsilon: Avoids singular bridge targets at exactly t=0 and t=1.
        n_ipf_iterations: Number of backward/forward outer alternations.
        inner_steps: Optimizer steps per half-iteration.
        batch_size: Batch size used when no TrainingConfig is supplied.
        cache_size: Number of endpoint pairs cached per half-iteration.
        fb_sequence: Half-iteration order. The paper implementation defaults
            to backward then forward.
        first_coupling: Initial endpoint coupling: ``"ref"``, ``"ind"``, or
            ``"ot"``. ``"ref"`` starts the first backward pass from a reference
            perturbation of source samples, matching the PyTorch code.
        mean_match: If True, networks predict endpoint means instead of drifts.
        loss_scale: Weight the MSE by the opposite bridge marginal std.
        skip_terminal_noise: Skip noise on the final sampling step.
    """

    num_steps: Optional[int] = None
    sigma: Optional[float] = None
    sde: str = "ve"
    epsilon: float = 1e-4
    n_ipf_iterations: int = 3
    inner_steps: int = 500
    batch_size: int = 128
    cache_size: int = 4096
    fb_sequence: Tuple[str, ...] = ("b", "f")
    first_coupling: str = "ref"
    mean_match: bool = False
    loss_scale: bool = True
    skip_terminal_noise: bool = True
    learning_rate: float = 1e-4
    ema_decay: float = 0.999
    use_ema: bool = True
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    time_embed_dim: int = 64
    ot_regularization: float = 0.1
    ot_iterations: int = 50
    network_factory: Optional[NetworkFactory] = None

    def __post_init__(self) -> None:
        if self.num_steps is not None and self.num_steps < 1:
            raise ValueError("num_steps must be positive.")
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        if self.sde.lower() not in {"ve", "vp"}:
            raise ValueError("sde must be 've' or 'vp'.")
        if not 0.0 < self.epsilon < 0.5:
            raise ValueError("epsilon must be in (0, 0.5).")
        if self.n_ipf_iterations < 1:
            raise ValueError("n_ipf_iterations must be positive.")
        if self.inner_steps < 1:
            raise ValueError("inner_steps must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.cache_size < 1:
            raise ValueError("cache_size must be positive.")
        if not self.fb_sequence:
            raise ValueError("fb_sequence must contain at least one direction.")
        for fb in self.fb_sequence:
            if fb not in {"f", "b"}:
                raise ValueError("fb_sequence entries must be 'f' or 'b'.")
        if self.first_coupling not in {"ref", "ind", "ot"}:
            raise ValueError("first_coupling must be 'ref', 'ind', or 'ot'.")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1).")


class DSBMSolver(SBSolver):
    """Diffusion Schrodinger Bridge Matching solver."""

    def __init__(
        self,
        problem: SBProblem,
        dsbm_config: Optional[DSBMConfig] = None,
        solver_config: Optional[SolverConfig] = None,
        **kwargs: Any,
    ):
        if dsbm_config is None and "config" in kwargs:
            candidate = kwargs["config"]
            if isinstance(candidate, DSBMConfig):
                dsbm_config = kwargs.pop("config")
        if solver_config is None and "config" in kwargs:
            solver_config = kwargs.pop("config")

        self.dsbm_config = dsbm_config or DSBMConfig()
        if self.dsbm_config.num_steps is None:
            self.dsbm_config.num_steps = problem.time_grid.num_steps
        if self.dsbm_config.sigma is None:
            self.dsbm_config.sigma = self._infer_scalar_sigma(problem)

        super().__init__(problem, config=solver_config)

        self._factory = self.dsbm_config.network_factory or MLPFactory(
            hidden_dims=self.dsbm_config.hidden_dims,
            time_embed_dim=self.dsbm_config.time_embed_dim,
        )
        self._ema_params: Optional[Params] = None
        self._last_direction: Optional[str] = None

    @property
    def solver_type(self) -> SolverType:
        return SolverType.DSBM

    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.DRIFT

    @property
    def num_steps(self) -> int:
        return int(self.dsbm_config.num_steps)

    @property
    def sigma(self) -> float:
        return float(self.dsbm_config.sigma)

    @property
    def alpha(self) -> float:
        return 0.5 if self.dsbm_config.sde.lower() == "vp" else 0.0

    def _infer_scalar_sigma(self, problem: SBProblem) -> float:
        x = jnp.zeros((1, problem.dim), dtype=jnp.float32)
        sigma = jnp.asarray(problem.reference.diffusion(x, problem.time_grid.t0))
        if sigma.ndim == 0:
            return float(sigma)
        if sigma.size == 1:
            return float(sigma.reshape(()))
        raise ValueError(
            "DSBMConfig.sigma must be set when the reference diffusion is not scalar."
        )

    def init_params(self, key: PRNGKey) -> Params:
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
        raise NotImplementedError("DSBMSolver uses an alternating train loop.")

    def train(
        self,
        key: PRNGKey,
        training_config: Optional[TrainingConfig] = None,
        callback: Optional[Any] = None,
    ) -> SolverResult:
        train_config = training_config or TrainingConfig(
            batch_size=self.dsbm_config.batch_size,
        )
        batch_size = (
            train_config.batch_size
            if training_config is not None
            else self.dsbm_config.batch_size
        )

        key, init_key = jax.random.split(key)
        params = self.init_params(init_key)
        opt_F = init_adam(params["F"])
        opt_B = init_adam(params["B"])
        self._ema_params = {"F": params["F"], "B": params["B"]}
        self._params = params

        loss_history: List[float] = []
        checkpoint_paths: List[str] = []
        last_metrics: Dict[str, Any] = {}
        global_step = 0
        previous_direction: Optional[str] = None

        for ipf_iter in range(self.dsbm_config.n_ipf_iterations):
            for direction in self.dsbm_config.fb_sequence:
                key, cache_key = jax.random.split(key)
                first_it = self._is_first_half_iteration(
                    ipf_iter,
                    direction,
                    previous_direction,
                )
                if first_it:
                    pair_cache = self._build_initial_pair_cache(
                        cache_key,
                        self.dsbm_config.cache_size,
                        direction,
                    )
                else:
                    if previous_direction is None:
                        raise ValueError(
                            "No previous DSBM half-iteration is available to build a cache."
                        )
                    pair_cache = self._build_previous_pair_cache(
                        cache_key,
                        self.dsbm_config.cache_size,
                        params,
                        previous_direction,
                    )

                for inner_step in range(self.dsbm_config.inner_steps):
                    key, batch_key, update_key = jax.random.split(key, 3)
                    batch = self._draw_pair_batch(
                        batch_key,
                        pair_cache,
                        batch_size,
                    )
                    if direction == "f":
                        params["F"], opt_F, loss = self._update_jit(
                            update_key,
                            params["F"],
                            opt_F,
                            batch,
                            direction="f",
                        )
                        if self.dsbm_config.use_ema:
                            self._ema_params = {
                                "F": self._ema_update(self._ema_params["F"], params["F"]),
                                "B": self._ema_params["B"],
                            }
                    else:
                        params["B"], opt_B, loss = self._update_jit(
                            update_key,
                            params["B"],
                            opt_B,
                            batch,
                            direction="b",
                        )
                        if self.dsbm_config.use_ema:
                            self._ema_params = {
                                "F": self._ema_params["F"],
                                "B": self._ema_update(self._ema_params["B"], params["B"]),
                            }

                    global_step += 1
                    metrics = {
                        "loss": loss,
                        "phase": "forward" if direction == "f" else "backward",
                        "ipf_iteration": ipf_iter,
                        "inner_step": inner_step,
                    }
                    last_metrics = metrics
                    loss_history.append(float(loss))
                    self._params = params
                    self._last_direction = direction

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

                previous_direction = direction

        self._params = params
        if not self.dsbm_config.use_ema:
            self._ema_params = params
        self._is_trained = True
        self._last_direction = previous_direction

        diagnostics = self._run_diagnostics(key, params)
        metadata = {
            "converged": True,
            "final_step": global_step,
            "solver_type": self.solver_type.name,
            "n_ipf_iterations": self.dsbm_config.n_ipf_iterations,
            "inner_steps": self.dsbm_config.inner_steps,
            "num_steps": self.num_steps,
            "sigma": self.sigma,
            "sde": self.dsbm_config.sde,
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

    def _is_first_half_iteration(
        self,
        ipf_iter: int,
        direction: str,
        previous_direction: Optional[str],
    ) -> bool:
        if ipf_iter != 0:
            return False
        if self.dsbm_config.first_coupling == "ind":
            return True
        return previous_direction is None or direction == "b"

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
        decay = self.dsbm_config.ema_decay
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
            raise ValueError("Solver not trained. Call train() first or provide params.")
        if use_ema and self.dsbm_config.use_ema and self._ema_params is not None:
            return self._ema_params
        return self._params

    def get_trained_params(self, use_ema: bool = True) -> Params:
        return self._active_params(None, use_ema=use_ema)

    def _network(self, net_params: Params, x: Array, t: Array) -> Array:
        x = jnp.atleast_2d(x)
        t = jnp.asarray(t, dtype=jnp.float32)
        if t.ndim == 0:
            t = jnp.full((x.shape[0],), t)
        if t.ndim > 1:
            t = t.reshape((t.shape[0],))
        return self._factory.forward(net_params, x, t)

    def _safe_t(self, t: Array) -> Array:
        eps = self.dsbm_config.epsilon
        return jnp.clip(t, eps, 1.0 - eps)

    def _coefficients(self, t: Array, direction: str) -> Tuple[Array, Array]:
        t = self._safe_t(t)
        alpha = self.alpha
        if self.dsbm_config.sde.lower() == "ve":
            if direction == "f":
                return -1.0 / (1.0 - t), 1.0 / (1.0 - t)
            return -1.0 / t, 1.0 / t

        if direction == "f":
            tau = 1.0 - t
        else:
            tau = t
        a = -alpha / jnp.tanh(alpha * tau)
        m = alpha / jnp.sinh(alpha * tau)
        return a, m

    def _bridge_drift_from_endpoints(
        self,
        x: Array,
        t: Array,
        endpoint: Array,
        direction: str,
    ) -> Array:
        t_col = self._safe_t(t)[:, None]
        a, m = self._coefficients(t_col, direction)
        return a * x + m * endpoint

    def _drift_from_prediction(
        self,
        pred: Array,
        x: Array,
        t: Array,
        direction: str,
    ) -> Array:
        if self.dsbm_config.mean_match:
            return self._bridge_drift_from_endpoints(x, t, pred, direction)
        if direction == "f":
            return pred - self.alpha * x
        return pred + self.alpha * x

    def _loss_scale(self, t: Array, direction: str) -> Array:
        t = self._safe_t(t)
        if self.dsbm_config.sde.lower() == "ve":
            scale_t = 1.0 - t if direction == "f" else t
            return self.sigma * jnp.sqrt(scale_t)
        if direction == "f":
            scale_t = 1.0 - t
        else:
            scale_t = t
        return self.sigma * jnp.sqrt(1.0 - jnp.exp(-scale_t))

    def _make_training_tuple(
        self,
        key: PRNGKey,
        pairs: Array,
        direction: str,
    ) -> Tuple[Array, Array, Array]:
        z0 = pairs[:, 0, :]
        z1 = pairs[:, 1, :]
        k_t, k_z = jax.random.split(key)
        t = jax.random.uniform(
            k_t,
            (pairs.shape[0],),
            minval=self.dsbm_config.epsilon,
            maxval=1.0 - self.dsbm_config.epsilon,
        )
        noise = jax.random.normal(k_z, z0.shape)
        t_col = t[:, None]
        z_t = (1.0 - t_col) * z0 + t_col * z1
        z_t = z_t + self.sigma * jnp.sqrt(t_col * (1.0 - t_col)) * noise

        if self.dsbm_config.mean_match:
            target = z1 if direction == "f" else z0
        else:
            endpoint = z1 if direction == "f" else z0
            drift = self._bridge_drift_from_endpoints(z_t, t, endpoint, direction)
            if direction == "f":
                target = drift + self.alpha * z_t
            else:
                target = drift - self.alpha * z_t
        return z_t, t, target

    def _loss(
        self,
        net_params: Params,
        key: PRNGKey,
        pairs: Array,
        *,
        direction: str,
    ) -> Array:
        x_t, t, target = self._make_training_tuple(key, pairs, direction)
        pred = self._network(net_params, x_t, t)
        if self.dsbm_config.loss_scale:
            scale = self._loss_scale(t, direction)[:, None]
        else:
            scale = 1.0
        return jnp.mean((scale * (pred - target)) ** 2)

    @partial(jax.jit, static_argnums=0, static_argnames=("direction",))
    def _update_jit(
        self,
        key: PRNGKey,
        net_params: Params,
        opt_state: Any,
        pairs: Array,
        *,
        direction: str,
    ) -> Tuple[Params, Any, Array]:
        loss, grads = jax.value_and_grad(self._loss)(
            net_params,
            key,
            pairs,
            direction=direction,
        )
        net_params, opt_state = adam_update(
            opt_state,
            grads,
            net_params,
            lr=self.dsbm_config.learning_rate,
        )
        return net_params, opt_state, loss

    def _draw_pair_batch(self, key: PRNGKey, pairs: Array, batch_size: int) -> Array:
        idx = jax.random.randint(key, (batch_size,), 0, pairs.shape[0])
        return pairs[idx]

    def _sinkhorn_match(self, x0: Array, x1: Array) -> Array:
        cost = jnp.sum((x0[:, None, :] - x1[None, :, :]) ** 2, axis=-1)
        scale = jnp.median(cost) + 1e-8
        kernel = jnp.exp(-cost / (self.dsbm_config.ot_regularization * scale + 1e-8))
        u = jnp.ones((x0.shape[0],), dtype=x0.dtype)
        v = jnp.ones((x1.shape[0],), dtype=x1.dtype)
        for _ in range(self.dsbm_config.ot_iterations):
            u = 1.0 / (kernel @ v + 1e-8)
            v = 1.0 / (kernel.T @ u + 1e-8)
        coupling = u[:, None] * kernel * v[None, :]
        return x1[jnp.argmax(coupling, axis=1)]

    def _build_initial_pair_cache(
        self,
        key: PRNGKey,
        cache_size: int,
        direction: str,
    ) -> Array:
        k0, k1, k2, k3 = jax.random.split(key, 4)
        x0 = self.problem.sample_source(k0, cache_size)
        x1 = self.problem.sample_target(k1, cache_size)

        if self.dsbm_config.first_coupling == "ref" and direction == "b":
            z0 = x0
            z1 = x0 + self.sigma * jax.random.normal(k2, x0.shape)
        elif self.dsbm_config.first_coupling == "ot":
            z0 = x0
            z1 = self._sinkhorn_match(x0, x1)
        else:
            z0 = x0
            perm = jax.random.permutation(k3, cache_size)
            z1 = x1[perm]
        return jnp.stack([z0, z1], axis=1)

    def _build_previous_pair_cache(
        self,
        key: PRNGKey,
        cache_size: int,
        params: Params,
        previous_direction: str,
    ) -> Array:
        k0, k1 = jax.random.split(key)
        if previous_direction == "f":
            x0 = self.problem.sample_source(k0, cache_size)
            paths = self._sample_chain_jit(k1, x0, params["F"], direction="f")
            return jnp.stack([x0, paths[:, -1, :]], axis=1)

        x1 = self.problem.sample_target(k0, cache_size)
        paths = self._sample_chain_jit(k1, x1, params["B"], direction="b")
        return jnp.stack([paths[:, 0, :], x1], axis=1)

    @partial(jax.jit, static_argnums=0, static_argnames=("direction",))
    def _sample_chain_jit(
        self,
        key: PRNGKey,
        start: Array,
        net_params: Params,
        *,
        direction: str,
    ) -> Array:
        dt = 1.0 / float(self.num_steps)
        x = jnp.atleast_2d(start)
        states = [x]

        for step in range(self.num_steps):
            key, noise_key = jax.random.split(key)
            if direction == "f":
                t = jnp.asarray(step * dt, dtype=jnp.float32)
            else:
                t = jnp.asarray(1.0 - step * dt, dtype=jnp.float32)
            t_batch = jnp.full((x.shape[0],), t)
            pred = self._network(net_params, x, t_batch)
            drift = self._drift_from_prediction(pred, x, t_batch, direction)
            x = x + drift * dt
            if not (
                self.dsbm_config.skip_terminal_noise and step == self.num_steps - 1
            ):
                x = x + self.sigma * jnp.sqrt(dt) * jax.random.normal(noise_key, x.shape)
            states.append(x)

        path = jnp.stack(states, axis=1)
        if direction == "b":
            path = path[:, ::-1, :]
        return path

    def sample(
        self,
        key: PRNGKey,
        num_samples: int,
        params: Optional[Params] = None,
        x0: Optional[Array] = None,
        direction: str = "forward",
        use_ema: bool = True,
    ) -> TrajectoryBatch:
        active_params = self._active_params(params, use_ema=use_ema)
        k0, k1 = jax.random.split(key)
        if direction in {"forward", "f"}:
            start = self.problem.sample_source(k0, num_samples) if x0 is None else x0
            paths = self._sample_chain_jit(k1, start, active_params["F"], direction="f")
        elif direction in {"backward", "b"}:
            start = self.problem.sample_target(k0, num_samples) if x0 is None else x0
            paths = self._sample_chain_jit(k1, start, active_params["B"], direction="b")
        else:
            raise ValueError("direction must be 'forward'/'f' or 'backward'/'b'.")

        times = jnp.linspace(
            self.problem.time_grid.t0,
            self.problem.time_grid.t1,
            self.num_steps + 1,
        )
        return TrajectoryBatch(paths=paths, times=times)

    def sample_backward(
        self,
        key: PRNGKey,
        num_samples: int,
        params: Optional[Params] = None,
        xN: Optional[Array] = None,
    ) -> TrajectoryBatch:
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
        active_params = self._active_params(params, use_ema=use_ema)

        def drift(x: Array, t: Scalar) -> Array:
            x_arr = jnp.asarray(x)
            was_unbatched = x_arr.ndim == 1
            x_batch = jnp.atleast_2d(x_arr)
            t_norm = (
                jnp.asarray(t, dtype=jnp.float32) - self.problem.time_grid.t0
            ) / (self.problem.time_grid.t1 - self.problem.time_grid.t0)
            t_batch = jnp.full((x_batch.shape[0],), t_norm)
            if direction in {"forward", "f"}:
                pred = self._network(active_params["F"], x_batch, t_batch)
                value = self._drift_from_prediction(pred, x_batch, t_batch, "f")
            elif direction in {"backward", "b"}:
                pred = self._network(active_params["B"], x_batch, t_batch)
                value = self._drift_from_prediction(pred, x_batch, t_batch, "b")
            else:
                raise ValueError("direction must be 'forward'/'f' or 'backward'/'b'.")
            return value[0] if was_unbatched else value

        return drift

    def extract_backward_drift(self, params: Optional[Params] = None) -> DriftFn:
        return self.extract_drift(params, direction="backward")

    def _checkpoint_state(self) -> Dict[str, Any]:
        return {
            "ema_params": self._ema_params,
            "last_direction": self._last_direction,
        }

    def _restore_checkpoint_state(self, state: Dict[str, Any]) -> None:
        self._ema_params = state.get("ema_params")
        self._last_direction = state.get("last_direction")

__all__ = ["DSBMConfig", "DSBMSolver"]
