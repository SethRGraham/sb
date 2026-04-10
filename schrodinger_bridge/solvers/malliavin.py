"""Malliavin / BEL score solver for diffusion Schrödinger bridges.

This module implements a score-style SB solver where the bridge correction is
learned from a Malliavin / Bismut-Elworthy-Li estimator instead of from the
closed-form Brownian-bridge score target used by :class:`ScoreBasedSolver`.

Design
------
We work under the reference dynamics started from the source marginal.
Let R be that path measure and let X_T denote the terminal state. If

    g(x_T) \approx d mu_1 / d(R_T)(x_T),

then the Doob-transformed path measure with Radon-Nikodym derivative g(X_T)
has the target terminal marginal mu_1. The corresponding bridge correction is
the score of the backward potential

    h_t(x) = E_R[g(X_T) | X_t = x].

The score satisfies

    grad log h_t(x)
      = E_R[g(X_T) H_t | X_t = x] / E_R[g(X_T) | X_t = x],

where H_t is a Malliavin / BEL estimator. Therefore the score can be learned
by weighted regression under R:

    min_s E_R[g(X_T) ||s(X_t, t) - H_t||^2].

The minimiser is exactly the conditional score above. This gives a principled
way to replace the analytic bridge score target with a Malliavin estimator.

Practical note
--------------
In general, the terminal density ratio g is unknown. This implementation uses
either:

1. a direct target density if `problem.target.log_prob` is available, together
   with a KDE estimate of the reference terminal density, or
2. KDE estimates for both target and reference terminal densities.

This keeps the solver sample-based and compatible with the rest of the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

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
    SolverType,
)
from ..network_factory import MLPFactory, NetworkFactory, sanity_check
from ..networks import AdamState, adam_update, init_adam
from .base import SBSolver


@dataclass
class MalliavinConfig:
    """Configuration for the Malliavin score solver."""

    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    time_embed_dim: int = 64
    learning_rate: float = 1e-4
    ema_decay: float = 0.999
    alpha_mode: str = "uniform"  # "uniform", "first", "last"
    reward_mode: str = "auto"    # "auto", "density_ratio", "kde_ratio"
    reward_bandwidth: float = 0.25
    reward_temperature: float = 1.0
    normalize_weights: bool = True
    reference_kde_multiplier: int = 1
    target_kde_multiplier: int = 1
    network_factory: Optional[NetworkFactory] = None


def _gaussian_kde_log_density(
    eval_points: Array,
    centers: Array,
    bandwidth: float,
) -> Array:
    """Simple isotropic Gaussian KDE log-density."""
    eval_points = jnp.atleast_2d(eval_points)
    centers = jnp.atleast_2d(centers)
    dim = eval_points.shape[-1]
    sq_dists = jnp.sum((eval_points[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    log_kernel = -0.5 * sq_dists / (bandwidth ** 2 + 1e-8)
    log_norm = -0.5 * dim * jnp.log(2.0 * jnp.pi * (bandwidth ** 2 + 1e-8))
    return jax.nn.logsumexp(log_kernel + log_norm, axis=1) - jnp.log(centers.shape[0])


class MalliavinScoreSolver(SBSolver):
    """Malliavin / BEL-driven score solver for SBs.

    This solver learns a score function s_theta(x, t) using reference-process
    simulations and weighted regression against a discrete BEL target.
    """

    def __init__(
        self,
        problem: SBProblem,
        malliavin_config: Optional[MalliavinConfig] = None,
        config: Optional[Union[MalliavinConfig, SolverConfig]] = None,
        solver_config: Optional[SolverConfig] = None,
        **kwargs,
    ):
        if malliavin_config is None and config is not None:
            if isinstance(config, MalliavinConfig):
                malliavin_config = config
                config = None

        filtered_kwargs = dict(kwargs)
        base_config = solver_config
        if base_config is None and isinstance(config, SolverConfig):
            base_config = config
        if base_config is not None:
            filtered_kwargs["config"] = base_config

        super().__init__(problem, **filtered_kwargs)
        self.malliavin_config = malliavin_config or MalliavinConfig()
        self._ema_params: Optional[Params] = None

        self._factory: NetworkFactory = (
            self.malliavin_config.network_factory
            or MLPFactory(
                hidden_dims=self.malliavin_config.hidden_dims,
                time_embed_dim=self.malliavin_config.time_embed_dim,
            )
        )

    @property
    def solver_type(self) -> SolverType:
        return SolverType.MALLIAVIN

    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.SCORE

    def init_params(self, key: PRNGKey) -> Params:
        params = self._factory.init(key, self.problem.dim, self.problem.dim)
        sanity_check(self._factory, key, self.problem.dim, self.problem.dim)
        return params

    def _init_optimizer(self, params: Params) -> AdamState:
        self._ema_params = params
        return init_adam(params)

    def _drift_jacobian(self, x: Array, t: Scalar) -> Array:
        """Jacobian of the reference drift with respect to state."""

        def drift_single(xi: Array) -> Array:
            return self.problem.reference.drift(xi[None, :], t)[0]

        return jax.vmap(jax.jacfwd(drift_single))(x)

    def _alpha_weights(self, num_steps: int, dt: float) -> Array:
        mode = self.malliavin_config.alpha_mode.lower()
        if mode == "first":
            weights = jnp.zeros((num_steps,))
            return weights.at[0].set(1.0 / dt)
        if mode == "last":
            weights = jnp.zeros((num_steps,))
            return weights.at[-1].set(1.0 / dt)
        return jnp.ones((num_steps,)) / (num_steps * dt)

    def _simulate_reference_rollout(
        self,
        key: PRNGKey,
        x0: Array,
    ) -> Tuple[Array, Array, Array]:
        """Simulate the reference diffusion and auxiliary linearised dynamics."""
        x0 = jnp.atleast_2d(x0)
        _, dim = x0.shape
        times = self.problem.time_grid.times
        dt = self.problem.time_grid.dt
        keys = jax.random.split(key, self.problem.time_grid.num_steps)

        x = x0
        paths = [x0]
        brownian_increments = []
        local_jacobians = []

        for step_idx, t in enumerate(times[:-1]):
            sigma = self.problem.reference.diffusion(x, t)
            if jnp.ndim(jnp.asarray(sigma)) != 0:
                raise NotImplementedError(
                    "MalliavinScoreSolver currently supports scalar diffusion "
                    "coefficients. Extending to matrix-valued diffusion is a "
                    "natural next step."
                )

            drift = self.problem.reference.drift(x, t)
            grad_b = self._drift_jacobian(x, t)

            noise = jax.random.normal(keys[step_idx], x.shape)
            dB = jnp.sqrt(dt) * noise
            x = x + drift * dt + sigma * dB

            paths.append(x)
            brownian_increments.append(dB)
            local_jacobians.append(jnp.eye(dim)[None, :, :] + grad_b * dt)

        return (
            jnp.stack(paths, axis=1),
            jnp.stack(brownian_increments, axis=1),
            jnp.stack(local_jacobians, axis=1),
        )

    def _estimate_bel_targets(
        self,
        paths: Array,
        dB: Array,
        local_jacobians: Array,
    ) -> Array:
        """Discrete BEL target for every interior time slice."""
        batch_size, num_times, dim = paths.shape
        num_steps = num_times - 1
        dt = self.problem.time_grid.dt
        times = self.problem.time_grid.times
        alpha_prime = self._alpha_weights(num_steps, dt)

        sigma_vals = []
        for t_idx, t in enumerate(times[:-1]):
            sigma_t = self.problem.reference.diffusion(paths[:, t_idx, :], t)
            sigma_vals.append(jnp.asarray(sigma_t))
        sigma_vals = jnp.stack(sigma_vals, axis=1)

        eye = jnp.broadcast_to(jnp.eye(dim), (batch_size, dim, dim))
        bel_targets = []

        for start_idx in range(num_steps):
            J = eye
            accum = jnp.zeros((batch_size, dim))
            horizon = max(times[-1] - times[start_idx], 1e-6)

            for future_idx in range(start_idx, num_steps):
                if future_idx > start_idx:
                    J = jnp.einsum(
                        "bij,bjk->bik",
                        local_jacobians[:, future_idx, :, :],
                        J,
                    )

                sigma_inv = 1.0 / (sigma_vals[:, future_idx] + 1e-8)
                increment = sigma_inv[:, None] * dB[:, future_idx, :]
                contrib = alpha_prime[future_idx] * jnp.einsum(
                    "bij,bj->bi",
                    jnp.swapaxes(J, 1, 2),
                    increment,
                )
                accum = accum + contrib

            bel_targets.append(accum / horizon)

        return jnp.stack(bel_targets, axis=1)

    def _compute_log_terminal_ratio(
        self,
        x_terminal: Array,
        target_bank: Array,
        reference_bank: Array,
    ) -> Array:
        """Approximate log(d mu_1 / d R_T) at terminal reference samples."""
        bw = self.malliavin_config.reward_bandwidth
        mode = self.malliavin_config.reward_mode.lower()

        if mode not in {"auto", "density_ratio", "kde_ratio"}:
            raise ValueError(f"Unknown reward_mode: {self.malliavin_config.reward_mode}")

        use_density = (
            mode == "density_ratio"
            or (mode == "auto" and getattr(self.problem.target, "has_density", False))
        )

        if use_density:
            log_target = self.problem.target.log_prob(x_terminal)
        else:
            log_target = _gaussian_kde_log_density(x_terminal, target_bank, bw)

        log_reference = _gaussian_kde_log_density(x_terminal, reference_bank, bw)
        return (log_target - log_reference) / max(self.malliavin_config.reward_temperature, 1e-8)

    def _terminal_weights(
        self,
        x_terminal: Array,
        target_bank: Array,
        reference_bank: Array,
    ) -> Array:
        log_w = self._compute_log_terminal_ratio(x_terminal, target_bank, reference_bank)
        log_w = log_w - jnp.max(log_w)
        w = jnp.exp(log_w)
        if self.malliavin_config.normalize_weights:
            w = w / (jnp.mean(w) + 1e-8)
        return w

    def _loss_fn(
        self,
        params: Params,
        key: PRNGKey,
        x0: Array,
        target_bank: Array,
        reference_bank: Array,
    ) -> Tuple[Scalar, Dict[str, Scalar]]:
        paths, dB, local_jacobians = self._simulate_reference_rollout(key, x0)
        bel_targets = self._estimate_bel_targets(paths, dB, local_jacobians)

        times = jnp.broadcast_to(
            self.problem.time_grid.times[:-1][None, :],
            (paths.shape[0], bel_targets.shape[1]),
        )
        pred = self._factory.forward(
            params,
            paths[:, :-1, :].reshape(-1, self.problem.dim),
            times.reshape(-1),
        ).reshape(bel_targets.shape)

        weights = self._terminal_weights(paths[:, -1, :], target_bank, reference_bank)
        sq_error = (pred - bel_targets) ** 2
        loss = jnp.mean(weights[:, None, None] * sq_error)

        metrics = {
            "loss": loss,
            "mean_weight": jnp.mean(weights),
            "max_weight": jnp.max(weights),
            "target_norm": jnp.mean(jnp.linalg.norm(bel_targets, axis=-1)),
            "prediction_norm": jnp.mean(jnp.linalg.norm(pred, axis=-1)),
        }
        return loss, metrics

    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: AdamState,
        batch_size: int,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        k0, k1, k2, k3 = jax.random.split(key, 4)
        x0 = self.problem.sample_source(k0, batch_size)

        target_bank = self.problem.sample_target(
            k1,
            max(batch_size, self.malliavin_config.target_kde_multiplier * batch_size),
        )
        reference_x0 = self.problem.sample_source(
            k2,
            max(batch_size, self.malliavin_config.reference_kde_multiplier * batch_size),
        )
        reference_paths, _, _ = self._simulate_reference_rollout(k3, reference_x0)
        reference_bank = reference_paths[:, -1, :]

        (loss, metrics), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(
            params, k3, x0, target_bank, reference_bank
        )
        new_params, new_opt_state = adam_update(
            opt_state,
            grads,
            params,
            lr=self.malliavin_config.learning_rate,
        )

        if self._ema_params is None:
            self._ema_params = new_params
        else:
            self._ema_params = jax.tree_util.tree_map(
                lambda ema, new: self.malliavin_config.ema_decay * ema
                + (1.0 - self.malliavin_config.ema_decay) * new,
                self._ema_params,
                new_params,
            )

        return new_params, new_opt_state, metrics

    def extract_drift(self, params: Params) -> DriftFn:
        score_params = self._ema_params if self._ema_params is not None else params
        factory = self._factory

        def drift(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))

            b_ref = self.problem.reference.drift(x, t)
            sigma = self.problem.reference.diffusion(x, t)
            score = factory.forward(score_params, x, t_arr)
            return b_ref + sigma ** 2 * score

        return drift

    def get_score_fn(self, params: Optional[Params] = None):
        if params is None:
            params = self._ema_params if self._ema_params is not None else self._params
        factory = self._factory

        def score(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))
            return factory.forward(params, x, t_arr)

        return score


# Backward-compatible alias for the initial draft class name.
MalliavinBridgeSolver = MalliavinScoreSolver

