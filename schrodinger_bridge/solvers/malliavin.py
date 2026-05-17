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

Reference:
    Schrödinger (1931) original formulation
    Pidstrigach et al. (2025) Conditioning Diffusions Using Malliavin Calculus 
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ..core.problem import SBProblem
from ..core.diffusion import (
    apply_diffusion,
    apply_diffusion_covariance,
    solve_diffusion_coefficient,
)
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
    reference_bank_size: int = 8192
    reference_bank_refresh_every: int = 100
    include_diffusion_jacobian: bool = False
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
        self._reference_bank: Optional[Array] = None
        self._reference_bank_age: int = 0

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

    def _diffusion_noise_jacobian(self, x: Array, t: Scalar, dB: Array) -> Array:
        """Jacobian of ``sigma(x,t) dB`` with respect to state.

        This is the stochastic tangent term in the linearized reference flow.
        It is opt-in because differentiating a state-dependent diffusion matrix
        is substantially more expensive than the drift-only tangent update.
        """

        def diffusion_noise_single(xi: Array, dB_i: Array) -> Array:
            sigma_i = self.problem.reference.diffusion(xi[None, :], t)
            return apply_diffusion(
                sigma_i,
                dB_i[None, :],
                is_scalar_diffusion=self.problem.reference.is_diffusion_scalar,
            )[0]

        return jax.vmap(
            jax.jacfwd(diffusion_noise_single, argnums=0),
            in_axes=(0, 0),
        )(x, dB)

    def _alpha_weights(self, num_steps: int, dt: float) -> Array:
        """Return discrete alpha-prime values, not normalized averages."""
        del dt
        mode = self.malliavin_config.alpha_mode.lower()
        if mode == "first":
            weights = jnp.zeros((num_steps,))
            return weights.at[0].set(1.0)
        if mode == "last":
            weights = jnp.zeros((num_steps,))
            return weights.at[-1].set(1.0)
        return jnp.ones((num_steps,))

    def _alpha_normalizers(self, alpha_prime: Array, dt: float) -> Array:
        """Compute A_{T|s} = integral_s^T alpha'_t dt on the grid."""
        alpha_dt = alpha_prime * dt
        return jnp.flip(jnp.cumsum(jnp.flip(alpha_dt)))

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
            drift = self.problem.reference.drift(x, t)
            grad_b = self._drift_jacobian(x, t)

            noise = jax.random.normal(keys[step_idx], x.shape)
            dB = jnp.sqrt(dt) * noise
            diffusion_noise = apply_diffusion(
                sigma,
                dB,
                is_scalar_diffusion=self.problem.reference.is_diffusion_scalar,
            )
            x = x + drift * dt + diffusion_noise

            paths.append(x)
            brownian_increments.append(dB)
            local_jacobian = jnp.eye(dim)[None, :, :] + grad_b * dt
            if self.malliavin_config.include_diffusion_jacobian:
                local_jacobian = local_jacobian + self._diffusion_noise_jacobian(
                    paths[-2],
                    t,
                    dB,
                )
            local_jacobians.append(local_jacobian)

        return (
            jnp.stack(paths, axis=1),
            jnp.stack(brownian_increments, axis=1),
            jnp.stack(local_jacobians, axis=1),
        )

    @partial(jax.jit, static_argnums=0)
    def _estimate_bel_targets(
        self,
        paths: Array,
        dB: Array,
        local_jacobians: Array,
    ) -> Array:
        """Discrete BEL target for every interior time slice.

        Uses the O(N) backward adjoint recurrence

            R_s = alpha'_s sigma_s^{-1} dB_s + J_{s+1|s}^T R_{s+1}

        followed by division by A_{T|s}. This is equivalent to the BEL
        sum but avoids the previous O(N^2) Python accumulation.
        """
        batch_size, num_times, dim = paths.shape
        num_steps = num_times - 1
        dt = self.problem.time_grid.dt
        times = self.problem.time_grid.times
        alpha_prime = self._alpha_weights(num_steps, dt)

        sigma_vals = []
        for t_idx, t in enumerate(times[:-1]):
            sigma_t = jnp.asarray(
                self.problem.reference.diffusion(paths[:, t_idx, :], t)
            )
            if sigma_t.ndim == 0:
                sigma_t = jnp.full((batch_size,), sigma_t)
            sigma_vals.append(sigma_t)
        sigma_vals = jnp.stack(sigma_vals, axis=0)

        scan_inputs = (
            jnp.flip(jnp.swapaxes(local_jacobians, 0, 1), axis=0),
            jnp.flip(jnp.swapaxes(dB, 0, 1), axis=0),
            jnp.flip(sigma_vals, axis=0),
            jnp.flip(alpha_prime, axis=0),
        )

        def scan_step(carry: Array, inputs: Tuple[Array, Array, Array, Array]):
            J_local, dB_step, sigma_step, alpha_step = inputs
            increment = solve_diffusion_coefficient(
                sigma_step,
                dB_step,
                is_scalar_diffusion=self.problem.reference.is_diffusion_scalar,
            )
            transported = jnp.einsum(
                "bij,bj->bi",
                jnp.swapaxes(J_local, 1, 2),
                carry,
            )
            carry = alpha_step * increment + transported
            return carry, carry

        init = jnp.zeros((batch_size, dim))
        _, reversed_targets = jax.lax.scan(scan_step, init, scan_inputs)
        unnormalized = jnp.flip(jnp.swapaxes(reversed_targets, 0, 1), axis=1)

        normalizers = self._alpha_normalizers(alpha_prime, dt)
        valid = normalizers > 1e-8
        targets = unnormalized / jnp.maximum(normalizers[None, :, None], 1e-8)
        return jnp.where(valid[None, :, None], targets, 0.0)

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
            raise ValueError(
                f"Unknown reward_mode: {self.malliavin_config.reward_mode}"
            )

        use_density = (
            mode == "density_ratio"
            or (mode == "auto" and getattr(self.problem.target, "has_density", False))
        )

        if use_density:
            log_target = self.problem.target.log_prob(x_terminal)
        else:
            log_target = _gaussian_kde_log_density(x_terminal, target_bank, bw)

        log_reference = _gaussian_kde_log_density(x_terminal, reference_bank, bw)
        return (log_target - log_reference) / max(
            self.malliavin_config.reward_temperature,
            1e-8,
        )

    def _terminal_weights(
        self,
        x_terminal: Array,
        target_bank: Array,
        reference_bank: Array,
    ) -> Array:
        log_w = self._compute_log_terminal_ratio(
            x_terminal,
            target_bank,
            reference_bank,
        )
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
        paths = jax.lax.stop_gradient(paths)
        bel_targets = jax.lax.stop_gradient(bel_targets)

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
        weight_sum = jnp.sum(weights)
        ess = weight_sum ** 2 / (jnp.sum(weights ** 2) + 1e-8)
        ess_fraction = ess / weights.shape[0]

        metrics = {
            "loss": loss,
            "mean_weight": jnp.mean(weights),
            "max_weight": jnp.max(weights),
            "ess_fraction": ess_fraction,
            "target_norm": jnp.mean(jnp.linalg.norm(bel_targets, axis=-1)),
            "prediction_norm": jnp.mean(jnp.linalg.norm(pred, axis=-1)),
        }
        return loss, metrics

    @partial(jax.jit, static_argnums=0)
    def _gradient_update_jit(
        self,
        params: Params,
        opt_state: AdamState,
        key: PRNGKey,
        x0: Array,
        target_bank: Array,
        reference_bank: Array,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        (_, metrics), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(
            params, key, x0, target_bank, reference_bank
        )
        new_params, new_opt_state = adam_update(
            opt_state,
            grads,
            params,
            lr=self.malliavin_config.learning_rate,
        )
        return new_params, new_opt_state, metrics

    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: AdamState,
        batch_size: int,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        k0, k1, k2, k3, k4 = jax.random.split(key, 5)
        cfg = self.malliavin_config
        x0 = self.problem.sample_source(k0, batch_size)

        target_bank = self.problem.sample_target(
            k1,
            max(batch_size, cfg.target_kde_multiplier * batch_size),
        )

        needs_refresh = (
            self._reference_bank is None
            or self._reference_bank_age >= cfg.reference_bank_refresh_every
        )
        if needs_refresh:
            ref_size = max(batch_size, cfg.reference_bank_size)
            reference_x0 = self.problem.sample_source(k2, ref_size)
            reference_paths, _, _ = self._simulate_reference_rollout(k3, reference_x0)
            self._reference_bank = jax.lax.stop_gradient(reference_paths[:, -1, :])
            self._reference_bank_age = 0
        else:
            self._reference_bank_age += 1

        new_params, new_opt_state, metrics = self._gradient_update_jit(
            params,
            opt_state,
            k4,
            x0,
            target_bank,
            self._reference_bank,
        )

        if self._ema_params is None:
            self._ema_params = new_params
        else:
            self._ema_params = jax.tree_util.tree_map(
                lambda ema, new: cfg.ema_decay * ema + (1.0 - cfg.ema_decay) * new,
                self._ema_params,
                new_params,
            )

        return new_params, new_opt_state, metrics

    def _checkpoint_state(self) -> Dict[str, Optional[Params]]:
        """Persist EMA parameters used for inference."""
        return {"ema_params": self._ema_params}

    def _restore_checkpoint_state(self, state: Dict[str, Optional[Params]]) -> None:
        self._ema_params = state.get("ema_params")

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
            return b_ref + apply_diffusion_covariance(
                sigma,
                score,
                is_scalar_diffusion=self.problem.reference.is_diffusion_scalar,
            )

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
