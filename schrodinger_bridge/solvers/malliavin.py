"""Malliavin / BEL score solver for diffusion Schrödinger bridges.

The default training path follows the amortised conditioning construction in
Pidstrigach et al. (2025). Reference paths are simulated from the source
marginal, terminal observations ``Y = G(X_T)`` are attached to each path, and a
network learns the conditional bridge score/control from BEL targets:

    min_u E int ||u(X_t, t, Y) - S_t||^2 dt.

For an endpoint bridge, ``G`` is the identity. Sampling a terminal condition
from the target marginal and running the learned conditional drift gives a
sample-based diffusion bridge without estimating a high-dimensional terminal
density ratio.

A legacy density-ratio mode is still available for low-dimensional experiments.
That mode uses terminal weights ``g(X_T)`` and may estimate densities with KDE;
it is not the paper-faithful training path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

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
    TrajectoryBatch,
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
    training_mode: str = "conditional"  # "conditional", "density_ratio"
    observation_dim: Optional[int] = None
    observation_fn: Optional[Callable[[Array], Array]] = None
    reward_mode: str = "auto"    # "auto", "density_ratio", "kde_ratio"
    reward_bandwidth: float = 0.25
    reward_temperature: float = 1.0
    normalize_weights: bool = True
    reference_kde_multiplier: int = 1
    target_kde_multiplier: int = 1
    reference_bank_size: int = 8192
    reference_bank_refresh_every: int = 100
    bel_num_rollouts: int = 1
    include_diffusion_jacobian: bool = False
    local_jacobian_spectral_clip: Optional[float] = None
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

    def _uses_conditional_training(self) -> bool:
        mode = self.malliavin_config.training_mode.lower()
        if mode in {"conditional", "amortized", "amortised", "paper"}:
            return True
        if mode in {"density_ratio", "weighted", "kde", "legacy"}:
            return False
        raise ValueError(f"Unknown training_mode: {self.malliavin_config.training_mode}")

    def _observation_dim(self) -> int:
        cfg = self.malliavin_config
        if cfg.observation_dim is not None:
            return int(cfg.observation_dim)
        if cfg.observation_fn is None:
            return self.problem.dim
        probe = jnp.zeros((1, self.problem.dim))
        observed = jnp.atleast_2d(cfg.observation_fn(probe))
        return int(observed.shape[-1])

    def _network_input_dim(self) -> int:
        if self._uses_conditional_training():
            return self.problem.dim + self._observation_dim()
        return self.problem.dim

    def _observe_terminal(self, x_terminal: Array) -> Array:
        x_terminal = jnp.atleast_2d(x_terminal)
        if self.malliavin_config.observation_fn is None:
            observed = x_terminal
        else:
            observed = self.malliavin_config.observation_fn(x_terminal)
        observed = jnp.atleast_2d(observed)
        expected = self._observation_dim()
        if observed.shape[-1] != expected:
            raise ValueError(
                f"observation_fn returned dim {observed.shape[-1]}, "
                f"expected {expected}."
            )
        return observed

    def _conditioned_network_input(self, x: Array, condition: Array) -> Array:
        x = jnp.atleast_2d(x)
        condition = jnp.atleast_2d(condition)
        if condition.shape[0] == 1 and x.shape[0] != 1:
            condition = jnp.broadcast_to(condition, (x.shape[0], condition.shape[-1]))
        if condition.shape[0] != x.shape[0]:
            raise ValueError(
                "condition batch size must be 1 or match the state batch size; "
                f"got {condition.shape[0]} and {x.shape[0]}."
            )
        return jnp.concatenate([x, condition], axis=-1)

    def init_params(self, key: PRNGKey) -> Params:
        input_dim = self._network_input_dim()
        params = self._factory.init(key, input_dim, self.problem.dim)
        sanity_check(self._factory, key, input_dim, self.problem.dim)
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

    def _clip_local_jacobian(self, local_jacobian: Array) -> Array:
        """Optionally project local tangent maps to a spectral-norm ball."""
        clip = self.malliavin_config.local_jacobian_spectral_clip
        if clip is None or clip <= 0:
            return local_jacobian
        max_sv = jnp.linalg.svd(local_jacobian, compute_uv=False)[..., 0]
        scale = jnp.maximum(max_sv / float(clip), 1.0)
        return local_jacobian / scale[:, None, None]

    def _alpha_weights(self, num_steps: int, dt: float) -> Array:
        """Return discrete alpha-prime values, not normalized averages."""
        del dt
        mode = self.malliavin_config.alpha_mode.lower()
        if mode == "last":
            weights = jnp.zeros((num_steps,))
            return weights.at[-1].set(1.0)
        return jnp.ones((num_steps,))

    def _alpha_normalizers(self, alpha_prime: Array, dt: float) -> Array:
        """Compute A_{T|s} = integral_s^T alpha'_t dt on the grid."""
        alpha_dt = alpha_prime * dt
        return jnp.flip(jnp.cumsum(jnp.flip(alpha_dt)))

    def _alpha_time_mask(self, num_steps: int, dt: float) -> Array:
        """Time slices included in the BEL regression loss."""
        mode = self.malliavin_config.alpha_mode.lower()
        if mode == "first":
            mask = jnp.zeros((num_steps,), dtype=bool)
            return mask.at[0].set(True)
        alpha_prime = self._alpha_weights(num_steps, dt)
        return self._alpha_normalizers(alpha_prime, dt) > 1e-8

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
        eye = jnp.eye(dim)

        def scan_step(x: Array, inputs: Tuple[Scalar, PRNGKey]):
            t, step_key = inputs
            sigma = self.problem.reference.diffusion(x, t)
            drift = self.problem.reference.drift(x, t)
            grad_b = self._drift_jacobian(x, t)

            noise = jax.random.normal(step_key, x.shape)
            dB = jnp.sqrt(dt) * noise
            diffusion_noise = apply_diffusion(
                sigma,
                dB,
                is_scalar_diffusion=self.problem.reference.is_diffusion_scalar,
            )
            x_next = x + drift * dt + diffusion_noise

            local_jacobian = eye[None, :, :] + grad_b * dt
            if self.malliavin_config.include_diffusion_jacobian:
                local_jacobian = local_jacobian + self._diffusion_noise_jacobian(
                    x,
                    t,
                    dB,
                )
            local_jacobian = self._clip_local_jacobian(local_jacobian)
            return x_next, (x_next, dB, local_jacobian)

        _, (path_steps, brownian_increments, local_jacobians) = jax.lax.scan(
            scan_step,
            x0,
            (times[:-1], keys),
        )

        return (
            jnp.concatenate([x0[:, None, :], jnp.swapaxes(path_steps, 0, 1)], axis=1),
            jnp.swapaxes(brownian_increments, 0, 1),
            jnp.swapaxes(local_jacobians, 0, 1),
        )

    def _cached_reference_bank_size(self, batch_size: int) -> int:
        cfg = self.malliavin_config
        multiplier_size = max(1, int(cfg.reference_kde_multiplier)) * int(batch_size)
        fixed_size = max(0, int(cfg.reference_bank_size))
        return max(int(batch_size), multiplier_size, fixed_size)

    def _bel_num_rollouts(self) -> int:
        return max(1, int(self.malliavin_config.bel_num_rollouts))

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
        valid = self._alpha_time_mask(num_steps, dt)
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

    def _bel_training_batch(
        self,
        key: PRNGKey,
        x0: Array,
        target_bank: Array,
        reference_bank: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        num_rollouts = self._bel_num_rollouts()
        rollout_keys = jax.random.split(key, num_rollouts)

        paths, dB, local_jacobians = jax.vmap(
            lambda rollout_key: self._simulate_reference_rollout(rollout_key, x0)
        )(rollout_keys)
        bel_targets = jax.vmap(
            lambda path, increments, jacobians: self._estimate_bel_targets(
                path,
                increments,
                jacobians,
            )
        )(paths, dB, local_jacobians)

        batch_size = x0.shape[0]
        terminal = paths[:, :, -1, :].reshape(num_rollouts * batch_size, -1)
        weights = self._terminal_weights(
            terminal,
            target_bank,
            reference_bank,
        ).reshape(num_rollouts, batch_size)
        metric_weights = weights.reshape(num_rollouts * batch_size)

        if self.malliavin_config.alpha_mode.lower() == "first":
            weight_sum = jnp.sum(weights, axis=0)
            weighted_targets = jnp.sum(
                weights[:, :, None, None] * bel_targets,
                axis=0,
            )
            targets = weighted_targets / jnp.maximum(weight_sum[:, None, None], 1e-8)
            return paths[0], targets, weight_sum / num_rollouts, metric_weights

        num_times = paths.shape[2]
        dim = paths.shape[3]
        loss_weights = weights.reshape(num_rollouts * batch_size)
        return (
            paths.reshape(num_rollouts * batch_size, num_times, dim),
            bel_targets.reshape(
                num_rollouts * batch_size,
                num_times - 1,
                dim,
            ),
            loss_weights,
            metric_weights,
        )

    def _conditional_bel_training_batch(
        self,
        key: PRNGKey,
        x0: Array,
    ) -> Tuple[Array, Array, Array, Array, Array]:
        num_rollouts = self._bel_num_rollouts()
        rollout_keys = jax.random.split(key, num_rollouts)

        paths, dB, local_jacobians = jax.vmap(
            lambda rollout_key: self._simulate_reference_rollout(rollout_key, x0)
        )(rollout_keys)
        bel_targets = jax.vmap(
            lambda path, increments, jacobians: self._estimate_bel_targets(
                path,
                increments,
                jacobians,
            )
        )(paths, dB, local_jacobians)

        batch_size = x0.shape[0]
        num_times = paths.shape[2]
        dim = paths.shape[3]
        terminal = paths[:, :, -1, :].reshape(num_rollouts * batch_size, dim)
        observations = self._observe_terminal(terminal)
        weights = jnp.ones((num_rollouts * batch_size,))

        return (
            paths.reshape(num_rollouts * batch_size, num_times, dim),
            observations,
            bel_targets.reshape(num_rollouts * batch_size, num_times - 1, dim),
            weights,
            weights,
        )

    def _loss_fn(
        self,
        params: Params,
        key: PRNGKey,
        x0: Array,
        target_bank: Array,
        reference_bank: Array,
    ) -> Tuple[Scalar, Dict[str, Scalar]]:
        paths, bel_targets, weights, metric_weights = self._bel_training_batch(
            key,
            x0,
            target_bank,
            reference_bank,
        )
        paths = jax.lax.stop_gradient(paths)
        bel_targets = jax.lax.stop_gradient(bel_targets)
        weights = jax.lax.stop_gradient(weights)
        metric_weights = jax.lax.stop_gradient(metric_weights)

        times = jnp.broadcast_to(
            self.problem.time_grid.times[:-1][None, :],
            (paths.shape[0], bel_targets.shape[1]),
        )
        pred = self._factory.forward(
            params,
            paths[:, :-1, :].reshape(-1, self.problem.dim),
            times.reshape(-1),
        ).reshape(bel_targets.shape)

        sq_error = (pred - bel_targets) ** 2
        time_mask = self._alpha_time_mask(
            bel_targets.shape[1],
            self.problem.time_grid.dt,
        )
        mask = time_mask[None, :, None]
        num_valid_times = jnp.maximum(jnp.sum(time_mask), 1)
        loss = jnp.sum(weights[:, None, None] * sq_error * mask) / (
            weights.shape[0] * num_valid_times * self.problem.dim
        )
        metric_weight_sum = jnp.sum(metric_weights)
        ess = metric_weight_sum ** 2 / (jnp.sum(metric_weights ** 2) + 1e-8)
        ess_fraction = ess / metric_weights.shape[0]
        loss_weight_sum = jnp.sum(weights)
        loss_ess = loss_weight_sum ** 2 / (jnp.sum(weights ** 2) + 1e-8)
        loss_ess_fraction = loss_ess / weights.shape[0]
        target_norms = jnp.linalg.norm(bel_targets, axis=-1)
        prediction_norms = jnp.linalg.norm(pred, axis=-1)
        alpha_prime = self._alpha_weights(
            bel_targets.shape[1],
            self.problem.time_grid.dt,
        )
        alpha_normalizers = self._alpha_normalizers(
            alpha_prime,
            self.problem.time_grid.dt,
        )

        metrics = {
            "loss": loss,
            "mean_weight": jnp.mean(metric_weights),
            "max_weight": jnp.max(metric_weights),
            "ess_fraction": ess_fraction,
            "loss_ess_fraction": loss_ess_fraction,
            "target_norm": jnp.sum(target_norms * time_mask[None, :])
            / (target_norms.shape[0] * num_valid_times),
            "prediction_norm": jnp.sum(prediction_norms * time_mask[None, :])
            / (prediction_norms.shape[0] * num_valid_times),
            "supervised_time_fraction": num_valid_times / bel_targets.shape[1],
            "bel_num_rollouts": jnp.asarray(
                self._bel_num_rollouts(),
                dtype=loss.dtype,
            ),
            "bel_effective_batch_size": jnp.asarray(
                weights.shape[0],
                dtype=loss.dtype,
            ),
            "alpha_normalizer_min": jnp.min(
                jnp.where(time_mask, alpha_normalizers, jnp.inf)
            ),
            "alpha_normalizer_max": jnp.max(
                jnp.where(time_mask, alpha_normalizers, 0.0)
            ),
        }
        return loss, metrics

    def _loss_fn_conditional(
        self,
        params: Params,
        key: PRNGKey,
        x0: Array,
    ) -> Tuple[Scalar, Dict[str, Scalar]]:
        (
            paths,
            observations,
            bel_targets,
            weights,
            metric_weights,
        ) = self._conditional_bel_training_batch(key, x0)
        paths = jax.lax.stop_gradient(paths)
        observations = jax.lax.stop_gradient(observations)
        bel_targets = jax.lax.stop_gradient(bel_targets)
        weights = jax.lax.stop_gradient(weights)
        metric_weights = jax.lax.stop_gradient(metric_weights)

        times = jnp.broadcast_to(
            self.problem.time_grid.times[:-1][None, :],
            (paths.shape[0], bel_targets.shape[1]),
        )
        flat_x = paths[:, :-1, :].reshape(-1, self.problem.dim)
        flat_observations = jnp.broadcast_to(
            observations[:, None, :],
            (paths.shape[0], bel_targets.shape[1], observations.shape[-1]),
        ).reshape(-1, observations.shape[-1])
        network_input = self._conditioned_network_input(flat_x, flat_observations)
        pred = self._factory.forward(
            params,
            network_input,
            times.reshape(-1),
        ).reshape(bel_targets.shape)

        sq_error = (pred - bel_targets) ** 2
        time_mask = self._alpha_time_mask(
            bel_targets.shape[1],
            self.problem.time_grid.dt,
        )
        mask = time_mask[None, :, None]
        num_valid_times = jnp.maximum(jnp.sum(time_mask), 1)
        loss = jnp.sum(weights[:, None, None] * sq_error * mask) / (
            weights.shape[0] * num_valid_times * self.problem.dim
        )
        metric_weight_sum = jnp.sum(metric_weights)
        ess = metric_weight_sum ** 2 / (jnp.sum(metric_weights ** 2) + 1e-8)
        ess_fraction = ess / metric_weights.shape[0]
        target_norms = jnp.linalg.norm(bel_targets, axis=-1)
        prediction_norms = jnp.linalg.norm(pred, axis=-1)
        alpha_prime = self._alpha_weights(
            bel_targets.shape[1],
            self.problem.time_grid.dt,
        )
        alpha_normalizers = self._alpha_normalizers(
            alpha_prime,
            self.problem.time_grid.dt,
        )

        metrics = {
            "loss": loss,
            "mean_weight": jnp.mean(metric_weights),
            "max_weight": jnp.max(metric_weights),
            "ess_fraction": ess_fraction,
            "loss_ess_fraction": ess_fraction,
            "target_norm": jnp.sum(target_norms * time_mask[None, :])
            / (target_norms.shape[0] * num_valid_times),
            "prediction_norm": jnp.sum(prediction_norms * time_mask[None, :])
            / (prediction_norms.shape[0] * num_valid_times),
            "supervised_time_fraction": num_valid_times / bel_targets.shape[1],
            "bel_num_rollouts": jnp.asarray(
                self._bel_num_rollouts(),
                dtype=loss.dtype,
            ),
            "bel_effective_batch_size": jnp.asarray(
                weights.shape[0],
                dtype=loss.dtype,
            ),
            "alpha_normalizer_min": jnp.min(
                jnp.where(time_mask, alpha_normalizers, jnp.inf)
            ),
            "alpha_normalizer_max": jnp.max(
                jnp.where(time_mask, alpha_normalizers, 0.0)
            ),
            "conditional_training": jnp.asarray(1.0, dtype=loss.dtype),
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

    @partial(jax.jit, static_argnums=0)
    def _gradient_update_conditional_jit(
        self,
        params: Params,
        opt_state: AdamState,
        key: PRNGKey,
        x0: Array,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        (_, metrics), grads = jax.value_and_grad(
            self._loss_fn_conditional,
            has_aux=True,
        )(params, key, x0)
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
        if self._uses_conditional_training():
            k0, k1 = jax.random.split(key)
            x0 = self.problem.sample_source(k0, batch_size)
            new_params, new_opt_state, metrics = self._gradient_update_conditional_jit(
                params,
                opt_state,
                k1,
                x0,
            )
            cfg = self.malliavin_config
            if self._ema_params is None:
                self._ema_params = new_params
            else:
                self._ema_params = jax.tree_util.tree_map(
                    lambda ema, new: cfg.ema_decay * ema + (1.0 - cfg.ema_decay) * new,
                    self._ema_params,
                    new_params,
                )
            return new_params, new_opt_state, metrics

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
            ref_size = self._cached_reference_bank_size(batch_size)
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

    def _default_condition(self) -> Array:
        if hasattr(self.problem.target, "mean"):
            return self._observe_terminal(jnp.asarray(self.problem.target.mean)[None, :])
        return jnp.zeros((1, self._observation_dim()))

    def _prepare_condition(self, condition: Optional[Array], batch_size: int) -> Array:
        if condition is None:
            condition = self._default_condition()
        condition = jnp.atleast_2d(condition)
        if condition.shape[0] == 1 and batch_size != 1:
            condition = jnp.broadcast_to(condition, (batch_size, condition.shape[-1]))
        if condition.shape[0] != batch_size:
            raise ValueError(
                "condition batch size must be 1 or match the state batch size; "
                f"got {condition.shape[0]} and {batch_size}."
            )
        expected = self._observation_dim()
        if condition.shape[-1] != expected:
            raise ValueError(
                f"condition dim must be {expected}; got {condition.shape[-1]}."
            )
        return condition

    def _sample_target_conditions(self, key: PRNGKey, num_samples: int) -> Array:
        target_samples = self.problem.sample_target(key, num_samples)
        return self._observe_terminal(target_samples)

    def extract_drift(
        self,
        params: Params,
        condition: Optional[Array] = None,
    ) -> DriftFn:
        score_params = self._ema_params if self._ema_params is not None else params
        factory = self._factory

        def drift(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))

            b_ref = self.problem.reference.drift(x, t)
            sigma = self.problem.reference.diffusion(x, t)
            if self._uses_conditional_training():
                prepared_condition = self._prepare_condition(condition, x.shape[0])
                network_input = self._conditioned_network_input(
                    x,
                    prepared_condition,
                )
            else:
                network_input = x
            score = factory.forward(score_params, network_input, t_arr)
            return b_ref + apply_diffusion_covariance(
                sigma,
                score,
                is_scalar_diffusion=self.problem.reference.is_diffusion_scalar,
            )

        return drift

    def sample(
        self,
        key: PRNGKey,
        num_samples: int,
        params: Optional[Params] = None,
        x0: Optional[Array] = None,
        condition: Optional[Array] = None,
    ) -> TrajectoryBatch:
        if params is None:
            if not self._is_trained:
                raise ValueError("Solver not trained. Call train() first or provide params.")
            params = self._params

        if not self._uses_conditional_training():
            return super().sample(key, num_samples, params=params, x0=x0)

        k0, k1, k2 = jax.random.split(key, 3)
        if x0 is None:
            x0 = self.problem.sample_source(k0, num_samples)
        else:
            x0 = jnp.atleast_2d(x0)
            num_samples = x0.shape[0]

        if condition is None:
            condition = self._sample_target_conditions(k1, num_samples)
        else:
            condition = self._prepare_condition(condition, num_samples)

        drift = self.extract_drift(params, condition=condition)
        return self.integrator.integrate(
            k2,
            x0,
            self.problem.time_grid,
            drift,
            self.problem.sigma,
            True,
        )

    def get_score_fn(
        self,
        params: Optional[Params] = None,
        condition: Optional[Array] = None,
    ):
        if params is None:
            params = self._ema_params if self._ema_params is not None else self._params
        factory = self._factory

        def score(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))
            if self._uses_conditional_training():
                prepared_condition = self._prepare_condition(condition, x.shape[0])
                network_input = self._conditioned_network_input(
                    x,
                    prepared_condition,
                )
            else:
                network_input = x
            return factory.forward(params, network_input, t_arr)

        return score


# Backward-compatible alias for the initial draft class name.
MalliavinBridgeSolver = MalliavinScoreSolver
