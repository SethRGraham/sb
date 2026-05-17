"""Forward-Backward SDE Solver for Schrödinger Bridges.

Implements Chen, Liu, Theodorou. "Likelihood Training of Schrödinger Bridge
using Forward-Backward SDEs Theory" (ICLR 2022).


SB optimality (Theorem 1) gives two coupled SDEs:

    Forward:  dX_t = [f + G*grad log Psi] dt + sigma dW_t,    X_0 ~ mu_0    (7a)
    Backward: dX_t = [f - G*grad log Psi_hat] dt + sigma dW_tilde,   X_T ~ mu_1    (7b)

Training (Algorithm 3) alternates:
    Phase 1: Cache forward trajectories -> train backward grad log Psi_hat via eq (18)
    Phase 2: Cache backward trajectories -> train forward grad log Psi via eq (19)

The per-policy loss (eqs 18/19) for the trainable network b_theta is:

    L(theta) = E[ 1/2 b^TG b + div(G*b - f) + b_hat_frozen^T G b ]
                                         ^^^^^^^^^^^^^^^^
                                         cross-term: essential!

Supported methods:
    sb_fbsde  - Paper-faithful two-policy alternating training (DEFAULT)
    deep_bsde - Legacy single-Z with BSDE consistency (Han/Jentzen/E 2018)
    soc       - Legacy direct stochastic optimal control
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ..core.diffusion import apply_diffusion, apply_diffusion_covariance
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
    DiagnosticReport,
)
from ..core.problem import SBProblem
from ..networks import (
    init_adam,
    adam_update,
    AdamState,
)
from ..network_factory import NetworkFactory, MLPFactory, sanity_check
from .base import SBSolver


# Diffusion geometry helpers
# The reference SDE is dX = f(x,t) dt + sigma(x,t) dW where sigma may be:
#   - Scalar:  sigma in R             -> G = sigma^2 * I
#   - Matrix:  sigma in R^{dxd}       -> G = sigma * sigma^T
# The SB optimal drift is f + G*grad log Psi = f + sigma*sigma^T*b.

def _apply_g(g: Array, v: Array) -> Array:
    """Apply diffusion coefficient: sigma * v.

    Handles scalar sigma (isotropic) and matrix sigma (general).
    """
    return apply_diffusion(g, v)


def _apply_GG(g: Array, v: Array) -> Array:
    """Apply diffusion tensor: G * v = sigma*sigma^T * v.

    For scalar sigma: returns sigma^2 * v.
    For matrix sigma: returns sigma @ (sigma^T @ v).
    """
    return apply_diffusion_covariance(g, v)


# Solution containers
class FBSDESolution(NamedTuple):
    """Solution of the FBSDE system (backward-compatible container)."""
    X: Array         # Forward process [batch, time, dim]
    Y: Array         # Value function [batch, time] (deep_bsde only; zeros otherwise)
    Z: Array         # Forward policy [batch, time, dim]
    times: Array


class SBFBSDESolution(NamedTuple):
    """Full two-policy SB-FBSDE solution."""
    X_forward: Array       # Forward trajectories mu_0->mu_1 [batch, time, dim]
    X_backward: Array      # Backward trajectories mu_1->mu_0 [batch, time, dim]
    forward_policy: Array  # grad log Psi along forward path [batch, time, dim]
    backward_policy: Array # grad log Psi_hat along backward path [batch, time, dim]
    times: Array


# Configuration
@dataclass
class FBSDEConfig:
    """Configuration for FBSDE solver.

    Attributes:
        method: 'sb_fbsde' (paper-faithful, default), 'deep_bsde', or 'soc'.

        --- sb_fbsde (Algorithm 3) ---
        num_stages: Alternating training stages (each = train bwd + fwd).
        steps_per_stage: Gradient steps per policy per stage.
        trajectory_batch_size: Trajectories cached per stage.

        --- Langevin corrector (Sec.3.3, Alg 4) ---
        use_langevin_corrector: Enable corrector during sampling.
        corrector_steps: Number of Langevin steps per predictor step.
        corrector_snr: Signal-to-noise ratio r (paper uses 0.05).

        --- Legacy (deep_bsde / soc) ---
        terminal_weight: Weight on terminal matching loss.
        running_weight: Weight on control regularization.
    """
    # Method
    method: str = 'sb_fbsde'

    # Architecture
    hidden_dims: Tuple[int, ...] = (256, 256)
    time_embed_dim: int = 64
    learning_rate: float = 2e-4  # Paper uses 2e-4 for toy datasets

    # sb_fbsde training (Algorithm 3)
    num_stages: int = 5
    steps_per_stage: int = 2000
    trajectory_batch_size: int = 512

    # Langevin corrector
    use_langevin_corrector: bool = False
    corrector_steps: int = 1
    corrector_snr: float = 0.05

    # Legacy
    terminal_weight: float = 1.0
    running_weight: float = 0.01

    # Network factories
    forward_network_factory: Optional[NetworkFactory] = None
    backward_network_factory: Optional[NetworkFactory] = None
    network_factory: Optional[NetworkFactory] = None        # backward compat alias
    value_network_factory: Optional[NetworkFactory] = None   # deep_bsde Y network


# Solver
class FBSDESolver(SBSolver):
    """FBSDE-based Schrödinger Bridge solver.

    Three training methods:

    sb_fbsde (default, paper-faithful):
        Two policy networks grad log Psi (forward) and grad log Psi_hat (backward) trained
        with alternating stages. Loss includes the cross-term b_hat^TG b that
        couples the two policies. Score grad log p_t = grad log Psi + grad log Psi_hat.

    deep_bsde (legacy):
        Single Z(x,t) control + Y(x,t) value network (Han/Jentzen/E 2018).

    soc (legacy):
        Direct stochastic optimal control with running + terminal cost.
    """

    def __init__(
        self,
        problem: SBProblem,
        fbsde_config: Optional[FBSDEConfig] = None,
        config: Optional[Union[FBSDEConfig, SolverConfig]] = None,
        solver_config: Optional[SolverConfig] = None,
        **kwargs,
    ):
        # - Handle config parameter flexibility -
        if fbsde_config is None and config is not None:
            if isinstance(config, FBSDEConfig):
                fbsde_config = config
                config = None
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if not isinstance(v, FBSDEConfig)}
        base_config = solver_config or (
            config if isinstance(config, SolverConfig) else None
        )
        if base_config is not None:
            filtered_kwargs['config'] = base_config

        super().__init__(problem, **filtered_kwargs)
        self.fbsde_config = fbsde_config or FBSDEConfig()

        # - Build network factories -
        def _default():
            return MLPFactory(
                hidden_dims=self.fbsde_config.hidden_dims,
                time_embed_dim=self.fbsde_config.time_embed_dim,
            )

        cfg = self.fbsde_config
        self._forward_factory: NetworkFactory = (
            cfg.forward_network_factory or cfg.network_factory or _default()
        )
        self._backward_factory: NetworkFactory = (
            cfg.backward_network_factory or _default()
        )
        self._y_factory: NetworkFactory = (
            cfg.value_network_factory or _default()
        )
        # backward compat alias
        self._z_factory = self._forward_factory

    # - Properties -
    @property
    def solver_type(self) -> SolverType:
        return SolverType.FBSDE

    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.CONTROL

    # Parameter initialization
    def init_params(self, key: PRNGKey) -> Params:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        dim = self.problem.dim

        if self.fbsde_config.method == 'sb_fbsde':
            fwd = self._forward_factory.init(k1, dim, dim)
            bwd = self._backward_factory.init(k2, dim, dim)
            sanity_check(self._forward_factory, k3, dim, dim)
            sanity_check(self._backward_factory, k4, dim, dim)
            return {'forward': fwd, 'backward': bwd}
        else:
            z = self._z_factory.init(k1, dim, dim)
            y = self._y_factory.init(k2, dim, 1)
            sanity_check(self._z_factory, k3, dim, dim)
            return {'z': z, 'y': y}

    # Network evaluation helpers
    # All networks output b = grad log Psi (raw score, shape [batch, dim]).
    # The sigma factors are applied separately via _apply_g / _apply_GG.

    def _forward_fn(self, params: Params, x: Array, t: Array) -> Array:
        """grad log Psi(x,t) - forward policy."""
        key = 'forward' if 'forward' in params else 'z'
        return self._forward_factory.forward(params[key], x, t)

    def _backward_fn(self, params: Params, x: Array, t: Array) -> Array:
        """grad log Psi_hat(x,t) - backward policy."""
        return self._backward_factory.forward(params['backward'], x, t)

    def _y_fn(self, params: Params, x: Array, t: Array) -> Array:
        """Y(x,t) - value function (deep_bsde only)."""
        return self._y_factory.forward(params['y'], x, t).squeeze(-1)

    # backward compat
    def _z_fn(self, params: Params, x: Array, t: Array) -> Array:
        return self._forward_fn(params, x, t)

    # SDE simulation - with proper diffusion geometry
    # Forward:  dX = [f + G*b_fwd] dt + sigma dW        (7a)
    # Backward: dX = [f - G*b_bwd] dt + sigma dW_tilde        (7b)
    # where G = sigma*sigma^T is the diffusion tensor.

    @partial(jax.jit, static_argnums=0)
    def _simulate_forward_sde(
        self, params: Params, key: PRNGKey, x0: Array,
    ) -> Array:
        """Simulate forward SDE (7a).

        Returns trajectories [batch, num_times, dim].
        """
        times = self.problem.time_grid.times
        dt = self.problem.time_grid.dt
        num_steps = self.problem.time_grid.num_steps
        batch_size = x0.shape[0]
        keys = jax.random.split(key, num_steps)

        def step_fn(x, inputs):
            t, step_key = inputs
            f = self.problem.reference.drift(x, t)
            g = self.problem.reference.diffusion(x, t)
            b = self._forward_fn(params, x, jnp.full((batch_size,), t))

            # Controlled drift: f + G*b where G = sigma*sigma^T
            drift = f + _apply_GG(g, b)

            # Noise: sigma*dW
            dW = jax.random.normal(step_key, x.shape) * jnp.sqrt(dt)
            noise = _apply_g(g, dW)

            x_next = x + drift * dt + noise
            return x_next, x_next

        _, X_steps = jax.lax.scan(step_fn, x0, (times[:-1], keys))
        X_traj = jnp.concatenate([x0[None], X_steps], axis=0)
        return jnp.transpose(X_traj, (1, 0, 2))

    @partial(jax.jit, static_argnums=0)
    def _simulate_backward_sde(
        self, params: Params, key: PRNGKey, x_T: Array,
    ) -> Array:
        """Simulate backward SDE (7b) from t=T to t=0.

        Euler-Maruyama going backward:
            X_{t-dt} = X_t - [f - G*b_hat]*dt + sigma*sqrt(dt)*eps
                     = X_t - f*dt + G*b_hat*dt + sigma*sqrt(dt)*eps

        Returns trajectories [batch, num_times, dim] in chronological order.
        """
        times = self.problem.time_grid.times
        dt = self.problem.time_grid.dt
        num_steps = self.problem.time_grid.num_steps
        batch_size = x_T.shape[0]
        keys = jax.random.split(key, num_steps)
        reversed_times = times[::-1]

        def step_fn(x, inputs):
            t, step_key = inputs
            f = self.problem.reference.drift(x, t)
            g = self.problem.reference.diffusion(x, t)
            b_hat = self._backward_fn(params, x, jnp.full((batch_size,), t))

            # Backward Euler: X_{t-dt} = X_t - (f - G*b_hat)dt + sigma*sqrt(dt) eps
            dW = jax.random.normal(step_key, x.shape) * jnp.sqrt(dt)
            x_prev = x - f * dt + _apply_GG(g, b_hat) * dt + _apply_g(g, dW)
            return x_prev, x_prev

        _, X_reversed = jax.lax.scan(step_fn, x_T, (reversed_times[:-1], keys))
        X_traj_rev = jnp.concatenate([x_T[None], X_reversed], axis=0)
        X_traj = X_traj_rev[::-1]  # flip to chronological order
        return jnp.transpose(X_traj, (1, 0, 2))

    # Paper's divergence-based loss (Theorem 4, eqs 18/19)
    # Eqs (18) and (19) share the same structure for the trainable policy:
    #   L(theta) = integral E[
    #       1/2 * ||sigma*b_theta||^2
    #       + div_x(G*b_theta - f)
    #       + (sigma*b_hat_frozen)^T (sigma*b_theta)
    #   ] dt
    # Key implementation details:
    #   - term 2: the network MUST be evaluated inside the JVP so the
    #     Hutchinson estimator captures db_theta/dx, not just dsigma/dx.
    #   - term 3: frozen policy uses stop_gradient to block backprop.
    #   - The -div(f) term is included for correct loss monitoring (gradient-neutral
    #     but makes the reported value match the actual bound).

    def _sb_fbsde_loss(
        self,
        trainable_params: Params,
        trainable_factory: NetworkFactory,
        frozen_params: Params,
        frozen_factory: NetworkFactory,
        x_batch: Array,
        t_batch: Array,
        key: PRNGKey,
    ) -> Tuple[Scalar, Dict[str, Scalar]]:
        """Paper's per-policy loss (eqs 18/19).

        Args:
            trainable_params: Params for the policy being optimized.
            trainable_factory: NetworkFactory for the trainable policy.
            frozen_params: Params for the OTHER (frozen) policy.
            frozen_factory: NetworkFactory for the frozen policy.
            x_batch: Training points [batch, dim] from cached trajectories.
            t_batch: Corresponding times [batch].
            key: PRNG key for Hutchinson estimator.
        """
        # - Evaluate both policies -
        b_train = trainable_factory.forward(trainable_params, x_batch, t_batch)
        b_frozen = jax.lax.stop_gradient(
            frozen_factory.forward(frozen_params, x_batch, t_batch)
        )

        # - Diffusion coefficient at each (x, t) -
        g_vals = jax.vmap(
            lambda x, t: self.problem.reference.diffusion(x, t)
        )(x_batch, t_batch)

        # - Term 1: 1/2 ||sigma*b_train||^2 -
        # = 1/2 b^T(sigma^Tsigma)b  ->  for scalar sigma: 1/2sigma^2||b||^2
        g_b = jax.vmap(_apply_g)(g_vals, b_train)       # sigma*b  [batch, dim]
        term1 = 0.5 * jnp.sum(g_b ** 2, axis=-1)        # [batch]

        # - Term 2: div_x (G*b_train - f) via Hutchinson -
        # The trainable network is evaluated INSIDE the JVP so that
        # the estimator captures db_theta/dx, not just dsigma/dx.
        v = 2.0 * jax.random.bernoulli(key, shape=x_batch.shape).astype(x_batch.dtype) - 1.0

        def _full_vector_field(x_):
            """G(x)*b_train(x,t) - f(x,t)  - the complete vector field."""
            g_ = jax.vmap(
                lambda xi, ti: self.problem.reference.diffusion(xi, ti)
            )(x_, t_batch)
            b_ = trainable_factory.forward(trainable_params, x_, t_batch)
            f_ = jax.vmap(
                lambda xi, ti: self.problem.reference.drift(xi, ti)
            )(x_, t_batch)
            return jax.vmap(_apply_GG)(g_, b_) - f_

        _, Jv = jax.jvp(_full_vector_field, (x_batch,), (v,))
        term2 = jnp.sum(v * Jv, axis=-1)  # Hutchinson trace estimate [batch]

        # - Term 3: (sigma*b_hat_frozen)^T (sigma*b_train) - the cross-term -
        g_b_frozen = jax.vmap(_apply_g)(g_vals, b_frozen)   # sigma*b_hat  [batch, dim]
        term3 = jnp.sum(g_b_frozen * g_b, axis=-1)          # [batch]

        loss = jnp.mean(term1 + term2 + term3)

        metrics = {
            'loss': loss,
            'score_norm': jnp.mean(jnp.sum(b_train ** 2, axis=-1)),
            'divergence': jnp.mean(term2),
            'cross_term': jnp.mean(term3),
        }
        return loss, metrics

    @partial(jax.jit, static_argnums=0)
    def _sb_backward_update_jit(
        self,
        backward_params: Params,
        backward_opt: AdamState,
        forward_params: Params,
        x_batch: Array,
        t_batch: Array,
        key: PRNGKey,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        (_, metrics), grads = jax.value_and_grad(
            self._sb_fbsde_loss, has_aux=True
        )(
            backward_params,
            self._backward_factory,
            forward_params,
            self._forward_factory,
            x_batch,
            t_batch,
            key,
        )
        new_params, new_opt = adam_update(
            backward_opt,
            grads,
            backward_params,
            lr=self.fbsde_config.learning_rate,
        )
        return new_params, new_opt, metrics

    @partial(jax.jit, static_argnums=0)
    def _sb_forward_update_jit(
        self,
        forward_params: Params,
        forward_opt: AdamState,
        backward_params: Params,
        x_batch: Array,
        t_batch: Array,
        key: PRNGKey,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        (_, metrics), grads = jax.value_and_grad(
            self._sb_fbsde_loss, has_aux=True
        )(
            forward_params,
            self._forward_factory,
            backward_params,
            self._backward_factory,
            x_batch,
            t_batch,
            key,
        )
        new_params, new_opt = adam_update(
            forward_opt,
            grads,
            forward_params,
            lr=self.fbsde_config.learning_rate,
        )
        return new_params, new_opt, metrics

    # SB-FBSDE training (Algorithm 3)
    def _sample_trajectory_data(
        self, trajectories: Array, key: PRNGKey, batch_size: int,
    ) -> Tuple[Array, Array]:
        """Sample random (x_t, t) pairs from cached trajectories.

        Avoids exact boundary times (t=0, t=T) for numerical stability of
        the score matching divergence estimator.
        """
        k1, k2 = jax.random.split(key)
        traj_batch, num_times, dim = trajectories.shape
        times = self.problem.time_grid.times

        traj_idx = jax.random.randint(k1, (batch_size,), 0, traj_batch)
        time_idx = jax.random.randint(k2, (batch_size,), 1, num_times - 1)

        x_batch = trajectories[traj_idx, time_idx, :]
        t_batch = times[time_idx]
        return x_batch, t_batch

    def _train_sb_fbsde(
        self,
        key: PRNGKey,
        training_config: Optional[TrainingConfig] = None,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> SolverResult:
        """Paper-faithful alternating stage training (Algorithm 3).

        Each stage:
          Phase 1: Cache forward trajectories (eq 7a) with current theta.
                   Train backward policy phi via loss (eq 18).
          Phase 2: Cache backward trajectories (eq 7b) with updated phi.
                   Train forward policy theta via loss (eq 19).
        """
        cfg = self.fbsde_config
        tc = training_config or TrainingConfig()

        # Initialize
        k_init, key = jax.random.split(key)
        params = self.init_params(k_init)
        fwd_opt = init_adam(params['forward'])
        bwd_opt = init_adam(params['backward'])

        loss_history = []
        checkpoint_paths = []
        global_step = 0

        for stage in range(cfg.num_stages):
            if self.config.verbose >= 1:
                print(f"\n=== Stage {stage + 1}/{cfg.num_stages} ===")

            # Phase 1: Train backward grad log Psi_hat
            # Cache forward trajectories from current forward policy theta
            k_traj, key = jax.random.split(key)
            k_s, k_sim = jax.random.split(k_traj)
            x0 = self.problem.sample_source(k_s, cfg.trajectory_batch_size)
            forward_trajs = self._simulate_forward_sde(params, k_sim, x0)

            if self.config.verbose >= 1:
                print(f"  Phase 1: Training backward policy "
                      f"({cfg.steps_per_stage} steps)...")

            for step in range(cfg.steps_per_stage):
                k_data, k_loss, key = jax.random.split(key, 3)

                x_batch, t_batch = self._sample_trajectory_data(
                    forward_trajs, k_data, tc.batch_size
                )

                # Eq (18): trainable = backward, frozen = forward.
                params['backward'], bwd_opt, metrics = self._sb_backward_update_jit(
                    params['backward'],
                    bwd_opt,
                    params['forward'],
                    x_batch,
                    t_batch,
                    k_loss,
                )

                loss_val = float(metrics['loss'])
                loss_history.append(loss_val)

                if self.config.verbose >= 2 and step % 500 == 0:
                    print(f"    step {step}: bwd_loss={loss_val:.4f}  "
                          f"cross={float(metrics['cross_term']):.4f}  "
                          f"div={float(metrics['divergence']):.4f}")

                if callback:
                    callback(global_step, {
                        **metrics, 'phase': 'backward', 'stage': stage
                    })
                global_step += 1

                checkpoint_path = self._maybe_save_checkpoint(
                    tc,
                    step=global_step,
                    params=params,
                    opt_state={'forward': fwd_opt, 'backward': bwd_opt},
                    loss_history=loss_history,
                    metrics={**metrics, 'phase': 'backward', 'stage': stage},
                )
                if checkpoint_path is not None:
                    checkpoint_paths.append(checkpoint_path)

            # Phase 2: Train forward grad log Psi
            # Cache backward trajectories from updated backward policy phi
            k_traj, key = jax.random.split(key)
            k_s, k_sim = jax.random.split(k_traj)
            xT = self.problem.sample_target(k_s, cfg.trajectory_batch_size)
            backward_trajs = self._simulate_backward_sde(params, k_sim, xT)

            if self.config.verbose >= 1:
                print(f"  Phase 2: Training forward policy "
                      f"({cfg.steps_per_stage} steps)...")

            for step in range(cfg.steps_per_stage):
                k_data, k_loss, key = jax.random.split(key, 3)

                x_batch, t_batch = self._sample_trajectory_data(
                    backward_trajs, k_data, tc.batch_size
                )

                # Eq (19): trainable = forward, frozen = backward.
                params['forward'], fwd_opt, metrics = self._sb_forward_update_jit(
                    params['forward'],
                    fwd_opt,
                    params['backward'],
                    x_batch,
                    t_batch,
                    k_loss,
                )

                loss_val = float(metrics['loss'])
                loss_history.append(loss_val)

                if self.config.verbose >= 2 and step % 500 == 0:
                    print(f"    step {step}: fwd_loss={loss_val:.4f}  "
                          f"cross={float(metrics['cross_term']):.4f}  "
                          f"div={float(metrics['divergence']):.4f}")

                if callback:
                    callback(global_step, {
                        **metrics, 'phase': 'forward', 'stage': stage
                    })
                global_step += 1

                checkpoint_path = self._maybe_save_checkpoint(
                    tc,
                    step=global_step,
                    params=params,
                    opt_state={'forward': fwd_opt, 'backward': bwd_opt},
                    loss_history=loss_history,
                    metrics={**metrics, 'phase': 'forward', 'stage': stage},
                )
                if checkpoint_path is not None:
                    checkpoint_paths.append(checkpoint_path)

            # - Stage summary -
            if self.config.verbose >= 1:
                window = min(200, len(loss_history))
                avg = sum(loss_history[-window:]) / window
                print(f"  Stage {stage + 1} complete - "
                      f"avg loss (last {window}): {avg:.6f}")

        # Store trained params
        self._params = params
        self._is_trained = True

        # Run diagnostics
        k_diag, _ = jax.random.split(key)
        diagnostics = self._run_diagnostics(k_diag, params)
        metadata = {
            'converged': True,
            'final_step': global_step,
            'solver_type': self.solver_type.name,
            'method': 'sb_fbsde',
            'num_stages': cfg.num_stages,
        }
        final_checkpoint_path = self._maybe_save_checkpoint(
            tc,
            step=global_step,
            params=params,
            opt_state={'forward': fwd_opt, 'backward': bwd_opt},
            loss_history=loss_history,
            metrics={'loss': loss_history[-1] if loss_history else 0.0},
            final=True,
            metadata=metadata,
        )
        if final_checkpoint_path is not None:
            checkpoint_paths.append(final_checkpoint_path)
            metadata['checkpoint_path'] = final_checkpoint_path
        if checkpoint_paths:
            metadata['checkpoint_paths'] = checkpoint_paths

        return SolverResult(
            params=params,
            loss_history=jnp.array(loss_history),
            diagnostics=diagnostics,
            metadata=metadata,
        )

    # Legacy training (deep_bsde / soc) - backward compatible
    def _solve_forward_sde_legacy(
        self, key: PRNGKey, params: Params, x0: Array,
    ) -> Tuple[Array, Array, Array]:
        """Forward SDE returning (X_traj, Z_traj, dW_traj) for deep_bsde/soc.

        Uses proper diffusion geometry: drift = f + G*z, noise = sigma*dW.
        """
        times = self.problem.time_grid.times
        dt = self.problem.time_grid.dt
        num_steps = self.problem.time_grid.num_steps
        keys = jax.random.split(key, num_steps)
        batch_size = x0.shape[0]

        def step_fn(x, inputs):
            t, step_key = inputs
            f = self.problem.reference.drift(x, t)
            g = self.problem.reference.diffusion(x, t)
            z = self._z_fn(params, x, jnp.full((batch_size,), t))

            # Controlled drift: f + G*z where G = sigma*sigma^T
            drift = f + _apply_GG(g, z)

            # Noise: sigma*dW
            dW_raw = jax.random.normal(step_key, x.shape) * jnp.sqrt(dt)
            noise = _apply_g(g, dW_raw)

            x_next = x + drift * dt + noise
            return x_next, (x_next, z, dW_raw)

        _, (X, Z, dW) = jax.lax.scan(step_fn, x0, (times[:-1], keys))
        z0 = self._z_fn(params, x0, jnp.zeros(batch_size))
        X = jnp.concatenate([x0[None], X], axis=0)
        Z = jnp.concatenate([z0[None], Z], axis=0)
        return (jnp.transpose(X, (1, 0, 2)),
                jnp.transpose(Z, (1, 0, 2)),
                jnp.transpose(dW, (1, 0, 2)))

    def _terminal_cost(self, x_T: Array, target_samples: Array) -> Array:
        """Approximate -log p_target(X_T) via nearest-neighbor.

        Handles both scalar and matrix diffusion coefficients.
        """
        dists_sq = jnp.sum((x_T[:, None] - target_samples[None, :]) ** 2, axis=-1)
        min_dist_sq = jnp.min(dists_sq, axis=-1)
        sigma = self.problem.reference.diffusion(None, 1.0)
        sigma = jnp.asarray(sigma)
        if jnp.ndim(sigma) == 0:
            return min_dist_sq / (2 * sigma ** 2)
        else:
            # For matrix diffusion, use Frobenius norm as scale
            return min_dist_sq / 2.0

    def _compute_loss_legacy(
        self, params: Params, key: PRNGKey, x0: Array, x1: Array,
    ) -> Tuple[Scalar, Dict[str, Scalar]]:
        """deep_bsde or soc loss (legacy, backward compatible)."""
        k1, _ = jax.random.split(key)
        batch_size = x0.shape[0]
        X_traj, Z_traj, dW_traj = self._solve_forward_sde_legacy(k1, params, x0)
        times = self.problem.time_grid.times
        dt = self.problem.time_grid.dt
        n_steps = len(times) - 1

        if self.fbsde_config.method == 'deep_bsde':
            X_T = X_traj[:, -1]
            g_XT = self._terminal_cost(X_T, x1)
            Y = self._y_fn(params, x0, jnp.zeros(batch_size))
            for i in range(n_steps):
                z_t, dW_t = Z_traj[:, i], dW_traj[:, i]
                Y = Y - 0.5 * jnp.sum(z_t ** 2, axis=-1) * dt + jnp.sum(z_t * dW_t, axis=-1)
            terminal_loss = jnp.mean((Y - g_XT) ** 2)
            dists = jnp.sum((X_T[:, None] - x1[None, :]) ** 2, axis=-1)
            endpoint_loss = jnp.mean(jnp.min(dists, axis=-1))
            control_cost = jnp.mean(jnp.sum(Z_traj ** 2, axis=-1))
            loss = (self.fbsde_config.terminal_weight * terminal_loss
                    + self.fbsde_config.running_weight * control_cost
                    + endpoint_loss)
            return loss, {'loss': loss, 'terminal_loss': terminal_loss,
                         'endpoint_loss': endpoint_loss, 'control_cost': control_cost}
        else:  # soc
            X_T = X_traj[:, -1]
            running = 0.5 * jnp.mean(jnp.sum(Z_traj ** 2, axis=-1)) * dt * len(times)
            dists = jnp.sum((X_T[:, None] - x1[None, :]) ** 2, axis=-1)
            terminal = jnp.mean(jnp.min(dists, axis=-1))
            loss = (self.fbsde_config.running_weight * running
                    + self.fbsde_config.terminal_weight * terminal)
            return loss, {'loss': loss, 'running_cost': running, 'terminal_cost': terminal}

    @partial(jax.jit, static_argnums=(0, 4))
    def _legacy_train_step_jit(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: AdamState,
        batch_size: int,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        k1, k2, k3 = jax.random.split(key, 3)
        x0 = self.problem.sample_source(k1, batch_size)
        x1 = self.problem.sample_target(k2, batch_size)
        (_, metrics), grads = jax.value_and_grad(
            self._compute_loss_legacy, has_aux=True
        )(params, k3, x0, x1)
        new_params, new_opt = adam_update(
            opt_state, grads, params, lr=self.fbsde_config.learning_rate
        )
        return new_params, new_opt, metrics

    def train_step(
        self, key: PRNGKey, params: Params, opt_state: AdamState, batch_size: int,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        """Single training step (deep_bsde / soc only)."""
        return self._legacy_train_step_jit(key, params, opt_state, batch_size)

    # Main train() dispatch
    def train(
        self,
        key: PRNGKey,
        training_config: Optional[TrainingConfig] = None,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> SolverResult:
        """Train the solver. Dispatches to method-specific training."""
        if self.fbsde_config.method == 'sb_fbsde':
            return self._train_sb_fbsde(key, training_config, callback)
        else:
            return super().train(key, training_config, callback)

    # Drift extraction (inference)
    def extract_drift(self, params: Params) -> DriftFn:
        """Extract forward drift: b*(x,t) = f + G*grad log Psi.

        Transports mu_0 -> mu_1. Handles both scalar and matrix diffusion.
        """
        fwd_factory = self._forward_factory
        key = 'forward' if 'forward' in params else 'z'
        fwd_params = params[key]

        def drift(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))
            f = self.problem.reference.drift(x, t)
            g = self.problem.reference.diffusion(x, t)
            b = fwd_factory.forward(fwd_params, x, t_arr)
            return f + _apply_GG(g, b)
        return drift

    def extract_backward_drift(self, params: Params) -> DriftFn:
        """Extract backward drift: f - G*grad log Psi_hat.

        For backward integration (T->0), transports mu_1 -> mu_0.
        Requires method='sb_fbsde' (needs backward policy).
        """
        if 'backward' not in params:
            raise ValueError("No backward policy. Use method='sb_fbsde'.")
        bwd_factory = self._backward_factory
        bwd_params = params['backward']

        def drift(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))
            f = self.problem.reference.drift(x, t)
            g = self.problem.reference.diffusion(x, t)
            b_hat = bwd_factory.forward(bwd_params, x, t_arr)
            return f - _apply_GG(g, b_hat)
        return drift

    def extract_score(self, params: Params) -> Callable[[Array, Scalar], Array]:
        """grad log p_t^SB = grad log Psi + grad log Psi_hat  (factorization principle).

        This is the time-dependent score of the SB marginal density.
        Clean because networks output b = grad log Psi directly.
        Requires method='sb_fbsde'.
        """
        if 'forward' not in params or 'backward' not in params:
            raise ValueError("Score requires both policies (method='sb_fbsde').")
        ff, bf = self._forward_factory, self._backward_factory
        fp, bp = params['forward'], params['backward']

        def score(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))
            return ff.forward(fp, x, t_arr) + bf.forward(bp, x, t_arr)
        return score

    # Langevin corrector (Sec.3.3, Algorithm 4)
    def _langevin_corrector_step(
        self, params: Params, x: Array, t: Scalar, key: PRNGKey,
    ) -> Array:
        """One Langevin corrector step (Algorithm 4).

        Uses both policies via factorization:
            score ~= grad log Psi + grad log Psi_hat

        Step size from eq (59): eps = 2r^2||eps_noise||^2 / ||sigma*score||^2
        Then: x' = x + eps*G*score + sqrt(2eps)*sigma*noise
        """
        batch = x.shape[0]
        t_b = jnp.full((batch,), t)
        g = self.problem.reference.diffusion(x, t)

        score = (self._forward_fn(params, x, t_b) +
                 self._backward_fn(params, x, t_b))

        eps = jax.random.normal(key, x.shape)

        # sigma*score and sigma*noise for step size computation
        g_score = _apply_g(g, score)
        score_norm_sq = jnp.sum(g_score ** 2) + 1e-8
        eps_norm_sq = jnp.sum(eps ** 2)

        r = self.fbsde_config.corrector_snr
        step_size = 2.0 * r ** 2 * eps_norm_sq / score_norm_sq

        return x + step_size * _apply_GG(g, score) + jnp.sqrt(2 * step_size) * _apply_g(g, eps)

    # Solution access
    def solve_fbsde(
        self, key: PRNGKey, params: Params, x0: Array,
    ) -> FBSDESolution:
        """Solve forward and return all components (backward-compatible)."""
        if self.fbsde_config.method == 'sb_fbsde':
            X = self._simulate_forward_sde(params, key, x0)
            times = self.problem.time_grid.times
            batch = x0.shape[0]
            Z = jnp.stack([
                self._forward_fn(params, X[:, i], jnp.full(batch, times[i]))
                for i in range(len(times))
            ], axis=1)
            Y = jnp.zeros((batch, len(times)))
            return FBSDESolution(X=X, Y=Y, Z=Z, times=times)
        else:
            X, Z, _ = self._solve_forward_sde_legacy(key, params, x0)
            times = self.problem.time_grid.times
            batch = x0.shape[0]
            Y = jnp.stack([
                self._y_fn(params, X[:, i], jnp.full(batch, times[i]))
                for i in range(len(times))
            ], axis=1)
            return FBSDESolution(X=X, Y=Y, Z=Z, times=times)

    def solve_sb_fbsde(
        self, key: PRNGKey, params: Params, num_samples: int = 256,
    ) -> SBFBSDESolution:
        """Solve full two-policy system (sb_fbsde only).

        Returns both forward and backward trajectories with policy evaluations.
        """
        if self.fbsde_config.method != 'sb_fbsde':
            raise ValueError("solve_sb_fbsde requires method='sb_fbsde'.")
        k1, k2, k3, k4 = jax.random.split(key, 4)
        times = self.problem.time_grid.times
        n = num_samples

        x0 = self.problem.sample_source(k1, n)
        X_fwd = self._simulate_forward_sde(params, k2, x0)

        xT = self.problem.sample_target(k3, n)
        X_bwd = self._simulate_backward_sde(params, k4, xT)

        fwd_p = jnp.stack([
            self._forward_fn(params, X_fwd[:, i], jnp.full(n, times[i]))
            for i in range(len(times))
        ], axis=1)
        bwd_p = jnp.stack([
            self._backward_fn(params, X_bwd[:, i], jnp.full(n, times[i]))
            for i in range(len(times))
        ], axis=1)

        return SBFBSDESolution(
            X_forward=X_fwd, X_backward=X_bwd,
            forward_policy=fwd_p, backward_policy=bwd_p, times=times,
        )
