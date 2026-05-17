"""Mirror Descent IPF (MD-IPF) solver for Schrodinger bridges.

This solver uses the mirror-descent view of Sinkhorn/IPF for entropic OT:

    min_pi <C, pi> + eps * KL(pi || mu otimes nu)
    subject to pi in Pi(mu_0, mu_1)

Mirror-descent update (KL geometry):

    pi_{k+1} = argmin_pi { <grad f(pi_k), pi> + (1/eta) * D_psi(pi || pi_k) }

For eta = 1 this recovers standard Sinkhorn projections. For eta < 1 this
gives damped updates, which can improve stability on hard problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

import jax
import jax.numpy as jnp

from ..core.diffusion import apply_diffusion
from ..core.types import (
    Array,
    DriftFn,
    Params,
    PRNGKey,
    RepresentationType,
    Scalar,
    SolverConfig,
    SolverType,
    SolverResult,
    DiagnosticReport,
    TrajectoryBatch,
    TrainingConfig,
)
from ..core.problem import SBProblem
from ..networks import (
    init_time_conditioned_mlp,
    time_conditioned_mlp_forward,
    TimeConditionedMLPConfig,
    init_adam,
    adam_update,
    AdamState,
)
from .base import SBSolver


class MDVariant(Enum):
    """Mirror descent algorithm variants."""
    STANDARD = "standard"          # eta = 1, standard Sinkhorn
    DAMPED = "damped"              # eta < 1, damped updates
    AVERAGED = "averaged"          # Iterate averaging (Polyak-Ruppert)
    ACCELERATED = "accelerated"    # Nesterov-style momentum in KL geometry
    ADAPTIVE = "adaptive"          # Adaptive step size


@dataclass
class MirrorDescentIPFConfig:
    """Configuration for Mirror Descent IPF solver.

    Sinkhorn can be interpreted as mirror descent with KL divergence.
    The step size eta controls damping of updates in log-space.

    Attributes:
        variant: Which MD variant to use
        step_size: Mirror descent step size (eta).
            - eta = 1.0: Standard Sinkhorn
            - eta < 1.0: Damped/conservative updates (recommended: 0.5-0.9)
        num_md_iterations: Number of mirror descent (Sinkhorn) iterations
        regularization: Entropic regularization eps for OT
        use_log_domain: Compute in log-domain for numerical stability
        averaging_start: When to start iterate averaging (0 = from beginning)
        momentum: Momentum coefficient for accelerated variant
        adaptive_beta: Adaptivity parameter for step size adjustment
        convergence_threshold: Stop when marginal error below this
        hidden_dims: Neural network hidden dimensions (for drift learning)
        learning_rate: Learning rate for neural network training
        steps_per_md_iteration: Training steps per MD iteration
    """
    variant: MDVariant = MDVariant.DAMPED
    step_size: float = 0.8  # Damped by default for stability
    num_md_iterations: int = 20
    regularization: float = 0.1
    use_log_domain: bool = True
    averaging_start: int = 5
    momentum: float = 0.9
    adaptive_beta: float = 0.5
    convergence_threshold: float = 1e-6
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    learning_rate: float = 1e-4
    steps_per_md_iteration: int = 500


class MirrorDescentIPFSolver(SBSolver):
    """Mirror Descent IPF solver for Schrodinger bridges.

    Each iteration alternates:
    1) Sinkhorn/mirror-descent update for the coupling.
    2) Velocity matching to train a continuous drift model.
    """
    
    def __init__(
        self,
        problem: SBProblem,
        md_config: Optional[MirrorDescentIPFConfig] = None,
        config: Optional[Union[MirrorDescentIPFConfig, SolverConfig]] = None,
        solver_config: Optional[SolverConfig] = None,
        **kwargs,
    ):
        """Initialize Mirror Descent IPF solver.
        
        Args:
            problem: SB problem specification.
            md_config: MD-IPF specific configuration.
            config: Alternative config (can be MDIPFConfig or SolverConfig).
            solver_config: Base solver configuration.
            **kwargs: Additional arguments for base class.
        """
        # Handle config parameter flexibility
        if md_config is None and config is not None:
            if isinstance(config, MirrorDescentIPFConfig):
                md_config = config
                config = None
        
        # Filter kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if not isinstance(v, MirrorDescentIPFConfig)}
        
        # Determine base class config
        base_config = None
        if solver_config is not None:
            base_config = solver_config
        elif config is not None and isinstance(config, SolverConfig):
            base_config = config
        
        if base_config is not None:
            filtered_kwargs['config'] = base_config
        
        super().__init__(problem, **filtered_kwargs)
        self.md_config = md_config or MirrorDescentIPFConfig()
        
        # State for MD iterations
        self._sinkhorn_f: Optional[Array] = None  # Log-domain potential
        self._sinkhorn_g: Optional[Array] = None  # Log-domain potential
        self._averaged_f: Optional[Array] = None  # Averaged potentials
        self._averaged_g: Optional[Array] = None
        self._md_iteration: int = 0
        self._momentum_f: Optional[Array] = None  # For accelerated variant
        self._momentum_g: Optional[Array] = None
    
    @property
    def solver_type(self) -> SolverType:
        return SolverType.IPF  # We're still fundamentally IPF
    
    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.CONTROL
    
    def init_params(self, key: PRNGKey) -> Params:
        """Initialize drift network and Sinkhorn potentials."""
        k1, k2 = jax.random.split(key)
        
        # Neural network for drift
        config = TimeConditionedMLPConfig(
            input_dim=self.problem.dim,
            output_dim=self.problem.dim,
            hidden_dims=self.md_config.hidden_dims,
        )
        drift_params = init_time_conditioned_mlp(k1, config)
        
        return {'drift': drift_params}
    
    def _compute_cost_matrix(self, x0: Array, x1: Array) -> Array:
        """Compute squared Euclidean cost matrix.
        
        C[i,j] = ||x0[i] - x1[j]||^2
        """
        return jnp.sum((x0[:, None, :] - x1[None, :, :]) ** 2, axis=-1)
    
    def _log_sinkhorn_step(
        self,
        f: Array,
        g: Array,
        C: Array,
        eps: float,
    ) -> Tuple[Array, Array]:
        """Single Sinkhorn step in log-domain.

        For uniform marginals:
            f <- -eps * logsumexp_j((-C_ij + g_j) / eps) - eps * log(n)
            g <- -eps * logsumexp_i((-C_ij + f_i) / eps) - eps * log(n)
        """
        n = C.shape[0]
        log_n = jnp.log(n)
        
        # f update: f_i = -eps * logsumexp_j((-C_ij + g_j)/eps) - eps*log(n)
        M_f = (-C + g[None, :]) / eps
        f_new = -eps * jax.scipy.special.logsumexp(M_f, axis=1) - eps * log_n
        
        # g update: g_j = -eps * logsumexp_i((-C_ij + f_i)/eps) - eps*log(n)  
        M_g = (-C + f_new[:, None]) / eps
        g_new = -eps * jax.scipy.special.logsumexp(M_g, axis=0) - eps * log_n
        
        return f_new, g_new
    
    def _mirror_descent_update(
        self,
        f_old: Array,
        g_old: Array,
        f_new: Array,
        g_new: Array,
        step_size: float,
    ) -> Tuple[Array, Array]:
        """Apply damped mirror descent update.
        """
        f_updated = step_size * f_new + (1 - step_size) * f_old
        g_updated = step_size * g_new + (1 - step_size) * g_old
        return f_updated, g_updated
    
    def _accelerated_update(
        self,
        f_old: Array,
        g_old: Array,
        f_new: Array,
        g_new: Array,
        momentum_f: Array,
        momentum_g: Array,
        step_size: float,
        momentum: float,
    ) -> Tuple[Array, Array, Array, Array]:
        """Nesterov-style accelerated update in KL geometry.
        """
        # Update momentum buffers
        new_momentum_f = f_new - f_old
        new_momentum_g = g_new - g_old
        
        # Apply damped update with momentum extrapolation
        f_updated = f_new + momentum * momentum_f
        g_updated = g_new + momentum * momentum_g
        
        return f_updated, g_updated, new_momentum_f, new_momentum_g
    
    def _compute_marginal_error(
        self,
        f: Array,
        g: Array,
        C: Array,
        eps: float,
    ) -> float:
        """Compute marginal constraint violation.
        
        Returns max(||pi1 - uniform||_1, ||pi*1 - uniform||_1)
        """
        n = C.shape[0]
        
        # Compute coupling pi = diag(exp(f/eps)) K diag(exp(g/eps))
        log_P = (f[:, None] + g[None, :] - C) / eps
        log_P = log_P - jax.scipy.special.logsumexp(log_P)  # Normalize
        P = jnp.exp(log_P)
        
        # Marginal errors
        row_marginal = P.sum(axis=1)
        col_marginal = P.sum(axis=0)
        uniform = jnp.ones(n) / n
        
        row_error = jnp.abs(row_marginal - uniform).sum()
        col_error = jnp.abs(col_marginal - uniform).sum()
        
        return float(max(row_error, col_error))
    
    def _run_mirror_descent_sinkhorn(
        self,
        x0: Array,
        x1: Array,
    ) -> Tuple[Array, Array, List[float]]:
        """Run mirror descent (Sinkhorn) iterations.
        """
        n = x0.shape[0]
        eps = self.md_config.regularization
        eta = self.md_config.step_size
        variant = self.md_config.variant
        
        # Compute cost matrix
        C = self._compute_cost_matrix(x0, x1)
        
        # Initialize potentials
        f = jnp.zeros(n)
        g = jnp.zeros(n)
        
        # For averaging
        f_avg = jnp.zeros(n)
        g_avg = jnp.zeros(n)
        avg_count = 0
        
        # For acceleration
        momentum_f = jnp.zeros(n)
        momentum_g = jnp.zeros(n)
        
        errors = []
        
        for k in range(self.md_config.num_md_iterations):
            # Standard Sinkhorn step
            f_new, g_new = self._log_sinkhorn_step(f, g, C, eps)
            
            # Apply variant-specific update
            if variant == MDVariant.STANDARD:
                f, g = f_new, g_new
                
            elif variant == MDVariant.DAMPED:
                f, g = self._mirror_descent_update(f, g, f_new, g_new, eta)
                
            elif variant == MDVariant.ACCELERATED:
                f, g, momentum_f, momentum_g = self._accelerated_update(
                    f, g, f_new, g_new, momentum_f, momentum_g,
                    eta, self.md_config.momentum
                )
                
            elif variant == MDVariant.AVERAGED:
                # Apply standard update first
                f, g = f_new, g_new
                # Then update average
                if k >= self.md_config.averaging_start:
                    avg_count += 1
                    alpha = 1.0 / avg_count
                    f_avg = (1 - alpha) * f_avg + alpha * f
                    g_avg = (1 - alpha) * g_avg + alpha * g
                    
            elif variant == MDVariant.ADAPTIVE:
                # Adaptive step size based on progress
                error = self._compute_marginal_error(f_new, g_new, C, eps)
                # Increase step size if making good progress
                if len(errors) > 0 and error < errors[-1]:
                    eta = min(1.0, eta / self.md_config.adaptive_beta)
                else:
                    eta = max(0.1, eta * self.md_config.adaptive_beta)
                f, g = self._mirror_descent_update(f, g, f_new, g_new, eta)
            
            # Track convergence
            error = self._compute_marginal_error(f, g, C, eps)
            errors.append(error)
            
            if error < self.md_config.convergence_threshold:
                break
        
        # Return averaged if using that variant
        if variant == MDVariant.AVERAGED and avg_count > 0:
            return f_avg, g_avg, errors
        
        return f, g, errors
    
    def _sample_from_coupling(
        self,
        key: PRNGKey,
        x0: Array,
        x1: Array,
        f: Array,
        g: Array,
    ) -> Tuple[Array, Array]:
        """Sample coupled pairs from Sinkhorn coupling.
        
        Returns (x0_coupled, x1_coupled) where pairs are drawn
        from the entropic OT coupling pi_eps.
        """
        n = x0.shape[0]
        eps = self.md_config.regularization
        C = self._compute_cost_matrix(x0, x1)
        
        # Coupling in log-domain
        log_P = (f[:, None] + g[None, :] - C) / eps
        P = jax.nn.softmax(log_P.flatten()).reshape(n, n)
        
        # Sample pairs
        flat_idx = jax.random.choice(key, n * n, shape=(n,), p=P.flatten())
        i_idx = flat_idx // n
        j_idx = flat_idx % n
        
        return x0[i_idx], x1[j_idx]
    
    def _sample_bridge_path(
        self,
        key: PRNGKey,
        x0: Array,
        x1: Array,
        t: Array,
    ) -> Tuple[Array, Array]:
        """Sample from Brownian bridge conditional and compute target velocity.
        
        Bridge: X_t | X_0=x0, X_1=x1 ~ N(mu_t, sigma_t^2)
        where mu_t = (1-t)x0 + t*x1, sigma_t = sigma*sqrt(t(1-t))
        
        Target velocity (OT direction): v = x1 - x0
        """
        sigma = self.problem.reference.diffusion(x0, t)
        t_col = t[:, None]
        
        # Bridge mean and std
        mu_t = (1 - t_col) * x0 + t_col * x1
        bridge_scale = jnp.sqrt(t_col * (1 - t_col) + 1e-6)
        
        # Sample
        noise = jax.random.normal(key, x0.shape)
        x_t = mu_t + bridge_scale * apply_diffusion(
            sigma,
            noise,
            is_scalar_diffusion=self.problem.reference.is_diffusion_scalar,
        )
        
        # OT velocity
        v_target = x1 - x0
        
        return x_t, v_target
    
    def _velocity_matching_loss(
        self,
        params: Params,
        key: PRNGKey,
        x0_coupled: Array,
        x1_coupled: Array,
    ) -> Tuple[Scalar, Dict[str, Scalar]]:
        """Velocity matching loss for drift learning.
        
        L = E_{t, x_t} [||v_theta(x_t, t) - (x1 - x0)||^2]
        
        This trains the drift network to predict the OT velocity.
        """
        batch_size = x0_coupled.shape[0]
        k1, k2 = jax.random.split(key)
        
        # Sample time
        t = jax.random.uniform(k1, (batch_size,), minval=0.01, maxval=0.99)
        
        # Sample bridge point and get target
        x_t, v_target = self._sample_bridge_path(k2, x0_coupled, x1_coupled, t)
        
        # Predict velocity
        v_pred = time_conditioned_mlp_forward(params['drift'], x_t, t)
        
        # MSE loss
        loss = jnp.mean(jnp.sum((v_pred - v_target) ** 2, axis=-1))
        
        return loss, {'loss': loss}

    @partial(jax.jit, static_argnums=0)
    def _velocity_update_jit(
        self,
        params: Params,
        opt_state: AdamState,
        key: PRNGKey,
        x0_coupled: Array,
        x1_coupled: Array,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        (_, metrics), grads = jax.value_and_grad(
            self._velocity_matching_loss, has_aux=True
        )(params, key, x0_coupled, x1_coupled)

        new_params, new_opt_state = adam_update(
            opt_state, grads, params,
            lr=self.md_config.learning_rate
        )
        return new_params, new_opt_state, metrics
    
    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: AdamState,
        batch_size: int,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        """Single training step with Sinkhorn coupling."""
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Sample source and target
        x0 = self.problem.sample_source(k1, batch_size)
        x1 = self.problem.sample_target(k2, batch_size)
        
        # Run mirror descent Sinkhorn
        f, g, errors = self._run_mirror_descent_sinkhorn(x0, x1)
        
        # Sample from coupling
        x0_coupled, x1_coupled = self._sample_from_coupling(k3, x0, x1, f, g)
        
        # Velocity matching update
        new_params, new_opt_state, metrics = self._velocity_update_jit(
            params,
            opt_state,
            k4,
            x0_coupled,
            x1_coupled,
        )
        
        # Add Sinkhorn metrics
        metrics['sinkhorn_error'] = errors[-1] if errors else 0.0
        metrics['sinkhorn_iterations'] = len(errors)
        
        return new_params, new_opt_state, metrics
    
    def train(
        self,
        key: PRNGKey,
        training_config=None,
        callback=None,
    ) -> SolverResult:
        """Train MD-IPF solver.
        
        The training alternates between:
        1. Mirror descent (Sinkhorn) to refine coupling
        2. Neural network training to learn continuous drift
        """
        k1, k2 = jax.random.split(key)
        
        # Initialize
        params = self.init_params(k1)
        opt_state = self._init_optimizer(params)
        config = training_config or TrainingConfig()
        
        all_losses = []
        all_sinkhorn_errors = []
        checkpoint_paths = []
        global_step = 0
        
        total_steps = (self.md_config.num_md_iterations * 
                       self.md_config.steps_per_md_iteration)
        
        if self.config.verbose >= 1:
            print(f"=== Mirror Descent IPF ({self.md_config.variant.value}) ===")
            print(f"Step size eta = {self.md_config.step_size}")
            print(f"Regularization eps = {self.md_config.regularization}")
        
        for md_iter in range(self.md_config.num_md_iterations):
            self._md_iteration = md_iter
            
            if self.config.verbose >= 1:
                print(f"\n--- MD Iteration {md_iter + 1}/{self.md_config.num_md_iterations} ---")
            
            for step in range(self.md_config.steps_per_md_iteration):
                k2, step_key = jax.random.split(k2)
                
                params, opt_state, metrics = self.train_step(
                    step_key, params, opt_state, config.batch_size
                )
                
                all_losses.append(float(metrics['loss']))
                all_sinkhorn_errors.append(float(metrics['sinkhorn_error']))
                global_step += 1

                checkpoint_path = self._maybe_save_checkpoint(
                    config,
                    step=global_step,
                    params=params,
                    opt_state=opt_state,
                    loss_history=all_losses,
                    metrics=metrics,
                    metadata={'md_iteration': md_iter},
                )
                if checkpoint_path is not None:
                    checkpoint_paths.append(checkpoint_path)
                
                if self.config.verbose >= 1 and step % 100 == 0:
                    print(f"  Step {step}: loss={metrics['loss']:.6f}, "
                          f"sinkhorn_err={metrics['sinkhorn_error']:.6f}")
        
        self._params = params
        self._is_trained = True
        
        # Diagnostics
        diagnostics = self._run_diagnostics(key, params)
        metadata = {
            'converged': True,
            'solver_type': 'MIRROR_DESCENT_IPF',
            'variant': self.md_config.variant.value,
            'step_size': self.md_config.step_size,
            'sinkhorn_errors': all_sinkhorn_errors,
        }
        final_checkpoint_path = self._maybe_save_checkpoint(
            config,
            step=global_step,
            params=params,
            opt_state=opt_state,
            loss_history=all_losses,
            metrics={'loss': all_losses[-1] if all_losses else 0.0},
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
            loss_history=jnp.array(all_losses),
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def _checkpoint_state(self) -> Dict[str, Any]:
        return {
            'sinkhorn_f': self._sinkhorn_f,
            'sinkhorn_g': self._sinkhorn_g,
            'averaged_f': self._averaged_f,
            'averaged_g': self._averaged_g,
            'md_iteration': self._md_iteration,
            'momentum_f': self._momentum_f,
            'momentum_g': self._momentum_g,
        }

    def _restore_checkpoint_state(self, state: Dict[str, Any]) -> None:
        self._sinkhorn_f = state.get('sinkhorn_f')
        self._sinkhorn_g = state.get('sinkhorn_g')
        self._averaged_f = state.get('averaged_f')
        self._averaged_g = state.get('averaged_g')
        self._md_iteration = state.get('md_iteration', 0)
        self._momentum_f = state.get('momentum_f')
        self._momentum_g = state.get('momentum_g')
    
    def extract_drift(self, params: Params) -> DriftFn:
        """Extract learned drift function."""
        def drift(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))
            
            # Reference drift
            b_ref = self.problem.reference.drift(x, t)
            
            # Learned velocity
            v = time_conditioned_mlp_forward(params['drift'], x, t_arr)
            
            return b_ref + v
        
        return drift


# Convenience functions
def create_md_ipf_solver(
    problem: SBProblem,
    variant: str = 'damped',
    step_size: float = 0.8,
    **kwargs,
) -> MirrorDescentIPFSolver:
    """Create MD-IPF solver with convenient defaults.
    
    Args:
        problem: SB problem
        variant: 'standard', 'damped', 'averaged', 'accelerated', 'adaptive'
        step_size: Mirror descent step size (eta)
        **kwargs: Additional config options
    
    Returns:
        Configured MD-IPF solver
    
    Example:
        solver = create_md_ipf_solver(problem, variant='damped', step_size=0.7)
        result = solver.train(key)
    """
    variant_enum = MDVariant(variant)
    config = MirrorDescentIPFConfig(
        variant=variant_enum,
        step_size=step_size,
        **kwargs,
    )
    return MirrorDescentIPFSolver(problem, md_config=config)


# Module exports
__all__ = [
    'MDVariant',
    'MirrorDescentIPFConfig',
    'MirrorDescentIPFSolver',
    'create_md_ipf_solver',
]
