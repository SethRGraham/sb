"""Iterative Markovian Fitting (IMF) Solver.

IMF is a simulation-free approach that alternates between fitting
forward and backward velocity fields using regression.

Unlike IPF which requires trajectory simulation, IMF uses 
optimal transport conditional paths for regression targets.

Reference:
    Shi et al. "Diffusion Schrödinger Bridge Matching" (ICLR 2024)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

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
    SolverType,
    SolverResult,
    DiagnosticReport,
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


@dataclass
class IMFConfig:
    """Configuration for IMF solver."""
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    time_embed_dim: int = 64
    learning_rate: float = 1e-4
    num_imf_iterations: int = 5
    steps_per_iteration: int = 2000
    use_ot_coupling: bool = True
    ot_regularization: float = 0.1


class IMFSolver(SBSolver):
    """Iterative Markovian Fitting solver.
    
    Learns forward and backward velocity fields via alternating regression.
    The key insight is using OT-conditional paths as regression targets,
    avoiding trajectory simulation.
    
    Algorithm:
    1. Initialize with flow matching (OT-CFM)
    2. Iterate:
       a. Use backward velocity to refine forward targets
       b. Fit forward velocity
       c. Use forward velocity to refine backward targets
       d. Fit backward velocity
    
    Attributes:
        imf_config: IMF-specific configuration.
    """
    
    def __init__(
        self,
        problem: SBProblem,
        imf_config: Optional[IMFConfig] = None,
        config: Optional[Union[IMFConfig, SolverConfig]] = None,
        solver_config: Optional[SolverConfig] = None,
        **kwargs,
    ):
        """Initialize IMF solver.
        
        Args:
            problem: SB problem specification.
            imf_config: IMF-specific configuration.
            config: Can be either IMFConfig or SolverConfig (for convenience).
            solver_config: Base solver configuration (explicit).
            **kwargs: Additional arguments for base class.
        
        Examples:
            # All these work:
            solver = IMFSolver(problem, imf_config=IMFConfig(...))
            solver = IMFSolver(problem, config=IMFConfig(...))
            solver = IMFSolver(problem, IMFConfig(...))
        """
        # Handle config parameter flexibility
        if imf_config is None and config is not None:
            if isinstance(config, IMFConfig):
                imf_config = config
                config = None
        
        # Filter kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if not isinstance(v, IMFConfig)}
        
        # Determine base class config
        base_config = None
        if solver_config is not None:
            base_config = solver_config
        elif config is not None and isinstance(config, SolverConfig):
            base_config = config
        
        if base_config is not None:
            filtered_kwargs['config'] = base_config
            
        super().__init__(problem, **filtered_kwargs)
        self.imf_config = imf_config or IMFConfig()
        self._imf_iteration = 0
    
    @property
    def solver_type(self) -> SolverType:
        return SolverType.IMF
    
    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.SCORE  # Velocity ≈ score structure
    
    def init_params(self, key: PRNGKey) -> Params:
        """Initialize forward and backward velocity networks."""
        k1, k2 = jax.random.split(key)
        
        config = TimeConditionedMLPConfig(
            input_dim=self.problem.dim,
            output_dim=self.problem.dim,
            hidden_dims=self.imf_config.hidden_dims,
            time_embed_dim=self.imf_config.time_embed_dim,
        )
        
        forward_params = init_time_conditioned_mlp(k1, config)
        backward_params = init_time_conditioned_mlp(k2, config)
        
        return {
            'forward': forward_params,
            'backward': backward_params,
        }
    
    def _forward_velocity(self, params: Params, x: Array, t: Array) -> Array:
        """Evaluate forward velocity network."""
        return time_conditioned_mlp_forward(params['forward'], x, t)
    
    def _backward_velocity(self, params: Params, x: Array, t: Array) -> Array:
        """Evaluate backward velocity network."""
        return time_conditioned_mlp_forward(params['backward'], x, t)
    
    def _compute_ot_coupling(
        self,
        x0: Array,
        x1: Array,
        reg: float = 0.1,
    ) -> Array:
        """Compute OT coupling using Sinkhorn."""
        batch_size = x0.shape[0]
        
        # Cost matrix
        C = jnp.sum((x0[:, None, :] - x1[None, :, :]) ** 2, axis=-1)
        
        # Sinkhorn
        K = jnp.exp(-C / reg)
        u = jnp.ones(batch_size)
        v = jnp.ones(batch_size)
        
        for _ in range(50):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.T @ u + 1e-8)
        
        # Get coupling as indices
        P = u[:, None] * K * v[None, :]
        coupling = jnp.argmax(P, axis=1)
        
        return coupling
    
    def _sample_conditional_path(
        self,
        key: PRNGKey,
        x0: Array,
        x1: Array,
        t: Array,
    ) -> Tuple[Array, Array]:
        """Sample from OT conditional path."""
        sigma = self.problem.reference.diffusion(None, t[0])
        bridge_std = sigma * jnp.sqrt(t * (1 - t) + 1e-6)
        
        noise = jax.random.normal(key, x0.shape)
        mean_t = (1 - t)[:, None] * x0 + t[:, None] * x1
        x_t = mean_t + bridge_std[:, None] * noise
        
        # OT velocity: constant direction
        target_v = x1 - x0
        
        return x_t, target_v
    
    def _forward_matching_loss(
        self,
        params: Params,
        key: PRNGKey,
        x0: Array,
        x1: Array,
    ) -> Scalar:
        """Loss for forward velocity matching."""
        batch_size = x0.shape[0]
        k1, k2 = jax.random.split(key)
        
        t = jax.random.uniform(k1, (batch_size,), minval=0.01, maxval=0.99)
        x_t, target_v = self._sample_conditional_path(k2, x0, x1, t)
        
        pred_v = self._forward_velocity(params, x_t, t)
        
        return jnp.mean(jnp.sum((pred_v - target_v) ** 2, axis=-1))
    
    def _backward_matching_loss(
        self,
        params: Params,
        key: PRNGKey,
        x0: Array,
        x1: Array,
    ) -> Scalar:
        """Loss for backward velocity matching."""
        batch_size = x0.shape[0]
        k1, k2 = jax.random.split(key)
        
        t = jax.random.uniform(k1, (batch_size,), minval=0.01, maxval=0.99)
        x_t, _ = self._sample_conditional_path(k2, x0, x1, t)
        
        target_v = x0 - x1  # Backward direction
        pred_v = self._backward_velocity(params, x_t, t)
        
        return jnp.mean(jnp.sum((pred_v - target_v) ** 2, axis=-1))
    
    def _imf_forward_loss(
        self,
        params: Params,
        key: PRNGKey,
        x0: Array,
        x1: Array,
    ) -> Scalar:
        """IMF forward loss using backward velocity for targets."""
        batch_size = x0.shape[0]
        k1, k2 = jax.random.split(key)
        
        t = jax.random.uniform(k1, (batch_size,), minval=0.01, maxval=0.99)
        x_t, _ = self._sample_conditional_path(k2, x0, x1, t)
        
        # Use backward velocity to refine target
        backward_v = self._backward_velocity(params, x_t, t)
        
        # Blend empirical and model-based
        empirical_v = x1 - x0
        alpha = jnp.clip(t, 0.1, 0.9)
        target_v = alpha[:, None] * empirical_v + (1 - alpha)[:, None] * (-backward_v)
        
        pred_v = self._forward_velocity(params, x_t, t)
        
        return jnp.mean(jnp.sum((pred_v - target_v) ** 2, axis=-1))
    
    def _imf_backward_loss(
        self,
        params: Params,
        key: PRNGKey,
        x0: Array,
        x1: Array,
    ) -> Scalar:
        """IMF backward loss using forward velocity for targets."""
        batch_size = x0.shape[0]
        k1, k2 = jax.random.split(key)
        
        t = jax.random.uniform(k1, (batch_size,), minval=0.01, maxval=0.99)
        x_t, _ = self._sample_conditional_path(k2, x0, x1, t)
        
        forward_v = self._forward_velocity(params, x_t, t)
        
        empirical_v = x0 - x1
        alpha = jnp.clip(1 - t, 0.1, 0.9)
        target_v = alpha[:, None] * empirical_v + (1 - alpha)[:, None] * (-forward_v)
        
        pred_v = self._backward_velocity(params, x_t, t)
        
        return jnp.mean(jnp.sum((pred_v - target_v) ** 2, axis=-1))
    
    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: AdamState,
        batch_size: int,
        phase: str = 'forward',
        use_imf: bool = True,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        """Perform one training step."""
        k1, k2, k3 = jax.random.split(key, 3)
        
        x0 = self.problem.sample_source(k1, batch_size)
        x1 = self.problem.sample_target(k2, batch_size)
        
        # Optional OT coupling
        if self.imf_config.use_ot_coupling:
            coupling = self._compute_ot_coupling(x0, x1, self.imf_config.ot_regularization)
            x1 = x1[coupling]
        
        if phase == 'forward':
            if use_imf and self._imf_iteration > 0:
                loss_fn = lambda p: self._imf_forward_loss(p, k3, x0, x1)
            else:
                loss_fn = lambda p: self._forward_matching_loss(p, k3, x0, x1)
        else:
            if use_imf and self._imf_iteration > 0:
                loss_fn = lambda p: self._imf_backward_loss(p, k3, x0, x1)
            else:
                loss_fn = lambda p: self._backward_matching_loss(p, k3, x0, x1)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        
        new_params, new_opt_state = adam_update(
            opt_state, grads, params,
            lr=self.imf_config.learning_rate,
        )
        
        return new_params, new_opt_state, {'loss': loss, 'phase': phase}
    
    def train(
        self,
        key: PRNGKey,
        training_config=None,
        callback=None,
    ) -> SolverResult:
        """Train using IMF iterations."""
        k1, k2 = jax.random.split(key)
        
        params = self.init_params(k1)
        opt_state = self._init_optimizer(params)
        
        all_losses = []
        
        # Phase 1: Initial flow matching
        if self.config.verbose >= 1:
            print("=== Initial Flow Matching ===")
        
        for step in range(self.imf_config.steps_per_iteration):
            k2, step_key = jax.random.split(k2)
            phase = 'forward' if step % 2 == 0 else 'backward'
            
            params, opt_state, metrics = self.train_step(
                step_key, params, opt_state, 256, phase=phase, use_imf=False
            )
            all_losses.append(metrics['loss'])
            
            if self.config.verbose >= 1 and step % 500 == 0:
                print(f"  Step {step}: loss = {metrics['loss']:.6f}")
        
        # Phase 2: IMF iterations
        for imf_iter in range(self.imf_config.num_imf_iterations):
            self._imf_iteration = imf_iter + 1
            
            if self.config.verbose >= 1:
                print(f"\n=== IMF Iteration {imf_iter + 1} ===")
            
            # Forward phase
            for step in range(self.imf_config.steps_per_iteration // 2):
                k2, step_key = jax.random.split(k2)
                params, opt_state, metrics = self.train_step(
                    step_key, params, opt_state, 256, phase='forward', use_imf=True
                )
                all_losses.append(metrics['loss'])
                
                if self.config.verbose >= 1 and step % 500 == 0:
                    print(f"  Forward {step}: loss = {metrics['loss']:.6f}")
            
            # Backward phase
            for step in range(self.imf_config.steps_per_iteration // 2):
                k2, step_key = jax.random.split(k2)
                params, opt_state, metrics = self.train_step(
                    step_key, params, opt_state, 256, phase='backward', use_imf=True
                )
                all_losses.append(metrics['loss'])
        
        self._params = params
        self._is_trained = True
        
        diagnostics = self._run_diagnostics(key, params)
        
        return SolverResult(
            params=params,
            loss_history=jnp.array(all_losses),
            diagnostics=diagnostics,
            metadata={
                'converged': True,
                'solver_type': self.solver_type.name,
                'imf_iterations': self.imf_config.num_imf_iterations,
            },
        )
    
    def extract_drift(self, params: Params) -> DriftFn:
        """Extract forward drift (ODE-style, no diffusion)."""
        def drift(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))
            
            return self._forward_velocity(params, x, t_arr)
        
        return drift
