"""Iterative Proportional Fitting (IPF) Solver for Schrödinger Bridges.

IPF alternates between:
1. Forward pass: Fix backward drift, learn forward drift to match target
2. Backward pass: Fix forward drift, learn backward drift to match source

This is the classical approach to SB, extending discrete Sinkhorn to continuous time.

Mathematical foundation:
- At convergence, the forward and backward drifts satisfy the SB optimality conditions
- The process alternately projects onto the forward and backward marginal constraints

Reference:
    Schrödinger (1931) original formulation
    Fortet (1940) IPF procedure
    De Bortoli et al. (2021) continuous-time neural IPF
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    TimeGrid,
    TrajectoryBatch,
)
from ..core.problem import SBProblem
from ..networks import (
    TimeConditionedMLPConfig,
    init_time_conditioned_mlp,
    time_conditioned_mlp_forward,
    init_adam,
    adam_update,
    AdamState,
)
from ..integrators import EulerMaruyama, sample_brownian_bridge
from .base import SBSolver


@dataclass
class IPFConfig:
    """Configuration for IPF solver.
    
    Attributes:
        hidden_dims: Network hidden dimensions.
        learning_rate: Learning rate.
        num_ipf_iterations: Number of forward-backward iterations.
        steps_per_iteration: Training steps per IPF iteration.
        use_warm_start: Warm start from previous iteration.
    """
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    learning_rate: float = 1e-4
    num_ipf_iterations: int = 10
    steps_per_iteration: int = 1000
    use_warm_start: bool = True


class IPFSolver(SBSolver):
    """Iterative Proportional Fitting solver.
    
    IPF works by alternating:
    1. Sample backward trajectories from target using current backward model
    2. Train forward model to match these trajectories (velocity matching)
    3. Sample forward trajectories from source using current forward model
    4. Train backward model to match these trajectories
    
    This converges to the Schrödinger Bridge when both marginal constraints are satisfied.
    
    Representation: Uses drift correction parameterization
        b*(x,t) = b_ref(x,t) + σ² · drift_correction(x,t)
    """
    
    def __init__(
        self,
        problem: SBProblem,
        ipf_config: Optional[IPFConfig] = None,
        config: Optional[Union[IPFConfig, SolverConfig]] = None,
        solver_config: Optional[SolverConfig] = None,
        **kwargs,
    ):
        """Initialize IPF solver.
        
        Args:
            problem: SB problem specification.
            ipf_config: IPF-specific configuration.
            config: Can be either IPFConfig or SolverConfig (for convenience).
            solver_config: Base solver configuration (explicit).
            **kwargs: Additional arguments for base class.
        
        Examples:
            # All these work:
            solver = IPFSolver(problem, ipf_config=IPFConfig(...))
            solver = IPFSolver(problem, config=IPFConfig(...))
            solver = IPFSolver(problem, IPFConfig(...))
        """
        # Handle config parameter flexibility
        if ipf_config is None and config is not None:
            if isinstance(config, IPFConfig):
                ipf_config = config
                config = None
        
        # Filter kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if not isinstance(v, IPFConfig)}
        
        # Determine base class config
        base_config = None
        if solver_config is not None:
            base_config = solver_config
        elif config is not None and isinstance(config, SolverConfig):
            base_config = config
        
        if base_config is not None:
            filtered_kwargs['config'] = base_config
            
        super().__init__(problem, **filtered_kwargs)
        self.ipf_config = ipf_config or IPFConfig()
        
        # Separate networks for forward and backward
        self._forward_params: Optional[Params] = None
        self._backward_params: Optional[Params] = None
        self._forward_opt: Optional[AdamState] = None
        self._backward_opt: Optional[AdamState] = None
        self._current_direction: str = 'forward'
        self._ipf_iteration: int = 0
    
    @property
    def solver_type(self) -> SolverType:
        return SolverType.IPF
    
    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.CONTROL  # Drift correction
    
    def init_params(self, key: PRNGKey) -> Params:
        """Initialize both forward and backward network parameters."""
        k1, k2 = jax.random.split(key)
        
        config = TimeConditionedMLPConfig(
            input_dim=self.problem.dim,
            output_dim=self.problem.dim,
            hidden_dims=self.ipf_config.hidden_dims,
        )
        
        self._forward_params = init_time_conditioned_mlp(k1, config)
        self._backward_params = init_time_conditioned_mlp(k2, config)
        
        self._forward_opt = init_adam(self._forward_params)
        self._backward_opt = init_adam(self._backward_params)
        
        # Return forward params as "main" params for compatibility
        return self._forward_params
    
    def _get_forward_drift(self, params: Params) -> DriftFn:
        """Get forward drift using current forward parameters."""
        def drift(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))
            
            b_ref = self.problem.reference.drift(x, t)
            sigma = self.problem.reference.diffusion(x, t)
            correction = time_conditioned_mlp_forward(params, x, t_arr)
            
            return b_ref + sigma ** 2 * correction
        return drift
    
    def _get_backward_drift(self, params: Params) -> DriftFn:
        """Get backward drift using current backward parameters."""
        def drift(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))
            
            b_ref = self.problem.reference.drift(x, t)
            sigma = self.problem.reference.diffusion(x, t)
            correction = time_conditioned_mlp_forward(params, x, t_arr)
            
            # Backward drift is negative of forward
            return -b_ref + sigma ** 2 * correction
        return drift
    
    def _sample_forward_trajectories(
        self,
        key: PRNGKey,
        batch_size: int,
    ) -> TrajectoryBatch:
        """Sample trajectories forward from source."""
        k1, k2 = jax.random.split(key)
        x0 = self.problem.sample_source(k1, batch_size)
        
        drift = self._get_forward_drift(self._forward_params)
        
        def diffusion(x, t):
            return self.problem.reference.diffusion(x, t)
        
        return self.integrator.integrate(
            k2, x0, self.problem.time_grid, drift, diffusion, True
        )
    
    def _sample_backward_trajectories(
        self,
        key: PRNGKey,
        batch_size: int,
    ) -> TrajectoryBatch:
        """Sample trajectories backward from target."""
        k1, k2 = jax.random.split(key)
        x1 = self.problem.sample_target(k1, batch_size)
        
        # Backward drift for reverse-time SDE
        backward_drift = self._get_backward_drift(self._backward_params)
        
        def diffusion(x, t):
            return self.problem.reference.diffusion(x, t)
        
        return self.integrator.integrate_backward(
            k2, x1, self.problem.time_grid, backward_drift, diffusion, True
        )
    
    def _velocity_matching_loss(
        self,
        params: Params,
        trajectory: TrajectoryBatch,
        direction: str,
    ) -> Tuple[Scalar, Dict[str, Scalar]]:
        """Velocity matching loss for training.
        
        Train drift to match the empirical velocity along sampled trajectories.
        """
        paths = trajectory.paths  # [batch, time, dim]
        times = trajectory.times
        dt = times[1] - times[0]
        
        batch_size, num_times, dim = paths.shape
        
        loss = 0.0
        count = 0
        
        for i in range(num_times - 1):
            x_t = paths[:, i, :]
            x_next = paths[:, i + 1, :]
            t = times[i]
            
            # Empirical velocity
            v_empirical = (x_next - x_t) / dt
            
            # Predicted drift
            t_batch = jnp.full((batch_size,), t)
            
            if direction == 'forward':
                drift = self._get_forward_drift(params)
            else:
                drift = self._get_backward_drift(params)
            
            v_pred = drift(x_t, t)
            
            # MSE
            loss = loss + jnp.mean((v_pred - v_empirical) ** 2)
            count += 1
        
        loss = loss / count
        
        return loss, {'loss': loss}
    
    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: Any,
        batch_size: int,
    ) -> Tuple[Params, Any, Dict[str, Scalar]]:
        """Perform one IPF training step."""
        k1, k2 = jax.random.split(key)
        
        if self._current_direction == 'forward':
            # Sample backward trajectories using backward model
            trajectories = self._sample_backward_trajectories(k1, batch_size)
            
            # Train forward model to match
            loss_fn = lambda p: self._velocity_matching_loss(p, trajectories, 'forward')
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self._forward_params)
            
            self._forward_params, self._forward_opt = adam_update(
                self._forward_opt, grads, self._forward_params,
                lr=self.ipf_config.learning_rate,
            )
            
            metrics['direction'] = 'forward'
            return self._forward_params, self._forward_opt, metrics
            
        else:  # backward
            # Sample forward trajectories using forward model
            trajectories = self._sample_forward_trajectories(k1, batch_size)
            
            # Train backward model to match
            loss_fn = lambda p: self._velocity_matching_loss(p, trajectories, 'backward')
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self._backward_params)
            
            self._backward_params, self._backward_opt = adam_update(
                self._backward_opt, grads, self._backward_params,
                lr=self.ipf_config.learning_rate,
            )
            
            metrics['direction'] = 'backward'
            return self._backward_params, self._backward_opt, metrics
    
    def train(self, key: PRNGKey, training_config=None, callback=None):
        """Train using IPF iterations."""
        from ..core.types import TrainingConfig
        config = training_config or TrainingConfig(
            num_iterations=self.ipf_config.steps_per_iteration,
        )
        
        k1, key = jax.random.split(key)
        self.init_params(k1)
        
        all_losses = []
        
        for ipf_iter in range(self.ipf_config.num_ipf_iterations):
            self._ipf_iteration = ipf_iter
            
            if self.config.verbose >= 1:
                print(f"\n=== IPF Iteration {ipf_iter + 1}/{self.ipf_config.num_ipf_iterations} ===")
            
            # Forward phase
            self._current_direction = 'forward'
            for step in range(self.ipf_config.steps_per_iteration):
                key, step_key = jax.random.split(key)
                _, _, metrics = self.train_step(step_key, None, None, config.batch_size)
                all_losses.append(float(metrics['loss']))
                
                if self.config.verbose >= 1 and step % 200 == 0:
                    print(f"  Forward step {step}: loss = {metrics['loss']:.6f}")
            
            # Backward phase
            self._current_direction = 'backward'
            for step in range(self.ipf_config.steps_per_iteration):
                key, step_key = jax.random.split(key)
                _, _, metrics = self.train_step(step_key, None, None, config.batch_size)
                all_losses.append(float(metrics['loss']))
                
                if self.config.verbose >= 1 and step % 200 == 0:
                    print(f"  Backward step {step}: loss = {metrics['loss']:.6f}")
        
        self._is_trained = True
        self._params = self._forward_params
        
        # Run diagnostics
        diagnostics = self._run_diagnostics(key, self._params)
        
        from ..core.types import SolverResult
        return SolverResult(
            params=self._forward_params,
            loss_history=jnp.array(all_losses),
            diagnostics=diagnostics,
            metadata={
                'forward_params': self._forward_params,
                'backward_params': self._backward_params,
                'ipf_iterations': self.ipf_config.num_ipf_iterations,
            },
        )
    
    def extract_drift(self, params: Params) -> DriftFn:
        """Extract forward drift."""
        return self._get_forward_drift(params)
