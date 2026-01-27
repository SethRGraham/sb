"""Forward-Backward SDE Solver for Schrödinger Bridges.

The SB can be characterized via a system of coupled FBSDEs:

Forward SDE:
    dX_t = [b(X_t, t) + σ²(t) Z_t] dt + σ(t) dW_t,  X_0 ~ μ_0

Backward SDE (value function):
    dY_t = -f(X_t, Z_t, t) dt + Z_t · dW_t,  Y_T = g(X_T)

The optimal control is u* = σ Z.

Reference:
    Chen et al. "Likelihood Training of Schrödinger Bridge using 
    Forward-Backward SDEs Theory" (ICLR 2022)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union

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


class FBSDESolution(NamedTuple):
    """Solution of the FBSDE system."""
    X: Array  # Forward process [batch, time, dim]
    Y: Array  # Value function [batch, time]
    Z: Array  # Control [batch, time, dim]
    times: Array


@dataclass
class FBSDEConfig:
    """Configuration for FBSDE solver."""
    hidden_dims: Tuple[int, ...] = (256, 256)
    time_embed_dim: int = 64
    learning_rate: float = 1e-4
    terminal_weight: float = 1.0
    running_weight: float = 0.01
    method: str = 'deep_bsde'  # 'deep_bsde' or 'soc'


class FBSDESolver(SBSolver):
    """FBSDE-based Schrödinger Bridge solver.
    
    Learns the control Z(x,t) directly via:
    1. Deep BSDE: Match backward equation + terminal condition
    2. SOC: Minimize the stochastic optimal control cost
    
    The drift is: b*(x,t) = b_ref(x,t) + σ²(t) Z(x,t)
    
    Attributes:
        fbsde_config: FBSDE-specific configuration.
    """
    
    def __init__(
        self,
        problem: SBProblem,
        fbsde_config: Optional[FBSDEConfig] = None,
        config: Optional[Union[FBSDEConfig, SolverConfig]] = None,
        solver_config: Optional[SolverConfig] = None,
        **kwargs,
    ):
        """Initialize FBSDE solver.
        
        Args:
            problem: SB problem specification.
            fbsde_config: FBSDE-specific configuration.
            config: Can be either FBSDEConfig or SolverConfig (for convenience).
            solver_config: Base solver configuration (explicit).
            **kwargs: Additional arguments for base class.
        
        Examples:
            # All these work:
            solver = FBSDESolver(problem, fbsde_config=FBSDEConfig(...))
            solver = FBSDESolver(problem, config=FBSDEConfig(...))
            solver = FBSDESolver(problem, FBSDEConfig(...))
            solver = FBSDESolver(problem, fbsde_config=FBSDEConfig(...), 
                                solver_config=SolverConfig(verbose=2))
        """
        # Handle config parameter flexibility
        # If fbsde_config not provided, check if config is an FBSDEConfig
        if fbsde_config is None and config is not None:
            if isinstance(config, FBSDEConfig):
                fbsde_config = config
                config = None  # Don't pass FBSDEConfig to base class
        
        # Filter kwargs to remove any FBSDEConfig that might have been passed
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if not isinstance(v, FBSDEConfig)}
        
        # Determine what config to pass to base class
        base_config = None
        if solver_config is not None:
            base_config = solver_config
        elif config is not None and isinstance(config, SolverConfig):
            base_config = config
        
        if base_config is not None:
            filtered_kwargs['config'] = base_config
            
        super().__init__(problem, **filtered_kwargs)
        self.fbsde_config = fbsde_config or FBSDEConfig()
    
    @property
    def solver_type(self) -> SolverType:
        return SolverType.FBSDE
    
    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.CONTROL
    
    def init_params(self, key: PRNGKey) -> Params:
        """Initialize Z network (and optionally Y network)."""
        k1, k2 = jax.random.split(key)
        
        # Z network: outputs control
        z_config = TimeConditionedMLPConfig(
            input_dim=self.problem.dim,
            output_dim=self.problem.dim,
            hidden_dims=self.fbsde_config.hidden_dims,
            time_embed_dim=self.fbsde_config.time_embed_dim,
        )
        z_params = init_time_conditioned_mlp(k1, z_config)
        
        # Y network: outputs value function (scalar)
        y_config = TimeConditionedMLPConfig(
            input_dim=self.problem.dim,
            output_dim=1,
            hidden_dims=self.fbsde_config.hidden_dims,
            time_embed_dim=self.fbsde_config.time_embed_dim,
        )
        y_params = init_time_conditioned_mlp(k2, y_config)
        
        return {'z': z_params, 'y': y_params}
    
    def _z_fn(self, params: Params, x: Array, t: Array) -> Array:
        """Evaluate Z network (control)."""
        return time_conditioned_mlp_forward(params['z'], x, t)
    
    def _y_fn(self, params: Params, x: Array, t: Array) -> Array:
        """Evaluate Y network (value function)."""
        return time_conditioned_mlp_forward(params['y'], x, t).squeeze(-1)
    
    def _solve_forward_sde(
        self,
        key: PRNGKey,
        params: Params,
        x0: Array,
    ) -> Tuple[Array, Array, Array]:
        """Solve forward SDE with current control Z."""
        times = self.problem.time_grid.times
        dt = self.problem.time_grid.dt
        num_steps = self.problem.time_grid.num_steps
        
        keys = jax.random.split(key, num_steps)
        batch_size = x0.shape[0]
        
        def step_fn(x, inputs):
            t, step_key = inputs
            
            # Controlled drift
            ref_drift = self.problem.reference.drift(x, t)
            sigma = self.problem.reference.diffusion(x, t)
            z = self._z_fn(params, x, jnp.full((batch_size,), t))
            controlled_drift = ref_drift + sigma ** 2 * z
            
            # Sample noise
            dW = jax.random.normal(step_key, x.shape) * jnp.sqrt(dt)
            
            # Euler step
            x_next = x + controlled_drift * dt + sigma * dW
            
            return x_next, (x_next, z, dW)
        
        _, (X_traj, Z_traj, dW_traj) = jax.lax.scan(
            step_fn, x0, (times[:-1], keys)
        )
        
        # Prepend initial
        z0 = self._z_fn(params, x0, jnp.zeros(batch_size))
        X_traj = jnp.concatenate([x0[None], X_traj], axis=0)
        Z_traj = jnp.concatenate([z0[None], Z_traj], axis=0)
        
        # Transpose to [batch, time, dim]
        X_traj = jnp.transpose(X_traj, (1, 0, 2))
        Z_traj = jnp.transpose(Z_traj, (1, 0, 2))
        dW_traj = jnp.transpose(dW_traj, (1, 0, 2))
        
        return X_traj, Z_traj, dW_traj
    
    def _terminal_cost(self, x_T: Array, target_samples: Array) -> Array:
        """Compute terminal cost g(X_T).
        
        Approximates -log p_target(X_T) using nearest neighbor distance.
        """
        # Distance to nearest target sample
        dists_sq = jnp.sum((x_T[:, None] - target_samples[None, :]) ** 2, axis=-1)
        min_dist_sq = jnp.min(dists_sq, axis=-1)
        
        sigma = self.problem.reference.diffusion(None, 1.0)
        return min_dist_sq / (2 * sigma ** 2)
    
    def _compute_loss(
        self,
        params: Params,
        key: PRNGKey,
        x0: Array,
        x1: Array,
    ) -> Tuple[Scalar, Dict[str, Scalar]]:
        """Compute FBSDE loss."""
        k1, k2 = jax.random.split(key)
        batch_size = x0.shape[0]
        
        # Solve forward SDE
        X_traj, Z_traj, dW_traj = self._solve_forward_sde(k1, params, x0)
        
        times = self.problem.time_grid.times
        dt = self.problem.time_grid.dt
        num_time_steps = len(times) - 1  # Number of steps (not time points)
        
        if self.fbsde_config.method == 'deep_bsde':
            # Deep BSDE loss: terminal matching + BSDE consistency
            
            # Terminal cost
            X_T = X_traj[:, -1]
            g_X_T = self._terminal_cost(X_T, x1)
            
            # Propagate Y using BSDE
            Y_0 = self._y_fn(params, x0, jnp.zeros(batch_size))
            Y_current = Y_0
            
            # FIX: dW_traj has shape [batch, num_steps, dim] where num_steps = len(times) - 1
            # The loop iterates num_time_steps times, and dW_traj[:, i] is always valid
            for i in range(num_time_steps):
                z_t = Z_traj[:, i]
                dW_t = dW_traj[:, i]  # Always valid: i ranges [0, num_steps-1]
                
                # Running cost: (1/2)|z|²
                f_t = 0.5 * jnp.sum(z_t ** 2, axis=-1)
                
                # BSDE step
                Y_current = Y_current - f_t * dt + jnp.sum(z_t * dW_t, axis=-1)
            
            Y_T = Y_current
            
            # Terminal matching loss
            terminal_loss = jnp.mean((Y_T - g_X_T) ** 2)
            
            # Endpoint loss
            dists_sq = jnp.sum((X_T[:, None] - x1[None, :]) ** 2, axis=-1)
            endpoint_loss = jnp.mean(jnp.min(dists_sq, axis=-1))
            
            # Control regularization
            control_cost = jnp.mean(jnp.sum(Z_traj ** 2, axis=-1))
            
            loss = (
                self.fbsde_config.terminal_weight * terminal_loss +
                self.fbsde_config.running_weight * control_cost +
                endpoint_loss
            )
            
            metrics = {
                'loss': loss,
                'terminal_loss': terminal_loss,
                'endpoint_loss': endpoint_loss,
                'control_cost': control_cost,
            }
            
        else:  # SOC method
            # Direct stochastic optimal control
            X_T = X_traj[:, -1]
            
            # Running cost
            running_cost = 0.5 * jnp.mean(jnp.sum(Z_traj ** 2, axis=-1)) * dt * len(times)
            
            # Terminal cost
            dists_sq = jnp.sum((X_T[:, None] - x1[None, :]) ** 2, axis=-1)
            terminal_cost = jnp.mean(jnp.min(dists_sq, axis=-1))
            
            loss = (
                self.fbsde_config.running_weight * running_cost +
                self.fbsde_config.terminal_weight * terminal_cost
            )
            
            metrics = {
                'loss': loss,
                'running_cost': running_cost,
                'terminal_cost': terminal_cost,
            }
        
        return loss, metrics
    
    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: AdamState,
        batch_size: int,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        """Perform one training step."""
        k1, k2, k3 = jax.random.split(key, 3)
        
        x0 = self.problem.sample_source(k1, batch_size)
        x1 = self.problem.sample_target(k2, batch_size)
        
        (loss, metrics), grads = jax.value_and_grad(
            self._compute_loss, has_aux=True
        )(params, k3, x0, x1)
        
        new_params, new_opt_state = adam_update(
            opt_state, grads, params,
            lr=self.fbsde_config.learning_rate,
        )
        
        return new_params, new_opt_state, metrics
    
    def extract_drift(self, params: Params) -> DriftFn:
        """Extract forward drift from Z network."""
        def drift(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))
            
            ref_drift = self.problem.reference.drift(x, t)
            sigma = self.problem.reference.diffusion(x, t)
            z = self._z_fn(params, x, t_arr)
            
            return ref_drift + sigma ** 2 * z
        
        return drift
    
    def solve_fbsde(
        self,
        key: PRNGKey,
        params: Params,
        x0: Array,
    ) -> FBSDESolution:
        """Solve the full FBSDE system and return all components."""
        X_traj, Z_traj, _ = self._solve_forward_sde(key, params, x0)
        
        times = self.problem.time_grid.times
        batch_size = x0.shape[0]
        
        # Compute Y along trajectory
        Y_traj = []
        for i in range(len(times)):
            x_t = X_traj[:, i]
            t = times[i]
            y_t = self._y_fn(params, x_t, jnp.full(batch_size, t))
            Y_traj.append(y_t)
        
        Y_traj = jnp.stack(Y_traj, axis=1)
        
        return FBSDESolution(
            X=X_traj,
            Y=Y_traj,
            Z=Z_traj,
            times=times,
        )
