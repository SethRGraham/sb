"""RKHS (Kernel-based) Schrödinger Bridge Solver.

A non-parametric approach using Reproducing Kernel Hilbert Spaces.
Does NOT require neural networks - uses kernel regression instead.

This solver represents the score/drift as a weighted sum of kernel gradients:
    s(x,t) = sum_i alpha_i(t) grad_x k(x, x_i)

Reference:
    Bunne et al. "Schrödinger Bridges Beat Diffusion Models on 
    Text-to-Speech Synthesis" (2023) - kernel methods section
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
    TrainingConfig,
)
from ..core.problem import SBProblem
from ..kernels import (
    gaussian_kernel,
    gaussian_kernel_gradient,
    median_heuristic,
    kernel_ridge_regression,
)
from .base import SBSolver


@dataclass
class RKHSConfig:
    """Configuration for RKHS solver."""
    bandwidth: Optional[float] = None  # Auto if None
    regularization: float = 1e-4
    num_inducing: int = 500  # Number of inducing points
    num_time_points: int = 20  # Discretization for time
    kernel_type: str = 'gaussian'


class RKHSSolver(SBSolver):
    """RKHS-based Schrödinger Bridge solver.
    
    Uses kernel methods to estimate the score function without
    neural networks. The solution is represented as:
    
        grad log p_t(x) ~= sum_i alpha_i(t) grad_x k(x, x_i)
    
    where {x_i} are inducing points and alpha_i(t) are learned coefficients.
    
    This is a closed-form solution at each time slice, making it
    fast to train but potentially limited in expressiveness.
    
    Attributes:
        rkhs_config: RKHS-specific configuration.
    """
    
    def __init__(
        self,
        problem: SBProblem,
        rkhs_config: Optional[RKHSConfig] = None,
        config: Optional[Union[RKHSConfig, SolverConfig]] = None,
        solver_config: Optional[SolverConfig] = None,
        **kwargs,
    ):
        """Initialize RKHS solver.
        
        Args:
            problem: SB problem specification.
            rkhs_config: RKHS-specific configuration.
            config: Can be either RKHSConfig or SolverConfig (for convenience).
            solver_config: Base solver configuration (explicit).
            **kwargs: Additional arguments for base class.
        
        Examples:
            # All these work:
            solver = RKHSSolver(problem, rkhs_config=RKHSConfig(...))
            solver = RKHSSolver(problem, config=RKHSConfig(...))
            solver = RKHSSolver(problem, RKHSConfig(...))
        """
        # Handle config parameter flexibility
        if rkhs_config is None and config is not None:
            if isinstance(config, RKHSConfig):
                rkhs_config = config
                config = None
        
        # Filter kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if not isinstance(v, RKHSConfig)}
        
        # Determine base class config
        base_config = None
        if solver_config is not None:
            base_config = solver_config
        elif config is not None and isinstance(config, SolverConfig):
            base_config = config
        
        if base_config is not None:
            filtered_kwargs['config'] = base_config
            
        super().__init__(problem, **filtered_kwargs)
        self.rkhs_config = rkhs_config or RKHSConfig()
        
        # Will be set during training
        self._inducing_points: Optional[Array] = None
        self._time_points: Optional[Array] = None
        self._coefficients: Optional[Array] = None
        self._bandwidth: Optional[float] = None
    
    @property
    def solver_type(self) -> SolverType:
        return SolverType.RKHS
    
    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.KERNEL
    
    @property
    def is_neural(self) -> bool:
        """RKHS solver does NOT use neural networks."""
        return False
    
    def init_params(self, key: PRNGKey) -> Params:
        """Initialize RKHS parameters (inducing points and coefficients)."""
        k1, k2 = jax.random.split(key)
        
        # Sample inducing points from source and target
        n_source = self.rkhs_config.num_inducing // 2
        n_target = self.rkhs_config.num_inducing - n_source
        
        source_pts = self.problem.sample_source(k1, n_source)
        target_pts = self.problem.sample_target(k2, n_target)
        
        inducing = jnp.concatenate([source_pts, target_pts], axis=0)
        
        # Time discretization
        time_points = jnp.linspace(0.01, 0.99, self.rkhs_config.num_time_points)
        
        # Initialize coefficients (will be solved in closed form)
        coeffs = jnp.zeros((self.rkhs_config.num_time_points, 
                           self.rkhs_config.num_inducing, 
                           self.problem.dim))
        
        # Compute bandwidth if not provided
        if self.rkhs_config.bandwidth is None:
            bandwidth = median_heuristic(inducing)
        else:
            bandwidth = self.rkhs_config.bandwidth
        
        return {
            'inducing_points': inducing,
            'time_points': time_points,
            'coefficients': coeffs,
            'bandwidth': bandwidth,
        }
    
    def _estimate_score_at_time(
        self,
        key: PRNGKey,
        t: float,
        x0_samples: Array,
        x1_samples: Array,
        inducing: Array,
        bandwidth: float,
    ) -> Array:
        """Estimate score coefficients at a single time point.
        
        Uses the bridge conditional to generate training data.
        """
        batch_size = x0_samples.shape[0]
        
        # Sample from bridge conditional at time t
        sigma = self.problem.reference.diffusion(None, t)
        bridge_std = sigma * jnp.sqrt(t * (1 - t))
        
        mean_t = (1 - t) * x0_samples + t * x1_samples
        noise = jax.random.normal(key, x0_samples.shape)
        x_t = mean_t + bridge_std * noise
        
        # True score at x_t (from bridge conditional)
        bridge_var = bridge_std ** 2 + 1e-8
        true_score = -(x_t - mean_t) / bridge_var
        
        # Kernel matrix: K(x_t, inducing)
        K = gaussian_kernel(x_t, inducing, bandwidth)  # [batch, num_inducing]
        
        # Solve kernel ridge regression for each dimension
        reg = self.rkhs_config.regularization
        K_reg = K.T @ K + reg * jnp.eye(K.shape[1])
        
        # Coefficients: alpha = (K^T K + lambdaI)^{-1} K^T y
        coeffs = jnp.linalg.solve(K_reg, K.T @ true_score)
        
        return coeffs
    
    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: Any,
        batch_size: int,
    ) -> Tuple[Params, Any, Dict[str, Scalar]]:
        """RKHS doesn't use iterative training - solve in closed form."""
        # This is a no-op for RKHS; actual solving happens in train()
        return params, opt_state, {'loss': 0.0}
    
    def train(
        self,
        key: PRNGKey,
        training_config=None,
        callback=None,
    ):
        """Solve the RKHS problem in closed form."""
        from ..core.types import SolverResult, DiagnosticReport
        config = training_config or TrainingConfig()
        
        k1, k2 = jax.random.split(key)
        
        # Initialize
        params = self.init_params(k1)
        
        inducing = params['inducing_points']
        time_points = params['time_points']
        bandwidth = params['bandwidth']
        
        self._inducing_points = inducing
        self._time_points = time_points
        self._bandwidth = bandwidth
        
        if self.config.verbose >= 1:
            print(f"RKHS Solver: {len(inducing)} inducing points, bandwidth={bandwidth:.4f}")
        
        # Sample training data
        batch_size = 2000
        x0_samples = self.problem.sample_source(k1, batch_size)
        x1_samples = self.problem.sample_target(k2, batch_size)
        
        # Solve for coefficients at each time point
        coefficients = []
        keys = jax.random.split(key, len(time_points))
        
        for i, t in enumerate(time_points):
            coeffs_t = self._estimate_score_at_time(
                keys[i], float(t), x0_samples, x1_samples, inducing, bandwidth
            )
            coefficients.append(coeffs_t)
            
            if self.config.verbose >= 1 and i % 5 == 0:
                print(f"  Solved time {t:.3f}")
        
        self._coefficients = jnp.stack(coefficients, axis=0)
        
        params['coefficients'] = self._coefficients
        self._params = params
        self._is_trained = True
        
        # Run diagnostics
        diagnostics = self._run_diagnostics(key, params)
        metadata = {
            'converged': True,
            'solver_type': self.solver_type.name,
            'bandwidth': bandwidth,
            'num_inducing': len(inducing),
        }
        checkpoint_paths = []
        final_checkpoint_path = self._maybe_save_checkpoint(
            config,
            step=0,
            params=params,
            opt_state=None,
            loss_history=[0.0],
            metrics={'loss': 0.0},
            final=True,
            metadata=metadata,
        )
        if final_checkpoint_path is not None:
            checkpoint_paths.append(final_checkpoint_path)
            metadata['checkpoint_path'] = final_checkpoint_path
            metadata['checkpoint_paths'] = checkpoint_paths
        
        return SolverResult(
            params=params,
            loss_history=jnp.array([0.0]),  # No iterative loss
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def _checkpoint_state(self) -> Dict[str, Any]:
        return {
            'inducing_points': self._inducing_points,
            'time_points': self._time_points,
            'coefficients': self._coefficients,
            'bandwidth': self._bandwidth,
        }

    def _restore_checkpoint_state(self, state: Dict[str, Any]) -> None:
        self._inducing_points = state.get('inducing_points')
        self._time_points = state.get('time_points')
        self._coefficients = state.get('coefficients')
        self._bandwidth = state.get('bandwidth')
        if self._params is not None:
            if self._inducing_points is None:
                self._inducing_points = self._params.get('inducing_points')
            if self._time_points is None:
                self._time_points = self._params.get('time_points')
            if self._coefficients is None:
                self._coefficients = self._params.get('coefficients')
            if self._bandwidth is None:
                self._bandwidth = self._params.get('bandwidth')
    
    def _interpolate_coefficients(self, t: float) -> Array:
        """Interpolate coefficients to arbitrary time."""
        time_points = self._time_points
        coefficients = self._coefficients
        
        # Find bracketing indices
        idx = jnp.searchsorted(time_points, t)
        idx = jnp.clip(idx, 1, len(time_points) - 1)
        
        t0, t1 = time_points[idx - 1], time_points[idx]
        c0, c1 = coefficients[idx - 1], coefficients[idx]
        
        # Linear interpolation
        alpha = (t - t0) / (t1 - t0 + 1e-8)
        return (1 - alpha) * c0 + alpha * c1
    
    def _score_fn(self, x: Array, t: Scalar) -> Array:
        """Evaluate score using kernel regression."""
        x = jnp.atleast_2d(x)
        
        # Handle JAX tracing - use jnp operations instead of float()
        t_scalar = jnp.asarray(t).reshape(())
        
        # Get interpolated coefficients using vectorized operations
        time_points = self._time_points
        
        # Find bracketing indices (differentiable)
        idx_float = jnp.sum(time_points <= t_scalar)
        idx = jnp.clip(idx_float, 1, len(time_points) - 1).astype(jnp.int32)
        idx_prev = idx - 1
        
        t0 = time_points[idx_prev]
        t1 = time_points[idx]
        c0 = self._coefficients[idx_prev]
        c1 = self._coefficients[idx]
        
        # Linear interpolation
        alpha = (t_scalar - t0) / (t1 - t0 + 1e-8)
        coeffs = (1 - alpha) * c0 + alpha * c1  # [num_inducing, dim]
        
        # Kernel gradient: grad_x k(x, x_i)
        K_grad = gaussian_kernel_gradient(
            x, self._inducing_points, self._bandwidth
        )  # [batch, num_inducing, dim]
        
        # Score: sum_i alpha_i grad_x k(x, x_i)
        score = jnp.einsum('ijk,jk->ik', K_grad, coeffs)
        
        return score
    
    def extract_drift(self, params: Params) -> DriftFn:
        """Extract forward drift from kernel representation."""
        def drift(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            
            ref_drift = self.problem.reference.drift(x, t)
            sigma = self.problem.reference.diffusion(x, t)
            score = self._score_fn(x, t)
            
            return ref_drift + sigma ** 2 * score
        
        return drift
    
    def get_score_fn(self) -> Callable:
        """Get the learned score function."""
        return self._score_fn
