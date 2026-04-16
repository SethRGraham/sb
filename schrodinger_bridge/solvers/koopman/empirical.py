"""Empirical Reference Dynamics for Data-Driven Schrödinger Bridges.

This module provides support for defining reference dynamics from empirical data
rather than closed-form SDEs. This is essential for applications like:
- Financial modeling (where true dynamics are unknown)
- Biological systems (complex, partially observed)
- Any domain where you have trajectory data but not equations

Key Classes:
===========
- EmpiricalReferenceDynamics: Reference process defined by trajectory data
- KernelDriftEstimator: Non-parametric drift estimation
- LocalLinearDrift: Locally linear drift approximation

The empirical reference integrates with Koopman methods, which can
extract eigenfunctions directly from the data.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import vmap

# Type aliases
Array = jnp.ndarray
PRNGKey = jax.Array
Scalar = Union[float, Array]
DriftFn = Callable[[Array, Scalar], Array]
DiffusionFn = Callable[[Array, Scalar], Union[Scalar, Array]]


# =============================================================================
# Empirical Reference Dynamics
# =============================================================================

class EmpiricalReferenceDynamics:
    """Reference dynamics estimated from empirical trajectory data.
    
    Instead of specifying drift b(x,t) and diffusion σ(x,t) analytically,
    we estimate them from observed trajectories.
    
    Estimation Methods:
    - Drift: Local averaging, kernel regression, or neural network
    - Diffusion: Local variance estimation or quadratic variation
    
    Attributes:
        trajectories: Historical trajectory data, shape [num_traj, num_times, dim].
        dt: Time step between observations.
        drift_estimator: Method for drift estimation.
        diffusion_estimator: Method for diffusion estimation.
    """
    
    def __init__(
        self,
        trajectories: Array,
        dt: float,
        drift_method: str = 'kernel',
        diffusion_method: str = 'local_variance',
        kernel_bandwidth: Optional[float] = None,
        regularization: float = 1e-6,
    ):
        """Initialize from trajectory data.
        
        Args:
            trajectories: Shape [num_traj, num_times, dim].
            dt: Time step.
            drift_method: 'kernel', 'local_linear', or 'knn'.
            diffusion_method: 'local_variance', 'quadratic_variation', or 'constant'.
            kernel_bandwidth: Bandwidth for kernel methods (auto if None).
            regularization: Regularization for matrix inversions.
        """
        self.trajectories = trajectories
        self.dt = dt
        self._dim = trajectories.shape[2]
        self.drift_method = drift_method
        self.diffusion_method = diffusion_method
        self.regularization = regularization
        
        # Extract state-derivative pairs
        self._extract_data()
        
        # Set bandwidth
        if kernel_bandwidth is None:
            self.bandwidth = self._compute_bandwidth()
        else:
            self.bandwidth = kernel_bandwidth
        
        # Estimate constant diffusion if using that method
        if diffusion_method == 'constant':
            self._sigma_constant = self._estimate_constant_diffusion()
        
        # Pre-compute for efficiency
        self._precompute()
    
    def _extract_data(self):
        """Extract (x, dx/dt) pairs from trajectories."""
        num_traj, num_times, dim = self.trajectories.shape
        
        X_list = []
        dX_list = []
        T_list = []
        
        times = jnp.linspace(0, 1, num_times)
        
        for traj_idx in range(num_traj):
            for t_idx in range(num_times - 1):
                X_list.append(self.trajectories[traj_idx, t_idx])
                dX = self.trajectories[traj_idx, t_idx + 1] - self.trajectories[traj_idx, t_idx]
                dX_list.append(dX / self.dt)
                T_list.append(times[t_idx])
        
        self._X_data = jnp.stack(X_list)  # [N, dim]
        self._dX_data = jnp.stack(dX_list)  # [N, dim]
        self._T_data = jnp.stack(T_list)  # [N]
        self._num_samples = len(X_list)
    
    def _compute_bandwidth(self) -> float:
        """Compute bandwidth using median heuristic."""
        # Subsample for efficiency
        n_sub = min(1000, self._num_samples)
        indices = jnp.linspace(0, self._num_samples - 1, n_sub).astype(int)
        X_sub = self._X_data[indices]
        
        # Pairwise distances
        dists_sq = jnp.sum((X_sub[:, None, :] - X_sub[None, :, :]) ** 2, axis=-1)
        
        # Median of non-zero distances
        mask = dists_sq > 0
        median_dist = jnp.sqrt(jnp.median(dists_sq[mask]))
        
        return float(median_dist)
    
    def _estimate_constant_diffusion(self) -> float:
        """Estimate constant diffusion from quadratic variation."""
        # dX² / dt ≈ σ² for small dt
        dX_squared = jnp.sum(self._dX_data ** 2, axis=-1) * self.dt
        return float(jnp.sqrt(jnp.mean(dX_squared)))
    
    def _precompute(self):
        """Precompute quantities for fast inference."""
        if self.drift_method == 'kernel':
            # Precompute kernel matrix for Nadaraya-Watson
            pass  # Done on-the-fly for memory efficiency
        elif self.drift_method == 'local_linear':
            # Could precompute local regression coefficients
            pass
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def is_time_homogeneous(self) -> bool:
        return False  # Empirical dynamics are generally time-dependent
    
    @property
    def is_diffusion_scalar(self) -> bool:
        return self.diffusion_method == 'constant'
    
    def drift(self, x: Array, t: Scalar) -> Array:
        """Estimate drift at (x, t) using the chosen method."""
        x = jnp.atleast_2d(x)
        batch_size = x.shape[0]
        
        if self.drift_method == 'kernel':
            return self._kernel_drift(x, t)
        elif self.drift_method == 'local_linear':
            return self._local_linear_drift(x, t)
        elif self.drift_method == 'knn':
            return self._knn_drift(x, t)
        else:
            raise ValueError(f"Unknown drift method: {self.drift_method}")
    
    def _kernel_drift(self, x: Array, t: Scalar) -> Array:
        """Nadaraya-Watson kernel regression for drift.
        
        b(x) = Σ_i K(x, x_i) (dx/dt)_i / Σ_i K(x, x_i)
        """
        # Kernel weights: K(x, x_i) = exp(-||x - x_i||² / (2h²))
        # Shape: [batch, num_samples]
        diff = x[:, None, :] - self._X_data[None, :, :]  # [batch, N, dim]
        dist_sq = jnp.sum(diff ** 2, axis=-1)  # [batch, N]
        
        weights = jnp.exp(-dist_sq / (2 * self.bandwidth ** 2))
        
        # Normalize weights
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-10)
        
        # Weighted average of derivatives
        drift = jnp.einsum('bn,nd->bd', weights, self._dX_data)
        
        return drift
    
    def _local_linear_drift(self, x: Array, t: Scalar) -> Array:
        """Locally weighted linear regression for drift.
        
        At each query point, fit: dx/dt = a + Bx locally.
        """
        batch_size = x.shape[0]
        
        def fit_single(x_query):
            # Kernel weights
            diff = self._X_data - x_query  # [N, dim]
            dist_sq = jnp.sum(diff ** 2, axis=-1)
            weights = jnp.exp(-dist_sq / (2 * self.bandwidth ** 2))
            
            # Weighted least squares: [1, x] @ [a; B] = dx/dt
            # Design matrix
            ones = jnp.ones((self._num_samples, 1))
            Phi = jnp.hstack([ones, self._X_data])  # [N, 1+dim]
            
            # Weighted normal equations
            W = jnp.diag(weights)
            PhiTW = Phi.T @ W
            A = PhiTW @ Phi + self.regularization * jnp.eye(1 + self._dim)
            b = PhiTW @ self._dX_data
            
            # Solve for coefficients
            coeffs = jnp.linalg.solve(A, b)  # [1+dim, dim]
            
            # Predict at query point
            query_features = jnp.concatenate([jnp.ones(1), x_query])
            return query_features @ coeffs
        
        return vmap(fit_single)(x)
    
    def _knn_drift(self, x: Array, t: Scalar, k: int = 20) -> Array:
        """K-nearest neighbors drift estimation."""
        batch_size = x.shape[0]
        
        def knn_single(x_query):
            # Compute distances
            dist_sq = jnp.sum((self._X_data - x_query) ** 2, axis=-1)
            
            # Find k nearest
            _, indices = jax.lax.top_k(-dist_sq, k)  # Negative for smallest
            
            # Average derivatives of neighbors
            return jnp.mean(self._dX_data[indices], axis=0)
        
        return vmap(knn_single)(x)
    
    def diffusion(self, x: Array, t: Scalar) -> Union[Scalar, Array]:
        """Estimate diffusion at (x, t)."""
        if self.diffusion_method == 'constant':
            return self._sigma_constant
        elif self.diffusion_method == 'local_variance':
            return self._local_variance_diffusion(x, t)
        elif self.diffusion_method == 'quadratic_variation':
            return self._quadratic_variation_diffusion(x, t)
        else:
            raise ValueError(f"Unknown diffusion method: {self.diffusion_method}")
    
    def _local_variance_diffusion(self, x: Array, t: Scalar) -> Array:
        """Estimate local diffusion from residual variance."""
        x = jnp.atleast_2d(x)
        
        # Get predicted drift
        drift_pred = self.drift(x, t)
        
        # Kernel weights
        diff = x[:, None, :] - self._X_data[None, :, :]
        dist_sq = jnp.sum(diff ** 2, axis=-1)
        weights = jnp.exp(-dist_sq / (2 * self.bandwidth ** 2))
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-10)
        
        # Residual variance: Var(dx/dt - b(x)) ≈ σ²/dt
        residuals = self._dX_data[None, :, :] - drift_pred[:, None, :]  # [batch, N, dim]
        residuals_sq = jnp.sum(residuals ** 2, axis=-1)  # [batch, N]
        
        local_var = jnp.einsum('bn,bn->b', weights, residuals_sq)
        
        # σ = sqrt(Var * dt)
        sigma = jnp.sqrt(local_var * self.dt + 1e-10)
        
        return sigma
    
    def _quadratic_variation_diffusion(self, x: Array, t: Scalar) -> Array:
        """Estimate diffusion from quadratic variation."""
        # For now, return constant estimate
        # Could implement local QV estimation
        return self._sigma_constant * jnp.ones(x.shape[0])
    
    def sample_trajectory(
        self,
        key: PRNGKey,
        x0: Array,
        num_steps: int,
    ) -> Array:
        """Simulate trajectory from the empirical dynamics.
        
        Uses Euler-Maruyama with estimated drift and diffusion.
        """
        x0 = jnp.atleast_2d(x0)
        batch_size = x0.shape[0]
        
        trajectory = [x0]
        x = x0
        
        keys = jax.random.split(key, num_steps)
        
        for i in range(num_steps):
            t = i * self.dt
            
            # Drift and diffusion
            b = self.drift(x, t)
            sigma = self.diffusion(x, t)
            
            # Euler-Maruyama step
            dW = jax.random.normal(keys[i], x.shape) * jnp.sqrt(self.dt)
            
            if jnp.ndim(sigma) == 0 or sigma.shape == ():
                x = x + b * self.dt + sigma * dW
            else:
                x = x + b * self.dt + sigma[:, None] * dW
            
            trajectory.append(x)
        
        return jnp.stack(trajectory, axis=1)  # [batch, num_steps+1, dim]
    
    def to_sde_coefficients(self):
        """Convert to SDECoefficients for compatibility."""
        from ..types import SDECoefficients
        return SDECoefficients(
            drift=self.drift,
            diffusion=lambda t: self.diffusion(None, t) if self.is_diffusion_scalar else self.diffusion,
            is_diffusion_scalar=self.is_diffusion_scalar,
        )


# =============================================================================
# Pre-fitted Drift Models
# =============================================================================

@dataclass
class KernelDriftModel:
    """Pre-fitted kernel drift model for fast inference.
    
    Stores the data needed for Nadaraya-Watson regression
    and provides efficient drift evaluation.
    """
    X_data: Array  # [N, dim]
    dX_data: Array  # [N, dim]
    bandwidth: float
    
    def __call__(self, x: Array, t: Scalar = None) -> Array:
        """Evaluate drift at x."""
        x = jnp.atleast_2d(x)
        
        diff = x[:, None, :] - self.X_data[None, :, :]
        dist_sq = jnp.sum(diff ** 2, axis=-1)
        weights = jnp.exp(-dist_sq / (2 * self.bandwidth ** 2))
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-10)
        
        return jnp.einsum('bn,nd->bd', weights, self.dX_data)


@dataclass  
class LocalLinearDriftModel:
    """Pre-fitted local linear drift model.
    
    Stores grid of local linear models for fast lookup.
    """
    grid_points: Array  # [num_grid, dim]
    coefficients: Array  # [num_grid, 1+dim, dim] - local [a, B] at each grid point
    bandwidth: float
    
    def __call__(self, x: Array, t: Scalar = None) -> Array:
        """Evaluate drift via interpolation of local models."""
        x = jnp.atleast_2d(x)
        
        # Find weights to grid points
        diff = x[:, None, :] - self.grid_points[None, :, :]
        dist_sq = jnp.sum(diff ** 2, axis=-1)
        weights = jnp.exp(-dist_sq / (2 * self.bandwidth ** 2))
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-10)
        
        # Evaluate each local model at query points
        ones = jnp.ones((x.shape[0], 1))
        features = jnp.hstack([ones, x])  # [batch, 1+dim]
        
        # Local predictions: [batch, num_grid, dim]
        local_preds = jnp.einsum('bf,gfd->bgd', features, self.coefficients)
        
        # Weighted combination
        return jnp.einsum('bg,bgd->bd', weights, local_preds)


# =============================================================================
# Factory Functions
# =============================================================================

def create_empirical_reference(
    trajectories: Array,
    dt: float,
    method: str = 'kernel',
    **kwargs,
) -> EmpiricalReferenceDynamics:
    """Create empirical reference dynamics from trajectory data.
    
    Args:
        trajectories: Shape [num_traj, num_times, dim].
        dt: Time step.
        method: Drift estimation method.
        **kwargs: Additional arguments.
        
    Returns:
        EmpiricalReferenceDynamics instance.
    """
    return EmpiricalReferenceDynamics(
        trajectories=trajectories,
        dt=dt,
        drift_method=method,
        **kwargs,
    )


def fit_drift_model(
    trajectories: Array,
    dt: float,
    method: str = 'kernel',
    **kwargs,
) -> Union[KernelDriftModel, LocalLinearDriftModel]:
    """Fit a drift model to trajectory data.
    
    Returns a callable drift function that can be used
    with standard SB solvers.
    """
    emp_ref = EmpiricalReferenceDynamics(
        trajectories=trajectories,
        dt=dt,
        drift_method=method,
        **kwargs,
    )
    
    if method == 'kernel':
        return KernelDriftModel(
            X_data=emp_ref._X_data,
            dX_data=emp_ref._dX_data,
            bandwidth=emp_ref.bandwidth,
        )
    else:
        raise NotImplementedError(f"Pre-fitted model for {method} not implemented")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'EmpiricalReferenceDynamics',
    'KernelDriftModel',
    'LocalLinearDriftModel',
    'create_empirical_reference',
    'fit_drift_model',
]
