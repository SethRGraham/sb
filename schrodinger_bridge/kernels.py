"""Kernel Utilities for RKHS-based Schrödinger Bridge methods.

This module provides kernel functions and RKHS operations for
non-parametric Schrödinger Bridge solvers.

RKHS methods represent the bridge solution as a weighted sum of
kernel functions, avoiding neural network training.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from .core.types import Array, PRNGKey, Scalar


# =============================================================================
# Kernel Functions
# =============================================================================

def gaussian_kernel(
    x: Array,
    y: Array,
    bandwidth: float = 1.0,
) -> Array:
    """Gaussian (RBF) kernel.
    
    k(x, y) = exp(-||x - y||² / (2σ²))
    
    Args:
        x: Points, shape [n, d].
        y: Points, shape [m, d].
        bandwidth: Kernel bandwidth σ.
        
    Returns:
        Kernel matrix, shape [n, m].
    """
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    sq_dists = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-sq_dists / (2 * bandwidth ** 2))


def laplacian_kernel(
    x: Array,
    y: Array,
    bandwidth: float = 1.0,
) -> Array:
    """Laplacian kernel.
    
    k(x, y) = exp(-||x - y|| / σ)
    
    Args:
        x: Points, shape [n, d].
        y: Points, shape [m, d].
        bandwidth: Kernel bandwidth.
        
    Returns:
        Kernel matrix, shape [n, m].
    """
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    dists = jnp.sqrt(jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1) + 1e-10)
    return jnp.exp(-dists / bandwidth)


def matern_kernel(
    x: Array,
    y: Array,
    bandwidth: float = 1.0,
    nu: float = 2.5,
) -> Array:
    """Matérn kernel.
    
    Special cases: ν=0.5 (Laplacian), ν→∞ (Gaussian).
    Common choices: ν=1.5 (once differentiable), ν=2.5 (twice differentiable).
    
    Args:
        x: Points, shape [n, d].
        y: Points, shape [m, d].
        bandwidth: Length scale.
        nu: Smoothness parameter.
        
    Returns:
        Kernel matrix, shape [n, m].
    """
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    dists = jnp.sqrt(jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1) + 1e-10)
    
    if nu == 0.5:
        return jnp.exp(-dists / bandwidth)
    elif nu == 1.5:
        scaled = jnp.sqrt(3) * dists / bandwidth
        return (1 + scaled) * jnp.exp(-scaled)
    elif nu == 2.5:
        scaled = jnp.sqrt(5) * dists / bandwidth
        return (1 + scaled + scaled ** 2 / 3) * jnp.exp(-scaled)
    else:
        raise ValueError(f"nu={nu} not implemented. Use 0.5, 1.5, or 2.5.")


def polynomial_kernel(
    x: Array,
    y: Array,
    degree: int = 3,
    c: float = 1.0,
) -> Array:
    """Polynomial kernel.
    
    k(x, y) = (x·y + c)^d
    
    Args:
        x: Points, shape [n, d].
        y: Points, shape [m, d].
        degree: Polynomial degree.
        c: Constant term.
        
    Returns:
        Kernel matrix, shape [n, m].
    """
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    return (jnp.dot(x, y.T) + c) ** degree


def imq_kernel(
    x: Array,
    y: Array,
    c: float = 1.0,
    beta: float = -0.5,
) -> Array:
    """Inverse Multiquadric (IMQ) kernel.
    
    k(x, y) = (c² + ||x - y||²)^β
    
    Common for MMD computation. With β = -0.5, it's characteristic.
    
    Args:
        x: Points, shape [n, d].
        y: Points, shape [m, d].
        c: Constant.
        beta: Exponent (typically negative).
        
    Returns:
        Kernel matrix, shape [n, m].
    """
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    sq_dists = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return (c ** 2 + sq_dists) ** beta


# =============================================================================
# Kernel Gradient Computation
# =============================================================================

def gaussian_kernel_gradient(
    x: Array,
    y: Array,
    bandwidth: float = 1.0,
) -> Array:
    """Gradient of Gaussian kernel w.r.t. first argument.
    
    ∇_x k(x, y) = -k(x, y) * (x - y) / σ²
    
    Args:
        x: Query points, shape [n, d].
        y: Reference points, shape [m, d].
        bandwidth: Kernel bandwidth.
        
    Returns:
        Gradient, shape [n, m, d].
    """
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    
    K = gaussian_kernel(x, y, bandwidth)  # [n, m]
    diff = x[:, None, :] - y[None, :, :]  # [n, m, d]
    
    return -K[:, :, None] * diff / (bandwidth ** 2)


def gaussian_kernel_laplacian(
    x: Array,
    y: Array,
    bandwidth: float = 1.0,
) -> Array:
    """Laplacian of Gaussian kernel w.r.t. first argument.
    
    Δ_x k(x, y) = k(x, y) * (||x-y||²/σ⁴ - d/σ²)
    
    Args:
        x: Query points, shape [n, d].
        y: Reference points, shape [m, d].
        bandwidth: Kernel bandwidth.
        
    Returns:
        Laplacian, shape [n, m].
    """
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    d = x.shape[1]
    
    K = gaussian_kernel(x, y, bandwidth)  # [n, m]
    sq_dists = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)  # [n, m]
    
    return K * (sq_dists / bandwidth ** 4 - d / bandwidth ** 2)


# =============================================================================
# Bandwidth Selection
# =============================================================================

def median_heuristic(x: Array, y: Optional[Array] = None) -> float:
    """Median heuristic for kernel bandwidth selection.
    
    Sets σ = median of pairwise distances.
    
    Args:
        x: Points, shape [n, d].
        y: Optional second set of points.
        
    Returns:
        Recommended bandwidth.
    """
    x = jnp.atleast_2d(x)
    if y is None:
        points = x
    else:
        y = jnp.atleast_2d(y)
        points = jnp.concatenate([x, y], axis=0)
    
    n = len(points)
    if n < 2:
        # Not enough points, return default
        return 1.0
    
    dists = jnp.sqrt(jnp.sum(
        (points[:, None, :] - points[None, :, :]) ** 2, axis=-1
    ) + 1e-10)
    
    # Get upper triangular (excluding diagonal)
    upper_tri_mask = jnp.triu(jnp.ones((n, n)), k=1)
    upper_dists = dists * upper_tri_mask
    
    # Flatten and remove zeros
    flat_dists = upper_dists.flatten()
    nonzero_dists = flat_dists[flat_dists > 0]
    
    if len(nonzero_dists) == 0:
        return 1.0
    
    return float(jnp.median(nonzero_dists))


def silverman_bandwidth(x: Array) -> float:
    """Silverman's rule of thumb for bandwidth.
    
    σ = (4/(d+2))^(1/(d+4)) * n^(-1/(d+4)) * std(x)
    
    Args:
        x: Points, shape [n, d].
        
    Returns:
        Recommended bandwidth.
    """
    x = jnp.atleast_2d(x)
    n, d = x.shape
    std = jnp.mean(jnp.std(x, axis=0))
    return float((4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4)) * std)


# =============================================================================
# RKHS Operations
# =============================================================================

@dataclass
class KernelDensityEstimate:
    """Kernel density estimate.
    
    Attributes:
        centers: Kernel centers, shape [n, d].
        weights: Kernel weights, shape [n].
        bandwidth: Kernel bandwidth.
        kernel_fn: Kernel function.
    """
    centers: Array
    weights: Array
    bandwidth: float
    kernel_fn: Callable = gaussian_kernel
    
    def __call__(self, x: Array) -> Array:
        """Evaluate density at points x.
        
        Args:
            x: Query points, shape [m, d].
            
        Returns:
            Density values, shape [m].
        """
        K = self.kernel_fn(x, self.centers, self.bandwidth)  # [m, n]
        return K @ self.weights
    
    def log_prob(self, x: Array) -> Array:
        """Log probability (may be unnormalized)."""
        return jnp.log(self(x) + 1e-10)
    
    def gradient(self, x: Array) -> Array:
        """Gradient of log density (score function).
        
        ∇log p(x) = ∇p(x) / p(x)
        """
        # Compute density and its gradient
        K = self.kernel_fn(x, self.centers, self.bandwidth)
        density = K @ self.weights  # [m]
        
        # Gradient of kernel sum
        K_grad = gaussian_kernel_gradient(x, self.centers, self.bandwidth)  # [m, n, d]
        density_grad = jnp.einsum('ijk,j->ik', K_grad, self.weights)  # [m, d]
        
        # Score
        return density_grad / (density[:, None] + 1e-10)


def fit_kde(
    samples: Array,
    bandwidth: Optional[float] = None,
    weights: Optional[Array] = None,
) -> KernelDensityEstimate:
    """Fit kernel density estimate to samples.
    
    Args:
        samples: Data points, shape [n, d].
        bandwidth: Kernel bandwidth (median heuristic if None).
        weights: Sample weights (uniform if None).
        
    Returns:
        KDE object.
    """
    samples = jnp.atleast_2d(samples)
    n = samples.shape[0]
    
    if bandwidth is None:
        bandwidth = median_heuristic(samples)
    
    if weights is None:
        weights = jnp.ones(n) / n
    
    return KernelDensityEstimate(
        centers=samples,
        weights=weights,
        bandwidth=bandwidth,
    )


# =============================================================================
# Kernel Mean Embedding
# =============================================================================

class KernelMeanEmbedding:
    """Kernel mean embedding of a distribution.
    
    μ_P = E_P[k(·, X)] ≈ (1/n) Σ k(·, x_i)
    
    Attributes:
        samples: Sample points, shape [n, d].
        weights: Sample weights, shape [n].
        bandwidth: Kernel bandwidth.
        kernel_fn: Kernel function.
    """
    
    def __init__(
        self,
        samples: Array,
        weights: Optional[Array] = None,
        bandwidth: Optional[float] = None,
        kernel_fn: Callable = gaussian_kernel,
    ):
        self.samples = jnp.atleast_2d(samples)
        n = self.samples.shape[0]
        self.weights = weights if weights is not None else jnp.ones(n) / n
        self.bandwidth = bandwidth if bandwidth is not None else median_heuristic(samples)
        self.kernel_fn = kernel_fn
    
    def __call__(self, x: Array) -> Array:
        """Evaluate embedding at test points.
        
        μ_P(x) = E_P[k(x, X)]
        
        Args:
            x: Test points, shape [m, d].
            
        Returns:
            Embedding values, shape [m].
        """
        K = self.kernel_fn(x, self.samples, self.bandwidth)
        return K @ self.weights
    
    def mmd_squared(self, other: 'KernelMeanEmbedding') -> float:
        """Compute squared MMD to another embedding.
        
        MMD²(P, Q) = ||μ_P - μ_Q||²_H
        """
        Kxx = self.kernel_fn(self.samples, self.samples, self.bandwidth)
        Kyy = self.kernel_fn(other.samples, other.samples, self.bandwidth)
        Kxy = self.kernel_fn(self.samples, other.samples, self.bandwidth)
        
        mmd2 = (
            self.weights @ Kxx @ self.weights
            + other.weights @ Kyy @ other.weights
            - 2 * self.weights @ Kxy @ other.weights
        )
        
        return float(jnp.maximum(mmd2, 0.0))


# =============================================================================
# Kernel Regression (for RKHS-based drift/score estimation)
# =============================================================================

def kernel_ridge_regression(
    X: Array,
    y: Array,
    bandwidth: float,
    reg: float = 1e-4,
    kernel_fn: Callable = gaussian_kernel,
) -> Callable[[Array], Array]:
    """Kernel ridge regression.
    
    Solves: min_f ||f||²_H + λ Σ(f(x_i) - y_i)²
    
    Args:
        X: Training inputs, shape [n, d].
        y: Training targets, shape [n] or [n, d_out].
        bandwidth: Kernel bandwidth.
        reg: Regularization strength λ.
        kernel_fn: Kernel function.
        
    Returns:
        Prediction function.
    """
    X = jnp.atleast_2d(X)
    y = jnp.atleast_1d(y)
    n = X.shape[0]
    
    # Compute kernel matrix
    K = kernel_fn(X, X, bandwidth)  # [n, n]
    
    # Solve (K + λI)α = y
    alpha = jnp.linalg.solve(K + reg * jnp.eye(n), y)
    
    def predict(x: Array) -> Array:
        """Predict at new points."""
        k = kernel_fn(x, X, bandwidth)  # [m, n]
        return k @ alpha
    
    return predict


def kernel_score_estimation(
    samples: Array,
    bandwidth: Optional[float] = None,
    reg: float = 1e-4,
) -> Callable[[Array], Array]:
    """Estimate score function using kernels.
    
    Uses the Stein gradient estimator:
    ∇log p(x) ≈ Σ_i w_i ∇_x k(x, x_i)
    
    where weights solve a linear system.
    
    Args:
        samples: Samples from p, shape [n, d].
        bandwidth: Kernel bandwidth.
        reg: Regularization.
        
    Returns:
        Score function.
    """
    samples = jnp.atleast_2d(samples)
    n, d = samples.shape
    
    if bandwidth is None:
        bandwidth = median_heuristic(samples)
    
    # Compute kernel matrix and its gradient
    K = gaussian_kernel(samples, samples, bandwidth)  # [n, n]
    
    # Stein identity: E[∇log p(X) k(X, x')] = -E[∇_X k(X, x')]
    # We solve for weights that give the score
    
    # Gram matrix with Laplacian
    K_lap = gaussian_kernel_laplacian(samples, samples, bandwidth)  # [n, n]
    
    # Solve regularized system
    weights = jnp.linalg.solve(K + reg * jnp.eye(n), -K_lap.sum(axis=0) / n)
    
    def score(x: Array) -> Array:
        """Compute score at query points."""
        K_grad = gaussian_kernel_gradient(x, samples, bandwidth)  # [m, n, d]
        return jnp.einsum('ijk,j->ik', K_grad, weights)
    
    return score


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Kernels
    'gaussian_kernel', 'laplacian_kernel', 'matern_kernel',
    'polynomial_kernel', 'imq_kernel',
    # Gradients
    'gaussian_kernel_gradient', 'gaussian_kernel_laplacian',
    # Bandwidth selection
    'median_heuristic', 'silverman_bandwidth',
    # KDE
    'KernelDensityEstimate', 'fit_kde',
    # Mean embedding
    'KernelMeanEmbedding',
    # Regression
    'kernel_ridge_regression', 'kernel_score_estimation',
]
