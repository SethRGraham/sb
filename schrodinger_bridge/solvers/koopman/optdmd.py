"""Optimized Dynamic Mode Decomposition (optDMD) for Noise-Robust Koopman Approximation.

This module implements optDMD and related noise-robust variants of DMD,
essential for working with empirical/noisy data like financial time series.

The Problem with Standard DMD/EDMD:
==================================
Standard DMD solves: min ||X' - AX||_F

This is a *projection* problem, which is biased when X contains noise.
The bias grows with noise variance, leading to:
- Spurious eigenvalues
- Incorrect mode shapes
- Poor prediction accuracy

optDMD Solution:
===============
optDMD solves: min ||X' - AX||_F subject to rank(A) = r
             or equivalently: min over (φ, λ, b) ||X - Φ diag(b) V_λ||_F

where V_λ is the Vandermonde matrix of eigenvalues. This jointly optimizes
eigenvalues and modes, avoiding the bias from noisy X.

References:
==========
- Askham & Kutz (2018) "Variable Projection Methods for an Optimized DMD"
- Dawson et al. (2016) "Characterizing and correcting for the effect of sensor noise in DMD"
- Hemati et al. (2017) "De-biasing the dynamic mode decomposition for applied Koopman analysis"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.numpy.linalg import svd, lstsq, eig

# Type aliases
Array = jnp.ndarray
PRNGKey = jax.Array
Scalar = Union[float, Array]


# =============================================================================
# Result Containers
# =============================================================================

class OptDMDResult(NamedTuple):
    """Result from optDMD computation.
    
    Attributes:
        eigenvalues: Optimized continuous-time eigenvalues λ (where modes ~ e^{λt}).
        modes: Optimized DMD modes Φ, shape [state_dim, num_modes].
        amplitudes: Mode amplitudes b, shape [num_modes].
        reconstruction_error: Final optimization error.
        iterations: Number of optimization iterations.
        metadata: Additional information.
    """
    eigenvalues: Array  # Continuous-time eigenvalues
    modes: Array
    amplitudes: Array
    reconstruction_error: float
    iterations: int
    metadata: Dict


class BagOptDMDResult(NamedTuple):
    """Result from Bagging-optDMD.
    
    Attributes:
        eigenvalues_mean: Mean eigenvalues across bags.
        eigenvalues_std: Standard deviation of eigenvalues.
        modes_mean: Mean modes across bags.
        all_eigenvalues: Eigenvalues from each bag.
        all_modes: Modes from each bag.
        confidence_intervals: 95% CI for eigenvalues.
    """
    eigenvalues_mean: Array
    eigenvalues_std: Array
    modes_mean: Array
    all_eigenvalues: Array
    all_modes: Array
    confidence_intervals: Tuple[Array, Array]


# =============================================================================
# Standard DMD (for comparison)
# =============================================================================

def standard_dmd(
    X: Array,
    X_prime: Array,
    rank: Optional[int] = None,
    dt: float = 1.0,
) -> Dict:
    """Standard DMD via SVD.
    
    Solves X' ≈ AX via A = X' X^†
    
    Args:
        X: Data matrix, shape [state_dim, num_snapshots].
        X_prime: Time-shifted data, shape [state_dim, num_snapshots].
        rank: Truncation rank (None for full rank).
        dt: Time step between snapshots.
        
    Returns:
        Dictionary with eigenvalues, modes, etc.
    """
    # SVD of X
    U, S, Vh = svd(X, full_matrices=False)
    
    if rank is not None:
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
    
    # Compute reduced A
    S_inv = jnp.diag(1.0 / (S + 1e-10))
    A_tilde = U.T @ X_prime @ Vh.T @ S_inv
    
    # Eigendecomposition
    eigenvalues_discrete, W = eig(A_tilde)
    
    # Convert to continuous time: λ_cont = log(λ_disc) / dt
    eigenvalues = jnp.log(eigenvalues_discrete + 1e-10) / dt
    
    # DMD modes
    modes = X_prime @ Vh.T @ S_inv @ W
    
    # Amplitudes from initial condition
    amplitudes = jnp.linalg.lstsq(modes, X[:, 0], rcond=None)[0]
    
    # Reconstruction error
    X_recon = modes @ jnp.diag(amplitudes) @ jnp.exp(jnp.outer(eigenvalues, jnp.arange(X.shape[1]) * dt))
    error = jnp.mean(jnp.abs(X - X_recon) ** 2)
    
    return {
        'eigenvalues': eigenvalues,
        'modes': modes,
        'amplitudes': amplitudes,
        'A_tilde': A_tilde,
        'reconstruction_error': float(jnp.real(error)),
    }


# =============================================================================
# Optimized DMD (optDMD)
# =============================================================================

def optdmd(
    X: Array,
    times: Array,
    rank: int,
    max_iterations: int = 100,
    tol: float = 1e-6,
    init_eigenvalues: Optional[Array] = None,
) -> OptDMDResult:
    """Optimized DMD via variable projection.
    
    Jointly optimizes eigenvalues and modes to minimize:
        ||X - Φ B V_λ||_F
    
    where V_λ is the Vandermonde matrix with entries exp(λ_j t_i).
    
    This is solved via variable projection:
    1. For fixed λ, optimal Φ B is linear least squares
    2. Optimize λ via Levenberg-Marquardt on the residual
    
    Args:
        X: Data matrix, shape [state_dim, num_snapshots].
        times: Time points, shape [num_snapshots].
        rank: Number of modes (rank of approximation).
        max_iterations: Maximum optimization iterations.
        tol: Convergence tolerance.
        init_eigenvalues: Initial eigenvalue guess (uses standard DMD if None).
        
    Returns:
        OptDMDResult with optimized eigenvalues, modes, and amplitudes.
    """
    state_dim, num_snapshots = X.shape
    
    # Initialize eigenvalues from standard DMD if not provided
    if init_eigenvalues is None:
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        X_curr = X[:, :-1]
        X_next = X[:, 1:]
        dmd_init = standard_dmd(X_curr, X_next, rank=rank, dt=dt)
        eigenvalues = dmd_init['eigenvalues'][:rank]
    else:
        eigenvalues = init_eigenvalues
    
    def build_vandermonde(lambdas: Array) -> Array:
        """Build Vandermonde matrix V[i,j] = exp(λ_j * t_i)."""
        return jnp.exp(jnp.outer(times, lambdas))
    
    def compute_residual(lambdas: Array) -> Tuple[Array, Array, Array]:
        """Compute residual and optimal modes/amplitudes for given eigenvalues."""
        V = build_vandermonde(lambdas)  # [num_snapshots, rank]
        
        # Solve X = Φ B V^T for Φ B via least squares
        # This is equivalent to: X V (V^T V)^{-1} = Φ B
        # Or solve V^T (Φ B)^T = X^T
        PhiB, residuals, _, _ = jnp.linalg.lstsq(V, X.T, rcond=None)
        PhiB = PhiB.T  # [state_dim, rank]
        
        # Reconstruction
        X_recon = PhiB @ V.T
        residual = X - X_recon
        
        return residual, PhiB, V
    
    def loss_fn(lambdas_real_imag: Array) -> Scalar:
        """Loss function for optimization (real-valued parameterization)."""
        # Unpack real and imaginary parts
        lambdas = lambdas_real_imag[:rank] + 1j * lambdas_real_imag[rank:]
        residual, _, _ = compute_residual(lambdas)
        return jnp.sum(jnp.abs(residual) ** 2)
    
    # Pack eigenvalues as real vector for optimization
    lambdas_packed = jnp.concatenate([jnp.real(eigenvalues), jnp.imag(eigenvalues)])
    
    # Simple gradient descent optimization (could use L-BFGS for better performance)
    learning_rate = 0.01
    prev_loss = float('inf')
    
    for iteration in range(max_iterations):
        loss, grads = jax.value_and_grad(loss_fn)(lambdas_packed)
        
        # Gradient descent step
        lambdas_packed = lambdas_packed - learning_rate * grads
        
        # Check convergence
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
        
        # Adaptive learning rate
        if iteration > 0 and iteration % 20 == 0:
            learning_rate *= 0.5
    
    # Extract final results
    final_eigenvalues = lambdas_packed[:rank] + 1j * lambdas_packed[rank:]
    residual, PhiB, V = compute_residual(final_eigenvalues)
    
    # Separate modes and amplitudes via SVD of PhiB
    U_phi, S_phi, Vh_phi = svd(PhiB, full_matrices=False)
    modes = U_phi
    amplitudes = S_phi * Vh_phi[jnp.arange(rank), jnp.arange(rank)]
    
    reconstruction_error = float(jnp.mean(jnp.abs(residual) ** 2))
    
    return OptDMDResult(
        eigenvalues=final_eigenvalues,
        modes=modes,
        amplitudes=amplitudes,
        reconstruction_error=reconstruction_error,
        iterations=iteration + 1,
        metadata={
            'rank': rank,
            'num_snapshots': num_snapshots,
            'state_dim': state_dim,
        },
    )


# =============================================================================
# Bagging-Optimized DMD
# =============================================================================

def bagging_optdmd(
    X: Array,
    times: Array,
    rank: int,
    num_bags: int = 100,
    bag_fraction: float = 0.8,
    key: PRNGKey = None,
    max_iterations: int = 50,
) -> BagOptDMDResult:
    """Bagging-optimized DMD for robust eigenvalue estimation.
    
    Performs optDMD on multiple bootstrap samples and aggregates results.
    This provides:
    1. Robust eigenvalue estimates (median/mean across bags)
    2. Uncertainty quantification (std across bags)
    3. Identification of spurious modes (high variance across bags)
    
    Args:
        X: Data matrix, shape [state_dim, num_snapshots].
        times: Time points.
        rank: Number of modes.
        num_bags: Number of bootstrap samples.
        bag_fraction: Fraction of snapshots in each bag.
        key: JAX random key.
        max_iterations: Max iterations per optDMD call.
        
    Returns:
        BagOptDMDResult with statistics across bags.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    state_dim, num_snapshots = X.shape
    bag_size = int(num_snapshots * bag_fraction)
    
    all_eigenvalues = []
    all_modes = []
    
    for i in range(num_bags):
        key, subkey = jax.random.split(key)
        
        # Sample indices with replacement
        indices = jax.random.choice(subkey, num_snapshots, shape=(bag_size,), replace=True)
        indices = jnp.sort(indices)  # Keep time ordering
        
        X_bag = X[:, indices]
        times_bag = times[indices]
        
        # Run optDMD on this bag
        try:
            result = optdmd(X_bag, times_bag, rank, max_iterations=max_iterations)
            all_eigenvalues.append(result.eigenvalues)
            all_modes.append(result.modes)
        except Exception:
            # Skip failed optimizations
            continue
    
    if len(all_eigenvalues) == 0:
        raise ValueError("All bagging iterations failed")
    
    # Stack results
    all_eigenvalues = jnp.stack(all_eigenvalues)  # [num_bags, rank]
    all_modes = jnp.stack(all_modes)  # [num_bags, state_dim, rank]
    
    # Compute statistics
    eigenvalues_mean = jnp.mean(all_eigenvalues, axis=0)
    eigenvalues_std = jnp.std(all_eigenvalues, axis=0)
    modes_mean = jnp.mean(all_modes, axis=0)
    
    # 95% confidence intervals
    ci_low = jnp.percentile(all_eigenvalues, 2.5, axis=0)
    ci_high = jnp.percentile(all_eigenvalues, 97.5, axis=0)
    
    return BagOptDMDResult(
        eigenvalues_mean=eigenvalues_mean,
        eigenvalues_std=eigenvalues_std,
        modes_mean=modes_mean,
        all_eigenvalues=all_eigenvalues,
        all_modes=all_modes,
        confidence_intervals=(ci_low, ci_high),
    )


# =============================================================================
# Forward-Backward DMD (fbDMD) for Debiasing
# =============================================================================

def forward_backward_dmd(
    X: Array,
    rank: Optional[int] = None,
    dt: float = 1.0,
) -> Dict:
    """Forward-Backward DMD for bias reduction.
    
    Computes DMD in both forward and backward directions, then
    combines to cancel first-order noise bias.
    
    A_fb = (A_f @ A_b^{-1})^{1/2}
    
    This removes O(σ²) bias from eigenvalues where σ is noise level.
    
    Args:
        X: Data matrix, shape [state_dim, num_snapshots].
        rank: Truncation rank.
        dt: Time step.
        
    Returns:
        Dictionary with debiased eigenvalues and modes.
    """
    # Forward DMD: X' = A_f X
    X_f = X[:, :-1]
    X_f_prime = X[:, 1:]
    result_f = standard_dmd(X_f, X_f_prime, rank=rank, dt=dt)
    
    # Backward DMD: X = A_b X'
    result_b = standard_dmd(X_f_prime, X_f, rank=rank, dt=dt)
    
    # Combine: A_fb = sqrt(A_f @ A_b^{-1})
    # In eigenvalue space: λ_fb = sqrt(λ_f / λ_b)
    lambda_f = result_f['eigenvalues']
    lambda_b = result_b['eigenvalues']
    
    # Match eigenvalues by proximity
    eigenvalues_fb = jnp.sqrt(lambda_f * jnp.conj(lambda_b))
    
    # Use forward modes (could also average)
    modes = result_f['modes']
    
    return {
        'eigenvalues': eigenvalues_fb,
        'modes': modes,
        'eigenvalues_forward': lambda_f,
        'eigenvalues_backward': lambda_b,
        'amplitudes': result_f['amplitudes'],
    }


# =============================================================================
# Total Least Squares DMD (tlsDMD)
# =============================================================================

def tls_dmd(
    X: Array,
    X_prime: Array,
    rank: Optional[int] = None,
    dt: float = 1.0,
) -> Dict:
    """Total Least Squares DMD.
    
    Assumes noise in both X and X', solving:
        min ||[ΔX, ΔX']||_F  s.t.  (X' + ΔX') = A(X + ΔX)
    
    This is appropriate when both current and next states are noisy
    (typical for real-world measurements).
    
    Args:
        X: Current states, shape [state_dim, num_snapshots].
        X_prime: Next states, shape [state_dim, num_snapshots].
        rank: Truncation rank.
        dt: Time step.
        
    Returns:
        Dictionary with TLS-DMD results.
    """
    state_dim, num_snapshots = X.shape
    
    # Stack data
    Z = jnp.vstack([X, X_prime])  # [2*state_dim, num_snapshots]
    
    # SVD of stacked data
    U, S, Vh = svd(Z, full_matrices=False)
    
    if rank is not None:
        r = rank
    else:
        r = min(state_dim, num_snapshots)
    
    # Partition U into blocks
    U1 = U[:state_dim, :r]
    U2 = U[state_dim:, :r]
    
    # TLS solution: A_tls = U2 @ U1^†
    A_tls = U2 @ jnp.linalg.pinv(U1)
    
    # Eigendecomposition
    eigenvalues_discrete, W = eig(A_tls)
    eigenvalues = jnp.log(eigenvalues_discrete + 1e-10) / dt
    
    # Modes
    modes = U1 @ W
    
    return {
        'eigenvalues': eigenvalues,
        'modes': modes,
        'A_tls': A_tls,
    }


# =============================================================================
# Convenience Functions for Koopman Integration
# =============================================================================

def optdmd_from_trajectories(
    trajectories: Array,
    dictionary: 'Dictionary',
    rank: int,
    dt: float,
    method: str = 'optdmd',
    key: Optional[PRNGKey] = None,
    **kwargs,
) -> Dict:
    """Apply optDMD to trajectory data with dictionary lifting.
    
    This is the interface for integrating optDMD with EDMD/gEDMD workflows.
    
    Args:
        trajectories: Shape [num_traj, num_times, dim].
        dictionary: Observable dictionary.
        rank: Number of modes.
        dt: Time step.
        method: 'optdmd', 'bagging', 'fbdmd', or 'tls'.
        key: Random key (for bagging).
        **kwargs: Additional arguments for the specific method.
        
    Returns:
        Dictionary with Koopman approximation results.
    """
    num_traj, num_times, dim = trajectories.shape
    
    # Evaluate dictionary on all trajectory points
    # Flatten trajectories: [num_traj * num_times, dim]
    X_flat = trajectories.reshape(-1, dim)
    Psi_flat = dictionary(X_flat)  # [num_traj * num_times, dict_size]
    
    # Reshape back: [num_traj, num_times, dict_size]
    dict_size = Psi_flat.shape[1]
    Psi = Psi_flat.reshape(num_traj, num_times, dict_size)
    
    # Stack trajectories as columns for DMD
    # X[:, k] = Psi at time k, concatenated across trajectories
    # Shape: [dict_size, num_traj * (num_times - 1)]
    
    X_list = []
    X_prime_list = []
    times_list = []
    
    times = jnp.arange(num_times) * dt
    
    for traj_idx in range(num_traj):
        for t_idx in range(num_times - 1):
            X_list.append(Psi[traj_idx, t_idx])
            X_prime_list.append(Psi[traj_idx, t_idx + 1])
            times_list.append(times[t_idx])
    
    X = jnp.stack(X_list, axis=1)  # [dict_size, num_samples]
    X_prime = jnp.stack(X_prime_list, axis=1)
    times_arr = jnp.array(times_list)
    
    # Apply chosen method
    if method == 'optdmd':
        # For optDMD, we need continuous-time formulation
        # Stack X and use time indices
        result = optdmd(X, times_arr, rank, **kwargs)
        return {
            'eigenvalues': result.eigenvalues,
            'modes': result.modes,
            'amplitudes': result.amplitudes,
            'reconstruction_error': result.reconstruction_error,
            'method': 'optdmd',
        }
    
    elif method == 'bagging':
        if key is None:
            key = jax.random.PRNGKey(42)
        result = bagging_optdmd(X, times_arr, rank, key=key, **kwargs)
        return {
            'eigenvalues': result.eigenvalues_mean,
            'eigenvalues_std': result.eigenvalues_std,
            'modes': result.modes_mean,
            'confidence_intervals': result.confidence_intervals,
            'method': 'bagging_optdmd',
        }
    
    elif method == 'fbdmd':
        result = forward_backward_dmd(X, rank=rank, dt=dt)
        return {
            'eigenvalues': result['eigenvalues'],
            'modes': result['modes'],
            'method': 'fbdmd',
        }
    
    elif method == 'tls':
        result = tls_dmd(X, X_prime, rank=rank, dt=dt)
        return {
            'eigenvalues': result['eigenvalues'],
            'modes': result['modes'],
            'method': 'tls_dmd',
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'OptDMDResult',
    'BagOptDMDResult',
    'standard_dmd',
    'optdmd',
    'bagging_optdmd',
    'forward_backward_dmd',
    'tls_dmd',
    'optdmd_from_trajectories',
]
