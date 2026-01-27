"""Extended Dynamic Mode Decomposition (EDMD) for Koopman Approximation.

This module implements EDMD and its variants for approximating the Koopman
operator from trajectory data.

Mathematical Background:
=======================
For a dynamical system x_{k+1} = f(x_k), the Koopman operator K acts on
observables g: X → ℝ by (Kg)(x) = g(f(x)).

EDMD approximates K by finding a matrix K̃ that minimizes:
    ||Ψ(X') - K̃ Ψ(X)||_F

where Ψ is a dictionary of observables, X = [x_0, x_1, ...] are snapshots,
and X' = [x_1, x_2, ...] are the time-shifted snapshots.

For Stochastic Systems (SDEs):
============================
When data comes from an SDE dX = b(X)dt + σdW, EDMD approximates the
*stochastic Koopman operator* (SKO), whose eigenfunctions are solutions
to the backward Kolmogorov equation:
    ∂_t φ + b·∇φ + (σ²/2)Δφ = λφ

This is precisely what we need for Schrödinger Bridges, where the
Schrödinger potential ψ solves a similar PDE!

References:
==========
- Williams et al. (2015) "A Data-Driven Approximation of the Koopman Operator"
- Klus et al. (2016) "On the numerical approximation of the Perron-Frobenius
  and Koopman operator"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.numpy.linalg import eig, lstsq, pinv, svd

from .dictionary import Dictionary, PolynomialDictionary

# Type aliases
Array = jnp.ndarray
PRNGKey = jax.Array
Scalar = Union[float, Array]


# =============================================================================
# EDMD Result Container
# =============================================================================

class EDMDResult(NamedTuple):
    """Result from EDMD computation.
    
    Attributes:
        K: Koopman matrix approximation, shape [D, D].
        eigenvalues: Koopman eigenvalues, shape [D].
        eigenvectors: Right eigenvectors (Koopman modes), shape [D, D].
        eigenfunctions_coeffs: Coefficients for eigenfunctions in dictionary basis.
        reconstruction_error: Fitting error.
        dictionary: The dictionary used.
        metadata: Additional information.
    """
    K: Array
    eigenvalues: Array
    eigenvectors: Array
    eigenfunction_coeffs: Array
    reconstruction_error: float
    dictionary: Dictionary
    metadata: Dict


# =============================================================================
# Core EDMD Algorithm
# =============================================================================

def edmd(
    X: Array,
    Y: Array,
    dictionary: Dictionary,
    regularization: float = 1e-6,
) -> EDMDResult:
    """Extended Dynamic Mode Decomposition.
    
    Finds the Koopman matrix K̃ that best satisfies:
        Ψ(Y) ≈ K̃ Ψ(X)
    
    in the least-squares sense.
    
    Args:
        X: Data snapshots at time t, shape [N, dim].
        Y: Data snapshots at time t+dt, shape [N, dim].
        dictionary: Observable dictionary.
        regularization: Tikhonov regularization for stability.
        
    Returns:
        EDMDResult with Koopman matrix and spectral decomposition.
    """
    # Evaluate dictionary on data
    Psi_X = dictionary(X)  # [N, D]
    Psi_Y = dictionary(Y)  # [N, D]
    
    N, D = Psi_X.shape
    
    # Solve least squares: K̃ = argmin ||Ψ(Y) - K̃ Ψ(X)||²_F
    # This is equivalent to: K̃ᵀ = (Ψ(X)ᵀΨ(X))⁻¹ Ψ(X)ᵀ Ψ(Y)
    # With regularization: K̃ᵀ = (Ψ(X)ᵀΨ(X) + λI)⁻¹ Ψ(X)ᵀ Ψ(Y)
    
    G = Psi_X.T @ Psi_X  # [D, D] - Gram matrix
    A = Psi_X.T @ Psi_Y  # [D, D]
    
    # Add regularization
    G_reg = G + regularization * jnp.eye(D)
    
    # Solve for K transpose
    K_T = jnp.linalg.solve(G_reg, A)
    K = K_T.T  # [D, D]
    
    # Compute reconstruction error
    Psi_Y_pred = Psi_X @ K_T
    error = jnp.mean((Psi_Y - Psi_Y_pred) ** 2)
    
    # Eigendecomposition for Koopman modes
    eigenvalues, eigenvectors = jnp.linalg.eig(K)
    
    # Left eigenvectors give eigenfunction coefficients
    # K v = λ v (right eigenvector)
    # wᵀ K = λ wᵀ (left eigenvector)
    # Eigenfunction: φ(x) = wᵀ Ψ(x)
    
    # For symmetric case, left = right conjugate transpose
    # For general case, need to compute separately
    eigenfunction_coeffs = jnp.linalg.inv(eigenvectors).T  # Left eigenvectors
    
    return EDMDResult(
        K=K,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigenfunction_coeffs=eigenfunction_coeffs,
        reconstruction_error=float(error),
        dictionary=dictionary,
        metadata={
            'num_samples': N,
            'dictionary_size': D,
            'regularization': regularization,
        },
    )


def extended_dmd(
    trajectories: Array,
    dictionary: Dictionary,
    dt: float = 1.0,
    regularization: float = 1e-6,
    use_time: bool = True,
) -> EDMDResult:
    """EDMD from trajectory data.
    
    Extracts snapshot pairs from full trajectories.
    
    Args:
        trajectories: Trajectory data, shape [batch, time, dim].
        dictionary: Observable dictionary.
        dt: Time step between snapshots.
        regularization: Tikhonov regularization.
        use_time: Whether to pass time information to dictionary.
        
    Returns:
        EDMDResult.
    """
    batch_size, num_times, dim = trajectories.shape
    
    # Extract all consecutive pairs
    X_list = []
    Y_list = []
    T_list = []
    
    times = jnp.linspace(0, 1, num_times)
    
    for b in range(batch_size):
        for i in range(num_times - 1):
            X_list.append(trajectories[b, i])
            Y_list.append(trajectories[b, i + 1])
            if use_time:
                T_list.append(times[i])
    
    X = jnp.stack(X_list)
    Y = jnp.stack(Y_list)
    
    if use_time:
        T = jnp.stack(T_list)
        # Modify dictionary call to include time
        Psi_X = dictionary(X, T)
        Psi_Y = dictionary(Y, T + dt / (num_times - 1))
    else:
        Psi_X = dictionary(X)
        Psi_Y = dictionary(Y)
    
    N, D = Psi_X.shape
    
    # EDMD solve
    G = Psi_X.T @ Psi_X + regularization * jnp.eye(D)
    A = Psi_X.T @ Psi_Y
    K_T = jnp.linalg.solve(G, A)
    K = K_T.T
    
    # Reconstruction error
    Psi_Y_pred = Psi_X @ K_T
    error = jnp.mean((Psi_Y - Psi_Y_pred) ** 2)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eig(K)
    eigenfunction_coeffs = jnp.linalg.inv(eigenvectors + 1e-10 * jnp.eye(D)).T
    
    return EDMDResult(
        K=K,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigenfunction_coeffs=eigenfunction_coeffs,
        reconstruction_error=float(error),
        dictionary=dictionary,
        metadata={
            'num_samples': N,
            'dictionary_size': D,
            'regularization': regularization,
            'num_trajectories': batch_size,
            'trajectory_length': num_times,
        },
    )


# =============================================================================
# Kernel EDMD (for high-dimensional dictionaries)
# =============================================================================

def kernel_edmd(
    X: Array,
    Y: Array,
    kernel_fn: Callable[[Array, Array], Array],
    regularization: float = 1e-6,
    num_eigenfunctions: int = 20,
) -> Dict:
    """Kernel EDMD for implicit high-dimensional dictionaries.
    
    Uses the kernel trick to work in a potentially infinite-dimensional
    feature space without explicitly computing the features.
    
    The kernel k(x, y) implicitly defines a feature map φ such that
    k(x, y) = ⟨φ(x), φ(y)⟩.
    
    Args:
        X: Data at time t, shape [N, dim].
        Y: Data at time t+dt, shape [N, dim].
        kernel_fn: Kernel function k(X, Y) → [N, M].
        regularization: Tikhonov regularization.
        num_eigenfunctions: Number of eigenfunctions to compute.
        
    Returns:
        Dictionary with eigenvalues, eigenfunction evaluators, etc.
    """
    N = X.shape[0]
    
    # Kernel matrices
    K_XX = kernel_fn(X, X)  # [N, N]
    K_XY = kernel_fn(X, Y)  # [N, N]
    
    # Regularized kernel EDMD
    # In kernel space, the Koopman operator approximation satisfies:
    # K_XY ≈ K_XX @ A for some A
    
    K_XX_reg = K_XX + regularization * jnp.eye(N)
    A = jnp.linalg.solve(K_XX_reg, K_XY)
    
    # Eigendecomposition of A gives approximate Koopman eigenvalues
    eigenvalues, eigenvectors = jnp.linalg.eig(A)
    
    # Sort by magnitude
    idx = jnp.argsort(jnp.abs(eigenvalues))[::-1][:num_eigenfunctions]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Eigenfunction evaluator
    def evaluate_eigenfunction(x_new: Array, idx: int) -> Array:
        """Evaluate i-th eigenfunction at new points."""
        K_xX = kernel_fn(x_new, X)  # [M, N]
        return K_xX @ eigenvectors[:, idx]
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'evaluate_eigenfunction': evaluate_eigenfunction,
        'K_XX': K_XX,
        'A': A,
        'X_train': X,
    }


# =============================================================================
# Koopman Mode Decomposition
# =============================================================================

def compute_koopman_modes(
    edmd_result: EDMDResult,
    observable_fn: Optional[Callable[[Array], Array]] = None,
) -> Dict:
    """Compute Koopman Mode Decomposition (KMD).
    
    Decomposes an observable g into Koopman modes:
        g(x) = Σ_j v_j φ_j(x)
    
    where φ_j are Koopman eigenfunctions and v_j are Koopman modes.
    
    Args:
        edmd_result: EDMD result with eigenfunctions.
        observable_fn: Observable to decompose (default: identity).
        
    Returns:
        Dictionary with modes, reconstruction function, etc.
    """
    K = edmd_result.K
    eigenvalues = edmd_result.eigenvalues
    eigenfunction_coeffs = edmd_result.eigenfunction_coeffs
    dictionary = edmd_result.dictionary
    D = K.shape[0]
    
    # For the identity observable on the dictionary,
    # the modes are just the right eigenvectors
    modes = edmd_result.eigenvectors
    
    def reconstruct(x: Array, t: Scalar = 0.0, num_modes: int = -1) -> Array:
        """Reconstruct observable at future time.
        
        Args:
            x: Current state.
            t: Time to propagate (in units where eigenvalues are per-unit-time).
            num_modes: Number of modes to use (-1 for all).
        """
        x = jnp.atleast_2d(x)
        psi = dictionary(x)  # [batch, D]
        
        if num_modes == -1:
            num_modes = D
        
        # φ_j(x) = eigenfunction_coeffs[j] · Ψ(x)
        eigenfunc_vals = psi @ eigenfunction_coeffs[:, :num_modes]  # [batch, num_modes]
        
        # Time evolution: λ^t for discrete, exp(λt) for continuous
        time_factors = jnp.exp(eigenvalues[:num_modes] * t)  # Assuming continuous
        
        # Reconstruct
        recon = eigenfunc_vals * time_factors @ modes[:num_modes, :]
        
        return recon
    
    def eigenfunction(x: Array, idx: int) -> Array:
        """Evaluate specific Koopman eigenfunction."""
        x = jnp.atleast_2d(x)
        psi = dictionary(x)
        return psi @ eigenfunction_coeffs[:, idx]
    
    return {
        'modes': modes,
        'eigenvalues': eigenvalues,
        'eigenfunction_coeffs': eigenfunction_coeffs,
        'reconstruct': reconstruct,
        'eigenfunction': eigenfunction,
    }


# =============================================================================
# EDMD for Score Function Approximation
# =============================================================================

def edmd_score_approximation(
    trajectories: Array,
    dictionary: Dictionary,
    sigma: float,
    regularization: float = 1e-6,
) -> Callable[[Array, Scalar], Array]:
    """Use EDMD to approximate the score function for SB.
    
    Key insight: For the SDE dX = b(X)dt + σdW, the Koopman eigenfunctions
    are related to the backward Kolmogorov equation solutions.
    
    The score ∇log p_t can be approximated by combining Koopman eigenfunctions:
        ∇log p_t(x) ≈ Σ_j c_j(t) ∇φ_j(x)
    
    Args:
        trajectories: Trajectory data, shape [batch, time, dim].
        dictionary: Observable dictionary (must support gradients).
        sigma: Diffusion coefficient.
        regularization: EDMD regularization.
        
    Returns:
        Score function approximation: score(x, t) → [batch, dim].
    """
    # Run EDMD
    result = extended_dmd(trajectories, dictionary, regularization=regularization)
    
    eigenvalues = result.eigenvalues
    eigenfunction_coeffs = result.eigenfunction_coeffs
    D = result.K.shape[0]
    
    # Identify relevant eigenfunctions (slow modes)
    # For SB, we want eigenfunctions with eigenvalues close to 1
    # (these correspond to near-stationary distributions)
    
    def score_fn(x: Array, t: Scalar) -> Array:
        """Approximate score function."""
        x = jnp.atleast_2d(x)
        batch_size = x.shape[0]
        
        if not hasattr(t, 'shape') or t.shape == ():
            t = t * jnp.ones(batch_size)
        
        # Get dictionary gradient: ∇Ψ(x, t)
        grad_psi = dictionary.gradient(x, t)  # [batch, D, dim]
        
        # Eigenfunction gradients: ∇φ_j(x) = eigenfunction_coeffs[j] · ∇Ψ(x)
        # [batch, D, dim] @ [D, D] → [batch, D, dim]
        grad_eigenfuncs = jnp.einsum('bdk,kj->bdj', grad_psi, eigenfunction_coeffs)
        
        # Time-dependent combination
        # For SB, use exponential decay based on eigenvalues
        # The terminal condition (t=1) determines the coefficients
        time_weights = jnp.exp(jnp.real(eigenvalues) * (1 - t[:, None]))  # [batch, D]
        
        # Score ≈ sum of eigenfunction gradients weighted by time
        # Actually, for proper SB, we need the ratio ∇ψ/ψ
        # This is a simplification that works for initialization
        score = jnp.einsum('bd,bdj->bj', time_weights, grad_eigenfuncs)
        
        # Normalize by σ² factor
        score = score / (sigma ** 2 + 1e-8)
        
        return score
    
    return score_fn


# =============================================================================
# EDMD Warm Start Utilities
# =============================================================================

def create_warm_start_drift(
    edmd_result: EDMDResult,
    reference_drift: Callable[[Array, Scalar], Array],
    sigma: float,
    num_modes: int = 10,
) -> Callable[[Array, Scalar], Array]:
    """Create warm-start drift function from EDMD result.
    
    Uses leading Koopman eigenfunctions to approximate the SB drift:
        b*(x,t) = b_ref(x,t) + σ² ∇log ψ(x,t)
    
    where ψ is approximated using Koopman eigenfunctions.
    
    Args:
        edmd_result: EDMD result.
        reference_drift: Reference SDE drift.
        sigma: Diffusion coefficient.
        num_modes: Number of Koopman modes to use.
        
    Returns:
        Approximate SB drift function.
    """
    dictionary = edmd_result.dictionary
    eigenvalues = edmd_result.eigenvalues
    eigenfunction_coeffs = edmd_result.eigenfunction_coeffs
    
    # Select leading modes (by eigenvalue magnitude closest to 1)
    # For SB, modes with |λ| ≈ 1 are most relevant
    mode_importance = jnp.abs(jnp.abs(eigenvalues) - 1)
    selected_idx = jnp.argsort(mode_importance)[:num_modes]
    
    selected_eigenvals = eigenvalues[selected_idx]
    selected_coeffs = eigenfunction_coeffs[:, selected_idx]
    
    def drift(x: Array, t: Scalar) -> Array:
        """Warm-start SB drift."""
        x = jnp.atleast_2d(x)
        batch_size = x.shape[0]
        
        if not hasattr(t, 'shape') or t.shape == ():
            t_arr = t * jnp.ones(batch_size)
        else:
            t_arr = t
        
        # Reference drift
        b_ref = reference_drift(x, t)
        
        # Compute eigenfunction values and gradients
        psi = dictionary(x, t_arr)  # [batch, D]
        grad_psi = dictionary.gradient(x, t_arr)  # [batch, D, dim]
        
        # Eigenfunction values: φ_j(x) = Ψ(x) · coeffs[:, j]
        phi_vals = psi @ selected_coeffs  # [batch, num_modes]
        
        # Eigenfunction gradients: ∇φ_j(x) = ∇Ψ(x) · coeffs[:, j]
        grad_phi = jnp.einsum('bdk,kj->bjd', grad_psi, selected_coeffs)  # [batch, num_modes, dim]
        
        # Time weights: for SB, ψ(x,t) evolves as ψ_j(x) exp(λ_j(T-t))
        # Terminal time T = 1
        time_factors = jnp.exp(jnp.real(selected_eigenvals) * (1 - t_arr[:, None]))
        
        # ψ ≈ Σ w_j φ_j exp(λ_j(1-t))
        psi_approx = jnp.sum(phi_vals * time_factors, axis=-1, keepdims=True) + 1e-8
        
        # ∇ψ ≈ Σ w_j ∇φ_j exp(λ_j(1-t))
        grad_psi_approx = jnp.einsum('bj,bjd->bd', time_factors, grad_phi)
        
        # ∇log ψ = ∇ψ / ψ
        grad_log_psi = grad_psi_approx / psi_approx
        
        # SB drift: b* = b_ref + σ² ∇log ψ
        drift_correction = sigma ** 2 * grad_log_psi
        
        return b_ref + drift_correction
    
    return drift


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'EDMDResult',
    'edmd',
    'extended_dmd',
    'kernel_edmd',
    'compute_koopman_modes',
    'edmd_score_approximation',
    'create_warm_start_drift',
]
