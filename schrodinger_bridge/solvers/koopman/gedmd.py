"""Generator Extended Dynamic Mode Decomposition (gEDMD).

This module implements gEDMD, which approximates the *generator* of the
Koopman operator rather than the operator itself.

Mathematical Background:
=======================
For an SDE: dX = b(X)dt + σ(X)dW

The Koopman generator L acts on smooth functions g by:
    (Lg)(x) = b(x)·∇g(x) + (1/2)Tr[σσᵀ∇²g(x)]
    
This is the infinitesimal generator of the Markov semigroup, and its
eigenfunctions satisfy:
    Lg = λg

For the backward Kolmogorov equation ∂_t u = Lu, eigenfunctions evolve as:
    u(x, t) = e^{λt} φ(x)

Connection to Schrödinger Bridge:
================================
The SB potential ψ(x,t) satisfies:
    ∂_t ψ + L_ref ψ = 0  (backward Kolmogorov for reference)
    
with terminal condition from μ₁. By approximating the generator via gEDMD,
we can directly identify the drift and diffusion of the SB!

Key advantages of gEDMD over EDMD:
- Works in continuous time (no discretization error)
- Directly gives drift and diffusion identification
- Better behaved eigenvalues (no need to take logarithms)

Reference:
=========
Klus et al. (2020) "Data-driven approximation of the Koopman generator:
Model reduction, system identification, and control"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from .dictionary import Dictionary, PolynomialDictionary

# Type aliases
Array = jnp.ndarray
PRNGKey = jax.Array
Scalar = Union[float, Array]


# =============================================================================
# gEDMD Result Container
# =============================================================================

class GEDMDResult(NamedTuple):
    """Result from gEDMD computation.
    
    Attributes:
        L: Generator matrix approximation, shape [D, D].
        eigenvalues: Generator eigenvalues (continuous-time).
        eigenvectors: Right eigenvectors.
        eigenfunction_coeffs: Coefficients for eigenfunctions.
        drift_coeffs: Identified drift coefficients (for SINDy-like recovery).
        diffusion_coeffs: Identified diffusion coefficients.
        reconstruction_error: Fitting error.
        dictionary: The dictionary used.
        metadata: Additional information.
    """
    L: Array
    eigenvalues: Array
    eigenvectors: Array
    eigenfunction_coeffs: Array
    drift_coeffs: Optional[Array]
    diffusion_coeffs: Optional[Array]
    reconstruction_error: float
    dictionary: Dictionary
    metadata: Dict


# =============================================================================
# Core gEDMD Algorithm
# =============================================================================

def gedmd(
    X: Array,
    dX_dt: Array,
    dictionary: Dictionary,
    sigma: Optional[float] = None,
    regularization: float = 1e-6,
) -> GEDMDResult:
    """Generator Extended Dynamic Mode Decomposition.
    
    Approximates the Koopman generator L from data by solving:
        dΨ/dt ≈ L Ψ
    
    where dΨ/dt is computed using the chain rule:
        dΨ/dt = ∇Ψ · dX/dt + (σ²/2) ΔΨ
    
    Args:
        X: State snapshots, shape [N, dim].
        dX_dt: Time derivatives, shape [N, dim].
        dictionary: Observable dictionary.
        sigma: Diffusion coefficient (if known, for Itô correction).
        regularization: Tikhonov regularization.
        
    Returns:
        GEDMDResult with generator matrix and spectral decomposition.
    """
    X = jnp.atleast_2d(X)
    N, dim = X.shape
    
    # Evaluate dictionary and derivatives
    Psi = dictionary(X)  # [N, D]
    grad_Psi = dictionary.gradient(X)  # [N, D, dim]
    
    D = Psi.shape[1]
    
    # Compute dΨ/dt using chain rule
    # dΨ/dt = ∇Ψ · dX/dt
    dPsi_dt = jnp.einsum('ndi,ni->nd', grad_Psi, dX_dt)  # [N, D]
    
    # Add Itô correction if diffusion is known
    if sigma is not None:
        laplacian_Psi = dictionary.laplacian(X)  # [N, D]
        dPsi_dt = dPsi_dt + 0.5 * sigma**2 * laplacian_Psi
    
    # Solve for generator: dΨ/dt = L Ψ
    # This is: L^T Ψ^T = (dΨ/dt)^T
    # Or: Ψ L^T = dΨ/dt in matrix form
    
    G = Psi.T @ Psi + regularization * jnp.eye(D)  # [D, D]
    A = Psi.T @ dPsi_dt  # [D, D]
    
    L_T = jnp.linalg.solve(G, A)
    L = L_T.T  # Generator matrix [D, D]
    
    # Reconstruction error
    dPsi_dt_pred = Psi @ L_T
    error = jnp.mean((dPsi_dt - dPsi_dt_pred) ** 2)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eig(L)
    
    # Left eigenvectors for eigenfunction coefficients
    eigenfunction_coeffs = jnp.linalg.inv(eigenvectors + 1e-10 * jnp.eye(D)).T
    
    return GEDMDResult(
        L=L,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigenfunction_coeffs=eigenfunction_coeffs,
        drift_coeffs=None,
        diffusion_coeffs=None,
        reconstruction_error=float(error),
        dictionary=dictionary,
        metadata={
            'num_samples': N,
            'dictionary_size': D,
            'regularization': regularization,
            'sigma': sigma,
        },
    )


def gedmd_from_trajectories(
    trajectories: Array,
    dictionary: Dictionary,
    dt: float,
    sigma: Optional[float] = None,
    regularization: float = 1e-6,
    time_derivative_method: str = 'finite_difference',
) -> GEDMDResult:
    """gEDMD from trajectory data.
    
    Estimates time derivatives from trajectories and applies gEDMD.
    
    Args:
        trajectories: Trajectory data, shape [batch, time, dim].
        dictionary: Observable dictionary.
        dt: Time step size.
        sigma: Known diffusion coefficient.
        regularization: Tikhonov regularization.
        time_derivative_method: 'finite_difference' or 'central_difference'.
        
    Returns:
        GEDMDResult.
    """
    batch_size, num_times, dim = trajectories.shape
    
    # Estimate time derivatives
    X_list = []
    dX_dt_list = []
    T_list = []
    
    times = jnp.linspace(0, 1, num_times)
    
    if time_derivative_method == 'finite_difference':
        # Forward difference: dX/dt ≈ (X_{k+1} - X_k) / dt
        for b in range(batch_size):
            for i in range(num_times - 1):
                X_list.append(trajectories[b, i])
                dX_dt_list.append((trajectories[b, i+1] - trajectories[b, i]) / dt)
                T_list.append(times[i])
    
    elif time_derivative_method == 'central_difference':
        # Central difference: dX/dt ≈ (X_{k+1} - X_{k-1}) / (2dt)
        for b in range(batch_size):
            for i in range(1, num_times - 1):
                X_list.append(trajectories[b, i])
                dX_dt_list.append(
                    (trajectories[b, i+1] - trajectories[b, i-1]) / (2 * dt)
                )
                T_list.append(times[i])
    
    X = jnp.stack(X_list)
    dX_dt = jnp.stack(dX_dt_list)
    T = jnp.stack(T_list)
    
    # Apply gEDMD with time information
    N = X.shape[0]
    
    Psi = dictionary(X, T)  # [N, D]
    grad_Psi = dictionary.gradient(X, T)  # [N, D, dim]
    
    D = Psi.shape[1]
    
    # dΨ/dt = ∇_x Ψ · dX/dt + ∂_t Ψ
    # For now, assume ∂_t Ψ is captured in the dictionary structure
    dPsi_dt = jnp.einsum('ndi,ni->nd', grad_Psi, dX_dt)
    
    if sigma is not None:
        laplacian_Psi = dictionary.laplacian(X, T)
        dPsi_dt = dPsi_dt + 0.5 * sigma**2 * laplacian_Psi
    
    # Solve for generator
    G = Psi.T @ Psi + regularization * jnp.eye(D)
    A = Psi.T @ dPsi_dt
    L_T = jnp.linalg.solve(G, A)
    L = L_T.T
    
    # Error
    dPsi_dt_pred = Psi @ L_T
    error = jnp.mean((dPsi_dt - dPsi_dt_pred) ** 2)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eig(L)
    eigenfunction_coeffs = jnp.linalg.inv(eigenvectors + 1e-10 * jnp.eye(D)).T
    
    return GEDMDResult(
        L=L,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigenfunction_coeffs=eigenfunction_coeffs,
        drift_coeffs=None,
        diffusion_coeffs=None,
        reconstruction_error=float(error),
        dictionary=dictionary,
        metadata={
            'num_samples': N,
            'dictionary_size': D,
            'dt': dt,
            'sigma': sigma,
            'time_derivative_method': time_derivative_method,
        },
    )


# =============================================================================
# SDE Identification via gEDMD
# =============================================================================

def gedmd_sde_identification(
    trajectories: Array,
    dictionary: Dictionary,
    dt: float,
    regularization: float = 1e-6,
    identify_diffusion: bool = True,
) -> Dict:
    """Identify SDE drift and diffusion from trajectories using gEDMD.
    
    For SDE: dX = b(X)dt + σ(X)dW
    
    The generator is: L = b·∇ + (1/2)σ²Δ
    
    By choosing appropriate dictionary functions, we can recover b and σ.
    
    Method:
    1. Use gEDMD to find L in dictionary basis
    2. Express b(x) = Σ_j ξ_j^b θ_j(x) for dictionary θ
    3. Similarly for σ(x) if identifying diffusion
    
    Args:
        trajectories: Trajectory data, shape [batch, time, dim].
        dictionary: Observable dictionary.
        dt: Time step.
        regularization: Tikhonov regularization.
        identify_diffusion: Whether to identify diffusion (requires quadratic terms).
        
    Returns:
        Dictionary with drift_fn, diffusion_fn, coefficients, etc.
    """
    batch_size, num_times, dim = trajectories.shape
    
    # Collect data points and derivatives
    X_list = []
    dX_dt_list = []
    dX_sq_dt_list = []  # For diffusion identification
    
    for b in range(batch_size):
        for i in range(num_times - 1):
            X_list.append(trajectories[b, i])
            dX = trajectories[b, i+1] - trajectories[b, i]
            dX_dt_list.append(dX / dt)
            
            if identify_diffusion:
                # Quadratic variation for diffusion
                dX_sq_dt_list.append(dX**2 / dt)
    
    X = jnp.stack(X_list)  # [N, dim]
    dX_dt = jnp.stack(dX_dt_list)  # [N, dim]
    
    N = X.shape[0]
    
    # Evaluate dictionary
    Psi = dictionary(X)  # [N, D]
    D = Psi.shape[1]
    
    # === Drift Identification ===
    # dX/dt ≈ b(X) = Θ(X) @ ξ_b
    # Solve: dX_dt = Psi @ ξ_b  (for each dimension)
    
    G = Psi.T @ Psi + regularization * jnp.eye(D)
    drift_coeffs = []
    
    for d in range(dim):
        A_d = Psi.T @ dX_dt[:, d]
        xi_d = jnp.linalg.solve(G, A_d)
        drift_coeffs.append(xi_d)
    
    drift_coeffs = jnp.stack(drift_coeffs, axis=-1)  # [D, dim]
    
    # Drift reconstruction error
    dX_dt_pred = Psi @ drift_coeffs
    drift_error = jnp.mean((dX_dt - dX_dt_pred) ** 2)
    
    # === Diffusion Identification ===
    diffusion_coeffs = None
    diffusion_error = None
    
    if identify_diffusion:
        dX_sq_dt = jnp.stack(dX_sq_dt_list)
        
        # E[dX²/dt] ≈ σ²(X)
        # Solve: dX_sq_dt = Psi @ ξ_σ²
        
        diffusion_coeffs = []
        for d in range(dim):
            A_d = Psi.T @ dX_sq_dt[:, d]
            xi_d = jnp.linalg.solve(G, A_d)
            diffusion_coeffs.append(xi_d)
        
        diffusion_coeffs = jnp.stack(diffusion_coeffs, axis=-1)  # [D, dim]
        
        # Diffusion error
        dX_sq_pred = Psi @ diffusion_coeffs
        diffusion_error = jnp.mean((dX_sq_dt - dX_sq_pred) ** 2)
    
    # === Create Functions ===
    def drift_fn(x: Array, t: Optional[Scalar] = None) -> Array:
        """Identified drift function b(x)."""
        x = jnp.atleast_2d(x)
        psi = dictionary(x, t) if t is not None else dictionary(x)
        return psi @ drift_coeffs
    
    def diffusion_fn(x: Array, t: Optional[Scalar] = None) -> Array:
        """Identified diffusion function σ(x)."""
        if diffusion_coeffs is None:
            raise ValueError("Diffusion not identified")
        x = jnp.atleast_2d(x)
        psi = dictionary(x, t) if t is not None else dictionary(x)
        sigma_sq = psi @ diffusion_coeffs
        return jnp.sqrt(jnp.maximum(sigma_sq, 1e-8))
    
    return {
        'drift_fn': drift_fn,
        'diffusion_fn': diffusion_fn if identify_diffusion else None,
        'drift_coeffs': drift_coeffs,
        'diffusion_coeffs': diffusion_coeffs,
        'drift_error': float(drift_error),
        'diffusion_error': float(diffusion_error) if diffusion_error else None,
        'dictionary': dictionary,
        'num_samples': N,
    }


def extract_drift_diffusion(
    gedmd_result: GEDMDResult,
    dim: int,
) -> Tuple[Callable, Callable]:
    """Extract drift and diffusion functions from gEDMD generator.
    
    For generator L = b·∇ + (σ²/2)Δ, we can recover b and σ by
    examining how L acts on simple observables.
    
    Using x_i as observable:
        Lx_i = b_i(x)
    
    Using x_i x_j as observable:
        L(x_i x_j) = x_i b_j + x_j b_i + σ_i² δ_{ij}
    
    Args:
        gedmd_result: gEDMD result with generator L.
        dim: State dimension.
        
    Returns:
        (drift_fn, diffusion_fn)
    """
    L = gedmd_result.L
    dictionary = gedmd_result.dictionary
    
    # This requires the dictionary to have specific structure
    # (linear terms first, then quadratic)
    
    # For a polynomial dictionary with degree ≥ 2:
    # Index 0: constant (1)
    # Index 1 to dim: linear (x_1, ..., x_d)
    # Following: quadratic terms
    
    # L applied to x_i gives b_i directly in the dictionary basis
    # We need to identify which dictionary elements correspond to what
    
    # Simplified approach: assume the drift is represented as Ψ @ ξ
    # where ξ is learned from the generator structure
    
    def drift_fn(x: Array, t: Optional[Scalar] = None) -> Array:
        """Extract drift from generator."""
        x = jnp.atleast_2d(x)
        batch_size = x.shape[0]
        
        psi = dictionary(x, t) if t is not None else dictionary(x)
        
        # Drift is related to first derivative terms
        # For linear observables, Lg = b·∇g
        # If g = x_i, then ∇g = e_i, so Lx_i = b_i
        
        # Extract from generator: first dim rows after constant
        # This is a simplification that works for polynomial dictionaries
        drift = jnp.zeros((batch_size, dim))
        
        for i in range(dim):
            # Row i+1 of L gives how x_i evolves
            # The contribution to the constant term gives b_i(x) ≈ L[i+1, :] @ Ψ(x)
            drift = drift.at[:, i].set(psi @ L[i+1, :])
        
        return drift
    
    def diffusion_fn(x: Array, t: Optional[Scalar] = None) -> Scalar:
        """Extract diffusion (assuming scalar)."""
        # For scalar diffusion, extract from quadratic terms
        # L(x²) = 2xb + σ²
        # Simplified: return constant from metadata if available
        sigma = gedmd_result.metadata.get('sigma', 1.0)
        return sigma
    
    return drift_fn, diffusion_fn


# =============================================================================
# gEDMD for Schrödinger Bridge
# =============================================================================

def gedmd_sb_drift(
    source_samples: Array,
    target_samples: Array,
    reference_trajectories: Array,
    dictionary: Dictionary,
    sigma: float,
    regularization: float = 1e-6,
) -> Callable[[Array, Scalar], Array]:
    """Use gEDMD to construct SB drift approximation.
    
    Strategy:
    1. Run gEDMD on reference trajectories to get generator L_ref
    2. Modify L to account for marginal constraints
    3. Extract drift correction from modified generator
    
    This gives an approximation to the optimal SB drift.
    
    Args:
        source_samples: Samples from source distribution, shape [N, dim].
        target_samples: Samples from target distribution, shape [M, dim].
        reference_trajectories: Trajectories from reference SDE.
        dictionary: Observable dictionary.
        sigma: Reference diffusion coefficient.
        regularization: gEDMD regularization.
        
    Returns:
        Approximate SB drift function.
    """
    batch_size, num_times, dim = reference_trajectories.shape
    dt = 1.0 / (num_times - 1)
    
    # Run gEDMD on reference trajectories
    result = gedmd_from_trajectories(
        reference_trajectories,
        dictionary,
        dt=dt,
        sigma=sigma,
        regularization=regularization,
    )
    
    L_ref = result.L
    eigenvalues = result.eigenvalues
    eigenfunction_coeffs = result.eigenfunction_coeffs
    D = L_ref.shape[0]
    
    # Compute boundary conditions from marginals
    # At t=0: distribution should match source
    # At t=1: distribution should match target
    
    # Evaluate dictionary on marginals
    Psi_source = dictionary(source_samples)  # [N, D]
    Psi_target = dictionary(target_samples)  # [M, D]
    
    # Mean dictionary values at boundaries
    psi_0 = jnp.mean(Psi_source, axis=0)  # [D]
    psi_1 = jnp.mean(Psi_target, axis=0)  # [D]
    
    # The SB potential ψ satisfies:
    # ∂_t ψ + L_ref ψ = 0
    # ψ(x, 1) determined by target marginal
    
    # Approximate ψ using eigenfunctions
    # ψ(x, t) ≈ Σ_j c_j exp(-λ_j(1-t)) φ_j(x)
    
    # Coefficients determined by terminal condition
    # At t=1: ψ(x, 1) ≈ Σ_j c_j φ_j(x) ∝ ρ_1(x)/ρ_ref(x)
    
    # Simplified: use eigenfunction decomposition of target indicator
    # c_j ≈ ⟨φ_j, target⟩
    
    # Project target mean onto eigenfunctions
    coeffs = eigenfunction_coeffs.T @ psi_1  # [D]
    
    def sb_drift(x: Array, t: Scalar) -> Array:
        """Approximate SB drift."""
        x = jnp.atleast_2d(x)
        batch_size = x.shape[0]
        
        if not hasattr(t, 'shape') or t.shape == ():
            t_arr = t * jnp.ones(batch_size)
        else:
            t_arr = t
        
        # Evaluate dictionary
        psi = dictionary(x, t_arr)  # [batch, D]
        grad_psi = dictionary.gradient(x, t_arr)  # [batch, D, dim]
        
        # Reference drift (from L_ref acting on linear observables)
        b_ref = jnp.zeros((batch_size, dim))
        for i in range(dim):
            b_ref = b_ref.at[:, i].set(psi @ L_ref[i+1, :])
        
        # Eigenfunction values
        phi = psi @ eigenfunction_coeffs  # [batch, D]
        
        # Eigenfunction gradients
        grad_phi = jnp.einsum('bdk,kj->bjd', grad_psi, eigenfunction_coeffs)
        
        # Time factors: exp(-λ(1-t))
        time_factors = jnp.exp(-jnp.real(eigenvalues) * (1 - t_arr[:, None]))
        
        # ψ(x, t) ≈ Σ c_j φ_j(x) exp(-λ_j(1-t))
        psi_approx = jnp.sum(coeffs * phi * time_factors, axis=-1, keepdims=True)
        psi_approx = jnp.maximum(psi_approx, 1e-8)
        
        # ∇ψ(x, t)
        grad_psi_approx = jnp.einsum('j,bj,bjd->bd', coeffs, time_factors, grad_phi)
        
        # SB drift correction: σ² ∇log ψ
        drift_correction = sigma**2 * grad_psi_approx / psi_approx
        
        return b_ref + drift_correction
    
    return sb_drift


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'GEDMDResult',
    'gedmd',
    'gedmd_from_trajectories',
    'gedmd_sde_identification',
    'extract_drift_diffusion',
    'gedmd_sb_drift',
]
