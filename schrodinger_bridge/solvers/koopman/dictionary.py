"""Dictionary Functions for Koopman-Based Methods.

This module provides various dictionary (observable) functions for
Extended Dynamic Mode Decomposition (EDMD) and related methods.

The choice of dictionary is crucial for approximating Koopman eigenfunctions.
Different dictionaries capture different features of the dynamics.

Mathematical Background:
=======================
EDMD approximates the Koopman operator K by finding matrix K̃ such that:
    Ψ(x_{k+1}) ≈ K̃ Ψ(x_k)

where Ψ(x) = [ψ₁(x), ψ₂(x), ..., ψ_D(x)]ᵀ is the dictionary of observables.

The quality of the approximation depends heavily on the dictionary:
- Polynomial: Good for polynomial dynamics, captures nonlinearity order
- Fourier: Good for periodic/oscillatory dynamics
- RBF: Universal approximation, adapts to data distribution
- Hermite: Natural for Gaussian/diffusion problems (orthogonal w.r.t. Gaussian)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations_with_replacement
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.scipy.special import factorial

# Type aliases (matching library conventions)
Array = jnp.ndarray
PRNGKey = jax.Array
Scalar = Union[float, Array]


# =============================================================================
# Abstract Dictionary Base Class
# =============================================================================

class Dictionary(abc.ABC):
    """Abstract base class for observable dictionaries.
    
    A dictionary maps states x ∈ ℝᵈ to a feature vector Ψ(x) ∈ ℝᴰ
    where D is the dictionary size.
    
    For time-dependent problems (like SB), the dictionary can also
    depend on time: Ψ(x, t).
    """
    
    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Number of dictionary elements."""
        pass
    
    @property
    @abc.abstractmethod
    def input_dim(self) -> int:
        """Input state dimension."""
        pass
    
    @abc.abstractmethod
    def __call__(self, x: Array, t: Optional[Scalar] = None) -> Array:
        """Evaluate dictionary at state x (and optionally time t).
        
        Args:
            x: State vector(s), shape [batch, dim] or [dim].
            t: Optional time, shape [] or [batch].
            
        Returns:
            Dictionary values, shape [batch, size] or [size].
        """
        pass
    
    def gradient(self, x: Array, t: Optional[Scalar] = None) -> Array:
        """Compute gradient ∇_x Ψ(x, t).
        
        Args:
            x: State vector(s), shape [batch, dim].
            t: Optional time.
            
        Returns:
            Gradients, shape [batch, size, dim].
        """
        x = jnp.atleast_2d(x)
        
        def single_grad(x_i, t_i):
            return jax.jacfwd(lambda xi: self(xi[None], t_i)[0])(x_i)
        
        if t is None:
            t = jnp.zeros(x.shape[0])
        t = jnp.atleast_1d(t)
        if t.shape[0] == 1:
            t = jnp.broadcast_to(t, (x.shape[0],))
            
        return jax.vmap(single_grad)(x, t)
    
    def laplacian(self, x: Array, t: Optional[Scalar] = None) -> Array:
        """Compute Laplacian Δ_x Ψ(x, t).
        
        Args:
            x: State vector(s), shape [batch, dim].
            t: Optional time.
            
        Returns:
            Laplacians, shape [batch, size].
        """
        x = jnp.atleast_2d(x)
        
        def single_laplacian(x_i, t_i):
            def psi_i(xi):
                return self(xi[None], t_i)[0]  # [size]
            
            hess = jax.hessian(psi_i)(x_i)  # [size, dim, dim]
            return jnp.trace(hess, axis1=1, axis2=2)  # [size]
        
        if t is None:
            t = jnp.zeros(x.shape[0])
        t = jnp.atleast_1d(t)
        if t.shape[0] == 1:
            t = jnp.broadcast_to(t, (x.shape[0],))
            
        return jax.vmap(single_laplacian)(x, t)


# =============================================================================
# Polynomial Dictionary
# =============================================================================

class PolynomialDictionary(Dictionary):
    """Polynomial dictionary up to specified degree.
    
    For dimension d and degree p, includes all monomials:
        {x₁^{a₁} x₂^{a₂} ... x_d^{a_d} : a₁ + a₂ + ... + a_d ≤ p}
    
    Example (d=2, p=2):
        [1, x, y, x², xy, y²]
    
    Attributes:
        dim: Input dimension.
        degree: Maximum polynomial degree.
        include_time: Whether to include time-dependent terms.
    """
    
    def __init__(
        self,
        dim: int,
        degree: int = 3,
        include_time: bool = True,
        time_degree: int = 2,
    ):
        self._dim = dim
        self._degree = degree
        self._include_time = include_time
        self._time_degree = time_degree
        
        # Precompute multi-indices
        self._multi_indices = self._compute_multi_indices()
        self._size = len(self._multi_indices)
        
        if include_time:
            # Add time-polynomial terms
            self._time_size = time_degree + 1
            self._size += self._time_size - 1  # -1 to avoid duplicate constant
            # Cross terms: x^a * t^b for b > 0
            self._size += len(self._multi_indices) * time_degree
    
    def _compute_multi_indices(self) -> List[Tuple[int, ...]]:
        """Compute all multi-indices for polynomials up to degree."""
        indices = []
        for total_deg in range(self._degree + 1):
            for combo in combinations_with_replacement(range(self._dim), total_deg):
                # Convert to exponent tuple
                exponents = [0] * self._dim
                for idx in combo:
                    exponents[idx] += 1
                indices.append(tuple(exponents))
        return indices
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def input_dim(self) -> int:
        return self._dim
    
    @property
    def degree(self) -> int:
        return self._degree
    
    def __call__(self, x: Array, t: Optional[Scalar] = None) -> Array:
        x = jnp.atleast_2d(x)
        batch_size = x.shape[0]
        
        # Compute spatial polynomial terms
        terms = []
        for exponents in self._multi_indices:
            # x₁^{a₁} * x₂^{a₂} * ...
            term = jnp.ones(batch_size)
            for i, exp in enumerate(exponents):
                if exp > 0:
                    term = term * (x[:, i] ** exp)
            terms.append(term)
        
        # Add time terms if requested
        if self._include_time and t is not None:
            t = jnp.atleast_1d(t)
            if t.shape[0] == 1:
                t = jnp.broadcast_to(t, (batch_size,))
            
            # Pure time polynomials (t, t², ...)
            for p in range(1, self._time_degree + 1):
                terms.append(t ** p)
            
            # Cross terms: spatial_poly * t^p
            for exponents in self._multi_indices:
                term = jnp.ones(batch_size)
                for i, exp in enumerate(exponents):
                    if exp > 0:
                        term = term * (x[:, i] ** exp)
                for p in range(1, self._time_degree + 1):
                    terms.append(term * (t ** p))
        
        return jnp.stack(terms, axis=-1)
    
    def get_term_names(self) -> List[str]:
        """Get human-readable names for each dictionary term."""
        names = []
        var_names = [f'x{i}' for i in range(self._dim)]
        
        for exponents in self._multi_indices:
            parts = []
            for i, exp in enumerate(exponents):
                if exp == 1:
                    parts.append(var_names[i])
                elif exp > 1:
                    parts.append(f'{var_names[i]}^{exp}')
            names.append('*'.join(parts) if parts else '1')
        
        if self._include_time:
            for p in range(1, self._time_degree + 1):
                names.append(f't^{p}' if p > 1 else 't')
            
            for exponents in self._multi_indices:
                base_parts = []
                for i, exp in enumerate(exponents):
                    if exp == 1:
                        base_parts.append(var_names[i])
                    elif exp > 1:
                        base_parts.append(f'{var_names[i]}^{exp}')
                base = '*'.join(base_parts) if base_parts else '1'
                for p in range(1, self._time_degree + 1):
                    t_part = f't^{p}' if p > 1 else 't'
                    names.append(f'{base}*{t_part}')
        
        return names


# =============================================================================
# Fourier Dictionary
# =============================================================================

class FourierDictionary(Dictionary):
    """Fourier basis dictionary for periodic/oscillatory dynamics.
    
    Includes sin and cos terms at various frequencies:
        {1, sin(2πk·x/L), cos(2πk·x/L) : k ∈ frequency set}
    
    Useful for problems with periodic boundary conditions or
    oscillatory dynamics.
    """
    
    def __init__(
        self,
        dim: int,
        num_frequencies: int = 5,
        max_frequency: float = 5.0,
        include_cross_terms: bool = True,
        include_time: bool = True,
    ):
        self._dim = dim
        self._num_frequencies = num_frequencies
        self._max_frequency = max_frequency
        self._include_cross = include_cross_terms
        self._include_time = include_time
        
        # Compute frequencies (log-spaced for multi-scale)
        self._frequencies = jnp.linspace(0.5, max_frequency, num_frequencies)
        
        # Compute size
        # 1 (constant) + 2*dim*num_freq (sin/cos for each dimension)
        self._size = 1 + 2 * dim * num_frequencies
        
        if include_cross_terms and dim > 1:
            # Add sin(k_i x_i + k_j x_j) type terms
            num_pairs = dim * (dim - 1) // 2
            self._size += 2 * num_pairs * num_frequencies
        
        if include_time:
            self._size += 2 * num_frequencies  # sin/cos in time
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def input_dim(self) -> int:
        return self._dim
    
    def __call__(self, x: Array, t: Optional[Scalar] = None) -> Array:
        x = jnp.atleast_2d(x)
        batch_size = x.shape[0]
        
        terms = [jnp.ones(batch_size)]  # Constant term
        
        # Single-dimension Fourier terms
        for d in range(self._dim):
            for freq in self._frequencies:
                arg = 2 * jnp.pi * freq * x[:, d]
                terms.append(jnp.sin(arg))
                terms.append(jnp.cos(arg))
        
        # Cross terms
        if self._include_cross and self._dim > 1:
            for i in range(self._dim):
                for j in range(i + 1, self._dim):
                    for freq in self._frequencies:
                        arg = 2 * jnp.pi * freq * (x[:, i] + x[:, j])
                        terms.append(jnp.sin(arg))
                        terms.append(jnp.cos(arg))
        
        # Time terms
        if self._include_time and t is not None:
            t = jnp.atleast_1d(t)
            if t.shape[0] == 1:
                t = jnp.broadcast_to(t, (batch_size,))
            
            for freq in self._frequencies:
                arg = 2 * jnp.pi * freq * t
                terms.append(jnp.sin(arg))
                terms.append(jnp.cos(arg))
        
        return jnp.stack(terms, axis=-1)


# =============================================================================
# Radial Basis Function (RBF) Dictionary
# =============================================================================

class RBFDictionary(Dictionary):
    """Radial Basis Function dictionary.
    
    Uses Gaussian RBFs centered at data-derived or specified centers:
        ψ_i(x) = exp(-||x - c_i||² / (2σ²))
    
    Provides universal approximation and adapts to data distribution.
    """
    
    def __init__(
        self,
        dim: int,
        centers: Optional[Array] = None,
        num_centers: int = 50,
        bandwidth: Optional[float] = None,
        include_time: bool = True,
    ):
        self._dim = dim
        self._num_centers = num_centers
        self._bandwidth = bandwidth
        self._include_time = include_time
        
        # Centers will be set later if not provided
        self._centers = centers
        self._centers_initialized = centers is not None
        
        # Size: num_centers + 1 (constant) + dim (linear)
        self._size = num_centers + 1 + dim
        if include_time:
            self._size += 1  # Linear time term
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def input_dim(self) -> int:
        return self._dim
    
    def set_centers_from_data(self, data: Array, key: Optional[PRNGKey] = None):
        """Set RBF centers from data (e.g., using k-means or random selection).
        
        Args:
            data: Data points, shape [N, dim].
            key: Random key for center selection.
        """
        if key is None:
            # Uniform spacing through data indices
            indices = jnp.linspace(0, len(data) - 1, self._num_centers).astype(int)
            self._centers = data[indices]
        else:
            # Random selection
            indices = jax.random.choice(key, len(data), (self._num_centers,), replace=False)
            self._centers = data[indices]
        
        # Auto-compute bandwidth using median heuristic
        if self._bandwidth is None:
            dists = jnp.sum((self._centers[:, None, :] - self._centers[None, :, :]) ** 2, axis=-1)
            self._bandwidth = jnp.sqrt(jnp.median(dists[dists > 0]))
        
        self._centers_initialized = True
    
    def __call__(self, x: Array, t: Optional[Scalar] = None) -> Array:
        x = jnp.atleast_2d(x)
        batch_size = x.shape[0]
        
        if not self._centers_initialized:
            raise ValueError("RBF centers not initialized. Call set_centers_from_data first.")
        
        terms = [jnp.ones(batch_size)]  # Constant
        
        # Linear terms
        for d in range(self._dim):
            terms.append(x[:, d])
        
        # RBF terms
        # ||x - c_i||² for all centers
        dists_sq = jnp.sum(
            (x[:, None, :] - self._centers[None, :, :]) ** 2,
            axis=-1
        )  # [batch, num_centers]
        
        rbf_values = jnp.exp(-dists_sq / (2 * self._bandwidth ** 2))
        for i in range(self._num_centers):
            terms.append(rbf_values[:, i])
        
        # Time term
        if self._include_time and t is not None:
            t = jnp.atleast_1d(t)
            if t.shape[0] == 1:
                t = jnp.broadcast_to(t, (batch_size,))
            terms.append(t)
        
        return jnp.stack(terms, axis=-1)


# =============================================================================
# Hermite Polynomial Dictionary
# =============================================================================

class HermiteDictionary(Dictionary):
    """Hermite polynomial dictionary.
    
    Uses probabilist's Hermite polynomials (He_n), which are orthogonal
    with respect to the standard Gaussian measure:
        ∫ He_m(x) He_n(x) exp(-x²/2) dx = n! δ_{mn}
    
    This is natural for diffusion processes and Gaussian-related problems,
    making it ideal for Schrödinger Bridge problems.
    
    The polynomials satisfy:
        He_0(x) = 1
        He_1(x) = x
        He_2(x) = x² - 1
        He_3(x) = x³ - 3x
        He_n(x) = x·He_{n-1}(x) - (n-1)·He_{n-2}(x)
    """
    
    def __init__(
        self,
        dim: int,
        max_order: int = 4,
        include_time: bool = True,
    ):
        self._dim = dim
        self._max_order = max_order
        self._include_time = include_time
        
        # Compute multi-indices for tensor product basis
        self._multi_indices = self._compute_multi_indices()
        self._size = len(self._multi_indices)
        
        if include_time:
            self._size += max_order  # He_1(t), He_2(t), ... for time
    
    def _compute_multi_indices(self) -> List[Tuple[int, ...]]:
        """Compute multi-indices for tensor product Hermite basis."""
        indices = []
        for total_order in range(self._max_order + 1):
            for combo in combinations_with_replacement(range(self._dim), total_order):
                orders = [0] * self._dim
                for idx in combo:
                    orders[idx] += 1
                if tuple(orders) not in indices:
                    indices.append(tuple(orders))
        return indices
    
    @staticmethod
    def _hermite_1d(x: Array, n: int) -> Array:
        """Evaluate probabilist's Hermite polynomial He_n(x)."""
        if n == 0:
            return jnp.ones_like(x)
        elif n == 1:
            return x
        else:
            # Recurrence: He_n(x) = x·He_{n-1}(x) - (n-1)·He_{n-2}(x)
            He_prev2 = jnp.ones_like(x)
            He_prev1 = x
            for k in range(2, n + 1):
                He_curr = x * He_prev1 - (k - 1) * He_prev2
                He_prev2 = He_prev1
                He_prev1 = He_curr
            return He_prev1
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def input_dim(self) -> int:
        return self._dim
    
    def __call__(self, x: Array, t: Optional[Scalar] = None) -> Array:
        x = jnp.atleast_2d(x)
        batch_size = x.shape[0]
        
        terms = []
        
        # Tensor product Hermite polynomials
        for orders in self._multi_indices:
            term = jnp.ones(batch_size)
            for d, order in enumerate(orders):
                if order > 0:
                    term = term * self._hermite_1d(x[:, d], order)
            terms.append(term)
        
        # Time Hermite terms
        if self._include_time and t is not None:
            t = jnp.atleast_1d(t)
            if t.shape[0] == 1:
                t = jnp.broadcast_to(t, (batch_size,))
            
            # Rescale time to [−1, 1] approximately
            t_scaled = 2 * t - 1
            for n in range(1, self._max_order + 1):
                terms.append(self._hermite_1d(t_scaled, n))
        
        return jnp.stack(terms, axis=-1)


# =============================================================================
# Composite Dictionary
# =============================================================================

class CompositeDictionary(Dictionary):
    """Combines multiple dictionaries.
    
    Useful for capturing different aspects of dynamics simultaneously,
    e.g., polynomial for local behavior + Fourier for periodic features.
    """
    
    def __init__(self, dictionaries: List[Dictionary]):
        if not dictionaries:
            raise ValueError("Must provide at least one dictionary")
        
        self._dictionaries = dictionaries
        self._dim = dictionaries[0].input_dim
        
        # Verify all have same input dimension
        for d in dictionaries:
            if d.input_dim != self._dim:
                raise ValueError(f"Dimension mismatch: {d.input_dim} vs {self._dim}")
        
        # Total size (minus redundant constants)
        self._size = sum(d.size for d in dictionaries) - (len(dictionaries) - 1)
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def input_dim(self) -> int:
        return self._dim
    
    def __call__(self, x: Array, t: Optional[Scalar] = None) -> Array:
        results = []
        for i, d in enumerate(self._dictionaries):
            vals = d(x, t)
            if i > 0:
                # Remove constant term (avoid duplicates)
                vals = vals[..., 1:]
            results.append(vals)
        return jnp.concatenate(results, axis=-1)


# =============================================================================
# Adaptive Dictionary Builder
# =============================================================================

def build_adaptive_dictionary(
    dim: int,
    data: Optional[Array] = None,
    key: Optional[PRNGKey] = None,
    include_polynomial: bool = True,
    include_fourier: bool = False,
    include_rbf: bool = True,
    include_hermite: bool = True,
    polynomial_degree: int = 3,
    fourier_frequencies: int = 5,
    rbf_centers: int = 30,
    hermite_order: int = 4,
    include_time: bool = True,
) -> Dictionary:
    """Build an adaptive dictionary based on problem characteristics.
    
    This is a convenience function that creates a composite dictionary
    suitable for most Schrödinger Bridge problems.
    
    Args:
        dim: State dimension.
        data: Optional data for RBF center initialization.
        key: Random key for RBF initialization.
        include_polynomial: Include polynomial terms.
        include_fourier: Include Fourier terms.
        include_rbf: Include RBF terms.
        include_hermite: Include Hermite polynomial terms.
        polynomial_degree: Maximum polynomial degree.
        fourier_frequencies: Number of Fourier frequencies.
        rbf_centers: Number of RBF centers.
        hermite_order: Maximum Hermite polynomial order.
        include_time: Include time-dependent terms.
        
    Returns:
        Composite dictionary suitable for EDMD/gEDMD.
    """
    dictionaries = []
    
    if include_hermite:
        # Hermite first (most relevant for Gaussian/diffusion)
        dictionaries.append(HermiteDictionary(
            dim=dim,
            max_order=hermite_order,
            include_time=include_time,
        ))
    
    if include_polynomial:
        dictionaries.append(PolynomialDictionary(
            dim=dim,
            degree=polynomial_degree,
            include_time=include_time and not include_hermite,  # Avoid duplicate
        ))
    
    if include_fourier:
        dictionaries.append(FourierDictionary(
            dim=dim,
            num_frequencies=fourier_frequencies,
            include_time=include_time and not (include_hermite or include_polynomial),
        ))
    
    if include_rbf and data is not None:
        rbf_dict = RBFDictionary(
            dim=dim,
            num_centers=rbf_centers,
            include_time=False,  # Already covered
        )
        rbf_dict.set_centers_from_data(data, key)
        dictionaries.append(rbf_dict)
    
    if len(dictionaries) == 1:
        return dictionaries[0]
    
    return CompositeDictionary(dictionaries)


# =============================================================================
# Specialized Dictionaries for Schrödinger Bridge
# =============================================================================

class SBDictionary(Dictionary):
    """Specialized dictionary for Schrödinger Bridge problems.
    
    Includes terms specifically relevant for SB:
    - Gaussian basis functions (related to heat kernel)
    - Gradient-like terms (related to score function)
    - Boundary layer terms for t → 0 and t → 1
    """
    
    def __init__(
        self,
        dim: int,
        num_gaussians: int = 20,
        polynomial_degree: int = 2,
    ):
        self._dim = dim
        self._num_gaussians = num_gaussians
        self._poly_degree = polynomial_degree
        
        # Size: constant + linear + quadratic + gaussians + boundary terms
        # Polynomial: sum_{k=0}^{degree} C(dim+k-1, k)
        from math import comb
        poly_size = sum(comb(dim + k - 1, k) for k in range(polynomial_degree + 1))
        
        self._size = poly_size + num_gaussians + 4  # +4 for boundary terms
        
        # Initialize Gaussian centers uniformly
        self._gaussian_centers = None
        self._gaussian_scales = None
    
    def initialize(self, key: PRNGKey, data_range: Tuple[float, float] = (-3.0, 3.0)):
        """Initialize Gaussian centers."""
        k1, k2 = jax.random.split(key)
        self._gaussian_centers = jax.random.uniform(
            k1, (self._num_gaussians, self._dim),
            minval=data_range[0], maxval=data_range[1]
        )
        self._gaussian_scales = jax.random.uniform(
            k2, (self._num_gaussians,),
            minval=0.5, maxval=2.0
        )
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def input_dim(self) -> int:
        return self._dim
    
    def __call__(self, x: Array, t: Optional[Scalar] = None) -> Array:
        x = jnp.atleast_2d(x)
        batch_size = x.shape[0]
        
        if t is None:
            t = 0.5 * jnp.ones(batch_size)
        t = jnp.atleast_1d(t)
        if t.shape[0] == 1:
            t = jnp.broadcast_to(t, (batch_size,))
        
        terms = []
        
        # Constant
        terms.append(jnp.ones(batch_size))
        
        # Linear
        for d in range(self._dim):
            terms.append(x[:, d])
        
        # Quadratic (if degree >= 2)
        if self._poly_degree >= 2:
            for i in range(self._dim):
                for j in range(i, self._dim):
                    terms.append(x[:, i] * x[:, j])
        
        # Gaussian basis functions
        if self._gaussian_centers is not None:
            for c, s in zip(self._gaussian_centers, self._gaussian_scales):
                dist_sq = jnp.sum((x - c) ** 2, axis=-1)
                terms.append(jnp.exp(-dist_sq / (2 * s ** 2)))
        else:
            # Fallback: use simple Gaussian at origin with different scales
            for scale in jnp.linspace(0.5, 3.0, self._num_gaussians):
                dist_sq = jnp.sum(x ** 2, axis=-1)
                terms.append(jnp.exp(-dist_sq / (2 * scale ** 2)))
        
        # Boundary layer terms (important for SB near t=0, t=1)
        eps = 1e-3
        terms.append(1.0 / (t + eps))  # Singular as t → 0
        terms.append(1.0 / (1 - t + eps))  # Singular as t → 1
        terms.append(jnp.sqrt(t * (1 - t) + eps))  # Bridge variance profile
        terms.append(t * (1 - t))  # Quadratic time profile
        
        return jnp.stack(terms, axis=-1)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'Dictionary',
    'PolynomialDictionary',
    'FourierDictionary',
    'RBFDictionary',
    'HermiteDictionary',
    'CompositeDictionary',
    'SBDictionary',
    'build_adaptive_dictionary',
]
