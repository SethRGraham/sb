"""Discrete-Time IPF-Koopman Solver for Schrödinger Bridges.

This module implements a discrete-time Schrödinger Bridge solver that combines
Koopman operator estimation (EDMD) with Iterative Proportional Fitting (IPF/Sinkhorn).

Why Discrete-Time?
=================
For low-frequency data (daily, weekly), estimating the continuous-time generator
via gEDMD is numerically unstable due to derivative estimation noise.

Instead, we directly model the discrete-time transition operator K_Δt and solve
the discrete SB problem via IPF.

The Discrete SB Problem:
=======================
Given:
- Source marginal μ₀
- Target marginal μ₁  
- Reference Markov transitions P_ref(x_{k+1} | x_k)

Find the path measure P* minimizing:
    KL(P* || P_ref) subject to P*_0 = μ₀, P*_N = μ₁

Solution: Iterative Proportional Fitting (Sinkhorn)
    π*_{ij} = α_i K_{ij} β_j
where K is the reference transition kernel and (α, β) satisfy marginal constraints.

References:
==========
- Cuturi (2013) "Sinkhorn Distances"
- Peyré & Cuturi (2019) "Computational Optimal Transport"
- Chen et al. (2016) "Entropic interpolation and Schrödinger bridges"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax, vmap

from .dictionary import Dictionary, PolynomialDictionary
from .optdmd import optdmd_from_trajectories, bagging_optdmd

# Type aliases
Array = jnp.ndarray
PRNGKey = jax.Array
Scalar = Union[float, Array]


# =============================================================================
# Result Containers
# =============================================================================

class DiscreteIPFResult(NamedTuple):
    """Result from discrete IPF-Koopman solver.
    
    Attributes:
        alpha: Left scaling factors (for source), shape [num_steps+1, num_particles].
        beta: Right scaling factors (for target), shape [num_steps+1, num_particles].
        marginals: Estimated marginals at each time step.
        transport_cost: Total transport cost.
        num_iterations: Sinkhorn iterations used.
        converged: Whether Sinkhorn converged.
    """
    alpha: Array
    beta: Array
    marginals: List[Array]
    transport_cost: float
    num_iterations: int
    converged: bool


class BridgePathResult(NamedTuple):
    """Sampled bridge paths.
    
    Attributes:
        paths: Sampled trajectories, shape [num_samples, num_steps+1, dim].
        log_weights: Log importance weights (if applicable).
        source_samples: Initial points.
        target_samples: Terminal points.
    """
    paths: Array
    log_weights: Optional[Array]
    source_samples: Array
    target_samples: Array


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DiscreteIPFConfig:
    """Configuration for discrete IPF-Koopman solver.
    
    Attributes:
        num_particles: Number of particles for discretization.
        sinkhorn_iterations: Maximum Sinkhorn iterations.
        sinkhorn_threshold: Convergence threshold for Sinkhorn.
        sinkhorn_regularization: Entropic regularization (ε).
        use_log_domain: Use log-domain Sinkhorn for stability.
        dictionary_type: Type of dictionary for Koopman.
        polynomial_degree: Degree for polynomial dictionary.
        koopman_rank: Rank truncation for Koopman operator.
        use_optdmd: Use optDMD for noise robustness.
        optdmd_bags: Number of bags for Bagging-optDMD.
    """
    num_particles: int = 1000
    sinkhorn_iterations: int = 100
    sinkhorn_threshold: float = 1e-6
    sinkhorn_regularization: float = 0.01
    use_log_domain: bool = True
    dictionary_type: str = 'polynomial'
    polynomial_degree: int = 3
    koopman_rank: int = 20
    use_optdmd: bool = True
    optdmd_bags: int = 50


# =============================================================================
# Sinkhorn Algorithm
# =============================================================================

def sinkhorn_log_domain(
    log_K: Array,
    log_mu: Array,
    log_nu: Array,
    num_iterations: int = 100,
    threshold: float = 1e-6,
) -> Tuple[Array, Array, bool]:
    """Sinkhorn algorithm in log domain for numerical stability.
    
    Solves: π* = diag(α) K diag(β) with π*1 = μ, π*ᵀ1 = ν
    
    In log domain: log π* = log α + log K + log β
    
    Args:
        log_K: Log transition kernel, shape [n, m].
        log_mu: Log source marginal, shape [n].
        log_nu: Log target marginal, shape [m].
        num_iterations: Maximum iterations.
        threshold: Convergence threshold.
        
    Returns:
        (log_alpha, log_beta, converged)
    """
    n, m = log_K.shape
    
    # Initialize
    log_alpha = jnp.zeros(n)
    log_beta = jnp.zeros(m)
    
    def sinkhorn_step(carry, _):
        log_alpha, log_beta, _ = carry
        
        # Update alpha: α_i = μ_i / (K β)_i
        log_Kb = jax.scipy.special.logsumexp(log_K + log_beta[None, :], axis=1)
        log_alpha_new = log_mu - log_Kb
        
        # Update beta: β_j = ν_j / (Kᵀ α)_j  
        log_Kta = jax.scipy.special.logsumexp(log_K.T + log_alpha_new[None, :], axis=1)
        log_beta_new = log_nu - log_Kta
        
        # Check convergence
        delta = jnp.max(jnp.abs(log_alpha_new - log_alpha))
        
        return (log_alpha_new, log_beta_new, delta), delta
    
    # Run iterations
    init_carry = (log_alpha, log_beta, jnp.inf)
    (log_alpha, log_beta, final_delta), deltas = lax.scan(
        sinkhorn_step, init_carry, None, length=num_iterations
    )
    
    converged = final_delta < threshold
    
    return log_alpha, log_beta, converged


def sinkhorn_standard(
    K: Array,
    mu: Array,
    nu: Array,
    num_iterations: int = 100,
    threshold: float = 1e-6,
) -> Tuple[Array, Array, bool]:
    """Standard Sinkhorn algorithm.
    
    Args:
        K: Transition kernel, shape [n, m].
        mu: Source marginal, shape [n].
        nu: Target marginal, shape [m].
        num_iterations: Maximum iterations.
        threshold: Convergence threshold.
        
    Returns:
        (alpha, beta, converged)
    """
    n, m = K.shape
    
    alpha = jnp.ones(n)
    beta = jnp.ones(m)
    
    def sinkhorn_step(carry, _):
        alpha, beta, _ = carry
        
        # Update alpha
        Kb = K @ beta
        alpha_new = mu / (Kb + 1e-10)
        
        # Update beta
        Kta = K.T @ alpha_new
        beta_new = nu / (Kta + 1e-10)
        
        delta = jnp.max(jnp.abs(alpha_new - alpha))
        
        return (alpha_new, beta_new, delta), delta
    
    init_carry = (alpha, beta, jnp.inf)
    (alpha, beta, final_delta), _ = lax.scan(
        sinkhorn_step, init_carry, None, length=num_iterations
    )
    
    converged = final_delta < threshold
    
    return alpha, beta, converged


# =============================================================================
# Koopman-Based Transition Kernel
# =============================================================================

def build_koopman_kernel(
    koopman_matrix: Array,
    source_features: Array,
    target_features: Array,
    regularization: float = 0.01,
) -> Array:
    """Build transition kernel from Koopman operator in feature space.
    
    The Koopman operator K acts on observables: (Kg)(x) = E[g(X') | X = x]
    
    In feature space: Ψ(X') ≈ K̃ Ψ(X)
    
    We construct a transition kernel between particles using feature-space
    distances weighted by the Koopman dynamics.
    
    Args:
        koopman_matrix: EDMD Koopman matrix, shape [D, D].
        source_features: Dictionary evaluated at source particles, shape [n, D].
        target_features: Dictionary evaluated at target particles, shape [m, D].
        regularization: Entropic regularization parameter.
        
    Returns:
        Transition kernel K, shape [n, m].
    """
    # Predict features at next time step: Ψ(X') ≈ K̃ Ψ(X)
    predicted_features = source_features @ koopman_matrix.T  # [n, D]
    
    # Compute distances between predictions and actual target features
    # ||Ψ_pred(x_i) - Ψ(y_j)||²
    diff = predicted_features[:, None, :] - target_features[None, :, :]  # [n, m, D]
    dist_sq = jnp.sum(diff ** 2, axis=-1)  # [n, m]
    
    # Gibbs kernel: K_ij ∝ exp(-||Ψ_pred(x_i) - Ψ(y_j)||² / (2ε))
    log_K = -dist_sq / (2 * regularization)
    
    # Normalize rows (transition probabilities)
    log_K = log_K - jax.scipy.special.logsumexp(log_K, axis=1, keepdims=True)
    
    return jnp.exp(log_K)


def build_multistep_kernels(
    koopman_matrix: Array,
    dictionary: Dictionary,
    particles: Array,
    num_steps: int,
    regularization: float = 0.01,
) -> List[Array]:
    """Build transition kernels for each time step.
    
    Args:
        koopman_matrix: Koopman matrix for single step.
        dictionary: Feature dictionary.
        particles: Particle locations, shape [num_particles, dim].
        num_steps: Number of time steps.
        regularization: Entropic regularization.
        
    Returns:
        List of transition kernels, one per step.
    """
    # Evaluate dictionary at particles
    features = dictionary(particles)  # [num_particles, D]
    
    kernels = []
    current_features = features
    
    for step in range(num_steps):
        # Build kernel from current to next step
        next_features = current_features @ koopman_matrix.T
        
        # Kernel based on feature-space distances
        diff = next_features[:, None, :] - features[None, :, :]
        dist_sq = jnp.sum(diff ** 2, axis=-1)
        
        log_K = -dist_sq / (2 * regularization)
        log_K = log_K - jax.scipy.special.logsumexp(log_K, axis=1, keepdims=True)
        K = jnp.exp(log_K)
        
        kernels.append(K)
        current_features = next_features
    
    return kernels


# =============================================================================
# Discrete IPF-Koopman Solver
# =============================================================================

class DiscreteIPFKoopmanSolver:
    """Discrete-time Schrödinger Bridge solver using IPF and Koopman.
    
    This solver is appropriate for:
    - Daily or lower frequency data
    - Noisy real-world observations
    - When you want to avoid continuous-time derivative estimation
    
    Algorithm:
    1. Fit Koopman operator K̃ to historical data via EDMD/optDMD
    2. Construct transition kernels from K̃
    3. Run Sinkhorn/IPF to match source and target marginals
    4. Sample bridge paths using the optimal coupling
    """
    
    def __init__(
        self,
        config: Optional[DiscreteIPFConfig] = None,
    ):
        """Initialize solver.
        
        Args:
            config: Solver configuration.
        """
        self.config = config or DiscreteIPFConfig()
        self._koopman_matrix: Optional[Array] = None
        self._dictionary: Optional[Dictionary] = None
        self._particles: Optional[Array] = None
        self._ipf_result: Optional[DiscreteIPFResult] = None
    
    def fit_koopman(
        self,
        trajectories: Array,
        dt: float,
        key: Optional[PRNGKey] = None,
    ) -> Dict:
        """Fit Koopman operator to trajectory data.
        
        Args:
            trajectories: Historical trajectories, shape [num_traj, num_times, dim].
            dt: Time step.
            key: Random key (for optDMD).
            
        Returns:
            Dictionary with Koopman results.
        """
        dim = trajectories.shape[-1]
        
        # Build dictionary
        if self.config.dictionary_type == 'polynomial':
            self._dictionary = PolynomialDictionary(
                dim=dim,
                degree=self.config.polynomial_degree,
                include_time=False,
            )
        else:
            raise ValueError(f"Unknown dictionary type: {self.config.dictionary_type}")
        
        # Fit Koopman
        if self.config.use_optdmd:
            if key is None:
                key = jax.random.PRNGKey(0)
            
            result = optdmd_from_trajectories(
                trajectories=trajectories,
                dictionary=self._dictionary,
                rank=self.config.koopman_rank,
                dt=dt,
                method='bagging',
                num_bags=self.config.optdmd_bags,
                key=key,
            )
        else:
            # Standard EDMD
            result = optdmd_from_trajectories(
                trajectories=trajectories,
                dictionary=self._dictionary,
                rank=self.config.koopman_rank,
                dt=dt,
                method='standard',
            )
        
        # Store Koopman matrix
        # For EDMD, we need to construct the matrix from eigenvalues/modes
        # This is a simplified version - full implementation would reconstruct K
        self._koopman_result = result
        
        return result
    
    def solve(
        self,
        source_samples: Array,
        target_samples: Array,
        num_steps: int,
        key: PRNGKey,
    ) -> DiscreteIPFResult:
        """Solve the discrete-time SB problem.
        
        Args:
            source_samples: Samples from source distribution, shape [n, dim].
            target_samples: Samples from target distribution, shape [m, dim].
            num_steps: Number of time steps.
            key: Random key.
            
        Returns:
            DiscreteIPFResult with optimal coupling.
        """
        if self._dictionary is None:
            raise ValueError("Must call fit_koopman first")
        
        n_source = len(source_samples)
        n_target = len(target_samples)
        
        # Use source/target samples as particles
        # (Could also use a separate particle set)
        
        # Evaluate dictionary
        source_features = self._dictionary(source_samples)
        target_features = self._dictionary(target_samples)
        
        # Build Koopman-based transition kernel
        # For multi-step, we need the full path
        # Simplified: use single-step kernel repeatedly
        
        # Uniform marginals (assuming equal weights)
        mu = jnp.ones(n_source) / n_source
        nu = jnp.ones(n_target) / n_target
        
        # Build kernel from Koopman dynamics
        # This uses the feature-space Koopman to define transitions
        K_matrix = self._koopman_result.get('K', jnp.eye(source_features.shape[1]))
        
        K = build_koopman_kernel(
            K_matrix,
            source_features,
            target_features,
            regularization=self.config.sinkhorn_regularization,
        )
        
        # Run Sinkhorn
        if self.config.use_log_domain:
            log_K = jnp.log(K + 1e-10)
            log_mu = jnp.log(mu + 1e-10)
            log_nu = jnp.log(nu + 1e-10)
            
            log_alpha, log_beta, converged = sinkhorn_log_domain(
                log_K, log_mu, log_nu,
                num_iterations=self.config.sinkhorn_iterations,
                threshold=self.config.sinkhorn_threshold,
            )
            
            alpha = jnp.exp(log_alpha)
            beta = jnp.exp(log_beta)
        else:
            alpha, beta, converged = sinkhorn_standard(
                K, mu, nu,
                num_iterations=self.config.sinkhorn_iterations,
                threshold=self.config.sinkhorn_threshold,
            )
        
        # Compute optimal coupling
        pi = jnp.diag(alpha) @ K @ jnp.diag(beta)
        
        # Transport cost
        cost = jnp.sum(pi * (-jnp.log(K + 1e-10)))
        
        self._ipf_result = DiscreteIPFResult(
            alpha=alpha,
            beta=beta,
            marginals=[mu, nu],
            transport_cost=float(cost),
            num_iterations=self.config.sinkhorn_iterations,
            converged=converged,
        )
        
        self._source_samples = source_samples
        self._target_samples = target_samples
        
        return self._ipf_result
    
    def sample_paths(
        self,
        key: PRNGKey,
        num_samples: int,
        num_steps: int = 10,
    ) -> BridgePathResult:
        """Sample bridge paths from the optimal coupling.
        
        Args:
            key: Random key.
            num_samples: Number of paths to sample.
            num_steps: Number of intermediate time steps.
            
        Returns:
            BridgePathResult with sampled paths.
        """
        if self._ipf_result is None:
            raise ValueError("Must call solve first")
        
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Compute optimal coupling matrix
        alpha = self._ipf_result.alpha
        beta = self._ipf_result.beta
        
        n_source = len(self._source_samples)
        n_target = len(self._target_samples)
        
        # Rebuild kernel
        source_features = self._dictionary(self._source_samples)
        target_features = self._dictionary(self._target_samples)
        K_matrix = self._koopman_result.get('K', jnp.eye(source_features.shape[1]))
        
        K = build_koopman_kernel(
            K_matrix,
            source_features,
            target_features,
            regularization=self.config.sinkhorn_regularization,
        )
        
        pi = jnp.diag(alpha) @ K @ jnp.diag(beta)
        
        # Sample source-target pairs from coupling
        pi_flat = pi.flatten()
        pi_flat = pi_flat / jnp.sum(pi_flat)  # Normalize
        
        indices = jax.random.choice(
            k1, n_source * n_target,
            shape=(num_samples,),
            p=pi_flat,
        )
        
        source_indices = indices // n_target
        target_indices = indices % n_target
        
        x0 = self._source_samples[source_indices]  # [num_samples, dim]
        x1 = self._target_samples[target_indices]  # [num_samples, dim]
        
        # Interpolate paths (linear + noise for now)
        # More sophisticated: use Koopman dynamics for intermediate steps
        times = jnp.linspace(0, 1, num_steps + 1)
        
        paths = []
        for t in times:
            # Linear interpolation with noise
            mean = (1 - t) * x0 + t * x1
            noise = jax.random.normal(k2, mean.shape) * 0.01 * jnp.sqrt(t * (1 - t) + 1e-6)
            k2, _ = jax.random.split(k2)
            paths.append(mean + noise)
        
        paths = jnp.stack(paths, axis=1)  # [num_samples, num_steps+1, dim]
        
        return BridgePathResult(
            paths=paths,
            log_weights=None,
            source_samples=x0,
            target_samples=x1,
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_discrete_ipf_solver(
    trajectories: Array,
    dt: float,
    config: Optional[DiscreteIPFConfig] = None,
    key: Optional[PRNGKey] = None,
) -> DiscreteIPFKoopmanSolver:
    """Create and initialize a discrete IPF-Koopman solver.
    
    Convenience function that creates solver and fits Koopman in one call.
    
    Args:
        trajectories: Historical trajectories for Koopman fitting.
        dt: Time step.
        config: Solver configuration.
        key: Random key.
        
    Returns:
        Initialized solver with fitted Koopman operator.
    """
    solver = DiscreteIPFKoopmanSolver(config)
    solver.fit_koopman(trajectories, dt, key)
    return solver


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'DiscreteIPFConfig',
    'DiscreteIPFResult',
    'BridgePathResult',
    'DiscreteIPFKoopmanSolver',
    'sinkhorn_log_domain',
    'sinkhorn_standard',
    'build_koopman_kernel',
    'create_discrete_ipf_solver',
]
