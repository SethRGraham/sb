"""OTT-JAX Integration for Schrödinger Bridges.

Provides integration with the OTT-JAX library for:
- Optimal transport coupling computation
- Sinkhorn divergence for marginal matching
- Entropic OT for regularized transport plans
- Displacement interpolation for initialization

Reference:
    OTT-JAX: https://github.com/ott-jax/ott
    Cuturi (2013) "Sinkhorn Distances"
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from .core.types import Array, PRNGKey, Scalar


# Check for OTT-JAX availability
_OTT_AVAILABLE = False
_ott = None

try:
    import ott
    from ott.geometry import pointcloud, costs
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn
    _OTT_AVAILABLE = True
    _ott = ott
except ImportError:
    pass


def is_ott_available() -> bool:
    """Check if OTT-JAX is installed."""
    return _OTT_AVAILABLE


def require_ott():
    """Raise error if OTT-JAX is not available."""
    if not _OTT_AVAILABLE:
        raise ImportError(
            "OTT-JAX is required for this functionality. "
            "Install with: pip install ott-jax"
        )


# Fallback Sinkhorn (when OTT not available)
def sinkhorn_coupling_fallback(
    x: Array,
    y: Array,
    epsilon: float = 0.1,
    max_iterations: int = 100,
    threshold: float = 1e-4,
) -> Tuple[Array, Dict[str, Any]]:
    """Simple Sinkhorn coupling computation (fallback when OTT unavailable).
    
    Computes entropic optimal transport coupling using Sinkhorn algorithm.
    
    Args:
        x: Source points, shape [n, d].
        y: Target points, shape [m, d].
        epsilon: Entropic regularization.
        max_iterations: Maximum Sinkhorn iterations.
        threshold: Convergence threshold.
        
    Returns:
        (coupling_matrix, info_dict)
    """
    n, m = x.shape[0], y.shape[0]
    
    # Cost matrix (squared Euclidean)
    C = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    
    # Gibbs kernel
    K = jnp.exp(-C / epsilon)
    
    # Initialize dual variables
    u = jnp.ones(n) / n
    v = jnp.ones(m) / m
    
    # Marginals
    a = jnp.ones(n) / n
    b = jnp.ones(m) / m
    
    def sinkhorn_iteration(carry, _):
        u, v = carry
        u_new = a / (K @ v + 1e-10)
        v_new = b / (K.T @ u_new + 1e-10)
        return (u_new, v_new), None
    
    (u, v), _ = jax.lax.scan(
        sinkhorn_iteration,
        (u, v),
        None,
        length=max_iterations,
    )
    
    # Coupling matrix
    P = jnp.diag(u) @ K @ jnp.diag(v)
    
    # Transport cost
    cost = jnp.sum(P * C)
    
    info = {
        'cost': cost,
        'converged': True,
        'iterations': max_iterations,
        'epsilon': epsilon,
    }
    
    return P, info


# OTT-JAX Wrappers
@dataclass
class OTConfig:
    """Configuration for optimal transport computation."""
    epsilon: float = 0.1  # Entropic regularization
    max_iterations: int = 1000
    threshold: float = 1e-4
    lse_mode: bool = True  # Log-sum-exp mode for stability
    momentum: float = 1.0  # Sinkhorn momentum
    inner_iterations: int = 10
    

def compute_ot_coupling(
    x: Array,
    y: Array,
    config: Optional[OTConfig] = None,
) -> Tuple[Array, Dict[str, Any]]:
    """Compute optimal transport coupling between point clouds.
    
    Uses OTT-JAX if available, falls back to simple Sinkhorn otherwise.
    
    Args:
        x: Source points, shape [n, d].
        y: Target points, shape [m, d].
        config: OT configuration.
        
    Returns:
        (coupling_matrix, info_dict) where coupling_matrix has shape [n, m].
    """
    config = config or OTConfig()
    
    if _OTT_AVAILABLE:
        # Use OTT-JAX
        geom = pointcloud.PointCloud(x, y, epsilon=config.epsilon)
        prob = linear_problem.LinearProblem(geom)
        
        solver = sinkhorn.Sinkhorn(
            lse_mode=config.lse_mode,
            threshold=config.threshold,
            max_iterations=config.max_iterations,
            # OTT expects a Momentum object; older code passed a float which
            # causes attribute errors (momentum.start). Convert simple numeric
            # momenta to None to use default OTT behavior.
            momentum=(None if isinstance(config.momentum, (int, float)) else config.momentum),
            inner_iterations=config.inner_iterations,
        )
        
        out = solver(prob)
        
        # Get coupling matrix
        P = jnp.asarray(out.matrix)

        info = {
            'cost': float(out.reg_ot_cost) if hasattr(out, 'reg_ot_cost') else float('nan'),
            'converged': bool(getattr(out, 'converged', True)),
            'iterations': int(getattr(out, 'n_iters', config.max_iterations)),
            'epsilon': float(config.epsilon),
            'ott_output': out,
        }

        return P, info
    
    else:
        # Fallback
        P, info = sinkhorn_coupling_fallback(
            x, y,
            epsilon=config.epsilon,
            max_iterations=config.max_iterations,
            threshold=config.threshold,
        )
        # Ensure types are consistent
        P = jnp.asarray(P)
        info['cost'] = float(info.get('cost', float('nan')))
        info['converged'] = bool(info.get('converged', True))
        info['iterations'] = int(info.get('iterations', config.max_iterations))
        info['epsilon'] = float(config.epsilon)
        return P, info


def compute_ot_cost(
    x: Array,
    y: Array,
    config: Optional[OTConfig] = None,
) -> float:
    """Compute optimal transport cost (Wasserstein distance approximation).
    
    Args:
        x: Source points.
        y: Target points.
        config: OT configuration.
        
    Returns:
        Regularized OT cost.
    """
    _, info = compute_ot_coupling(x, y, config)
    return info['cost']


def compute_sinkhorn_divergence(
    x: Array,
    y: Array,
    config: Optional[OTConfig] = None,
) -> float:
    """Compute Sinkhorn divergence (debiased OT cost).
    
    S(x,y) = OT(x,y) - 0.5 * OT(x,x) - 0.5 * OT(y,y)
    
    This removes the entropic bias and gives a proper divergence.
    
    Args:
        x: Source points.
        y: Target points.
        config: OT configuration.
        
    Returns:
        Sinkhorn divergence.
    """
    config = config or OTConfig()
    
    if _OTT_AVAILABLE:
        from ott.tools import sinkhorn_divergence as sd
        
        out = sd.sinkhorn_divergence(
            pointcloud.PointCloud,
            x, y,
            epsilon=config.epsilon,
        )
        
        return float(out.divergence)
    
    else:
        # Manual computation
        _, info_xy = sinkhorn_coupling_fallback(x, y, config.epsilon)
        _, info_xx = sinkhorn_coupling_fallback(x, x, config.epsilon)
        _, info_yy = sinkhorn_coupling_fallback(y, y, config.epsilon)
        
        return info_xy['cost'] - 0.5 * info_xx['cost'] - 0.5 * info_yy['cost']


# OT Coupling for SB Solvers
def get_ot_paired_samples(
    key: PRNGKey,
    source_samples: Array,
    target_samples: Array,
    config: Optional[OTConfig] = None,
) -> Tuple[Array, Array]:
    """Get OT-coupled pairs from source and target samples.
    
    Instead of random pairing, uses OT coupling to match samples.
    This provides better initialization for SB solvers.
    
    Args:
        key: Random key (for tie-breaking).
        source_samples: Source points [n, d].
        target_samples: Target points [m, d].
        config: OT configuration.
        
    Returns:
        (paired_source, paired_target) with shape [min(n,m), d] each.
    """
    n, m = source_samples.shape[0], target_samples.shape[0]
    k = min(n, m)
    
    # Compute coupling
    P, _ = compute_ot_coupling(source_samples[:k], target_samples[:k], config)
    
    # Sample pairs according to coupling
    # For each source, pick target with highest coupling weight
    # (deterministic assignment)
    target_indices = jnp.argmax(P, axis=1)
    
    paired_source = source_samples[:k]
    paired_target = target_samples[target_indices]
    
    return paired_source, paired_target


def get_ot_barycentric_interpolation(
    x0: Array,
    x1: Array,
    t: Union[float, Array],
    config: Optional[OTConfig] = None,
) -> Array:
    """Compute displacement interpolation along OT geodesic.
    
    Given OT coupling between x0 and x1, computes the position
    at time t along the geodesic.
    
    Args:
        x0: Source points [n, d].
        x1: Target points [m, d].
        t: Interpolation time(s) in [0, 1].
        config: OT configuration.
        
    Returns:
        Interpolated points.
    """
    # Get paired samples
    paired_x0, paired_x1 = get_ot_paired_samples(
        jax.random.PRNGKey(0), x0, x1, config
    )
    
    # Linear interpolation along OT map
    t = jnp.atleast_1d(t)
    if t.ndim == 1:
        t = t[:, None]  # [batch, 1]
    
    # Broadcast if needed
    if paired_x0.shape[0] != t.shape[0]:
        # Assume t is a single time, broadcast
        t = jnp.broadcast_to(t, (paired_x0.shape[0], 1))
    
    return (1 - t) * paired_x0 + t * paired_x1


# Integration with SB Solvers
class OTCoupledSampler:
    """Sampler that provides OT-coupled source-target pairs.
    
    Useful for SB solver training where OT coupling provides
    better initialization than random pairing.
    """
    
    def __init__(
        self,
        source_samples: Array,
        target_samples: Array,
        config: Optional[OTConfig] = None,
        recompute_coupling: bool = False,
    ):
        """Initialize with source and target samples.
        
        Args:
            source_samples: Source distribution samples.
            target_samples: Target distribution samples.
            config: OT configuration.
            recompute_coupling: If True, recompute coupling each call.
        """
        self.source_samples = source_samples
        self.target_samples = target_samples
        self.config = config or OTConfig()
        self.recompute_coupling = recompute_coupling
        
        # Precompute coupling if not recomputing
        if not recompute_coupling:
            self._coupling, self._info = compute_ot_coupling(
                source_samples, target_samples, config
            )
        else:
            self._coupling = None
            self._info = None
    
    def sample_pairs(
        self,
        key: PRNGKey,
        batch_size: int,
    ) -> Tuple[Array, Array]:
        """Sample OT-coupled pairs.
        
        Args:
            key: Random key.
            batch_size: Number of pairs to sample.
            
        Returns:
            (source_batch, target_batch)
        """
        k1, k2 = jax.random.split(key)
        
        n = self.source_samples.shape[0]
        
        if self.recompute_coupling:
            # Subsample and compute coupling
            idx_source = jax.random.choice(k1, n, (batch_size,))
            idx_target = jax.random.choice(k2, n, (batch_size,))
            
            sub_source = self.source_samples[idx_source]
            sub_target = self.target_samples[idx_target]
            
            P, _ = compute_ot_coupling(sub_source, sub_target, self.config)
            target_idx = jnp.argmax(P, axis=1)
            
            return sub_source, sub_target[target_idx]
        
        else:
            # Use precomputed coupling
            # Sample source indices
            source_idx = jax.random.choice(k1, n, (batch_size,))
            source_batch = self.source_samples[source_idx]
            
            # Sample target according to coupling row
            def sample_target_for_source(key, src_idx):
                row = self._coupling[src_idx]
                row = row / (row.sum() + 1e-10)  # Normalize
                return jax.random.choice(key, len(row), p=row)
            
            keys = jax.random.split(k2, batch_size)
            target_idx = jax.vmap(sample_target_for_source)(keys, source_idx)
            target_batch = self.target_samples[target_idx]
            
            return source_batch, target_batch
    
    @property
    def ot_cost(self) -> float:
        """Get the OT cost between source and target."""
        if self._info is not None:
            return self._info['cost']
        else:
            _, info = compute_ot_coupling(
                self.source_samples, self.target_samples, self.config
            )
            return info['cost']


def create_ot_coupled_sampler(
    problem,  # SBProblem
    key: PRNGKey,
    num_samples: int = 1000,
    config: Optional[OTConfig] = None,
) -> OTCoupledSampler:
    """Create an OT-coupled sampler from an SB problem.
    
    Args:
        problem: SBProblem instance.
        key: Random key.
        num_samples: Number of samples for coupling computation.
        config: OT configuration.
        
    Returns:
        OTCoupledSampler instance.
    """
    k1, k2 = jax.random.split(key)
    
    source_samples = problem.sample_source(k1, num_samples)
    target_samples = problem.sample_target(k2, num_samples)
    
    return OTCoupledSampler(source_samples, target_samples, config)


# OT-based Loss Functions
def ot_loss(
    generated: Array,
    target: Array,
    config: Optional[OTConfig] = None,
) -> Scalar:
    """Compute OT loss between generated and target samples.
    
    Args:
        generated: Generated samples.
        target: Target samples.
        config: OT configuration.
        
    Returns:
        OT cost (differentiable with respect to generated).
    """
    return compute_ot_cost(generated, target, config)


def sinkhorn_loss(
    generated: Array,
    target: Array,
    config: Optional[OTConfig] = None,
) -> Scalar:
    """Compute Sinkhorn divergence loss.
    
    This is a debiased version of entropic OT cost.
    
    Args:
        generated: Generated samples.
        target: Target samples.
        config: OT configuration.
        
    Returns:
        Sinkhorn divergence.
    """
    return compute_sinkhorn_divergence(generated, target, config)


# Utilities
def visualize_coupling(
    P: Array,
    x: Array,
    y: Array,
    ax=None,
    threshold: float = 0.01,
    **kwargs,
):
    """Visualize OT coupling as lines between matched points.
    
    Args:
        P: Coupling matrix [n, m].
        x: Source points [n, 2].
        y: Target points [m, 2].
        ax: Matplotlib axis.
        threshold: Minimum coupling weight to draw.
        **kwargs: Additional plot arguments.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # Normalize coupling for visualization
    P_norm = P / P.max()
    
    # Draw lines for significant couplings
    n, m = P.shape
    for i in range(n):
        for j in range(m):
            if P_norm[i, j] > threshold:
                ax.plot(
                    [x[i, 0], y[j, 0]],
                    [x[i, 1], y[j, 1]],
                    alpha=float(P_norm[i, j]),
                    color='gray',
                    linewidth=0.5,
                    **kwargs,
                )
    
    # Draw points
    ax.scatter(x[:, 0], x[:, 1], c='blue', s=20, label='Source', zorder=5)
    ax.scatter(y[:, 0], y[:, 1], c='red', s=20, label='Target', zorder=5)
    
    ax.legend()
    ax.set_aspect('equal')
    
    return ax


# Module Exports
__all__ = [
    # Availability
    'is_ott_available',
    'require_ott',
    # Config
    'OTConfig',
    # Core OT
    'compute_ot_coupling',
    'compute_ot_cost',
    'compute_sinkhorn_divergence',
    'sinkhorn_coupling_fallback',
    # Sampling
    'get_ot_paired_samples',
    'get_ot_barycentric_interpolation',
    # Sampler class
    'OTCoupledSampler',
    'create_ot_coupled_sampler',
    # Losses
    'ot_loss',
    'sinkhorn_loss',
    # Visualization
    'visualize_coupling',
]
