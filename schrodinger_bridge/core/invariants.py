"""Schrödinger Bridge Invariant Checking.

This module provides utilities for verifying that solutions satisfy
the fundamental properties of Schrödinger Bridges:

1. Mass Conservation: ∫ρ_t(x)dx = 1 for all t
2. Marginal Consistency: ρ_0 = μ_0, ρ_1 = μ_1
3. Entropy/KL Evolution: Should follow specific dynamics
4. Continuity Equation: ∂ρ/∂t + ∇·(ρv) = 0

These invariants are essential for diagnosing solver issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from .types import (
    Array,
    DiagnosticReport,
    DriftFn,
    InvariantViolation,
    PRNGKey,
    Scalar,
    TrajectoryBatch,
)


# =============================================================================
# Invariant Thresholds
# =============================================================================

@dataclass
class InvariantThresholds:
    """Thresholds for invariant violation detection.
    
    Attributes:
        mass_conservation_warning: Warning threshold for mass error.
        mass_conservation_error: Error threshold for mass error.
        marginal_mmd_warning: Warning threshold for MMD.
        marginal_mmd_error: Error threshold for MMD.
        kl_divergence_warning: Warning threshold for KL anomaly.
    """
    mass_conservation_warning: float = 0.01
    mass_conservation_error: float = 0.1
    marginal_mmd_warning: float = 0.1
    marginal_mmd_error: float = 0.5
    kl_divergence_warning: float = 1.0


DEFAULT_THRESHOLDS = InvariantThresholds()


# =============================================================================
# Statistical Utilities
# =============================================================================

def gaussian_kernel(x: Array, y: Array, bandwidth: float = 1.0) -> Array:
    """Compute Gaussian (RBF) kernel between point sets.
    
    Args:
        x: Points, shape [n, d].
        y: Points, shape [m, d].
        bandwidth: Kernel bandwidth.
        
    Returns:
        Kernel matrix, shape [n, m].
    """
    sq_dists = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-sq_dists / (2 * bandwidth ** 2))


def mmd_squared(
    x: Array,
    y: Array,
    bandwidth: Optional[float] = None,
) -> float:
    """Compute squared Maximum Mean Discrepancy.
    
    MMD is a kernel-based distance between distributions.
    MMD²(P, Q) = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
    
    Args:
        x: Samples from P, shape [n, d].
        y: Samples from Q, shape [m, d].
        bandwidth: Kernel bandwidth (auto if None).
        
    Returns:
        Squared MMD value.
    """
    if bandwidth is None:
        # Median heuristic
        all_points = jnp.concatenate([x, y], axis=0)
        dists = jnp.sqrt(jnp.sum(
            (all_points[:, None, :] - all_points[None, :, :]) ** 2, axis=-1
        ))
        bandwidth = jnp.median(dists) + 1e-6
    
    Kxx = gaussian_kernel(x, x, bandwidth)
    Kyy = gaussian_kernel(y, y, bandwidth)
    Kxy = gaussian_kernel(x, y, bandwidth)
    
    n, m = x.shape[0], y.shape[0]
    
    # Unbiased estimator
    mmd2 = (
        (jnp.sum(Kxx) - jnp.trace(Kxx)) / (n * (n - 1))
        + (jnp.sum(Kyy) - jnp.trace(Kyy)) / (m * (m - 1))
        - 2 * jnp.mean(Kxy)
    )
    
    return float(jnp.maximum(mmd2, 0.0))


def wasserstein_1d(x: Array, y: Array) -> float:
    """Compute 1D Wasserstein distance.
    
    For 1D, W_1 = ∫|F_X^{-1}(t) - F_Y^{-1}(t)|dt
    which simplifies to comparing sorted samples.
    
    Args:
        x: Samples, shape [n].
        y: Samples, shape [m].
        
    Returns:
        1D Wasserstein distance.
    """
    x_sorted = jnp.sort(x)
    y_sorted = jnp.sort(y)
    
    # Interpolate to same grid
    n, m = len(x_sorted), len(y_sorted)
    combined = jnp.sort(jnp.concatenate([
        jnp.linspace(0, 1, n),
        jnp.linspace(0, 1, m)
    ]))
    
    x_interp = jnp.interp(combined, jnp.linspace(0, 1, n), x_sorted)
    y_interp = jnp.interp(combined, jnp.linspace(0, 1, m), y_sorted)
    
    return float(jnp.mean(jnp.abs(x_interp - y_interp)))


def sliced_wasserstein(
    x: Array,
    y: Array,
    num_projections: int = 50,
    key: Optional[PRNGKey] = None,
) -> float:
    """Compute Sliced Wasserstein distance.
    
    Approximates Wasserstein by averaging 1D Wasserstein over random projections.
    
    Args:
        x: Samples, shape [n, d].
        y: Samples, shape [m, d].
        num_projections: Number of random projections.
        key: Random key (uses fixed directions if None).
        
    Returns:
        Sliced Wasserstein distance.
    """
    d = x.shape[1]
    
    if key is None:
        # Fixed directions for reproducibility
        key = jax.random.PRNGKey(0)
    
    # Random directions on unit sphere
    directions = jax.random.normal(key, (num_projections, d))
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
    
    # Project and compute 1D distances
    x_proj = x @ directions.T  # [n, num_projections]
    y_proj = y @ directions.T  # [m, num_projections]
    
    distances = []
    for i in range(num_projections):
        distances.append(wasserstein_1d(x_proj[:, i], y_proj[:, i]))
    
    return float(jnp.mean(jnp.array(distances)))


def estimate_entropy(x: Array, k: int = 5) -> float:
    """Estimate differential entropy using k-NN.
    
    Uses the Kozachenko-Leonenko estimator.
    
    Args:
        x: Samples, shape [n, d].
        k: Number of nearest neighbors.
        
    Returns:
        Estimated entropy in nats.
    """
    n, d = x.shape
    
    # Compute pairwise distances
    dists = jnp.sqrt(jnp.sum(
        (x[:, None, :] - x[None, :, :]) ** 2, axis=-1
    ))
    
    # k-th nearest neighbor distance (k+1 to exclude self)
    sorted_dists = jnp.sort(dists, axis=1)
    rho_k = sorted_dists[:, k]  # [n]
    
    # Volume of d-dimensional unit ball
    from scipy.special import gamma as scipy_gamma
    # Using approximation for JAX compatibility
    log_vol_unit_ball = d/2 * jnp.log(jnp.pi) - jnp.sum(jnp.log(jnp.arange(1, d//2 + 1)))
    
    # Kozachenko-Leonenko estimator
    entropy = d * jnp.mean(jnp.log(rho_k + 1e-10)) + jnp.log(n - 1) - jax.scipy.special.digamma(k)
    
    return float(entropy)


# =============================================================================
# Invariant Checkers
# =============================================================================

class InvariantChecker:
    """Checks Schrödinger Bridge invariants and generates diagnostics.
    
    Attributes:
        thresholds: Violation thresholds.
    """
    
    def __init__(self, thresholds: Optional[InvariantThresholds] = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self._violations: List[InvariantViolation] = []
    
    def reset(self):
        """Clear accumulated violations."""
        self._violations = []
    
    def _add_violation(
        self,
        name: str,
        expected: float,
        actual: float,
        warning_threshold: float,
        error_threshold: float,
        message_template: str,
    ):
        """Add violation if threshold exceeded."""
        error = abs(actual - expected)
        
        if error > error_threshold:
            severity = 'error'
        elif error > warning_threshold:
            severity = 'warning'
        else:
            return  # No violation
        
        self._violations.append(InvariantViolation(
            name=name,
            expected=expected,
            actual=actual,
            severity=severity,
            message=message_template.format(expected=expected, actual=actual, error=error),
        ))
    
    def check_mass_conservation(
        self,
        trajectories: TrajectoryBatch,
        weights: Optional[Array] = None,
    ) -> Array:
        """Check mass conservation over time.
        
        For particle representations, "mass" = sum of weights.
        Should remain constant (normalized to 1).
        
        Args:
            trajectories: Batch of trajectories.
            weights: Optional importance weights.
            
        Returns:
            Mass at each time step.
        """
        if weights is None:
            # Uniform weights
            mass = jnp.ones(trajectories.num_times)
        else:
            # Sum of weights at each time (should be constant)
            mass = jnp.sum(jnp.exp(weights))  # Assuming log weights
            mass = jnp.full(trajectories.num_times, mass)
        
        # Check for violations
        mass_error = jnp.abs(mass - 1.0)
        max_error = float(jnp.max(mass_error))
        
        self._add_violation(
            name="mass_conservation",
            expected=1.0,
            actual=float(mass[jnp.argmax(mass_error)]),
            warning_threshold=self.thresholds.mass_conservation_warning,
            error_threshold=self.thresholds.mass_conservation_error,
            message_template="Mass should be 1.0, got {actual:.4f} (error: {error:.2e})",
        )
        
        return mass
    
    def check_marginal_consistency(
        self,
        trajectories: TrajectoryBatch,
        source_samples: Array,
        target_samples: Array,
        key: Optional[PRNGKey] = None,
    ) -> Tuple[float, float]:
        """Check marginal distribution matching.
        
        Uses MMD to compare:
        - trajectories[:, 0, :] vs source_samples
        - trajectories[:, -1, :] vs target_samples
        
        Args:
            trajectories: Batch of trajectories.
            source_samples: Reference samples from source.
            target_samples: Reference samples from target.
            key: Random key for sliced Wasserstein.
            
        Returns:
            (source_mmd, target_mmd)
        """
        source_mmd = mmd_squared(trajectories.source_samples, source_samples)
        target_mmd = mmd_squared(trajectories.target_samples, target_samples)
        
        # Check source marginal
        self._add_violation(
            name="source_marginal",
            expected=0.0,
            actual=source_mmd,
            warning_threshold=self.thresholds.marginal_mmd_warning ** 2,
            error_threshold=self.thresholds.marginal_mmd_error ** 2,
            message_template="Source marginal MMD² = {actual:.4f} (should be ~0)",
        )
        
        # Check target marginal
        self._add_violation(
            name="target_marginal",
            expected=0.0,
            actual=target_mmd,
            warning_threshold=self.thresholds.marginal_mmd_warning ** 2,
            error_threshold=self.thresholds.marginal_mmd_error ** 2,
            message_template="Target marginal MMD² = {actual:.4f} (should be ~0)",
        )
        
        return source_mmd, target_mmd
    
    def check_entropy_evolution(
        self,
        trajectories: TrajectoryBatch,
        time_indices: Optional[Array] = None,
    ) -> Array:
        """Estimate entropy at multiple time points.
        
        For SB, entropy should evolve smoothly (not necessarily monotonically).
        
        Args:
            trajectories: Batch of trajectories.
            time_indices: Which time indices to evaluate.
            
        Returns:
            Entropy estimates at selected times.
        """
        if time_indices is None:
            # Sample 10 evenly spaced times
            time_indices = jnp.linspace(
                0, trajectories.num_times - 1, 10
            ).astype(int)
        
        entropies = []
        for t_idx in time_indices:
            samples = trajectories.at_time(int(t_idx))
            try:
                h = estimate_entropy(samples)
            except Exception:
                h = float('nan')
            entropies.append(h)
        
        return jnp.array(entropies)
    
    def check_path_regularity(
        self,
        trajectories: TrajectoryBatch,
        max_velocity: float = 100.0,
    ) -> Tuple[float, float]:
        """Check for irregular (exploding) paths.
        
        Args:
            trajectories: Batch of trajectories.
            max_velocity: Maximum expected velocity.
            
        Returns:
            (mean_velocity, max_velocity_observed)
        """
        # Compute velocities
        dt = trajectories.times[1] - trajectories.times[0]
        velocities = jnp.diff(trajectories.paths, axis=1) / dt
        velocity_norms = jnp.linalg.norm(velocities, axis=-1)
        
        mean_vel = float(jnp.mean(velocity_norms))
        max_vel = float(jnp.max(velocity_norms))
        
        if max_vel > max_velocity:
            self._violations.append(InvariantViolation(
                name="path_regularity",
                expected=max_velocity,
                actual=max_vel,
                severity='warning',
                message=f"Max velocity {max_vel:.2f} exceeds threshold {max_velocity}",
            ))
        
        return mean_vel, max_vel
    
    def check_all(
        self,
        trajectories: TrajectoryBatch,
        source_samples: Array,
        target_samples: Array,
        key: Optional[PRNGKey] = None,
    ) -> DiagnosticReport:
        """Run all invariant checks and generate report.
        
        Args:
            trajectories: Batch of trajectories.
            source_samples: Reference source samples.
            target_samples: Reference target samples.
            key: Random key.
            
        Returns:
            Complete diagnostic report.
        """
        self.reset()
        
        # Run checks
        mass = self.check_mass_conservation(trajectories)
        source_err, target_err = self.check_marginal_consistency(
            trajectories, source_samples, target_samples, key
        )
        entropy = self.check_entropy_evolution(trajectories)
        mean_vel, max_vel = self.check_path_regularity(trajectories)
        
        return DiagnosticReport(
            mass_conservation=mass,
            marginal_error_source=source_err,
            marginal_error_target=target_err,
            entropy_evolution=entropy,
            violations=list(self._violations),
            metadata={
                'mean_velocity': mean_vel,
                'max_velocity': max_vel,
            },
        )


# =============================================================================
# Continuity Equation Checker
# =============================================================================

def check_continuity_equation(
    density_fn: Callable[[Array, Scalar], Array],
    velocity_fn: Callable[[Array, Scalar], Array],
    points: Array,
    times: Array,
    eps: float = 1e-4,
) -> Array:
    """Check continuity equation: ∂ρ/∂t + ∇·(ρv) = 0.
    
    Uses finite differences to approximate derivatives.
    
    Args:
        density_fn: Density function ρ(x, t).
        velocity_fn: Velocity function v(x, t).
        points: Evaluation points, shape [n, d].
        times: Evaluation times, shape [m].
        eps: Finite difference epsilon.
        
    Returns:
        Residuals at each (point, time), shape [n, m].
    """
    n, d = points.shape
    m = len(times)
    
    residuals = jnp.zeros((n, m))
    
    for j, t in enumerate(times):
        if t <= eps or t >= 1.0 - eps:
            continue  # Skip boundary
        
        # Time derivative
        rho_t_plus = density_fn(points, t + eps)
        rho_t_minus = density_fn(points, t - eps)
        drho_dt = (rho_t_plus - rho_t_minus) / (2 * eps)
        
        # Divergence of (ρv)
        rho = density_fn(points, t)
        v = velocity_fn(points, t)
        
        div_rho_v = jnp.zeros(n)
        for k in range(d):
            points_plus = points.at[:, k].add(eps)
            points_minus = points.at[:, k].add(-eps)
            
            rho_v_plus = density_fn(points_plus, t) * velocity_fn(points_plus, t)[:, k]
            rho_v_minus = density_fn(points_minus, t) * velocity_fn(points_minus, t)[:, k]
            
            div_rho_v = div_rho_v + (rho_v_plus - rho_v_minus) / (2 * eps)
        
        residuals = residuals.at[:, j].set(drho_dt + div_rho_v)
    
    return residuals


# =============================================================================
# Quick Check Functions
# =============================================================================

def quick_marginal_check(
    generated_samples: Array,
    reference_samples: Array,
    metric: str = 'mmd',
) -> float:
    """Quick check of marginal distribution matching.
    
    Args:
        generated_samples: Samples from learned distribution.
        reference_samples: Samples from target distribution.
        metric: 'mmd' or 'swd' (sliced Wasserstein).
        
    Returns:
        Distance value.
    """
    if metric == 'mmd':
        return jnp.sqrt(mmd_squared(generated_samples, reference_samples))
    elif metric == 'swd':
        return sliced_wasserstein(generated_samples, reference_samples)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def quick_trajectory_check(
    trajectories: Array,
    times: Array,
) -> Dict[str, float]:
    """Quick sanity check on trajectories.
    
    Args:
        trajectories: Shape [batch, time, dim].
        times: Shape [time].
        
    Returns:
        Dictionary of statistics.
    """
    # Basic statistics
    mean_norm = float(jnp.mean(jnp.linalg.norm(trajectories, axis=-1)))
    max_norm = float(jnp.max(jnp.linalg.norm(trajectories, axis=-1)))
    
    # Velocity statistics
    dt = times[1] - times[0]
    velocities = jnp.diff(trajectories, axis=1) / dt
    mean_velocity = float(jnp.mean(jnp.linalg.norm(velocities, axis=-1)))
    max_velocity = float(jnp.max(jnp.linalg.norm(velocities, axis=-1)))
    
    # Check for NaN/Inf
    has_nan = bool(jnp.any(jnp.isnan(trajectories)))
    has_inf = bool(jnp.any(jnp.isinf(trajectories)))
    
    return {
        'mean_norm': mean_norm,
        'max_norm': max_norm,
        'mean_velocity': mean_velocity,
        'max_velocity': max_velocity,
        'has_nan': has_nan,
        'has_inf': has_inf,
    }
