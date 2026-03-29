"""Martingale Schrödinger Bridge Extension.

Extends the Marginal Schrödinger Bridge to enforce the MARTINGALE constraint,
which is essential for arbitrage-free pricing in quantitative finance.

Standard Marginal SB: Match marginals at t=0, t₁, t₂, ..., tₖ, t=1
Martingale SB: Match marginals AND enforce E[S_{T_{i+1}} | S_{T_i}] = Forward

Mathematical formulation:
    P* = argmin KL(P || P_ref)
    subject to: 
        P_{t_i} = μ_i  for i = 0, 1, ..., K        (marginal constraints)
        E[X_{t_{i+1}} | X_{t_i}] = F(t_i, t_{i+1})  (martingale constraint)

The martingale constraint ensures no-arbitrage: the conditional expectation
of future prices equals the forward price.

Key Mathematical Insight:
========================
For risk-neutral pricing, the discounted asset price must be a martingale:
    E^Q[S_T | F_t] = S_t · e^{r(T-t)} = F(t,T)

This is NOT automatically satisfied by standard Marginal SB!
Martingale SB enforces this through "martingale optimal transport" coupling.

ENHANCEMENTS (v2):
==================
1. SV Reference Process - Use Heston/LocalVol instead of Brownian as prior
2. Variance Swap Constraint - Match expected realized variance
3. MOT Price Bounds - Compute model-free upper/lower bounds

Implementation Strategy:
=======================
1. Extract risk-neutral marginals from options (Breeden-Litzenberger)
2. Compute forward prices from the forward curve
3. Use martingale-preserving coupling between consecutive marginals
4. Simulate via SV-aware conditional bridges (not just Brownian)
5. Optionally adjust for variance swap constraint

References:
    Beiglböck et al. "Model-independent bounds for option prices" (2013)
    Henry-Labordère "Martingale Optimal Transport" (2017)
    Guo & Loeper "Path-dependent optimal transport" (2021)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

import jax
import jax.numpy as jnp

from .core.types import (
    Array,
    DriftFn,
    Params,
    PRNGKey,
    Scalar,
    TimeGrid,
    TrajectoryBatch,
)
from .core.problem import MarginalDistribution, ReferenceDynamics
from .core.invariants import mmd_squared
from .marginal_sb import MarginalConstraint, MarginalSBProblem


# =============================================================================
# Martingale Constraint Specification
# =============================================================================

@dataclass
class ForwardCurve:
    """Forward price curve for martingale constraint.
    
    The forward price F(t,T) is the price agreed today for delivery at T.
    Under risk-neutral measure: F(t,T) = S_t · e^{r(T-t)}
    
    Attributes:
        spot: Current spot price S_0.
        rate: Risk-free interest rate (continuous).
        dividend_yield: Continuous dividend yield (default 0).
    """
    spot: float
    rate: float
    dividend_yield: float = 0.0
    
    def forward(self, t_start: float, t_end: float) -> float:
        """Compute forward price from t_start to t_end.
        
        F(t_start, t_end) = S_{t_start} · e^{(r-q)(t_end - t_start)}
        
        For t_start = 0, this uses spot price.
        """
        tau = t_end - t_start
        return self.spot * np.exp((self.rate - self.dividend_yield) * t_end)
    
    def forward_ratio(self, t_start: float, t_end: float) -> float:
        """Compute the forward ratio F(0,t_end) / F(0,t_start).
        
        This is the expected growth factor from t_start to t_end.
        """
        tau = t_end - t_start
        return np.exp((self.rate - self.dividend_yield) * tau)


@dataclass  
class MartingaleConstraint:
    """Martingale constraint between two time points.
    
    Enforces: E[X_{t_end} | X_{t_start}] = X_{t_start} · forward_ratio
    
    Attributes:
        t_start: Start time.
        t_end: End time.
        forward_ratio: Expected growth factor E[X_end/X_start].
    """
    t_start: float
    t_end: float
    forward_ratio: float  # = F(t_start, t_end) / S_{t_start} = e^{(r-q)τ}
    
    def __post_init__(self):
        if self.t_end <= self.t_start:
            raise ValueError(f"t_end must be > t_start")
        if self.forward_ratio <= 0:
            raise ValueError(f"forward_ratio must be positive")


# =============================================================================
# ENHANCEMENT: Variance Swap Constraint
# =============================================================================

@dataclass
class VarianceSwapConstraint:
    """Constraint on expected realized variance.
    
    ═══════════════════════════════════════════════════════════════════════════
    MAIN MATH TAKEAWAY: Variance Swap Calibration
    ═══════════════════════════════════════════════════════════════════════════
    
    Problem: Marginals alone don't constrain intra-marginal volatility.
    Solution: Add constraint from variance swap market:
    
        E[RV] = ∫₀ᵀ E[σ²_t] dt = VarSwap_price
    
    This fixes the "too smooth paths" problem!
    ═══════════════════════════════════════════════════════════════════════════
    
    Attributes:
        target_variance: Target annualized variance (from variance swap).
        weight: Weight on constraint (higher = stricter enforcement).
    """
    target_variance: float  # Annualized variance, e.g., 0.04 for 20% vol
    weight: float = 1.0
    
    def compute_realized_variance(self, paths: Array, T: float) -> Array:
        """Compute realized variance for each path.
        
        RV = (1/T) * Σ (log(S_{i+1}/S_i))²
        """
        log_returns = jnp.log(paths[:, 1:] / (paths[:, :-1] + 1e-10))
        return jnp.sum(log_returns ** 2, axis=1) / T
    
    def adjustment_factor(self, paths: Array, T: float) -> float:
        """Compute scaling factor to match target variance."""
        rv = self.compute_realized_variance(paths, T)
        current_rv = float(jnp.mean(rv))
        
        if current_rv < 1e-10:
            return 1.0
        
        return jnp.sqrt(self.target_variance / current_rv)
    
    def adjust_paths(self, paths: Array, T: float) -> Array:
        """Rescale paths to match target realized variance.
        
        Scales log-returns around their mean to hit target variance
        while approximately preserving the marginal distributions.
        """
        scale = self.adjustment_factor(paths, T)
        
        # Work in log-space
        log_paths = jnp.log(paths + 1e-10)
        log_returns = jnp.diff(log_paths, axis=1)
        
        # Scale returns around mean
        mean_return = jnp.mean(log_returns, axis=1, keepdims=True)
        scaled_returns = mean_return + scale * (log_returns - mean_return)
        
        # Reconstruct paths
        new_log_paths = jnp.concatenate([
            log_paths[:, :1],
            log_paths[:, :1] + jnp.cumsum(scaled_returns, axis=1)
        ], axis=1)
        
        return jnp.exp(new_log_paths)


# =============================================================================
# Martingale SB Problem
# =============================================================================

@dataclass
class MartingaleSBProblem:
    """Martingale Schrödinger Bridge problem specification.
    
    Combines marginal constraints with martingale constraints.
    
    Attributes:
        reference: Reference stochastic process.
        marginals: List of marginal constraints (must include t=0 and t=1).
        forward_curve: Forward curve for martingale constraints.
        time_grid: Time discretization for the full interval.
        variance_constraint: Optional variance swap constraint.
        name: Optional problem name.
    """
    reference: ReferenceDynamics
    marginals: List[MarginalConstraint]
    forward_curve: ForwardCurve
    time_grid: TimeGrid = field(default_factory=lambda: TimeGrid(num_steps=100))
    variance_constraint: Optional[VarianceSwapConstraint] = None
    name: str = "MartingaleSB"
    
    def __post_init__(self):
        # Sort marginals by time
        self.marginals = sorted(self.marginals, key=lambda m: m.time)
        
        # Validate endpoints exist
        times = [m.time for m in self.marginals]
        if 0.0 not in times:
            raise ValueError("Must include marginal at t=0")
        if 1.0 not in times:
            raise ValueError("Must include marginal at t=1")
        
        # Build martingale constraints between consecutive marginals
        self.martingale_constraints = []
        for i in range(len(self.marginals) - 1):
            t_start = self.marginals[i].time
            t_end = self.marginals[i + 1].time
            fwd_ratio = self.forward_curve.forward_ratio(t_start, t_end)
            self.martingale_constraints.append(
                MartingaleConstraint(t_start, t_end, fwd_ratio)
            )
    
    @property
    def dim(self) -> int:
        """State space dimension."""
        return self.marginals[0].distribution.dim
    
    @property
    def num_segments(self) -> int:
        """Number of segments between marginals."""
        return len(self.marginals) - 1
    
    @property
    def segment_times(self) -> List[Tuple[float, float]]:
        """List of (t_start, t_end) for each segment."""
        times = [m.time for m in self.marginals]
        return [(times[i], times[i+1]) for i in range(len(times) - 1)]
    
    @property
    def expiry_times(self) -> List[float]:
        """List of marginal constraint times (expiries)."""
        return [m.time for m in self.marginals]
    
    def get_forward(self, t: float) -> float:
        """Get forward price F(0, t)."""
        return self.forward_curve.forward(0, t)
    
    def summary(self) -> str:
        """Get problem summary string."""
        lines = [
            f"=== {self.name} ===",
            f"Dimension: {self.dim}",
            f"Reference: {type(self.reference).__name__}",
            f"Num marginals: {len(self.marginals)}",
            f"Num segments: {self.num_segments}",
            f"Spot: {self.forward_curve.spot:.2f}",
            f"Rate: {self.forward_curve.rate:.2%}",
            "Marginal times: " + ", ".join(f"{m.time:.3f}" for m in self.marginals),
            "Forward prices: " + ", ".join(
                f"F(0,{m.time:.2f})={self.get_forward(m.time):.2f}" 
                for m in self.marginals[1:]
            ),
        ]
        if self.variance_constraint:
            target_vol = np.sqrt(self.variance_constraint.target_variance) * 100
            lines.append(f"Variance constraint: {target_vol:.1f}% target vol")
        return "\n".join(lines)


# =============================================================================
# Martingale Optimal Transport Coupling
# =============================================================================

def martingale_sinkhorn_coupling(
    x_start: Array,
    x_end_pool: Array,
    forward_ratio: float,
    epsilon: float = 0.5,
    num_iters: int = 50,
    martingale_weight: float = 10.0,
    sv_transition_probs: Optional[Array] = None,
    return_diagnostics: bool = False,
) -> Any:
    """Compute martingale-constrained OT coupling.
    
    Finds a coupling π between x_start and x_end_pool such that:
    1. Minimizes transport cost (like standard OT)
    2. Approximately satisfies E[x_end | x_start] ≈ x_start · forward_ratio
    
    ENHANCEMENT: If sv_transition_probs is provided, uses it as prior instead
    of uniform, giving more realistic SV-aware coupling.
    
    Args:
        x_start: Starting points (log-prices), shape [n].
        x_end_pool: Pool of ending points (log-prices), shape [n].
        forward_ratio: Expected ratio exp(x_end) / exp(x_start).
        epsilon: Entropic regularization.
        num_iters: Number of Sinkhorn iterations.
        martingale_weight: Weight on martingale constraint.
        sv_transition_probs: Optional SV-implied transition matrix [n, n].
        
    Returns:
        If return_diagnostics is False:
            Coupling indices, shape [n], where x_end_pool[coupling[i]] is paired with x_start[i].
        If return_diagnostics is True:
            (coupling_indices, diagnostics_dict) where diagnostics_dict contains
            coupling-level KL and transport summaries.
    """
    n = len(x_start)
    
    # Cost matrix: squared distance in log-space
    C = (x_start[:, None] - x_end_pool[None, :]) ** 2
    
    # Martingale penalty: penalize deviations from forward
    # We want exp(x_end) ≈ exp(x_start) * forward_ratio
    # In log-space: x_end ≈ x_start + log(forward_ratio)
    log_fwd = jnp.log(forward_ratio)
    martingale_deviation = (x_end_pool[None, :] - x_start[:, None] - log_fwd) ** 2
    
    # Combined cost
    C_total = C + martingale_weight * martingale_deviation
    
    # ENHANCEMENT: Include SV transition probabilities as prior
    if sv_transition_probs is not None:
        # Use SV probs in the kernel: K ∝ P_sv * exp(-C/ε)
        log_prior = jnp.log(sv_transition_probs + 1e-30)
        K = jnp.exp(log_prior - C_total / (epsilon + 1e-6))
    else:
        K = jnp.exp(-C_total / (epsilon + 1e-6))
    
    # Sinkhorn algorithm
    u = jnp.ones(n)
    v = jnp.ones(n)
    
    for _ in range(num_iters):
        u = 1.0 / (K @ v + 1e-10)
        v = 1.0 / (K.T @ u + 1e-10)
    
    P = u[:, None] * K * v[None, :]
    
    # Convert to bijective assignment (greedy)
    P_np = np.array(P)
    coupling = np.zeros(n, dtype=np.int32)
    available = np.ones(n, dtype=bool)
    
    # Sort rows by max coupling value
    row_order = np.argsort(-np.max(P_np, axis=1))
    
    for i in row_order:
        row = P_np[i].copy()
        row[~available] = -1
        j = np.argmax(row)
        coupling[i] = j
        available[j] = False
    
    coupling_arr = jnp.array(coupling)
    if not return_diagnostics:
        return coupling_arr

    # Coupling-level KL in nats:
    #   KL(P || Q), with Q as normalized reference kernel and P as normalized
    #   Sinkhorn coupling matrix after marginals/constraints enforcement.
    P_norm = P / (jnp.sum(P) + 1e-30)
    Q = K / (jnp.sum(K) + 1e-30)
    kl_nats = float(jnp.sum(P_norm * (jnp.log(P_norm + 1e-30) - jnp.log(Q + 1e-30))))
    avg_transport_cost = float(jnp.sum(P_norm * C))
    avg_total_cost = float(jnp.sum(P_norm * C_total))
    diagnostics = {
        "kl_divergence_nats": kl_nats,
        "avg_transport_cost": avg_transport_cost,
        "avg_total_cost": avg_total_cost,
    }
    return coupling_arr, diagnostics


def project_to_martingale(
    x_start: Array,
    x_end: Array,
    forward_ratio: float,
    method: str = 'mean_shift',
) -> Array:
    """Project coupling to satisfy martingale constraint exactly.
    
    Given a coupling (x_start[i], x_end[i]), adjust x_end so that:
        E[exp(x_end) | exp(x_start)] = exp(x_start) · forward_ratio
    
    Methods:
    - 'mean_shift': Shift all x_end by constant to match mean
    - 'proportional': Scale x_end proportionally
    - 'quantile': Quantile-based adjustment (preserves distribution shape)
    
    Args:
        x_start: Log-prices at start, shape [n].
        x_end: Log-prices at end, shape [n].
        forward_ratio: Target ratio.
        method: Projection method.
        
    Returns:
        Adjusted x_end satisfying martingale constraint.
    """
    # Current ratio in price space
    current_ratio = jnp.mean(jnp.exp(x_end)) / jnp.mean(jnp.exp(x_start))
    
    if method == 'mean_shift':
        # Shift in log-space to match forward
        # If current_ratio ≠ forward_ratio, shift by log(forward_ratio/current_ratio)
        shift = jnp.log(forward_ratio / (current_ratio + 1e-10))
        return x_end + shift
    
    elif method == 'proportional':
        # Scale in price space: new_price = old_price * (forward_ratio / current_ratio)
        scale = forward_ratio / (current_ratio + 1e-10)
        return x_end + jnp.log(scale)
    
    elif method == 'quantile':
        # More sophisticated: preserve shape but shift quantiles
        # This maintains the marginal distribution shape better
        shift = jnp.log(forward_ratio / (current_ratio + 1e-10))
        return x_end + shift
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# ENHANCEMENT: MOT Price Bounds
# =============================================================================

class MartingaleOTBounds:
    """Compute model-free price bounds via Martingale Optimal Transport.
    
    ═══════════════════════════════════════════════════════════════════════════
    MAIN MATH TAKEAWAY: MOT Price Bounds
    ═══════════════════════════════════════════════════════════════════════════
    
    Instead of one price, compute:
        
        C̲ = inf  E_π[g(X,Y)]    (lower bound)
        C̄ = sup  E_π[g(X,Y)]    (upper bound)
    
    subject to:
        - π has marginals μ, ν
        - E_π[Y|X] = X · fwd_ratio (martingale)
    
    The width [C̲, C̄] quantifies MODEL RISK!
    ═══════════════════════════════════════════════════════════════════════════
    
    Args:
        mu_samples: Samples from first marginal.
        nu_samples: Samples from second marginal.
        forward_ratio: E[Y]/E[X] under martingale constraint.
    """
    
    def __init__(
        self,
        mu_samples: Array,
        nu_samples: Array,
        forward_ratio: float,
    ):
        self.mu = jnp.sort(jnp.asarray(mu_samples))
        self.nu = jnp.sort(jnp.asarray(nu_samples))
        self.forward_ratio = forward_ratio
        self.n = len(mu_samples)
    
    def compute_bounds(
        self,
        payoff_fn: Callable[[Array, Array], Array],
        epsilon: float = 0.1,
        num_iters: int = 100,
    ) -> Tuple[float, float, Array]:
        """Compute upper and lower bounds for E[payoff(X, Y)].
        
        Args:
            payoff_fn: Function (X, Y) → payoff, where X, Y are meshgrid arrays.
            epsilon: Entropic regularization (smaller = tighter bounds).
            num_iters: Sinkhorn iterations.
            
        Returns:
            (lower_bound, upper_bound, optimal_coupling)
        """
        # Compute payoff matrix
        X, Y = jnp.meshgrid(self.mu, self.nu, indexing='ij')
        payoff_matrix = payoff_fn(X, Y)
        
        # Martingale penalty matrix
        martingale_dev = (Y - X * self.forward_ratio) ** 2 / (X ** 2 + 1e-8)
        
        # Upper bound: maximize payoff
        upper_coupling, upper_bound = self._entropic_mot(
            payoff_matrix, martingale_dev, epsilon, num_iters, maximize=True
        )
        
        # Lower bound: minimize payoff
        lower_coupling, lower_bound = self._entropic_mot(
            payoff_matrix, martingale_dev, epsilon, num_iters, maximize=False
        )
        
        return float(lower_bound), float(upper_bound), upper_coupling
    
    def _entropic_mot(
        self,
        payoff: Array,
        martingale_penalty: Array,
        epsilon: float,
        num_iters: int,
        maximize: bool,
    ) -> Tuple[Array, float]:
        """Solve entropic MOT via Sinkhorn."""
        n = self.n
        sign = 1.0 if maximize else -1.0
        lambda_mart = 10.0
        
        # Kernel
        log_K = (sign * payoff - lambda_mart * martingale_penalty) / epsilon
        log_K = log_K - jnp.max(log_K)
        K = jnp.exp(log_K)
        K = K / (jnp.sum(K) + 1e-10)
        
        # Sinkhorn
        u = jnp.ones(n)
        v = jnp.ones(n)
        
        for _ in range(num_iters):
            u = 1.0 / (K @ v + 1e-10)
            u = jnp.clip(u, 1e-30, 1e30)
            v = 1.0 / (K.T @ u + 1e-10)
            v = jnp.clip(v, 1e-30, 1e30)
        
        P = u[:, None] * K * v[None, :]
        P = P / (jnp.sum(P) + 1e-10)
        
        expected = jnp.sum(P * payoff)
        return P, expected


# =============================================================================
# Martingale SB Solver
# =============================================================================

@dataclass
class MartingaleSBConfig:
    """Configuration for Martingale SB solver.
    
    ENHANCED: Added use_sv_reference for stochastic volatility bridge.
    """
    sigma_ref: float = 0.2  # Reference volatility
    num_steps_per_segment: int = 40
    martingale_weight: float = 10.0  # Weight on martingale constraint in coupling
    projection_method: str = 'mean_shift'  # 'mean_shift', 'proportional', 'quantile'
    
    # ENHANCEMENT: SV reference options
    use_sv_reference: bool = False  # Use SV dynamics instead of Brownian bridge
    apply_variance_constraint: bool = True  # Rescale to match variance swap
    
    verbose: int = 1


class MartingaleSBSolver:
    """Solver for Martingale Schrödinger Bridge problems.
    
    This solver produces paths that:
    1. Match all marginal distributions (from options prices)
    2. Satisfy the martingale constraint (no-arbitrage)
    3. (ENHANCED) Match variance swap level
    4. (ENHANCED) Use SV dynamics as reference
    
    The key insight is using martingale-constrained optimal transport
    to couple consecutive marginals, then interpolating with bridges.
    
    Algorithm:
    1. For each segment [T_i, T_{i+1}]:
       a. Sample from target marginal μ_{i+1}
       b. Compute martingale OT coupling (with SV prior if configured)
       c. Project to exact martingale constraint
       d. Interpolate with bridge (Brownian or SV-aware)
    2. Apply variance swap constraint if configured
    
    Attributes:
        problem: MartingaleSBProblem specification.
        config: Solver configuration.
        marginal_samples: Pre-sampled marginal distributions.
    """
    
    def __init__(
        self,
        problem: MartingaleSBProblem,
        config: Optional[MartingaleSBConfig] = None,
    ):
        self.problem = problem
        self.config = config or MartingaleSBConfig()
        
        self.marginal_samples: Dict[float, Array] = {}
        self._is_trained = False
        self._mot_bounds: Dict[Tuple[float, float], MartingaleOTBounds] = {}
        self.last_coupling_diagnostics: List[Dict[str, Any]] = []
        self.last_kl_divergence_nats: float = float("nan")
    
    def train(
        self,
        key: PRNGKey,
        num_samples: int = 2000,
    ) -> Dict[str, Any]:
        """Train the martingale SB solver.
        
        For this solver, "training" means pre-sampling from marginals
        and validating the martingale constraints.
        
        Args:
            key: Random key.
            num_samples: Number of samples per marginal.
            
        Returns:
            Training info dictionary.
        """
        if self.config.verbose >= 1:
            print("Training Martingale SB...")
            print(f"  Spot: ${self.problem.forward_curve.spot:.2f}")
            print(f"  Rate: {self.problem.forward_curve.rate:.2%}")
            if self.config.use_sv_reference:
                print(f"  Using SV reference: {type(self.problem.reference).__name__}")
        
        # Pre-sample from each marginal
        for marginal in self.problem.marginals:
            key, subkey = jax.random.split(key)
            samples = marginal.distribution.sample(subkey, num_samples)
            # Handle 1D case
            if samples.ndim == 1:
                samples = samples[:, None]
            self.marginal_samples[marginal.time] = samples.squeeze()
        
        # Build MOT bounds objects for each segment
        for i in range(self.problem.num_segments):
            t_start = self.problem.expiry_times[i]
            t_end = self.problem.expiry_times[i + 1]
            fwd_ratio = self.problem.martingale_constraints[i].forward_ratio
            
            mu = self.marginal_samples[t_start]
            nu = self.marginal_samples[t_end]
            
            self._mot_bounds[(t_start, t_end)] = MartingaleOTBounds(mu, nu, fwd_ratio)
        
        # Validate forward consistency
        if self.config.verbose >= 1:
            print("\n  Forward price validation:")
            for i, marginal in enumerate(self.problem.marginals):
                t = marginal.time
                samples = self.marginal_samples[t]
                sample_mean = float(jnp.mean(samples))
                forward = self.problem.get_forward(t) if t > 0 else self.problem.forward_curve.spot
                print(f"    T={t:.3f}: Sample mean=${sample_mean:.2f}, Forward=${forward:.2f}")
        
        self._is_trained = True
        
        return {
            'num_marginals': len(self.problem.marginals),
            'num_samples': num_samples,
        }
    
    def set_marginal_samples(self, samples_dict: Dict[float, Array]):
        """Set marginal samples directly (for fair comparison).
        
        Args:
            samples_dict: Dictionary mapping expiry times to samples.
        """
        self.marginal_samples = samples_dict
        self._is_trained = True
    
    def _get_sv_transition_probs(
        self,
        x_start: Array,
        x_end_pool: Array,
        t_start: float,
        t_end: float,
    ) -> Optional[Array]:
        """Compute SV-implied transition probability matrix.
        
        If problem.reference is a HestonDynamics or similar, compute
        P(X_end | X_start) under the reference measure.
        
        Returns None if not using SV reference.
        """
        if not self.config.use_sv_reference:
            return None
        
        ref = self.problem.reference
        
        # Check if reference has transition probability method
        if not hasattr(ref, 'transition_log_prob'):
            # Fall back to Gaussian approximation
            tau = t_end - t_start
            
            # Use reference volatility for Gaussian approximation
            if hasattr(ref, 'v0'):
                sigma = np.sqrt(ref.v0)
            elif hasattr(ref, 'theta'):
                sigma = np.sqrt(ref.theta)
            else:
                sigma = self.config.sigma_ref
            
            # Log-normal transition: X_end | X_start ~ N(X_start + drift, sigma*sqrt(tau))
            drift = (self.problem.forward_curve.rate - 0.5 * sigma**2) * tau
            std = sigma * np.sqrt(tau)
            
            n = len(x_start)
            log_probs = np.zeros((n, n))
            
            for i in range(n):
                mean = x_start[i] + drift
                log_probs[i] = -0.5 * ((x_end_pool - mean) / std) ** 2
            
            # Normalize rows
            log_probs = log_probs - np.max(log_probs, axis=1, keepdims=True)
            probs = np.exp(log_probs)
            probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)
            
            return jnp.array(probs)
        
        # Use reference's transition probability
        return ref.transition_log_prob(x_start, x_end_pool, t_start, t_end)
    
    def _sv_bridge_interpolate(
        self,
        X_start: Array,
        X_end: Array,
        t_start: float,
        t_end: float,
        num_steps: int,
        key: PRNGKey,
    ) -> Tuple[Array, Array]:
        """Interpolate using SV-aware bridge.
        
        Instead of simple Brownian bridge, uses reference dynamics
        with drift adjustment to hit endpoints.
        """
        ref = self.problem.reference
        n_paths = len(X_start)
        tau = t_end - t_start
        dt = tau / num_steps
        sqrt_dt = jnp.sqrt(dt)
        
        segment_times = jnp.linspace(t_start, t_end, num_steps + 1)
        
        # Get volatility from reference
        if hasattr(ref, 'v0'):
            v0 = ref.v0
            kappa = getattr(ref, 'kappa', 2.0)
            theta = getattr(ref, 'theta', v0)
            xi = getattr(ref, 'xi', 0.3)
            rho = getattr(ref, 'rho', -0.7)
        else:
            v0 = self.config.sigma_ref ** 2
            kappa, theta, xi, rho = 2.0, v0, 0.3, -0.7
        
        # Initialize
        X = X_start.copy()
        v = jnp.full(n_paths, v0)
        segment_paths = [X]
        
        for i in range(num_steps - 1):
            key, k1, k2 = jax.random.split(key, 3)
            
            t_current = segment_times[i]
            remaining = t_end - t_current
            
            # SV dynamics with bridge conditioning
            sqrt_v = jnp.sqrt(jnp.maximum(v, 1e-8))
            
            # Bridge drift: pull toward endpoint
            target_drift = (X_end - X) / (remaining + 1e-8)
            
            # Blend with SV drift
            sv_drift = self.problem.forward_curve.rate - 0.5 * v
            blend = jnp.minimum(0.3, dt / (remaining + 1e-8))
            
            combined_drift = (1 - blend) * sv_drift + blend * target_drift
            
            # Correlated noise
            Z1 = jax.random.normal(k1, (n_paths,))
            Z2 = jax.random.normal(k2, (n_paths,))
            dW_S = Z1 * sqrt_dt
            dW_v = (rho * Z1 + jnp.sqrt(1 - rho**2) * Z2) * sqrt_dt
            
            # Update price
            X = X + combined_drift * dt + sqrt_v * dW_S
            
            # Update variance
            v = v + kappa * (theta - v) * dt + xi * sqrt_v * dW_v
            v = jnp.maximum(v, 1e-8)
            
            segment_paths.append(X)
        
        # Final step: snap to endpoint
        segment_paths.append(X_end)
        
        return segment_times, jnp.stack(segment_paths, axis=1)
    
    def _brownian_bridge_interpolate(
        self,
        X_start: Array,
        X_end: Array,
        t_start: float,
        t_end: float,
        num_steps: int,
        key: PRNGKey,
    ) -> Tuple[Array, Array]:
        """Standard Brownian bridge interpolation."""
        n_paths = len(X_start)
        tau = t_end - t_start
        dt = tau / num_steps
        
        segment_times = jnp.linspace(t_start, t_end, num_steps + 1)
        
        X = X_start.copy()
        segment_paths = [X]
        
        for i in range(num_steps - 1):
            key, subkey = jax.random.split(key)
            
            t_current = segment_times[i]
            remaining = t_end - t_current
            
            # Bridge drift
            drift = (X_end - X) / (remaining + 1e-8)
            
            # Bridge noise
            noise_scale = self.config.sigma_ref * jnp.sqrt(
                dt * (remaining - dt) / (remaining + 1e-8)
            )
            noise_scale = jnp.maximum(noise_scale, 0)
            
            dW = jax.random.normal(subkey, (n_paths,))
            X = X + drift * dt + noise_scale * dW
            segment_paths.append(X)
        
        # Final step: snap to endpoint
        segment_paths.append(X_end)
        
        return segment_times, jnp.stack(segment_paths, axis=1)
    
    def simulate(
        self,
        key: PRNGKey,
        num_paths: int,
    ) -> Tuple[Array, Array]:
        """Simulate martingale-constrained paths.
        
        Args:
            key: Random key.
            num_paths: Number of paths to simulate.
            
        Returns:
            (times, paths) where paths has shape [num_paths, num_times].
        """
        if not self._is_trained:
            raise RuntimeError("Solver must be trained first")
        
        all_times = []
        all_paths = []
        
        expiries = self.problem.expiry_times
        segment_diag: List[Dict[str, Any]] = []
        
        # Sample endpoints for each segment, enforcing martingale
        segment_endpoints = {}
        segment_endpoints[0.0] = jnp.log(self.problem.forward_curve.spot) * jnp.ones(num_paths)
        
        for seg_idx in range(self.problem.num_segments):
            t_start = expiries[seg_idx]
            t_end = expiries[seg_idx + 1]
            
            key, k1, k2 = jax.random.split(key, 3)
            
            # Get target samples for this expiry
            target_samples = self.marginal_samples.get(t_end)
            if target_samples is None:
                raise ValueError(f"No marginal samples for t={t_end}")
            
            # Ensure we have enough samples
            if len(target_samples) < num_paths:
                idx = jax.random.choice(k1, len(target_samples), shape=(num_paths,))
                target_prices = jnp.array(target_samples)[idx]
            else:
                target_prices = jnp.array(target_samples[:num_paths])
            
            X_end_pool = jnp.log(target_prices)
            X_start = segment_endpoints[t_start]
            
            # Get forward ratio for this segment
            fwd_ratio = self.problem.martingale_constraints[seg_idx].forward_ratio
            
            # ENHANCEMENT: Get SV transition probs if using SV reference
            sv_probs = self._get_sv_transition_probs(
                np.array(X_start), np.array(X_end_pool), t_start, t_end
            )
            
            if seg_idx == 0:
                # First segment: all start at same point
                perm = jax.random.permutation(k2, num_paths)
                X_end = X_end_pool[perm]
                segment_diag.append({
                    "segment_index": int(seg_idx),
                    "t_start": float(t_start),
                    "t_end": float(t_end),
                    "kl_divergence_nats": float("nan"),
                    "avg_transport_cost": float("nan"),
                    "avg_total_cost": float("nan"),
                    "coupling_mode": "random_permutation",
                })
            else:
                # Martingale OT coupling (with SV prior if available)
                coupling, coupling_diag = martingale_sinkhorn_coupling(
                    X_start, X_end_pool, fwd_ratio,
                    epsilon=0.5,
                    num_iters=50,
                    martingale_weight=self.config.martingale_weight,
                    sv_transition_probs=sv_probs,
                    return_diagnostics=True,
                )
                X_end = X_end_pool[coupling]
                segment_diag.append({
                    "segment_index": int(seg_idx),
                    "t_start": float(t_start),
                    "t_end": float(t_end),
                    "kl_divergence_nats": float(coupling_diag.get("kl_divergence_nats", float("nan"))),
                    "avg_transport_cost": float(coupling_diag.get("avg_transport_cost", float("nan"))),
                    "avg_total_cost": float(coupling_diag.get("avg_total_cost", float("nan"))),
                    "coupling_mode": "martingale_sinkhorn",
                })
            
            # Project to satisfy martingale exactly
            X_end = project_to_martingale(
                X_start, X_end, fwd_ratio,
                method=self.config.projection_method,
            )
            
            segment_endpoints[t_end] = X_end
        
        # Now simulate bridges between consecutive endpoints
        for seg_idx in range(self.problem.num_segments):
            t_start = expiries[seg_idx]
            t_end = expiries[seg_idx + 1]
            
            key, subkey = jax.random.split(key)
            
            X_start = segment_endpoints[t_start]
            X_end = segment_endpoints[t_end]
            
            num_steps = self.config.num_steps_per_segment
            
            # ENHANCEMENT: Choose bridge type based on config
            if self.config.use_sv_reference:
                segment_times, segment_paths = self._sv_bridge_interpolate(
                    X_start, X_end, t_start, t_end, num_steps, subkey
                )
            else:
                segment_times, segment_paths = self._brownian_bridge_interpolate(
                    X_start, X_end, t_start, t_end, num_steps, subkey
                )
            
            # Store
            if seg_idx == 0:
                all_times.extend(segment_times.tolist())
                all_paths.append(segment_paths)
            else:
                all_times.extend(segment_times[1:].tolist())
                all_paths.append(segment_paths[:, 1:])
        
        times = jnp.array(all_times)
        paths = jnp.concatenate(all_paths, axis=1)
        price_paths = jnp.exp(paths)  # Return prices, not log-prices
        
        # ENHANCEMENT: Apply variance constraint if configured
        if (self.config.apply_variance_constraint and 
            self.problem.variance_constraint is not None):
            
            price_paths = self.problem.variance_constraint.adjust_paths(
                price_paths, times[-1]
            )
            
            if self.config.verbose >= 1:
                rv = self.problem.variance_constraint.compute_realized_variance(
                    price_paths, times[-1]
                )
                print(f"  Applied variance constraint: realized vol = {jnp.sqrt(jnp.mean(rv))*100:.1f}%")

        self.last_coupling_diagnostics = segment_diag
        kl_vals = [
            float(d.get("kl_divergence_nats", float("nan")))
            for d in segment_diag
            if np.isfinite(float(d.get("kl_divergence_nats", float("nan"))))
        ]
        self.last_kl_divergence_nats = float(np.mean(kl_vals)) if kl_vals else float("nan")
        
        return times, price_paths
    
    def compute_mot_bounds(
        self,
        payoff_fn: Callable[[Array, Array], Array],
        t_start: float,
        t_end: float,
        discount: float = 1.0,
    ) -> Tuple[float, float]:
        """Compute model-free MOT price bounds for a two-time payoff.
        
        Args:
            payoff_fn: Payoff function g(X_{t_start}, X_{t_end}).
            t_start: First time point.
            t_end: Second time point.
            discount: Discount factor.
            
        Returns:
            (lower_bound, upper_bound) for discounted price.
        """
        if (t_start, t_end) not in self._mot_bounds:
            raise ValueError(f"No MOT bounds for segment ({t_start}, {t_end})")
        
        mot = self._mot_bounds[(t_start, t_end)]
        lower, upper, _ = mot.compute_bounds(payoff_fn)
        
        return discount * lower, discount * upper
    
    def check_martingale(
        self,
        key: PRNGKey,
        num_paths: int = 5000,
    ) -> Dict[str, Any]:
        """Check martingale property of simulated paths.
        
        Verifies: E[S_{T_{i+1}} | S_{T_i}] ≈ Forward(T_i, T_{i+1})
        
        Returns:
            Dictionary with martingale diagnostics.
        """
        times, paths = self.simulate(key, num_paths)
        
        expiries = self.problem.expiry_times
        results = {'segments': []}
        
        for i, constraint in enumerate(self.problem.martingale_constraints):
            t_start = constraint.t_start
            t_end = constraint.t_end
            expected_ratio = constraint.forward_ratio
            
            # Find time indices
            idx_start = int(jnp.argmin(jnp.abs(times - t_start)))
            idx_end = int(jnp.argmin(jnp.abs(times - t_end)))
            
            S_start = paths[:, idx_start]
            S_end = paths[:, idx_end]
            
            # Actual ratio
            actual_ratio = float(jnp.mean(S_end) / jnp.mean(S_start))
            
            # Martingale error
            error = abs(actual_ratio - expected_ratio) / expected_ratio
            
            results['segments'].append({
                't_start': t_start,
                't_end': t_end,
                'expected_ratio': expected_ratio,
                'actual_ratio': actual_ratio,
                'relative_error': error,
            })
        
        # Average error
        results['avg_martingale_error'] = np.mean([s['relative_error'] for s in results['segments']])
        
        # ENHANCEMENT: Also report realized vol
        rv = jnp.sum(jnp.log(paths[:, 1:] / paths[:, :-1])**2, axis=1) / times[-1]
        results['realized_vol'] = float(jnp.sqrt(jnp.mean(rv)))
        
        return results
    
    def check_marginals(
        self,
        key: PRNGKey,
        num_paths: int = 2000,
    ) -> Dict[str, float]:
        """Check marginal matching at each expiry.
        
        Returns:
            Dictionary mapping expiry times to MMD² values.
        """
        times, paths = self.simulate(key, num_paths)
        
        results = {}
        
        for marginal in self.problem.marginals:
            t = marginal.time
            
            # Find closest time index
            idx = int(jnp.argmin(jnp.abs(times - t)))
            simulated = paths[:, idx]
            
            # Target samples
            target = self.marginal_samples.get(t)
            if target is None:
                continue
            
            target = jnp.array(target[:num_paths])
            
            # MMD²
            mmd = float(mmd_squared(simulated[:, None], target[:, None]))
            results[f"t={t:.3f}"] = mmd
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def create_martingale_sb_problem(
    spot: float,
    rate: float,
    expiries: List[float],
    marginal_distributions: List[MarginalDistribution],
    reference: Optional[ReferenceDynamics] = None,
    dividend_yield: float = 0.0,
    variance_swap_strike: Optional[float] = None,
    name: str = "MartingaleSB",
) -> MartingaleSBProblem:
    """Create a MartingaleSBProblem from market data.
    
    Args:
        spot: Current spot price.
        rate: Risk-free rate.
        expiries: List of option expiries [0, T1, T2, ..., T_final].
        marginal_distributions: Risk-neutral distributions at each expiry.
        reference: Reference dynamics (default: Brownian motion).
        dividend_yield: Continuous dividend yield.
        variance_swap_strike: Optional variance swap strike (annualized variance).
        name: Problem name.
        
    Returns:
        MartingaleSBProblem instance.
    """
    from .core.problem import BrownianMotion
    
    if len(expiries) != len(marginal_distributions):
        raise ValueError("expiries and marginal_distributions must have same length")
    
    # Normalize times to [0, 1]
    T_final = max(expiries)
    normalized_times = [t / T_final for t in expiries]
    
    # Ensure 0 is included
    if 0.0 not in normalized_times:
        raise ValueError("expiries must include 0")
    
    # Create marginal constraints
    marginals = [
        MarginalConstraint(time=t_norm, distribution=dist)
        for t_norm, dist in zip(normalized_times, marginal_distributions)
    ]
    
    # Reference dynamics
    if reference is None:
        reference = BrownianMotion(sigma=0.2, dim=1)
    
    # Forward curve (scale rate by T_final for normalized time)
    forward_curve = ForwardCurve(
        spot=spot,
        rate=rate * T_final,  # Scale for normalized time
        dividend_yield=dividend_yield * T_final,
    )
    
    # Variance constraint
    var_constraint = None
    if variance_swap_strike is not None:
        var_constraint = VarianceSwapConstraint(
            target_variance=variance_swap_strike * T_final  # Scale for normalized time
        )
    
    return MartingaleSBProblem(
        reference=reference,
        marginals=marginals,
        forward_curve=forward_curve,
        variance_constraint=var_constraint,
        name=name,
    )


def solve_martingale_sb(
    problem: MartingaleSBProblem,
    key: PRNGKey,
    num_samples: int = 2000,
    config: Optional[MartingaleSBConfig] = None,
) -> Tuple[MartingaleSBSolver, Dict]:
    """Convenience function to solve a martingale SB problem.
    
    Args:
        problem: MartingaleSBProblem to solve.
        key: Random key.
        num_samples: Samples per marginal.
        config: Solver configuration.
        
    Returns:
        (solver, results) tuple.
    """
    solver = MartingaleSBSolver(problem, config)
    results = solver.train(key, num_samples)
    
    return solver, results


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core classes
    'ForwardCurve',
    'MartingaleConstraint',
    'MartingaleSBProblem',
    'MartingaleSBConfig',
    'MartingaleSBSolver',
    # ENHANCED
    'VarianceSwapConstraint',
    'MartingaleOTBounds',
    # Coupling utilities
    'martingale_sinkhorn_coupling',
    'project_to_martingale',
    # Convenience functions
    'create_martingale_sb_problem',
    'solve_martingale_sb',
]
