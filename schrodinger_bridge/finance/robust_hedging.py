"""Robust Hedging via Entropic Martingale Optimal Transport.

This module provides model-free option pricing bounds and hedge extraction
using the Entropic MOT framework.

MATHEMATICAL FOUNDATION
-
The primal problem (robust pricing bounds):
    sup/inf_{pi  in Pi_M(mu,nu)} E_pi[g(X,Y)]
    
where Pi_M(mu,nu) is the set of **martingale couplings** (E[Y|X] = X).

With entropic regularization:
    sup_pi { E_pi[g] - eps*KL(pi || R) }

MAIN MATH TAKEAWAY
-
The dual problem gives hedging portfolios:

    inf_{phi,psi,h} { integralphi dmu + integralpsi dnu + eps*(regularization term) }

where:
    - phi(x): Static options at T_1
    - psi(y): Static options at T₂
    - h(x): Delta hedge

The Sinkhorn scalings (u, v) give us the duals directly:
    phi(x) = eps * log(u(x))
    psi(y) = eps * log(v(y))

REGULARIZATION TRADE-OFF
-
| eps         | Bounds   | Stability | Interpretation          |
|-----------|----------|-----------|-------------------------|
| eps -> 0     | Tight    | Fragile   | Classical MOT (extremal)|
| eps -> ∞     | Wide     | Robust    | Reference model         |
| moderate  | Balanced | Good      | "Robust but not paranoid"|

References:
    Beiglböck et al. "Model-independent bounds for option prices" (2013)
    Nutz & Stebegg "Canonical Martingale Optimal Transport" (2018)

Author: Schrödinger Bridge Library
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple, Union

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

Array = jnp.ndarray
Scalar = Union[float, Array]
PRNGKey = jax.Array


# DUAL POTENTIALS (HEDGE POSITIONS)
# -----------------------------------------------------------------------------
@dataclass
class DualPotentials:
    """Container for dual potentials from entropic MOT.
    
    These represent the optimal static hedge positions:
    - phi(x): Option positions at time T_1
    - psi(y): Option positions at time T₂
    - h(x): Delta hedge (shares of underlying at T_1)
    
    The super-replication inequality (approximately) holds:
        phi(X) + psi(Y) + h(X)*(Y - X) >= g(X, Y) - O(eps)
    """
    phi_values: Array
    psi_values: Array
    delta_values: Array
    x_grid: Array
    y_grid: Array
    epsilon: float
    
    def phi(self, x: Array) -> Array:
        """Interpolate phi at arbitrary points."""
        return jnp.interp(x, self.x_grid, self.phi_values)
    
    def psi(self, y: Array) -> Array:
        """Interpolate psi at arbitrary points."""
        return jnp.interp(y, self.y_grid, self.psi_values)
    
    def delta(self, x: Array) -> Array:
        """Interpolate delta hedge at arbitrary points."""
        return jnp.interp(x, self.x_grid, self.delta_values)
    
    def hedge_pnl(self, x: Array, y: Array) -> Array:
        """Compute hedge P&L: phi(x) + psi(y) + h(x)(y - x)."""
        return self.phi(x) + self.psi(y) + self.delta(x) * (y - x)
    
    def to_option_portfolio(
        self,
        strikes_t1: Array,
        strikes_t2: Array,
    ) -> Dict[str, Array]:
        """Convert dual potentials to option portfolio weights.
        
        Uses Breeden-Litzenberger: phi''(K) gives butterfly weights.
        """
        dx = self.x_grid[1] - self.x_grid[0]
        dy = self.y_grid[1] - self.y_grid[0]
        
        phi_weights = jnp.zeros_like(strikes_t1)
        for i, K in enumerate(strikes_t1):
            idx = jnp.argmin(jnp.abs(self.x_grid - K))
            if 0 < idx < len(self.x_grid) - 1:
                phi_weights = phi_weights.at[i].set(
                    (self.phi_values[idx+1] - 2*self.phi_values[idx] + self.phi_values[idx-1]) / dx**2
                )
        
        psi_weights = jnp.zeros_like(strikes_t2)
        for i, K in enumerate(strikes_t2):
            idx = jnp.argmin(jnp.abs(self.y_grid - K))
            if 0 < idx < len(self.y_grid) - 1:
                psi_weights = psi_weights.at[i].set(
                    (self.psi_values[idx+1] - 2*self.psi_values[idx] + self.psi_values[idx-1]) / dy**2
                )
        
        return {
            'strikes_t1': strikes_t1,
            'weights_t1': phi_weights,
            'strikes_t2': strikes_t2,
            'weights_t2': psi_weights,
            'delta_hedge': jnp.mean(self.delta_values),
        }


# RESULT CONTAINER
# -----------------------------------------------------------------------------
@dataclass
class RobustHedgingResult:
    """Complete result from robust hedging computation."""
    upper_bound: float
    lower_bound: float
    upper_dual: DualPotentials
    lower_dual: DualPotentials
    primal_value: float
    coupling: Array
    epsilon: float
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    
    def price_interval(self) -> Tuple[float, float]:
        """Return (lower, upper) price bounds."""
        return (self.lower_bound, self.upper_bound)
    
    def mid_price(self) -> float:
        """Return midpoint of price interval."""
        return (self.lower_bound + self.upper_bound) / 2
    
    def uncertainty(self) -> float:
        """Return width of price interval (model uncertainty)."""
        return self.upper_bound - self.lower_bound
    
    def get_super_replicating_hedge(self) -> DualPotentials:
        """Get hedge that super-replicates the payoff."""
        return self.upper_dual
    
    def get_sub_replicating_hedge(self) -> DualPotentials:
        """Get hedge that sub-replicates the payoff."""
        return self.lower_dual


# ENTROPIC MOT SOLVER
# -----------------------------------------------------------------------------
class EntropicMOTSolver:
    """Solver for Entropic Martingale Optimal Transport.
    
    NOTE: This is NOT an SBSolver subclass - it solves a different problem
    (2-marginal OT with martingale constraint vs. continuous path measure).
    
    The algorithm uses modified Sinkhorn iteration that enforces:
    1. Marginal constraints: pi has marginals mu, nu
    2. Martingale constraint: E_pi[Y|X] = X * forward_ratio
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        num_iters: int = 200,
        tol: float = 1e-6,
        martingale_weight: float = 10.0,
    ):
        """Initialize EntropicMOTSolver.
        
        Args:
            epsilon: Entropic regularization (larger = more stable, wider bounds).
            num_iters: Maximum Sinkhorn iterations.
            tol: Convergence tolerance.
            martingale_weight: Penalty weight for martingale violation.
        """
        self.epsilon = epsilon
        self.num_iters = num_iters
        self.tol = tol
        self.martingale_weight = martingale_weight
    
    def solve(
        self,
        x_samples: Array,
        y_samples: Array,
        payoff_fn: Callable[[Array, Array], Array],
        forward_ratio: float = 1.0,
        compute_lower: bool = True,
    ) -> RobustHedgingResult:
        """Solve entropic MOT and extract dual potentials.
        
        Args:
            x_samples: Samples from mu (T_1 marginal), shape [n].
            y_samples: Samples from nu (T₂ marginal), shape [n].
            payoff_fn: Payoff function g(x, y).
            forward_ratio: E[Y]/E[X] = exp(r*tau) for martingale.
            compute_lower: Whether to compute lower bound.
            
        Returns:
            RobustHedgingResult with bounds and hedge portfolios.
        """
        x = jnp.sort(jnp.asarray(x_samples))
        y = jnp.sort(jnp.asarray(y_samples))
        
        # Compute payoff matrix
        G = payoff_fn(x[:, None], y[None, :])
        
        # Upper bound
        upper_result = self._sinkhorn_mot(x, y, G, forward_ratio, maximize=True)
        
        # Lower bound
        if compute_lower:
            lower_result = self._sinkhorn_mot(x, y, G, forward_ratio, maximize=False)
        else:
            lower_result = upper_result
        
        # Extract dual potentials
        upper_dual = self._extract_duals(
            x, y, G, upper_result['u'], upper_result['v'], 
            forward_ratio, maximize=True
        )
        lower_dual = self._extract_duals(
            x, y, G, lower_result['u'], lower_result['v'],
            forward_ratio, maximize=False
        )
        
        return RobustHedgingResult(
            upper_bound=upper_result['value'],
            lower_bound=lower_result['value'] if compute_lower else -jnp.inf,
            upper_dual=upper_dual,
            lower_dual=lower_dual,
            primal_value=upper_result['value'],
            coupling=upper_result['coupling'],
            epsilon=self.epsilon,
            convergence_info={
                'upper_iters': upper_result['iters'],
                'lower_iters': lower_result.get('iters', 0),
            }
        )
    
    def _sinkhorn_mot(
        self,
        x: Array,
        y: Array,
        G: Array,
        forward_ratio: float,
        maximize: bool,
    ) -> Dict[str, Any]:
        """Modified Sinkhorn for martingale OT."""
        n = len(x)
        sign = -1.0 if maximize else 1.0
        
        # Martingale penalty
        martingale_penalty = ((y[None, :] - x[:, None] * forward_ratio) / (x[:, None] + 1e-8)) ** 2
        
        # Combined cost
        G_scaled = G / (jnp.abs(G).max() + 1e-8)
        C = sign * G_scaled + self.martingale_weight * martingale_penalty
        C = C - C.min()
        
        # Gibbs kernel
        log_K = -C / (self.epsilon + 1e-6)
        log_K = log_K - logsumexp(log_K)
        K = jnp.exp(log_K)
        K = jnp.clip(K, 1e-30, 1e30)
        
        # Sinkhorn iterations
        u = jnp.ones(n) / n
        v = jnp.ones(n) / n
        
        for i in range(self.num_iters):
            u_new = 1.0 / (K @ v + 1e-10)
            u_new = jnp.clip(u_new, 1e-30, 1e30)
            v_new = 1.0 / (K.T @ u_new + 1e-10)
            v_new = jnp.clip(v_new, 1e-30, 1e30)
            
            if jnp.max(jnp.abs(u_new - u)) < self.tol:
                u, v = u_new, v_new
                break
            u, v = u_new, v_new
        
        P = u[:, None] * K * v[None, :]
        P = P / (P.sum() + 1e-10)
        value = float(jnp.sum(P * G))
        
        return {'coupling': P, 'u': u, 'v': v, 'value': value, 'iters': i + 1}
    
    def _extract_duals(
        self,
        x: Array,
        y: Array,
        G: Array,
        u: Array,
        v: Array,
        forward_ratio: float,
        maximize: bool,
    ) -> DualPotentials:
        """Extract dual potentials from Sinkhorn scaling.
        
        phi(x) = eps * log(u(x))
        psi(y) = eps * log(v(y))
        """
        phi = self.epsilon * jnp.log(u + 1e-10)
        psi = self.epsilon * jnp.log(v + 1e-10)
        
        if maximize:
            phi = -phi
            psi = -psi
        
        # Delta hedge via finite differences
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        delta = jnp.zeros_like(phi)
        delta = delta.at[1:-1].set((phi[2:] - phi[:-2]) / (2 * dx))
        delta = delta.at[0].set((phi[1] - phi[0]) / dx)
        delta = delta.at[-1].set((phi[-1] - phi[-2]) / dx)
        delta = delta + jnp.log(forward_ratio)
        
        return DualPotentials(
            phi_values=phi,
            psi_values=psi,
            delta_values=delta,
            x_grid=x,
            y_grid=y,
            epsilon=self.epsilon,
        )
    
    def sensitivity_analysis(
        self,
        x_samples: Array,
        y_samples: Array,
        payoff_fn: Callable,
        forward_ratio: float,
        epsilon_range: Array,
    ) -> Dict[str, Array]:
        """Analyze how bounds change with regularization."""
        uppers, lowers = [], []
        original_eps = self.epsilon
        
        for eps in epsilon_range:
            self.epsilon = float(eps)
            result = self.solve(x_samples, y_samples, payoff_fn, forward_ratio)
            uppers.append(result.upper_bound)
            lowers.append(result.lower_bound)
        
        self.epsilon = original_eps
        
        return {
            'epsilons': jnp.asarray(epsilon_range),
            'upper_bounds': jnp.asarray(uppers),
            'lower_bounds': jnp.asarray(lowers),
        }


# PAYOFF FUNCTIONS
# -----------------------------------------------------------------------------
def european_call_payoff(strike: float) -> Callable:
    """European call: max(y - K, 0)."""
    def payoff(x: Array, y: Array) -> Array:
        return jnp.maximum(y - strike, 0)
    return payoff


def european_put_payoff(strike: float) -> Callable:
    """European put: max(K - y, 0)."""
    def payoff(x: Array, y: Array) -> Array:
        return jnp.maximum(strike - y, 0)
    return payoff


def forward_start_call_payoff(strike_ratio: float) -> Callable:
    """Forward-start call: max(Y - k*X, 0).
    
    This depends on the COUPLING, not just marginals.
    Perfect example for MOT!
    """
    def payoff(x: Array, y: Array) -> Array:
        return jnp.maximum(y - strike_ratio * x, 0)
    return payoff


def variance_swap_payoff(strike_var: float) -> Callable:
    """Variance swap: (log(Y/X))^2 - K_var."""
    def payoff(x: Array, y: Array) -> Array:
        log_return = jnp.log(y / (x + 1e-10))
        return log_return ** 2 - strike_var
    return payoff


def lookback_proxy_payoff() -> Callable:
    """Lookback proxy: max(Y - X, 0).
    
    True lookback is path-dependent; this is 2-marginal approximation.
    """
    def payoff(x: Array, y: Array) -> Array:
        return jnp.maximum(y - x, 0)
    return payoff


def barrier_down_out_call_payoff(strike: float, barrier: float) -> Callable:
    """Down-and-out call (2-marginal approximation)."""
    def payoff(x: Array, y: Array) -> Array:
        vanilla = jnp.maximum(y - strike, 0)
        knocked_out = (x < barrier) | (y < barrier)
        return jnp.where(knocked_out, 0.0, vanilla)
    return payoff


# CONVENIENCE FUNCTION
# -----------------------------------------------------------------------------
def compute_robust_price_bounds(
    mu_samples: Array,
    nu_samples: Array,
    payoff_fn: Callable,
    forward_ratio: float = 1.0,
    epsilon: float = 0.1,
    martingale_weight: float = 10.0,
) -> Tuple[float, float]:
    """Quick price bounds computation.
    
    Args:
        mu_samples: Samples from T_1 marginal.
        nu_samples: Samples from T₂ marginal.
        payoff_fn: Payoff function g(x, y).
        forward_ratio: exp(r*tau) for martingale.
        epsilon: Entropic regularization.
        martingale_weight: Martingale penalty weight.
    
    Returns:
        (lower_bound, upper_bound) price interval.
    """
    solver = EntropicMOTSolver(epsilon=epsilon, martingale_weight=martingale_weight)
    result = solver.solve(mu_samples, nu_samples, payoff_fn, forward_ratio)
    return result.price_interval()


# MODULE EXPORTS
# -----------------------------------------------------------------------------
__all__ = [
    'DualPotentials',
    'RobustHedgingResult',
    'EntropicMOTSolver',
    'european_call_payoff',
    'european_put_payoff',
    'forward_start_call_payoff',
    'variance_swap_payoff',
    'lookback_proxy_payoff',
    'barrier_down_out_call_payoff',
    'compute_robust_price_bounds',
]
