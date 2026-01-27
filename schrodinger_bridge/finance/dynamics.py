"""Financial Reference Dynamics for Schrödinger Bridges.

This module provides stochastic volatility models as reference processes
for the Schrödinger Bridge problem:

    P* = argmin KL(P || R)  subject to marginal constraints

MAIN MATH TAKEAWAY
==================

The reference R determines what "maximum entropy" means:

| Model    | Pros                              | Cons                           |
|----------|-----------------------------------|--------------------------------|
| LocalVol | Matches all vanilla prices        | Unrealistic smile dynamics     |
| Heston   | Realistic vol clustering          | Limited smile flexibility      |
| SABR     | Excellent single-expiry fit       | Doesn't extend across expiries |
| RoughVol | Best short-term dynamics (H≈0.1)  | Hardest to calibrate           |

IMPORTANT: There's no free lunch! The SB finds min KL(P || R), so exotic
prices depend on R even when marginals are pinned to market!

ENHANCEMENTS (v2):
==================
1. transition_log_prob() - Compute transition probabilities for SV-aware coupling
2. conditional_mean_var() - Compute conditional moments for bridge construction
3. simulate_bridge() - SV-aware bridge simulation between two points

HESTON: THE WORKHORSE
=====================

Heston is the most widely used stochastic vol model:

    dS = rS dt + √v S dW^S
    dv = κ(θ - v) dt + ξ√v dW^v
    d⟨W^S, W^v⟩ = ρ dt

Key features:
- Mean-reverting variance (κ pulls v toward θ)
- Vol-of-vol ξ creates fat tails
- Correlation ρ < 0 creates the "leverage effect" (skew)

Feller condition 2κθ > ξ² ensures v > 0, but most equity calibrations 
violate this (need high ξ to fit observed skew).

Author: Schrödinger Bridge Library
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union
import warnings

import jax
import jax.numpy as jnp
from scipy.stats import norm
import numpy as np

# Handle imports for both standalone and package use
try:
    from ..types import Array, PRNGKey, Scalar
    from ..problem import ReferenceDynamics
except ImportError:
    # Standalone use
    Array = jnp.ndarray
    Scalar = Union[float, Array]
    PRNGKey = jax.Array
    
    # Define abstract base if not available
    import abc
    class ReferenceDynamics(abc.ABC):
        @abc.abstractmethod
        def drift(self, x: Array, t: Scalar) -> Array:
            pass
        
        @abc.abstractmethod
        def diffusion(self, x: Array, t: Scalar) -> Union[Scalar, Array]:
            pass
        
        @property
        @abc.abstractmethod
        def is_time_homogeneous(self) -> bool:
            pass


# =============================================================================
# LOCAL VOLATILITY (DUPIRE)
# =============================================================================

class LocalVolatilityDynamics(ReferenceDynamics):
    """Local Volatility (Dupire) reference dynamics.
    
    SDE: dS = rS dt + σ_loc(S,t) S dW
    
    In log-space: dX = (r - σ²/2) dt + σ_loc(eˣ, t) dW
    
    DUPIRE'S FORMULA
    ================
    Given market call prices C(K,T), the local vol is:
    
        σ²_loc(K,T) = (∂C/∂T + rK·∂C/∂K) / (½K²·∂²C/∂K²)
    
    This is remarkable: option prices uniquely determine σ_loc to match 
    ALL vanilla prices simultaneously.
    
    LIMITATION: Local vol produces unrealistic forward smile dynamics.
    The smile tends to flatten as we look forward in time, which doesn't
    match how markets actually behave.
    """
    
    def __init__(
        self,
        vol_surface: Union[Callable, Tuple[Array, Array, Array]],
        rate: float = 0.05,
        dividend_yield: float = 0.0,
        spot: float = 100.0,
    ):
        """Initialize local vol dynamics.
        
        Args:
            vol_surface: Either callable (S, t) → σ, or tuple 
                         (strikes, times, vol_grid) for interpolation.
            rate: Risk-free rate.
            dividend_yield: Dividend yield.
            spot: Spot price (for normalization).
        """
        self.rate = rate
        self.dividend_yield = dividend_yield
        self.spot = spot
        self._dim = 1
        
        if callable(vol_surface):
            self._vol_fn = vol_surface
        else:
            strikes, times, vol_grid = vol_surface
            self._strikes = jnp.asarray(strikes)
            self._times = jnp.asarray(times)
            self._vol_grid = jnp.asarray(vol_grid)
            self._vol_fn = self._interpolate_vol
    
    def _interpolate_vol(self, S: Array, t: Scalar) -> Array:
        """Bilinear interpolation of local vol surface."""
        S = jnp.atleast_1d(S)
        
        t_idx = jnp.clip(
            jnp.searchsorted(self._times, t) - 1,
            0, len(self._times) - 2
        )
        t_frac = (t - self._times[t_idx]) / (
            self._times[t_idx + 1] - self._times[t_idx] + 1e-10
        )
        t_frac = jnp.clip(t_frac, 0, 1)
        
        vol_t0 = jnp.interp(S, self._strikes, self._vol_grid[t_idx])
        vol_t1 = jnp.interp(S, self._strikes, self._vol_grid[t_idx + 1])
        
        return vol_t0 * (1 - t_frac) + vol_t1 * t_frac
    
    def drift(self, x: Array, t: Scalar) -> Array:
        """Drift in log-space: (r - q - σ²/2)."""
        x = jnp.atleast_1d(x)
        S = jnp.exp(x) * self.spot
        sigma = self._vol_fn(S, t)
        return jnp.full_like(x, self.rate - self.dividend_yield) - 0.5 * sigma ** 2
    
    def diffusion(self, x: Array, t: Scalar) -> Array:
        """Diffusion: σ_loc(S, t)."""
        x = jnp.atleast_1d(x)
        S = jnp.exp(x) * self.spot
        return self._vol_fn(S, t)
    
    @property
    def is_time_homogeneous(self) -> bool:
        return False
    
    @property
    def is_diffusion_scalar(self) -> bool:
        return False  # State-dependent
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def simulate(
        self,
        key: PRNGKey,
        x0: Array,
        num_steps: int,
        T: float,
    ) -> Tuple[Array, Array]:
        """Simulate paths under local vol dynamics (Euler-Maruyama)."""
        x0 = jnp.atleast_1d(x0)
        num_paths = len(x0)
        dt = T / num_steps
        times = jnp.linspace(0, T, num_steps + 1)
        
        paths = jnp.zeros((num_paths, num_steps + 1))
        paths = paths.at[:, 0].set(x0)
        
        keys = jax.random.split(key, num_steps)
        
        for i in range(num_steps):
            x = paths[:, i]
            t = times[i]
            dW = jax.random.normal(keys[i], (num_paths,)) * jnp.sqrt(dt)
            paths = paths.at[:, i + 1].set(
                x + self.drift(x, t) * dt + self.diffusion(x, t) * dW
            )
        
        return times, paths
    
    @classmethod
    def from_implied_vols(
        cls,
        strikes: Array,
        expiries: Array,
        iv_surface: Array,
        spot: float,
        rate: float,
    ) -> 'LocalVolatilityDynamics':
        """Create LocalVol from implied volatility surface.
        
        NOTE: This uses IV as a proxy for local vol. True Dupire requires
        differentiating call prices, which needs numerical care.
        """
        return cls(
            vol_surface=(strikes, expiries, iv_surface),
            rate=rate,
            spot=spot,
        )


# =============================================================================
# HESTON STOCHASTIC VOLATILITY (ENHANCED)
# =============================================================================

class HestonDynamics(ReferenceDynamics):
    """Heston Stochastic Volatility reference dynamics.
    
    SDE system (correlated Brownian motions):
        dS = rS dt + √v · S dW^S
        dv = κ(θ - v) dt + ξ√v dW^v
        d⟨W^S, W^v⟩ = ρ dt
    
    In log-space for numerical stability:
        dX = (r - v/2) dt + √v dW^S
        dv = κ(θ - v) dt + ξ√v dW^v
    
    KEY PARAMETERS
    ==============
    - κ (kappa): Mean reversion speed. Higher κ → faster return to θ
    - θ (theta): Long-run variance. θ = 0.04 means 20% long-run vol
    - ξ (xi): Vol-of-vol. Creates fat tails and smile curvature
    - ρ (rho): Spot-vol correlation. ρ < 0 creates skew (leverage effect)
    - v₀: Initial variance
    
    FELLER CONDITION: 2κθ > ξ²
    If satisfied, variance stays strictly positive.
    Most equity calibrations VIOLATE this (need high ξ for skew).
    
    ENHANCEMENTS (v2):
    ==================
    - transition_log_prob(): Compute log P(X_end | X_start) for SV coupling
    - conditional_mean_var(): E[X_end | X_start] and Var[X_end | X_start]
    - simulate_bridge(): SV-aware bridge simulation
    """
    
    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        v0: float = 0.04,
        rate: float = 0.05,
        spot: float = 100.0,
    ):
        """Initialize Heston dynamics.
        
        Args:
            kappa: Mean reversion speed of variance.
            theta: Long-run variance level.
            xi: Volatility of volatility.
            rho: Spot-vol correlation (usually negative for equities).
            v0: Initial variance.
            rate: Risk-free rate.
            spot: Spot price.
        """
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        self.rate = rate
        self.spot = spot
        self._dim = 2  # (log_S, v)
        
        # Check Feller condition
        if 2 * kappa * theta <= xi ** 2:
            warnings.warn(
                f"Feller condition violated: 2κθ = {2*kappa*theta:.4f} ≤ ξ² = {xi**2:.4f}. "
                "Variance may hit zero — consider using reflection."
            )
    
    def drift(self, x: Array, t: Scalar) -> Array:
        """Drift for (log_S, v) system."""
        x = jnp.atleast_2d(x)
        v = jnp.maximum(x[:, 1], 1e-8)  # Floor variance
        
        drift_logS = self.rate - 0.5 * v
        drift_v = self.kappa * (self.theta - v)
        
        return jnp.stack([drift_logS, drift_v], axis=-1)
    
    def diffusion(self, x: Array, t: Scalar) -> Array:
        """Diffusion matrix (Cholesky factor for correlated BMs).
        
        For correlation ρ, we use:
            dW^S = dZ₁
            dW^v = ρ dZ₁ + √(1-ρ²) dZ₂
        
        So L = [[√v, 0], [ρξ√v, ξ√v√(1-ρ²)]]
        """
        x = jnp.atleast_2d(x)
        v = jnp.maximum(x[:, 1], 1e-8)
        sqrt_v = jnp.sqrt(v)
        
        L = jnp.zeros((len(x), 2, 2))
        L = L.at[:, 0, 0].set(sqrt_v)
        L = L.at[:, 1, 0].set(self.rho * self.xi * sqrt_v)
        L = L.at[:, 1, 1].set(self.xi * sqrt_v * jnp.sqrt(1 - self.rho**2))
        
        return L
    
    @property
    def is_time_homogeneous(self) -> bool:
        return True
    
    @property
    def is_diffusion_scalar(self) -> bool:
        return False  # Matrix-valued
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def initial_state(self) -> Array:
        """Return initial state (log_S₀, v₀)."""
        return jnp.array([0.0, self.v0])
    
    # =========================================================================
    # ENHANCEMENT: Transition Probability Methods
    # =========================================================================
    
    def conditional_mean_var(
        self,
        x_start: Array,
        t_start: float,
        t_end: float,
        v_start: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Compute conditional moments E[X_end | X_start] and Var[X_end | X_start].
        
        ═══════════════════════════════════════════════════════════════════════════
        MAIN MATH: Heston Conditional Moments
        ═══════════════════════════════════════════════════════════════════════════
        
        Under Heston, log-price is approximately Gaussian conditional on variance path:
        
            E[log(S_T) | log(S_t)] ≈ log(S_t) + (r - θ/2)τ
            Var[log(S_T) | log(S_t)] ≈ θτ + (v_t - θ)(1 - e^{-κτ})/κ
        
        This is an approximation; true Heston has characteristic function solution.
        ═══════════════════════════════════════════════════════════════════════════
        
        Args:
            x_start: Starting log-prices, shape [n].
            t_start: Start time.
            t_end: End time.
            v_start: Starting variances (default: use v0).
            
        Returns:
            (mean, variance) each shape [n].
        """
        x_start = np.atleast_1d(x_start)
        tau = t_end - t_start
        
        if v_start is None:
            v_start = np.full_like(x_start, self.v0)
        
        # Expected variance over [t, T]
        exp_kappa_tau = np.exp(-self.kappa * tau)
        E_v = self.theta + (v_start - self.theta) * exp_kappa_tau
        
        # Mean of log-price
        mean = x_start + (self.rate - 0.5 * E_v) * tau
        
        # Variance of log-price (integrated variance)
        # Var = ∫_t^T E[v_s] ds ≈ θτ + (v_t - θ)(1 - exp(-κτ))/κ
        integrated_var = self.theta * tau + (v_start - self.theta) * (1 - exp_kappa_tau) / self.kappa
        
        return mean, np.maximum(integrated_var, 1e-8)
    
    def transition_log_prob(
        self,
        x_start: Array,
        x_end: Array,
        t_start: float,
        t_end: float,
        v_start: Optional[Array] = None,
    ) -> Array:
        """Compute log transition probability matrix log P(x_end | x_start).
        
        ═══════════════════════════════════════════════════════════════════════════
        MAIN MATH: SV Transition Probabilities for OT Coupling
        ═══════════════════════════════════════════════════════════════════════════
        
        Instead of using uniform prior in Sinkhorn, use SV-implied transitions:
        
            K_{ij} ∝ P(X_end = x_j | X_start = x_i) · exp(-C_{ij}/ε)
        
        This gives couplings that respect SV dynamics, not just marginals!
        ═══════════════════════════════════════════════════════════════════════════
        
        Args:
            x_start: Starting points, shape [n].
            x_end: Ending points, shape [m].
            t_start: Start time.
            t_end: End time.
            v_start: Starting variances (default: use v0).
            
        Returns:
            Log probability matrix, shape [n, m].
        """
        x_start = np.atleast_1d(x_start)
        x_end = np.atleast_1d(x_end)
        n, m = len(x_start), len(x_end)
        
        # Get conditional moments for each starting point
        mean, var = self.conditional_mean_var(x_start, t_start, t_end, v_start)
        std = np.sqrt(var)
        
        # Gaussian log-density: log p(x_end | x_start) = -0.5 * ((x - μ)/σ)² - log(σ√2π)
        log_prob = np.zeros((n, m))
        
        for i in range(n):
            z = (x_end - mean[i]) / (std[i] + 1e-10)
            log_prob[i] = -0.5 * z**2 - np.log(std[i] + 1e-10) - 0.5 * np.log(2 * np.pi)
        
        return log_prob
    
    def simulate_bridge(
        self,
        key: PRNGKey,
        x_start: Array,
        x_end: Array,
        t_start: float,
        t_end: float,
        num_steps: int,
        v_start: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array]:
        """Simulate SV-aware bridge from x_start to x_end.
        
        ═══════════════════════════════════════════════════════════════════════════
        MAIN MATH: SV Reference Bridge
        ═══════════════════════════════════════════════════════════════════════════
        
        Standard Brownian bridge: X_t | X_0, X_T ~ N(linear_interp, t(T-t)/T)
        
        SV bridge: Simulate Heston with drift adjusted to hit endpoint:
        
            dX = [μ_SV + λ(t)(X_end - X)/(T-t)] dt + √v dW
        
        where λ(t) blends from 0 (pure SV) to 1 (pure bridge) as t → T.
        
        This preserves volatility clustering while hitting endpoints!
        ═══════════════════════════════════════════════════════════════════════════
        
        Args:
            key: Random key.
            x_start: Starting log-prices, shape [n_paths].
            x_end: Ending log-prices, shape [n_paths].
            t_start: Start time.
            t_end: End time.
            num_steps: Number of time steps.
            v_start: Starting variances (default: use v0).
            
        Returns:
            (times, log_price_paths, variance_paths)
        """
        x_start = jnp.atleast_1d(x_start)
        x_end = jnp.atleast_1d(x_end)
        n_paths = len(x_start)
        
        if v_start is None:
            v_start = jnp.full(n_paths, self.v0)
        
        tau = t_end - t_start
        dt = tau / num_steps
        sqrt_dt = jnp.sqrt(dt)
        
        times = jnp.linspace(t_start, t_end, num_steps + 1)
        
        # Initialize
        X = x_start.copy()
        v = v_start.copy()
        
        log_paths = [X]
        var_paths = [v]
        
        keys = jax.random.split(key, num_steps)
        
        for i in range(num_steps - 1):
            k1, k2 = jax.random.split(keys[i])
            
            t_current = times[i]
            remaining = t_end - t_current
            
            # SV drift
            sv_drift = self.rate - 0.5 * v
            
            # Bridge drift: pull toward endpoint
            bridge_drift = (x_end - X) / (remaining + 1e-8)
            
            # Blend: increase bridge influence as we approach end
            blend = jnp.minimum(0.5, dt / (remaining + 1e-8))
            combined_drift = (1 - blend) * sv_drift + blend * bridge_drift
            
            # Correlated noise
            sqrt_v = jnp.sqrt(jnp.maximum(v, 1e-8))
            Z1 = jax.random.normal(k1, (n_paths,))
            Z2 = jax.random.normal(k2, (n_paths,))
            
            dW_X = Z1 * sqrt_dt
            dW_v = (self.rho * Z1 + jnp.sqrt(1 - self.rho**2) * Z2) * sqrt_dt
            
            # Update
            X = X + combined_drift * dt + sqrt_v * dW_X
            v = v + self.kappa * (self.theta - v) * dt + self.xi * sqrt_v * dW_v
            v = jnp.maximum(v, 1e-8)
            
            log_paths.append(X)
            var_paths.append(v)
        
        # Final step: snap to endpoint
        log_paths.append(x_end)
        var_paths.append(v)
        
        return times, jnp.stack(log_paths, axis=1), jnp.stack(var_paths, axis=1)
    
    def simulate(
        self,
        key: PRNGKey,
        num_paths: int,
        num_steps: int,
        T: float,
    ) -> Tuple[Array, Array, Array]:
        """Simulate Heston paths with variance reflection.
        
        Returns:
            (times, S_paths, v_paths) where paths have shape [num_paths, num_steps+1].
        """
        dt = T / num_steps
        times = jnp.linspace(0, T, num_steps + 1)
        
        log_S = jnp.zeros((num_paths, num_steps + 1))
        v = jnp.zeros((num_paths, num_steps + 1))
        v = v.at[:, 0].set(self.v0)
        
        keys = jax.random.split(key, num_steps)
        
        for i in range(num_steps):
            k1, k2 = jax.random.split(keys[i])
            Z1 = jax.random.normal(k1, (num_paths,))
            Z2 = jax.random.normal(k2, (num_paths,))
            
            sqrt_v_i = jnp.sqrt(jnp.maximum(v[:, i], 1e-8))
            dW_S = Z1 * jnp.sqrt(dt)
            dW_v = (self.rho * Z1 + jnp.sqrt(1 - self.rho**2) * Z2) * jnp.sqrt(dt)
            
            log_S = log_S.at[:, i+1].set(
                log_S[:, i] + (self.rate - 0.5 * v[:, i]) * dt + sqrt_v_i * dW_S
            )
            
            v_new = v[:, i] + self.kappa * (self.theta - v[:, i]) * dt + self.xi * sqrt_v_i * dW_v
            v = v.at[:, i+1].set(jnp.maximum(v_new, 1e-8))  # Reflection
        
        S_paths = self.spot * jnp.exp(log_S)
        return times, S_paths, v


# =============================================================================
# SABR MODEL
# =============================================================================

class SABRDynamics(ReferenceDynamics):
    """SABR Stochastic Alpha-Beta-Rho dynamics.
    
    SDE (forward space):
        dF = σ F^β dW^F
        dσ = α σ dW^σ
        d⟨W^F, W^σ⟩ = ρ dt
    
    KEY PARAMETERS
    ==============
    - α (alpha): Vol-of-vol. Controls smile curvature.
    - β (beta): CEV exponent. β=1 is lognormal, β=0 is normal.
    - ρ (rho): Correlation. Controls skew direction.
    - σ₀: Initial volatility.
    
    HAGAN'S FORMULA
    ===============
    SABR has a famous closed-form approximation for implied volatility,
    making it the industry standard for single-expiry smile fitting.
    
    LIMITATION: SABR is a single-expiry model. It doesn't extend cleanly
    to term structure (different params per expiry = inconsistent dynamics).
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.5,
        rho: float = -0.3,
        sigma0: float = 0.20,
        forward0: float = 100.0,
    ):
        """Initialize SABR dynamics.
        
        Args:
            alpha: Vol-of-vol parameter.
            beta: CEV exponent (0 = normal, 1 = lognormal).
            rho: Correlation between forward and vol.
            sigma0: Initial volatility.
            forward0: Initial forward price.
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.sigma0 = sigma0
        self.forward0 = forward0
        self._dim = 2
    
    def drift(self, x: Array, t: Scalar) -> Array:
        """Drift is zero (martingale dynamics)."""
        x = jnp.atleast_2d(x)
        return jnp.zeros_like(x)
    
    def diffusion(self, x: Array, t: Scalar) -> Array:
        """Diffusion matrix for correlated dynamics."""
        x = jnp.atleast_2d(x)
        F = jnp.maximum(x[:, 0], 1e-8)
        sigma = jnp.maximum(x[:, 1], 1e-8)
        
        L = jnp.zeros((len(x), 2, 2))
        L = L.at[:, 0, 0].set(sigma * F ** self.beta)
        L = L.at[:, 1, 0].set(self.rho * self.alpha * sigma)
        L = L.at[:, 1, 1].set(jnp.sqrt(1 - self.rho**2) * self.alpha * sigma)
        
        return L
    
    @property
    def is_time_homogeneous(self) -> bool:
        return True
    
    @property
    def is_diffusion_scalar(self) -> bool:
        return False
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def implied_vol(self, K: float, T: float) -> float:
        """Compute Black implied vol using Hagan's approximation (2002).
        
        This is the famous SABR formula that made SABR practical.
        """
        F = self.forward0
        
        if abs(F - K) < 1e-10:
            # ATM formula
            sigma_atm = self.sigma0 / F ** (1 - self.beta) * (
                1 + T * (
                    (1 - self.beta) ** 2 * self.sigma0 ** 2 / (24 * F ** (2 - 2*self.beta))
                    + self.rho * self.beta * self.alpha * self.sigma0 / (4 * F ** (1 - self.beta))
                    + (2 - 3 * self.rho ** 2) * self.alpha ** 2 / 24
                ))
            return float(sigma_atm)
        
        # General Hagan formula
        log_FK = jnp.log(F / K)
        FK_mid = jnp.sqrt(F * K)
        
        z = self.alpha / self.sigma0 * FK_mid ** (1 - self.beta) * log_FK
        x_z = jnp.log((jnp.sqrt(1 - 2*self.rho*z + z**2) + z - self.rho) / (1 - self.rho))
        
        sigma_B = (
            self.sigma0 * z / (x_z * FK_mid ** (1 - self.beta))
            * (1 + T * (
                (1 - self.beta) ** 2 * self.sigma0 ** 2 / (24 * FK_mid ** (2 - 2*self.beta))
                + self.rho * self.beta * self.alpha * self.sigma0 / (4 * FK_mid ** (1 - self.beta))
                + (2 - 3 * self.rho ** 2) * self.alpha ** 2 / 24
            ))
        )
        
        return float(sigma_B)


# =============================================================================
# ROUGH VOLATILITY
# =============================================================================

class RoughVolatilityDynamics(ReferenceDynamics):
    """Rough Volatility reference dynamics (Markovian approximation).
    
    The Rough Bergomi model:
        dS = S √v dW
        v_t = ξ(t) · exp(η·W^H_t - η²t^{2H}/2)
    
    where W^H is fractional Brownian motion with Hurst parameter H < 0.5.
    
    KEY INSIGHT (Gatheral, Jaisson, Rosenbaum 2018)
    ===============================================
    Empirical Hurst parameter H ≈ 0.05-0.15 for equity markets!
    This is MUCH rougher than standard Brownian motion (H = 0.5).
    
    Rough vol explains:
    - Observed term structure of ATM skew
    - Short-term smile dynamics
    - Why standard models underestimate short-dated skew
    
    CHALLENGE: Fractional BM is not Markovian (infinite-dimensional state).
    We approximate with a sum of OU processes:
        W^H_t ≈ Σ_k w_k Y_k(t)  where dY_k = -λ_k Y_k dt + dW
    """
    
    def __init__(
        self,
        H: float = 0.1,
        eta: float = 1.0,
        xi: Callable[[float], float] = lambda t: 0.04,
        num_factors: int = 5,
        rate: float = 0.05,
        spot: float = 100.0,
    ):
        """Initialize rough vol dynamics.
        
        Args:
            H: Hurst parameter (< 0.5 for roughness, empirical ≈ 0.1).
            eta: Volatility of volatility.
            xi: Forward variance curve ξ(t).
            num_factors: Number of OU factors in Markovian approximation.
            rate: Risk-free rate.
            spot: Spot price.
        """
        self.H = H
        self.eta = eta
        self.xi = xi
        self.num_factors = num_factors
        self.rate = rate
        self.spot = spot
        
        self._dim = 1 + num_factors
        self._lambdas = jnp.array([2.0 ** k for k in range(num_factors)])
        self._weights = self._compute_weights()
    
    def _compute_weights(self) -> Array:
        """Compute weights for Markovian approximation."""
        weights = self._lambdas ** (self.H - 0.5)
        return weights / jnp.sum(weights)
    
    def _variance(self, factors: Array, t: Scalar) -> Array:
        """Compute variance from factors."""
        weighted_sum = jnp.sum(self._weights * factors, axis=-1)
        return self.xi(t) * jnp.exp(
            self.eta * weighted_sum - 0.5 * self.eta ** 2 * t ** (2 * self.H)
        )
    
    def drift(self, x: Array, t: Scalar) -> Array:
        """Drift for (log_S, factors) system."""
        x = jnp.atleast_2d(x)
        factors = x[:, 1:]
        
        v = self._variance(factors, t)
        drift_logS = self.rate - 0.5 * v
        drift_factors = -self._lambdas[None, :] * factors
        
        return jnp.concatenate([drift_logS[:, None], drift_factors], axis=-1)
    
    def diffusion(self, x: Array, t: Scalar) -> Array:
        """Diffusion for rough vol model."""
        x = jnp.atleast_2d(x)
        factors = x[:, 1:]
        
        v = self._variance(factors, t)
        sqrt_v = jnp.sqrt(jnp.maximum(v, 1e-8))
        
        diff = jnp.zeros((len(x), self._dim, self._dim))
        diff = diff.at[:, 0, 0].set(sqrt_v)
        
        for k in range(self.num_factors):
            diff = diff.at[:, k + 1, k + 1].set(jnp.sqrt(2 * self._lambdas[k]))
        
        return diff
    
    @property
    def is_time_homogeneous(self) -> bool:
        return False
    
    @property
    def is_diffusion_scalar(self) -> bool:
        return False
    
    @property
    def dim(self) -> int:
        return self._dim


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_vol_surface_from_sabr(
    sabr: SABRDynamics,
    strikes: Array,
    expiries: Array,
) -> Array:
    """Generate implied vol surface from SABR parameters.
    
    Args:
        sabr: SABRDynamics instance.
        strikes: Strike grid.
        expiries: Expiry grid.
        
    Returns:
        Implied vol grid, shape [n_T, n_K].
    """
    iv_grid = jnp.zeros((len(expiries), len(strikes)))
    
    for i, T in enumerate(expiries):
        for j, K in enumerate(strikes):
            iv = sabr.implied_vol(float(K), float(T))
            iv_grid = iv_grid.at[i, j].set(iv)
    
    return iv_grid


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'LocalVolatilityDynamics',
    'HestonDynamics',
    'SABRDynamics',
    'RoughVolatilityDynamics',
    'create_vol_surface_from_sabr',
]
