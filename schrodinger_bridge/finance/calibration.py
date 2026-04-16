"""Calibration Tools: Risk-Neutral Density Extraction.

This module provides tools to extract risk-neutral probability distributions
from observed option prices - the key input for Schrödinger Bridge methods.

MAIN MATH TAKEAWAY: BREEDEN-LITZENBERGER
-
The fundamental theorem connecting options to probabilities:

    d^2C/dK^2 = e^{-rT} * p(K)

where p(K) is the risk-neutral probability density at strike K!

INTUITION: A butterfly spread centered at K with small width dK pays off
~1 if S_T  in [K - dK, K + dK], and 0 otherwise. Its price must be:
    
    Butterfly price ~= e^{-rT} * P(S_T  in [K - dK, K + dK]) ~= e^{-rT} * p(K) * 2dK

Since butterfly = C(K-dK) - 2C(K) + C(K+dK) ~= d^2C/dK^2 * dK^2, we get the result.

WHY THIS MATTERS FOR SB
-
1. Option prices -> Risk-neutral marginals at each expiry
2. Marginals become constraints for MartingaleSBSolver
3. SB finds LEAST committed model consistent with these marginals
4. Result: Model-free bounds on exotic prices!

USAGE
-
```python
from schrodinger_bridge.finance.calibration import (
    breeden_litzenberger_density,
    extract_risk_neutral_distribution,
)

# From market data
strikes = jnp.array([90, 95, 100, 105, 110])
call_prices = jnp.array([12.5, 9.0, 6.2, 4.0, 2.3])

# Extract density
density = breeden_litzenberger_density(strikes, call_prices, forward=100, r=0.05, T=1.0)

# Get samples for SB
dist = extract_risk_neutral_distribution(strikes, call_prices, forward=100, r=0.05, T=1.0)
samples = dist.sample(key, 1000)
```

Author: Schrödinger Bridge Library
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr

Array = jnp.ndarray
Scalar = Union[float, Array]
PRNGKey = jax.Array


# BREEDEN-LITZENBERGER DENSITY EXTRACTION
# -----------------------------------------------------------------------------
def breeden_litzenberger_density(
    strikes: Array,
    call_prices: Array,
    forward: float,
    r: float,
    T: float,
    method: str = 'finite_difference',
) -> Tuple[Array, Array]:
    """Extract risk-neutral density from call prices via Breeden-Litzenberger.
    
    The Breeden-Litzenberger formula:
        p(K) = e^{rT} * d^2C/dK^2
    
    Args:
        strikes: Strike prices, shape [n], must be sorted ascending.
        call_prices: Corresponding call prices, shape [n].
        forward: Forward price F = S*e^{(r-q)T}.
        r: Risk-free rate.
        T: Time to expiry.
        method: 'finite_difference' or 'spline'.
        
    Returns:
        (density_strikes, density_values) where density integrates to 1.
        
    Example:
        >>> K = jnp.linspace(80, 120, 21)
        >>> C = jnp.array([...])  # market call prices
        >>> K_d, p_d = breeden_litzenberger_density(K, C, 100, 0.05, 1.0)
    """
    strikes = jnp.asarray(strikes)
    call_prices = jnp.asarray(call_prices)
    
    # Discount factor
    discount = jnp.exp(-r * T)
    
    if method == 'finite_difference':
        # Second derivative via central differences
        # d^2C/dK^2 ~= (C_{i+1} - 2C_i + C_{i-1}) / (dK)^2
        
        n = len(strikes)
        if n < 3:
            raise ValueError("Need at least 3 strikes for finite difference")
        
        # Use interior points only (can't compute 2nd derivative at boundaries)
        density_strikes = strikes[1:-1]
        
        # Variable spacing: use average of neighboring intervals
        dK_left = strikes[1:-1] - strikes[:-2]
        dK_right = strikes[2:] - strikes[1:-1]
        dK_avg = (dK_left + dK_right) / 2
        
        # Central difference
        d2C = (call_prices[2:] - 2 * call_prices[1:-1] + call_prices[:-2]) / (dK_avg ** 2)
        
        # Convert to density
        density_values = d2C / discount
        
        # Floor at zero (numerical noise can give negative densities)
        density_values = jnp.maximum(density_values, 0)
        
        # Normalize to integrate to 1
        integral = jnp.trapezoid(density_values, density_strikes)
        if integral > 1e-10:
            density_values = density_values / integral
        
        return density_strikes, density_values
    
    elif method == 'spline':
        # Fit cubic spline and differentiate twice
        # This is smoother but requires scipy
        raise NotImplementedError("Spline method requires scipy; use finite_difference")
    
    else:
        raise ValueError(f"Unknown method: {method}")


def extrapolate_tails(
    strikes: Array,
    density: Array,
    forward: float,
    left_tail: str = 'lognormal',
    right_tail: str = 'lognormal',
    tail_vol: float = 0.3,
) -> Tuple[Array, Array]:
    """Extrapolate density tails beyond observed strikes.
    
    The Breeden-Litzenberger density is only defined between min/max strikes.
    We need to extrapolate tails for proper sampling.
    
    Args:
        strikes: Interior strike grid.
        density: Density values at strikes.
        forward: Forward price.
        left_tail: 'lognormal', 'powerlaw', or 'flat'.
        right_tail: Same options.
        tail_vol: Volatility for lognormal extrapolation.
        
    Returns:
        (extended_strikes, extended_density) with tails.
    """
    K_min, K_max = strikes[0], strikes[-1]
    dK = strikes[1] - strikes[0]
    
    # Extend grid
    n_left = int((K_min - 0.01 * forward) / dK)
    n_right = int((3 * forward - K_max) / dK)
    
    left_K = jnp.linspace(max(0.01 * forward, K_min - n_left * dK), K_min, n_left + 1)[:-1]
    right_K = jnp.linspace(K_max, min(3 * forward, K_max + n_right * dK), n_right + 1)[1:]
    
    # Lognormal tail density
    def lognormal_density(K, mu, sigma):
        return jnp.exp(-0.5 * ((jnp.log(K) - mu) / sigma) ** 2) / (K * sigma * jnp.sqrt(2 * jnp.pi))
    
    mu = jnp.log(forward)
    sigma = tail_vol
    
    # Match at boundaries
    if left_tail == 'lognormal':
        scale_left = density[0] / (lognormal_density(K_min, mu, sigma) + 1e-10)
        left_density = scale_left * lognormal_density(left_K, mu, sigma)
    else:
        left_density = jnp.zeros_like(left_K)
    
    if right_tail == 'lognormal':
        scale_right = density[-1] / (lognormal_density(K_max, mu, sigma) + 1e-10)
        right_density = scale_right * lognormal_density(right_K, mu, sigma)
    else:
        right_density = jnp.zeros_like(right_K)
    
    # Concatenate
    full_strikes = jnp.concatenate([left_K, strikes, right_K])
    full_density = jnp.concatenate([left_density, density, right_density])
    
    # Renormalize
    integral = jnp.trapezoid(full_density, full_strikes)
    if integral > 1e-10:
        full_density = full_density / integral
    
    return full_strikes, full_density


# RISK-NEUTRAL DISTRIBUTION CLASS
# -----------------------------------------------------------------------------
@dataclass
class RiskNeutralDistribution:
    """Risk-neutral distribution extracted from option prices.
    
    Provides sampling and density evaluation for use with SB methods.
    
    Attributes:
        strikes: Strike grid.
        density: Probability density values.
        cdf: Cumulative distribution values.
        forward: Forward price (mean under risk-neutral measure).
        expiry: Time to expiry.
    """
    strikes: Array
    density: Array
    cdf: Array
    forward: float
    expiry: float
    
    def pdf(self, x: Array) -> Array:
        """Evaluate probability density at x."""
        return jnp.interp(x, self.strikes, self.density, left=0.0, right=0.0)
    
    def log_pdf(self, x: Array) -> Array:
        """Evaluate log probability density at x."""
        return jnp.log(self.pdf(x) + 1e-10)
    
    def sample(self, key: PRNGKey, n: int) -> Array:
        """Sample n points from the distribution.
        
        Uses inverse CDF sampling.
        """
        u = jr.uniform(key, (n,))
        return jnp.interp(u, self.cdf, self.strikes)
    
    def mean(self) -> float:
        """Expected value (should equal forward)."""
        return float(jnp.trapezoid(self.strikes * self.density, self.strikes))
    
    def variance(self) -> float:
        """Variance of the distribution."""
        mu = self.mean()
        return float(jnp.trapezoid((self.strikes - mu) ** 2 * self.density, self.strikes))
    
    def std(self) -> float:
        """Standard deviation."""
        return float(jnp.sqrt(self.variance()))
    
    def quantile(self, p: float) -> float:
        """p-th quantile of the distribution."""
        return float(jnp.interp(p, self.cdf, self.strikes))
    
    @property
    def dim(self) -> int:
        """Dimension (always 1 for univariate)."""
        return 1


def extract_risk_neutral_distribution(
    strikes: Array,
    call_prices: Array,
    forward: float,
    r: float,
    T: float,
    extrapolate: bool = True,
    tail_vol: float = 0.3,
) -> RiskNeutralDistribution:
    """Extract complete risk-neutral distribution from call prices.
    
    This is the main entry point for calibration.
    
    Args:
        strikes: Strike prices (sorted).
        call_prices: Corresponding call prices.
        forward: Forward price.
        r: Risk-free rate.
        T: Time to expiry.
        extrapolate: Whether to extrapolate tails.
        tail_vol: Volatility for tail extrapolation.
        
    Returns:
        RiskNeutralDistribution object ready for SB methods.
        
    Example:
        >>> dist = extract_risk_neutral_distribution(K, C, F, r, T)
        >>> samples = dist.sample(key, 1000)  # For SB solver
        >>> print(f"Mean: {dist.mean():.2f}, Std: {dist.std():.2f}")
    """
    # Extract density
    K_density, density = breeden_litzenberger_density(strikes, call_prices, forward, r, T)
    
    # Extrapolate tails
    if extrapolate:
        K_density, density = extrapolate_tails(K_density, density, forward, tail_vol=tail_vol)
    
    # Compute CDF
    cdf = jnp.zeros_like(density)
    for i in range(1, len(density)):
        cdf = cdf.at[i].set(cdf[i-1] + 0.5 * (density[i] + density[i-1]) * (K_density[i] - K_density[i-1]))
    
    # Ensure CDF ends at 1
    cdf = cdf / (cdf[-1] + 1e-10)
    
    return RiskNeutralDistribution(
        strikes=K_density,
        density=density,
        cdf=cdf,
        forward=forward,
        expiry=T,
    )


def sample_from_density(
    key: PRNGKey,
    strikes: Array,
    density: Array,
    n: int,
) -> Array:
    """Sample from arbitrary density via inverse CDF.
    
    Utility function when you have raw (strikes, density) arrays.
    """
    # Build CDF
    cdf = jnp.cumsum(density) * (strikes[1] - strikes[0])
    cdf = cdf / cdf[-1]
    
    # Inverse CDF sampling
    u = jr.uniform(key, (n,))
    return jnp.interp(u, cdf, strikes)


# VOLATILITY SURFACE TOOLS
# -----------------------------------------------------------------------------
def interpolate_vol_surface(
    strikes: Array,
    expiries: Array,
    iv_grid: Array,
    K_query: Scalar,
    T_query: Scalar,
    method: str = 'bilinear',
) -> Scalar:
    """Interpolate implied volatility surface.
    
    Args:
        strikes: Strike grid, shape [n_K].
        expiries: Expiry grid, shape [n_T].
        iv_grid: Implied vol grid, shape [n_T, n_K].
        K_query: Query strike.
        T_query: Query expiry.
        method: Interpolation method ('bilinear', 'cubic').
        
    Returns:
        Interpolated implied volatility.
    """
    # Find surrounding grid points
    T_idx = jnp.clip(jnp.searchsorted(expiries, T_query) - 1, 0, len(expiries) - 2)
    K_idx = jnp.clip(jnp.searchsorted(strikes, K_query) - 1, 0, len(strikes) - 2)
    
    # Bilinear weights
    T_frac = (T_query - expiries[T_idx]) / (expiries[T_idx + 1] - expiries[T_idx] + 1e-10)
    K_frac = (K_query - strikes[K_idx]) / (strikes[K_idx + 1] - strikes[K_idx] + 1e-10)
    
    T_frac = jnp.clip(T_frac, 0, 1)
    K_frac = jnp.clip(K_frac, 0, 1)
    
    # Bilinear interpolation
    iv_00 = iv_grid[T_idx, K_idx]
    iv_01 = iv_grid[T_idx, K_idx + 1]
    iv_10 = iv_grid[T_idx + 1, K_idx]
    iv_11 = iv_grid[T_idx + 1, K_idx + 1]
    
    iv_0 = iv_00 * (1 - K_frac) + iv_01 * K_frac
    iv_1 = iv_10 * (1 - K_frac) + iv_11 * K_frac
    
    return iv_0 * (1 - T_frac) + iv_1 * T_frac


def total_variance_surface(
    strikes: Array,
    expiries: Array,
    iv_grid: Array,
) -> Array:
    """Convert implied vol surface to total variance surface.
    
    Total variance: w(K, T) = sigma^2(K, T) * T
    
    Total variance is often easier to interpolate (more linear).
    
    Returns:
        Total variance grid, shape [n_T, n_K].
    """
    return iv_grid ** 2 * expiries[:, None]


def implied_vol_from_total_variance(
    total_var: Array,
    expiries: Array,
) -> Array:
    """Convert total variance back to implied vol.
    
    sigma(K, T) = sqrt(w(K, T) / T)
    """
    return jnp.sqrt(total_var / expiries[:, None])


# ARBITRAGE CHECKS
# -----------------------------------------------------------------------------
def check_butterfly_arbitrage(
    strikes: Array,
    call_prices: Array,
) -> Tuple[bool, Array]:
    """Check for butterfly arbitrage in call prices.
    
    Butterfly arbitrage: C(K-dK) - 2C(K) + C(K+dK) < 0
    This would imply negative probability density!
    
    Returns:
        (is_arbitrage_free, violations) where violations[i] < 0 indicates arbitrage.
    """
    n = len(strikes)
    if n < 3:
        return True, jnp.array([])
    
    butterflies = call_prices[:-2] - 2 * call_prices[1:-1] + call_prices[2:]
    
    # Should be non-negative
    is_arb_free = jnp.all(butterflies >= -1e-6)
    
    return bool(is_arb_free), butterflies


def check_calendar_arbitrage(
    expiries: Array,
    atm_prices: Array,
) -> Tuple[bool, Array]:
    """Check for calendar arbitrage in ATM option prices.
    
    Calendar arbitrage: C(K, T_1) > C(K, T₂) for T_1 < T₂
    Option prices should increase with expiry (time value).
    
    Returns:
        (is_arbitrage_free, violations).
    """
    diffs = jnp.diff(atm_prices)
    
    # Should be non-negative
    is_arb_free = jnp.all(diffs >= -1e-6)
    
    return bool(is_arb_free), diffs


# MODULE EXPORTS
# -----------------------------------------------------------------------------
__all__ = [
    # Core density extraction
    'breeden_litzenberger_density',
    'extrapolate_tails',
    'extract_risk_neutral_distribution',
    'sample_from_density',
    # Distribution class
    'RiskNeutralDistribution',
    # Surface tools
    'interpolate_vol_surface',
    'total_variance_surface',
    'implied_vol_from_total_variance',
    # Arbitrage checks
    'check_butterfly_arbitrage',
    'check_calendar_arbitrage',
]
