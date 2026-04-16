"""Option Pricing and Greeks.

This module provides classical option pricing formulas and sensitivities.

MATHEMATICAL FOUNDATION
-
Black-Scholes assumes geometric Brownian motion under risk-neutral measure:
    dS = rS dt + sigmaS dW

The resulting call price has the famous closed form:
    C = S*N(d_1) - K*e^{-rT}*N(d₂)

where:
    d_1 = [ln(S/K) + (r + sigma^2/2)T] / (sigmasqrtT)
    d₂ = d_1 - sigmasqrtT

MAIN MATH TAKEAWAY
-
The Black-Scholes formula can be understood as:

    C = e^{-rT} * E^Q[max(S_T - K, 0)]
      = e^{-rT} * [E^Q[S_T * 1_{S_T > K}] - K * P^Q(S_T > K)]
      = S * N(d_1) - K*e^{-rT} * N(d₂)

The two N(*) terms are:
- N(d₂) = Risk-neutral probability of finishing in-the-money
- N(d_1) = Delta-weighted probability (probability under stock measure)

GREEKS INTERPRETATION
-
Greeks measure sensitivity to inputs:
- Δ (Delta) = dC/dS = Shares to hold for delta-neutral hedge
- Γ (Gamma) = d^2C/dS^2 = Convexity, measures delta instability
- nu (Vega)  = dC/dsigma = Sensitivity to volatility (NOT in BS model!)
- Θ (Theta) = dC/dt = Time decay
- rho (Rho)   = dC/dr = Interest rate sensitivity

Author: Schrödinger Bridge Library
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

# Type aliases
Array = jnp.ndarray
Scalar = Union[float, Array]


# NORMAL CDF AND PDF (for clarity)
# -----------------------------------------------------------------------------
def _N(x: Scalar) -> Scalar:
    """Standard normal CDF: N(x) = P(Z <= x)."""
    return norm.cdf(x)


def _n(x: Scalar) -> Scalar:
    """Standard normal PDF: n(x) = (1/sqrt2pi)*exp(-x^2/2)."""
    return norm.pdf(x)


# BLACK-SCHOLES FORMULA
# -----------------------------------------------------------------------------
def _compute_d1_d2(
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    sigma: Scalar,
    q: Scalar = 0.0,
) -> Tuple[Scalar, Scalar]:
    """Compute d_1 and d₂ for Black-Scholes.
    
    d_1 = [ln(S/K) + (r - q + sigma^2/2)T] / (sigmasqrtT)
    d₂ = d_1 - sigmasqrtT
    
    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry (years).
        r: Risk-free rate (continuous).
        sigma: Volatility.
        q: Dividend yield (continuous).
    """
    sqrt_T = jnp.sqrt(jnp.maximum(T, 1e-10))
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T + 1e-10)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def black_scholes_call(
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    sigma: Scalar,
    q: Scalar = 0.0,
) -> Scalar:
    """Black-Scholes call option price.
    
    C = S*e^{-qT}*N(d_1) - K*e^{-rT}*N(d₂)
    
    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry (years).
        r: Risk-free rate (continuous compounding).
        sigma: Volatility (annualized).
        q: Dividend yield (continuous).
        
    Returns:
        Call option price.
        
    Example:
        >>> black_scholes_call(100, 100, 1.0, 0.05, 0.20)
        10.45...  # ATM 1-year call with 20% vol
    """
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma, q)
    return S * jnp.exp(-q * T) * _N(d1) - K * jnp.exp(-r * T) * _N(d2)


def black_scholes_put(
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    sigma: Scalar,
    q: Scalar = 0.0,
) -> Scalar:
    """Black-Scholes put option price.
    
    P = K*e^{-rT}*N(-d₂) - S*e^{-qT}*N(-d_1)
    
    Or via put-call parity: P = C - S*e^{-qT} + K*e^{-rT}
    """
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma, q)
    return K * jnp.exp(-r * T) * _N(-d2) - S * jnp.exp(-q * T) * _N(-d1)


def black_scholes_price(
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    sigma: Scalar,
    is_call: bool = True,
    q: Scalar = 0.0,
) -> Scalar:
    """Unified Black-Scholes price for call or put."""
    if is_call:
        return black_scholes_call(S, K, T, r, sigma, q)
    else:
        return black_scholes_put(S, K, T, r, sigma, q)


# GREEKS
# -----------------------------------------------------------------------------
def bs_delta(
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    sigma: Scalar,
    is_call: bool = True,
    q: Scalar = 0.0,
) -> Scalar:
    """Delta: dC/dS or dP/dS.
    
    Call delta: Δ_C = e^{-qT}*N(d_1)   in [0, 1]
    Put delta:  Δ_P = -e^{-qT}*N(-d_1)  in [-1, 0]
    
    INTERPRETATION: Number of shares to hold for delta-neutral hedge.
    """
    d1, _ = _compute_d1_d2(S, K, T, r, sigma, q)
    discount_q = jnp.exp(-q * T)
    if is_call:
        return discount_q * _N(d1)
    else:
        return -discount_q * _N(-d1)


def bs_gamma(
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    sigma: Scalar,
    q: Scalar = 0.0,
) -> Scalar:
    """Gamma: d^2C/dS^2 (same for call and put).
    
    Γ = e^{-qT}*n(d_1) / (S*sigma*sqrtT)
    
    INTERPRETATION: Rate of change of delta. High gamma near ATM at expiry.
    Gamma is always positive - options are convex in spot.
    """
    d1, _ = _compute_d1_d2(S, K, T, r, sigma, q)
    sqrt_T = jnp.sqrt(jnp.maximum(T, 1e-10))
    return jnp.exp(-q * T) * _n(d1) / (S * sigma * sqrt_T + 1e-10)


def bs_vega(
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    sigma: Scalar,
    q: Scalar = 0.0,
) -> Scalar:
    """Vega: dC/dsigma (same for call and put).
    
    nu = S*e^{-qT}*sqrtT*n(d_1)
    
    INTERPRETATION: Sensitivity to volatility. Note that sigma is assumed
    constant in BS, so vega is "out of model" - but essential for trading!
    
    Returns vega per 1% vol move (divide by 100 for per-point).
    """
    d1, _ = _compute_d1_d2(S, K, T, r, sigma, q)
    sqrt_T = jnp.sqrt(jnp.maximum(T, 1e-10))
    # Standard vega (per unit vol)
    return S * jnp.exp(-q * T) * sqrt_T * _n(d1)


def bs_theta(
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    sigma: Scalar,
    is_call: bool = True,
    q: Scalar = 0.0,
) -> Scalar:
    """Theta: -dC/dT (time decay, per year).
    
    Note the negative sign: theta measures how much value you LOSE per day.
    
    For calls:
        Θ = -[S*e^{-qT}*n(d_1)*sigma/(2sqrtT)] - r*K*e^{-rT}*N(d₂) + q*S*e^{-qT}*N(d_1)
    
    INTERPRETATION: Long options have negative theta (time decay).
    Theta is the "rent" you pay for gamma exposure.
    """
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma, q)
    sqrt_T = jnp.sqrt(jnp.maximum(T, 1e-10))
    
    discount_q = jnp.exp(-q * T)
    discount_r = jnp.exp(-r * T)
    
    # Common term (time decay of intrinsic randomness)
    term1 = -S * discount_q * _n(d1) * sigma / (2 * sqrt_T + 1e-10)
    
    if is_call:
        return term1 - r * K * discount_r * _N(d2) + q * S * discount_q * _N(d1)
    else:
        return term1 + r * K * discount_r * _N(-d2) - q * S * discount_q * _N(-d1)


def bs_rho(
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    sigma: Scalar,
    is_call: bool = True,
    q: Scalar = 0.0,
) -> Scalar:
    """Rho: dC/dr (interest rate sensitivity).
    
    Call rho: rho_C = K*T*e^{-rT}*N(d₂)
    Put rho:  rho_P = -K*T*e^{-rT}*N(-d₂)
    
    Returns rho per 1% rate move (divide by 100 for per-bp).
    """
    _, d2 = _compute_d1_d2(S, K, T, r, sigma, q)
    discount_r = jnp.exp(-r * T)
    
    if is_call:
        return K * T * discount_r * _N(d2)
    else:
        return -K * T * discount_r * _N(-d2)


def compute_all_greeks(
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    sigma: Scalar,
    is_call: bool = True,
    q: Scalar = 0.0,
) -> Dict[str, Scalar]:
    """Compute all Greeks at once.
    
    Returns:
        Dictionary with 'delta', 'gamma', 'vega', 'theta', 'rho'.
    """
    return {
        'price': black_scholes_price(S, K, T, r, sigma, is_call, q),
        'delta': bs_delta(S, K, T, r, sigma, is_call, q),
        'gamma': bs_gamma(S, K, T, r, sigma, q),
        'vega': bs_vega(S, K, T, r, sigma, q),
        'theta': bs_theta(S, K, T, r, sigma, is_call, q),
        'rho': bs_rho(S, K, T, r, sigma, is_call, q),
    }


# IMPLIED VOLATILITY
# -----------------------------------------------------------------------------
def implied_volatility(
    price: Scalar,
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    is_call: bool = True,
    q: Scalar = 0.0,
    initial_guess: Scalar = 0.2,
    tol: float = 1e-6,
    max_iters: int = 50,
) -> Scalar:
    """Compute implied volatility using Newton-Raphson.
    
    Finds sigma such that BS(S, K, T, r, sigma) = market_price.
    
    The Newton update is: sigma_{n+1} = sigma_n - (BS(sigma_n) - price) / vega(sigma_n)
    
    Args:
        price: Market option price.
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        is_call: True for call, False for put.
        q: Dividend yield.
        initial_guess: Starting volatility guess.
        tol: Convergence tolerance.
        max_iters: Maximum iterations.
        
    Returns:
        Implied volatility.
        
    Example:
        >>> iv = implied_volatility(10.45, 100, 100, 1.0, 0.05)
        >>> print(f"{iv:.2%}")  # Should be ~20%
    """
    sigma = initial_guess
    
    for _ in range(max_iters):
        bs_price = black_scholes_price(S, K, T, r, sigma, is_call, q)
        vega = bs_vega(S, K, T, r, sigma, q)
        
        # Avoid division by zero
        vega = jnp.maximum(vega, 1e-10)
        
        diff = bs_price - price
        sigma_new = sigma - diff / vega
        
        # Keep sigma positive
        sigma_new = jnp.maximum(sigma_new, 1e-6)
        
        if jnp.abs(sigma_new - sigma) < tol:
            return sigma_new
        
        sigma = sigma_new
    
    return sigma


# Alias for consistency
implied_volatility_newton = implied_volatility


def implied_volatility_bisection(
    price: Scalar,
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    is_call: bool = True,
    q: Scalar = 0.0,
    sigma_low: float = 0.001,
    sigma_high: float = 3.0,
    tol: float = 1e-6,
    max_iters: int = 100,
) -> Scalar:
    """Implied volatility via bisection (more robust, slower).
    
    Use when Newton fails (e.g., deep OTM options).
    """
    for _ in range(max_iters):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = black_scholes_price(S, K, T, r, sigma_mid, is_call, q)
        
        if jnp.abs(price_mid - price) < tol:
            return sigma_mid
        
        if price_mid > price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    
    return (sigma_low + sigma_high) / 2


# BACHELIER (NORMAL) MODEL
# -----------------------------------------------------------------------------
def bachelier_call(
    F: Scalar,
    K: Scalar,
    T: Scalar,
    sigma_n: Scalar,
    discount: Scalar = 1.0,
) -> Scalar:
    """Bachelier (normal) model call price.
    
    Assumes dF = sigma_n dW (arithmetic, not geometric).
    
    C = discount * [(F - K)*N(d) + sigma_n*sqrtT*n(d)]
    
    where d = (F - K) / (sigma_n*sqrtT)
    
    Args:
        F: Forward price.
        K: Strike price.
        T: Time to expiry.
        sigma_n: Normal volatility (in price units, not %).
        discount: Discount factor e^{-rT}.
    """
    sqrt_T = jnp.sqrt(jnp.maximum(T, 1e-10))
    d = (F - K) / (sigma_n * sqrt_T + 1e-10)
    return discount * ((F - K) * _N(d) + sigma_n * sqrt_T * _n(d))


def bachelier_put(
    F: Scalar,
    K: Scalar,
    T: Scalar,
    sigma_n: Scalar,
    discount: Scalar = 1.0,
) -> Scalar:
    """Bachelier (normal) model put price."""
    sqrt_T = jnp.sqrt(jnp.maximum(T, 1e-10))
    d = (F - K) / (sigma_n * sqrt_T + 1e-10)
    return discount * ((K - F) * _N(-d) + sigma_n * sqrt_T * _n(d))


def bachelier_implied_vol(
    price: Scalar,
    F: Scalar,
    K: Scalar,
    T: Scalar,
    is_call: bool = True,
    discount: Scalar = 1.0,
    tol: float = 1e-6,
    max_iters: int = 50,
) -> Scalar:
    """Implied normal volatility for Bachelier model."""
    # Initial guess based on ATM approximation: C ~= 0.4 * sigma_n * sqrtT
    sigma_n = price / (0.4 * jnp.sqrt(T) * discount + 1e-10)
    
    price_fn = bachelier_call if is_call else bachelier_put
    
    for _ in range(max_iters):
        model_price = price_fn(F, K, T, sigma_n, discount)
        
        # Vega for Bachelier: dC/dsigma_n = discount * sqrtT * n(d)
        sqrt_T = jnp.sqrt(jnp.maximum(T, 1e-10))
        d = (F - K) / (sigma_n * sqrt_T + 1e-10)
        vega = discount * sqrt_T * _n(d)
        vega = jnp.maximum(vega, 1e-10)
        
        diff = model_price - price
        sigma_n_new = sigma_n - diff / vega
        sigma_n_new = jnp.maximum(sigma_n_new, 1e-10)
        
        if jnp.abs(sigma_n_new - sigma_n) < tol:
            return sigma_n_new
        
        sigma_n = sigma_n_new
    
    return sigma_n


# PUT-CALL PARITY
# -----------------------------------------------------------------------------
def put_call_parity_call(
    put_price: Scalar,
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    q: Scalar = 0.0,
) -> Scalar:
    """Compute call price from put using put-call parity.
    
    C - P = S*e^{-qT} - K*e^{-rT}
    """
    return put_price + S * jnp.exp(-q * T) - K * jnp.exp(-r * T)


def put_call_parity_put(
    call_price: Scalar,
    S: Scalar,
    K: Scalar,
    T: Scalar,
    r: Scalar,
    q: Scalar = 0.0,
) -> Scalar:
    """Compute put price from call using put-call parity."""
    return call_price - S * jnp.exp(-q * T) + K * jnp.exp(-r * T)


# MODULE EXPORTS
# -----------------------------------------------------------------------------
__all__ = [
    # Black-Scholes
    'black_scholes_call',
    'black_scholes_put',
    'black_scholes_price',
    # Greeks
    'bs_delta',
    'bs_gamma',
    'bs_vega',
    'bs_theta',
    'bs_rho',
    'compute_all_greeks',
    # Implied vol
    'implied_volatility',
    'implied_volatility_newton',
    'implied_volatility_bisection',
    # Bachelier
    'bachelier_call',
    'bachelier_put',
    'bachelier_implied_vol',
    # Parity
    'put_call_parity_call',
    'put_call_parity_put',
]
