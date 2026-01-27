"""Term Structure: Yield Curves and Forward Prices.

This module provides tools for discount factors, forward rates, and yield
curve construction — essential inputs for derivatives pricing.

MAIN CONCEPTS
=============

Discount Factor D(t):
    Present value of $1 received at time t
    D(t) = exp(-r·t) for continuous compounding

Zero Rate r(t):
    Rate that gives D(t) = exp(-r(t)·t)
    Also called spot rate or zero-coupon yield

Forward Rate f(t₁, t₂):
    Rate locked in today for borrowing from t₁ to t₂
    Related to zero rates: exp(-r(t₂)·t₂) = exp(-r(t₁)·t₁) · exp(-f·(t₂-t₁))

MAIN MATH TAKEAWAY
==================

All three are equivalent ways to express the same information:

    D(t) = exp(-r(t)·t) = exp(-∫₀ᵗ f(s) ds)

The instantaneous forward rate is:
    f(t) = -d/dt[log D(t)] = r(t) + t·r'(t)

WHY THIS MATTERS FOR SB
=======================

The forward curve determines the martingale constraint:
    E^Q[S_T | F_t] = F(t, T) = S_t · D(t)/D(T)

For MartingaleSBSolver, you need to specify forward ratios between
marginal times, which come from the yield curve.

Author: Schrödinger Bridge Library
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp

Array = jnp.ndarray
Scalar = Union[float, Array]


# =============================================================================
# BASIC FUNCTIONS
# =============================================================================

def discount_factor(
    rate: float,
    T: float,
    compounding: str = 'continuous',
) -> float:
    """Compute discount factor.
    
    Args:
        rate: Interest rate.
        T: Time to maturity.
        compounding: 'continuous', 'annual', 'semi', 'quarterly'.
        
    Returns:
        Discount factor D(T).
    """
    if compounding == 'continuous':
        return float(jnp.exp(-rate * T))
    elif compounding == 'annual':
        return float(1 / (1 + rate) ** T)
    elif compounding == 'semi':
        return float(1 / (1 + rate / 2) ** (2 * T))
    elif compounding == 'quarterly':
        return float(1 / (1 + rate / 4) ** (4 * T))
    else:
        raise ValueError(f"Unknown compounding: {compounding}")


def zero_rate(
    discount: float,
    T: float,
    compounding: str = 'continuous',
) -> float:
    """Compute zero rate from discount factor.
    
    Args:
        discount: Discount factor D(T).
        T: Time to maturity.
        compounding: Compounding convention.
        
    Returns:
        Zero rate r(T).
    """
    if T <= 0:
        return 0.0
    
    if compounding == 'continuous':
        return float(-jnp.log(discount) / T)
    elif compounding == 'annual':
        return float((1 / discount) ** (1 / T) - 1)
    elif compounding == 'semi':
        return float(2 * ((1 / discount) ** (1 / (2 * T)) - 1))
    elif compounding == 'quarterly':
        return float(4 * ((1 / discount) ** (1 / (4 * T)) - 1))
    else:
        raise ValueError(f"Unknown compounding: {compounding}")


def forward_rate(
    D_t1: float,
    D_t2: float,
    t1: float,
    t2: float,
    compounding: str = 'continuous',
) -> float:
    """Compute forward rate between two times.
    
    The forward rate f(t₁, t₂) satisfies:
        D(t₂) = D(t₁) · exp(-f·(t₂-t₁))   [continuous]
    
    Args:
        D_t1: Discount factor to t₁.
        D_t2: Discount factor to t₂.
        t1: Start time.
        t2: End time.
        compounding: Compounding convention.
        
    Returns:
        Forward rate f(t₁, t₂).
    """
    tau = t2 - t1
    if tau <= 0:
        return 0.0
    
    if compounding == 'continuous':
        return float(-jnp.log(D_t2 / D_t1) / tau)
    elif compounding in ('annual', 'simple'):
        return float((D_t1 / D_t2 - 1) / tau)
    else:
        raise ValueError(f"Forward rate not implemented for {compounding}")


def instantaneous_forward(
    zero_rates: Array,
    times: Array,
) -> Array:
    """Compute instantaneous forward curve from zero rates.
    
    f(t) = r(t) + t · dr/dt
    
    Args:
        zero_rates: Zero rates r(t) at each time.
        times: Corresponding times.
        
    Returns:
        Instantaneous forward rates f(t).
    """
    # f(t) = d/dt[r(t)·t] = r(t) + t·r'(t)
    rt = zero_rates * times
    
    # Numerical derivative
    forwards = jnp.zeros_like(zero_rates)
    forwards = forwards.at[1:-1].set((rt[2:] - rt[:-2]) / (times[2:] - times[:-2]))
    forwards = forwards.at[0].set((rt[1] - rt[0]) / (times[1] - times[0]))
    forwards = forwards.at[-1].set((rt[-1] - rt[-2]) / (times[-1] - times[-2]))
    
    return forwards


# =============================================================================
# YIELD CURVE CLASSES
# =============================================================================

class YieldCurve(abc.ABC):
    """Abstract base class for yield curves."""
    
    @abc.abstractmethod
    def discount(self, T: float) -> float:
        """Discount factor D(T)."""
        pass
    
    @abc.abstractmethod
    def zero_rate(self, T: float) -> float:
        """Zero rate r(T)."""
        pass
    
    def forward_rate(self, t1: float, t2: float) -> float:
        """Forward rate f(t₁, t₂)."""
        D1 = self.discount(t1)
        D2 = self.discount(t2)
        return forward_rate(D1, D2, t1, t2)
    
    def forward_price(self, spot: float, T: float, dividend_yield: float = 0.0) -> float:
        """Forward price F(0, T) = S · exp((r - q)·T).
        
        Args:
            spot: Current spot price.
            T: Time to delivery.
            dividend_yield: Continuous dividend yield.
        """
        r = self.zero_rate(T)
        return spot * jnp.exp((r - dividend_yield) * T)


@dataclass
class FlatYieldCurve(YieldCurve):
    """Flat (constant) yield curve.
    
    Simplest case: r(T) = r for all T.
    """
    rate: float
    
    def discount(self, T: float) -> float:
        return float(jnp.exp(-self.rate * T))
    
    def zero_rate(self, T: float) -> float:
        return self.rate


@dataclass
class InterpolatedYieldCurve(YieldCurve):
    """Yield curve from interpolated zero rates.
    
    Supports linear and cubic spline interpolation.
    """
    times: Array
    zero_rates: Array
    method: str = 'linear'
    
    def __post_init__(self):
        self.times = jnp.asarray(self.times)
        self.zero_rates = jnp.asarray(self.zero_rates)
        
        # Ensure starts at 0
        if self.times[0] != 0:
            self.times = jnp.concatenate([jnp.array([0.0]), self.times])
            self.zero_rates = jnp.concatenate([self.zero_rates[:1], self.zero_rates])
    
    def discount(self, T: float) -> float:
        r = self.zero_rate(T)
        return float(jnp.exp(-r * T))
    
    def zero_rate(self, T: float) -> float:
        if T <= 0:
            return float(self.zero_rates[0])
        return float(jnp.interp(T, self.times, self.zero_rates))


# =============================================================================
# FORWARD PRICE CURVE
# =============================================================================

@dataclass
class ForwardPriceCurve:
    """Forward price curve for an asset.
    
    Stores either:
    - Direct forward prices at specific times, or
    - Spot + yield curve for computation
    
    Used by MartingaleSBSolver to compute forward ratios.
    """
    spot: float
    yield_curve: Optional[YieldCurve] = None
    dividend_yield: float = 0.0
    
    # Alternative: direct forward prices
    forward_times: Optional[Array] = None
    forward_prices: Optional[Array] = None
    
    def forward(self, T: float) -> float:
        """Compute forward price F(0, T).
        
        F(0, T) = S₀ · exp((r(T) - q) · T)
        """
        if self.forward_times is not None and self.forward_prices is not None:
            # Interpolate from stored forwards
            return float(jnp.interp(T, self.forward_times, self.forward_prices))
        
        if self.yield_curve is not None:
            return self.yield_curve.forward_price(self.spot, T, self.dividend_yield)
        
        # Default: no discounting
        return self.spot
    
    def forward_ratio(self, t1: float, t2: float) -> float:
        """Compute forward ratio F(0, t₂) / F(0, t₁).
        
        This is the expected growth factor from t₁ to t₂ under risk-neutral measure.
        Used for martingale constraints in SB.
        """
        F1 = self.forward(t1) if t1 > 0 else self.spot
        F2 = self.forward(t2)
        return F2 / F1
    
    @classmethod
    def from_rate(
        cls,
        spot: float,
        rate: float,
        dividend_yield: float = 0.0,
    ) -> 'ForwardPriceCurve':
        """Create from flat rate."""
        return cls(
            spot=spot,
            yield_curve=FlatYieldCurve(rate),
            dividend_yield=dividend_yield,
        )
    
    @classmethod
    def from_forwards(
        cls,
        spot: float,
        times: Array,
        forwards: Array,
    ) -> 'ForwardPriceCurve':
        """Create from observed forward prices."""
        return cls(
            spot=spot,
            forward_times=jnp.asarray(times),
            forward_prices=jnp.asarray(forwards),
        )


# =============================================================================
# BOOTSTRAPPING
# =============================================================================

def bootstrap_zero_curve(
    swap_rates: Array,
    swap_tenors: Array,
    frequency: int = 2,  # Semi-annual
) -> Tuple[Array, Array]:
    """Bootstrap zero curve from swap rates.
    
    Swap rate S_n satisfies: S_n · Σ D(t_i) · Δt = 1 - D(T_n)
    
    We solve iteratively for D(T_i), then convert to zero rates.
    
    Args:
        swap_rates: Par swap rates.
        swap_tenors: Corresponding tenors in years.
        frequency: Payment frequency per year.
        
    Returns:
        (times, zero_rates)
    """
    dt = 1.0 / frequency
    
    times = []
    discounts = []
    
    # D(0) = 1
    times.append(0.0)
    discounts.append(1.0)
    
    for i, (rate, tenor) in enumerate(zip(swap_rates, swap_tenors)):
        n_payments = int(tenor * frequency)
        payment_times = jnp.arange(1, n_payments + 1) * dt
        
        # Sum of known discounts
        sum_known = 0.0
        for t in payment_times[:-1]:
            if t in times:
                idx = times.index(t)
                sum_known += discounts[idx] * dt
            else:
                # Interpolate
                D_t = jnp.interp(t, jnp.array(times), jnp.array(discounts))
                sum_known += float(D_t) * dt
        
        # Solve for D(T): rate · (sum_known + D(T)·dt) = 1 - D(T)
        # rate · sum_known + rate · D(T) · dt = 1 - D(T)
        # D(T) · (1 + rate · dt) = 1 - rate · sum_known
        D_T = (1 - rate * sum_known) / (1 + rate * dt)
        
        times.append(tenor)
        discounts.append(D_T)
    
    times = jnp.array(times)
    discounts = jnp.array(discounts)
    
    # Convert to zero rates
    zero_rates = jnp.zeros_like(times)
    zero_rates = zero_rates.at[1:].set(-jnp.log(discounts[1:]) / times[1:])
    zero_rates = zero_rates.at[0].set(zero_rates[1])  # Extrapolate at 0
    
    return times, zero_rates


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Basic functions
    'discount_factor',
    'zero_rate',
    'forward_rate',
    'instantaneous_forward',
    # Yield curve classes
    'YieldCurve',
    'FlatYieldCurve',
    'InterpolatedYieldCurve',
    # Forward curve
    'ForwardPriceCurve',
    # Bootstrapping
    'bootstrap_zero_curve',
]
