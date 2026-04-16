"""Risk Measures: VaR, CVaR, and Portfolio Risk.

This module provides standard risk measures used in quantitative finance.

MAIN CONCEPTS
-
Value at Risk (VaR):
    VaR_alpha = quantile of loss distribution at level alpha
    "With probability 1-alpha, loss won't exceed VaR_alpha"

Expected Shortfall (ES) / Conditional VaR (CVaR):
    ES_alpha = E[Loss | Loss > VaR_alpha]
    "Average loss in the worst alpha cases"

MAIN MATH TAKEAWAY
-
VaR is just a quantile - it tells you the threshold but nothing about
what happens beyond that threshold. CVaR fixes this:

    CVaR_alpha = (1/alpha) integral_0^alpha VaR_u du

CVaR is also called Expected Shortfall (ES) and is the regulatory standard
(Basel III) because it's:
1. Coherent (subadditive): CVaR(A+B) <= CVaR(A) + CVaR(B)
2. Captures tail risk properly
3. More stable than VaR

WHY THIS MATTERS FOR SB
-
The Schrödinger Bridge gives you a distribution over future prices.
These risk measures help you:
1. Quantify uncertainty in the SB-generated forecasts
2. Set position limits based on tail risk
3. Compare risk under different reference processes

Author: Schrödinger Bridge Library
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

Array = jnp.ndarray
Scalar = Union[float, Array]


# VALUE AT RISK (VaR)
# -----------------------------------------------------------------------------
def value_at_risk(
    losses: Array,
    alpha: float = 0.05,
) -> float:
    """Compute Value at Risk from loss samples.
    
    VaR_alpha = alpha-quantile of loss distribution
    
    Interpretation: "With probability 1-alpha, loss won't exceed VaR_alpha"
    
    Args:
        losses: Array of loss values (positive = loss, negative = gain).
        alpha: Confidence level (default 5% = 95% VaR).
        
    Returns:
        VaR at level alpha.
        
    Example:
        >>> losses = portfolio_values[0] - portfolio_values[-1]  # P&L
        >>> var_95 = value_at_risk(losses, alpha=0.05)
        >>> print(f"95% VaR: ${var_95:.2f}")
    """
    return float(jnp.percentile(losses, 100 * (1 - alpha)))


def historical_var(
    returns: Array,
    portfolio_value: float,
    alpha: float = 0.05,
    horizon_days: int = 1,
) -> float:
    """Historical VaR from return series.
    
    Scales returns by portfolio value and time horizon.
    
    Args:
        returns: Daily return series (e.g., log returns).
        portfolio_value: Current portfolio value.
        alpha: Confidence level.
        horizon_days: Time horizon in days.
        
    Returns:
        VaR in currency units.
    """
    # Scale to horizon (assumes i.i.d. - crude approximation)
    scaled_returns = returns * jnp.sqrt(horizon_days)
    
    # Loss = negative return
    losses = -scaled_returns * portfolio_value
    
    return value_at_risk(losses, alpha)


def parametric_var(
    mu: float,
    sigma: float,
    portfolio_value: float,
    alpha: float = 0.05,
    horizon_days: int = 1,
) -> float:
    """Parametric (Gaussian) VaR.
    
    Assumes returns ~ N(mu, sigma^2).
    
    VaR_alpha = -portfolio * (mu*T - sigma*sqrtT * z_alpha)
    
    where z_alpha is the alpha-quantile of standard normal.
    
    Args:
        mu: Mean return (annualized).
        sigma: Volatility (annualized).
        portfolio_value: Current portfolio value.
        alpha: Confidence level.
        horizon_days: Time horizon.
        
    Returns:
        Parametric VaR.
    """
    T = horizon_days / 252  # Annualize
    z_alpha = float(norm.ppf(alpha))
    
    # VaR = worst case loss at alpha level
    var = -portfolio_value * (mu * T + sigma * jnp.sqrt(T) * z_alpha)
    
    return float(var)


# EXPECTED SHORTFALL (CVaR)
# -----------------------------------------------------------------------------
def expected_shortfall(
    losses: Array,
    alpha: float = 0.05,
) -> float:
    """Compute Expected Shortfall (CVaR) from loss samples.
    
    ES_alpha = E[Loss | Loss > VaR_alpha]
    
    This is the average loss in the worst alpha% of cases.
    
    Args:
        losses: Array of loss values.
        alpha: Confidence level (default 5% = 95% ES).
        
    Returns:
        Expected shortfall at level alpha.
        
    Example:
        >>> es_95 = expected_shortfall(losses, alpha=0.05)
        >>> print(f"95% ES: ${es_95:.2f}")
    """
    var = value_at_risk(losses, alpha)
    tail_losses = losses[losses >= var]
    
    if len(tail_losses) == 0:
        return var
    
    return float(jnp.mean(tail_losses))


# Alias
conditional_value_at_risk = expected_shortfall


def parametric_es(
    mu: float,
    sigma: float,
    portfolio_value: float,
    alpha: float = 0.05,
    horizon_days: int = 1,
) -> float:
    """Parametric (Gaussian) Expected Shortfall.
    
    For Gaussian: ES_alpha = -portfolio * (mu*T - sigma*sqrtT * phi(z_alpha)/alpha)
    
    where phi is standard normal PDF, z_alpha is alpha-quantile.
    
    Args:
        mu: Mean return (annualized).
        sigma: Volatility (annualized).
        portfolio_value: Current portfolio value.
        alpha: Confidence level.
        horizon_days: Time horizon.
        
    Returns:
        Parametric ES.
    """
    T = horizon_days / 252
    z_alpha = float(norm.ppf(alpha))
    phi_z = float(norm.pdf(z_alpha))
    
    es = -portfolio_value * (mu * T - sigma * jnp.sqrt(T) * phi_z / alpha)
    
    return float(es)


# GREEKS-BASED RISK
# -----------------------------------------------------------------------------
def delta_var(
    delta: float,
    spot: float,
    spot_vol: float,
    portfolio_value: float,
    alpha: float = 0.05,
    horizon_days: int = 1,
) -> float:
    """Delta-based VaR for option positions.
    
    Approximates P&L using first-order Greeks:
        dV ~= Δ * dS
    
    Then computes VaR of dS assuming lognormal spot.
    
    Args:
        delta: Portfolio delta.
        spot: Current spot price.
        spot_vol: Spot volatility.
        portfolio_value: Notional (for scaling).
        alpha: Confidence level.
        horizon_days: Time horizon.
        
    Returns:
        Delta-based VaR.
    """
    T = horizon_days / 252
    z_alpha = float(norm.ppf(alpha))
    
    # Worst-case spot move at alpha level (lognormal)
    spot_move = spot * (jnp.exp(spot_vol * jnp.sqrt(T) * z_alpha) - 1)
    
    # P&L from delta
    pnl = delta * spot_move
    
    return float(jnp.abs(pnl))


def delta_gamma_var(
    delta: float,
    gamma: float,
    spot: float,
    spot_vol: float,
    alpha: float = 0.05,
    horizon_days: int = 1,
) -> float:
    """Delta-Gamma VaR (second-order approximation).
    
    P&L approximation:
        dV ~= Δ*dS + 1/2Γ*(dS)^2
    
    For Gaussian dS, this gives a non-central chi-squared distribution.
    We use Monte Carlo for simplicity.
    
    Args:
        delta: Portfolio delta.
        gamma: Portfolio gamma.
        spot: Current spot.
        spot_vol: Spot volatility.
        alpha: Confidence level.
        horizon_days: Time horizon.
        
    Returns:
        Delta-gamma VaR.
    """
    T = horizon_days / 252
    sigma_S = spot * spot_vol * jnp.sqrt(T)
    
    # Monte Carlo
    key = jax.random.PRNGKey(42)
    dS = sigma_S * jax.random.normal(key, (10000,))
    
    # Delta-gamma P&L
    pnl = delta * dS + 0.5 * gamma * dS ** 2
    losses = -pnl  # Loss = negative P&L
    
    return value_at_risk(losses, alpha)


def portfolio_var(
    weights: Array,
    returns: Array,
    alpha: float = 0.05,
) -> float:
    """Portfolio VaR from historical returns.
    
    Args:
        weights: Portfolio weights, shape [n_assets].
        returns: Historical returns, shape [n_days, n_assets].
        alpha: Confidence level.
        
    Returns:
        Portfolio VaR.
    """
    # Portfolio returns
    portfolio_returns = returns @ weights
    
    # Portfolio loss
    losses = -portfolio_returns
    
    return value_at_risk(losses, alpha)


def portfolio_es(
    weights: Array,
    returns: Array,
    alpha: float = 0.05,
) -> float:
    """Portfolio Expected Shortfall from historical returns."""
    portfolio_returns = returns @ weights
    losses = -portfolio_returns
    return expected_shortfall(losses, alpha)


# TAIL STATISTICS
# -----------------------------------------------------------------------------
def tail_ratio(
    returns: Array,
    alpha: float = 0.05,
) -> float:
    """Compute tail ratio: ratio of upside to downside tail risk.
    
    Tail ratio = ES of gains / ES of losses
    
    > 1 means upside tail is fatter (good for long positions).
    < 1 means downside tail is fatter (bad for long positions).
    """
    gains = returns[returns > 0]
    losses = -returns[returns < 0]
    
    es_up = expected_shortfall(gains, alpha) if len(gains) > 0 else 0
    es_down = expected_shortfall(losses, alpha) if len(losses) > 0 else 1e-10
    
    return float(es_up / es_down)


def max_drawdown(prices: Array) -> Tuple[float, int, int]:
    """Compute maximum drawdown from price series.
    
    Max drawdown = max peak-to-trough decline.
    
    Returns:
        (max_drawdown, peak_idx, trough_idx)
    """
    # Running maximum
    running_max = jnp.maximum.accumulate(prices)
    
    # Drawdown at each point
    drawdowns = (running_max - prices) / running_max
    
    # Maximum drawdown
    max_dd_idx = int(jnp.argmax(drawdowns))
    max_dd = float(drawdowns[max_dd_idx])
    
    # Find corresponding peak
    peak_idx = int(jnp.argmax(prices[:max_dd_idx + 1]))
    
    return max_dd, peak_idx, max_dd_idx


def calmar_ratio(
    returns: Array,
    prices: Array,
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio: annualized return / max drawdown.
    
    Higher is better - measures return per unit of drawdown risk.
    """
    ann_return = float(jnp.mean(returns) * periods_per_year)
    max_dd, _, _ = max_drawdown(prices)
    
    return ann_return / (max_dd + 1e-10)


# RISK DECOMPOSITION
# -----------------------------------------------------------------------------
def component_var(
    weights: Array,
    cov_matrix: Array,
    portfolio_value: float,
    alpha: float = 0.05,
    horizon_days: int = 1,
) -> Dict[str, Array]:
    """Decompose VaR into component contributions.
    
    Component VaR tells you how much each position contributes to total VaR.
    
    Returns:
        Dict with 'total_var', 'marginal_var', 'component_var', 'percent_contribution'.
    """
    T = horizon_days / 252
    z_alpha = float(norm.ppf(1 - alpha))
    
    # Portfolio variance
    port_var = weights @ cov_matrix @ weights * T
    port_vol = jnp.sqrt(port_var)
    
    # Total VaR
    total_var = portfolio_value * port_vol * z_alpha
    
    # Marginal VaR: dVaR/dw_i
    marginal_var = portfolio_value * z_alpha * (cov_matrix @ weights * T) / (port_vol + 1e-10)
    
    # Component VaR: w_i × marginal VaR
    component_var = weights * marginal_var
    
    # Percent contribution
    pct_contribution = component_var / (total_var + 1e-10)
    
    return {
        'total_var': total_var,
        'marginal_var': marginal_var,
        'component_var': component_var,
        'percent_contribution': pct_contribution,
    }


# MODULE EXPORTS
# -----------------------------------------------------------------------------
__all__ = [
    # Basic VaR
    'value_at_risk',
    'historical_var',
    'parametric_var',
    # Expected Shortfall
    'expected_shortfall',
    'conditional_value_at_risk',
    'parametric_es',
    # Greeks-based
    'delta_var',
    'delta_gamma_var',
    # Portfolio
    'portfolio_var',
    'portfolio_es',
    # Tail statistics
    'tail_ratio',
    'max_drawdown',
    'calmar_ratio',
    # Decomposition
    'component_var',
]
