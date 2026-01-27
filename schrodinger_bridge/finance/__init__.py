"""Quantitative Finance Subpackage for Schrödinger Bridge Library.

This module provides tools for derivatives pricing, calibration, and risk
management using optimal transport and Schrödinger Bridge methods.

CORE PHILOSOPHY
===============
Standard quant models (Black-Scholes, Heston, etc.) specify a single probability
measure. The SB framework is fundamentally different:

    "What's the LEAST committed model consistent with observed option prices?"

This is powerful because:
1. You get model-FREE price bounds for exotics
2. The calibration is to MARGINALS, not parameters
3. Martingale constraint ensures no-arbitrage automatically

MAIN COMPONENTS
===============

1. OPTIONS PRICING (options.py)
   - Black-Scholes formula and Greeks
   - Implied volatility solver (Newton-Raphson)
   - Bachelier (normal) model

2. CALIBRATION (calibration.py)
   - Breeden-Litzenberger density extraction
   - Risk-neutral distribution from option prices
   - Surface interpolation

3. STOCHASTIC VOLATILITY DYNAMICS (dynamics.py)
   - Heston model
   - Local volatility (Dupire)
   - SABR model
   - Rough volatility

4. ROBUST HEDGING (robust_hedging.py)
   - Entropic Martingale Optimal Transport
   - Model-free price bounds
   - Hedge portfolio extraction

5. RISK MEASURES (risk.py)
   - Value at Risk (VaR)
   - Expected Shortfall (CVaR)
   - Greeks-based risk

6. TERM STRUCTURE (curves.py)
   - Discount factors
   - Forward rates
   - Yield curve interpolation

QUICK START
===========

```python
import jax.random as jr
from schrodinger_bridge.finance import (
    # Option pricing
    black_scholes_call,
    implied_volatility,
    # Calibration
    breeden_litzenberger_density,
    extract_risk_neutral_distribution,
    # Dynamics for SB reference
    HestonDynamics,
    LocalVolatilityDynamics,
    # Robust pricing
    EntropicMOTSolver,
    forward_start_call_payoff,
    # Risk
    value_at_risk,
    expected_shortfall,
)

# Extract risk-neutral density from options
density = breeden_litzenberger_density(strikes, call_prices, forward, discount)

# Compute model-free bounds on exotic
solver = EntropicMOTSolver(epsilon=0.1)
result = solver.solve(samples_T1, samples_T2, forward_start_call_payoff(1.1))
print(f"Price in [{result.lower_bound:.4f}, {result.upper_bound:.4f}]")
```

Author: Schrödinger Bridge Library
"""

# =============================================================================
# OPTIONS PRICING
# =============================================================================
from .options import (
    # Black-Scholes
    black_scholes_call,
    black_scholes_put,
    black_scholes_price,
    # Greeks
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_rho,
    compute_all_greeks,
    # Implied volatility
    implied_volatility,
    implied_volatility_newton,
    # Bachelier (normal model)
    bachelier_call,
    bachelier_put,
    bachelier_implied_vol,
    # Utilities
    put_call_parity_call,
    put_call_parity_put,
)

# =============================================================================
# CALIBRATION
# =============================================================================
from .calibration import (
    # Breeden-Litzenberger
    breeden_litzenberger_density,
    extract_risk_neutral_distribution,
    # Risk-neutral sampling
    RiskNeutralDistribution,
    sample_from_density,
    # Surface tools
    interpolate_vol_surface,
    total_variance_surface,
)

# =============================================================================
# STOCHASTIC VOLATILITY DYNAMICS
# =============================================================================
from .dynamics import (
    # Models
    LocalVolatilityDynamics,
    HestonDynamics,
    SABRDynamics,
    RoughVolatilityDynamics,
    # Utilities
    create_vol_surface_from_sabr,
)

# =============================================================================
# ROBUST HEDGING
# =============================================================================
from .robust_hedging import (
    # Core classes
    EntropicMOTSolver,
    DualPotentials,
    RobustHedgingResult,
    # Payoff functions
    european_call_payoff,
    european_put_payoff,
    forward_start_call_payoff,
    variance_swap_payoff,
    lookback_proxy_payoff,
    barrier_down_out_call_payoff,
    # Convenience
    compute_robust_price_bounds,
)

# =============================================================================
# RISK MEASURES
# =============================================================================
from .risk import (
    # VaR and CVaR
    value_at_risk,
    expected_shortfall,
    conditional_value_at_risk,
    # Distribution-based
    parametric_var,
    historical_var,
    # Greeks risk
    delta_var,
    portfolio_var,
)

# =============================================================================
# TERM STRUCTURE
# =============================================================================
from .curves import (
    # Discount factors
    discount_factor,
    forward_rate,
    zero_rate,
    # Yield curve
    YieldCurve,
    FlatYieldCurve,
    InterpolatedYieldCurve,
    # Forward curve
    ForwardPriceCurve,
)

# =============================================================================
# MODULE METADATA
# =============================================================================

__all__ = [
    # Options
    'black_scholes_call',
    'black_scholes_put',
    'black_scholes_price',
    'bs_delta',
    'bs_gamma',
    'bs_vega',
    'bs_theta',
    'bs_rho',
    'compute_all_greeks',
    'implied_volatility',
    'implied_volatility_newton',
    'bachelier_call',
    'bachelier_put',
    'bachelier_implied_vol',
    'put_call_parity_call',
    'put_call_parity_put',
    # Calibration
    'breeden_litzenberger_density',
    'extract_risk_neutral_distribution',
    'RiskNeutralDistribution',
    'sample_from_density',
    'interpolate_vol_surface',
    'total_variance_surface',
    # Dynamics
    'LocalVolatilityDynamics',
    'HestonDynamics',
    'SABRDynamics',
    'RoughVolatilityDynamics',
    'create_vol_surface_from_sabr',
    # Robust hedging
    'EntropicMOTSolver',
    'DualPotentials',
    'RobustHedgingResult',
    'european_call_payoff',
    'european_put_payoff',
    'forward_start_call_payoff',
    'variance_swap_payoff',
    'lookback_proxy_payoff',
    'barrier_down_out_call_payoff',
    'compute_robust_price_bounds',
    # Risk
    'value_at_risk',
    'expected_shortfall',
    'conditional_value_at_risk',
    'parametric_var',
    'historical_var',
    'delta_var',
    'portfolio_var',
    # Curves
    'discount_factor',
    'forward_rate',
    'zero_rate',
    'YieldCurve',
    'FlatYieldCurve',
    'InterpolatedYieldCurve',
    'ForwardPriceCurve',
]
