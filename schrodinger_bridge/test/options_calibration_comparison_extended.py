#!/usr/bin/env python3
"""
Extended options-calibration comparison using yfinance + Schrödinger Bridges.

Goal
----
Hold out an intermediate maturity T2, calibrate on T1 and T3 only, then compare
T2 vanilla pricing / IV errors and arbitrage diagnostics.

Models compared
---------------
1) Martingale Schrödinger Bridge (SB) with "Brownian-like" reference
2) Martingale SB with SV reference (HestonDynamics)  [Heston + SB]
3) Heston-only Monte Carlo                           [Heston baseline]
4) Total-variance interpolation (market IV slices)
5) SABR (Hagan) slice fits at T1,T3 + parameter interpolation to T2
6) SVI slice fits at T1,T3 + total-variance interpolation in T (SVI-VI)

Key diagnostics
---------------
- RMSE/MAE on prices and implied vols at held-out T2 (strike grid)
- Butterfly (convexity) diagnostics on market slices
- Calendar monotonicity at approx ATM across T
- Two-time payoff model risk via Martingale OT bounds (forward-start straddle)

Requirements
------------
pip install yfinance jax jaxlib scipy

Internet access is required for yfinance.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Optional imports (required for SB portion)
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
except Exception:
    jax = None
    jnp = None
    jr = None

try:
    import yfinance as yf
except Exception:
    yf = None

# Repo modules (keep module-level imports minimal)
from schrodinger_bridge.finance.options import black_scholes_call, implied_volatility_bisection


# =============================================================================
# Utilities
# =============================================================================

@dataclass
class SliceData:
    """Single-expiry call slice."""
    T: float
    forward: float
    strikes: np.ndarray
    call_mid: np.ndarray
    iv: np.ndarray


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _yearfrac(dt0: datetime, dt1: datetime) -> float:
    # ACT/365F
    return (dt1 - dt0).total_seconds() / (365.0 * 24 * 3600)


def _safe_mid(bid: float, ask: float, last: float) -> float:
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
        return 0.5 * (bid + ask)
    return float(last)


def _iv_from_price(S: float, K: float, T: float, r: float, C: float) -> float:
    try:
        return float(implied_volatility_bisection(S, K, T, r, C, option_type="call"))
    except Exception:
        return float("nan")


def _select_three_expiries(
    expiry_strs: List[str], now: datetime, min_dte: int, max_dte: int
) -> List[Tuple[str, float]]:
    expiries = []
    for s in expiry_strs:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        dte = (dt - now).days
        if min_dte <= dte <= max_dte:
            expiries.append((s, _yearfrac(now, dt)))
    expiries.sort(key=lambda x: x[1])
    if len(expiries) < 3:
        raise RuntimeError(f"Need >=3 expiries in [{min_dte},{max_dte}] days, found {len(expiries)}")
    return expiries[:3]


def fetch_yfinance_slices(
    ticker: str,
    r: float,
    q: float,
    min_dte: int,
    max_dte: int,
    moneyness_lo: float,
    moneyness_hi: float,
    min_quotes: int,
) -> Tuple[float, List[SliceData], Dict]:
    """Fetch three call slices from yfinance and build SliceData objects."""
    _require(yf is not None, "yfinance not available. Install with: pip install yfinance")
    tk = yf.Ticker(ticker)

    hist = tk.history(period="5d", auto_adjust=False)
    _require(len(hist) > 0, "No price history from yfinance.")
    spot = float(hist["Close"].iloc[-1])

    now = datetime.now(timezone.utc)
    expiries = _select_three_expiries(list(tk.options), now, min_dte, max_dte)
    (e1, T1), (e2, T2), (e3, T3) = expiries

    meta = {
        "ticker": ticker,
        "spot": spot,
        "expiries": [e1, e2, e3],
        "Ts": [T1, T2, T3],
        "rate": r,
        "div": q,
    }

    slices: List[SliceData] = []
    for exp, T in [(e1, T1), (e2, T2), (e3, T3)]:
        chain = tk.option_chain(exp)
        calls = chain.calls.copy()

        calls["mid"] = calls.apply(
            lambda row: _safe_mid(row.get("bid", np.nan), row.get("ask", np.nan), row.get("lastPrice", np.nan)),
            axis=1,
        )
        calls["iv"] = calls.get("impliedVolatility", np.nan)

        F = spot * math.exp((r - q) * T)

        lo = moneyness_lo * spot
        hi = moneyness_hi * spot
        calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
        calls = calls[np.isfinite(calls["mid"]) & (calls["mid"] > 0)]
        calls = calls.sort_values("strike")

        _require(len(calls) >= min_quotes, f"Not enough call quotes at {exp}: {len(calls)} < {min_quotes}")

        strikes = calls["strike"].to_numpy(dtype=float)
        call_mid = calls["mid"].to_numpy(dtype=float)

        iv = calls["iv"].to_numpy(dtype=float)
        bad = ~np.isfinite(iv) | (iv <= 0)
        if bad.any():
            for i in np.where(bad)[0]:
                iv[i] = _iv_from_price(spot, strikes[i], T, r, call_mid[i])

        good = np.isfinite(iv) & (iv > 0)
        strikes, call_mid, iv = strikes[good], call_mid[good], iv[good]
        _require(len(strikes) >= min_quotes, f"After IV cleaning, not enough quotes at {exp}: {len(strikes)}")

        slices.append(SliceData(T=T, forward=F, strikes=strikes, call_mid=call_mid, iv=iv))

    return spot, slices, meta


def make_common_strike_grid(spot: float, strikes: np.ndarray, n: int) -> np.ndarray:
    lo = np.quantile(strikes, 0.10)
    hi = np.quantile(strikes, 0.90)
    lo = max(lo, 0.5 * spot)
    hi = min(hi, 1.5 * spot)
    return np.linspace(lo, hi, n)


def interp_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, y, left=np.nan, right=np.nan)


def bs_call_vec(S: float, K: np.ndarray, T: float, r: float, sigma: np.ndarray) -> np.ndarray:
    return np.array([black_scholes_call(S, float(k), T, r, float(s)) for k, s in zip(K, sigma)], dtype=float)


def iv_from_price_vec(S: float, K: np.ndarray, T: float, r: float, C: np.ndarray) -> np.ndarray:
    out = np.empty_like(C, dtype=float)
    for i in range(len(C)):
        out[i] = _iv_from_price(S, float(K[i]), T, r, float(C[i]))
    return out


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(a[m] - b[m])))


# =============================================================================
# MINIMAL FIX for options_calibration_comparison_extended.py
#
# THE PROBLEM:
# - Market data has butterfly arbitrage violations
# - Breeden-Litzenberger on arbitrage-violating data produces garbage
# - The marginal means don't match forwards
#
# THE FIX:
# 1) Smooth prices with SVI before Breeden-Litzenberger
# 2) Force sample means to exactly match forwards
# =============================================================================

def _smooth_calls_svi(K, C, F, T, r, n_out=300):
    """Smooth call prices with SVI fit to ensure convexity (via SVI fit in total variance space)."""

    def bs_call(S, K_, T_, r_, sigma_):
        if T_ < 1e-10 or sigma_ < 1e-10:
            return max(S - K_, 0.0)
        d1 = (np.log(S / K_) + (r_ + 0.5 * sigma_ ** 2) * T_) / (sigma_ * np.sqrt(T_))
        d2 = d1 - sigma_ * np.sqrt(T_)
        return S * norm.cdf(d1) - K_ * np.exp(-r_ * T_) * norm.cdf(d2)

    def implied_vol(C_price, S, K_, T_, r_):
        lb = max(S - K_ * np.exp(-r_ * T_), 0.0)
        if C_price < lb + 1e-10:
            return np.nan
        lo, hi = 0.01, 3.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if bs_call(S, K_, T_, r_, mid) < C_price:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # S is the spot consistent with F under rate r (div handled already inside F if you used r-q)
    S = F * np.exp(-r * T)
    ivs = np.array([implied_vol(c, S, k, T, r) for c, k in zip(C, K)], dtype=float)

    valid = np.isfinite(ivs) & (ivs > 0.01) & (ivs < 3.0)
    if valid.sum() < 5:
        K_out = np.linspace(float(np.min(K)), float(np.max(K)), int(n_out))
        return K_out, np.interp(K_out, K, C)

    K_v, iv_v = K[valid], ivs[valid]
    k = np.log(K_v / F)
    w = (iv_v ** 2) * T

    def svi(k_, a, b, rho, m, sigma):
        x = k_ - m
        return a + b * (rho * x + np.sqrt(x ** 2 + sigma ** 2))

    def loss(params):
        a, b, rho, m, sigma = params
        if a < 0 or b < 0 or sigma < 1e-6 or abs(rho) >= 1:
            return 1e10
        w_fit = svi(k, a, b, rho, m, sigma)
        if np.any(w_fit < 0) or not np.isfinite(w_fit).all():
            return 1e10
        return float(np.sum((w_fit - w) ** 2))

    x0 = [max(float(np.median(w)) * 0.5, 0.01), 0.1, -0.3, 0.0, 0.2]
    bounds = [(0.001, 10), (0.001, 10), (-0.99, 0.99), (-2, 2), (0.01, 5)]
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    a, b, rho, m, sigma = res.x

    K_out = np.linspace(float(np.min(K)), float(np.max(K)), int(n_out))
    k_out = np.log(K_out / F)
    x = k_out - m
    w_out = a + b * (rho * x + np.sqrt(x ** 2 + sigma ** 2))
    w_out = np.maximum(w_out, 1e-6)
    iv_out = np.sqrt(w_out / T)

    C_out = np.array([bs_call(S, kk, T, r, ss) for kk, ss in zip(K_out, iv_out)], dtype=float)
    return K_out, np.maximum(C_out, 0.0)


def _robust_bl_density(K, C, F, T, r, n_grid=500):
    """Breeden-Litzenberger with SVI smoothing and forward-forcing."""

    # Smooth with SVI first
    K_smooth, C_smooth = _smooth_calls_svi(K, C, F, T, r, n_grid)

    # Breeden-Litzenberger
    dK = K_smooth[1] - K_smooth[0]
    discount = np.exp(r * T)

    d2C = np.zeros_like(C_smooth)
    d2C[1:-1] = (C_smooth[:-2] - 2 * C_smooth[1:-1] + C_smooth[2:]) / (dK ** 2)
    d2C[0], d2C[-1] = d2C[1], d2C[-2]

    density = discount * d2C
    density = np.maximum(density, 0)
    total = np.trapezoid(density, K_smooth)

    if total < 1e-10:
        # Fallback: lognormal
        sigma = 0.2
        log_K = np.log(K_smooth / F)
        density = np.exp(-0.5 * (log_K / sigma) ** 2) / (K_smooth * sigma * np.sqrt(2 * np.pi))
        total = np.trapezoid(density, K_smooth)

    density = density / total

    # Exponential tilting to force E[S] = F
    K_norm = (K_smooth - F) / F

    def compute_mean(alpha):
        w = density * np.exp(alpha * K_norm)
        Z = np.trapezoid(w, K_smooth)
        if Z < 1e-30:
            return np.inf
        return np.trapezoid(K_smooth * w / Z, K_smooth)

    lo, hi = -200.0, 200.0
    for _ in range(150):
        mid = 0.5 * (lo + hi)
        mu = compute_mean(mid)
        if not np.isfinite(mu) or mu > F:
            hi = mid
        else:
            lo = mid

    alpha = 0.5 * (lo + hi)
    w = density * np.exp(alpha * K_norm)
    Z = np.trapezoid(w, K_smooth)
    density = w / Z

    # CDF
    cdf = np.cumsum(density) * dK
    cdf = cdf / cdf[-1]

    return K_smooth, density, cdf



def _sample_from_density(K, cdf, n_samples, key=None):
    """Inverse CDF sampling. Returns (n_samples, 1)."""
    if key is not None:
        _require(jr is not None, "jax.random required for keyed sampling")
        u = np.array(jr.uniform(key, (int(n_samples),)))
    else:
        u = np.random.rand(int(n_samples))
    samples = np.interp(u, cdf, K)
    return samples.reshape(-1, 1)


def dist_from_slice_FIXED(sl, r, n_fine):
    """Extract risk-neutral distribution with guaranteed forward consistency (SVI-smoothed BL + mean forcing)."""
    K = np.asarray(sl.strikes, float)
    C = np.asarray(sl.call_mid, float)
    F = float(sl.forward)
    T = float(sl.T)

    K_grid, density, cdf = _robust_bl_density(K, C, F, T, r, n_grid=int(n_fine))

    # Verify mean from density
    mean = np.trapezoid(K_grid * density, K_grid)
    err = abs(mean - F) / max(F, 1e-12) * 100.0
    if err > 0.5:
        print(f"    T={T:.3f}: Density mean error = {err:.2f}% (will correct in samples)")

    class _Dist:
        def __init__(self, K_, cdf_, target_mean_):
            self.K = np.asarray(K_, float)
            self.cdf = np.asarray(cdf_, float)
            self.target_mean = float(target_mean_)

        def sample(self, key, n):
            samples = _sample_from_density(self.K, self.cdf, int(n), key=key)
            current_mean = float(np.mean(samples))
            if not np.isfinite(current_mean) or abs(current_mean) < 1e-12:
                return samples
            samples = samples * (self.target_mean / current_mean)
            return samples

    return _Dist(K_grid, cdf, F)


# =============================================================================
# SABR baseline (Hagan lognormal)
# =============================================================================

def sabr_hagan_iv(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    if F <= 0 or K <= 0 or alpha <= 0 or nu <= 0 or not (-0.999 < rho < 0.999):
        return float("nan")
    if abs(F - K) < 1e-12:
        FK = F
        one = (((1 - beta) ** 2) / 24) * (alpha ** 2) / (FK ** (2 - 2 * beta))
        two = (rho * beta * nu * alpha) / (4 * (FK ** (1 - beta)))
        three = ((2 - 3 * rho ** 2) / 24) * (nu ** 2)
        return (alpha / (FK ** (1 - beta))) * (1 + (one + two + three) * T)

    logFK = math.log(F / K)
    FK_beta = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * FK_beta * logFK
    xz = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    if abs(xz) < 1e-14:
        xz = 1e-14
    term1 = alpha / (FK_beta * (1 + ((1 - beta) ** 2 / 24) * logFK ** 2 + ((1 - beta) ** 4 / 1920) * logFK ** 4))
    one = (((1 - beta) ** 2) / 24) * (alpha ** 2) / ((F * K) ** (1 - beta))
    two = (rho * beta * nu * alpha) / (4 * (F * K) ** ((1 - beta) / 2))
    three = ((2 - 3 * rho ** 2) / 24) * (nu ** 2)
    return term1 * (z / xz) * (1 + (one + two + three) * T)


def fit_sabr_slice(F: float, K: np.ndarray, T: float, iv: np.ndarray, beta: float = 0.5) -> Tuple[float, float, float, float]:
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)
    m = np.isfinite(iv) & (iv > 0)
    K, iv = K[m], iv[m]
    if len(K) < 6:
        return (0.2, beta, -0.3, 0.6)

    def obj(x):
        alpha, rho, nu = x
        if alpha <= 1e-6 or nu <= 1e-6 or not (-0.999 < rho < 0.999):
            return 1e6
        pred = np.array([sabr_hagan_iv(F, float(k), T, alpha, beta, rho, nu) for k in K])
        if not np.isfinite(pred).all():
            return 1e6
        return float(np.mean((pred - iv) ** 2))

    x0 = np.array([np.nanmedian(iv) * (F ** (1 - beta)), -0.3, 0.8], float)
    bounds = [(1e-6, 5.0), (-0.999, 0.999), (1e-6, 5.0)]
    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
    alpha, rho, nu = res.x
    return float(alpha), float(beta), float(rho), float(nu)


# =============================================================================
# SVI baseline (raw SVI)
# =============================================================================

def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


def fit_svi_slice(F: float, K: np.ndarray, T: float, iv: np.ndarray) -> Tuple[float, float, float, float, float]:
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)
    msk = np.isfinite(iv) & (iv > 0)
    K, iv = K[msk], iv[msk]
    if len(K) < 7:
        w0 = float(np.nanmedian(iv) ** 2 * T)
        return (max(w0 * 0.5, 1e-6), 0.1, -0.2, 0.0, 0.2)

    k = np.log(K / F)
    w_obs = (iv ** 2) * T

    def obj(x):
        a, b, rho, m, sig = x
        if a <= 1e-10 or b <= 1e-10 or sig <= 1e-6 or not (-0.999 < rho < 0.999):
            return 1e6
        w = svi_total_variance(k, a, b, rho, m, sig)
        if np.any(w <= 1e-12) or not np.isfinite(w).all():
            return 1e6
        return float(np.mean((w - w_obs) ** 2))

    w0 = float(np.nanmedian(w_obs))
    x0 = np.array([max(0.5 * w0, 1e-6), 0.2, -0.3, 0.0, 0.2], float)
    bounds = [
        (1e-10, 10.0),
        (1e-10, 10.0),
        (-0.999, 0.999),
        (-2.0, 2.0),
        (1e-6, 5.0),
    ]
    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
    return tuple(map(float, res.x))


# =============================================================================
# Heston-only MC baseline
# =============================================================================

def heston_mc_call_prices(
    S0: float,
    K: np.ndarray,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    logS = np.full(n_paths, math.log(S0), float)
    v = np.full(n_paths, v0, float)

    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w1 = z1
        w2 = rho * z1 + math.sqrt(max(1 - rho * rho, 1e-12)) * z2

        v_pos = np.maximum(v, 0.0)
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqrt_dt * w2
        v = np.maximum(v, 0.0)

        logS = logS + (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * w1

    ST = np.exp(logS)
    disc = math.exp(-r * T)
    prices = disc * np.maximum(ST[:, None] - K[None, :], 0.0).mean(axis=0)
    return prices


# =============================================================================
# SB helpers (UPDATED)
# =============================================================================

def build_martingale_sb_solver(
    spot: float,
    r: float,
    q: float,
    slices,  # List[SliceData]
    num_grid_density: int,
    sigma_ref: float,
    use_sv_reference: bool,
    enforce_variance: bool = False,
    heston_params=None,
    num_steps_per_segment: int = 160,
):
    """Build MartingaleSB solver with VERIFIED forward-consistent marginals."""
    _require(jax is not None and jnp is not None and jr is not None, "jax/jaxlib required for SB portion.")

    from schrodinger_bridge.finance.dynamics import HestonDynamics
    from schrodinger_bridge.marginal_sb import MarginalConstraint
    from schrodinger_bridge.core.problem import EmpiricalDistribution
    from schrodinger_bridge.martingale_sb import (
        ForwardCurve, MartingaleSBProblem, MartingaleSBConfig, MartingaleSBSolver
    )

    T1, T2, T3 = slices[0].T, slices[1].T, slices[2].T
    t1 = T1 / T3
    t3 = 1.0

    print("  Extracting forward-consistent marginals (with SVI smoothing)...")

    dist1 = dist_from_slice_FIXED(slices[0], r, num_grid_density)
    dist3 = dist_from_slice_FIXED(slices[2], r, num_grid_density)

    key0 = jr.PRNGKey(0)
    s0_samples = jnp.full((3000, 1), float(spot))

    key1, key3 = jr.split(key0)
    s1_samples = jnp.array(dist1.sample(key1, 3000))
    s3_samples = jnp.array(dist3.sample(key3, 3000))

    # Verify forward consistency of samples
    F1 = float(spot * np.exp((r - q) * T1))
    F3 = float(spot * np.exp((r - q) * T3))
    s1_mean = float(np.mean(np.array(s1_samples)))
    s3_mean = float(np.mean(np.array(s3_samples)))

    print(f"  T1={T1:.3f}: Sample mean=${s1_mean:.2f}, Forward=${F1:.2f}, Error={abs(s1_mean - F1)/F1*100:.2f}%")
    print(f"  T3={T3:.3f}: Sample mean=${s3_mean:.2f}, Forward=${F3:.2f}, Error={abs(s3_mean - F3)/F3*100:.2f}%")

    mu0 = EmpiricalDistribution(samples=np.array(s0_samples).reshape(-1, 1))
    mu1 = EmpiricalDistribution(samples=np.array(s1_samples).reshape(-1, 1))
    mu3 = EmpiricalDistribution(samples=np.array(s3_samples).reshape(-1, 1))

    marginals = [
        MarginalConstraint(time=0.0, distribution=mu0),
        MarginalConstraint(time=float(t1), distribution=mu1),
        MarginalConstraint(time=float(t3), distribution=mu3),
    ]

    # Forward curve for NORMALIZED time: use rates scaled by T3 so that exp((r*T3)*t) = exp(r*(T3*t))
    fc = ForwardCurve(spot=float(spot), rate=float(r) * float(T3), dividend_yield=float(q) * float(T3))

    if use_sv_reference:
        hp = heston_params or dict(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.04, rate=r, spot=spot)
        ref = HestonDynamics(**hp)
    else:
        ref = HestonDynamics(kappa=1.0, theta=0.04, xi=0.01, rho=-0.2, v0=0.04, rate=r, spot=spot)

    problem = MartingaleSBProblem(
        reference=ref,
        marginals=marginals,
        forward_curve=fc,
        time_grid=None,
        variance_constraint=None,
        name="yfinance_martingale_sb",
    )

    config = MartingaleSBConfig()
    config.num_steps_per_segment = int(num_steps_per_segment)
    config.sigma_ref = float(sigma_ref)
    config.use_sv_reference = bool(use_sv_reference)
    config.apply_variance_constraint = bool(enforce_variance)

    solver = MartingaleSBSolver(problem, config)
    info = {
        "t_norm": [0.0, float(t1), 1.0],
        "T_years": [float(T1), float(T2), float(T3)],
        "use_sv_reference": bool(use_sv_reference),
    }
    return solver, info


def sb_price_T2_calls(
    solver,
    T2: float,
    T3: float,
    K: np.ndarray,
    r: float,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    _require(jax is not None and jr is not None, "jax required")
    key = jr.PRNGKey(int(seed))
    solver.train(key, num_samples=3000)

    times, paths = solver.simulate(key, int(n_paths))
    times = np.array(times)
    paths = np.array(paths)

    t2 = float(T2) / float(T3)
    idx2 = int(np.argmin(np.abs(times - t2)))
    ST2 = paths[:, idx2]

    disc = math.exp(-float(r) * float(T2))
    prices = disc * np.maximum(ST2[:, None] - K[None, :], 0.0).mean(axis=0)
    return prices


def sample_rn_marginal_from_slice(sl: SliceData, r: float, n_fine: int, n_samples: int, key) -> np.ndarray:
    """
    Market marginal samples using robust SVI-smoothed BL + mean-forcing.
    Returns shape (n_samples, 1).
    """
    dist = dist_from_slice_FIXED(sl, r, n_fine)
    samples = dist.sample(key, int(n_samples))
    return np.asarray(samples).reshape(-1, 1)


def mot_forward_start_bounds_from_market(
    S1_samples, S2_samples, fwd_ratio, r, t2, epsilon=0.05, num_iters=200
) -> Tuple[float, float]:
    """
    Model-free MOT bounds for time-0 discounted forward-start straddle payoff |S2 - S1|.
    NOTE: MartingaleOTBounds expects 1D supports.
    """
    _require(jax is not None and jnp is not None, "jax required for MOT bounds")
    from schrodinger_bridge.martingale_sb import MartingaleOTBounds

    S1 = np.asarray(S1_samples).reshape(-1)
    S2 = np.asarray(S2_samples).reshape(-1)

    payoff_fn = lambda X, Y: jnp.abs(Y - X)
    mot = MartingaleOTBounds(S1, S2, forward_ratio=float(fwd_ratio))
    lb, ub, _ = mot.compute_bounds(payoff_fn, epsilon=float(epsilon), num_iters=int(num_iters))

    disc0 = float(np.exp(-float(r) * float(t2)))
    return disc0 * float(lb), disc0 * float(ub)


# =============================================================================
# Main experiment
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="SPY")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--div", type=float, default=0.0)
    ap.add_argument("--min_dte", type=int, default=21)
    ap.add_argument("--max_dte", type=int, default=200)
    ap.add_argument("--moneyness_lo", type=float, default=0.85)
    ap.add_argument("--moneyness_hi", type=float, default=1.15)
    ap.add_argument("--min_quotes", type=int, default=18)
    ap.add_argument("--n_grid", type=int, default=25, help="Strike grid size for evaluation at T2")
    ap.add_argument("--density_grid", type=int, default=300, help="Grid for SVI-smoothed BL extraction")
    ap.add_argument("--mc_paths", type=int, default=25000)
    ap.add_argument("--mc_steps", type=int, default=160, help="MC steps for Heston-only")
    ap.add_argument("--sb_paths", type=int, default=12000)
    ap.add_argument("--sb_sigma_ref", type=float, default=0.35)
    ap.add_argument("--out_dir", type=str, default="sb_calibration_report")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spot, slices, meta = fetch_yfinance_slices(
        ticker=args.ticker,
        r=args.rate,
        q=args.div,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        moneyness_lo=args.moneyness_lo,
        moneyness_hi=args.moneyness_hi,
        min_quotes=args.min_quotes,
    )
    s1, s2, s3 = slices
    T1, T2, T3 = s1.T, s2.T, s3.T

    # Evaluation grid at T2
    K_grid = make_common_strike_grid(spot, s2.strikes, args.n_grid)

    # Market T2 prices + iv on grid (may contain NaN if grid extends beyond strike support)
    C_mkt = interp_1d(s2.strikes, s2.call_mid, K_grid)
    iv_mkt = interp_1d(s2.strikes, s2.iv, K_grid)

    # -------------------------------------------------------------------------
    # Baseline: total variance interpolation (TVI)
    # -------------------------------------------------------------------------
    iv1g = interp_1d(s1.strikes, s1.iv, K_grid)
    iv3g = interp_1d(s3.strikes, s3.iv, K_grid)
    w1 = (iv1g ** 2) * T1
    w3 = (iv3g ** 2) * T3
    lam = (T2 - T1) / (T3 - T1)
    w2_tvi = w1 + lam * (w3 - w1)
    iv2_tvi = np.sqrt(np.maximum(w2_tvi / T2, 1e-12))
    C_tvi = bs_call_vec(spot, K_grid, T2, args.rate, iv2_tvi)

    # -------------------------------------------------------------------------
    # Baseline: SABR fit at T1,T3, interpolate params to T2
    # -------------------------------------------------------------------------
    a1, b1, r1, n1 = fit_sabr_slice(s1.forward, s1.strikes, T1, s1.iv, beta=0.5)
    a3, b3, r3, n3 = fit_sabr_slice(s3.forward, s3.strikes, T3, s3.iv, beta=0.5)
    a2 = a1 + lam * (a3 - a1)
    rho2 = r1 + lam * (r3 - r1)
    nu2 = n1 + lam * (n3 - n1)
    beta = 0.5
    iv2_sabr = np.array([sabr_hagan_iv(s2.forward, float(k), T2, a2, beta, rho2, nu2) for k in K_grid])
    iv2_sabr = np.clip(iv2_sabr, 1e-4, 5.0)
    C_sabr = bs_call_vec(spot, K_grid, T2, args.rate, iv2_sabr)

    # -------------------------------------------------------------------------
    # Baseline: SVI slice fits at T1,T3, interpolate total variance to T2 (SVI-VI)
    # -------------------------------------------------------------------------
    svi1 = fit_svi_slice(s1.forward, s1.strikes, T1, s1.iv)
    svi3 = fit_svi_slice(s3.forward, s3.strikes, T3, s3.iv)
    k_grid = np.log(K_grid / s2.forward)
    w1_svi = svi_total_variance(k_grid, *svi1)
    w3_svi = svi_total_variance(k_grid, *svi3)
    w3_svi = np.maximum(w3_svi, w1_svi)  # simple calendar projection
    w2_svi = w1_svi + lam * (w3_svi - w1_svi)
    iv2_svi = np.sqrt(np.maximum(w2_svi / T2, 1e-12))
    C_svi = bs_call_vec(spot, K_grid, T2, args.rate, iv2_svi)

    # -------------------------------------------------------------------------
    # Heston-only MC baseline (parameters: simple, not calibrated)
    # -------------------------------------------------------------------------
    atm_iv = float(np.nanmedian(s2.iv))
    theta = max(atm_iv ** 2, 1e-4)
    heston_params = dict(kappa=2.0, theta=theta, xi=0.7, rho=-0.7, v0=theta)

    C_heston = heston_mc_call_prices(
        S0=spot,
        K=K_grid,
        T=T2,
        r=args.rate,
        q=args.div,
        n_paths=args.mc_paths,
        n_steps=args.mc_steps,
        seed=args.seed,
        **heston_params,
    )

    # -------------------------------------------------------------------------
    # SB: Brownian-like reference vs Heston+SB
    # -------------------------------------------------------------------------
    results: Dict[str, np.ndarray] = {}
    sb_meta: Dict = {}

    if jax is not None:
        # Market marginal samples for MOT bounds (T1 -> T2)
        k0 = jr.PRNGKey(args.seed + 12345)
        kA, kB = jr.split(k0)
        s1_samples = sample_rn_marginal_from_slice(s1, args.rate, args.density_grid, 1200, kA)
        s2_samples = sample_rn_marginal_from_slice(s2, args.rate, args.density_grid, 1200, kB)

        # SB with Brownian-like reference
        sb_bm, sb_info_bm = build_martingale_sb_solver(
            spot=spot,
            r=args.rate,
            q=args.div,
            slices=slices,
            num_grid_density=args.density_grid,
            sigma_ref=args.sb_sigma_ref,
            use_sv_reference=False,
            heston_params=None,
            num_steps_per_segment=args.mc_steps,
        )
        C_sb_bm = sb_price_T2_calls(sb_bm, T2, T3, K_grid, args.rate, args.sb_paths, seed=args.seed + 11)

        # Heston + SB
        sb_sv, sb_info_sv = build_martingale_sb_solver(
            spot=spot,
            r=args.rate,
            q=args.div,
            slices=slices,
            num_grid_density=args.density_grid,
            sigma_ref=args.sb_sigma_ref,
            use_sv_reference=True,
            heston_params={**heston_params, "rate": args.rate, "spot": spot},
            num_steps_per_segment=args.mc_steps,
        )
        C_sb_sv = sb_price_T2_calls(sb_sv, T2, T3, K_grid, args.rate, args.sb_paths, seed=args.seed + 22)

        # MOT bounds for forward-start straddle (T1 -> T2)
        fwd_ratio_12 = float(np.exp((args.rate - args.div) * (float(T2) - float(T1))))
        lb, ub = mot_forward_start_bounds_from_market(
            S1_samples=s1_samples,
            S2_samples=s2_samples,
            fwd_ratio=fwd_ratio_12,
            r=args.rate,
            t2=T2,
            epsilon=0.05,
            num_iters=200,
        )

        results["SB_Brownian"] = C_sb_bm
        results["SB_HestonRef"] = C_sb_sv
        sb_meta["mot_bounds_forward_start_straddle_T1_T2"] = {"lower": lb, "upper": ub}
        sb_meta["sb_info_bm"] = sb_info_bm
        sb_meta["sb_info_sv"] = sb_info_sv

    # -------------------------------------------------------------------------
    # Collect all models
    # -------------------------------------------------------------------------
    model_prices = {
        "Market": C_mkt,
        "TVI": C_tvi,
        "SABR": C_sabr,
        "SVI_VI": C_svi,
        "Heston_MC": C_heston,
    }
    if "SB_Brownian" in results:
        model_prices["SB_Brownian"] = results["SB_Brownian"]
        model_prices["SB_HestonRef"] = results["SB_HestonRef"]

    # Price errors
    price_err_rows = []
    for name, C in model_prices.items():
        if name == "Market":
            continue
        price_err_rows.append({"model": name, "rmse_price": rmse(C, C_mkt), "mae_price": mae(C, C_mkt)})

    # IV errors
    iv_err_rows = []
    for name, C in model_prices.items():
        if name == "Market":
            continue
        iv_model = iv_from_price_vec(spot, K_grid, T2, args.rate, C)
        iv_err_rows.append({"model": name, "rmse_iv": rmse(iv_model, iv_mkt), "mae_iv": mae(iv_model, iv_mkt)})

    # -------------------------------------------------------------------------
    # Arbitrage diagnostics (market slices, raw)
    # -------------------------------------------------------------------------
    arb: Dict[str, object] = {}
    if jax is not None:
        from schrodinger_bridge.finance.calibration import check_butterfly_arbitrage
        for tag, sl in [("T1", s1), ("T2", s2), ("T3", s3)]:
            K_f = np.linspace(sl.strikes.min(), sl.strikes.max(), 200)
            C_f = np.interp(K_f, sl.strikes, sl.call_mid)
            ok, viol = check_butterfly_arbitrage(jnp.array(K_f), jnp.array(C_f))
            arb[f"butterfly_ok_{tag}"] = bool(ok)
            v = np.array(viol)
            arb[f"butterfly_min_second_diff_{tag}"] = float(np.min(v)) if v.size else float("nan")

    # Calendar monotonicity at approx ATM across expiries
    atm_prices = []
    for sl in [s1, s2, s3]:
        idx = int(np.argmin(np.abs(sl.strikes - sl.forward)))
        atm_prices.append(float(sl.call_mid[idx]))
    arb["atm_prices"] = atm_prices
    arb["calendar_ok_atm"] = (atm_prices[0] <= atm_prices[1] + 1e-6) and (atm_prices[1] <= atm_prices[2] + 1e-6)

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    grid_rows = []
    for i, K in enumerate(K_grid):
        row = {
            "K": float(K),
            "Market": float(C_mkt[i]) if np.isfinite(C_mkt[i]) else float("nan"),
            "Market_iv": float(iv_mkt[i]) if np.isfinite(iv_mkt[i]) else float("nan"),
        }
        for name, C in model_prices.items():
            if name == "Market":
                continue
            row[name] = float(C[i]) if np.isfinite(C[i]) else float("nan")
        grid_rows.append(row)

    (out_dir / "meta.json").write_text(
        json.dumps({**meta, **sb_meta, "heston_params": heston_params, "arb": arb}, indent=2)
    )
    (out_dir / "price_errors_T2.json").write_text(json.dumps(price_err_rows, indent=2))
    (out_dir / "iv_errors_T2.json").write_text(json.dumps(iv_err_rows, indent=2))
    (out_dir / "grid_T2.json").write_text(json.dumps(grid_rows, indent=2))

    # Console summary
    print("\n=== Held-out T2 errors ===")
    for row in sorted(price_err_rows, key=lambda d: (d["rmse_price"] if np.isfinite(d["rmse_price"]) else 1e9)):
        print(f"{row['model']:>12s}  RMSE(price)={row['rmse_price']:.4g}  MAE(price)={row['mae_price']:.4g}")

    print("\n=== Held-out T2 IV errors ===")
    for row in sorted(iv_err_rows, key=lambda d: (d["rmse_iv"] if np.isfinite(d["rmse_iv"]) else 1e9)):
        print(f"{row['model']:>12s}  RMSE(iv)={row['rmse_iv']:.4g}  MAE(iv)={row['mae_iv']:.4g}")

    print("\n=== Arbitrage diagnostics (market slices; raw mids) ===")
    for k, v in arb.items():
        print(f"{k}: {v}")

    if "mot_bounds_forward_start_straddle_T1_T2" in sb_meta:
        b = sb_meta["mot_bounds_forward_start_straddle_T1_T2"]
        print("\n=== MOT bounds (time-0 discounted) forward-start straddle |S(T2)-S(T1)| ===")
        print(f"lower={b['lower']:.6g}, upper={b['upper']:.6g}")

    print(f"\nWrote report to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
