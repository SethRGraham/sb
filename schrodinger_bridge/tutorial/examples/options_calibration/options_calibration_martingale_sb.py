#!/usr/bin/env python3
"""
OPTIONS CALIBRATION WITH MARTINGALE SCHRÖDINGER BRIDGE
======================================================

This example demonstrates using the MartingaleSBSolver from our library
to calibrate asset dynamics to option prices while ensuring no-arbitrage.

We compare two approaches:
- MarginalSBSolver: Matches option-implied distributions but may violate no-arbitrage  
- MartingaleSBSolver: Matches distributions AND enforces E[S_T|S_t] = Forward

KEY MATHEMATICAL INSIGHT
========================
For risk-neutral pricing, discounted asset prices must be martingales:

    E^Q[S_T | F_t] = S_t · e^{r(T-t)} = F(t,T)

Standard Marginal SB does NOT enforce this! The MartingaleSBSolver adds:
1. Martingale-penalized OT coupling between consecutive marginals  
2. Exact projection ensuring E[S_end]/E[S_start] = e^{rτ}

BREEDEN-LITZENBERGER FORMULA
============================
Risk-neutral density from option prices:

    p(S_T = K) = e^{rT} × ∂²C/∂K²

This extracts the market-implied distribution from the option smile.

USAGE
=====
    python options_calibration_martingale_sb.py

Author: Built with Claude (Anthropic)
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid

import jax
import jax.numpy as jnp

# =============================================================================
# IMPORTS FROM OUR EXISTING LIBRARY
# =============================================================================

from schrodinger_bridge import (
    BrownianMotion,
    TimeGrid,
    mmd_squared,
)
from schrodinger_bridge.core.problem import MarginalDistribution
from schrodinger_bridge.marginal_sb import (
    MarginalConstraint,
    MarginalSBProblem,
    MarginalSBSolver,
    MarginalSBConfig,
)
from schrodinger_bridge.martingale_sb import (
    ForwardCurve,
    MartingaleSBProblem,
    MartingaleSBSolver,
    MartingaleSBConfig,
)


# =============================================================================
# PART 1: OPTIONS PRICING UTILITIES
# =============================================================================

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call option price.
    
    C = S·N(d₁) - K·e^{-rT}·N(d₂)
    
    where d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
          d₂ = d₁ - σ√T
    """
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


@dataclass
class OptionData:
    """Container for option market data at one expiry."""
    expiry: float
    strikes: np.ndarray
    ivs: np.ndarray
    forward: float


def generate_market_data(
    spot: float = 100.0,
    rate: float = 0.05,
    expiries: List[float] = None,
    base_vol: float = 0.20,
    skew: float = -0.10,
    smile: float = 0.02,
) -> List[OptionData]:
    """Generate realistic options market data with volatility smile.
    
    Creates a vol surface with:
    - Base ATM vol that decays with sqrt(T)
    - Negative skew (crash fear)
    - Smile (fat tails)
    """
    if expiries is None:
        expiries = [1/12, 0.25, 0.5, 1.0]
    
    market_data = []
    
    for T in expiries:
        forward = spot * np.exp(rate * T)
        
        num_strikes = 25
        log_moneyness = np.linspace(-0.4, 0.4, num_strikes)
        strikes = forward * np.exp(log_moneyness)
        
        # Vol smile: σ(K) = base/√(T+0.1) + skew·m + smile·m²
        term_adj = 1.0 / np.sqrt(T + 0.1)
        ivs = base_vol * term_adj + skew * log_moneyness + smile * log_moneyness**2
        ivs = np.maximum(ivs, 0.05)
        
        market_data.append(OptionData(
            expiry=T,
            strikes=strikes,
            ivs=ivs,
            forward=forward,
        ))
    
    return market_data


# =============================================================================
# PART 2: BREEDEN-LITZENBERGER DENSITY EXTRACTION
# =============================================================================

def extract_risk_neutral_density(
    market: OptionData,
    spot: float,
    rate: float,
    num_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract risk-neutral density via Breeden-Litzenberger.
    
    The formula: p(K) = e^{rT} × ∂²C/∂K²
    
    We fit a spline to call prices and differentiate twice.
    """
    T = market.expiry
    
    call_prices = np.array([
        black_scholes_call(spot, K, T, rate, iv)
        for K, iv in zip(market.strikes, market.ivs)
    ])
    
    spline = CubicSpline(market.strikes, call_prices)
    
    K_min = market.strikes.min() * 0.8
    K_max = market.strikes.max() * 1.2
    K_fine = np.linspace(K_min, K_max, num_points)
    
    # Second derivative = risk-neutral density (times e^{rT})
    d2C = spline(K_fine, 2)
    density = np.exp(rate * T) * d2C
    density = np.maximum(density, 0)
    
    total = trapezoid(density, K_fine)
    if total > 0:
        density = density / total
    
    return K_fine, density


def sample_from_density(
    key: jax.Array,
    strikes: np.ndarray,
    density: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    """Sample from risk-neutral density using inverse CDF."""
    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]
    
    u = np.array(jax.random.uniform(key, (num_samples,)))
    samples = np.interp(u, cdf, strikes)
    
    return samples


# =============================================================================
# PART 3: CUSTOM DISTRIBUTION CLASS FOR THE LIBRARY
# =============================================================================

class OptionsImpliedDistribution(MarginalDistribution):
    """Distribution extracted from options prices.
    
    This wraps the Breeden-Litzenberger extracted density so it can
    be used with our existing MarginalSBSolver and MartingaleSBSolver.
    """
    
    def __init__(
        self, 
        T: float, 
        spot: float, 
        rate: float, 
        market: Optional[OptionData] = None,
        n_samples: int = 10000,
    ):
        """Initialize from market data.
        
        Args:
            T: Expiry time (0 for spot distribution)
            spot: Spot price
            rate: Risk-free rate
            market: OptionData for this expiry (None for t=0)
            n_samples: Number of samples to pre-generate
        """
        self.T = T
        self.spot = spot
        self.rate = rate
        self._dim = 1
        
        if T > 0 and market is not None:
            K_fine, density = extract_risk_neutral_density(market, spot, rate)
            self._K_fine = K_fine
            self._density = density
            
            key = jax.random.PRNGKey(int(T * 1000))
            self._samples = sample_from_density(key, K_fine, density, n_samples)
        else:
            self._samples = spot * np.ones(n_samples)
            self._K_fine = None
            self._density = None
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def has_density(self) -> bool:
        return self._density is not None
    
    def sample(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Sample from the distribution."""
        indices = jax.random.choice(key, len(self._samples), shape=(n,))
        return jnp.array(self._samples)[indices, None]


# =============================================================================
# PART 4: MAIN COMPARISON USING EXISTING LIBRARY
# =============================================================================

def run_comparison(
    spot: float = 100.0,
    rate: float = 0.05,
    num_paths: int = 3000,
) -> Dict:
    """Compare MarginalSBSolver vs MartingaleSBSolver.
    
    This demonstrates how to use the existing library classes.
    """
    
    key = jax.random.PRNGKey(42)
    
    print("\n" + "="*70)
    print("OPTIONS CALIBRATION: MARGINAL SB vs MARTINGALE SB")
    print("="*70)
    print("\nUsing existing library classes:")
    print("  - MarginalSBSolver (from schrodinger_bridge.marginal_sb)")
    print("  - MartingaleSBSolver (from schrodinger_bridge.martingale_sb)")
    
    # -------------------------------------------------------------------------
    # Step 1: Generate market data
    # -------------------------------------------------------------------------
    expiries = [1/12, 0.25, 0.5, 1.0]  # 1M, 3M, 6M, 1Y
    market_data = generate_market_data(spot=spot, rate=rate, expiries=expiries)
    
    print(f"\nMarket Configuration:")
    print(f"  Spot:     ${spot:.2f}")
    print(f"  Rate:     {rate:.2%}")
    print(f"  Expiries: {expiries}")
    
    print(f"\nForward Prices (no-arbitrage requirement):")
    for T in expiries:
        F = spot * np.exp(rate * T)
        print(f"  F(0, {T:.3f}) = ${F:.2f}")
    
    # -------------------------------------------------------------------------
    # Step 2: Create distributions using OptionsImpliedDistribution
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Creating OptionsImpliedDistribution objects...")
    
    all_times = [0.0] + expiries
    distributions = [OptionsImpliedDistribution(0.0, spot, rate, None)]
    
    for T, market in zip(expiries, market_data):
        dist = OptionsImpliedDistribution(T, spot, rate, market)
        distributions.append(dist)
        print(f"  T={T:.3f}: mean=${np.mean(dist._samples):.2f}, fwd=${market.forward:.2f}")
    
    # -------------------------------------------------------------------------
    # Step 3: Normalize times to [0,1] and create MarginalConstraint objects
    # -------------------------------------------------------------------------
    T_max = max(expiries)
    normalized_times = [t / T_max for t in all_times]
    
    marginal_constraints = [
        MarginalConstraint(time=t_norm, distribution=dist)
        for t_norm, dist in zip(normalized_times, distributions)
    ]
    
    reference = BrownianMotion(sigma=0.15, dim=1)
    
    # -------------------------------------------------------------------------
    # Step 4: Create MARGINAL SB Problem
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Creating MarginalSBProblem (no martingale constraint)...")
    
    marginal_problem = MarginalSBProblem(
        reference=reference,
        marginals=marginal_constraints,
        time_grid=TimeGrid(num_steps=100),
        name="Marginal SB",
    )
    print(marginal_problem.summary())
    
    # -------------------------------------------------------------------------
    # Step 5: Create MARTINGALE SB Problem
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Creating MartingaleSBProblem (with martingale constraint)...")
    
    forward_curve = ForwardCurve(
        spot=spot,
        rate=rate * T_max,  # Scale rate for normalized time [0,1]
    )
    
    martingale_problem = MartingaleSBProblem(
        reference=reference,
        marginals=marginal_constraints,
        forward_curve=forward_curve,
        time_grid=TimeGrid(num_steps=100),
        name="Martingale SB",
    )
    print(martingale_problem.summary())
    
    # -------------------------------------------------------------------------
    # Step 6: Solve with MarginalSBSolver
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("[1/2] Solving with MarginalSBSolver...")
    
    marginal_config = MarginalSBConfig(
        segment_solver_type='doob',
        num_iterations=100,
        verbose=0,
    )
    marginal_solver = MarginalSBSolver(marginal_problem, marginal_config)
    
    key, subkey = jax.random.split(key)
    start = time.time()
    marginal_result = marginal_solver.train(subkey)
    marginal_time = time.time() - start
    print(f"  Completed in {marginal_time:.2f}s")
    
    key, subkey = jax.random.split(key)
    marginal_traj = marginal_solver.sample(subkey, num_paths)
    
    # -------------------------------------------------------------------------
    # Step 7: Solve with MartingaleSBSolver
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("[2/2] Solving with MartingaleSBSolver...")
    
    martingale_config = MartingaleSBConfig(
        martingale_weight=15.0,
        num_steps_per_segment=30,
        sigma_ref=0.15,
        verbose=0,
    )
    martingale_solver = MartingaleSBSolver(martingale_problem, martingale_config)
    
    key, subkey = jax.random.split(key)
    start = time.time()
    martingale_result = martingale_solver.train(subkey, num_samples=num_paths)
    martingale_time = time.time() - start
    print(f"  Completed in {martingale_time:.2f}s")
    
    key, subkey = jax.random.split(key)
    martingale_times, martingale_paths = martingale_solver.simulate(subkey, num_paths)
    
    # -------------------------------------------------------------------------
    # Step 8: Evaluate martingale property
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("EVALUATING MARTINGALE PROPERTY")
    print("="*70)
    print("\nThe martingale property requires:")
    print("  E[S_{T_j} | S_{T_i}] = S_{T_i} × e^{r(T_j - T_i)} = Forward")
    
    results = {}
    
    # --- Marginal SB Evaluation ---
    marginal_times_np = np.array(marginal_traj.times)
    marginal_paths_np = np.array(marginal_traj.paths[:, :, 0])
    
    print(f"\n{'='*55}")
    print("MARGINAL SB (should VIOLATE martingale):")
    print("-" * 55)
    
    marginal_errors = []
    for i in range(len(normalized_times) - 1):
        t_start, t_end = normalized_times[i], normalized_times[i+1]
        
        idx_start = np.argmin(np.abs(marginal_times_np - t_start))
        idx_end = np.argmin(np.abs(marginal_times_np - t_end))
        
        S_start = marginal_paths_np[:, idx_start]
        S_end = marginal_paths_np[:, idx_end]
        
        actual = np.mean(S_end) / np.mean(S_start)
        expected = np.exp(rate * T_max * (t_end - t_start))
        error = abs(actual - expected) / expected * 100
        marginal_errors.append(error)
        
        status = "✓" if error < 0.1 else "✗"
        print(f"  [{t_start:.2f}→{t_end:.2f}]: E[ratio]={actual:.5f}, "
              f"Forward={expected:.5f}, err={error:.3f}% {status}")
    
    # --- Martingale SB Evaluation ---
    martingale_times_np = np.array(martingale_times)
    martingale_paths_np = np.array(martingale_paths)
    
    print(f"\n{'='*55}")
    print("MARTINGALE SB (should SATISFY martingale):")
    print("-" * 55)
    
    martingale_errors = []
    for i in range(len(normalized_times) - 1):
        t_start, t_end = normalized_times[i], normalized_times[i+1]
        
        idx_start = np.argmin(np.abs(martingale_times_np - t_start))
        idx_end = np.argmin(np.abs(martingale_times_np - t_end))
        
        S_start = martingale_paths_np[:, idx_start]
        S_end = martingale_paths_np[:, idx_end]
        
        actual = np.mean(S_end) / np.mean(S_start)
        expected = np.exp(rate * T_max * (t_end - t_start))
        error = abs(actual - expected) / expected * 100
        martingale_errors.append(error)
        
        status = "✓" if error < 0.1 else "✗"
        print(f"  [{t_start:.2f}→{t_end:.2f}]: E[ratio]={actual:.5f}, "
              f"Forward={expected:.5f}, err={error:.3f}% {status}")
    
    # Store results
    results['Marginal SB'] = {
        'times': marginal_times_np,
        'paths': marginal_paths_np,
        'errors': marginal_errors,
        'max_error': max(marginal_errors),
    }
    results['Martingale SB'] = {
        'times': martingale_times_np,
        'paths': martingale_paths_np,
        'errors': martingale_errors,
        'max_error': max(martingale_errors),
    }
    
    # -------------------------------------------------------------------------
    # Step 9: Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n" + "-" * 60)
    print(f"{'Method':<25} {'Max Martingale Error':<20} {'Arbitrage-Free?'}")
    print("-" * 60)
    
    max_m = max(marginal_errors)
    max_mt = max(martingale_errors)
    
    print(f"{'MarginalSBSolver':<25} {max_m:<20.3f}% {'NO ✗' if max_m > 0.1 else 'YES ✓'}")
    print(f"{'MartingaleSBSolver':<25} {max_mt:<20.3f}% {'NO ✗' if max_mt > 0.1 else 'YES ✓'}")
    print("-" * 60)
    
    print("\n┌" + "─"*68 + "┐")
    print("│" + " "*20 + "KEY TAKEAWAYS" + " "*35 + "│")
    print("├" + "─"*68 + "┤")
    print("│                                                                    │")
    print("│  1. MarginalSBSolver: Matches distributions but MAY VIOLATE        │")
    print("│     no-arbitrage (martingale property not enforced)                │")
    print("│                                                                    │")
    print("│  2. MartingaleSBSolver: Matches distributions AND ENFORCES         │")
    print("│     no-arbitrage via martingale-constrained OT coupling            │")
    print("│                                                                    │")
    print("│  3. For derivatives pricing: ALWAYS use MartingaleSBSolver!        │")
    print("│                                                                    │")
    print("└" + "─"*68 + "┘")
    
    return results, normalized_times, spot, rate, T_max


# =============================================================================
# PART 5: VISUALIZATION
# =============================================================================

def create_visualization(
    results: Dict,
    normalized_times: List[float],
    spot: float,
    rate: float,
    T_max: float,
    save_dir: str = ".",
):
    """Create comparison plots."""
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    t_fwd = np.linspace(0, 1, 100)
    fwd_curve = spot * np.exp(rate * T_max * t_fwd)
    
    n_show = 50
    
    # --- Panel 1: Marginal SB ---
    ax = axes[0]
    times = results['Marginal SB']['times']
    paths = results['Marginal SB']['paths']
    
    for i in range(n_show):
        ax.plot(times, paths[i], alpha=0.12, lw=0.4, color='steelblue')
    ax.plot(t_fwd, fwd_curve, 'r--', lw=2.5, label='Forward F(0,t)', zorder=10)
    ax.plot(times, np.mean(paths, axis=0), 'b-', lw=2.5, label='Mean E[S_t]', zorder=10)
    ax.set_xlabel('Time (normalized)', fontsize=11)
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.set_title('MarginalSBSolver\n(no martingale constraint)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(70, 145)
    
    # --- Panel 2: Martingale SB ---
    ax = axes[1]
    times = results['Martingale SB']['times']
    paths = results['Martingale SB']['paths']
    
    for i in range(n_show):
        ax.plot(times, paths[i], alpha=0.12, lw=0.4, color='forestgreen')
    ax.plot(t_fwd, fwd_curve, 'r--', lw=2.5, label='Forward F(0,t)', zorder=10)
    ax.plot(times, np.mean(paths, axis=0), 'g-', lw=2.5, label='Mean E[S_t]', zorder=10)
    ax.set_xlabel('Time (normalized)', fontsize=11)
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.set_title('MartingaleSBSolver\n(with no-arbitrage constraint)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(70, 145)
    
    # --- Panel 3: Error comparison ---
    ax = axes[2]
    segments = [f'[{normalized_times[i]:.2f},{normalized_times[i+1]:.2f}]' 
                for i in range(len(normalized_times)-1)]
    x = np.arange(len(segments))
    width = 0.35
    
    ax.bar(x - width/2, results['Marginal SB']['errors'], width, 
           label='MarginalSBSolver', color='#e74c3c', alpha=0.85)
    ax.bar(x + width/2, results['Martingale SB']['errors'], width,
           label='MartingaleSBSolver', color='#27ae60', alpha=0.85)
    ax.axhline(0.1, color='orange', ls='--', lw=1.5, label='0.1% threshold')
    ax.set_xlabel('Time Segment', fontsize=11)
    ax.set_ylabel('Martingale Error (%)', fontsize=11)
    ax.set_title('No-Arbitrage Violation\n(lower = better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(segments, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    path = os.path.join(save_dir, 'martingale_sb_options_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved visualization to: {path}")
    
    return path


# =============================================================================
# PART 6: USING THE LIBRARY'S CONVENIENCE FUNCTIONS
# =============================================================================

def demo_convenience_functions():
    """Show how to use the library's convenience functions."""
    print("\n" + "="*70)
    print("ALTERNATIVE: USING CONVENIENCE FUNCTIONS")
    print("="*70)
    
    print("""
    The library also provides convenience functions for common workflows:
    
    from schrodinger_bridge.martingale_sb import (
        create_martingale_sb_problem,
        solve_martingale_sb,
    )
    
    # Create problem from market parameters
    problem = create_martingale_sb_problem(
        spot=100.0,
        rate=0.05,
        expiries=[0.0, 0.25, 0.5, 1.0],
        marginal_distributions=[dist_0, dist_1, dist_2, dist_3],
    )
    
    # Solve in one line
    solver, results = solve_martingale_sb(problem, key, num_samples=3000)
    
    # Check martingale property
    mart_check = solver.check_martingale(key)
    print(f"Average martingale error: {mart_check['avg_martingale_error']:.4%}")
    
    # Check marginal matching
    mmd_check = solver.check_marginals(key)
    for t, mmd in mmd_check.items():
        print(f"  {t}: MMD² = {mmd:.6f}")
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the full comparison using our existing library."""
    
    results, normalized_times, spot, rate, T_max = run_comparison(
        spot=100.0,
        rate=0.05,
        num_paths=3000,
    )
    
    # Create visualization
    print("\nCreating visualization...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    create_visualization(results, normalized_times, spot, rate, T_max, output_dir)
    
    # Show convenience functions
    demo_convenience_functions()
    
    print("\n✓ Complete!")
    
    return results


if __name__ == "__main__":
    main()
