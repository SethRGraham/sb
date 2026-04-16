# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "jax",
#     "jaxlib", 
#     "numpy",
#     "matplotlib",
#     "schrodinger-bridge",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(42)
    return jax, jnp, key, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # 📈 Schrödinger Bridges for Options Pricing
    
    **Using Empirical Reference Dynamics with Koopman Methods**

    This notebook demonstrates how to use Schrödinger Bridges for quantitative finance
    when you **cannot assume Brownian motion** as the reference process.

    ## The Problem

    In options pricing, we want to find a stochastic process that:
    1. **Starts** from the current market state (source distribution)
    2. **Ends** at the risk-neutral distribution implied by option prices (target)
    3. Is **closest to historical dynamics** (empirical reference, not Brownian motion!)

    ## Critical Design Choices for Interday Options

    | Choice | Recommendation | Why |
    |--------|---------------|-----|
    | **Time formulation** | Discrete-time (IPF) | Daily data → noisy derivatives |
    | **Koopman method** | EDMD on $K_{\Delta t}$, not gEDMD | Generator estimation unstable for $\Delta t = 1/252$ |
    | **Noise handling** | optDMD or Bagging | Market data is noisy |
    | **State space** | Low-dimensional factors | Curse of dimensionality |

    ---
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. Why Discrete-Time for Daily Data?

    ### The Derivative Estimation Problem

    gEDMD requires time derivatives: $\dot{X} \approx (X_{k+1} - X_k)/\Delta t$

    For daily data ($\Delta t = 1/252 \approx 0.004$):

    $$\text{Var}(\dot{X}) \approx \frac{2\sigma^2_{\text{noise}}}{(\Delta t)^2} \approx 62,500 \times \sigma^2_{\text{noise}}$$

    **This noise amplification destroys the generator estimate!**

    ### The Correct Approach: Discrete-Time Markov Chain

    Instead of approximating the continuous generator $\mathcal{L}$, 
    we directly model the **discrete-time transition operator**:

    $$K_{\Delta t}: \quad \mathbb{E}[g(X_{k+1}) | X_k = x] = (K_{\Delta t} g)(x)$$

    EDMD approximates this as a matrix $\tilde{K}$ in dictionary space:

    $$\Psi(X_{k+1}) \approx \tilde{K} \Psi(X_k)$$

    **No derivative estimation needed!**

    ### Discrete-Time Schrödinger Bridge

    The discrete SB finds the minimum-entropy path measure:

    $$P^* = \arg\min_{P: P_0 = \mu_0, P_N = \mu_1} \text{KL}(P \| P_{\text{ref}})$$

    This is solved by **Iterative Proportional Fitting (IPF)** / Sinkhorn iteration,
    operating on the transition matrices induced by $\tilde{K}$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. State Space Design for Options

    ### The Problem with Raw Prices

    Using raw prices $S_t$ or even log-prices $\log S_t$ has issues:
    - High-dimensional if multiple assets
    - Non-stationary
    - Mixes different types of risk factors

    ### Recommended: Low-Dimensional Factor State

    Define state $z_t$ as a vector of **interpretable factors**:

    ```python
    z_t = [
        r_t,           # Log-return (or cumulative return)
        IV_ATM_t,      # At-the-money implied volatility
        skew_t,        # IV skew (25δ put - 25δ call)
        term_t,        # Term structure slope
        tau,           # Time-to-maturity (CRITICAL!)
    ]
    ```

    ### Why Include Time-to-Maturity $\tau$?

    The SB process is inherently **time-inhomogeneous** over $[0, T]$.

    A Koopman model fit on historical data gives a **time-homogeneous** operator.

    **Fix**: Augment state with $\tau$ so the dictionary can capture time-dependence:

    $$\Psi(z, \tau) = [1, z, z^2, \tau, \tau^2, z\tau, \ldots]$$

    Now the "time-homogeneous" Koopman operator on $(z, \tau)$ space 
    captures the time-inhomogeneous dynamics on $z$ space!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. Extracting the Risk-Neutral Target (Correctly!)

    ### The Breeden-Litzenberger Formula

    The risk-neutral density can be extracted from call prices:

    $$p_{\text{RN}}(K) = e^{rT} \frac{\partial^2 C}{\partial K^2}$$

    ### ⚠️ Critical: Don't Differentiate Raw Prices!

    Numerical second derivatives of noisy call prices are **extremely unstable**.

    **Wrong approach:**
    ```python
    # BAD: Direct numerical differentiation
    d2C_dK2 = jnp.gradient(jnp.gradient(call_prices, dK), dK)  # Noise explosion!
    ```

    **Correct approach:**
    1. **Fit an arbitrage-free call surface** (convex in $K$, monotone decreasing)
    2. **Differentiate the fitted surface analytically**

    ### Implementation with Arbitrage-Free Fitting

    ```python
    def extract_rn_density_safe(strikes, call_prices, r, T, method='spline'):
        '''Extract risk-neutral density with proper smoothing.'''
        
        # Step 1: Fit arbitrage-free call surface
        if method == 'spline':
            # Monotone convex spline (e.g., Hyman filter)
            from scipy.interpolate import PchipInterpolator
            # Ensure monotonicity and convexity
            call_interp = PchipInterpolator(strikes, call_prices)
            
        elif method == 'svi':
            # SVI parametric fit (preferred for options)
            # C(K) comes from Black-Scholes with SVI implied vol
            svi_params = fit_svi(strikes, implied_vols)
            call_interp = lambda K: bs_call(K, svi_iv(K, svi_params), T, r)
        
        # Step 2: Differentiate analytically (or with fine grid)
        K_fine = jnp.linspace(strikes[0], strikes[-1], 1000)
        C_fine = call_interp(K_fine)
        
        # Second derivative with smoothing
        dK = K_fine[1] - K_fine[0]
        d2C = jnp.gradient(jnp.gradient(C_fine, dK), dK)
        
        # Step 3: Construct density
        density = jnp.exp(r * T) * d2C
        density = jnp.maximum(density, 0)  # Enforce non-negativity
        density = density / (jnp.sum(density) * dK)  # Normalize
        
        return K_fine, density
    ```

    ### Alternative: Use Implied Volatility Directly

    Often cleaner to work with the **implied volatility surface** 
    and derive the density from a parametric model (SVI, SABR).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. The Empirical Reference: Historical Dynamics

    ### What We're Modeling

    The **reference measure** $P_{\text{ref}}$ is the historical (physical, $\mathbb{P}$) dynamics.

    We estimate this from historical factor trajectories:
    $$z_0 \to z_1 \to z_2 \to \cdots \to z_N$$

    ### Creating the Empirical Reference

    ```python
    from schrodinger_bridge.solvers.koopman import EmpiricalReferenceDynamics

    # historical_factors: shape [num_windows, num_days, num_factors]
    # e.g., 500 different 30-day windows, daily observations, 5 factors
    historical_factors = construct_factor_histories(market_data)

    # Create empirical reference
    # NOTE: This estimates the discrete-time transition, not continuous drift
    empirical_ref = EmpiricalReferenceDynamics(
        trajectories=historical_factors,
        dt=1/252,  # Daily
        drift_method='kernel',  # For discrete transitions, this estimates E[z_{k+1}|z_k]
        diffusion_method='local_variance',
        kernel_bandwidth=None,  # Auto via median heuristic
    )
    ```

    ### Augmenting with Time-to-Maturity

    ```python
    def augment_with_tau(trajectories, T_expiry):
        '''Add time-to-maturity as a state coordinate.'''
        num_traj, num_days, num_factors = trajectories.shape
        
        # tau decreases from T to 0 over the window
        tau = jnp.linspace(T_expiry, 0, num_days)
        tau_expanded = jnp.broadcast_to(tau[None, :, None], (num_traj, num_days, 1))
        
        return jnp.concatenate([trajectories, tau_expanded], axis=-1)

    # Now state is [r, IV_ATM, skew, term, tau]
    augmented_trajectories = augment_with_tau(historical_factors, T_expiry=30/252)
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Noise-Robust Koopman Estimation

    ### Why Standard EDMD Fails on Market Data

    Standard EDMD solves: $\min \|\Psi(X') - K\Psi(X)\|_F^2$

    This is **biased** when $X$ contains noise:
    - Eigenvalues shrink toward zero
    - Spurious modes appear
    - Poor out-of-sample prediction

    ### optDMD: Joint Optimization

    optDMD optimizes eigenvalues and modes jointly:

    $$\min_{\lambda, \phi, b} \|X - \Phi \text{diag}(b) V_\lambda\|_F^2$$

    where $V_\lambda$ is the Vandermonde matrix of eigenvalues.

    ### Bagging-optDMD: Uncertainty Quantification

    Bootstrap aggregation provides:
    1. **Robust estimates** (median/mean across bags)
    2. **Uncertainty quantification** (std across bags)
    3. **Spurious mode detection** (high variance = spurious)

    ### Implementation

    ```python
    from schrodinger_bridge.solvers.koopman import (
        optdmd_from_trajectories,
        RBFDictionary,
        PolynomialDictionary,
        CompositeDictionary,
    )

    # Build dictionary with time-to-maturity
    dim = num_factors + 1  # +1 for tau
    
    poly_dict = PolynomialDictionary(dim=dim, degree=2, include_time=False)
    rbf_dict = RBFDictionary(dim=dim, num_centers=30)
    rbf_dict.set_centers_from_data(augmented_trajectories.reshape(-1, dim))
    
    dictionary = CompositeDictionary([poly_dict, rbf_dict])

    # Noise-robust Koopman estimation
    koopman_result = optdmd_from_trajectories(
        trajectories=augmented_trajectories,
        dictionary=dictionary,
        rank=15,  # Number of modes to keep
        dt=1/252,
        method='bagging',  # Bootstrap for robustness
        num_bags=100,
        key=jax.random.PRNGKey(42),
    )

    # Check which modes are reliable
    reliable_modes = koopman_result['eigenvalues_std'] < 0.1  # Low uncertainty
    print(f"Reliable modes: {jnp.sum(reliable_modes)} / {len(reliable_modes)}")
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. Discrete-Time IPF-Koopman Solver

    ### The Algorithm

    1. **Fit Koopman operator** $\tilde{K}$ on historical data (with optDMD)
    2. **Project marginals** to dictionary space:
       - Source: $\mu_0$ (current market state)
       - Target: $\mu_1$ (risk-neutral from options)
    3. **Run IPF/Sinkhorn** to find optimal coupling
    4. **Sample bridge paths** and map back to state space

    ### Why IPF?

    IPF solves the discrete-time SB exactly (up to dictionary approximation):

    $$\pi^*_{ij} = \alpha_i \tilde{K}_{ij} \beta_j$$

    where $\alpha, \beta$ are found by Sinkhorn iteration to match marginals.

    ### Implementation

    ```python
    from schrodinger_bridge.solvers.koopman import DiscreteIPFKoopmanSolver

    # Define source: small uncertainty around current state
    current_state = jnp.array([0.0, 0.2, -0.05, 0.01, 30/252])  # [r, IV, skew, term, tau]
    
    class PointMassWithNoise(MarginalDistribution):
        def __init__(self, center, noise_std=0.01):
            self.center = center
            self.noise_std = noise_std
            self._dim = len(center)
        
        @property
        def dim(self):
            return self._dim
        
        def sample(self, key, num_samples):
            noise = jax.random.normal(key, (num_samples, self._dim)) * self.noise_std
            return self.center + noise

    source = PointMassWithNoise(current_state, noise_std=0.01)

    # Define target: risk-neutral distribution (in factor space!)
    # This requires mapping the K-space RN density to z-space
    target = FactorSpaceRNDistribution(K_grid, rn_density, factor_mapping)

    # Solve discrete-time SB
    solver = DiscreteIPFKoopmanSolver(
        source=source,
        target=target,
        koopman_operator=koopman_result,  # From optDMD
        dictionary=dictionary,
        num_time_steps=30,  # Days to expiry
        sinkhorn_iterations=100,
        sinkhorn_regularization=0.01,
    )

    result = solver.solve(key)
    bridge_paths = result.sample_paths(key, num_samples=10000)
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Interpretation: What Does the SB Give Us?

    ### The Drift Adjustment

    The SB modifies the reference (historical) dynamics to hit the target (risk-neutral):

    $$P^*_{\text{transition}} \propto P^{\mathbb{P}}_{\text{transition}} \cdot \psi(z_{k+1}, \tau_{k+1}) / \psi(z_k, \tau_k)$$

    where $\psi$ are the Schrödinger potentials.

    ### Is This "The Market Price of Risk"?

    **Partially, with important caveats:**

    ✅ The SB finds an entropy-minimizing $\mathbb{P} \to \tilde{\mathbb{Q}}$ change of measure

    ✅ This $\tilde{\mathbb{Q}}$ is consistent with the option-implied terminal marginal

    ⚠️ This is **not unique**: many measures match the same terminal marginal

    ⚠️ The "true" market price of risk (if it exists) would require the **full term structure**

    **Correct interpretation**: 

    > The SB drift adjustment represents the **minimum-entropy Girsanov kernel** 
    > that transforms historical dynamics to match the risk-neutral terminal distribution.
    > It is *a* market price of risk consistent with observed option prices, 
    > specifically the one that deviates least from historical behavior.

    ### What This Is Useful For

    1. **Pricing path-dependent options**: The SB gives a full path distribution, not just terminal
    2. **Scenario generation**: Paths that are "realistic" (close to historical) but hit RN terminal
    3. **Model calibration**: The required drift adjustment reveals market-implied dynamics
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8. Complete Workflow

    ```python
    import jax
    import jax.numpy as jnp
    from schrodinger_bridge.solvers.koopman import (
        EmpiricalReferenceDynamics,
        DiscreteIPFKoopmanSolver,
        optdmd_from_trajectories,
        PolynomialDictionary,
        RBFDictionary,
        CompositeDictionary,
    )

    key = jax.random.PRNGKey(42)

    # ============================================================
    # STEP 1: Construct factor state space
    # ============================================================
    
    # Load and process market data
    # factors: [log_return, IV_ATM, skew, term_slope]
    historical_factors = construct_factors_from_market_data(raw_data)
    
    # Augment with time-to-maturity
    T_expiry = 30 / 252  # 30 days
    historical_augmented = augment_with_tau(historical_factors, T_expiry)
    
    dim = historical_augmented.shape[-1]  # num_factors + 1

    # ============================================================
    # STEP 2: Build dictionary and fit Koopman operator
    # ============================================================
    
    # Data-adaptive dictionary
    poly_dict = PolynomialDictionary(dim=dim, degree=2)
    rbf_dict = RBFDictionary(dim=dim, num_centers=30)
    rbf_dict.set_centers_from_data(historical_augmented.reshape(-1, dim))
    dictionary = CompositeDictionary([poly_dict, rbf_dict])
    
    # Noise-robust Koopman estimation
    k1, k2 = jax.random.split(key)
    koopman_result = optdmd_from_trajectories(
        trajectories=historical_augmented,
        dictionary=dictionary,
        rank=15,
        dt=1/252,
        method='bagging',
        num_bags=100,
        key=k1,
    )

    # ============================================================
    # STEP 3: Extract risk-neutral target (with proper smoothing!)
    # ============================================================
    
    # Fit arbitrage-free IV surface first
    svi_params = fit_svi_surface(strikes, maturities, implied_vols)
    
    # Extract density for our target maturity
    K_grid, rn_density = extract_rn_density_from_svi(
        svi_params, T_expiry, r=0.05, S0=100
    )
    
    # Map to factor space (this is problem-specific!)
    target_in_factor_space = map_price_density_to_factors(
        K_grid, rn_density, current_factors
    )

    # ============================================================
    # STEP 4: Define source and target distributions
    # ============================================================
    
    # Source: current state with small uncertainty
    current_state = jnp.concatenate([current_factors, jnp.array([T_expiry])])
    source = PointMassWithNoise(current_state, noise_std=0.01)
    
    # Target: risk-neutral in factor space
    target = FactorSpaceDistribution(target_in_factor_space)

    # ============================================================
    # STEP 5: Solve discrete-time SB via IPF-Koopman
    # ============================================================
    
    solver = DiscreteIPFKoopmanSolver(
        source=source,
        target=target,
        koopman_operator=koopman_result['K'],
        dictionary=dictionary,
        num_time_steps=30,
        sinkhorn_reg=0.01,
    )
    
    result = solver.solve(k2)
    
    # ============================================================
    # STEP 6: Sample paths and price options
    # ============================================================
    
    bridge_paths = result.sample_paths(key, num_samples=10000)
    # bridge_paths: [10000, 31, dim] - paths in factor space
    
    # Map back to prices for payoff calculation
    price_paths = map_factors_to_prices(bridge_paths, S0=100)
    
    # Price a path-dependent option (e.g., Asian)
    asian_payoffs = jnp.maximum(jnp.mean(price_paths, axis=1) - K_strike, 0)
    asian_price = jnp.exp(-r * T_expiry) * jnp.mean(asian_payoffs)
    
    print(f"Asian call price: {asian_price:.4f}")
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 9. Summary: Best Practices for Finance

    ### Do's ✅

    | Practice | Reason |
    |----------|--------|
    | Use **discrete-time IPF**, not gEDMD | Daily data → noisy derivatives |
    | Use **optDMD or Bagging** | Market data is noisy |
    | Work in **low-dimensional factor space** | Avoids curse of dimensionality |
    | **Augment state with $\tau$** | Captures time-inhomogeneity |
    | Fit **arbitrage-free surface** before B-L | Raw numerical derivatives explode |
    | Use **regularization** in EDMD | Handles ill-conditioning |

    ### Don'ts ❌

    | Anti-pattern | Problem |
    |--------------|---------|
    | gEDMD on daily data | Derivative estimation amplifies noise 62,500× |
    | Raw price differentiation for B-L | Numerical noise dominates |
    | High-dimensional raw state | Dictionary size explodes |
    | Time-homogeneous model for bridge | Misses $t$-dependence of SB |
    | Claiming "the" market price of risk | It's *a* min-entropy one, not unique |

    ### The Pipeline

    ```
    Historical Factors              Option Prices
         │                               │
         ▼                               ▼
    ┌────────────────┐          ┌─────────────────┐
    │ Augment with τ │          │ Fit arb-free    │
    │                │          │ IV surface      │
    └───────┬────────┘          └────────┬────────┘
            │                            │
            ▼                            ▼
    ┌────────────────┐          ┌─────────────────┐
    │ optDMD/Bagging │          │ Extract RN      │
    │ → K̃_Δt        │          │ density (B-L)   │
    └───────┬────────┘          └────────┬────────┘
            │                            │
            └──────────┬─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Discrete IPF    │
              │ on K̃_Δt        │
              └───────┬─────────┘
                      │
                      ▼
              ┌─────────────────┐
              │ Bridge Paths    │
              │ → Option Prices │
              └─────────────────┘
    ```

    ---

    **Happy trading! 📊**
    """
    )
    return


if __name__ == "__main__":
    app.run()
