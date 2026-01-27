# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "jax",
#     "jaxlib",
#     "numpy",
#     "matplotlib",
#     "pillow",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def imports():
    """Import all required libraries."""
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import animation
    from PIL import Image
    import io

    # Set up nice plotting defaults
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.size'] = 11

    return Image, LineCollection, io, jax, jnp, mo, np, plt


@app.cell
def title(mo):
    """Display the tutorial title and overview."""
    mo.md(
        r"""
        # Stochastic Differential Equations & Schrödinger Bridges

        **A Visual Guide to Forward SDEs, Numerical Simulation, and the Inverse Problem**

        ---

        ## What You'll Learn

        | Topic | Key Question |
        |-------|--------------|
        | **Forward SDEs** | Given a drift, how do particles evolve randomly over time? |
        | **Euler-Maruyama** | How do we simulate SDEs on a computer? |
        | **Physical Intuition** | What do drift and diffusion *mean* physically? |
        | **Schrödinger Bridge** | Given start and end distributions, what drift connects them? |
        | **Drift Learning** | How do we solve the inverse problem? |

        ---

        ## 🎯 The Core Math Takeaway

        > **An SDE is fully determined by its drift $b(x,t)$.** 
        > 
        > - **Forward problem:** Given drift → find how distribution evolves
        > - **Inverse problem (Schrödinger Bridge):** Given start and end distributions → find the drift
        """
    )
    return


@app.cell
def sde_definition(mo):
    """Explain the mathematical definition of an SDE."""
    mo.md(
        r"""
        ## 1. Stochastic Differential Equations (SDEs)

        ### The Equation

        An SDE describes how a random quantity $X_t$ evolves over time:

        $$
        dX_t = \underbrace{b(X_t, t)}_{\text{drift}} \, dt + \underbrace{\sigma(t)}_{\text{diffusion}} \, dW_t
        $$

        | Term | Meaning | Physical Intuition |
        |------|---------|-------------------|
        | $X_t$ | State at time $t$ | Position of a particle |
        | $b(X_t, t)$ | Drift function | Deterministic force/velocity |
        | $\sigma(t)$ | Diffusion coefficient | Noise amplitude |
        | $dW_t$ | Wiener process increment | Random "kick" (white noise) |

        ---

        ### 🧠 Physical Intuition: The Drunk Walker

        Imagine a person walking on a street:

        ```
        Without drift (σ > 0, b = 0):     With drift (σ > 0, b > 0):

            ↖ ↗                               → → →
           ↙ X ↗                             ↗ X → →
            ↘ ↙                               → ↗ →

        "Drunk walker"                    "Drunk walker with destination"
        (pure random walk)                (biased toward the right)
        ```

        - **Drift $b(x,t)$**: The "intended direction" — where the walker *wants* to go
        - **Diffusion $\sigma$**: How "drunk" they are — the magnitude of random stumbles
        - **$dW_t$**: The random stumble at each instant (Gaussian noise)
        """
    )
    return


@app.cell
def sde_intuition_expanded(mo):
    """Deeper intuition about drift and diffusion."""
    mo.md(
        r"""
        ### Understanding Drift vs Diffusion

        #### The Drift $b(x, t)$: Deterministic Tendency

        The drift tells particles where they "want" to go. Common examples:

        | Drift Function | Behavior | Real-World Analog |
        |----------------|----------|-------------------|
        | $b = 0$ | No preferred direction | Gas molecules at equilibrium |
        | $b = c$ (constant) | Steady flow | River current |
        | $b = -\theta(x - \mu)$ | Pull toward $\mu$ | Spring returning to rest |
        | $b = \mu x$ | Exponential growth/decay | Population dynamics |

        #### The Diffusion $\sigma$: Random Fluctuations

        The diffusion coefficient controls the "spread" of randomness:

        | $\sigma$ Value | Behavior |
        |----------------|----------|
        | $\sigma = 0$ | Deterministic ODE (no randomness) |
        | $\sigma$ small | Tight trajectories, low variance |
        | $\sigma$ large | Wild trajectories, high variance |

        ---

        ### 🔑 Key Insight: Drift is the Control Knob

        > If you want particles to go somewhere specific, **change the drift**.
        > The diffusion just adds noise around that tendency.

        This is why the Schrödinger Bridge problem reduces to **finding the right drift**.
        """
    )
    return


@app.cell
def euler_maruyama_explanation(mo):
    """Explain the Euler-Maruyama numerical method."""
    mo.md(
        r"""
        ## 2. Euler-Maruyama: Solving SDEs Numerically

        We can't solve most SDEs analytically, so we **simulate** them.

        ### From Continuous to Discrete

        The SDE $dX_t = b(X_t, t)\,dt + \sigma\,dW_t$ becomes:

        $$
        \boxed{X_{t+\Delta t} = X_t + b(X_t, t) \cdot \Delta t + \sigma \cdot \sqrt{\Delta t} \cdot Z}
        $$

        where $Z \sim \mathcal{N}(0, I)$ is standard Gaussian noise.

        ### Why $\sqrt{\Delta t}$?

        This is crucial! The Wiener process has the property:

        $$W_{t+\Delta t} - W_t \sim \mathcal{N}(0, \Delta t)$$

        So $dW_t \approx \sqrt{\Delta t} \cdot Z$ where $Z \sim \mathcal{N}(0, 1)$.

        ### The Algorithm

        ```python
        def euler_maruyama(x0, drift_fn, sigma, dt, num_steps):
            x = x0
            for k in range(num_steps):
                t = k * dt
                Z = sample_gaussian()           # Z ~ N(0, I)
                x = x + drift_fn(x, t) * dt + sigma * sqrt(dt) * Z
            return x
        ```
        """
    )
    return


@app.cell
def define_sde_functions(jax, jnp):
    """Define core SDE simulation functions."""

    def euler_maruyama_step(key, x, t, dt, drift_fn, sigma):
        """Single Euler-Maruyama step."""
        x = jnp.atleast_2d(x)
        drift = drift_fn(x, t)
        noise = jax.random.normal(key, x.shape)
        x_next = x + drift * dt + sigma * jnp.sqrt(dt) * noise
        return x_next

    def simulate_sde(key, x0, drift_fn, sigma, T=1.0, num_steps=100):
        """Simulate SDE trajectories using Euler-Maruyama."""
        x0 = jnp.atleast_2d(x0)
        batch_size, dim = x0.shape
        dt = T / num_steps
        times = jnp.linspace(0, T, num_steps + 1)

        trajectories = jnp.zeros((batch_size, num_steps + 1, dim))
        trajectories = trajectories.at[:, 0, :].set(x0)

        keys = jax.random.split(key, num_steps)

        x = x0
        for i in range(num_steps):
            x = euler_maruyama_step(keys[i], x, times[i], dt, drift_fn, sigma)
            trajectories = trajectories.at[:, i + 1, :].set(x)

        return times, trajectories

    return (simulate_sde,)


@app.cell
def forward_sde_intro(mo):
    """Introduce forward SDE examples."""
    mo.md(
        r"""
        ## 3. Forward Problem: Given Drift, Simulate Evolution

        Let's see how different drift functions create different behaviors.

        ### Three Fundamental SDEs

        | Name | Drift $b(x,t)$ | SDE | Behavior |
        |------|---------------|-----|----------|
        | **Brownian Motion** | $0$ | $dX = \sigma\,dW$ | Pure random walk |
        | **Ornstein-Uhlenbeck** | $-\theta(X - \mu)$ | $dX = -\theta(X-\mu)dt + \sigma\,dW$ | Mean-reverting |
        | **Geometric BM** | $\mu X$ | $dX = \mu X\,dt + \sigma\,dW$ | Exponential + noise |
        """
    )
    return


@app.cell
def define_drift_functions(jnp):
    """Define various drift functions for demonstration."""

    def zero_drift(x, t):
        """Brownian motion: no drift."""
        return jnp.zeros_like(x)

    def ou_drift(x, t, theta=2.0, mu=0.0):
        """Ornstein-Uhlenbeck: mean-reverting toward mu."""
        return -theta * (x - mu)

    def linear_drift(x, t, mu=0.5):
        """Linear drift: exponential tendency."""
        return mu * x

    return linear_drift, ou_drift, zero_drift


@app.cell
def visualize_sde_comparison(
    jax,
    jnp,
    linear_drift,
    np,
    ou_drift,
    plt,
    simulate_sde,
    zero_drift,
):
    def _():
        """Visualize three fundamental SDE types side by side."""
        key = jax.random.PRNGKey(42)
        n_particles = 50
        x0_sde = jnp.zeros((n_particles, 1))
        sigma_sde = 0.5
        T_sde = 2.0

        k1, k2, k3 = jax.random.split(key, 3)

        times_sde, traj_bm = simulate_sde(k1, x0_sde, zero_drift, sigma_sde, T=T_sde)
        _, traj_ou = simulate_sde(k2, x0_sde, lambda x, t: ou_drift(x, t), sigma_sde, T=T_sde)
        _, traj_linear = simulate_sde(k3, x0_sde, lambda x, t: linear_drift(x, t), sigma_sde, T=T_sde)

        fig_sde, axes_sde = plt.subplots(1, 3, figsize=(14, 4))

        processes = [
            ("Brownian Motion\n$b(x,t) = 0$", traj_bm, 'blue'),
            ("Ornstein-Uhlenbeck\n$b(x,t) = -\\theta(x - \\mu)$", traj_ou, 'green'),
            ("Linear Drift\n$b(x,t) = \\mu x$", traj_linear, 'orange'),
        ]

        for ax, (title_str, traj, color) in zip(axes_sde, processes):
            for i in range(min(25, n_particles)):
                ax.plot(np.array(times_sde), np.array(traj[i, :, 0]), 
                        alpha=0.4, linewidth=0.8, color=color)
            mean_traj = np.array(traj[:, :, 0].mean(axis=0))
            ax.plot(np.array(times_sde), mean_traj, 'k-', linewidth=2, label='Mean')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time $t$')
            ax.set_title(title_str, fontsize=11)
            ax.legend(loc='upper left')

        axes_sde[0].set_ylabel('Position $X_t$')
        fig_sde.suptitle('Forward Problem: How Different Drifts Shape Evolution', 
                         fontsize=13, fontweight='bold')
        return plt.tight_layout()


    _()
    return


@app.cell
def brownian_motion_deep_dive(mo):
    """Deep dive into Brownian motion."""
    mo.md(
        r"""
        ### 3.1 Brownian Motion: The Foundation

        **SDE:** $dX_t = \sigma \, dW_t$ (no drift, $b = 0$)

        This is the simplest SDE — pure randomness with no preferred direction.

        #### Key Properties

        | Property | Formula | Meaning |
        |----------|---------|---------|
        | Mean | $\mathbb{E}[X_t] = X_0$ | Expected position doesn't change |
        | Variance | $\text{Var}(X_t) = \sigma^2 t$ | Spread grows linearly with time |
        | Distribution | $X_t \sim \mathcal{N}(X_0, \sigma^2 t)$ | Gaussian at all times |

        #### Physical Examples
        - Pollen grain in water (Robert Brown, 1827)
        - Stock price fluctuations (random component)
        - Heat diffusion in a material
        """
    )
    return


@app.cell
def visualize_brownian_properties(jax, jnp, np, plt, simulate_sde, zero_drift):
    def _():
        """Visualize Brownian motion properties: variance grows with time."""
        key_bm = jax.random.PRNGKey(100)
        n_particles_bm = 500
        x0_bm = jnp.zeros((n_particles_bm, 1))
        sigma_bm = 0.5

        times_bm, traj_bm = simulate_sde(key_bm, x0_bm, zero_drift, sigma_bm, T=2.0, num_steps=200)

        fig_bm, axes_bm = plt.subplots(1, 2, figsize=(12, 4))

        for i in range(30):
            axes_bm[0].plot(np.array(times_bm), np.array(traj_bm[i, :, 0]), 
                         alpha=0.3, linewidth=0.5, color='steelblue')
        axes_bm[0].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='$E[X_t] = 0$')
        axes_bm[0].fill_between(np.array(times_bm), 
                             -2*sigma_bm*np.sqrt(np.array(times_bm)), 
                             2*sigma_bm*np.sqrt(np.array(times_bm)),
                             alpha=0.2, color='red', label='$\\pm 2\\sigma\\sqrt{t}$')
        axes_bm[0].set_xlabel('Time $t$')
        axes_bm[0].set_ylabel('Position $X_t$')
        axes_bm[0].set_title('Brownian Motion: Trajectories Spread Over Time')
        axes_bm[0].legend()

        empirical_var_bm = np.array(traj_bm[:, :, 0].var(axis=0))
        theoretical_var_bm = sigma_bm**2 * np.array(times_bm)

        axes_bm[1].plot(np.array(times_bm), empirical_var_bm, 'b-', linewidth=2, label='Empirical Var$(X_t)$')
        axes_bm[1].plot(np.array(times_bm), theoretical_var_bm, 'r--', linewidth=2, label='Theory: $\\sigma^2 t$')
        axes_bm[1].set_xlabel('Time $t$')
        axes_bm[1].set_ylabel('Variance')
        axes_bm[1].set_title('Variance Grows Linearly with Time')
        axes_bm[1].legend()
        return plt.tight_layout()


    _()
    return


@app.cell
def ou_process_deep_dive(mo):
    """Deep dive into Ornstein-Uhlenbeck process."""
    mo.md(
        r"""
        ### 3.2 Ornstein-Uhlenbeck: Mean-Reverting Process

        **SDE:** $dX_t = -\theta(X_t - \mu) \, dt + \sigma \, dW_t$

        The drift pulls particles back toward the mean $\mu$ with strength $\theta$.

        #### Key Properties

        | Property | Formula | Meaning |
        |----------|---------|---------|
        | Mean | $\mathbb{E}[X_t] \to \mu$ as $t \to \infty$ | Converges to $\mu$ |
        | Stationary Var | $\frac{\sigma^2}{2\theta}$ | Variance stabilizes |
        | Relaxation time | $\tau = 1/\theta$ | Time to "forget" initial condition |

        #### Physical Examples
        - Particle in a potential well (thermal fluctuations)
        - Interest rates (Vasicek model)
        - Neuron membrane voltage

        #### The Drift Acts Like a Spring

        ```
        x > μ:  drift = -θ(x - μ) < 0  →  pulled LEFT toward μ
        x < μ:  drift = -θ(x - μ) > 0  →  pushed RIGHT toward μ
        x = μ:  drift = 0              →  no force (equilibrium)
        ```
        """
    )
    return


@app.cell
def visualize_ou_properties(jax, np, ou_drift, plt, simulate_sde):
    def _():
        """Visualize OU process mean reversion."""
        key_ou = jax.random.PRNGKey(200)
        n_particles_ou = 300
        x0_spread_ou = 3.0 * jax.random.normal(key_ou, (n_particles_ou, 1))

        sigma_ou = 0.5
        theta_ou = 2.0
        mu_ou = 0.0

        k1_ou, _ = jax.random.split(key_ou)
        times_ou, traj_ou = simulate_sde(
            k1_ou, x0_spread_ou, lambda x, t: ou_drift(x, t, theta=theta_ou, mu=mu_ou), 
            sigma_ou, T=3.0, num_steps=300
        )

        fig_ou, axes_ou = plt.subplots(1, 2, figsize=(12, 4))

        for i in range(40):
            axes_ou[0].plot(np.array(times_ou), np.array(traj_ou[i, :, 0]), 
                         alpha=0.3, linewidth=0.5, color='seagreen')

        mean_traj_ou = np.array(traj_ou[:, :, 0].mean(axis=0))
        axes_ou[0].plot(np.array(times_ou), mean_traj_ou, 'k-', linewidth=2.5, label='Mean trajectory')
        axes_ou[0].axhline(y=mu_ou, color='red', linestyle='--', linewidth=2, label=f'Target $\\mu = {mu_ou}$')
        axes_ou[0].set_xlabel('Time $t$')
        axes_ou[0].set_ylabel('Position $X_t$')
        axes_ou[0].set_title('OU Process: Particles Converge to Mean')
        axes_ou[0].legend()

        empirical_var_ou = np.array(traj_ou[:, :, 0].var(axis=0))
        stationary_var_ou = sigma_ou**2 / (2 * theta_ou)

        axes_ou[1].plot(np.array(times_ou), empirical_var_ou, 'g-', linewidth=2, label='Empirical Var$(X_t)$')
        axes_ou[1].axhline(y=stationary_var_ou, color='red', linestyle='--', linewidth=2, 
                        label=f'Stationary: $\\sigma^2/(2\\theta) = {stationary_var_ou:.3f}$')
        axes_ou[1].set_xlabel('Time $t$')
        axes_ou[1].set_ylabel('Variance')
        axes_ou[1].set_title('Variance Stabilizes (Unlike Brownian Motion!)')
        axes_ou[1].legend()
        return plt.tight_layout()


    _()
    return


@app.cell
def forward_problem_summary(mo):
    """Summarize the forward problem."""
    mo.md(
        r"""
        ### 3.3 Forward Problem Summary

        We've seen how different drifts create different behaviors:

        | Drift | Long-term Behavior | Variance |
        |-------|-------------------|----------|
        | $b = 0$ | Particles diffuse forever | $\to \infty$ |
        | $b = -\theta(x - \mu)$ | Particles concentrate at $\mu$ | $\to \sigma^2/(2\theta)$ |
        | $b = \mu x$ | Exponential spread | $\to \infty$ (fast) |

        ---

        ### 🔑 The Forward Problem is "Easy"

        > Given drift $b(x,t)$ → Just simulate with Euler-Maruyama!

        **But what if we want the reverse?**

        > Given where particles should START and END → Find the drift?

        This is the **inverse problem**, and it's much harder. Enter the Schrödinger Bridge.
        """
    )
    return


@app.cell
def inverse_problem_intro(mo):
    """Introduce the inverse problem."""
    mo.md(
        r"""
        ## 4. The Inverse Problem: Schrödinger Bridges

        ### Forward vs Inverse

        | | Forward Problem | Inverse Problem |
        |--|-----------------|-----------------|
        | **Given** | Drift $b(x,t)$, initial dist $\mu_0$ | Initial dist $\mu_0$, final dist $\mu_1$ |
        | **Find** | Final distribution $\mu_1$ | Drift $b^*(x,t)$ |
        | **Difficulty** | Easy (simulate!) | Hard (optimization) |
        | **Method** | Euler-Maruyama | Schrödinger Bridge |

        ---

        ### The Schrödinger Bridge Problem

        **Given:** Source distribution $\mu_0$ and target distribution $\mu_1$

        **Find:** The stochastic process $P^*$ that:
        1. Starts distributed as $\mu_0$
        2. Ends distributed as $\mu_1$  
        3. Is **closest to Brownian motion** (minimum "effort")

        $$
        P^* = \arg\min_{P: P_0 = \mu_0, P_1 = \mu_1} \text{KL}(P \| P_{\text{Brownian}})
        $$

        The KL divergence measures how "far" our process is from pure Brownian motion.
        Minimizing it means: **use the least amount of drift necessary**.
        """
    )
    return


@app.cell
def sb_drift_formula(mo):
    """Present the key drift formula for Schrödinger Bridges."""
    mo.md(
        r"""
        ### The Key Formula: Drift Points Toward Expected Destination

        The Schrödinger Bridge is itself an SDE with a **special drift**:

        $$
        dX_t = b^*(X_t, t) \, dt + \sigma \, dW_t
        $$

        where the optimal drift is:

        $$
        \boxed{b^*(x, t) = \frac{\mathbb{E}[X_1 \mid X_t = x] - x}{1 - t}}
        $$

        ---

        ### 🧠 Intuition: GPS Navigation for Particles

        Imagine each particle has a GPS that knows where it should end up:

        ```
        At time t, particle at position x:

        1. GPS computes: "On average, I should end up at E[X₁ | X_t = x]"
        2. Direction to go: E[X₁ | X_t = x] - x  
        3. How urgent: divide by remaining time (1 - t)
        4. Result: drift = (destination - current) / time_remaining
        ```

        As $t \to 1$, the urgency increases! (Denominator shrinks)
        """
    )
    return


@app.cell
def gaussian_case_theory(mo):
    """Theory for the Gaussian Schrödinger Bridge."""
    mo.md(
        r"""
        ## 5. The Gaussian Case: Analytical Solution

        When both $\mu_0$ and $\mu_1$ are Gaussian, we get a **closed-form solution!**

        ### Setup

        $$
        \mu_0 = \mathcal{N}(m_0, \Sigma_0), \quad \mu_1 = \mathcal{N}(m_1, \Sigma_1)
        $$

        ### Special Case: Equal Covariances ($\Sigma_0 = \Sigma_1$)

        When the shapes are the same, the optimal transport is just **translation**:

        $$
        T(x) = x + (m_1 - m_0)
        $$

        And the Schrödinger Bridge drift is remarkably simple:

        $$
        \boxed{b^*(x, t) = m_1 - m_0 \quad \text{(constant everywhere!)}}
        $$

        ---

        ### Why Constant Drift?

        With equal covariances, every particle should move in the same direction 
        by the same amount. No particle needs special treatment — they all just 
        "translate" together while diffusing.
        """
    )
    return


@app.cell
def define_gaussian_sb_drift(jnp):
    """Define the Gaussian Schrödinger Bridge drift function."""

    def gaussian_sb_drift(x, t, m0, m1):
        """Compute Schrödinger Bridge drift for Gaussian case (equal covariances)."""
        x = jnp.atleast_2d(x)
        drift = jnp.broadcast_to(m1 - m0, x.shape)
        return drift

    return (gaussian_sb_drift,)


@app.cell
def visualize_gaussian_sb(
    LineCollection,
    gaussian_sb_drift,
    jax,
    jnp,
    np,
    plt,
    simulate_sde,
):
    def _():
        """Visualize Gaussian Schrödinger Bridge trajectories."""
        # Define source and target means (exported for use in drift field visualization)
        m0_gsb = jnp.array([-2.0, 0.0])
        m1_gsb = jnp.array([2.0, 0.0])
        sigma_gsb = 0.3

        key_gsb = jax.random.PRNGKey(123)
        n_particles_gsb = 200
        x0_gsb = m0_gsb + sigma_gsb * jax.random.normal(key_gsb, (n_particles_gsb, 2))

        def sb_drift_gsb(x, t):
            return gaussian_sb_drift(x, t, m0_gsb, m1_gsb)

        key_sim_gsb = jax.random.PRNGKey(456)
        times_gsb, traj_gsb = simulate_sde(key_sim_gsb, x0_gsb, sb_drift_gsb, sigma_gsb, T=1.0, num_steps=100)

        fig_gsb, ax_gsb = plt.subplots(figsize=(12, 5))

        for i in range(min(60, n_particles_gsb)):
            points = np.array(traj_gsb[i]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='viridis', alpha=0.4, linewidth=0.8)
            lc.set_array(np.array(times_gsb[:-1]))
            ax_gsb.add_collection(lc)

        ax_gsb.scatter(np.array(x0_gsb[:, 0]), np.array(x0_gsb[:, 1]), 
                       c='blue', alpha=0.3, s=20, label='Source $\\mu_0$')
        ax_gsb.scatter(np.array(traj_gsb[:, -1, 0]), np.array(traj_gsb[:, -1, 1]), 
                       c='red', alpha=0.3, s=20, label='Target $\\mu_1$')

        ax_gsb.scatter([float(m0_gsb[0])], [float(m0_gsb[1])], c='blue', s=200, marker='*', 
                       edgecolors='black', linewidths=2, zorder=10)
        ax_gsb.scatter([float(m1_gsb[0])], [float(m1_gsb[1])], c='red', s=200, marker='*', 
                       edgecolors='black', linewidths=2, zorder=10)

        ax_gsb.annotate('', xy=(float(m1_gsb[0])-0.3, float(m1_gsb[1])), 
                        xytext=(float(m0_gsb[0])+0.3, float(m0_gsb[1])),
                        arrowprops=dict(arrowstyle='->', lw=3, color='purple'))
        ax_gsb.text(0, 0.8, '$b^* = m_1 - m_0$', fontsize=14, ha='center', color='purple')

        ax_gsb.set_xlim(-4, 4)
        ax_gsb.set_ylim(-2, 2)
        ax_gsb.set_aspect('equal')
        ax_gsb.set_xlabel('$x$', fontsize=12)
        ax_gsb.set_ylabel('$y$', fontsize=12)
        ax_gsb.set_title('Gaussian Schrödinger Bridge: Constant Drift Transport', 
                         fontsize=13, fontweight='bold')
        ax_gsb.legend(loc='upper right')

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax_gsb, label='Time $t$')
        return plt.tight_layout()


    _()
    return


@app.cell
def visualize_drift_field(gaussian_sb_drift, jnp, m0_gsb, m1_gsb, np, plt):
    """Visualize the drift vector field at multiple time points."""

    def plot_drift_at_time(t_val, ax):
        x_range = np.linspace(-4, 4, 15)
        y_range = np.linspace(-2, 2, 10)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)

        points = jnp.stack([X_grid.flatten(), Y_grid.flatten()], axis=-1)
        drifts = gaussian_sb_drift(points, t_val, m0_gsb, m1_gsb)

        U = np.array(drifts[:, 0]).reshape(X_grid.shape)
        V = np.array(drifts[:, 1]).reshape(X_grid.shape)
        magnitude = np.sqrt(U**2 + V**2)

        ax.quiver(X_grid, Y_grid, U, V, magnitude, cmap='coolwarm', alpha=0.8)
        ax.scatter([float(m0_gsb[0])], [float(m0_gsb[1])], c='blue', s=120, marker='o', 
                   edgecolors='black', linewidths=2, zorder=10)
        ax.scatter([float(m1_gsb[0])], [float(m1_gsb[1])], c='red', s=120, marker='o', 
                   edgecolors='black', linewidths=2, zorder=10)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(f'$t = {t_val:.1f}$', fontsize=11)

    fig_drift, axes_drift = plt.subplots(2, 3, figsize=(13, 7))
    axes_flat = axes_drift.flatten()

    time_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    for ax, t_val in zip(axes_flat, time_vals):
        plot_drift_at_time(t_val, ax)

    fig_drift.suptitle('Drift Vector Field (Constant for Equal-Covariance Gaussians)', 
                       fontsize=13, fontweight='bold')
    plt.tight_layout()
    return


@app.cell
def non_gaussian_intro(mo):
    """Introduce the non-Gaussian case."""
    mo.md(
        r"""
        ## 6. Beyond Gaussians: Learning the Drift

        Real distributions are rarely Gaussian. Examples:
        - **Two Moons**: Crescent-shaped clusters
        - **Swiss Roll**: Spiral manifold  
        - **Mixture Models**: Multi-modal distributions

        For these, we must **learn** the drift $b^*(x, t)$.

        ---

        ### The Kernel Method

        Recall the drift formula: $b^*(x, t) = \frac{\mathbb{E}[X_1 \mid X_t = x] - x}{1 - t}$

        We can estimate $\mathbb{E}[X_1 \mid X_t = x]$ using **kernel regression**:

        $$
        \mathbb{E}[X_1 \mid X_t = x] \approx \sum_i w_i(x, t) \cdot x_1^{(i)}
        $$

        The weights $w_i$ measure: "How likely is it that the particle at $x$ came from 
        source point $x_0^{(i)}$ and is heading to target point $x_1^{(i)}$?"
        """
    )
    return


@app.cell
def define_kernel_sb_drift(jax, jnp):
    """Define kernel-based Schrödinger Bridge drift."""

    def kernel_sb_drift(x, t, source_samples, target_samples, sigma_ref):
        """Compute kernel-based Schrödinger Bridge drift."""
        x = jnp.atleast_2d(x)
        t_scalar = jnp.asarray(t).reshape(())

        remaining_time = jnp.maximum(1.0 - t_scalar, 1e-4)
        bridge_var = sigma_ref ** 2 * jnp.maximum(t_scalar, 0.01) * remaining_time

        mu_t = (1.0 - t_scalar) * source_samples + t_scalar * target_samples

        diff = x[:, None, :] - mu_t[None, :, :]
        sq_dist = jnp.sum(diff ** 2, axis=-1)

        log_weights = -sq_dist / (2 * bridge_var + 1e-8)
        weights = jax.nn.softmax(log_weights, axis=-1)

        expected_target = jnp.einsum('bi,id->bd', weights, target_samples)
        drift = (expected_target - x) / remaining_time

        return drift

    return (kernel_sb_drift,)


@app.cell
def define_sample_distributions(jax, jnp):
    """Define sampling functions for various distributions."""

    def sample_two_moons(key, n_samples, noise=0.1):
        """Sample from the two moons distribution."""
        k1, k2, k3 = jax.random.split(key, 3)
        n_per_moon = n_samples // 2

        theta1 = jax.random.uniform(k1, (n_per_moon,)) * jnp.pi
        x1 = jnp.cos(theta1)
        y1 = jnp.sin(theta1)
        moon1 = jnp.stack([x1, y1], axis=-1)

        theta2 = jax.random.uniform(k2, (n_samples - n_per_moon,)) * jnp.pi
        x2 = 1 - jnp.cos(theta2)
        y2 = 0.5 - jnp.sin(theta2)
        moon2 = jnp.stack([x2, y2], axis=-1)

        samples = jnp.concatenate([moon1, moon2], axis=0)
        samples = samples + noise * jax.random.normal(k3, samples.shape)
        return samples

    def sample_gaussian_2d(key, n_samples, mean, std):
        """Sample from 2D isotropic Gaussian."""
        return mean + std * jax.random.normal(key, (n_samples, 2))

    return sample_gaussian_2d, sample_two_moons


@app.cell
def visualize_two_moons_problem(
    jax,
    jnp,
    np,
    plt,
    sample_gaussian_2d,
    sample_two_moons,
):
    """Visualize the Gaussian to Two Moons transport problem."""
    key_dist = jax.random.PRNGKey(789)
    k_src, k_tgt = jax.random.split(key_dist)

    source_samples = sample_gaussian_2d(k_src, 500, jnp.array([-2.0, 0.0]), 0.5)
    target_samples = sample_two_moons(k_tgt, 500, noise=0.08)

    fig_dist, axes_dist = plt.subplots(1, 2, figsize=(12, 5))

    axes_dist[0].scatter(np.array(source_samples[:, 0]), np.array(source_samples[:, 1]),
                    c='blue', alpha=0.5, s=20)
    axes_dist[0].set_title('Source: Gaussian $\\mu_0$', fontsize=12)
    axes_dist[0].set_aspect('equal')
    axes_dist[0].set_xlim(-4, 3)
    axes_dist[0].set_ylim(-2, 2)

    axes_dist[1].scatter(np.array(target_samples[:, 0]), np.array(target_samples[:, 1]),
                    c='red', alpha=0.5, s=20)
    axes_dist[1].set_title('Target: Two Moons $\\mu_1$', fontsize=12)
    axes_dist[1].set_aspect('equal')
    axes_dist[1].set_xlim(-4, 3)
    axes_dist[1].set_ylim(-2, 2)

    fig_dist.suptitle('The Challenge: Transport Gaussian → Two Moons', 
                      fontsize=13, fontweight='bold')
    plt.tight_layout()

    return source_samples, target_samples


@app.cell
def visualize_kernel_transport(
    LineCollection,
    jax,
    kernel_sb_drift,
    np,
    plt,
    simulate_sde,
    source_samples,
    target_samples,
):
    def _():
        """Visualize kernel-based transport from Gaussian to Two Moons."""
        sigma_kt = 0.5

        def two_moons_drift(x, t):
            return kernel_sb_drift(x, t, source_samples, target_samples, sigma_kt)

        key_kt = jax.random.PRNGKey(999)
        n_transport = 100
        times_kt, traj_kt = simulate_sde(
            key_kt, source_samples[:n_transport], two_moons_drift, sigma_kt, T=1.0, num_steps=80
        )

        fig_kt, ax_kt = plt.subplots(figsize=(11, 6))

        ax_kt.scatter(np.array(source_samples[:, 0]), np.array(source_samples[:, 1]),
                      c='blue', alpha=0.12, s=12, label='Source $\\mu_0$')
        ax_kt.scatter(np.array(target_samples[:, 0]), np.array(target_samples[:, 1]),
                      c='red', alpha=0.12, s=12, label='Target $\\mu_1$')

        for i in range(60):
            points = np.array(traj_kt[i]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='viridis', alpha=0.5, linewidth=0.8)
            lc.set_array(np.array(times_kt[:-1]))
            ax_kt.add_collection(lc)

        ax_kt.scatter(np.array(traj_kt[:, -1, 0]), np.array(traj_kt[:, -1, 1]),
                      c='limegreen', alpha=0.5, s=35, marker='x', label='Transported', linewidths=1.5)

        ax_kt.set_xlim(-4, 3)
        ax_kt.set_ylim(-2, 2)
        ax_kt.set_aspect('equal')
        ax_kt.set_xlabel('$x$', fontsize=12)
        ax_kt.set_ylabel('$y$', fontsize=12)
        ax_kt.set_title('Kernel-Based Schrödinger Bridge: Gaussian → Two Moons', 
                        fontsize=13, fontweight='bold')
        ax_kt.legend(loc='upper left')
        return plt.tight_layout(), traj_kt, times_kt


    plot, traj_kt, times_kt = _()
    return times_kt, traj_kt


@app.cell
def create_animation(
    Image,
    io,
    np,
    plt,
    source_samples,
    target_samples,
    times_kt,
    traj_kt,
):
    """Create animated GIF of transport evolution."""
    paths_anim = np.array(traj_kt)
    times_arr_anim = np.array(times_kt)
    n_particles_anim, n_frames_anim, _ = paths_anim.shape

    all_x_anim = np.concatenate([paths_anim[:, :, 0].flatten(), 
                            np.array(source_samples[:, 0]), 
                            np.array(target_samples[:, 0])])
    all_y_anim = np.concatenate([paths_anim[:, :, 1].flatten(), 
                            np.array(source_samples[:, 1]), 
                            np.array(target_samples[:, 1])])

    x_min_anim, x_max_anim = all_x_anim.min() - 0.5, all_x_anim.max() + 0.5
    y_min_anim, y_max_anim = all_y_anim.min() - 0.5, all_y_anim.max() + 0.5

    frames_anim = []
    frame_indices_anim = range(0, n_frames_anim, 2)

    for frame_idx in frame_indices_anim:
        fig_frame, ax_frame = plt.subplots(figsize=(10, 6))

        ax_frame.scatter(np.array(source_samples[:, 0]), np.array(source_samples[:, 1]),
                   c='blue', alpha=0.1, s=10, label='Source $\\mu_0$')
        ax_frame.scatter(np.array(target_samples[:, 0]), np.array(target_samples[:, 1]),
                   c='red', alpha=0.1, s=10, label='Target $\\mu_1$')

        positions_anim = paths_anim[:, frame_idx, :2]
        progress_anim = frame_idx / n_frames_anim
        colors_anim = plt.cm.viridis(np.full(len(positions_anim), progress_anim))
        ax_frame.scatter(positions_anim[:, 0], positions_anim[:, 1], c=colors_anim, s=35, alpha=0.7)

        ax_frame.text(0.02, 0.98, f't = {times_arr_anim[frame_idx]:.3f}', transform=ax_frame.transAxes,
                fontsize=14, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_frame.set_xlim(x_min_anim, x_max_anim)
        ax_frame.set_ylim(y_min_anim, y_max_anim)
        ax_frame.set_aspect('equal')
        ax_frame.set_xlabel('$x$', fontsize=12)
        ax_frame.set_ylabel('$y$', fontsize=12)
        ax_frame.set_title('Schrödinger Bridge: Gaussian → Two Moons', fontsize=13, fontweight='bold')
        ax_frame.legend(loc='upper right')

        buf_anim = io.BytesIO()
        plt.tight_layout()
        fig_frame.savefig(buf_anim, format='png', dpi=100, bbox_inches='tight')
        buf_anim.seek(0)
        frames_anim.append(Image.open(buf_anim).copy())
        buf_anim.close()
        plt.close(fig_frame)

    gif_buffer_anim = io.BytesIO()
    frames_anim[0].save(
        gif_buffer_anim, format='GIF',
        save_all=True, append_images=frames_anim[1:],
        duration=100, loop=0
    )
    gif_buffer_anim.seek(0)
    gif_image = Image.open(gif_buffer_anim)

    return


@app.cell
def display_gif(mo):
    """Display the animated GIF."""
    mo.md(
        r"""
        ## 7. Animated Transport

        Watch particles flow from the Gaussian source to the Two Moons target:
        """
    )
    return


@app.cell
def summary(mo):
    """Display summary of key formulas."""
    mo.md(
        r"""
        ## 8. Summary: Key Formulas

        ### The SDE

        $$
        dX_t = b(X_t, t) \, dt + \sigma \, dW_t
        $$

        ### Euler-Maruyama Update

        $$
        X_{t+\Delta t} = X_t + b(X_t, t) \cdot \Delta t + \sigma \sqrt{\Delta t} \cdot Z, \quad Z \sim \mathcal{N}(0, I)
        $$

        ### Forward Problem Examples

        | Process | Drift | Behavior |
        |---------|-------|----------|
        | Brownian | $b = 0$ | Random walk, variance $\to \infty$ |
        | OU | $b = -\theta(x-\mu)$ | Mean-reverting to $\mu$ |
        | Linear | $b = \mu x$ | Exponential growth + noise |

        ### Schrödinger Bridge Drift

        $$
        \boxed{b^*(x, t) = \frac{\mathbb{E}[X_1 \mid X_t = x] - x}{1 - t}}
        $$

        ### Gaussian Case (Equal Covariances)

        $$
        b^*(x, t) = m_1 - m_0 \quad \text{(constant)}
        $$

        ### Kernel Estimation

        $$
        \mathbb{E}[X_1 \mid X_t = x] \approx \sum_i w_i(x, t) \cdot x_1^{(i)}
        $$

        ---

        ### 🎯 Main Takeaway

        > **Forward Problem:** Given drift → simulate with Euler-Maruyama  
        > **Inverse Problem:** Given endpoints → find drift via Schrödinger Bridge
        >
        > The drift $b^*(x,t)$ is the answer to both problems!
        """
    )
    return


@app.cell
def library_integration(mo):
    """Show how to use the schrodinger_bridge library."""
    mo.md(
        r"""
        ## 9. Using the `schrodinger_bridge` Library

        Your library provides production-ready implementations:

        ```python
        from schrodinger_bridge import (
            SBProblem, BrownianMotion, GaussianDistribution, 
            TwoMoonsDistribution, TimeGrid
        )
        from schrodinger_bridge.solvers import DoobHTransformSolver

        # Define problem
        problem = SBProblem(
            reference=BrownianMotion(sigma=0.5, dim=2),
            source=GaussianDistribution(mean=[-2, 0], cov=0.25, dim=2),
            target=TwoMoonsDistribution(),
            time_grid=TimeGrid(num_steps=100),
        )

        # Solve (DoobHTransformSolver uses kernel method for non-Gaussian)
        solver = DoobHTransformSolver(problem)
        result = solver.train(jax.random.PRNGKey(0))

        # Extract the learned drift
        drift_fn = solver.extract_drift(result.params)

        # Sample trajectories
        trajectories = solver.sample(jax.random.PRNGKey(1), num_samples=500)
        ```

        ### Solver Guide

        | Scenario | Best Solver |
        |----------|-------------|
        | Both Gaussian | `DoobHTransformSolver` (analytical) |
        | General, interpretable | `DoobHTransformSolver` (kernel) |
        | Complex distributions | `ScoreBasedSolver` (neural) |
        | Optimal control | `FBSDESolver` |
        """
    )
    return


@app.cell
def conclusion(mo):
    """Concluding remarks."""
    mo.md(
        r"""
        ---

        ## 🏆 Congratulations!

        You now understand:

        1. ✅ **SDEs** model random dynamics via drift (direction) and diffusion (noise)
        2. ✅ **Euler-Maruyama** numerically solves SDEs using $\sqrt{\Delta t}$ noise scaling
        3. ✅ **Brownian motion** has no drift; **OU process** is mean-reverting
        4. ✅ **Schrödinger Bridge** is the inverse problem: find drift from endpoints
        5. ✅ **The key formula**: $b^*(x,t) = (\mathbb{E}[X_1 | X_t=x] - x)/(1-t)$
        6. ✅ **Gaussian case** has analytical solution; general case needs kernel/neural methods

        **Next steps:** Explore neural approaches (score-based, FBSDE) in your library!
        """
    )
    return


if __name__ == "__main__":
    app.run()
