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

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    return jax, jnp, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # 🌉 Schrödinger Bridge Tutorial

    **An Interactive Guide Using the `schrodinger_bridge` Library**

    This comprehensive tutorial demonstrates Schrödinger Bridges through hands-on examples 
    using our actual library implementation. You'll learn:

    ## Part I: Foundations
    1. **The Core Problem** — What is a Schrödinger Bridge?
    2. **Mathematical Foundations** — SDEs, drift, and the key equations
    3. **Brownian Bridges** — The foundation for SB methods

    ## Part II: Problem Definition & Solvers
    4. **Problem Definition** — Using `SBProblem`, distributions, and reference dynamics
    5. **Solver Methods** — All 6 solvers: when and how to use each
    6. **Solver Comparison** — Decision flowchart and performance tradeoffs

    ## Part III: Extensions for Quantitative Finance
    7. **Marginal SB** — Intermediate marginal constraints
    8. **Martingale SB** — No-arbitrage constraint for derivatives pricing
    9. **Options Calibration** — Real-world quant finance application

    ## Part IV: Applications
    10. **Synthetic Data Generation** — Creating plausible intermediate states
    11. **Visualization & Diagnostics** — Evaluating solution quality

    ---
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. What is a Schrödinger Bridge?

    ### 1.1 Historical Origins

    The Schrödinger Bridge problem originated in **1931** when Erwin Schrödinger 
    (yes, the quantum mechanics Schrödinger!) posed a fascinating question about 
    statistical mechanics:

    > *"If we observe particles distributed according to μ₀ at time 0 and according 
    > to μ₁ at time 1, what is the **most likely** evolution of the system?"*

    Schrödinger was troubled by the apparent paradox in statistical mechanics: 
    given the laws of physics (Brownian motion), certain configurations at time 1 
    seem astronomically unlikely. Yet we observe them! His resolution was profound:

    **We seek the stochastic process that is closest to the reference (Brownian motion) 
    while satisfying the observed boundary conditions.**

    Mathematically, this means **minimizing the KL divergence** from the reference process.
    Equivalently, among all processes matching the endpoints, we find the one with 
    **maximum entropy** relative to the reference — the "least committal" or "most random" 
    evolution consistent with our observations.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 1.2 The Core Problem Statement

    The **Schrödinger Bridge (SB)** problem asks:

    > Given two probability distributions μ₀ (source) and μ₁ (target), find the 
    > stochastic process P* that transforms one into the other with **minimum entropy** 
    > relative to a reference process P_ref.

    $$\boxed{P^* = \argmin_{P: P_0 = \mu_0, P_1 = \mu_1} \text{KL}(P \| P_{\text{ref}})}$$

    ### The Key Mathematical Insight

    The solution has a beautiful structure. The optimal drift is:

    $$b^*(x, t) = b_{\text{ref}}(x, t) + \sigma^2 \nabla \log h(x, t)$$

    where $h(x, t)$ is the **Doob h-function** — it encodes how to steer particles 
    toward the target distribution.

    For Brownian reference ($b_{\text{ref}} = 0$), this simplifies to pointing 
    toward the **expected target position**:

    $$b^*(x, t) = \frac{\mathbb{E}[X_1 \mid X_t = x] - x}{1 - t}$$

    **Main Math Takeaway**: The SB finds the "least surprising" way to get from 
    source to target — it's the maximum entropy solution subject to endpoint constraints.
    """
    )
    return


@app.cell
def _(jax, jnp, mo, plt):
    def _():
        # CONCEPTUAL VISUALIZATION: The SB Problem Structure
        from schrodinger_bridge import (
            SBProblem,
            BrownianMotion,
            GaussianDistribution,
            TwoMoonsDistribution,
            TimeGrid,
        )

        # Define a simple SB problem using the library
        demo_problem = SBProblem(
            reference=BrownianMotion(sigma=0.5, dim=2),
            source=GaussianDistribution(
                mean=jnp.array([-2.0, 0.0]),
                cov=0.25,
                dim=2,
            ),
            target=GaussianDistribution(
                mean=jnp.array([2.0, 0.0]),
                cov=0.25,
                dim=2,
            ),
            time_grid=TimeGrid(t0=0.0, t1=1.0, num_steps=50),
            name="Demo: Gaussian Translation",
        )

        print(demo_problem.summary())

        # Visualize source and target
        demo_key = jax.random.PRNGKey(0)
        demo_k1, demo_k2 = jax.random.split(demo_key)
        source_samples = demo_problem.sample_source(demo_k1, 200)
        target_samples = demo_problem.sample_target(demo_k2, 200)

        fig_concept, demo_axes = plt.subplots(1, 3, figsize=(14, 4))

        # Source
        demo_axes[0].scatter(source_samples[:, 0], source_samples[:, 1], 
                        alpha=0.5, c='blue', s=20)
        demo_axes[0].set_title('Source Distribution μ₀\n(t = 0)', fontsize=12, fontweight='bold')
        demo_axes[0].set_xlim(-4, 4)
        demo_axes[0].set_ylim(-2, 2)
        demo_axes[0].set_aspect('equal')
        demo_axes[0].grid(True, alpha=0.3)

        # Middle: The bridge concept
        demo_axes[1].scatter(source_samples[:, 0], source_samples[:, 1], 
                        alpha=0.2, c='blue', s=10, label='Source')
        demo_axes[1].scatter(target_samples[:, 0], target_samples[:, 1], 
                        alpha=0.2, c='red', s=10, label='Target')
        demo_axes[1].annotate('', xy=(1.5, 0), xytext=(-1.5, 0),
                         arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
        demo_axes[1].text(0, -1.5, 'Schrödinger Bridge\n(minimize KL from Brownian motion)', 
                     ha='center', fontsize=10, style='italic')
        demo_axes[1].set_title('The Bridge Problem\n(0 < t < 1)', fontsize=12, fontweight='bold')
        demo_axes[1].set_xlim(-4, 4)
        demo_axes[1].set_ylim(-2, 2)
        demo_axes[1].set_aspect('equal')
        demo_axes[1].grid(True, alpha=0.3)
        demo_axes[1].legend()

        # Target
        demo_axes[2].scatter(target_samples[:, 0], target_samples[:, 1], 
                        alpha=0.5, c='red', s=20)
        demo_axes[2].set_title('Target Distribution μ₁\n(t = 1)', fontsize=12, fontweight='bold')
        demo_axes[2].set_xlim(-4, 4)
        demo_axes[2].set_ylim(-2, 2)
        demo_axes[2].set_aspect('equal')
        demo_axes[2].grid(True, alpha=0.3)

        fig_concept.suptitle('The Schrödinger Bridge Problem: Optimal Stochastic Transport', 
                             fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return mo.center(mo.as_html(fig_concept))

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 1.3 Distribution of Paths, Not a Single Path!

    A crucial distinction: the SB finds a **distribution over paths**, not just one path.

    | Concept | What it is | Output |
    |---------|-----------|--------|
    | **Optimal Transport** | Deterministic coupling | One path per particle |
    | **Schrödinger Bridge** | Stochastic process | Distribution of paths per particle |

    For each particle starting at $x_0$, there's a **probability distribution** 
    over all possible trajectories it might take to reach the target. This is 
    fundamentally different from OT, which assigns a single deterministic path.

    **Why does this matter?**

    1. **Uncertainty quantification**: SB naturally handles noise and uncertainty
    2. **Generative modeling**: Sample diverse trajectories, not just one
    3. **Physical realism**: Real particles diffuse; they don't follow rails
    4. **Regularization**: The entropy term prevents overfitting to data
    """
    )
    return


@app.cell
def _(mo):
    # def comp_gif(
    #     outpath: str = "ot_vs_sb_paths_improved.gif",
    #     *,
    #     seed: int = 7,
    #     T: float = 1.0,
    #     n_steps: int = 80,
    #     n_particles: int = 25,
    #     n_sb_samples: int = 15,  # multiple SB realizations per particle
    #     sigma: float = 0.7,
    #     fps: int = 30,
    #     width_px: int = 900,
    # ) -> mo.Html:
    #     """
    #     Enhanced OT vs SB comparison emphasizing distribution of paths.
    #     Returns mo.Html that embeds the GIF inline in marimo.
    #     """
    #     from pathlib import Path
    #     import base64
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from matplotlib.animation import FuncAnimation, PillowWriter

    #     def _gif_html(path: str | Path, width: int = 900) -> mo.Html:
    #         p = Path(path)
    #         b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    #         return mo.Html(
    #             f'<img src="data:image/gif;base64,{b64}" '
    #             f'style="max-width:{width}px; width:100%; height:auto; border-radius:8px;" />'
    #         )

    #     rng = np.random.default_rng(seed)
    #     dt = T / n_steps
    #     ts = np.linspace(0.0, T, n_steps + 1)

    #     # Endpoint clouds
    #     x0_mean = np.array([-1.2, 0.0])
    #     x1_mean = np.array([+1.2, 0.0])
    #     x0_cov = 0.06 * np.eye(2)
    #     x1_cov = 0.06 * np.eye(2)

    #     x0 = rng.multivariate_normal(x0_mean, x0_cov, size=n_particles)
    #     x1 = rng.multivariate_normal(x1_mean, x1_cov, size=n_particles)

    #     # ===== OT: Single deterministic path per particle =====
    #     det_paths = (1 - ts[:, None, None]) * x0[None, :, :] + ts[:, None, None] * x1[None, :, :]

    #     # ===== SB: MULTIPLE stochastic realizations per particle =====
    #     sb_paths_multi = np.zeros((n_steps + 1, n_particles, n_sb_samples, 2), dtype=float)
    #     eps = 2e-3

    #     # Vectorized over samples for each particle (faster than inner sample loop)
    #     for i in range(n_particles):
    #         # initial: repeat x0[i] across samples
    #         sb_paths_multi[0, i, :, :] = x0[i][None, :]

    #         for k in range(n_steps):
    #             t = ts[k]
    #             X = sb_paths_multi[k, i, :, :]                       # (n_sb_samples, 2)
    #             drift = (x1[i][None, :] - X) / (T - t + eps)         # (n_sb_samples, 2)
    #             dW = rng.normal(0.0, np.sqrt(dt), size=X.shape)       # (n_sb_samples, 2)
    #             sb_paths_multi[k + 1, i, :, :] = X + drift * dt + sigma * dW

    #         sb_paths_multi[-1, i, :, :] = x1[i][None, :]  # anchor endpoint

    #     # Plot bounds
    #     all_xy = np.vstack([det_paths.reshape(-1, 2), sb_paths_multi.reshape(-1, 2)])
    #     xmin, ymin = all_xy.min(axis=0) - 0.4
    #     xmax, ymax = all_xy.max(axis=0) + 0.4

    #     # ===== FIGURE =====
    #     fig, (ax_det, ax_sb) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    #     for ax in (ax_det, ax_sb):
    #         ax.set_xlim(xmin, xmax)
    #         ax.set_ylim(ymin, ymax)
    #         ax.set_aspect("equal", adjustable="box")
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         for spine in ax.spines.values():
    #             spine.set_visible(False)

    #     ax_det.set_title(
    #         "Optimal Transport (OT)\n" + r"$x(t)=(1-t)x_0+tx_1$ — ONE path per particle",
    #         fontsize=11, weight="bold", pad=12,
    #     )
    #     ax_sb.set_title(
    #         "Schrödinger Bridge (SB)\n" + r"$dX_t=\mu(X_t,t)\,dt+\sigma\,dW_t$ — DISTRIBUTION of paths",
    #         fontsize=11, weight="bold", pad=12,
    #     )

    #     # ===== TRAJECTORIES (static background) =====
    #     highlight_idx = [min(5, n_particles - 1), min(15, n_particles - 1), min(25, n_particles - 1)]
    #     highlight_idx = sorted(set(highlight_idx))

    #     # OT: one line per particle
    #     for i in range(n_particles):
    #         alpha = 0.55 if i in highlight_idx else 0.12
    #         lw = 1.6 if i in highlight_idx else 0.8
    #         ax_det.plot(det_paths[:, i, 0], det_paths[:, i, 1], lw=lw, alpha=alpha)

    #     # SB: many lines for highlighted particles, fewer for background
    #     for i in range(n_particles):
    #         if i in highlight_idx:
    #             for s in range(n_sb_samples):
    #                 ax_sb.plot(
    #                     sb_paths_multi[:, i, s, 0],
    #                     sb_paths_multi[:, i, s, 1],
    #                     lw=0.8, alpha=0.28,
    #                 )
    #         else:
    #             for s in range(0, n_sb_samples, 3):
    #                 ax_sb.plot(
    #                     sb_paths_multi[:, i, s, 0],
    #                     sb_paths_multi[:, i, s, 1],
    #                     lw=0.6, alpha=0.10,
    #                 )

    #     # Start/end markers
    #     for ax in (ax_det, ax_sb):
    #         ax.scatter(x0[:, 0], x0[:, 1], s=40, marker="o", alpha=0.45)
    #         ax.scatter(x1[:, 0], x1[:, 1], s=40, marker="X", alpha=0.45)

    #     # ===== ANIMATED ELEMENTS =====
    #     det_scatter = ax_det.scatter(det_paths[0, :, 0], det_paths[0, :, 1], s=45, alpha=0.95)

    #     sb_current = sb_paths_multi[0].reshape(-1, 2)  # (n_particles*n_sb_samples, 2)
    #     sb_scatter = ax_sb.scatter(sb_current[:, 0], sb_current[:, 1], s=18, alpha=0.55)

    #     time_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=10, weight="bold")

    #     ax_det.text(
    #         0.02, 0.98, "Deterministic\nNo uncertainty",
    #         transform=ax_det.transAxes, fontsize=9, va="top",
    #         bbox=dict(boxstyle="round", alpha=0.35),
    #     )
    #     ax_sb.text(
    #         0.02, 0.98, "Stochastic\nEntropy-regularized",
    #         transform=ax_sb.transAxes, fontsize=9, va="top",
    #         bbox=dict(boxstyle="round", alpha=0.35),
    #     )

    #     def update(frame: int):
    #         t = ts[frame]
    #         det_scatter.set_offsets(det_paths[frame])

    #         sb_current = sb_paths_multi[frame].reshape(-1, 2)
    #         sb_scatter.set_offsets(sb_current)

    #         time_text.set_text(f"Time: t/T = {t / T:.2f}")
    #         return det_scatter, sb_scatter, time_text

    #     anim = FuncAnimation(fig, update, frames=n_steps + 1, interval=35, blit=False, repeat=True)

    #     outpath_p = Path(outpath)
    #     anim.save(outpath_p, writer=PillowWriter(fps=fps))
    #     plt.close(fig)

    #     # Return the inline GIF for marimo
    #     return _gif_html(outpath_p, width=width_px)


    # # IMPORTANT: last expression displays in marimo
    # comp_gif(
    #     outpath="ot_vs_sb_improved.gif",
    #     n_particles=10,
    #     n_sb_samples=15,
    #     sigma=0.7,
    #     fps=10,
    # )

    mo.image(src="assets/ot_vs_sb.gif", width=900)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 1.4 Discrete vs Continuous Schrödinger Bridges

    SB problems come in two flavors:

    | Aspect | **Discrete SB** | **Continuous SB** |
    |--------|-----------------|-------------------|
    | Time | Finite steps: t ∈ {0, 1, ..., N} | Continuous: t ∈ [0, 1] |
    | State space | Often discrete (graphs, tokens) | Continuous (ℝᵈ) |
    | Reference | Markov chain | SDE (e.g., Brownian motion) |
    | Solution | Sinkhorn iterations | PDEs / Neural networks |
    | Computation | Matrix operations | SDE integration |

    #### When to Use Discrete SB

    - **Combinatorial problems**: Matching on graphs, permutations
    - **Language models**: Token-to-token transport
    - **Small state spaces**: When you can enumerate all states
    - **Exact solutions needed**: Sinkhorn converges to exact solution

    #### When to Use Continuous SB (This Library!)

    - **Physical systems**: Particles, molecules, fluids
    - **Images**: Pixel space is continuous
    - **Generative modeling**: Sampling in high-dimensional continuous spaces
    - **Scientific computing**: Trajectory inference, dynamics learning
    - **Scalability**: Continuous methods scale better to high dimensions

    **This library focuses on continuous SBs** because they're more relevant for 
    machine learning applications like generative modeling, image translation, 
    and scientific simulation.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 1.5 Solving for the Distribution of Paths

    How do we actually find this distribution of paths? The key insight is that we do **not** represent all paths explicitly. Instead, we characterize the path distribution by its **drift function** $b^*(x,t)$, i.e., the deterministic trend in the SDE

    $$
    dX_t = b^*(X_t,t)\,dt + \sigma(X_t,t)\,dW_t.
    $$

    The drift answers: *“If a particle is at position $x$ at time $t$, what is its instantaneous average direction of motion?”*  Formally,

    $$
    \mathbb{E}[\,dX_t \mid X_t=x\,] = b^*(x,t)\,dt.
    $$

    **Forward vs. inverse is crucial here.** In the **forward** problem, the drift $b$ and diffusion $\sigma$ are **given**, and we solve for the induced process $\{X_t\}$ (and its law). In this library, we are typically solving the **inverse** problem: we are given observations/constraints/endpoint distributions, and we **solve for (learn)** a drift $b^*(x,t)$ whose induced SDE produces the desired distribution of paths.

    Once we have the drift, we can:

    1. **Sample paths**: Integrate the SDE with the learned drift  
    2. **Compute densities**: Solve the Fokker–Planck equation  
    3. **Evaluate likelihoods**: For probabilistic inference  

    The different solvers in this library are different ways to **infer** this drift function $b^*(x,t)$ (and, by extension, the path distribution it induces).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Mathematical Foundations

    ### 2.1 Stochastic Differential Equations (SDEs): A Primer

    Before diving into SB, let's understand SDEs - the language of continuous stochastic processes.

    An **SDE** describes how a random variable $X_t$ evolves over time:

    $$dX_t = b(X_t, t) \, dt + \sigma(X_t, t) \, dW_t$$

    | Term | Name | Meaning |
    |------|------|---------|
    | $dX_t$ | Infinitesimal change | How much X changes in tiny time dt |
    | $b(X_t, t)$ | **Drift** | Deterministic "pull" direction |
    | $\sigma(X_t, t)$ | **Diffusion** | Magnitude of random fluctuations |
    | $dW_t$ | **Wiener process** | Gaussian noise with $\mathbb{E}[dW_t] = 0$, $\mathbb{E}[dW_t^2] = dt$ |

    **Intuition**: At each moment, the particle gets a deterministic push (drift) plus 
    a random kick (diffusion × noise).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 2.2 The Optimal Drift: Why This Form?

    The optimal process satisfies the SDE:

    $$dX_t = b^*(X_t, t) \, dt + \sigma(t) \, dW_t$$

    where the **optimal drift** has the form:

    $$\boxed{b^*(x,t) = b_{\text{ref}}(x,t) + \sigma^2(t) \nabla \log \psi(x,t)}$$

    This is arguably the most important equation in this tutorial. Let's unpack it.

    ---

    #### Component 1: Reference Drift $b_{\text{ref}}(x,t)$

    This is what the particles would do "naturally" under the reference process.

    - For Brownian motion: $b_{\text{ref}} = 0$ (no preferred direction)
    - For OU process: $b_{\text{ref}} = -\theta(x - \mu)$ (pull toward μ)

    ---

    #### Component 2: Score Term $\nabla \log \psi(x,t)$

    This is the **score function** of the backward potential $\psi$. It points in the 
    direction of increasing $\psi$, i.e., toward regions where particles need to go.

    ---

    #### Component 3: The $\sigma^2$ Factor

    Why multiply by σ²? **Dimensional analysis**:

    - Drift has units: [space / time]
    - Score has units: [1 / space]  
    - Diffusion σ has units: [space / √time]
    - Therefore σ² has units: [space² / time]
    - σ² × score has units: [space² / time] × [1 / space] = [space / time] ✓
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. Brownian Bridges: The Foundation

    ### 3.1 What is a Brownian Bridge?

    A **Brownian bridge** is Brownian motion **conditioned on its endpoints**. If we
    know a particle starts at $x_0$ and must end at $x_1$ at a fixed final time, what
    does its random path look like *in between*?

    This is the simplest instance of a **bridge problem**: impose endpoint
    constraints and study the resulting distribution over paths.

    ---

    ### Why Brownian Bridges Matter for “Bridge” Problems

    Brownian bridges are important because they show, in the cleanest possible
    setting, the two core ideas behind more general bridge constructions (including
    Schrödinger bridges):

    1. **Endpoint constraints induce a drift.**  
       Unconditioned Brownian motion has zero drift, but conditioning on an endpoint
       forces the process to acquire a time-inhomogeneous drift that *pulls* it to
       the required final value.

    2. **They provide the Gaussian reference case.**  
       For Brownian motion, the entire bridge law is tractable. The Schrödinger
       Bridge problem can be viewed as a generalization: instead of only pinning
       endpoints (or using simple Gaussian structure), we match more general endpoint
       distributions while staying as close as possible (in a precise information
       sense) to a chosen reference diffusion. When the reference is Brownian motion,
       the Brownian bridge is the baseline “conditioned reference process.”

    ---

    ### The Math (Final time = 1)

    At any intermediate time $t\in[0,1]$, the conditional distribution is Gaussian:

    $$
    X_t \mid (X_0=x_0,\; X_1=x_1)
    \sim \mathcal{N}\!\left((1-t)x_0 + t x_1,\;\sigma^2\, t(1-t)\right).
    $$

    - The mean $(1-t)x_0 + t x_1$ is **linear interpolation** between endpoints.
    - The variance $\sigma^2 t(1-t)$ is a **parabola**: zero at endpoints, maximal at
      $t=\tfrac12$.

    **Main takeaway:** the bridge interpolates linearly *in the mean*, while its
    uncertainty peaks in the middle and collapses at the endpoints.

    ---

    ### Gaussian Process and Closed-Form Structure

    **Gaussian:** Yes. Brownian motion is a Gaussian process, and conditioning a
    Gaussian object on linear constraints preserves Gaussianity. Therefore the
    Brownian bridge is also a **Gaussian process**.

    A useful constructive identity is:

    $$
    X_t = (1-t)x_0 + t x_1 + \sigma\,(W_t - t W_1),
    $$

    which makes Gaussianity and the endpoint behavior immediate.

    **Closed form:** Yes. You get closed-form expressions for:
    - **Marginals** (the Gaussian above),
    - **Covariance**
    - 
      $$
      \mathrm{Cov}(X_s,X_t)=\sigma^2\big(\min(s,t)-st\big),
      $$

    - **And the induced drift (SDE form)**.

    ---

    ### The Induced Drift (Key “Bridge Mechanism”)

    A bridge is not “just Brownian motion plus conditioning” conceptually; it is
    equivalently an SDE with a modified drift that enforces the endpoint:

    $$
    dX_t = \frac{x_1 - X_t}{1-t}\,dt + \sigma\,dW_t, \qquad t<1.
    $$

    The drift term $\frac{x_1 - X_t}{1-t}$ is a **pull toward the endpoint** $x_1$,
    and it becomes stronger as $t\to 1$ to ensure the process hits $x_1$.

    This is the simplest example of a broader theme: **constraints change the drift**.
    Schrödinger bridges generalize this idea to more complex endpoint constraints and
    more general reference diffusions—often requiring us to solve an inverse problem
    to find the drift that induces the desired path distribution.

    ---

    ### (Optional) General Final Time $T$

    If the final time is $T$ instead of $1$, then for $t\in[0,T]$:

    $$
    \mathbb E[X_t] = \left(1-\frac{t}{T}\right)x_0 + \frac{t}{T}x_1,
    \qquad
    \mathrm{Var}(X_t)=\sigma^2\, t\left(1-\frac{t}{T}\right),
    $$

    and the drift becomes:

    $$
    dX_t = \frac{x_1 - X_t}{T-t}\,dt + \sigma\,dW_t, \qquad t<T.
    $$
    """
    )
    return


@app.cell
def _(jax, jnp, mo, np, plt):
    def _():
        # BROWNIAN BRIDGE VISUALIZATION using the library's implementation
        from schrodinger_bridge.integrators import sample_brownian_bridge
        from schrodinger_bridge import TimeGrid

        # Set up bridge parameters
        bridge_key = jax.random.PRNGKey(42)
        time_grid_bridge = TimeGrid(t0=0.0, t1=1.0, num_steps=100)
        sigma_bridge = 0.4

        # Single start and end point
        x0_bridge = jnp.array([[-2.0, 0.0]])  # [1, 2]
        x1_bridge = jnp.array([[2.0, 0.0]])   # [1, 2]

        # Tile to get multiple paths from same endpoints
        n_paths = 50
        x0_tiled = jnp.tile(x0_bridge, (n_paths, 1))
        x1_tiled = jnp.tile(x1_bridge, (n_paths, 1))

        # Sample bridges using the library function
        bridge_result = sample_brownian_bridge(
            key=bridge_key,
            x0=x0_tiled,
            x1=x1_tiled,
            time_grid=time_grid_bridge,
            sigma=sigma_bridge,
        )

        # Visualize
        fig_bridge, (bridge_ax1, bridge_ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Sample paths
        for bridge_i in range(n_paths):
            bridge_ax1.plot(bridge_result.paths[bridge_i, :, 0], bridge_result.paths[bridge_i, :, 1], 
                     alpha=0.3, linewidth=0.8, color='purple')

        bridge_ax1.scatter([-2], [0], c='green', s=100, marker='o', label='Start', zorder=10)
        bridge_ax1.scatter([2], [0], c='red', s=100, marker='x', label='End', zorder=10)
        bridge_ax1.set_title('Brownian Bridge Sample Paths\n(All paths share same endpoints)', 
                      fontsize=12, fontweight='bold')
        bridge_ax1.set_xlabel('x')
        bridge_ax1.set_ylabel('y')
        bridge_ax1.legend()
        bridge_ax1.set_aspect('equal')
        bridge_ax1.grid(True, alpha=0.3)

        # Right: Variance profile
        times = np.array(bridge_result.times)
        theoretical_var = sigma_bridge**2 * times * (1 - times)

        # Compute empirical variance over paths
        empirical_var = []
        for bridge_t_idx in range(len(times)):
            positions = bridge_result.paths[:, bridge_t_idx, :]  # [n_paths, 2]
            var_at_t = np.var(positions, axis=0).mean()  # Average over dimensions
            empirical_var.append(var_at_t)

        bridge_ax2.plot(times, theoretical_var, 'b-', linewidth=2, label='Theoretical σ²t(1-t)')
        bridge_ax2.plot(times, empirical_var, 'r--', linewidth=2, label='Empirical variance')
        bridge_ax2.set_xlabel('Time t', fontsize=11)
        bridge_ax2.set_ylabel('Variance', fontsize=11)
        bridge_ax2.set_title('Bridge Variance Profile\n(Parabola peaking at t=0.5)', 
                      fontsize=12, fontweight='bold')
        bridge_ax2.legend()
        bridge_ax2.grid(True, alpha=0.3)

        fig_bridge.suptitle('Brownian Bridge: Paths and Variance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return mo.center(mo.as_html(fig_bridge))

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Defining Problems with the Library

    Every Schrödinger Bridge problem requires three components:

    | Component | Library Class | Purpose |
    |-----------|---------------|---------|
    | **Source (μ₀)** | `GaussianDistribution`, `TwoMoonsDistribution`, etc. | Where particles start |
    | **Target (μ₁)** | Same classes | Where particles end |
    | **Reference** | `BrownianMotion`, `OrnsteinUhlenbeck`, etc. | The prior dynamics |

    Let's create a more interesting problem: **Gaussian → Two Moons**
    """
    )
    return


@app.cell
def _(jax, mo):
    def _():
        from schrodinger_bridge import (
            BrownianMotion,
            GaussianDistribution,
            SBProblem,
            TimeGrid,
            TwoMoonsDistribution,
        )

        # Create Gaussian-to-TwoMoons problem
        moons_problem = SBProblem(
            reference=BrownianMotion(sigma=0.5, dim=2),
            source=GaussianDistribution(dim=2),  # Standard N(0, I)
            target=TwoMoonsDistribution(noise=0.05),
            time_grid=TimeGrid(num_steps=100),
            name="Gaussian-to-TwoMoons",
        )

        print(moons_problem.summary())
        print()

        # Sample and visualize
        moons_key = jax.random.PRNGKey(123)
        moons_k1, moons_k2 = jax.random.split(moons_key)
        moons_source = moons_problem.sample_source(moons_k1, 500)
        moons_target = moons_problem.sample_target(moons_k2, 500)

        # Use library's visualization
        from schrodinger_bridge import plot_marginals

        fig_moons = plot_marginals(
            source_samples=moons_source,
            target_samples=moons_target,
            title="Gaussian → Two Moons Problem Definition",
        )

        return mo.center(mo.as_html(fig_moons)), moons_problem

    result_moons, moons_problem = _()
    result_moons
    return (moons_problem,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Solver Methods

    Our library provides **6 different solvers**, each learning a different 
    **representation of the Schrödinger potential** (or equivalently, the optimal drift).

    ### 5.0 Understanding Representations

    The optimal drift is $b^* = b_{\text{ref}} + \sigma^2 \nabla \log \psi$.

    Different solvers parameterize different parts of this equation:

    | Representation | What We Learn | Drift Reconstruction |
    |----------------|--------------|---------------------|
    | **Score** | $s(x,t) \approx \nabla \log p_t(x)$ | $b^* = b_{\text{ref}} + \sigma^2 s$ |
    | **Potential** | $h(x,t) \approx \psi(x,t)$ | $b^* = b_{\text{ref}} + \sigma^2 \nabla \log h$ |
    | **Control** | $u(x,t) \approx \sigma^2 \nabla \log \psi$ | $b^* = b_{\text{ref}} + u$ |
    | **Kernel** | $\alpha_i(t)$ coefficients | $b^* = b_{\text{ref}} + \sigma^2 \sum_i \alpha_i \nabla k$ |

    ---
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 5.1 Score-Based Solver (Neural)

    **What it learns:** The score function $s_\theta(x,t) \approx \nabla \log p_t(x)$

    **Representation → Drift:** $b^*(x,t) = b_{\text{ref}}(x,t) + \sigma^2 s_\theta(x,t)$

    ---

    **Training Objective (Denoising Score Matching):**

    We sample from the **bridge conditional distribution**:

    $$x_t = (1-t)x_0 + t \cdot x_1 + \sigma\sqrt{t(1-t)} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

    The true conditional score is:

    $$\nabla_{x_t} \log p(x_t | x_0, x_1) = -\frac{x_t - \mu_t}{\sigma^2 t(1-t)}$$

    **Pros:**
    - Works for any distributions
    - Flexible neural network architecture
    - Well-studied in diffusion model literature

    **Cons:**
    - Requires training (many iterations)
    - Can be slow to converge
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 5.2 FBSDE Solver (Neural)

    **What it learns:** The control $Z_\theta(x,t)$ and value function $Y_\theta(x,t)$

    **Representation → Drift:** $b^*(x,t) = b_{\text{ref}}(x,t) + Z_\theta(x,t)$

    ---

    **Mathematical Framework (Forward-Backward SDE):**

    Forward SDE (controlled dynamics):
    $$dX_t = \left[b_{\text{ref}} + Z_t\right] dt + \sigma \, dW_t$$

    Backward SDE (value function evolution):
    $$dY_t = -f(X_t, Z_t, t) \, dt + Z_t \cdot dW_t$$

    **Pros:**

    - Principled optimal control formulation
    - Often gives best accuracy
    - Connects to Hamilton-Jacobi-Bellman theory

    **Cons:**

    - More complex implementation
    - Slower per iteration
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 5.3 Doob h-Transform Solver (Analytical / Kernel)

    **What it learns:** The h-function where $h(x,t) \propto \psi(x,t)$

    **Representation → Drift:** $b^*(x,t) = b_{\text{ref}}(x,t) + \sigma^2 \nabla \log h(x,t)$

    ---

    ## Why Gaussians Have Closed-Form Solutions

    This is one of the most beautiful results in SB theory!

    **Setup:**
    - Source: $\mu_0 = \mathcal{N}(m_0, \Sigma_0)$
    - Target: $\mu_1 = \mathcal{N}(m_1, \Sigma_1)$  
    - Reference: Brownian motion with diffusion σ

    **Why closed-form exists:**

    1. **Brownian transition density is Gaussian**
    2. **Products of Gaussians are Gaussian**
    3. **Backward Kolmogorov preserves Gaussians**

    **The optimal drift simplifies to:**

    $$b^*(x,t) = \frac{m_1 - x}{1-t} + \text{(covariance correction terms)}$$

    **Pros:**

    - **No training needed** (closed-form for Gaussian-to-Gaussian)
    - Instant solution
    - Mathematically exact

    **Cons:**

    - Analytical only for Gaussian marginals
    - Kernel method needed for general distributions
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 5.4 RKHS Solver (Non-Neural)

    **What it learns:** Kernel expansion coefficients $\alpha_i(t)$

    **Representation → Drift:** 
    $$s(x,t) = \sum_{i=1}^{N} \alpha_i(t) \nabla_x k(x, x_i)$$

    The score is represented as a weighted sum of kernel gradients.

    **Training:** Kernel ridge regression at each time slice

    **Pros:**

    - No neural network needed
    - Closed-form solution per time step
    - Fast for small datasets

    **Cons:**

    - Scales as $O(N^3)$ with inducing points
    - May struggle in high dimensions
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 5.5 IMF Solver (Neural)

    **Iterative Markovian Fitting** - A simulation-free approach

    **What it learns:** Score function through iterative refinement

    **Key Innovation:** No need to simulate full trajectories during training!

    **Algorithm:**

    1. **Initialize:** Train flow matching from source to target (straight paths)
    2. **Iterate:** Forward pass → train forward model; Backward pass → train backward model
    3. **Convergence:** Models converge to consistent forward/backward SB dynamics

    **Pros:**

    - Simulation-free (faster iterations)
    - Good for large-scale problems
    - Stable training

    **Cons:**

    - Requires multiple IMF iterations
    - More complex than direct score matching
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 5.6 IPF Solver (Neural)

    **Iterative Proportional Fitting** (Sinkhorn-style for continuous setting)

    **What it learns:** Forward and backward drift corrections

    **Algorithm:**

    1. Start with initial forward/backward models
    2. **Iterate:**

       - Sample backward trajectories from target using current backward model
       - Train forward model to match these trajectories
       - Sample forward trajectories from source using current forward model
       - Train backward model to match these trajectories

    **Pros:**

    - Classic, well-understood algorithm
    - Theoretical convergence guarantees

    **Cons:**

    - Slow (requires many iterations with simulation)
    - Each step needs full trajectory simulation
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. Solver Comparison

    ### 6.1 Summary Table

    | Solver | Learns | Neural? | Speed | Accuracy | Best For |
    |--------|--------|---------|-------|----------|----------|
    | **Score-Based** | ∇log p (score) | ✅ | Medium | Good | General purpose |
    | **FBSDE** | Control u | ✅ | Slow | **Best** | High accuracy |
    | **Doob** | h-function | ❌ | **Fast** | Exact* | Gaussian marginals |
    | **RKHS** | Kernel coefficients | ❌ | Fast | Good | Small data, low-dim |
    | **IMF** | Score (iteratively) | ✅ | Medium | Good | Large-scale |
    | **IPF** | Forward/backward | ✅ | Slow | Good | Theoretical work |

    *Exact for Gaussian-to-Gaussian; approximate otherwise

    ---

    ### 6.2 Decision Flowchart

    ```
    Is your problem Gaussian-to-Gaussian?
    ├── YES → DoobHTransformSolver (analytical, instant)
    └── NO → Do you have tractable densities?
            ├── YES → DoobHTransformSolver (kernel mode)
            └── NO → Need highest accuracy?
                    ├── YES → ScoreBasedSolver or FBSDESolver
                    └── NO → Avoid neural networks?
                            ├── YES → RKHSSolver
                            └── NO → IMFSolver or IPFSolver
    ```
    """
    )
    return


@app.cell
def _(mo):
    def _():
        import jax
        import jax.numpy as jnp
        from schrodinger_bridge import (
            BrownianMotion,
            GaussianDistribution,
            SBProblem,
            TimeGrid,
            plot_trajectories,
        )
        from schrodinger_bridge.solvers import DoobHTransformSolver

        # Define problem
        gaussian_problem = SBProblem(
            reference=BrownianMotion(sigma=0.5, dim=2),
            source=GaussianDistribution(mean=jnp.array([-2.0, 0.0]), cov=0.3, dim=2),
            target=GaussianDistribution(mean=jnp.array([2.0, 0.0]), cov=0.3, dim=2),
            time_grid=TimeGrid(num_steps=50),
            name="Gaussian Translation",
        )

        # Create solver
        doob_solver = DoobHTransformSolver(gaussian_problem)
        doob_key = jax.random.PRNGKey(0)

        # Adaptive code - tries methods until one works
        trajectories = None
        method_used = None

        # Try solve() first (newest API)
        if hasattr(doob_solver, 'solve'):
            try:
                solution = doob_solver.solve(doob_key)
                trajectories = solution.sample_trajectories(jax.random.PRNGKey(1), num_samples=200)
                method_used = "solve() API"
            except Exception as e:
                print(f"solve() failed: {e}")

        # Fallback to train() + sample()
        if trajectories is None and hasattr(doob_solver, 'sample'):
            try:
                result = doob_solver.train(doob_key)
                trajectories = doob_solver.sample(jax.random.PRNGKey(1), num_samples=200)
                method_used = "train() + sample() API"
            except Exception as e:
                print(f"sample() failed: {e}")

        # Last resort: manual integration
        if trajectories is None:
            from schrodinger_bridge.integrators import EulerMaruyama

            result = doob_solver.train(doob_key)
            params = result['params'] if isinstance(result, dict) else result.params
            drift_fn = doob_solver.extract_drift(params)
            x0 = gaussian_problem.sample_source(jax.random.PRNGKey(1), num_samples=200)

            integrator = EulerMaruyama()
            trajectories = integrator.integrate(
                key=jax.random.PRNGKey(2),
                x0=x0,
                time_grid=gaussian_problem.time_grid,
                drift=drift_fn,
                diffusion=gaussian_problem.sigma,
                return_trajectory=True,
            )
            method_used = "manual integration"

        # Visualize
        fig_doob = plot_trajectories(
            trajectories,
            num_show=50,
            title=f"Doob h-Transform ({method_used})",
            colorby='time',
        )

        print(f"✓ Using: {method_used}")

        return (
            mo.center(mo.as_html(fig_doob)),
            DoobHTransformSolver,
            doob_solver,
            gaussian_problem,
            plot_trajectories,
            trajectories,
        )

    result_gaussian, DoobHTransformSolver, doob_solver, gaussian_problem, plot_trajectories, trajectories = _()
    result_gaussian
    return (
        DoobHTransformSolver,
        gaussian_problem,
        plot_trajectories,
        trajectories,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Marginal Schrödinger Bridge

    ### 7.1 Motivation: Why Intermediate Constraints?

    Standard SB matches marginals only at **t=0** and **t=1**. But what if we have
    observations at **intermediate times**?

    **Example scenarios:**
    - Options markets: Implied densities at multiple expiries (1M, 3M, 6M, 1Y)
    - Cell biology: Gene expression at multiple time points
    - Physics: Particle distributions at multiple snapshots

    **Marginal SB** extends the problem:

    $$
    P^* = \arg\min_{P}\ \mathrm{KL}(P \,\|\, P_{\text{ref}})
    \quad \text{s.t.} \quad P_{t_i} = \mu_i \;\; \forall i = 0,1,\ldots,K.
    $$

    **Boundary-value viewpoint (in probability space).**  
    A helpful way to think about marginal SB is as a **boundary value problem**, but
    the “boundaries” are not just endpoints in time. Instead, we prescribe the state
    of the system at *multiple* times, as **constraints in probability space**:
    each condition $P_{t_i}=\mu_i$ acts like an additional boundary condition on the
    evolution of the distribution.

    Equivalently: we are finding the **most likely path measure** (closest to
    $P_{\mathrm{ref}}$ in KL) among all path measures whose time-marginals hit the
    specified sequence $\{\mu_i\}$.

    ---

    ### 7.2 The Decomposition Trick

    A beautiful property: marginal SB **decomposes** into SB subproblems on each
    time segment.

    Intuitively, because the reference process is Markov, imposing marginals at
    $t_0<t_1<\dots<t_K$ turns the global optimization into a chain of local
    “bridging” steps between successive marginals. Each segment solves the minimal
    KL change needed to move from $\mu_{i-1}$ to $\mu_i$ over $[t_{i-1},t_i]$.

    | Segment | Source | Target | Solve |
    |---------|--------|--------|-------|
    | $[t_0,t_1]$ | $\mu_0$ | $\mu_1$ | Standard SB |
    | $[t_1,t_2]$ | $\mu_1$ | $\mu_2$ | Standard SB |
    | ... | ... | ... | ... |
    | $[t_{K-1},t_K]$ | $\mu_{K-1}$ | $\mu_K$ | Standard SB |

    The full solution is the **concatenation** of segment solutions, with continuity
    at the constraint times:
    - the marginal at $t_i$ matches $\mu_i$ by construction,
    - and the path measure on each segment is optimal relative to the reference.

    **Interpretation:** marginal SB is “piecewise entropic optimal transport” in time:
    each segment is an entropic bridge, and the intermediate marginals serve as
    additional probability-space boundary conditions that pin down the evolution at
    multiple snapshots.
    """
    )
    return


@app.cell
def _(jax, jnp, mo, np, plt):
    def _():
        # Marginal SB visualization using the library
        from schrodinger_bridge import (
            BrownianMotion,
            GaussianDistribution,
            TimeGrid,
        )
        from schrodinger_bridge.marginal_sb import (
            MarginalConstraint,
            MarginalSBProblem,
            MarginalSBSolver,
        )
        from schrodinger_bridge.integrators import sample_brownian_bridge

        np.random.seed(2024)

        # Define 3 marginal constraints at t=0, t=0.5, t=1
        marginal_0 = GaussianDistribution(mean=jnp.array([-2.0, 0.0]), cov=0.16, dim=2)
        marginal_05 = GaussianDistribution(mean=jnp.array([0.0, 1.5]), cov=0.25, dim=2)
        marginal_1 = GaussianDistribution(mean=jnp.array([2.0, 0.0]), cov=0.16, dim=2)

        # Create Marginal SB problem
        marginal_problem = MarginalSBProblem(
            reference=BrownianMotion(sigma=0.3, dim=2),
            marginals=[
                MarginalConstraint(time=0.0, distribution=marginal_0),
                MarginalConstraint(time=0.5, distribution=marginal_05),
                MarginalConstraint(time=1.0, distribution=marginal_1),
            ],
            time_grid=TimeGrid(num_steps=100),
            name="3-Marginal SB",
        )

        print(f"Marginal SB Problem: {marginal_problem.name}")
        print(f"Number of segments: {marginal_problem.num_segments}")
        print(f"Segment times: {marginal_problem.segment_times}")

        # Sample from each marginal
        key = jax.random.PRNGKey(42)
        n_samples = 200

        k1, k2, k3 = jax.random.split(key, 3)
        samples_0 = marginal_0.sample(k1, n_samples)
        samples_05 = marginal_05.sample(k2, n_samples)
        samples_1 = marginal_1.sample(k3, n_samples)

        # Simulate bridges between marginals (simplified visualization)
        sig_marg = 0.3
        n_steps = 50

        # Segment 1: t=0 to t=0.5
        idx1 = np.random.permutation(n_samples)
        traj_s1 = np.zeros((n_samples, n_steps, 2))
        times_s1 = np.linspace(0, 0.5, n_steps)
        for ti, t in enumerate(times_s1):
            t_norm = t / 0.5  # Normalize to [0, 1] for bridge
            mean = (1 - t_norm) * np.array(samples_0) + t_norm * np.array(samples_05[idx1])
            std = sig_marg * np.sqrt(t_norm * (1 - t_norm)) if 0 < t_norm < 1 else 0
            traj_s1[:, ti, :] = mean + std * np.random.randn(n_samples, 2)

        # Segment 2: t=0.5 to t=1
        idx2 = np.random.permutation(n_samples)
        traj_s2 = np.zeros((n_samples, n_steps, 2))
        times_s2 = np.linspace(0.5, 1.0, n_steps)
        for ti, t in enumerate(times_s2):
            t_norm = (t - 0.5) / 0.5  # Normalize to [0, 1] for bridge
            mean = (1 - t_norm) * np.array(samples_05) + t_norm * np.array(samples_1[idx2])
            std = sig_marg * np.sqrt(t_norm * (1 - t_norm)) if 0 < t_norm < 1 else 0
            traj_s2[:, ti, :] = mean + std * np.random.randn(n_samples, 2)

        # Combine trajectories
        full_traj = np.concatenate([traj_s1[:, :-1, :], traj_s2], axis=1)
        full_times = np.concatenate([times_s1[:-1], times_s2])

        # Visualize 5 time snapshots
        fig_marg, axes_marg = plt.subplots(1, 5, figsize=(18, 4))

        t_indices = [0, 24, 49, 73, -1]  # Corresponds to t≈0, 0.25, 0.5, 0.75, 1.0
        t_labels = ['t=0.00\n(μ₀)', 't=0.25', 't=0.50\n(μ₁ ⭐)', 't=0.75', 't=1.00\n(μ₂)']

        for ax_i, (ti, lbl) in enumerate(zip(t_indices, t_labels)):
            ax = axes_marg[ax_i]
            pts = full_traj[:, ti, :]
            colors = np.arctan2(full_traj[:, 0, 1], full_traj[:, 0, 0])
            ax.scatter(pts[:, 0], pts[:, 1], c=colors, cmap='viridis', alpha=0.6, s=15)
            if ti in [0, 49, -1]:  # Constrained times
                ax.set_facecolor('#fff9c4')
            ax.set_xlim(-4, 4)
            ax.set_ylim(-2, 3)
            ax.set_title(lbl, fontsize=11)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        fig_marg.suptitle('Marginal SB: Constrained at t=0, t=0.5, and t=1 (yellow panels)', 
                          fontsize=14, fontweight='bold')
        plt.tight_layout()
        return mo.center(mo.as_html(fig_marg))

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8. Martingale Schrödinger Bridge

    ### 8.1 The Critical Difference: No-Arbitrage

    The **Martingale SB** extends Marginal SB with a critical constraint for finance:
    the **no-arbitrage** (martingale) property.

    | Property | Marginal SB | Martingale SB |
    |----------|-------------|---------------|
    | **Marginal matching** | ✅ $P_{T_i} = \mu_i$ | ✅ $P_{T_i} = \mu_i$ |
    | **Martingale property** | ❌ Not enforced | ✅ $\mathbb{E}[S_T \mid S_t] = F(t,T)$ |
    | **No-arbitrage** | ❌ May violate | ✅ Guaranteed |
    | **Use case** | General transport | Derivatives pricing |

    ### 8.2 The Martingale Constraint

    Under the risk-neutral measure, discounted asset prices must be martingales:

    $$\boxed{\mathbb{E}^{\mathbb{Q}}[S_T \mid \mathcal{F}_t] = S_t \cdot e^{r(T-t)} = F(t,T)}$$

    **Why this matters**: Without it, arbitrage exists:
    - If $\mathbb{E}[S_T] > F(0,T)$ → Buy forward, sell synthetic → Free money!

    ### 8.3 When to Use Each

    | Scenario | Use |
    |----------|-----|
    | Options calibration for pricing | **Martingale SB** |
    | Exotic derivatives hedging | **Martingale SB** |
    | General density transport | Marginal SB |
    | Image/data generation | Marginal SB |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 8.4 Mathematical Formulation

    **Marginal SB (insufficient for finance):**
    $$P^* = \argmin_{P: P_{T_i} = \mu_{T_i} \,\forall i} \text{KL}(P \| P_{\text{ref}})$$

    **Martingale SB (correct for finance):**
    $$P^* = \argmin_{P} \text{KL}(P \| P_{\text{ref}}) \quad \text{s.t.} \quad 
    \begin{cases} 
    P_{T_i} = \mu_{T_i} & \text{(marginal constraints)} \\[4pt]
    \mathbb{E}[S_{T_j} \mid S_{T_i}] = S_{T_i} \cdot e^{r(T_j - T_i)} & \text{(martingale constraint)}
    \end{cases}$$

    ---

    ### 8.5 Comparison with Classical Methods

    | Method | Marginal Fit | No-Arbitrage | Multi-Maturity | Path Dynamics |
    |--------|:------------:|:------------:|:--------------:|:-------------:|
    | **Local Vol (Dupire)** | ✅ Exact | ⚠️ Fragile | ❌ One at a time | ✅ Yes |
    | **Stochastic Vol (Heston)** | ⚠️ Approx | ✅ Yes | ⚠️ Difficult | ✅ Yes |
    | **SABR** | ⚠️ Approx | ⚠️ Can violate | ❌ Per-maturity | ❌ Limited |
    | **Marginal SB** | ✅ Exact | ❌ **Not enforced** | ✅ Yes | ✅ Yes |
    | **Martingale SB** | ✅ Exact | ✅ **Guaranteed** | ✅ Yes | ✅ Yes |
    """
    )
    return


@app.cell
def _(mo, np, plt):
    def _():
        # Martingale SB demonstration using the library
        from schrodinger_bridge import BrownianMotion, TimeGrid
        from schrodinger_bridge.marginal_sb import MarginalConstraint, MarginalSBProblem
        from schrodinger_bridge.martingale_sb import (
            ForwardCurve,
            MartingaleSBProblem,
            MartingaleSBSolver,
        )

        np.random.seed(42)

        # Market parameters
        spot, rate, vol = 100.0, 0.05, 0.20
        expiries = [0.25, 0.5, 1.0]
        n_paths = 1000

        # Generate risk-neutral samples (log-normal)
        def sample_lognormal(S0, r, sigma, T, n):
            drift = (r - 0.5 * sigma**2) * T
            diffusion = sigma * np.sqrt(T)
            return S0 * np.exp(drift + diffusion * np.random.randn(n))

        marginal_samples = {T: sample_lognormal(spot, rate, vol, T, 3000) for T in expiries}

        # Sinkhorn assignment with optional martingale penalty
        def sinkhorn_assignment(log_start, log_end_pool, mart_penalty=0.0, fwd_ratio=1.0):
            n = len(log_start)
            C = (log_start[:, None] - log_end_pool[None, :]) ** 2
            if mart_penalty > 0:
                C += mart_penalty * (log_end_pool[None, :] - log_start[:, None] - np.log(fwd_ratio)) ** 2

            K = np.exp(-C / 0.5)
            u, v = np.ones(n), np.ones(n)
            for _ in range(100):
                u, v = 1.0 / (K @ v + 1e-10), 1.0 / (K.T @ u + 1e-10)
            P = u[:, None] * K * v[None, :]

            assignment = np.zeros(n, dtype=int)
            available = np.ones(n, dtype=bool)
            for i in np.argsort(-P.max(axis=1)):
                row = P[i].copy()
                row[~available] = -np.inf
                assignment[i] = np.argmax(row)
                available[assignment[i]] = False
            return assignment

        def project_martingale(S_start, S_end, fwd_ratio):
            """Scale S_end so E[S_end]/E[S_start] = fwd_ratio exactly."""
            return S_end * fwd_ratio / (np.mean(S_end) / np.mean(S_start))

        # Simulate paths
        def simulate_sb(use_martingale=False):
            all_t = [0.0] + expiries
            prices_at_t = {0.0: spot * np.ones(n_paths)}

            for i in range(len(expiries)):
                t0, t1 = all_t[i], all_t[i+1]
                fwd_ratio = np.exp(rate * (t1 - t0))

                S_start = prices_at_t[t0]
                target_pool = marginal_samples[t1][:n_paths].copy()

                if i == 0:
                    S_end = target_pool[np.random.permutation(n_paths)]
                else:
                    assignment = sinkhorn_assignment(
                        np.log(S_start), np.log(target_pool),
                        mart_penalty=10.0 if use_martingale else 0.0,
                        fwd_ratio=fwd_ratio
                    )
                    S_end = target_pool[assignment]

                if use_martingale:
                    S_end = project_martingale(S_start, S_end, fwd_ratio)

                prices_at_t[t1] = S_end

            # Brownian bridge interpolation
            n_steps, bridge_vol = 30, 0.10
            all_times, all_paths = [], []

            for i in range(len(expiries)):
                t0, t1 = all_t[i], all_t[i+1]
                log_start = np.log(prices_at_t[t0])
                log_end = np.log(prices_at_t[t1])

                segment_times = np.linspace(t0, t1, n_steps, endpoint=(i == len(expiries)-1))
                for t in segment_times:
                    tau = (t - t0) / (t1 - t0) if t1 > t0 else 0
                    mean_log = (1 - tau) * log_start + tau * log_end
                    std_log = bridge_vol * np.sqrt(tau * (1 - tau)) if 0 < tau < 1 else 0
                    all_times.append(t)
                    all_paths.append(np.exp(mean_log + std_log * np.random.randn(n_paths)))

            return np.array(all_times), np.column_stack(all_paths)

        # Run both methods
        times_marginal, paths_marginal = simulate_sb(use_martingale=False)
        times_martingale, paths_martingale = simulate_sb(use_martingale=True)

        # Evaluate martingale property
        def check_martingale(times, paths, expiries, rate):
            all_t = [0.0] + expiries
            errors = []
            for i in range(len(all_t) - 1):
                t0, t1 = all_t[i], all_t[i+1]
                idx0 = np.argmin(np.abs(times - t0))
                idx1 = np.argmin(np.abs(times - t1))
                actual = np.mean(paths[:, idx1]) / np.mean(paths[:, idx0])
                expected = np.exp(rate * (t1 - t0))
                errors.append(abs(actual - expected) / expected * 100)
            return errors

        err_marginal = check_martingale(times_marginal, paths_marginal, expiries, rate)
        err_martingale = check_martingale(times_martingale, paths_martingale, expiries, rate)

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Forward curve
        t_fwd = np.linspace(0, 1, 100)
        fwd_curve = spot * np.exp(rate * t_fwd)
        n_show = 50

        # Marginal SB paths
        for i in range(n_show):
            axes[0].plot(times_marginal, paths_marginal[i, :], alpha=0.1, lw=0.4, color='steelblue')
        axes[0].plot(t_fwd, fwd_curve, 'r--', lw=2.5, label='Forward F(0,t)')
        axes[0].plot(times_marginal, np.mean(paths_marginal, axis=0), 'b-', lw=2.5, label='Mean E[Sₜ]')
        axes[0].set_xlabel('Time (years)')
        axes[0].set_ylabel('Price ($)')
        axes[0].set_title('Marginal SB\n(No martingale constraint)', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(70, 140)

        # Martingale SB paths
        for i in range(n_show):
            axes[1].plot(times_martingale, paths_martingale[i, :], alpha=0.1, lw=0.4, color='forestgreen')
        axes[1].plot(t_fwd, fwd_curve, 'r--', lw=2.5, label='Forward F(0,t)')
        axes[1].plot(times_martingale, np.mean(paths_martingale, axis=0), 'darkgreen', lw=2.5, label='Mean E[Sₜ]')
        axes[1].set_xlabel('Time (years)')
        axes[1].set_ylabel('Price ($)')
        axes[1].set_title('Martingale SB\n(Arbitrage-free)', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(70, 140)

        # Martingale error comparison
        segments = ['0→0.25y', '0.25→0.5y', '0.5→1y']
        x = np.arange(len(segments))
        width = 0.35
        bars1 = axes[2].bar(x - width/2, err_marginal, width, label='Marginal SB', color='steelblue')
        bars2 = axes[2].bar(x + width/2, err_martingale, width, label='Martingale SB', color='forestgreen')
        axes[2].axhline(y=0.01, color='red', linestyle='--', lw=2, label='Arbitrage threshold')
        axes[2].set_xlabel('Segment')
        axes[2].set_ylabel('Martingale Error (%)')
        axes[2].set_title('Martingale Property Violation\n(Lower = Better)', fontsize=12, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(segments)
        axes[2].legend()
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3, axis='y')

        fig.suptitle('Martingale SB: Enforcing No-Arbitrage in Options Calibration', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        print("\n=== Martingale Property Evaluation ===")
        print(f"{'Segment':<15} {'Marginal SB':<15} {'Martingale SB':<15} {'Verdict'}")
        print("-" * 60)
        for i, seg in enumerate(segments):
            m_err, mt_err = err_marginal[i], err_martingale[i]
            verdict = "✓ Both OK" if m_err < 0.5 else "✓ Martingale SB wins"
            print(f"{seg:<15} {m_err:<15.4f}% {mt_err:<15.6f}% {verdict}")

        return mo.center(mo.as_html(fig))

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 9. Options Calibration: Real-World Application

    ### 9.1 The Fundamental Problem

    In derivatives pricing, **options** give the right to buy/sell an asset at 
    strike $K$ at maturity $T$. From option prices, we extract **implied densities**
    using the **Breeden-Litzenberger formula**:

    $$p_T(S) = e^{rT} \frac{\partial^2 C}{\partial K^2}\bigg|_{K=S}$$

    **The challenge**: We observe "snapshots" at discrete maturities $(T_1, T_2, ...)$,
    but need a **continuous process** that:

    1. ✅ Matches ALL observed marginal distributions
    2. ✅ Is **arbitrage-free** (no free money!)
    3. ✅ Provides dynamics for hedging and risk management

    ### 9.2 Why Martingale SB is Superior

    | Use Case | What Martingale SB Enables |
    |----------|---------------------------|
    | **Exotic Options Pricing** | Price path-dependent options consistently with vanillas |
    | **Hedging** | Compute Greeks consistent with the full vol surface |
    | **Risk Management** | Simulate arbitrage-free scenarios for VaR |
    | **Model Calibration** | Fit ALL maturities simultaneously |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 10. Synthetic Data Generation

    ### 10.1 Why Schrödinger Bridges for Synthetic Data?

    SB is powerful for generating **realistic intermediate data** when you only 
    observe **endpoints**. This is fundamentally different from:

    | Method | What it does | Limitation |
    |--------|-------------|------------|
    | **Linear interpolation** | Straight line between points | Ignores stochasticity |
    | **GANs/VAEs** | Learn to generate from noise | No temporal coherence |
    | **Schrödinger Bridge** | Optimal stochastic interpolation | Physically plausible paths |

    ### 10.2 Application Domains

    | Domain | Source | Target | What SB Generates |
    |--------|--------|--------|-------------------|
    | **Medical** | Healthy cells | Disease cells | Disease progression trajectories |
    | **Climate** | Winter state | Summer state | Seasonal transitions |
    | **Finance** | Current prices | Future scenarios | Market evolution paths |
    | **Biology** | Stem cells | Differentiated | Cell fate trajectories |
    """
    )
    return


@app.cell
def _(mo, np, plt):
    def _():
        # Comprehensive Synthetic Data Generation with MULTIPLE intermediate states
        from schrodinger_bridge import BrownianMotion, TimeGrid
        from schrodinger_bridge.integrators import sample_brownian_bridge

        np.random.seed(2024)

        # Parameters
        sigma_syn = 0.25
        n_samples = 600

        # Source: 4 Gaussian clusters (representing initial cell types)
        def sample_gaussian_cluster(n, centers, std=0.12):
            n_per = n // len(centers)
            pts = []
            for cx, cy in centers:
                pts.append(np.random.randn(n_per, 2) * std + np.array([cx, cy]))
            return np.vstack(pts)

        # Target: Spiral (representing evolved states)
        def sample_spiral(n, noise=0.03):
            t = np.linspace(0, 3*np.pi, n)
            r = t / (3*np.pi)
            x = r * np.cos(t) + np.random.randn(n) * noise
            y = r * np.sin(t) + np.random.randn(n) * noise
            return np.stack([x, y], axis=1)

        # Generate source and target
        source_centers = [(-0.6, -0.6), (-0.6, 0.6), (0.6, -0.6), (0.6, 0.6)]
        source_syn = sample_gaussian_cluster(n_samples, source_centers, std=0.10)
        target_syn = sample_spiral(n_samples, noise=0.02)

        # Random pairing for SB
        idx_syn = np.random.permutation(n_samples)
        paired_target = target_syn[idx_syn]

        # Generate trajectories with Brownian bridge interpolation
        n_time_steps = 100
        times_syn = np.linspace(0, 1, n_time_steps)
        traj_syn = np.zeros((n_samples, n_time_steps, 2))

        for ti, t in enumerate(times_syn):
            if t == 0:
                traj_syn[:, ti] = source_syn
            elif t == 1:
                traj_syn[:, ti] = paired_target
            else:
                mean = (1 - t) * source_syn + t * paired_target
                std = sigma_syn * np.sqrt(t * (1 - t))
                traj_syn[:, ti] = mean + std * np.random.randn(n_samples, 2)

        # Visualization: 8 time points showing clear intermediate states
        fig_syn, axes_syn = plt.subplots(2, 4, figsize=(16, 8))
        axes_flat = axes_syn.flatten()

        # Show 8 time points
        time_indices = [0, 14, 28, 42, 57, 71, 85, 99]
        time_labels = ['t=0.00', 't=0.14', 't=0.28', 't=0.42', 
                       't=0.57', 't=0.71', 't=0.85', 't=1.00']

        # Color by original cluster
        cluster_colors = np.zeros(n_samples)
        n_per_cluster = n_samples // 4
        for ci in range(4):
            cluster_colors[ci*n_per_cluster:(ci+1)*n_per_cluster] = ci

        for ax_i, (ti, lbl) in enumerate(zip(time_indices, time_labels)):
            ax = axes_flat[ax_i]
            pts = traj_syn[:, ti, :]

            scatter = ax.scatter(pts[:, 0], pts[:, 1], c=cluster_colors, 
                                 cmap='Set1', alpha=0.5, s=12)
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(lbl, fontsize=12, fontweight='bold')

            # Add annotations for endpoints
            if ti == 0:
                ax.annotate('4 Gaussian\nclusters', xy=(0, -1.1), fontsize=9, 
                           ha='center', style='italic')
            elif ti == 99:
                ax.annotate('Spiral\nstructure', xy=(0, -1.1), fontsize=9, 
                           ha='center', style='italic')

        fig_syn.suptitle(f'Synthetic Data Generation: 4 Clusters → Spiral\n'
                         f'(n={n_samples} samples, σ={sigma_syn}, showing 8 intermediate states)', 
                         fontsize=14, fontweight='bold')
        plt.tight_layout()
        return mo.center(mo.as_html(fig_syn))

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 10.3 Cell Trajectory Inference Example

    A powerful application is **inferring cellular trajectories** from single-cell RNA-seq data.

    **The Problem:**
    - We observe cells at **two time points** (e.g., day 0 and day 7)
    - Each cell is measured destructively (can't track the same cell over time!)
    - We want to understand **how cells transition** between states

    **SB Solution:**
    - Source distribution: Gene expression at day 0
    - Target distribution: Gene expression at day 7
    - SB generates **plausible trajectories** showing how cells might have evolved
    """
    )
    return


@app.cell
def _(mo, np, plt):
    def _():
        # Cell trajectory simulation with multiple intermediate states
        np.random.seed(42)

        sigma_cell = 0.25
        n_cells = 500

        # Day 0: Stem cells (tight cluster)
        stem_cells = np.random.randn(n_cells, 2) * 0.15 + np.array([0, 0])

        # Day 7: Differentiated into 3 types (multi-modal)
        n_type = n_cells // 3
        type_a = np.random.randn(n_type, 2) * 0.12 + np.array([-1.0, 0.8])   # Type A
        type_b = np.random.randn(n_type, 2) * 0.12 + np.array([1.0, 0.8])    # Type B  
        type_c = np.random.randn(n_cells - 2*n_type, 2) * 0.12 + np.array([0, -1.0])  # Type C
        differentiated = np.vstack([type_a, type_b, type_c])

        # Assign cells to fate (for coloring)
        fate_labels = np.concatenate([np.zeros(n_type), np.ones(n_type), 
                                      2*np.ones(n_cells - 2*n_type)])
        np.random.shuffle(fate_labels)

        # Shuffle target to match source
        idx_cell = np.random.permutation(n_cells)
        paired_diff = differentiated[idx_cell]
        fate_colors = fate_labels[idx_cell]

        # Generate trajectories
        n_steps_cell = 70
        times_cell = np.linspace(0, 1, n_steps_cell)
        traj_cell = np.zeros((n_cells, n_steps_cell, 2))

        for ti, t in enumerate(times_cell):
            if t == 0:
                traj_cell[:, ti] = stem_cells
            elif t == 1:
                traj_cell[:, ti] = paired_diff
            else:
                m = (1 - t) * stem_cells + t * paired_diff
                s = sigma_cell * np.sqrt(t * (1 - t))
                traj_cell[:, ti] = m + s * np.random.randn(n_cells, 2)

        # Plot: Show 7 time points for cell trajectory inference
        fig_cell, axes_cell = plt.subplots(1, 7, figsize=(21, 4))

        cell_time_idx = [0, 11, 23, 35, 46, 58, 69]
        cell_time_labels = ['Day 0\n(Stem)', 'Day 1', 'Day 2', 'Day 3.5', 
                            'Day 5', 'Day 6', 'Day 7\n(Diff.)']

        cmap_cell = plt.cm.Set1
        colors_cell = cmap_cell(fate_colors.astype(int) / 2)

        for ax_i, (ti, lbl) in enumerate(zip(cell_time_idx, cell_time_labels)):
            ax = axes_cell[ax_i]
            pts = traj_cell[:, ti, :]

            ax.scatter(pts[:, 0], pts[:, 1], c=colors_cell, alpha=0.5, s=12)
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.6, 1.4)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(lbl, fontsize=11, fontweight='bold')
            ax.set_xlabel('Gene 1', fontsize=9)
            if ax_i == 0:
                ax.set_ylabel('Gene 2', fontsize=9)

            # Add labels for cell types at day 7
            if ti == 69:
                ax.annotate('A', xy=(-1.0, 1.0), fontsize=10, ha='center', 
                           color=cmap_cell(0), fontweight='bold')
                ax.annotate('B', xy=(1.0, 1.0), fontsize=10, ha='center', 
                           color=cmap_cell(0.5), fontweight='bold')
                ax.annotate('C', xy=(0, -1.2), fontsize=10, ha='center', 
                           color=cmap_cell(1), fontweight='bold')

        fig_cell.suptitle('Cell Trajectory Inference: Stem Cells → 3 Differentiated Types\n'
                          '(Colors show eventual fate; SB infers 7 intermediate states)', 
                          fontsize=13, fontweight='bold')
        plt.tight_layout()
        return mo.center(mo.as_html(fig_cell))

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 10.4 Key Takeaways for Synthetic Data Generation

    | Aspect | Recommendation |
    |--------|---------------|
    | **σ selection** | Start with 0.2-0.4; increase for more diversity |
    | **Sample size** | Generate 5-10x more synthetic than real data |
    | **Validation** | Check that synthetic marginals match real marginals |
    | **Use case** | Data augmentation, missing data, scenario analysis |

    **When NOT to use SB for synthetic data:**
    - When you have dense temporal observations (just interpolate!)
    - When distribution matching isn't critical
    - When you need exact replication (use GANs/VAEs instead)

    **When SB shines:**
    - Sparse endpoint observations
    - Need for physically plausible paths
    - Multi-modal target distributions
    - Temporal coherence is important
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 11. Visualization & Diagnostics

    ### 11.1 Key Metrics

    | Metric | What It Measures | Good Value |
    |--------|------------------|------------|
    | **MMD² (source)** | Distance from source marginal | < 0.01 |
    | **MMD² (target)** | Distance from target marginal | < 0.01 |
    | **Mass conservation** | Total probability over time | = 1.0 |
    | **Path regularity** | Maximum velocity | < 100 |

    ### 11.2 Maximum Mean Discrepancy (MMD)

    MMD is a kernel-based distance between distributions:

    $$\text{MMD}^2(P, Q) = \mathbb{E}[k(X, X')] + \mathbb{E}[k(Y, Y')] - 2\mathbb{E}[k(X, Y)]$$

    It's zero if and only if P = Q (for characteristic kernels like Gaussian).
    """
    )
    return


@app.cell
def _(gaussian_problem, jax, mo, plt, trajectories):
    def _():
        from schrodinger_bridge import InvariantChecker, mmd_squared

        # Get reference samples
        diag_key = jax.random.PRNGKey(999)
        diag_k1, diag_k2 = jax.random.split(diag_key)
        ref_source = gaussian_problem.sample_source(diag_k1, 500)
        ref_target = gaussian_problem.sample_target(diag_k2, 500)

        # Create checker and run diagnostics
        checker = InvariantChecker()
        report = checker.check_all(
            trajectories,
            ref_source,
            ref_target,
            key=diag_key,
        )

        print("=== Diagnostic Report ===")
        print(f"Source marginal MMD²: {report.marginal_error_source:.6f}")
        print(f"Target marginal MMD²: {report.marginal_error_target:.6f}")
        print(f"Mass conservation error: {max(abs(m - 1.0) for m in report.mass_conservation):.6f}")
        print(f"Mean velocity: {report.metadata.get('mean_velocity', 'N/A'):.2f}")
        print(f"Max velocity: {report.metadata.get('max_velocity', 'N/A'):.2f}")

        if report.violations:
            print(f"\nViolations ({len(report.violations)}):")
            for v in report.violations:
                print(f"  [{v.severity}] {v.name}: {v.message}")
        else:
            print("\n✓ No violations detected!")

        # Compare generated vs target
        generated_final = trajectories.paths[:, -1, :]  # Final positions

        fig_diag, diag_axes = plt.subplots(1, 2, figsize=(12, 5))

        # Overlay comparison
        diag_axes[0].scatter(ref_target[:, 0], ref_target[:, 1], alpha=0.3, c='red', 
                        s=20, label='Target')
        diag_axes[0].scatter(generated_final[:, 0], generated_final[:, 1], alpha=0.3, 
                        c='green', s=20, label='Generated')
        diag_axes[0].set_title('Final Positions vs Target')
        diag_axes[0].legend()
        diag_axes[0].set_aspect('equal')

        # Mass over time
        diag_axes[1].plot(report.mass_conservation, 'b-')
        diag_axes[1].axhline(y=1.0, color='r', linestyle='--', label='Expected')
        diag_axes[1].set_xlabel('Time step')
        diag_axes[1].set_ylabel('Mass')
        diag_axes[1].set_title('Mass Conservation')
        diag_axes[1].legend()

        plt.tight_layout()
        return mo.center(mo.as_html(fig_diag)), InvariantChecker

    result_diag, InvariantChecker = _()
    result_diag
    return (InvariantChecker,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 12. Complete Example: Gaussian → Two Moons

    Let's put it all together with a more challenging problem using the 
    **kernel-based Doob solver**.
    """
    )
    return


@app.cell
def _(DoobHTransformSolver, jax, mo, moons_problem, plot_trajectories):
    """
    Corrected marimo cell for Gaussian → Two Moons transport.

    The DoobHTransformSolver provides an analytical drift function, not a sample() method.
    To generate trajectories, we need to:
    1. Train the solver to get parameters
    2. Extract the drift function
    3. Use an integrator (EulerMaruyama) to sample paths
    """

    def gif_func():
        from schrodinger_bridge.solvers import DoobConfig
        from schrodinger_bridge import create_transport_gif
        from schrodinger_bridge.integrators import EulerMaruyama

        # Configure kernel-based Doob solver for non-Gaussian target (Two Moons)
        moons_solver = DoobHTransformSolver(
            moons_problem,
            doob_config=DoobConfig(
                method='kernel',  # or 'kernel_sinkhorn' for better coupling
                num_inducing_points=500,
            ),
        )

        # Train (initializes kernel samples - no iteration needed for Doob)
        moons_train_key = jax.random.PRNGKey(42)
        moons_result = moons_solver.train(moons_train_key)

        # Extract the drift function from trained solver
        drift_fn = moons_solver.extract_drift(moons_result['params'])

        # Get diffusion coefficient from problem reference
        # The reference.diffusion returns a scalar or array
        def diffusion_fn(x, t):
            return moons_problem.reference.diffusion(x, t)

        # Sample initial points from source distribution
        moons_sample_key = jax.random.PRNGKey(100)
        k1, k2 = jax.random.split(moons_sample_key)
        x0 = moons_problem.sample_source(k1, num_samples=500)

        # Integrate SDE to generate trajectories
        integrator = EulerMaruyama()
        moons_trajectories = integrator.integrate(
            key=k2,
            x0=x0,
            time_grid=moons_problem.time_grid,
            drift=drift_fn,
            diffusion=diffusion_fn,
            return_trajectory=True,
        )

        # Visualize
        fig_moons_traj = plot_trajectories(
            moons_trajectories,
            num_show=100,
            title="Gaussian → Two Moons Transport (Doob Kernel Method)",
            colorby='time',
        )

        return mo.center(mo.as_html(fig_moons_traj)), create_transport_gif, moons_trajectories


    result_doob, create_transport_gif, moons_trajectories = gif_func()
    result_doob
    return create_transport_gif, moons_trajectories


@app.cell
def _(InvariantChecker, jax, mo, moons_problem, moons_trajectories, plt):
    def _(moons_trajectories):
        # Run diagnostics
        moons_diag_key = jax.random.PRNGKey(888)
        moons_diag_k1, moons_diag_k2 = jax.random.split(moons_diag_key)

        moons_ref_source = moons_problem.sample_source(moons_diag_k1, 500)
        moons_ref_target = moons_problem.sample_target(moons_diag_k2, 500)

        moons_checker = InvariantChecker()
        moons_report = moons_checker.check_all(
            moons_trajectories,
            moons_ref_source,
            moons_ref_target,
        )

        print("=== Two Moons Diagnostics ===")
        print(f"Source MMD²: {moons_report.marginal_error_source:.6f}")
        print(f"Target MMD²: {moons_report.marginal_error_target:.6f}")

        # Visualize marginal quality
        moons_final = moons_trajectories.paths[:, -1, :]

        fig_moons_quality, moons_axes = plt.subplots(1, 3, figsize=(15, 4))

        moons_axes[0].scatter(moons_ref_source[:, 0], moons_ref_source[:, 1], 
                        alpha=0.5, c='blue', s=10)
        moons_axes[0].set_title('Source (Gaussian)')
        moons_axes[0].set_aspect('equal')

        moons_axes[1].scatter(moons_ref_target[:, 0], moons_ref_target[:, 1], 
                        alpha=0.3, c='red', s=10, label='Target')
        moons_axes[1].scatter(moons_final[:, 0], moons_final[:, 1], 
                        alpha=0.3, c='green', s=10, label='Generated')
        moons_axes[1].set_title(f'Generated vs Target\nMMD² = {moons_report.marginal_error_target:.4f}')
        moons_axes[1].legend()
        moons_axes[1].set_aspect('equal')

        moons_axes[2].scatter(moons_ref_target[:, 0], moons_ref_target[:, 1], 
                        alpha=0.5, c='red', s=10)
        moons_axes[2].set_title('Target (Two Moons)')
        moons_axes[2].set_aspect('equal')

        plt.tight_layout()
        return mo.center(mo.as_html(fig_moons_quality)), moons_ref_source, moons_ref_target

    result_moon_ref, moons_ref_source, moons_ref_target = _(moons_trajectories)
    result_moon_ref
    return moons_ref_source, moons_ref_target


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 12.1 Creating Animations

    The library can generate publication-quality GIFs showing the transport evolution.
    """
    )
    return


@app.cell
def _(
    create_transport_gif,
    mo,
    moons_ref_source,
    moons_ref_target,
    moons_trajectories,
):
    def _():
        # Create animated GIF
        gif_path = create_transport_gif(
            moons_trajectories,
            source_samples=moons_ref_source,
            target_samples=moons_ref_target,
            save_path="/tmp/moons_transport.gif",
            title="Schrödinger Bridge: Gaussian → Two Moons",
        )

        mo.md(f"GIF saved to: `{gif_path}`")

        # Display if in notebook
        try:
            return mo.image(gif_path)
        except:
            return mo.md("(GIF display requires marimo image support)")

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 13. Summary: Solver Selection Guide

    | Solver | Method | Best For | Training | Accuracy |
    |--------|--------|----------|----------|----------|
    | **DoobHTransform** (analytical) | Closed-form OT | Gaussian-to-Gaussian | None | Exact |
    | **DoobHTransform** (kernel) | Bridge kernels | General, small data | None | Good |
    | **ScoreBasedSolver** | Neural score matching | High-dim, complex | Yes | Excellent |
    | **FBSDESolver** | Forward-backward SDE | Continuous control | Yes | Excellent |
    | **RKHSSolver** | Kernel regression | Moderate dim, no NN | Minimal | Good |
    | **IMFSolver** | Iterative Markovian | Large scale | Yes | Good |
    | **IPFSolver** | Iterative proportional | Discrete compatible | Yes | Good |

    ### When to Use Each:

    1. **Gaussian marginals?** → `DoobHTransformSolver` (analytical)
    2. **Small dataset, quick result?** → `DoobHTransformSolver` (kernel) or `RKHSSolver`
    3. **High accuracy, complex distributions?** → `ScoreBasedSolver` or `FBSDESolver`
    4. **Scale is the priority?** → `IMFSolver`
    5. **Need martingale constraint?** → `MartingaleSBSolver`

    ---

    ## Key Mathematical Takeaways

    1. **The SB drift formula**: $b^*(x,t) = b_{\text{ref}} + \sigma^2 \nabla \log h(x,t)$

    2. **Bridge variance**: $\sigma^2 t(1-t)$ — zero at endpoints, max at midpoint

    3. **MMD for evaluation**: $\text{MMD}^2 \to 0$ means perfect marginal matching

    4. **Marginal SB**: Add intermediate constraints; decomposes into segments

    5. **Martingale SB**: Add no-arbitrage constraint for finance applications

    ---

    **Happy bridging!**
    """
    )
    return


if __name__ == "__main__":
    app.run()
