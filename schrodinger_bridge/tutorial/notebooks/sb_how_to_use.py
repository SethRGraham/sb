# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "jax",
#     "jaxlib", 
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Schrödinger Bridge Library: Implementation Guide

    **A Practical Tutorial for Using the `schrodinger_bridge` Package**

    This guide covers everything you need to know to use the library in your own code:

    1. **Installation & Setup** — Dependencies and device configuration
    2. **Architecture: The Three Layers** — How the library is structured (start here!)
    3. **Problem Definition** — Setting up source, target, and reference dynamics
    4. **Choosing a Solver** — When to use each of the 6 solver methods
    5. **Training & Inference** — Complete workflow from definition to sampling
    6. **Neural Network Approaches** — NetworkFactory, custom architectures, ICNNs, OTT-JAX
    7. **Visualization & GIFs** — Creating publication-quality animations
    8. **Marginal SB** — Multi-time-point constraints
    9. **Advanced Topics** — Custom distributions, integrators, diagnostics

    ---
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. Installation & Setup

    ### 1.1 Dependencies

    The library is built on **pure JAX** with minimal dependencies:

    ```bash
    # Core (required)
    pip install jax jaxlib numpy

    # Visualization (recommended)
    pip install matplotlib pillow

    # OTT-JAX integration (optional, for optimal transport coupling)
    pip install ott-jax

    # Install the library itself
    pip install schrodinger-bridge
    # or from source:
    # pip install -e .
    ```

    ### 1.2 GPU/TPU Setup

    JAX automatically detects accelerators. To verify your setup:

    ```python
    from schrodinger_bridge import print_device_info

    print_device_info()
    # Output:
    # JAX Device Configuration
    #   Backend: gpu
    #   Device type: GPU
    #   Device count: 1
    #   Platform: cuda
    ```

    ### 1.3 Basic Imports

    ```python
    import jax
    import jax.numpy as jnp

    # Core problem definition
    from schrodinger_bridge import (
        SBProblem,
        BrownianMotion,
        GaussianDistribution,
        TwoMoonsDistribution,
        TimeGrid,
    )

    # Solvers
    from schrodinger_bridge.solvers import (
        ScoreBasedSolver,
        FBSDESolver,
        DoobHTransformSolver,
        RKHSSolver,
        IMFSolver,
        IPFSolver,
    )

    # Visualization
    from schrodinger_bridge import (
        plot_marginals,
        plot_trajectories,
        create_transport_gif,
    )

    # Network Factories (for custom architectures — Section 6)
    from schrodinger_bridge.network_factory import (
        NetworkFactory,    # ABC for subclassing
        MLPFactory,        # default (backward compatible)
        UNetFactory,       # 2D/3D spatial data
        TransformerFactory,# sequence / multichannel data
        CustomFactory,     # escape hatch for loose functions
        sanity_check,      # pre-training validation
    )
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Architecture: The Two Layers

    Before diving into the API, it helps to understand how the library is organized.
    Everything rests on a single mathematical result, and the code is layered so that
    each level adds one new constraint on top of the level below.

    ---
    """
    )
    return


@app.cell
def _(mo):

    diagram = r"""
    flowchart TD
      L1["Layer 1 — Base SBSolver (6 interchangeable methods)<br/>
      <b>solves</b>: min KL(P ‖ P_ref) s.t. P₀ = μ₀, P₁ = μ₁<br/>
      <b>output</b>: corrected drift  b*(x,t) = b_ref + σ² f(x,t)"]

      L2["Layer 2 — MarginalSBSolver (multi-time-point)<br/>
      <b>adds</b>: intermediate marginal constraints at t₁, t₂, …<br/>
      <b>decomposes</b>: into pairwise segments, each solved by a Layer-1 solver"]

      L1 --> L2
    """

    mo.mermaid(diagram).center()

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 2.1 Layer 1 — The Base Solver and the Universal Drift Equation

    Every solver in the library solves the same variational problem:

    $$P^* \;=\; \arg\min_{P}\; \text{KL}(P \,\|\, P_{\text{ref}}) \qquad \text{subject to} \quad P_0 = \mu_0,\;\; P_1 = \mu_1$$

    The **Schrödinger Bridge** $P^*$ is the stochastic process closest (in relative entropy)
    to a reference process $P_{\text{ref}}$, while matching prescribed distributions at the
    endpoints.

    The solution is always an SDE whose drift decomposes into two parts:

    $$\boxed{\; b^*(x,t) \;=\; \underbrace{b_{\text{ref}}(x,t)}_{\text{reference drift}} \;+\; \sigma^2(t) \cdot \underbrace{f(x,t)}_{\text{learned correction}} \;}$$

    This is the single most important equation in the library.  Every solver learns a
    different representation of $f(x,t)$, but the decomposition is always the same:
    the optimal process is the reference process **plus** a correction scaled by the
    diffusion coefficient $\sigma^2$.

    > **Bottom Line:** The SB solution never replaces the reference dynamics — it
    > *corrects* them.  The correction $f$ steers probability mass from $\mu_0$ to $\mu_1$
    > while staying as close to the reference as possible.  Think of it as the
    > minimum-energy steering law.

    #### What the code looks like

    ```
    schrodinger_bridge/
    ├── core/
    │   ├── types.py        # SolverResult, TrajectoryBatch, DiagnosticReport
    │   ├── problem.py      # SBProblem  (source + target + reference + time grid)
    │   └── invariants.py   # InvariantChecker, MMD, mass conservation tests
    └── solvers/
        ├── base.py         # SBSolver (abstract), SBSolution
        ├── score_based.py  # ScoreBasedSolver    — learns ∇ log pₜ
        ├── fbsde.py        # FBSDESolver         — learns control Z(x,t)
        ├── ipf.py          # IPFSolver           — Sinkhorn / IPFP iterations
        ├── imf.py          # IMFSolver           — simulation-free fitting
        ├── doob.py         # DoobHTransformSolver— learns (or computes) h(x,t)
        └── rkhs.py         # RKHSSolver          — kernel weights, no neural net
    ```

    The base class `SBSolver` defines three methods every solver must implement:

    ```python
    class SBSolver:
        def train(self, key, config) -> SolverResult:   ...  # fit the correction f
        def sample(self, key, num_samples) -> TrajectoryBatch:  ...  # simulate paths
        def solve(self, key, config) -> SBSolution:     ...  # reusable solution object
    ```

    `SBSolution` wraps the learned parameters and exposes `get_forward_drift()`,
    which returns the corrected drift $b^*$ as a callable `drift_fn(x, t)`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### The six representations of f(x,t)

    All six solvers find the **same** optimal process — they differ only in how they
    parameterize the correction term $f$.  The table below maps each solver to its
    mathematical object, training procedure, and the resulting drift formula.

    | Solver | What it learns | How it trains | Drift $b^*(x,t)$ |
    |--------|---------------|---------------|-------------------|
    | **Doob** | h-function $h(x,t)$ | Analytical (Gaussian) or kernel regression | $b_{\text{ref}} + \sigma^2 \nabla\!\log h$ |
    | **Score-Based** | Score $\nabla\!\log p_t(x)$ | Denoising score matching on bridge paths | $b_{\text{ref}} + \sigma^2 \nabla\!\log p_t$ |
    | **FBSDE** | Control $Z(x,t)$ | Coupled forward-backward SDE optimization | $b_{\text{ref}} + \sigma^2 Z$ |
    | **RKHS** | Kernel weights $\alpha_i$ | Closed-form regression (no gradient descent) | $b_{\text{ref}} + \sigma^2 \sum_i \alpha_i \nabla k(x, x_i)$ |
    | **IMF** | Velocity $v(x,t)$ | Iterative fitting, simulation-free | $v_{\text{forward}}(x,t)$ |
    | **IPF** | Drift correction | Alternating half-bridge projections (Sinkhorn) | $b_{\text{ref}} + \sigma^2\,\text{correction}$ |

    A deeper look at each:

    **Doob h-transform** — The h-function satisfies the backward Kolmogorov PDE:

    $$\frac{\partial h}{\partial t} + b_{\text{ref}} \cdot \nabla h + \tfrac{\sigma^2}{2}\,\Delta h = 0, \qquad h(x, 1) = p_{\mu_1}(x)$$

    For Gaussian-to-Gaussian problems, $h$ has a **closed-form** solution and training
    is instantaneous.  For general targets, the solver uses kernel density estimation
    at the terminal time and propagates backward.

    **Score-Based** — Learns $s_\theta(x,t) \approx \nabla\!\log p_t(x)$ via denoising
    score matching.  Training samples Brownian bridge paths
    $x_0 \to x_1$, adds noise at time $t$, and regresses against the score of the
    noisy conditional.  This is the workhorse for general problems.

    **FBSDE** — Frames the SB as a stochastic optimal control problem.  The forward
    SDE (state) and backward SDE (adjoint / value function) are solved jointly:

    $$dX_t = b^*\,dt + \sigma\,dW_t, \qquad dY_t = -Z_t\,dW_t$$

    The control $Z$ is the gradient of the value function.  Unique benefit: you also
    get the value function $Y(x,t)$, which no other solver provides.

    **RKHS** — Expands $f$ in a reproducing kernel Hilbert space:
    $f(x,t) = \sum_i \alpha_i(t)\,\nabla k(x, x_i)$ where $k$ is a Gaussian or
    Matérn kernel.  Coefficients $\alpha_i$ are found by closed-form least-squares
    at each time slice — no neural network, no gradient descent.

    **IMF (Iterative Markovian Fitting)** — Iteratively fits a velocity field
    $v(x,t)$ by alternating between forward simulation and backward projection.
    "Simulation-free" means each iteration only requires sample pairs, not full
    path rollouts.  Uses OT coupling (`use_ot_coupling=True`) for warm-starting.

    **IPF (Iterative Proportional Fitting / Sinkhorn)** — The classical approach.
    Alternates "half-bridge" projections: project onto the source marginal constraint,
    then onto the target marginal constraint.  Each half-step solves a conditioned
    bridge problem.  Convergence is guaranteed by Csiszár's theorem on alternating
    KL projections.  Slowest solver, but most interpretable — each iteration
    monotonically reduces KL divergence to the true solution.
    """
    )
    return


@app.cell
def _(mo):
    _sec_2_2 = mo.md(
        r"""
    ### 2.2 Layer 2 — Multi-Marginal: MarginalSBSolver

    Layer 1 handles two-endpoint problems ($t=0$ and $t=1$).  Many applications
    require matching distributions at **intermediate times** as well — for example,
    constraining the process to pass through a specific distribution at $t=0.5$.

    The multi-marginal Schrödinger Bridge extends the objective:

    $$P^* = \arg\min_{P}\; \text{KL}(P \,\|\, P_{\text{ref}}) \qquad \text{subject to} \quad P_{t_i} = \mu_i \;\;\forall\, i \in \{0, 1, \ldots, n\}$$

    #### How MarginalSBSolver decomposes the problem

    Rather than solving one giant constrained optimization, `MarginalSBSolver`
    **decomposes** the multi-marginal problem into pairwise segments:

    ```
    Marginals:    μ₀ ──────── μ₁ ──────── μ₂ ──────── μ₃
    Times:        t=0.0       t=0.3       t=0.7       t=1.0
                     │           │           │
                     ▼           ▼           ▼
    Segments:   [Segment 0]  [Segment 1]  [Segment 2]
                 μ₀ → μ₁      μ₁ → μ₂      μ₂ → μ₃
                 (Layer 1)    (Layer 1)    (Layer 1)
    ```

    Each segment is a standard two-endpoint SB — solved by **any** Layer 1 solver.
    The `segment_solver_type` config controls which one:

    ```python
    # Use Doob for each segment (fastest)
    config = MarginalSBConfig(segment_solver_type='doob')

    # Use score matching for each segment (most accurate)
    config = MarginalSBConfig(segment_solver_type='score')
    ```

    The `coupling_method` controls how endpoints are handed off between segments
    (e.g., `'sequential'` passes the terminal samples of segment $k$ as the
    source samples of segment $k+1$).

    #### Code structure

    ```
    schrodinger_bridge/
    └── marginal_sb.py    # MarginalSBProblem, MarginalSBSolver, MarginalSBConfig
    ```

    The key classes:

    ```python
    # Define the multi-marginal problem
    problem = MarginalSBProblem(
        reference=BrownianMotion(sigma=0.3, dim=2),
        marginals=[
            MarginalConstraint(time=0.0, distribution=source_dist),
            MarginalConstraint(time=0.5, distribution=mid_dist),    # intermediate!
            MarginalConstraint(time=1.0, distribution=target_dist),
        ],
    )
    # problem.num_segments == 2  (one per adjacent pair)

    # Solve — internally creates 2 Layer-1 solvers
    solver = MarginalSBSolver(problem, MarginalSBConfig(segment_solver_type='score'))
    solver.train(key)

    # Sample full trajectory (stitched across segments)
    trajectories = solver.sample(key, num_samples=500)

    # Verify marginals are matched at all times
    mmd_results = solver.check_marginal_consistency(key)
    ```
    """
    )

    _sec_2_4 = mo.md(
        r"""
    ### 2.3 How the Layers Compose

    The two layers form a strict hierarchy — each one adds exactly one new constraint:

    | Layer | Class | Constraints | What's new |
    |-------|-------|-------------|------------|
    | 1 | `SBSolver` (6 variants) | $P_0 = \mu_0,\; P_1 = \mu_1$ | Base two-endpoint bridge |
    | 2 | `MarginalSBSolver` | $P_{t_i} = \mu_i \;\forall\,i$ | Intermediate time constraints |

    Importantly, **you can use any layer independently**:

    - Doing generative modeling (Gaussian → TwoMoons)?  Use **Layer 1** directly.
    - Modeling a process that must pass through 5 waypoints?  Use **Layer 2**.

    And within each layer, you choose the solver that fits your problem:

    ```python
    # Layer 1 — pick any of the 6 solvers
    solver = ScoreBasedSolver(problem)              # neural, general purpose
    solver = DoobHTransformSolver(problem)           # instant for Gaussians
    solver = RKHSSolver(problem)                     # no neural net needed

    # Layer 2 — pick a Layer-1 solver for the segments
    solver = MarginalSBSolver(problem, MarginalSBConfig(segment_solver_type='doob'))
    solver = MarginalSBSolver(problem, MarginalSBConfig(segment_solver_type='score'))
    ```

    The rest of this tutorial focuses on the **API details** of each component,
    starting with how to define problems (Section 3) and how to choose among the
    Layer 1 solvers (Section 4).
    """
    )

    mo.output.replace(mo.accordion(
        {
            "2.2  Layer 2 — Multi-Marginal: MarginalSBSolver": _sec_2_2,
            "2.3  How the Layers Compose": _sec_2_4,
        }
    ))

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. Problem Definition

    Every Schrödinger Bridge problem requires three components:

    | Component | What It Is | Examples |
    |-----------|-----------|----------|
    | **Source ($\mu_0$)** | Initial distribution | Gaussian, data samples |
    | **Target ($\mu_1$)** | Final distribution | TwoMoons, SwissRoll, data |
    | **Reference** | Prior stochastic process | Brownian motion, OU process |

    ### 3.1 Built-in Distributions

    ```python
    from schrodinger_bridge import (
        GaussianDistribution,
        TwoMoonsDistribution,
        SwissRollDistribution,
        MixtureDistribution,
    )

    # Standard Gaussian (mean=0, cov=I)
    source = GaussianDistribution(dim=2)

    # Gaussian with custom parameters
    source = GaussianDistribution(
        mean=jnp.array([1.0, -1.0]),
        cov=jnp.array([[0.5, 0.1], [0.1, 0.3]]),
        dim=2,
    )

    # Two moons (classic ML benchmark)
    target = TwoMoonsDistribution(noise=0.05, offset=0.5)

    # Swiss roll
    target = SwissRollDistribution(noise=0.1, project_2d=True)

    # Mixture of Gaussians
    target = MixtureDistribution(
        components=[
            GaussianDistribution(mean=jnp.array([-1, 0]), cov=0.1, dim=2),
            GaussianDistribution(mean=jnp.array([1, 0]), cov=0.1, dim=2),
        ],
        weights=jnp.array([0.5, 0.5]),
    )
    ```

    ### 3.2 Reference Dynamics

    ```python
    from schrodinger_bridge import (
        BrownianMotion,
        OrnsteinUhlenbeck,
        VarianceExploding,
        VariancePreserving,
    )

    # Standard Brownian motion: dX = σ dW
    ref = BrownianMotion(sigma=0.5, dim=2)

    # Ornstein-Uhlenbeck: dX = -θ(X - μ) dt + σ dW
    ref = OrnsteinUhlenbeck(theta=1.0, mu=jnp.zeros(2), sigma=0.5, dim=2)

    # Variance Exploding (for diffusion models)
    ref = VarianceExploding(sigma_min=0.01, sigma_max=50.0, dim=2)

    # Variance Preserving (DDPM-style)
    ref = VariancePreserving(beta_min=0.1, beta_max=20.0, dim=2)
    ```

    ### 3.3 Putting It Together: SBProblem

    ```python
    from schrodinger_bridge import SBProblem, TimeGrid

    problem = SBProblem(
        reference=BrownianMotion(sigma=0.5, dim=2),
        source=GaussianDistribution(dim=2),
        target=TwoMoonsDistribution(noise=0.05),
        time_grid=TimeGrid(t0=0.0, t1=1.0, num_steps=100),
        name="Gaussian-to-TwoMoons",
    )

    # Print summary
    print(problem.summary())
    # Gaussian-to-TwoMoons
    # Dimension: 2
    # Reference: BrownianMotion
    # Source: GaussianDistribution
    # Target: TwoMoonsDistribution
    # Time: [0.0, 1.0]
    # Steps: 100
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 3.4 Custom Distributions

    For your own data, subclass `MarginalDistribution`:

    ```python
    from schrodinger_bridge import MarginalDistribution
    import jax
    import jax.numpy as jnp

    class DataDistribution(MarginalDistribution):
        '''Distribution from empirical data samples.'''

        def __init__(self, data: jnp.ndarray):
            self.data = data
            self._dim = data.shape[1]

        @property
        def dim(self) -> int:
            return self._dim

        def sample(self, key, num_samples: int) -> jnp.ndarray:
            '''Sample with replacement from data.'''
            indices = jax.random.choice(key, len(self.data), shape=(num_samples,))
            return self.data[indices]

        # Optional: implement log_prob for some solvers
        def log_prob(self, x):
            # Could use KDE or leave as NotImplementedError
            raise NotImplementedError("Use kernel-based log_prob estimation")

    # Usage
    my_data = jnp.array([[1, 2], [3, 4], [5, 6], ...])  # Your data
    source = DataDistribution(my_data)
    ```

    For distributions with tractable densities:

    ```python
    class CustomDensity(MarginalDistribution):
        '''Distribution with known density.'''

        @property
        def dim(self) -> int:
            return 2

        @property
        def has_density(self) -> bool:
            return True

        def sample(self, key, num_samples):
            # Implement sampling (rejection, MCMC, etc.)
            ...

        def log_prob(self, x):
            # Return log p(x) - enables more solver options
            return -0.5 * jnp.sum(x ** 2, axis=-1)  # Example: Gaussian
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Choosing a Solver

    The library provides **6 distinct solver methods**. Here's when to use each:

    ### 4.1 Decision Flowchart

    ```
    Is your problem Gaussian-to-Gaussian?
    ├── YES → Use DoobHTransformSolver (analytical, instant)
    └── NO → Do you have tractable densities?
            ├── YES → Use DoobHTransformSolver (kernel mode)
            └── NO → Do you need highest accuracy?
                    ├── YES → Use FBSDESolver or ScoreBasedSolver
                    └── NO → Do you want to avoid neural networks?
                            ├── YES → Use RKHSSolver
                            └── NO → Is scalability the priority?
                                    ├── YES → Use IMFSolver
                                    └── NO → Use IPFSolver for interpretability
    ```

    ### 4.2 Solver Comparison Table

    | Solver | Neural? | Speed | Accuracy | Best For |
    |--------|---------|-------|----------|----------|
    | **Doob (analytical)** | ❌ | ⚡⚡⚡ | ⭐⭐⭐ | Gaussian problems |
    | **Doob (kernel)** | ❌ | ⚡⚡ | ⭐⭐ | Quick prototyping |
    | **RKHS** | ❌ | ⚡⚡ | ⭐⭐ | No-training-needed |
    | **Score-Based** | ✅ | ⚡ | ⭐⭐⭐ | General purpose |
    | **FBSDE** | ✅ | ⚡ | ⭐⭐⭐ | Optimal control view |
    | **IMF** | ✅ | ⚡⚡ | ⭐⭐ | Large-scale, simulation-free |
    | **IPF** | ✅ | 🐢 | ⭐⭐⭐ | Classical, interpretable |

    ### 4.3 Mathematical Representations

    Each solver uses a different internal representation:

    | Solver | Learns | Drift Formula |
    |--------|--------|---------------|
    | Doob | h-function | b*(x,t) = b_ref + σ² ∇log h(x,t) |
    | Score-Based | Score ∇log p_t | b*(x,t) = b_ref + σ² ∇log p_t(x) |
    | FBSDE | Control Z(x,t) | b*(x,t) = b_ref + σ² Z(x,t) |
    | RKHS | Kernel weights | b*(x,t) = b_ref + σ² Σᵢ αᵢ ∇k(x, xᵢ) |
    | IMF | Forward velocity | b*(x,t) = v_forward(x,t) |
    | IPF | Drift correction | b*(x,t) = b_ref + σ² correction(x,t) |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Training & Inference

    ### 5.1 Complete Workflow

    ```python
    import jax
    from schrodinger_bridge import SBProblem, BrownianMotion, GaussianDistribution, TwoMoonsDistribution
    from schrodinger_bridge.solvers import ScoreBasedSolver, ScoreBasedConfig
    from schrodinger_bridge import TrainingConfig

    # 1. Define problem
    problem = SBProblem(
        reference=BrownianMotion(sigma=0.5, dim=2),
        source=GaussianDistribution(dim=2),
        target=TwoMoonsDistribution(noise=0.05),
    )

    # 2. Create solver with config
    config = ScoreBasedConfig(
        hidden_dims=(256, 256, 256),
        learning_rate=1e-4,
        use_bridge_matching=True,
    )
    solver = ScoreBasedSolver(problem, config=config)

    # 3. Train
    key = jax.random.PRNGKey(42)
    training_config = TrainingConfig(
        num_iterations=5000,
        batch_size=256,
        eval_every=500,
    )
    result = solver.train(key, training_config)

    # 4. Check convergence
    print(f"Final loss: {result.final_loss:.6f}")
    print(f"Converged: {result.converged}")
    print(result.diagnostics.summary())

    # 5. Sample trajectories
    key, sample_key = jax.random.split(key)
    trajectories = solver.sample(sample_key, num_samples=500)

    # trajectories.paths: [500, 101, 2] - 500 samples, 101 time steps, 2D
    # trajectories.times: [101] - time points
    # trajectories.source_samples: [500, 2] - samples at t=0
    # trajectories.target_samples: [500, 2] - samples at t=1
    ```

    ### 5.2 Using the Solution Object

    ```python
    # Alternative: get a reusable solution object
    solution = solver.solve(key, training_config)

    # Sample from solution
    new_trajectories = solution.sample_trajectories(key, num_samples=1000)

    # Sample just the endpoint (t=1)
    endpoints = solution.sample_endpoint(key, num_samples=1000)

    # Sample at a specific time
    samples_at_t05 = solution.evaluate_at_time(key, t=0.5, num_samples=1000)

    # Get the learned drift function
    drift_fn = solution.get_forward_drift()
    # drift_fn(x, t) returns the drift at position x, time t
    ```
    """
    )
    return


@app.cell
def _(mo):
    _sec_5_3 = mo.md(
        r"""
    ### 5.3 Solver-Specific Examples

    #### Doob h-Transform (Fastest for Gaussian)

    ```python
    from schrodinger_bridge.solvers import DoobHTransformSolver, DoobConfig

    # For Gaussian-to-Gaussian, this is INSTANT (analytical solution)
    problem = SBProblem(
        reference=BrownianMotion(sigma=0.5, dim=2),
        source=GaussianDistribution(mean=jnp.array([-2, 0]), cov=0.3, dim=2),
        target=GaussianDistribution(mean=jnp.array([2, 0]), cov=0.3, dim=2),
    )

    config = DoobConfig(method='auto')  # Auto-detects analytical case
    solver = DoobHTransformSolver(problem, config=config)

    result = solver.train(key)  # Instant - no training needed!
    print(f"Method used: {result.metadata['method']}")  # 'analytical'

    # For non-Gaussian, falls back to kernel method
    config = DoobConfig(
        method='kernel',
        kernel_bandwidth=None,  # Auto via median heuristic
        num_inducing_points=500,
    )
    ```

    #### RKHS Solver (No Neural Networks)

    ```python
    from schrodinger_bridge.solvers import RKHSSolver, RKHSConfig

    config = RKHSConfig(
        bandwidth=None,  # Auto
        regularization=1e-4,
        num_inducing=500,
        num_time_points=20,
    )
    solver = RKHSSolver(problem, rkhs_config=config)

    result = solver.train(key)
    # No gradient descent - closed-form at each time slice!
    ```

    #### FBSDE Solver (Optimal Control Perspective)

    ```python
    from schrodinger_bridge.solvers import FBSDESolver, FBSDEConfig

    config = FBSDEConfig(
        hidden_dims=(256, 256),
        learning_rate=1e-4,
        terminal_weight=1.0,
        running_weight=0.01,
        method='deep_bsde',  # or 'soc' for stochastic optimal control
    )
    solver = FBSDESolver(problem, fbsde_config=config)

    result = solver.train(key, TrainingConfig(num_iterations=10000))

    # Get the full FBSDE solution (including value function)
    fbsde_solution = solver.solve_fbsde(key, result.params, x0_samples)
    # fbsde_solution.X: forward process
    # fbsde_solution.Y: value function
    # fbsde_solution.Z: optimal control
    ```

    #### IMF Solver (Simulation-Free)

    ```python
    from schrodinger_bridge.solvers import IMFSolver, IMFConfig

    config = IMFConfig(
        hidden_dims=(256, 256, 256),
        num_imf_iterations=5,
        steps_per_iteration=2000,
        use_ot_coupling=True,  # Use OT for better sample pairing
    )
    solver = IMFSolver(problem, imf_config=config)

    result = solver.train(key)
    ```
    """
    )

    mo.output.replace(mo.accordion(
        {"5.3  Solver-Specific Examples (Doob, RKHS, FBSDE, IMF)": _sec_5_3}
    ))

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. Neural Network Approaches

    Every neural solver learns a correction $f_\theta(x,t)$ that plugs into the universal drift:

    $$b^*(x,t) = b_{\text{ref}}(x,t) + \sigma^2(t) \cdot f_\theta(x,t)$$

    The **NetworkFactory** protocol decouples *what the solver needs* (the function $f$)
    from *how the network is built* (MLP, U-Net, Transformer, your own design).  The
    factory has exactly two methods:

    ```python
    class NetworkFactory(ABC):
        def init(self, key, input_dim, output_dim) -> params   # JAX pytree of arrays
        def forward(self, params, x, t) -> output              # [batch, output_dim]
    ```

    **Contract:** `x` is `[batch, input_dim]`, `t` is `[batch]`, output is
    `[batch, output_dim]`. The solver sets both dimensions — don't assume they're
    equal (FBSDE's value network Y has `output_dim=1`).

    > **Bottom Line:** $f : \mathbb{R}^d \times [0,1] \to \mathbb{R}^{d'}$ is just a
    > function.  Whether it's parameterized by an MLP, U-Net, or Transformer is invisible
    > to the solver.  JAX's pytree autodiff gives you `jax.grad(loss)(params)` for free
    > regardless of architecture.

    Each solver uses the factory through the same three-line pattern:

    ```python
    params = self._factory.init(key, dim, output_dim)   # once at setup
    output = self._factory.forward(params, x_t, t)      # every training step
    grads  = jax.grad(loss_fn)(params)                   # automatic
    ```

    **Critical JAX rule:** params must contain *only* differentiable arrays — never
    Python ints or tuples.  Config metadata (spatial shapes, token counts) lives on
    `self`, weights live in `params`.

    #### How each solver interprets the output

    | Solver | $f(x,t)$ represents | `output_dim` |
    |--------|---------------------|--------------|
    | **ScoreBasedSolver** | $\nabla\!\log p_t(x)$ (score) | $D$ |
    | **FBSDESolver** (Z) | $Z(x,t)$ (stochastic control) | $D$ |
    | **FBSDESolver** (Y) | $Y(x,t)$ (value function) | $1$ |
    | **IMFSolver** | $v(x,t)$ (velocity field) | $D$ |
    | **IPFSolver** | drift correction | $D$ |

    #### Built-in factories

    | Factory | Best for | What it does internally |
    |---------|----------|------------------------|
    | `MLPFactory` | General purpose (default) | Time-conditioned MLP with sinusoidal embedding |
    | `UNetFactory` | 2D/3D spatial data (images, volumes) | Reshapes flat $\to$ spatial, encoder-decoder with skips, flattens back |
    | `TransformerFactory` | Multichannel / sequence data | Treats input as tokens with self-attention + time conditioning |
    | `CustomFactory` | Quick experiments | Wraps loose `init_fn` / `forward_fn` pairs |

    **Default behavior is unchanged.** If you don't pass a `network_factory`, every
    solver creates an `MLPFactory` from its `hidden_dims` config.  All existing code
    works without modification.

    See the **6.1 Factory Examples** accordion below for complete working examples of
    every factory type.
    """
    )
    return


@app.cell
def _(mo):
    _sec_6_default = mo.md(
        r"""
    #### Default — Nothing Changes

    If you never pass a `network_factory`, the solver internally creates an `MLPFactory`
    from the `hidden_dims` and `time_embed_dim` in its config.  These two calls are
    **exactly equivalent**:

    ```python
    from schrodinger_bridge import (
        SBProblem, BrownianMotion, GaussianDistribution,
        TwoMoonsDistribution, TimeGrid,
    )
    from schrodinger_bridge.solvers import ScoreBasedSolver, ScoreBasedConfig

    problem = SBProblem(
        reference=BrownianMotion(sigma=0.4, dim=2),
        source=GaussianDistribution(dim=2),
        target=TwoMoonsDistribution(noise=0.05),
        time_grid=TimeGrid(num_steps=100),
    )

    # Option A: the classic way (unchanged)
    solver = ScoreBasedSolver(problem, config=ScoreBasedConfig(hidden_dims=(128, 128)))

    # Option B: explicit factory (same result)
    from schrodinger_bridge.network_factory import MLPFactory

    solver = ScoreBasedSolver(problem, config=ScoreBasedConfig(
        network_factory=MLPFactory(hidden_dims=(128, 128)),
    ))
    ```

    Every solver that previously accepted `hidden_dims` still does.  The factory is
    purely additive — you only touch it when you want a non-MLP architecture.
    """
    )

    _sec_6_unet = mo.md(
        r"""
    #### U-Net for Image Data

    For spatial data (images, fields), a U-Net with skip connections vastly
    outperforms an MLP.  The key design: the **solver always works with flat vectors**
    `[batch, dim]`, and the **factory reshapes internally**.  Your `SBProblem`
    definition doesn't change at all.

    ```python
    from schrodinger_bridge.network_factory import UNetFactory

    dim_mnist = 28 * 28 * 1  # = 784, flattened

    problem_images = SBProblem(
        reference=BrownianMotion(sigma=0.3, dim=dim_mnist),
        source=GaussianDistribution(dim=dim_mnist),
        target=GaussianDistribution(dim=dim_mnist),  # placeholder — use real data
        time_grid=TimeGrid(num_steps=50),
    )

    solver = ScoreBasedSolver(
        problem_images,
        config=ScoreBasedConfig(
            network_factory=UNetFactory(
                spatial_shape=(28, 28, 1),   # (H, W, C) — factory asserts dim == H*W*C
                channels=(32, 64),           # feature channels at each encoder level
            ),
            learning_rate=2e-4,
        ),
    )
    ```

    What happens under the hood during `forward(params, x, t)`:

    ```
    x: [batch, 784]  →  reshape  →  [batch, 28, 28, 1]
                         ↓
                    encoder (conv + pool + time injection)
                         ↓
                    bottleneck
                         ↓
                    decoder (upsample + skip concat + conv)
                         ↓
                    [batch, 28, 28, 1]  →  flatten  →  [batch, 784]
    ```

    The factory also supports **3D volumes** — just pass a 4-tuple for `spatial_shape`.
    Keep channels small to avoid memory blowup:

    ```python
    solver_3d = ScoreBasedSolver(
        problem_3d,
        config=ScoreBasedConfig(
            network_factory=UNetFactory(
                spatial_shape=(32, 32, 32, 1),  # (D, H, W, C)
                channels=(8, 16),               # small for 3D!
            ),
        ),
    )
    ```
    """
    )

    _sec_6_transformer = mo.md(
        r"""
    #### Transformer for Multivariate State Data

    When modeling a joint distribution over multiple interacting channels, each channel
    can be represented as a **token** and self-attention captures cross-channel
    dependencies.

    ```python
    from schrodinger_bridge.network_factory import TransformerFactory

    num_channels = 10

    problem = SBProblem(
        reference=BrownianMotion(sigma=0.2, dim=num_channels),
        source=GaussianDistribution(dim=num_channels),
        target=GaussianDistribution(dim=num_channels),
        time_grid=TimeGrid(num_steps=100),
    )

    solver = ScoreBasedSolver(
        problem,
        config=ScoreBasedConfig(
            network_factory=TransformerFactory(
                token_dim=1,                # each channel is a scalar
                num_tokens=num_channels,    # 10 tokens
                num_heads=2,
                num_layers=3,
                hidden_dim=64,
            ),
            learning_rate=1e-4,
        ),
    )
    ```

    The factory reshapes `[batch, 10]` → `[batch, 10, 1]` (10 tokens of dim 1),
    runs self-attention with time conditioning, then flattens back.

    For channels with multi-dimensional features (e.g., position + velocity), increase
    `token_dim` and adjust `dim` accordingly:

    ```python
    # 10 channels x 3 features each = dim 30
    TransformerFactory(token_dim=3, num_tokens=10, ...)
    ```
    """
    )

    _sec_6_subclass = mo.md(
        r"""
    #### Custom Subclass — The Recommended Pattern

    For serious custom architectures, **subclass `NetworkFactory`** and implement
    `init()` and `forward()`.  This example uses Random Fourier Features to give the
    MLP a multi-scale frequency basis over input space.

    > **Math:** For $x \in \mathbb{R}^d$, random Fourier features compute
    > $\varphi(x) = [\sin(Bx),\, \cos(Bx)]$ where $B \in \mathbb{R}^{m \times d}$
    > has i.i.d. Gaussian entries scaled by $\sigma$.  By Bochner's theorem, the inner
    > product $\langle \varphi(x), \varphi(y) \rangle \approx k(x - y)$ approximates
    > a shift-invariant kernel.  This lets the network learn high-frequency corrections
    > that a vanilla MLP with smooth activations would struggle with.

    ```python
    from schrodinger_bridge.network_factory import NetworkFactory, sanity_check
    from schrodinger_bridge.networks import (
        init_mlp_params, mlp_forward, sinusoidal_embedding, swish,
        init_linear_params, linear_forward,
    )

    class FourierMLPFactory(NetworkFactory):

        def __init__(self, fourier_dim: int = 128, fourier_scale: float = 10.0):
            self.fourier_dim = fourier_dim
            self.fourier_scale = fourier_scale

        def init(self, key, input_dim, output_dim):
            k1, k2, k3 = jax.random.split(key, 3)
            return {
                'B': jax.random.normal(k1, (input_dim, self.fourier_dim)) * self.fourier_scale,
                'mlp': init_mlp_params(k2, [2 * self.fourier_dim + 64, 256, 256, output_dim]),
                'time_proj': init_linear_params(k3, 64, 64),
            }

        def forward(self, params, x, t):
            proj = x @ params['B']
            fourier_x = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
            t_emb = linear_forward(params['time_proj'], sinusoidal_embedding(t, 64))
            h = jnp.concatenate([fourier_x, t_emb], axis=-1)
            return mlp_forward(params['mlp'], h, swish)
    ```

    Use it with any neural solver:

    ```python
    solver = ScoreBasedSolver(
        problem,
        config=ScoreBasedConfig(
            network_factory=FourierMLPFactory(fourier_dim=128, fourier_scale=10.0),
        ),
    )
    ```

    You can **validate** a factory before training with `sanity_check`.  It verifies
    output shapes, checks for NaN/Inf, and tests single-sample edge cases:

    ```python
    key = jax.random.PRNGKey(0)
    sanity_check(FourierMLPFactory(), key, input_dim=2, output_dim=2)
    # Passes silently, or raises AssertionError with a clear message
    ```
    """
    )

    _sec_6_custom = mo.md(
        r"""
    #### CustomFactory Escape Hatch

    For quick one-off experiments where subclassing feels like overkill, wrap
    two lambdas:

    ```python
    from schrodinger_bridge.network_factory import CustomFactory

    solver = ScoreBasedSolver(
        problem,
        config=ScoreBasedConfig(
            network_factory=CustomFactory(
                init_fn=lambda key, d_in, d_out: {
                    'mlp': init_mlp_params(key, [d_in + 64, 128, 128, d_out]),
                },
                forward_fn=lambda params, x, t: mlp_forward(
                    params['mlp'],
                    jnp.concatenate([x, sinusoidal_embedding(t, 64)], axis=-1),
                    swish,
                ),
            ),
        ),
    )
    ```

    This is fine for prototyping.  For anything you'll reuse, graduate to a proper
    subclass (previous example) — it's more readable and `sanity_check` works with it.
    """
    )

    _sec_6_fbsde = mo.md(
        r"""
    #### Two-Factory Solvers (FBSDE)

    FBSDESolver is unique: it learns **two** functions with different output dimensions.
    The Z-network (control, $\mathbb{R}^d \to \mathbb{R}^d$) gets `network_factory`,
    and the Y-network (value, $\mathbb{R}^d \to \mathbb{R}^1$) gets
    `value_network_factory`.  If you only set one, the other defaults to `MLPFactory`.

    ```python
    from schrodinger_bridge.solvers import FBSDESolver, FBSDEConfig
    from schrodinger_bridge.network_factory import UNetFactory

    solver = FBSDESolver(
        problem_images,
        config=FBSDEConfig(
            # Z-network: spatial convolutions for the control
            network_factory=UNetFactory(spatial_shape=(28, 28, 1), channels=(32, 64)),
            # Y-network: scalar output, MLP is fine (default)
            # value_network_factory=MLPFactory(...)  ← optional override
        ),
    )
    ```

    The solver internally calls:

    ```python
    z_params = self._z_factory.init(key1, dim, dim)     # control: R^d → R^d
    y_params = self._y_factory.init(key2, dim, 1)        # value:   R^d → R^1
    ```

    IMFSolver and IPFSolver also use two networks (forward + backward), but both
    have the same `output_dim=D`, so a single `network_factory` config suffices
    — the solver calls `init()` twice with different keys.
    """
    )

    _sec_6_design = mo.md(
        r"""
    #### Design Rules & Gotchas

    **Rule 1 — params = arrays only.** Never put Python ints, tuples, or strings in
    the params dict.  `jax.grad` traverses every leaf and rejects non-float types.
    `jax.jit` tries to trace them and fails on concrete-value-dependent reshapes.
    Store config on `self`, weights in `params`.

    ```python
    # ✗ BAD — int in pytree breaks grad and jit
    params = {'w': jax.random.normal(key, (3, 3)), 'output_dim': 3}

    # ✓ GOOD — config on self, only arrays in params
    self._output_dim = output_dim
    params = {'w': jax.random.normal(key, (3, 3))}
    ```

    **Rule 2 — spatial reshape is the factory's job.**  The solver always passes flat
    `[batch, dim]` vectors.  If your architecture needs spatial structure, reshape in
    `forward()` and flatten before returning.  The solver, problem definition,
    integrators, and visualization code never see spatial dimensions.

    **Rule 3 — run `sanity_check` before training.**  It catches the top failure
    modes (wrong output shape, NaN params, single-sample edge cases) in milliseconds,
    before you burn 30 minutes on a training run that was doomed from the start.

    ```python
    from schrodinger_bridge.network_factory import sanity_check

    sanity_check(my_factory, key, input_dim=784, output_dim=784)
    ```

    **Rule 4 — `output_dim` is set by the solver, not by you.**  Score, control, and
    velocity networks use `output_dim=D`.  FBSDE's value network uses `output_dim=1`.
    Your factory must handle both — the simplest way is to thread `output_dim` through
    to the final layer width.
    """
    )

    mo.output.replace(mo.accordion(
        {
            "6.1  Default — Backward Compatible": _sec_6_default,
            "6.2  U-Net for Image / Spatial Data": _sec_6_unet,
            "6.3  Transformer for Multivariate State Data": _sec_6_transformer,
            "6.4  Custom Subclass (Recommended Pattern)": _sec_6_subclass,
            "6.5  CustomFactory Escape Hatch": _sec_6_custom,
            "6.6  Two-Factory Solvers (FBSDE)": _sec_6_fbsde,
            "6.7  Design Rules & Gotchas": _sec_6_design,
        }
    ))

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 6.8 Time Embedding

    Time is embedded using **sinusoidal positional encoding** (used internally by
    all built-in factories):

    ```python
    from schrodinger_bridge.networks import sinusoidal_embedding

    t = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    embedding = sinusoidal_embedding(t, dim=64)
    # Shape: [5, 64] — each time gets a 64-dimensional embedding
    ```

    ### 6.9 Input Convex Neural Networks (ICNN)

    For optimal transport applications, ICNNs ensure the potential is convex:

    ```python
    from schrodinger_bridge.networks import (
        init_icnn_params,
        icnn_forward,
        icnn_gradient,
    )

    params = init_icnn_params(key, input_dim=2, hidden_dims=(256, 256, 256))

    # Evaluate convex potential φ(x)
    x = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    potential = icnn_forward(params, x)         # [batch] — guaranteed convex in x

    # Optimal transport map T(x) = ∇φ(x)
    transport_map = icnn_gradient(params, x)    # [batch, dim]
    ```

    **Key ICNN properties:**
    - Output is **convex** in input x (by construction)
    - Gradient gives the **optimal transport map** (Brenier's theorem)
    - Uses softplus on internal weights to ensure non-negativity
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 6.10 OTT-JAX Integration

    The library integrates with [OTT-JAX](https://github.com/ott-jax/ott) for optimal transport:

    ```python
    from schrodinger_bridge.ott_integration import (
        is_ott_available,
        compute_ot_coupling,
        compute_sinkhorn_divergence,
        get_ot_paired_samples,
        OTConfig,
        OTCoupledSampler,
    )

    # Check if OTT is installed
    if is_ott_available():
        print("OTT-JAX available!")
    else:
        print("Using fallback Sinkhorn (install ott-jax for full features)")

    # Compute OT coupling between samples
    config = OTConfig(
        epsilon=0.1,  # Entropic regularization
        max_iterations=1000,
        threshold=1e-4,
    )

    source_samples = problem.sample_source(key, 500)
    target_samples = problem.sample_target(key, 500)

    coupling, info = compute_ot_coupling(source_samples, target_samples, config)
    print(f"OT cost: {info['cost']:.4f}")

    # Get OT-paired samples (better than random pairing!)
    paired_source, paired_target = get_ot_paired_samples(
        key, source_samples, target_samples, config
    )

    # Sinkhorn divergence (debiased OT cost)
    divergence = compute_sinkhorn_divergence(source_samples, target_samples, config)
    ```

    #### Using OT-Coupled Sampler for Training

    ```python
    # Create sampler that provides OT-coupled pairs
    from schrodinger_bridge.ott_integration import create_ot_coupled_sampler

    ot_sampler = create_ot_coupled_sampler(
        problem,
        key,
        num_samples=1000,
        config=OTConfig(epsilon=0.1),
    )

    # Sample paired batches for training
    source_batch, target_batch = ot_sampler.sample_pairs(key, batch_size=256)

    # This gives better initialization than random pairing!
    ```

    #### Visualizing OT Coupling

    ```python
    from schrodinger_bridge.ott_integration import visualize_coupling
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    visualize_coupling(
        coupling,
        source_samples[:100],
        target_samples[:100],
        ax=ax,
        threshold=0.01,
    )
    plt.title("Optimal Transport Coupling")
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Visualization & GIFs

    The library provides comprehensive visualization utilities.

    ### 7.1 Static Plots

    ```python
    from schrodinger_bridge import (
        plot_marginals,
        plot_trajectories,
        plot_diagnostics,
        plot_velocity_field,
        VisualizationConfig,
    )

    # Configure visualization
    config = VisualizationConfig(
        figsize=(12, 4),
        dpi=150,
        cmap='viridis',
        alpha=0.6,
        point_size=15,
        line_width=0.5,
        fps=20,
    )

    # Plot source, target, and generated samples
    fig = plot_marginals(
        source_samples=source_samples,
        target_samples=target_samples,
        generated_samples=trajectories.target_samples,
        config=config,
        title="Schrödinger Bridge: Marginals",
        save_path="marginals.png",
    )

    # Plot trajectories
    fig = plot_trajectories(
        trajectories,
        num_show=50,  # Number of trajectories to display
        colorby='time',  # or 'trajectory'
        title="Transport Paths",
        save_path="trajectories.png",
    )

    # Plot training diagnostics
    fig = plot_diagnostics(
        loss_history=result.loss_history,
        diagnostics=result.diagnostics,
        title="Training Progress",
    )
    ```

    ### 7.2 Velocity Field Visualization

    ```python
    # Get the learned drift function
    drift_fn = solver.extract_drift(result.params)

    # Plot velocity field at different times
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        fig = plot_velocity_field(
            velocity_fn=drift_fn,
            t=t,
            xlim=(-3, 3),
            ylim=(-3, 3),
            resolution=20,
            samples=trajectories.at_time(int(t * 100)),  # Overlay samples
            title=f"Velocity Field at t={t:.2f}",
            save_path=f"velocity_t{t:.2f}.png",
        )
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 7.3 Creating Animated GIFs

    The library makes it easy to create publication-quality animations:

    ```python
    from schrodinger_bridge import create_transport_gif, create_comparison_gif

    # Single solver GIF
    gif_path = create_transport_gif(
        trajectories,
        source_samples=source_samples,
        target_samples=target_samples,
        save_path="transport_animation.gif",
        config=VisualizationConfig(
            figsize=(8, 8),
            dpi=100,
            fps=20,
            interval=50,  # ms between frames
            alpha=0.7,
            point_size=20,
        ),
        title="Schrödinger Bridge Transport",
    )
    print(f"Saved to: {gif_path}")
    ```

    #### Comparison GIF (Multiple Methods Side-by-Side)

    ```python
    # Train multiple solvers
    score_solver = ScoreBasedSolver(problem)
    score_result = score_solver.train(key)
    score_traj = score_solver.sample(key, 300)

    fbsde_solver = FBSDESolver(problem)
    fbsde_result = fbsde_solver.train(key)
    fbsde_traj = fbsde_solver.sample(key, 300)

    rkhs_solver = RKHSSolver(problem)
    rkhs_result = rkhs_solver.train(key)
    rkhs_traj = rkhs_solver.sample(key, 300)

    # Create comparison GIF
    gif_path = create_comparison_gif(
        trajectories_dict={
            'Score-Based': score_traj,
            'FBSDE': fbsde_traj,
            'RKHS': rkhs_traj,
        },
        source_samples=source_samples,
        target_samples=target_samples,
        save_path="solver_comparison.gif",
    )
    ```

    ### 7.4 Custom Animation with Matplotlib

    For full control, create animations manually:

    ```python
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np

    # Extract trajectory data
    paths = np.array(trajectories.paths)  # [batch, time, dim]
    times = np.array(trajectories.times)
    n_frames = len(times)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    # Background: source and target
    ax.scatter(*source_samples.T, c='blue', alpha=0.1, s=10, label='Source')
    ax.scatter(*target_samples.T, c='red', alpha=0.1, s=10, label='Target')

    # Animated scatter
    scatter = ax.scatter([], [], c='green', s=20, alpha=0.7)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, va='top')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.legend()

    def animate(frame):
        positions = paths[:, frame, :2]
        scatter.set_offsets(positions)
        time_text.set_text(f't = {times[frame]:.3f}')
        return scatter, time_text

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True)
    anim.save('custom_animation.gif', writer='pillow', fps=20)
    plt.close()
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 7.5 Embedding GIFs in Marimo

    To display GIFs inline in a Marimo notebook:

    ```python
    import base64
    import tempfile
    import os

    # Create and save the GIF
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
        tmp_path = tmp.name

    create_transport_gif(trajectories, save_path=tmp_path)

    # Read and encode as base64
    with open(tmp_path, 'rb') as f:
        gif_base64 = base64.b64encode(f.read()).decode('utf-8')

    os.unlink(tmp_path)  # Clean up

    # Display in Marimo
    gif_html = f'<img src="data:image/gif;base64,{gif_base64}" style="max-width:100%;" />'
    mo.Html(gif_html)
    ```

    Or wrap it in a reusable function:

    ```python
    def display_gif_in_marimo(gif_path_or_trajectories, **kwargs):
        '''Display a GIF in Marimo from trajectories or path.'''
        import base64
        import tempfile
        import os

        if isinstance(gif_path_or_trajectories, str):
            path = gif_path_or_trajectories
        else:
            with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
                path = tmp.name
            create_transport_gif(gif_path_or_trajectories, save_path=path, **kwargs)

        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')

        if not isinstance(gif_path_or_trajectories, str):
            os.unlink(path)

        return mo.Html(f'<img src="data:image/gif;base64,{b64}" />')
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8. Marginal Schrödinger Bridges

    For problems with **intermediate time constraints**:

    ### 8.1 Problem Setup

    ```python
    from schrodinger_bridge.marginal_sb import (
        MarginalConstraint,
        MarginalSBProblem,
        MarginalSBSolver,
        MarginalSBConfig,
        create_marginal_sb_problem,
    )

    # Define marginal constraints at multiple times
    marginals = [
        MarginalConstraint(
            time=0.0,
            distribution=GaussianDistribution(mean=jnp.array([-2, 0]), cov=0.2, dim=2),
        ),
        MarginalConstraint(
            time=0.5,
            distribution=GaussianDistribution(mean=jnp.array([0, 1]), cov=0.3, dim=2),
        ),
        MarginalConstraint(
            time=1.0,
            distribution=GaussianDistribution(mean=jnp.array([2, 0]), cov=0.2, dim=2),
        ),
    ]

    problem = MarginalSBProblem(
        reference=BrownianMotion(sigma=0.3, dim=2),
        marginals=marginals,
        name="3-Marginal SB",
    )

    print(problem.summary())
    # 3-Marginal SB
    # Dimension: 2
    # Reference: BrownianMotion
    # Num marginals: 3
    # Num segments: 2
    # Marginal times: 0.000, 0.500, 1.000
    ```

    ### 8.2 Solving Marginal SB

    ```python
    config = MarginalSBConfig(
        segment_solver_type='score',  # or 'fbsde', 'doob', 'rkhs'
        coupling_method='sequential',
        num_iterations=2000,
    )

    solver = MarginalSBSolver(problem, config)
    results = solver.train(key)

    # Sample trajectories that pass through all marginals
    trajectories = solver.sample(key, num_samples=500)

    # Check marginal consistency
    mmd_results = solver.check_marginal_consistency(key, num_samples=500)
    for t, mmd in mmd_results.items():
        print(f"{t}: MMD = {mmd:.6f}")
    ```

    ### 8.3 Convenience Functions

    ```python
    from schrodinger_bridge.marginal_sb import solve_marginal_sb

    # One-liner solution
    solver, results = solve_marginal_sb(
        problem,
        key,
        solver_type='score',
        num_iterations=2000,
        verbose=1,
    )
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 9. Advanced Topics

    ### 9.1 Custom Integrators

    Choose the right SDE integrator for your problem:

    ```python
    from schrodinger_bridge.integrators import (
        EulerMaruyama,
        Heun,
        Milstein,
        AdaptiveIntegrator,
        SpectralIntegrator,
        create_integrator,
        IntegratorType,
    )

    # Basic Euler-Maruyama (default)
    integrator = EulerMaruyama()

    # Heun (second-order in drift)
    integrator = Heun()

    # Adaptive step-size
    from schrodinger_bridge.integrators import AdaptiveConfig
    integrator = AdaptiveIntegrator(AdaptiveConfig(
        rtol=1e-3,
        atol=1e-4,
        dt_min=1e-6,
        dt_max=0.1,
    ))

    # Spectral (for linear SDEs - exact solution!)
    A = jnp.array([[-1, 0], [0, -2]])  # Linear drift matrix
    integrator = SpectralIntegrator(linear_drift_matrix=A)

    # Use with solver
    solver = ScoreBasedSolver(
        problem,
        integrator_type=IntegratorType.HEUN,
    )
    ```

    ### 9.2 Brownian Bridge Sampling

    ```python
    from schrodinger_bridge.integrators import sample_brownian_bridge

    # Sample paths conditioned on endpoints
    bridge = sample_brownian_bridge(
        key,
        x0=source_samples,  # Start points
        x1=target_samples,  # End points
        time_grid=TimeGrid(num_steps=100),
        sigma=0.5,
    )

    # bridge.paths: [batch, 101, dim]
    # Exact endpoints guaranteed: bridge.paths[:, 0] == x0, bridge.paths[:, -1] == x1
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 9.3 Invariant Checking & Diagnostics

    The library automatically checks SB invariants:

    ```python
    from schrodinger_bridge import InvariantChecker

    checker = InvariantChecker()

    # Run all checks on trajectories
    report = checker.check_all(
        trajectories,
        source_samples,
        target_samples,
        key,
    )

    print(report.summary())
    # Diagnostic Report
    # Mass conservation max error: 2.31e-04
    # Source marginal error: 1.23e-02
    # Target marginal error: 8.45e-03
    # Violations (0):

    # Access individual metrics
    print(f"Source MMD: {report.marginal_error_source:.6f}")
    print(f"Target MMD: {report.marginal_error_target:.6f}")

    # Check for warnings/errors
    if report.has_errors:
        for violation in report.violations:
            print(f"ERROR: {violation}")
    ```

    #### MMD Computation

    ```python
    from schrodinger_bridge import mmd_squared

    # Compare two distributions
    mmd = mmd_squared(generated_samples, target_samples)
    print(f"MMD² = {mmd:.6f}")

    # Good values are < 0.1, excellent < 0.01
    ```

    ### 9.4 Device & Memory Management

    ```python
    from schrodinger_bridge import (
        get_device_info,
        print_device_info,
        check_memory_for_batch,
        process_in_batches,
    )

    # Check device configuration
    info = get_device_info()
    print(f"Using: {info.kind.value.upper()} x {info.count}")

    # Check if batch fits in memory
    can_fit, recommended = check_memory_for_batch(
        batch_size=1000,
        dim=2,
        num_steps=100,
    )
    if not can_fit:
        print(f"Reduce batch size to {recommended}")

    # Process large data in batches
    def my_fn(batch):
        return solver.sample(key, len(batch))

    results = process_in_batches(
        my_fn,
        large_data,
        batch_size=256,
        show_progress=True,
    )
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 9.5 Kernel Methods

    For non-parametric approaches without neural networks:

    ```python
    from schrodinger_bridge.kernels import (
        gaussian_kernel,
        laplacian_kernel,
        matern_kernel,
        median_heuristic,
        fit_kde,
        KernelMeanEmbedding,
        kernel_score_estimation,
    )

    # Kernel density estimation
    kde = fit_kde(samples, bandwidth=None)  # Auto bandwidth
    density = kde(query_points)
    score = kde.gradient(query_points)  # ∇log p(x)

    # Kernel mean embedding
    embedding = KernelMeanEmbedding(samples, bandwidth=0.5)

    # Compute MMD between embeddings
    embedding2 = KernelMeanEmbedding(other_samples)
    mmd2 = embedding.mmd_squared(embedding2)

    # Estimate score function from samples
    score_fn = kernel_score_estimation(samples, bandwidth=0.5, reg=1e-4)
    scores = score_fn(query_points)
    ```

    ### 9.6 Complete Example: End-to-End Pipeline

    ```python
    import jax
    import jax.numpy as jnp
    from schrodinger_bridge import (
        SBProblem, BrownianMotion, GaussianDistribution, TwoMoonsDistribution,
        TimeGrid, TrainingConfig, plot_trajectories, create_transport_gif,
    )
    from schrodinger_bridge.solvers import ScoreBasedSolver, ScoreBasedConfig

    # 1. Setup
    key = jax.random.PRNGKey(42)

    problem = SBProblem(
        reference=BrownianMotion(sigma=0.4, dim=2),
        source=GaussianDistribution(dim=2),
        target=TwoMoonsDistribution(noise=0.05),
        time_grid=TimeGrid(num_steps=100),
    )

    # 2. Create and train solver
    solver = ScoreBasedSolver(
        problem,
        config=ScoreBasedConfig(
            hidden_dims=(256, 256, 256),
            learning_rate=1e-4,
        ),
    )

    result = solver.train(
        key,
        TrainingConfig(num_iterations=5000, batch_size=256),
    )
    print(f"Training complete. Final loss: {result.final_loss:.6f}")

    # 3. Sample trajectories
    key, sample_key = jax.random.split(key)
    trajectories = solver.sample(sample_key, num_samples=500)

    # 4. Evaluate quality
    print(result.diagnostics.summary())

    # 5. Visualize
    plot_trajectories(trajectories, num_show=50, save_path="paths.png")

    # 6. Create animation
    source = problem.sample_source(key, 500)
    target = problem.sample_target(key, 500)
    create_transport_gif(
        trajectories,
        source_samples=source,
        target_samples=target,
        save_path="transport.gif",
    )

    print("Done! Check paths.png and transport.gif")
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 10. API Reference Summary

    ### Core Classes

    | Class | Purpose |
    |-------|---------|
    | `SBProblem` | Define source, target, reference |
    | `TimeGrid` | Time discretization |
    | `TrajectoryBatch` | Container for sampled paths |
    | `SolverResult` | Training output |
    | `DiagnosticReport` | Quality metrics |

    ### Distributions

    | Class | Description |
    |-------|-------------|
    | `GaussianDistribution` | Multivariate Gaussian |
    | `TwoMoonsDistribution` | Classic ML benchmark |
    | `SwissRollDistribution` | Spiral manifold |
    | `MixtureDistribution` | Mixture of distributions |

    ### Reference Dynamics

    | Class | SDE |
    |-------|-----|
    | `BrownianMotion` | dX = σ dW |
    | `OrnsteinUhlenbeck` | dX = -θ(X-μ) dt + σ dW |
    | `VarianceExploding` | dX = σ(t) dW |
    | `VariancePreserving` | dX = -½β(t)X dt + √β(t) dW |

    ### Solvers

    | Solver | Method |
    |--------|--------|
    | `DoobHTransformSolver` | h-transform (analytical/kernel) |
    | `ScoreBasedSolver` | Denoising score matching |
    | `FBSDESolver` | Forward-backward SDE |
    | `RKHSSolver` | Kernel regression |
    | `IMFSolver` | Iterative Markovian fitting |
    | `IPFSolver` | Iterative proportional fitting |

    ### Network Factories

    | Factory | Architecture |
    |---------|-------------|
    | `MLPFactory` | Time-conditioned MLP (default) |
    | `UNetFactory` | Encoder-decoder with skip connections (2D/3D) |
    | `TransformerFactory` | Self-attention over tokens |
    | `CustomFactory` | Wraps loose `init_fn` / `forward_fn` |
    | `NetworkFactory` | ABC — subclass for custom architectures |

    ### Visualization

    | Function | Output |
    |----------|--------|
    | `plot_marginals` | Source/target/generated comparison |
    | `plot_trajectories` | Sample paths |
    | `plot_velocity_field` | Drift vector field |
    | `plot_diagnostics` | Loss and metrics |
    | `create_transport_gif` | Animated evolution |
    | `create_comparison_gif` | Multi-solver animation |

    ---

    **Happy bridging! 🌉**
    """
    )
    return


if __name__ == "__main__":
    app.run()
