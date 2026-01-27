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
    2. **Problem Definition** — Setting up source, target, and reference dynamics
    3. **Choosing a Solver** — When to use each of the 6 solver methods
    4. **Training & Inference** — Complete workflow from definition to sampling
    5. **Neural Network Approaches** — Score networks, ICNNs, and OTT-JAX integration
    6. **Visualization & GIFs** — Creating publication-quality animations
    7. **Marginal SB** — Multi-time-point constraints
    8. **Advanced Topics** — Custom distributions, integrators, diagnostics

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
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Problem Definition

    Every Schrödinger Bridge problem requires three components:

    | Component | What It Is | Examples |
    |-----------|-----------|----------|
    | **Source (μ₀)** | Initial distribution | Gaussian, data samples |
    | **Target (μ₁)** | Final distribution | TwoMoons, SwissRoll, data |
    | **Reference** | Prior stochastic process | Brownian motion, OU process |

    ### 2.1 Built-in Distributions

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

    ### 2.2 Reference Dynamics

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

    ### 2.3 Putting It Together: SBProblem

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
    # === Gaussian-to-TwoMoons ===
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
    ### 2.4 Custom Distributions

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
    ## 3. Choosing a Solver

    The library provides **6 distinct solver methods**. Here's when to use each:

    ### 3.1 Decision Flowchart

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

    ### 3.2 Solver Comparison Table

    | Solver | Neural? | Speed | Accuracy | Best For |
    |--------|---------|-------|----------|----------|
    | **Doob (analytical)** | ❌ | ⚡⚡⚡ | ⭐⭐⭐ | Gaussian problems |
    | **Doob (kernel)** | ❌ | ⚡⚡ | ⭐⭐ | Quick prototyping |
    | **RKHS** | ❌ | ⚡⚡ | ⭐⭐ | No-training-needed |
    | **Score-Based** | ✅ | ⚡ | ⭐⭐⭐ | General purpose |
    | **FBSDE** | ✅ | ⚡ | ⭐⭐⭐ | Optimal control view |
    | **IMF** | ✅ | ⚡⚡ | ⭐⭐ | Large-scale, simulation-free |
    | **IPF** | ✅ | 🐢 | ⭐⭐⭐ | Classical, interpretable |

    ### 3.3 Mathematical Representations

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
    ## 4. Training & Inference

    ### 4.1 Complete Workflow

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

    ### 4.2 Using the Solution Object

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
    mo.md(
        r"""
    ### 4.3 Solver-Specific Examples

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
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Neural Network Approaches

    ### 5.1 Network Architecture

    The library uses **pure JAX** neural networks (no Flax, Equinox, etc.):

    ```python
    from schrodinger_bridge.networks import (
        init_score_network,
        score_network_forward,
        init_potential_network,
        potential_network_forward,
        TimeConditionedMLPConfig,
    )

    # Initialize a score network
    key = jax.random.PRNGKey(0)
    params = init_score_network(
        key,
        dim=2,
        hidden_dims=(256, 256, 256),
        time_embed_dim=64,
    )

    # Forward pass
    x = jnp.array([[0.0, 1.0], [1.0, 0.0]])  # [batch, dim]
    t = jnp.array([0.5, 0.5])  # [batch]
    score = score_network_forward(params, x, t)  # [batch, dim]
    ```

    ### 5.2 Time Embedding

    Time is embedded using **sinusoidal positional encoding**:

    ```python
    from schrodinger_bridge.networks import sinusoidal_embedding

    t = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    embedding = sinusoidal_embedding(t, dim=64)
    # Shape: [5, 64] - each time gets a 64-dimensional embedding
    ```

    ### 5.3 Input Convex Neural Networks (ICNN)

    For optimal transport applications, ICNNs ensure the potential is convex:

    ```python
    from schrodinger_bridge.networks import (
        init_icnn_params,
        icnn_forward,
        icnn_gradient,
    )

    # Initialize ICNN
    params = init_icnn_params(
        key,
        input_dim=2,
        hidden_dims=(256, 256, 256),
    )

    # Evaluate convex potential φ(x)
    x = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    potential = icnn_forward(params, x)  # [batch] - guaranteed convex in x

    # Get optimal transport map T(x) = ∇φ(x)
    transport_map = icnn_gradient(params, x)  # [batch, dim]
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
    ### 5.4 OTT-JAX Integration

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
    ## 6. Visualization & GIFs

    The library provides comprehensive visualization utilities.

    ### 6.1 Static Plots

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

    ### 6.2 Velocity Field Visualization

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
    ### 6.3 Creating Animated GIFs

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

    ### 6.4 Custom Animation with Matplotlib

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
    ### 6.5 Embedding GIFs in Marimo

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
    ## 7. Marginal Schrödinger Bridges

    For problems with **intermediate time constraints**:

    ### 7.1 Problem Setup

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
    # === 3-Marginal SB ===
    # Dimension: 2
    # Reference: BrownianMotion
    # Num marginals: 3
    # Num segments: 2
    # Marginal times: 0.000, 0.500, 1.000
    ```

    ### 7.2 Solving Marginal SB

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

    ### 7.3 Convenience Functions

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
    ## 8. Advanced Topics

    ### 8.1 Custom Integrators

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

    ### 8.2 Brownian Bridge Sampling

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
    ### 8.3 Invariant Checking & Diagnostics

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
    # === Diagnostic Report ===
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

    ### 8.4 Device & Memory Management

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
    ### 8.5 Kernel Methods

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

    ### 8.6 Complete Example: End-to-End Pipeline

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
    ## 9. API Reference Summary

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
