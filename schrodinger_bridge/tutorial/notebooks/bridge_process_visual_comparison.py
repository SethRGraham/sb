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

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as bp_mo
    import jax as bp_jax
    import jax.numpy as bp_jnp
    import numpy as bp_np

    try:
        import matplotlib.pyplot as bp_plt
        bp_plot_import_error = None
    except ImportError as bp_exc:
        bp_plt = None
        bp_plot_import_error = str(bp_exc)

    from schrodinger_bridge import (
        BrownianMotion as bp_BrownianMotion,
        GaussianDistribution as bp_GaussianDistribution,
        SBProblem as bp_SBProblem,
        ScoreBasedConfig as bp_ScoreBasedConfig,
        ScoreBasedSolver as bp_ScoreBasedSolver,
        TimeGrid as bp_TimeGrid,
        TrainingConfig as bp_TrainingConfig,
    )

    return (
        bp_BrownianMotion,
        bp_GaussianDistribution,
        bp_SBProblem,
        bp_ScoreBasedConfig,
        bp_ScoreBasedSolver,
        bp_TimeGrid,
        bp_TrainingConfig,
        bp_jax,
        bp_jnp,
        bp_mo,
        bp_np,
        bp_plot_import_error,
        bp_plt,
    )


@app.cell
def _(bp_mo, bp_plot_import_error, bp_plt):
    if bp_plot_import_error is not None:
        bp_dependency_notice = bp_mo.md(
            f"""
            ## Missing notebook dependency

            This notebook needs `matplotlib`, but your current Python environment does not have it.

            Install the project notebook/dev dependencies from the repo root:

            ```bash
            python -m pip install -e .[dev]
            ```

            or minimally:

            ```bash
            python -m pip install matplotlib
            ```

            Original import error: `{bp_plot_import_error}`
            """
        )
    else:
        bp_dependency_notice = bp_mo.md("")

    bp_palette = {
        "bg": "#f6f1e8",
        "panel": "#fffdfa",
        "grid": "#d8c9b2",
        "text": "#2b241c",
        "source": "#1d4ed8",
        "target": "#d04d2d",
        "stochastic": "#0f766e",
        "flow": "#b7791f",
        "accent": "#7c3aed",
    }

    if bp_plt is not None:
        bp_plt.rcParams.update(
            {
                "figure.facecolor": bp_palette["bg"],
                "axes.facecolor": bp_palette["panel"],
                "axes.edgecolor": bp_palette["grid"],
                "axes.labelcolor": bp_palette["text"],
                "axes.titlecolor": bp_palette["text"],
                "xtick.color": bp_palette["text"],
                "ytick.color": bp_palette["text"],
                "grid.color": bp_palette["grid"],
                "font.size": 11,
            }
        )

    bp_dependency_notice
    return (bp_palette,)


@app.cell
def _(bp_mo):
    bp_mo.md(
        r"""
        # Learned Path Measure as a Generative Diffusion Model

        A solved Schrödinger Bridge gives you a **path measure** over trajectories, not just a terminal map.
        That makes it useful as a generative model:

        1. start from a simple **source distribution** like a Gaussian prior,
        2. learn the bridge process that transports that prior toward a **target distribution**,
        3. sample new trajectories from the learned process to generate target-like data.

        In diffusion-model language:

        - `source` plays the role of the easy prior,
        - `BridgeProcess.sample_paths(...)` gives noisy generative trajectories,
        - `BridgeProcess.sample_flow(...)` gives the deterministic probability-flow sampler.

        If we write the learned path measure as `P_theta(X_{0:T})`, then generation is:

        1. sample `X_0` from the source prior,
        2. sample a full trajectory `X_{0:T}` from the learned bridge,
        3. use `X_T` as the generated sample from the transported distribution.

        This notebook is built to answer one concrete question:

        > How do I use the learned path measure to propagate a source distribution into a target one?
        """
    )
    return


@app.cell
def _(bp_mo):
    bp_iterations_slider = bp_mo.ui.slider(20, 160, value=80, step=20, label="Training iterations")
    bp_batch_slider = bp_mo.ui.slider(32, 256, value=128, step=32, label="Batch size")
    bp_samples_slider = bp_mo.ui.slider(64, 384, value=224, step=32, label="Generated samples")
    bp_steps_slider = bp_mo.ui.slider(20, 80, value=40, step=10, label="Time steps")
    bp_sigma_slider = bp_mo.ui.slider(0.15, 0.60, value=0.35, step=0.05, label="Reference sigma")
    bp_seed_input = bp_mo.ui.number(value=0, label="Random seed")

    bp_mo.vstack(
        [
            bp_mo.md("## Controls"),
            bp_mo.hstack([bp_iterations_slider, bp_batch_slider, bp_samples_slider]),
            bp_mo.hstack([bp_steps_slider, bp_sigma_slider, bp_seed_input]),
        ]
    )
    return (
        bp_batch_slider,
        bp_iterations_slider,
        bp_samples_slider,
        bp_seed_input,
        bp_sigma_slider,
        bp_steps_slider,
    )


@app.cell
def _(
    bp_BrownianMotion,
    bp_GaussianDistribution,
    bp_SBProblem,
    bp_ScoreBasedConfig,
    bp_ScoreBasedSolver,
    bp_TimeGrid,
    bp_TrainingConfig,
    bp_batch_slider,
    bp_iterations_slider,
    bp_jax,
    bp_jnp,
    bp_seed_input,
    bp_sigma_slider,
    bp_steps_slider,
):
    bp_master_key = bp_jax.random.PRNGKey(int(bp_seed_input.value))

    bp_problem = bp_SBProblem(
        reference=bp_BrownianMotion(sigma=float(bp_sigma_slider.value), dim=2),
        source=bp_GaussianDistribution(
            mean=bp_jnp.array([-2.0, 0.0]),
            cov=0.35,
            dim=2,
        ),
        target=bp_GaussianDistribution(
            mean=bp_jnp.array([2.0, 0.0]),
            cov=bp_jnp.array([[0.7, 0.35], [0.35, 0.45]]),
            dim=2,
        ),
        time_grid=bp_TimeGrid(num_steps=int(bp_steps_slider.value)),
        name="Generative Bridge Demo",
    )

    bp_solver = bp_ScoreBasedSolver(
        bp_problem,
        bp_ScoreBasedConfig(hidden_dims=(64, 64), learning_rate=1e-3),
    )
    bp_training_config = bp_TrainingConfig(
        num_iterations=int(bp_iterations_slider.value),
        batch_size=int(bp_batch_slider.value),
        eval_every=max(1, int(bp_iterations_slider.value) // 4),
        patience=int(bp_iterations_slider.value) + 5,
    )

    bp_solution = bp_solver.solve(bp_master_key, bp_training_config)
    bp_process = bp_solution.as_process()
    return bp_master_key, bp_problem, bp_process, bp_solution, bp_training_config


@app.cell
def _(bp_mo, bp_solution, bp_training_config):
    bp_mo.md(
        f"""
        ## Solved Bridge

        - Solver: `{bp_solution.solver_type.name}`
        - Representation: `{bp_solution.representation.name}`
        - Training iterations: `{bp_training_config.num_iterations}`
        - Runtime generative object: `bp_solution.as_process()`

        Once trained, the learned bridge can be treated as a **generative path measure** over trajectories connecting the source prior to the target distribution.
        """
    )
    return


@app.cell
def _(bp_mo):
    bp_path_measure_panel = bp_mo.md(
        r"""
        ## From Path Measure to Generative Model

        The key object is the **law of entire trajectories**. Once the bridge is trained, you do not only know where mass should end up at `t = 1`; you know how to sample whole paths that start at the source and evolve toward the target.

        For a practical generative workflow, that means:

        - the source distribution acts like the easy latent prior,
        - the learned bridge dynamics propagate those prior samples forward in time,
        - the terminal marginal of the sampled paths becomes your generated dataset.

        In symbols, if `X_{0:T} ~ P_theta` is sampled from the learned bridge path measure, then the generated sample is simply `X_T`.
        """
    )
    bp_path_measure_panel
    return


@app.cell
def _(bp_mo):
    bp_usage_block = bp_mo.md(
        r"""
        ## Minimal Generative Workflow

        ```python
        solution = solver.solve(key, train_cfg)
        process = solution.as_process()

        # Sample latent/prior points from the source marginal
        x0 = problem.sample_source(key, num_samples=256)

        # Stochastic generative sampler: draw paths from the learned path measure
        stochastic_paths = process.sample_paths(key, num_samples=256, x0=x0)
        generated_samples = stochastic_paths.paths[:, -1, :]

        # Deterministic sampler: probability-flow ODE
        flow_paths = process.sample_flow(key, num_samples=256, x0=x0)
        flow_samples = flow_paths.paths[:, -1, :]
        ```
        """
    )
    bp_usage_block
    return


@app.cell
def _(bp_jax, bp_master_key, bp_np, bp_problem, bp_process, bp_samples_slider):
    bp_num_generated = int(bp_samples_slider.value)
    (
        _bp_key_unused,
        bp_key_source,
        bp_key_paths,
        bp_key_flow,
        bp_key_target,
    ) = bp_jax.random.split(bp_master_key, 5)

    bp_source_initial = bp_problem.sample_source(bp_key_source, bp_num_generated)
    bp_stochastic_paths = bp_process.sample_paths(
        bp_key_paths,
        num_samples=bp_num_generated,
        x0=bp_source_initial,
    )
    bp_flow_paths = bp_process.sample_flow(
        bp_key_flow,
        num_samples=bp_num_generated,
        x0=bp_source_initial,
    )
    bp_target_samples = bp_problem.sample_target(bp_key_target, bp_num_generated)

    bp_time_points = bp_np.array(bp_stochastic_paths.times)
    bp_source_np = bp_np.array(bp_source_initial)
    bp_stochastic_np = bp_np.array(bp_stochastic_paths.paths)
    bp_flow_np = bp_np.array(bp_flow_paths.paths)
    bp_target_np = bp_np.array(bp_target_samples)

    bp_generated_stochastic = bp_stochastic_np[:, -1, :]
    bp_generated_flow = bp_flow_np[:, -1, :]

    bp_target_mean = bp_target_np.mean(axis=0)
    bp_stochastic_mean_error = float(bp_np.linalg.norm(bp_generated_stochastic.mean(axis=0) - bp_target_mean))
    bp_flow_mean_error = float(bp_np.linalg.norm(bp_generated_flow.mean(axis=0) - bp_target_mean))

    return (
        bp_flow_mean_error,
        bp_flow_np,
        bp_generated_flow,
        bp_generated_stochastic,
        bp_source_np,
        bp_stochastic_mean_error,
        bp_stochastic_np,
        bp_target_mean,
        bp_target_np,
        bp_time_points,
    )


@app.cell
def _(bp_np, bp_stochastic_np, bp_time_points):
    bp_snapshot_indices = bp_np.array(
        [0, len(bp_time_points) // 2, len(bp_time_points) - 1],
        dtype=int,
    )
    bp_snapshot_times = bp_time_points[bp_snapshot_indices]
    bp_snapshot_clouds = [bp_stochastic_np[:, int(bp_idx), :] for bp_idx in bp_snapshot_indices]
    return bp_snapshot_clouds, bp_snapshot_times


@app.cell
def _(bp_flow_mean_error, bp_mo, bp_stochastic_mean_error, bp_target_mean):
    bp_diagnostics_panel = bp_mo.hstack(
        [
            bp_mo.md(
                f"""
                ## Endpoint Diagnostics

                - Target mean: `{bp_target_mean.round(3)}`
                - Stochastic generative mean error: `{bp_stochastic_mean_error:.3f}`
                - Probability-flow mean error: `{bp_flow_mean_error:.3f}`
                """
            ),
            bp_mo.md(
                """
                ## How To Read This

                - The **stochastic** sampler uses the learned path measure directly, so it keeps diffusion noise alive during generation.
                - The **probability-flow** sampler removes the diffusion noise and follows a deterministic transport path induced by the score.
                - Both start from the same source prior, then push that prior toward the target distribution.
                """
            ),
        ],
        widths="equal",
    )
    bp_diagnostics_panel
    return


@app.cell
def _(
    bp_mo,
    bp_palette,
    bp_plt,
    bp_snapshot_clouds,
    bp_snapshot_times,
    bp_source_np,
    bp_target_np,
):
    if bp_plt is None:
        bp_snapshot_panel = bp_mo.md(
            """
            Install `matplotlib` to render the marginal propagation snapshots.
            """
        )
    else:
        bp_snapshot_fig, bp_snapshot_axes = bp_plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
        bp_snapshot_fig.patch.set_facecolor(bp_palette["bg"])

        bp_snapshot_titles = [
            "Source prior at t = 0",
            "Bridge marginal at mid-time",
            "Generated marginal at t = 1",
        ]

        for bp_snapshot_ax, bp_snapshot_cloud, bp_snapshot_time, bp_snapshot_title in zip(
            bp_snapshot_axes,
            bp_snapshot_clouds,
            bp_snapshot_times,
            bp_snapshot_titles,
        ):
            bp_snapshot_ax.scatter(
                bp_target_np[:, 0],
                bp_target_np[:, 1],
                s=16,
                c=bp_palette["target"],
                alpha=0.10,
                label="target",
            )
            bp_snapshot_ax.scatter(
                bp_source_np[:, 0],
                bp_source_np[:, 1],
                s=16,
                c=bp_palette["source"],
                alpha=0.08,
                label="source",
            )
            bp_snapshot_ax.scatter(
                bp_snapshot_cloud[:, 0],
                bp_snapshot_cloud[:, 1],
                s=18,
                c=bp_palette["stochastic"],
                alpha=0.42,
                label="propagated samples",
            )
            bp_snapshot_ax.set_title(
                f"{bp_snapshot_title}\n(saved t = {float(bp_snapshot_time):.2f})",
                fontweight="bold",
            )
            bp_snapshot_ax.set_xlabel("x")
            bp_snapshot_ax.set_ylabel("y")
            bp_snapshot_ax.set_aspect("equal")
            bp_snapshot_ax.grid(alpha=0.35)

        bp_snapshot_axes[0].legend(loc="best")
        bp_snapshot_fig.tight_layout()

        bp_snapshot_panel = bp_mo.vstack(
            [
                bp_mo.md(
                    """
                    ## Propagating the Prior Through the Learned Path Measure

                    This is the generative story in distribution form: draw source samples once, then let the learned bridge move those samples through time. The three panels below show the source marginal, an interior bridge marginal, and the final generated marginal.
                    """
                ),
                bp_mo.as_html(bp_snapshot_fig),
            ]
        )

    bp_snapshot_panel
    return


@app.cell
def _(
    bp_flow_np,
    bp_generated_flow,
    bp_generated_stochastic,
    bp_mo,
    bp_palette,
    bp_plt,
    bp_source_np,
    bp_stochastic_np,
    bp_target_np,
    bp_time_points,
):
    if bp_plt is None:
        bp_plot_panel = bp_mo.md(
            """
            Plot rendering is disabled until `matplotlib` is installed in the active environment.
            Once installed, rerun the notebook and the generative comparison panels will appear here.
            """
        )
    else:
        def bp_plot_bundle(bp_ax, bp_paths, bp_title, bp_color):
            bp_show = min(56, bp_paths.shape[0])
            for bp_idx in range(bp_show):
                bp_traj = bp_paths[bp_idx]
                bp_ax.plot(bp_traj[:, 0], bp_traj[:, 1], color=bp_color, alpha=0.14, linewidth=0.9)

            bp_ax.scatter(
                bp_paths[:bp_show, 0, 0],
                bp_paths[:bp_show, 0, 1],
                c=bp_palette["source"],
                s=24,
                label="source prior",
                zorder=3,
            )
            bp_ax.scatter(
                bp_paths[:bp_show, -1, 0],
                bp_paths[:bp_show, -1, 1],
                c=bp_palette["target"],
                s=20,
                label="generated endpoint",
                zorder=3,
            )
            bp_ax.set_title(bp_title, fontweight="bold")
            bp_ax.set_xlabel("x")
            bp_ax.set_ylabel("y")
            bp_ax.grid(alpha=0.35)
            bp_ax.set_aspect("equal")
            bp_ax.legend(loc="best")

        bp_fig, bp_axes = bp_plt.subplots(2, 2, figsize=(15, 11), dpi=150)
        bp_fig.patch.set_facecolor(bp_palette["bg"])

        bp_axes[0, 0].scatter(
            bp_source_np[:, 0],
            bp_source_np[:, 1],
            s=18,
            c=bp_palette["source"],
            alpha=0.30,
            label="source prior",
        )
        bp_axes[0, 0].scatter(
            bp_target_np[:, 0],
            bp_target_np[:, 1],
            s=18,
            c=bp_palette["target"],
            alpha=0.18,
            label="target distribution",
        )
        bp_axes[0, 0].set_title("Prior and Target", fontweight="bold")
        bp_axes[0, 0].set_xlabel("x")
        bp_axes[0, 0].set_ylabel("y")
        bp_axes[0, 0].set_aspect("equal")
        bp_axes[0, 0].grid(alpha=0.35)
        bp_axes[0, 0].legend(loc="best")

        bp_axes[0, 1].scatter(
            bp_target_np[:, 0],
            bp_target_np[:, 1],
            s=18,
            c=bp_palette["target"],
            alpha=0.16,
            label="target",
        )
        bp_axes[0, 1].scatter(
            bp_generated_stochastic[:, 0],
            bp_generated_stochastic[:, 1],
            s=18,
            c=bp_palette["stochastic"],
            alpha=0.45,
            label="generated via path measure",
        )
        bp_axes[0, 1].scatter(
            bp_generated_flow[:, 0],
            bp_generated_flow[:, 1],
            s=18,
            c=bp_palette["flow"],
            alpha=0.42,
            label="generated via flow",
        )
        bp_axes[0, 1].set_title("Generated Endpoints", fontweight="bold")
        bp_axes[0, 1].set_xlabel("x")
        bp_axes[0, 1].set_ylabel("y")
        bp_axes[0, 1].set_aspect("equal")
        bp_axes[0, 1].grid(alpha=0.35)
        bp_axes[0, 1].legend(loc="best")

        bp_plot_bundle(
            bp_axes[1, 0],
            bp_stochastic_np,
            "Stochastic Generative Paths",
            bp_palette["stochastic"],
        )
        bp_plot_bundle(
            bp_axes[1, 1],
            bp_flow_np,
            "Deterministic Probability-Flow Paths",
            bp_palette["flow"],
        )

        bp_fig.suptitle(
            "Using the learned bridge path measure as a generative diffusion model",
            fontsize=16,
            fontweight="bold",
            y=0.985,
        )
        bp_fig.tight_layout()

        bp_plot_panel = bp_mo.vstack(
            [
                bp_mo.as_html(bp_fig),
                bp_mo.md(
                    """
                    The top row shows the generative story at the distribution level: start from a simple prior and propagate it until the generated endpoints match the target geometry.  
                    The bottom row shows the trajectory-level view of the same process: the learned path measure itself can be sampled to create stochastic generative paths, while the probability-flow ODE gives a deterministic sampler built from the same learned bridge.
                    """
                ),
            ]
        )

    bp_plot_panel
    return


@app.cell
def _(bp_flow_np, bp_mo, bp_np, bp_stochastic_np, bp_time_points):
    bp_stochastic_mean_x = bp_stochastic_np[:, :, 0].mean(axis=0)
    bp_stochastic_std_x = bp_stochastic_np[:, :, 0].std(axis=0)
    bp_flow_mean_x = bp_flow_np[:, :, 0].mean(axis=0)
    bp_flow_std_x = bp_flow_np[:, :, 0].std(axis=0)

    bp_summary_table = bp_mo.md(
        f"""
        ## Transport Summary

        - Initial mean x: `{bp_stochastic_np[:, 0, 0].mean():.3f}`
        - Final stochastic mean x: `{bp_stochastic_mean_x[-1]:.3f}`
        - Final flow mean x: `{bp_flow_mean_x[-1]:.3f}`
        - Final stochastic spread x: `{bp_stochastic_std_x[-1]:.3f}`
        - Final flow spread x: `{bp_flow_std_x[-1]:.3f}`
        - Saved time points: `{len(bp_time_points)}`
        """
    )
    bp_summary_table
    return


@app.cell
def _(bp_mo):
    bp_mo.md(
        r"""
        ## Next Step

        If you want this notebook to demonstrate the backend abstraction too, the natural extension is a backend selector that compares:

        - `solution.as_process(backend="native")`
        - `solution.as_process(backend="diffrax")`

        while keeping the generative story identical.
        """
    )
    return


if __name__ == "__main__":
    app.run()
