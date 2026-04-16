"""Compare stochastic bridge sampling with probability-flow sampling.

This example trains a score-based Schrödinger Bridge, materializes the solved
process via ``solution.as_process()``, and compares:

1. Stochastic bridge paths sampled from the learned SDE
2. Deterministic probability-flow paths using the same initial states
3. The terminal point clouds against the target distribution

Run from the repository root:

    python schrodinger_bridge/tutorial/examples/bridge_process/bridge_process_vs_flow.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "This example requires matplotlib. Install it with "
        "`python -m pip install matplotlib` or `python -m pip install -r requirements-dev.txt`."
    ) from exc

from schrodinger_bridge import (
    BrownianMotion,
    GaussianDistribution,
    SBProblem,
    ScoreBasedConfig,
    ScoreBasedSolver,
    TimeGrid,
    TrainingConfig,
)


def _plot_paths(ax, paths: jnp.ndarray, title: str, color: str, num_show: int = 40) -> None:
    """Plot a subset of 2D trajectories on a single axis."""
    num_show = min(num_show, paths.shape[0])

    for idx in range(num_show):
        traj = paths[idx]
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.18, linewidth=0.8)

    ax.scatter(paths[:num_show, 0, 0], paths[:num_show, 0, 1], s=18, c="#1d4ed8", label="start")
    ax.scatter(paths[:num_show, -1, 0], paths[:num_show, -1, 1], s=18, c="#dc2626", label="end")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="best")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    figure_path = out_dir / "bridge_process_vs_flow.png"

    key = jax.random.PRNGKey(0)

    problem = SBProblem(
        reference=BrownianMotion(sigma=0.35, dim=2),
        source=GaussianDistribution(
            mean=jnp.array([-2.0, 0.0]),
            cov=0.35,
            dim=2,
        ),
        target=GaussianDistribution(
            mean=jnp.array([2.0, 0.0]),
            cov=jnp.array([[0.7, 0.35], [0.35, 0.45]]),
            dim=2,
        ),
        time_grid=TimeGrid(num_steps=40),
        name="BridgeProcess vs Probability Flow",
    )

    solver = ScoreBasedSolver(
        problem,
        ScoreBasedConfig(hidden_dims=(64, 64), learning_rate=1e-3),
    )

    print("Training score-based bridge...")
    solution = solver.solve(
        key,
        TrainingConfig(
            num_iterations=100,
            batch_size=128,
            eval_every=20,
            patience=200,
        ),
    )
    process = solution.as_process()

    key, k_init, k_stochastic, k_flow, k_target = jax.random.split(key, 5)
    x0 = problem.sample_source(k_init, 256)

    print("Sampling stochastic bridge paths...")
    stochastic_paths = process.sample_paths(k_stochastic, num_samples=x0.shape[0], x0=x0)

    print("Sampling probability-flow paths...")
    flow_paths = process.sample_flow(k_flow, num_samples=x0.shape[0], x0=x0)

    target_samples = problem.sample_target(k_target, x0.shape[0])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=140)

    _plot_paths(
        axes[0],
        stochastic_paths.paths,
        title="Stochastic Bridge Paths",
        color="#0f766e",
    )
    _plot_paths(
        axes[1],
        flow_paths.paths,
        title="Probability-Flow Paths",
        color="#7c3aed",
    )

    axes[2].scatter(
        target_samples[:, 0],
        target_samples[:, 1],
        s=16,
        c="#dc2626",
        alpha=0.18,
        label="target",
    )
    axes[2].scatter(
        stochastic_paths.paths[:, -1, 0],
        stochastic_paths.paths[:, -1, 1],
        s=16,
        c="#0f766e",
        alpha=0.45,
        label="stochastic endpoint",
    )
    axes[2].scatter(
        flow_paths.paths[:, -1, 0],
        flow_paths.paths[:, -1, 1],
        s=16,
        c="#7c3aed",
        alpha=0.45,
        label="flow endpoint",
    )
    axes[2].set_title("Endpoint Comparison")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    axes[2].legend(loc="best")

    fig.suptitle("BridgeProcess: stochastic sampling vs probability flow", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote comparison figure to: {figure_path}")


if __name__ == "__main__":
    main()
