"""Compare MalliavinScoreSolver against ScoreBasedSolver on two benchmark tasks.

This is meant as a practical quality check, not a formal benchmark.

Suggested reading of the output:
- Gaussian -> Gaussian:
  Malliavin should at least produce stable transport and sensible endpoint means.
- Gaussian -> TwoMoons:
  compare endpoint MMD^2 and visual quality against the score solver.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from schrodinger_bridge import (
    SBProblem,
    BrownianMotion,
    GaussianDistribution,
    TwoMoonsDistribution,
    TimeGrid,
    MalliavinScoreSolver,
    MalliavinConfig,
    ScoreBasedSolver,
    ScoreBasedConfig,
    TrainingConfig,
    mmd_squared,
    create_transport_gif,
)


def run_case(name: str, problem: SBProblem, train_cfg: TrainingConfig):
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    malliavin = MalliavinScoreSolver(
        problem,
        MalliavinConfig(
            hidden_dims=(64, 64),
            reward_bandwidth=0.25,
        ),
    )
    score = ScoreBasedSolver(
        problem,
        ScoreBasedConfig(
            hidden_dims=(64, 64),
            learning_rate=1e-3,
        ),
    )

    t0 = time.time()
    malliavin.train(k1, train_cfg)
    malliavin_time = time.time() - t0

    t0 = time.time()
    score.train(k2, train_cfg)
    score_time = time.time() - t0

    traj_m = malliavin.sample(k3, num_samples=500)
    traj_s = score.sample(k4, num_samples=500)
    target = problem.sample_target(k5, 500)

    mmd_m = mmd_squared(traj_m.paths[:, -1, :], target)
    mmd_s = mmd_squared(traj_s.paths[:, -1, :], target)

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(f"Malliavin MMD^2: {mmd_m:.6f}  time={malliavin_time:.2f}s")
    print(f"Score     MMD^2: {mmd_s:.6f}  time={score_time:.2f}s")

    create_transport_gif(
        traj_m,
        source_samples=problem.sample_source(jax.random.PRNGKey(11), 300),
        target_samples=problem.sample_target(jax.random.PRNGKey(12), 300),
        save_path=f"{name.lower().replace(' ', '_')}_malliavin.gif",
    )
    create_transport_gif(
        traj_s,
        source_samples=problem.sample_source(jax.random.PRNGKey(13), 300),
        target_samples=problem.sample_target(jax.random.PRNGKey(14), 300),
        save_path=f"{name.lower().replace(' ', '_')}_score.gif",
    )


def main():
    train_cfg = TrainingConfig(
        num_iterations=200,
        batch_size=64,
        eval_every=20,
        patience=50,
    )

    gaussian_problem = SBProblem(
        reference=BrownianMotion(sigma=0.5, dim=2),
        source=GaussianDistribution(
            mean=jnp.array([-2.0, 0.0]),
            cov=0.3,
            dim=2,
        ),
        target=GaussianDistribution(
            mean=jnp.array([2.0, 0.0]),
            cov=0.3,
            dim=2,
        ),
        time_grid=TimeGrid(num_steps=30),
        name="GaussianToGaussian",
    )

    moons_problem = SBProblem(
        reference=BrownianMotion(sigma=0.5, dim=2),
        source=GaussianDistribution(dim=2),
        target=TwoMoonsDistribution(noise=0.05),
        time_grid=TimeGrid(num_steps=30),
        name="GaussianToTwoMoons",
    )

    run_case("Gaussian To Gaussian", gaussian_problem, train_cfg)
    run_case("Gaussian To TwoMoons", moons_problem, train_cfg)


if __name__ == "__main__":
    main()
