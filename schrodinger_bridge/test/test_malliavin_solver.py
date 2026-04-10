#!/usr/bin/env python3
"""Smoke tests for the Malliavin score solver."""

import jax
import jax.numpy as jnp

from schrodinger_bridge import (
    SBProblem,
    TimeGrid,
    BrownianMotion,
    GaussianDistribution,
    TwoMoonsDistribution,
    MalliavinScoreSolver,
    MalliavinConfig,
    ScoreBasedSolver,
    ScoreBasedConfig,
    TrainingConfig,
)


def test_malliavin_smoke():
    problem = SBProblem(
        reference=BrownianMotion(sigma=0.5, dim=2),
        source=GaussianDistribution(dim=2),
        target=TwoMoonsDistribution(noise=0.05),
        time_grid=TimeGrid(num_steps=20),
    )

    solver = MalliavinScoreSolver(
        problem,
        MalliavinConfig(
            hidden_dims=(16, 16),
            reward_bandwidth=0.3,
        ),
    )

    res = solver.train(
        jax.random.PRNGKey(0),
        TrainingConfig(num_iterations=1, batch_size=8, eval_every=1, patience=5),
    )
    traj = solver.sample(jax.random.PRNGKey(1), num_samples=8)

    assert "loss" in res.diagnostics.metadata or res.loss_history.shape[0] >= 1
    assert traj.paths.shape == (8, 21, 2)


def test_malliavin_compare_api_to_score_solver():
    problem = SBProblem(
        reference=BrownianMotion(sigma=0.5, dim=2),
        source=GaussianDistribution(
            mean=jnp.array([-1.0, 0.0]),
            cov=0.5,
            dim=2,
        ),
        target=GaussianDistribution(
            mean=jnp.array([1.0, 0.0]),
            cov=0.5,
            dim=2,
        ),
        time_grid=TimeGrid(num_steps=10),
    )

    cfg = TrainingConfig(num_iterations=1, batch_size=8, eval_every=1, patience=5)

    malliavin = MalliavinScoreSolver(
        problem,
        MalliavinConfig(hidden_dims=(16, 16), reward_bandwidth=0.4),
    )
    score = ScoreBasedSolver(
        problem,
        ScoreBasedConfig(hidden_dims=(16, 16), learning_rate=1e-3),
    )

    malliavin.train(jax.random.PRNGKey(2), cfg)
    score.train(jax.random.PRNGKey(3), cfg)

    traj_m = malliavin.sample(jax.random.PRNGKey(4), num_samples=4)
    traj_s = score.sample(jax.random.PRNGKey(5), num_samples=4)

    assert traj_m.paths.shape == traj_s.paths.shape == (4, 11, 2)

