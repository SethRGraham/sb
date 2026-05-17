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


def _bel_test_solver(alpha_mode="uniform"):
    problem = SBProblem(
        reference=BrownianMotion(sigma=1.0, dim=1),
        source=GaussianDistribution(dim=1),
        target=GaussianDistribution(dim=1),
        time_grid=TimeGrid(t0=0.0, t1=2.0, num_steps=4),
    )
    return MalliavinScoreSolver(
        problem,
        MalliavinConfig(alpha_mode=alpha_mode),
    )


def test_malliavin_alpha_weights_are_alpha_prime_not_preaveraged():
    solver = _bel_test_solver("uniform")
    dt = solver.problem.time_grid.dt

    assert jnp.allclose(solver._alpha_weights(4, dt), jnp.ones((4,)))
    assert jnp.allclose(
        solver._alpha_normalizers(jnp.ones((4,)), dt),
        jnp.array([2.0, 1.5, 1.0, 0.5]),
    )


def test_malliavin_bel_targets_use_A_T_s_normalization():
    paths = jnp.zeros((1, 5, 1))
    dB = jnp.array([[[1.0], [2.0], [3.0], [4.0]]])
    local_jacobians = jnp.ones((1, 4, 1, 1))

    uniform = _bel_test_solver("uniform")._estimate_bel_targets(
        paths,
        dB,
        local_jacobians,
    )
    first = _bel_test_solver("first")._estimate_bel_targets(
        paths,
        dB,
        local_jacobians,
    )
    last = _bel_test_solver("last")._estimate_bel_targets(
        paths,
        dB,
        local_jacobians,
    )

    assert jnp.allclose(uniform[0, :, 0], jnp.array([5.0, 6.0, 7.0, 8.0]))
    assert jnp.allclose(first[0, :, 0], jnp.array([2.0, 0.0, 0.0, 0.0]))
    assert jnp.allclose(last[0, :, 0], jnp.array([8.0, 8.0, 8.0, 8.0]))


def test_malliavin_train_step_reports_ess_and_reuses_reference_bank():
    problem = SBProblem(
        reference=BrownianMotion(sigma=0.5, dim=2),
        source=GaussianDistribution(dim=2),
        target=GaussianDistribution(
            mean=jnp.array([0.5, -0.25]),
            cov=0.7,
            dim=2,
        ),
        time_grid=TimeGrid(num_steps=2),
    )
    solver = MalliavinScoreSolver(
        problem,
        MalliavinConfig(
            hidden_dims=(8,),
            reference_bank_size=12,
            reference_bank_refresh_every=2,
            reward_bandwidth=0.5,
        ),
    )
    params = solver.init_params(jax.random.PRNGKey(0))
    opt_state = solver._init_optimizer(params)

    params, opt_state, metrics = solver.train_step(
        jax.random.PRNGKey(1),
        params,
        opt_state,
        batch_size=4,
    )
    first_bank = solver._reference_bank

    assert first_bank.shape == (12, 2)
    assert solver._reference_bank_age == 0
    assert "ess_fraction" in metrics
    assert 0.0 < float(metrics["ess_fraction"]) <= 1.0

    params, opt_state, _ = solver.train_step(
        jax.random.PRNGKey(2),
        params,
        opt_state,
        batch_size=4,
    )
    assert solver._reference_bank is first_bank
    assert solver._reference_bank_age == 1

    params, opt_state, _ = solver.train_step(
        jax.random.PRNGKey(3),
        params,
        opt_state,
        batch_size=4,
    )
    assert solver._reference_bank is first_bank
    assert solver._reference_bank_age == 2

    solver.train_step(
        jax.random.PRNGKey(4),
        params,
        opt_state,
        batch_size=4,
    )
    assert solver._reference_bank is not first_bank
    assert solver._reference_bank_age == 0


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
