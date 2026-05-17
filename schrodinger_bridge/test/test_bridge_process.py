import pytest

import jax
import jax.numpy as jnp

from schrodinger_bridge import (
    BridgeProcess,
    BrownianMotion,
    DoobConfig,
    DoobHTransformSolver,
    GaussianDistribution,
    RepresentationType,
    SBProblem,
    ScoreBasedConfig,
    ScoreBasedSolver,
    SolverType,
    TimeGrid,
    TrainingConfig,
    create_gaussian_to_gaussian,
)


def test_solution_as_process_sampling_api():
    problem = create_gaussian_to_gaussian(
        source_mean=jnp.array([-1.0, 0.0]),
        source_cov=0.5,
        target_mean=jnp.array([1.0, 0.0]),
        target_cov=0.75,
        sigma=0.3,
        time_grid=TimeGrid(num_steps=12),
    )

    solver = DoobHTransformSolver(problem, DoobConfig(method="analytical"))
    solution = solver.solve(jax.random.PRNGKey(0))
    process = solution.as_process()

    coeffs = process.coefficients()
    drift = coeffs.drift(jnp.zeros((3, problem.dim)), 0.4)

    assert drift.shape == (3, problem.dim)

    paths = process.sample_paths(jax.random.PRNGKey(1), num_samples=6)
    assert paths.paths.shape == (6, 13, problem.dim)

    endpoint = process.sample_endpoint(jax.random.PRNGKey(2), num_samples=6)
    assert endpoint.shape == (6, problem.dim)

    marginal = process.sample_marginal(jax.random.PRNGKey(3), t=0.5, num_samples=6)
    assert marginal.shape == (6, problem.dim)

    solution_endpoint = solution.sample_endpoint(jax.random.PRNGKey(4), num_samples=6)
    assert solution_endpoint.shape == (6, problem.dim)


def test_bridge_process_probability_flow_sampling():
    problem = SBProblem(
        reference=BrownianMotion(sigma=0.4, dim=2),
        source=GaussianDistribution(dim=2),
        target=GaussianDistribution(mean=jnp.array([1.0, 1.0]), cov=0.5, dim=2),
        time_grid=TimeGrid(num_steps=10),
    )

    process = BridgeProcess(
        problem=problem,
        solver_type=SolverType.SCORE_BASED,
        representation_type=RepresentationType.SCORE,
        params=None,
        forward_drift_fn=lambda x, t: jnp.ones_like(jnp.atleast_2d(x)),
        backward_drift_fn=lambda x, t: -jnp.ones_like(jnp.atleast_2d(x)),
        score_fn=lambda x, t: jnp.ones_like(jnp.atleast_2d(x)),
    )

    pf_drift = process.probability_flow_drift(jnp.zeros((2, 2)), 0.25)
    assert pf_drift.shape == (2, 2)
    assert jnp.all(jnp.isfinite(pf_drift))

    flow_paths = process.sample_flow(jax.random.PRNGKey(5), num_samples=4)
    assert flow_paths.paths.shape == (4, 11, 2)

    reverse_process = process.reverse()
    reverse_paths = reverse_process.sample_paths(jax.random.PRNGKey(6), num_samples=4)
    assert reverse_paths.paths.shape == (4, 11, 2)


def test_bridge_process_reverse_fails_when_unavailable():
    problem = create_gaussian_to_gaussian(
        source_mean=jnp.array([-0.5, 0.0]),
        source_cov=0.5,
        target_mean=jnp.array([0.5, 0.0]),
        target_cov=0.75,
        sigma=0.25,
        time_grid=TimeGrid(num_steps=8),
    )

    solver = DoobHTransformSolver(problem, DoobConfig(method="analytical"))
    process = solver.solve(jax.random.PRNGKey(7)).as_process()

    assert not process.has_reverse()

    try:
        process.reverse()
    except ValueError as exc:
        assert "Backward drift not available" in str(exc)
    else:
        raise AssertionError("Expected reverse() to fail when no backward drift exists.")


def test_score_solution_exposes_bridge_process_probability_flow():
    problem = SBProblem(
        reference=BrownianMotion(sigma=0.3, dim=2),
        source=GaussianDistribution(dim=2),
        target=GaussianDistribution(mean=jnp.array([1.0, -1.0]), cov=0.6, dim=2),
        time_grid=TimeGrid(num_steps=10),
    )

    solver = ScoreBasedSolver(
        problem,
        ScoreBasedConfig(hidden_dims=(16, 16), learning_rate=1e-3),
    )
    solution = solver.solve(
        jax.random.PRNGKey(8),
        TrainingConfig(num_iterations=1, batch_size=8, eval_every=1, patience=5),
    )
    process = solution.as_process()

    assert process.has_score()
    assert not process.has_reverse()

    score = process.score(jnp.zeros((4, 2)), 0.5)
    assert score.shape == (4, 2)
    assert jnp.all(jnp.isfinite(score))

    flow_paths = process.sample_flow(jax.random.PRNGKey(9), num_samples=4)
    assert flow_paths.paths.shape == (4, 11, 2)


def test_diffrax_backend_sampling_api():
    pytest.importorskip("diffrax")

    problem = create_gaussian_to_gaussian(
        source_mean=jnp.array([-1.0, 0.0]),
        source_cov=0.5,
        target_mean=jnp.array([1.0, 0.0]),
        target_cov=0.75,
        sigma=0.3,
        time_grid=TimeGrid(num_steps=12),
    )

    solver = DoobHTransformSolver(problem, DoobConfig(method="analytical"))
    process = solver.solve(jax.random.PRNGKey(10)).as_process(backend="diffrax")

    paths = process.sample_paths(jax.random.PRNGKey(11), num_samples=4)
    assert paths.paths.shape == (4, 13, problem.dim)

    endpoint = process.sample_endpoint(jax.random.PRNGKey(12), num_samples=4)
    assert endpoint.shape == (4, problem.dim)

    marginal = process.sample_marginal(jax.random.PRNGKey(13), t=0.5, num_samples=4)
    assert marginal.shape == (4, problem.dim)


def test_diffrax_backend_probability_flow_and_reverse():
    pytest.importorskip("diffrax")

    problem = SBProblem(
        reference=BrownianMotion(sigma=0.4, dim=2),
        source=GaussianDistribution(dim=2),
        target=GaussianDistribution(mean=jnp.array([1.0, 1.0]), cov=0.5, dim=2),
        time_grid=TimeGrid(num_steps=10),
    )

    process = BridgeProcess(
        problem=problem,
        solver_type=SolverType.SCORE_BASED,
        representation_type=RepresentationType.SCORE,
        params=None,
        forward_drift_fn=lambda x, t: jnp.ones_like(jnp.atleast_2d(x)),
        backward_drift_fn=lambda x, t: -jnp.ones_like(jnp.atleast_2d(x)),
        score_fn=lambda x, t: jnp.ones_like(jnp.atleast_2d(x)),
        backend="diffrax",
    )

    flow_paths = process.sample_flow(jax.random.PRNGKey(14), num_samples=4)
    assert flow_paths.paths.shape == (4, 11, 2)

    reverse_paths = process.reverse().sample_paths(jax.random.PRNGKey(15), num_samples=4)
    assert reverse_paths.paths.shape == (4, 11, 2)
