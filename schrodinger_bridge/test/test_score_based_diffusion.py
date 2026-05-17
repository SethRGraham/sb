import jax
import jax.numpy as jnp

from schrodinger_bridge.core.problem import (
    GaussianDistribution,
    ReferenceDynamics,
    SBProblem,
    TimeGrid,
)
from schrodinger_bridge.solvers.score_based import ScoreBasedConfig, ScoreBasedSolver


class DiagonalReference(ReferenceDynamics):
    def __init__(self):
        self._sigma = jnp.array([0.5, 1.25])

    def drift(self, x, t):
        return jnp.zeros_like(jnp.atleast_2d(x))

    def diffusion(self, x, t):
        x = jnp.atleast_2d(x)
        return jnp.broadcast_to(self._sigma, x.shape)

    @property
    def is_time_homogeneous(self):
        return True

    @property
    def is_diffusion_scalar(self):
        return False

    @property
    def dim(self):
        return 2


class MatrixReference(ReferenceDynamics):
    def __init__(self):
        self._sigma = jnp.array([[0.5, 0.2], [0.0, 1.1]])

    def drift(self, x, t):
        return jnp.zeros_like(jnp.atleast_2d(x))

    def diffusion(self, x, t):
        return self._sigma

    @property
    def is_time_homogeneous(self):
        return True

    @property
    def is_diffusion_scalar(self):
        return False

    @property
    def dim(self):
        return 2


class CovarianceReference(MatrixReference):
    def __init__(self):
        self._sigma = jnp.array([[0.8, 0.15], [0.15, 1.4]])


def _make_problem(reference):
    return SBProblem(
        reference=reference,
        source=GaussianDistribution(dim=2),
        target=GaussianDistribution(
            mean=jnp.array([0.5, -0.25]),
            cov=0.7,
            dim=2,
        ),
        time_grid=TimeGrid(num_steps=4),
    )


def _exercise_solver(reference, batch_size, config=None):
    problem = _make_problem(reference)
    solver = ScoreBasedSolver(
        problem,
        config or ScoreBasedConfig(hidden_dims=(8,), time_embed_dim=8),
    )
    params = solver.init_params(jax.random.PRNGKey(0))
    x0, x1 = problem.sample_pair(jax.random.PRNGKey(1), batch_size)
    t = jnp.linspace(0.2, 0.8, batch_size)

    x_t, score = solver._sample_bridge_point(jax.random.PRNGKey(2), x0, x1, t)
    loss, metrics = solver._loss_fn(params, jax.random.PRNGKey(3), x0, x1)
    drift = solver.extract_drift(params)(x_t, t)

    assert x_t.shape == (batch_size, problem.dim)
    assert score.shape == x_t.shape
    assert drift.shape == x_t.shape
    assert jnp.all(jnp.isfinite(x_t))
    assert jnp.all(jnp.isfinite(score))
    assert jnp.all(jnp.isfinite(loss))
    assert jnp.all(jnp.isfinite(metrics["score_norm"]))
    assert jnp.all(jnp.isfinite(drift))


def test_score_based_solver_accepts_diagonal_diffusion():
    _exercise_solver(DiagonalReference(), batch_size=3)


def test_score_based_solver_accepts_full_matrix_diffusion_when_batch_matches_dim():
    _exercise_solver(MatrixReference(), batch_size=2)


def test_score_based_solver_accepts_full_covariance_matrix():
    _exercise_solver(
        CovarianceReference(),
        batch_size=2,
        config=ScoreBasedConfig(
            hidden_dims=(8,),
            time_embed_dim=8,
            diffusion_matrix_is_covariance=True,
        ),
    )
