import jax
import jax.numpy as jnp

from schrodinger_bridge import (
    BrownianMotion,
    GaussianDistribution,
    SBProblem,
    ScoreBasedConfig,
    ScoreBasedSolver,
    SolverConfig,
    TimeGrid,
    TrainingConfig,
)


def _small_problem():
    return SBProblem(
        reference=BrownianMotion(sigma=0.5, dim=2),
        source=GaussianDistribution(dim=2),
        target=GaussianDistribution(
            mean=jnp.array([0.5, -0.25]),
            cov=0.7,
            dim=2,
        ),
        time_grid=TimeGrid(num_steps=4),
    )


def _small_solver(problem):
    return ScoreBasedSolver(
        problem,
        ScoreBasedConfig(hidden_dims=(8,), time_embed_dim=8, learning_rate=1e-3),
        solver_config=SolverConfig(verbose=0),
    )


def _assert_trees_allclose(left, right):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        assert jnp.allclose(left_leaf, right_leaf)


def test_training_saves_periodic_and_final_checkpoints(tmp_path):
    problem = _small_problem()
    solver = _small_solver(problem)
    config = TrainingConfig(
        num_iterations=1,
        batch_size=8,
        eval_every=1,
        checkpoint_every=1,
        checkpoint_dir=str(tmp_path),
        patience=5,
    )

    result = solver.train(jax.random.PRNGKey(0), config)

    step_files = sorted(tmp_path.glob("checkpoint_score_based_step_*.pkl"))
    final_files = sorted(tmp_path.glob("checkpoint_score_based_final.pkl"))

    assert len(step_files) == 1
    assert len(final_files) == 1
    assert result.metadata["checkpoint_path"] == str(final_files[0])
    assert result.metadata["checkpoint_paths"] == [str(step_files[0]), str(final_files[0])]


def test_load_checkpoint_restores_score_solver_state(tmp_path):
    problem = _small_problem()
    solver = _small_solver(problem)
    config = TrainingConfig(
        num_iterations=1,
        batch_size=8,
        eval_every=1,
        checkpoint_every=1,
        checkpoint_dir=str(tmp_path),
        patience=5,
    )
    result = solver.train(jax.random.PRNGKey(1), config)

    restored = _small_solver(problem)
    payload = restored.load_checkpoint(result.metadata["checkpoint_path"])

    x = problem.sample_source(jax.random.PRNGKey(2), 4)
    t = jnp.linspace(0.2, 0.8, 4)
    original_score = solver.get_score_fn()(x, t)
    restored_score = restored.get_score_fn()(x, t)

    assert payload["solver_class"] == "ScoreBasedSolver"
    assert payload["step"] == 1
    assert restored._is_trained
    _assert_trees_allclose(restored.get_trained_params(use_ema=False), result.params)
    _assert_trees_allclose(restored.get_trained_params(use_ema=True), solver._ema_params)
    assert jnp.allclose(restored_score, original_score)
    assert restored_score.shape == x.shape
    assert jnp.all(jnp.isfinite(restored_score))
