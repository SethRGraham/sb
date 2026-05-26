import jax
import jax.numpy as jnp

from schrodinger_bridge import (
    BrownianMotion,
    DSBMConfig,
    DSBMSolver,
    GaussianDistribution,
    SBProblem,
    SolverConfig,
    TimeGrid,
    TrainingConfig,
    get_solver,
)


def _problem():
    return SBProblem(
        reference=BrownianMotion(sigma=0.4, dim=2),
        source=GaussianDistribution(dim=2),
        target=GaussianDistribution(
            mean=jnp.array([1.0, -0.5]),
            cov=0.6,
            dim=2,
        ),
        time_grid=TimeGrid(num_steps=4),
    )


def _config():
    return DSBMConfig(
        num_steps=4,
        sigma=0.4,
        hidden_dims=(16, 16),
        time_embed_dim=8,
        learning_rate=1e-3,
        n_ipf_iterations=1,
        inner_steps=2,
        batch_size=8,
        cache_size=16,
        first_coupling="ind",
    )


def test_dsbm_imports_and_factory_lookup():
    problem = _problem()
    solver = get_solver(
        "dsbm",
        problem,
        dsbm_config=DSBMConfig(
            num_steps=2,
            sigma=0.4,
            hidden_dims=(8,),
            time_embed_dim=4,
            inner_steps=1,
            cache_size=8,
        ),
        solver_config=SolverConfig(verbose=0),
    )
    assert isinstance(solver, DSBMSolver)


def test_dsbm_short_train_sample_and_drift():
    problem = _problem()
    solver = DSBMSolver(
        problem,
        dsbm_config=_config(),
        solver_config=SolverConfig(verbose=0),
    )
    result = solver.train(
        jax.random.PRNGKey(0),
        TrainingConfig(
            num_iterations=1,
            batch_size=8,
            checkpoint_dir=None,
            eval_every=100,
            patience=10,
        ),
    )

    assert result.params["F"] is not None
    assert result.params["B"] is not None
    assert result.loss_history.shape[0] == 4

    paths = solver.sample(jax.random.PRNGKey(1), num_samples=5)
    assert paths.paths.shape == (5, 5, 2)
    assert paths.times.shape == (5,)
    assert jnp.all(jnp.isfinite(paths.paths))

    drift = solver.extract_drift(result.params)
    value = drift(jnp.zeros((3, 2)), 0.5)
    assert value.shape == (3, 2)
    assert jnp.all(jnp.isfinite(value))
