import jax
import jax.numpy as jnp

from schrodinger_bridge import (
    BrownianMotion,
    GaussianDistribution,
    IPFDSBConfig,
    IPFDSBSolver,
    NetworkFactory,
    ReferenceDynamics,
    SBProblem,
    SolverConfig,
    TimeGrid,
    TrainingConfig,
    get_solver,
)


class ConstantDriftReference(ReferenceDynamics):
    def drift(self, x, t):
        del t
        return jnp.ones_like(jnp.atleast_2d(x)) * 0.75

    def diffusion(self, x, t):
        del x, t
        return 0.0

    @property
    def is_time_homogeneous(self):
        return True

    @property
    def dim(self):
        return 2


class ShiftMeanMapFactory(NetworkFactory):
    def init(self, key, input_dim, output_dim):
        del key, input_dim
        return {"shift": jnp.zeros((output_dim,))}

    def forward(self, params, x, t):
        del t
        return x + params["shift"][None, :]


def _small_problem(reference=None):
    return SBProblem(
        reference=reference or BrownianMotion(sigma=0.5, dim=2),
        source=GaussianDistribution(dim=2),
        target=GaussianDistribution(
            mean=jnp.array([0.5, -0.25]),
            cov=0.7,
            dim=2,
        ),
        time_grid=TimeGrid(num_steps=3),
    )


def _small_config():
    return IPFDSBConfig(
        N=3,
        gamma=1.0 / 3.0,
        hidden_dims=(8,),
        time_embed_dim=8,
        learning_rate=1e-3,
        inner_steps=1,
        batch_size=8,
        cache_size=8,
        n_ipf_iterations=1,
        ema_decay=0.0,
    )


def _small_solver(problem):
    return IPFDSBSolver(
        problem,
        _small_config(),
        solver_config=SolverConfig(verbose=0),
    )


def _assert_trees_allclose(left, right):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        assert jnp.allclose(left_leaf, right_leaf)


def test_reference_forward_map_is_mean_map_not_raw_drift():
    problem = _small_problem(ConstantDriftReference())
    solver = _small_solver(problem)
    params = solver.init_params(jax.random.PRNGKey(0))
    x0 = jnp.zeros((2, 2))

    paths = solver._sample_forward_chain(
        jax.random.PRNGKey(1),
        x0,
        params,
        F_is_reference=True,
    )

    expected = jnp.arange(4, dtype=jnp.float32)[:, None] * (solver.gamma * 0.75)
    expected = jnp.broadcast_to(expected[None, :, :], paths.shape)

    assert paths.shape == (2, 4, 2)
    assert jnp.allclose(paths, expected)


def test_ipf_dsb_trains_two_mean_maps_and_samples_forward_and_backward():
    problem = _small_problem()
    solver = _small_solver(problem)

    result = solver.train(jax.random.PRNGKey(2))
    forward = solver.sample(jax.random.PRNGKey(3), 4)
    backward = solver.sample_backward(jax.random.PRNGKey(4), 4)
    drift = solver.extract_drift(result.params)
    x = problem.sample_source(jax.random.PRNGKey(5), 4)
    t = jnp.linspace(0.0, 0.9, 4)

    assert set(result.params) == {"F", "B"}
    assert result.loss_history.shape == (2,)
    assert result.metadata["solver_type"] == "DSB"
    assert forward.paths.shape == (4, 4, 2)
    assert backward.paths.shape == (4, 4, 2)
    assert forward.times.shape == (4,)
    assert drift(x, t).shape == x.shape
    assert jnp.all(jnp.isfinite(result.loss_history))
    assert jnp.all(jnp.isfinite(forward.paths))
    assert jnp.all(jnp.isfinite(backward.paths))


def test_extract_drift_uses_forward_and_backward_mean_maps():
    problem = _small_problem()
    config = IPFDSBConfig(
        N=2,
        gamma=0.5,
        network_factory=ShiftMeanMapFactory(),
    )
    solver = IPFDSBSolver(problem, config, solver_config=SolverConfig(verbose=0))
    params = {
        "F": {"shift": jnp.array([0.25, -0.5])},
        "B": {"shift": jnp.array([-0.75, 0.5])},
    }
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    t = jnp.array([0.0, 0.8])

    forward_drift = solver.extract_drift(params, direction="forward")
    backward_drift = solver.extract_drift(params, direction="backward")
    single_forward = forward_drift(x[0], 0.0)

    assert jnp.allclose(forward_drift(x, t), jnp.array([[0.5, -1.0], [0.5, -1.0]]))
    assert jnp.allclose(backward_drift(x, t), jnp.array([[-1.5, 1.0], [-1.5, 1.0]]))
    assert single_forward.shape == (2,)
    assert jnp.allclose(single_forward, jnp.array([0.5, -1.0]))


def test_ipf_dsb_checkpoint_restores_raw_and_ema_params(tmp_path):
    problem = _small_problem()
    solver = _small_solver(problem)
    train_config = TrainingConfig(
        batch_size=8,
        eval_every=10,
        checkpoint_every=1,
        checkpoint_dir=str(tmp_path),
    )

    result = solver.train(jax.random.PRNGKey(6), train_config)
    restored = _small_solver(problem)
    payload = restored.load_checkpoint(result.metadata["checkpoint_path"])

    step_files = sorted(tmp_path.glob("checkpoint_dsb_step_*.pkl"))
    final_files = sorted(tmp_path.glob("checkpoint_dsb_final.pkl"))

    assert payload["solver_class"] == "IPFDSBSolver"
    assert payload["step"] == 2
    assert len(step_files) == 2
    assert len(final_files) == 1
    assert restored._is_trained
    assert not restored._F_is_reference
    _assert_trees_allclose(restored.get_trained_params(use_ema=False), result.params)
    _assert_trees_allclose(
        restored.get_trained_params(use_ema=True),
        solver._ema_params,
    )


def test_get_solver_accepts_dsb_aliases():
    problem = _small_problem()

    assert isinstance(
        get_solver("ipf_dsb", problem, dsb_config=_small_config()),
        IPFDSBSolver,
    )
    assert isinstance(
        get_solver("dsb", problem, dsb_config=_small_config()),
        IPFDSBSolver,
    )
    assert isinstance(
        get_solver("diffusion_sb", problem, dsb_config=_small_config()),
        IPFDSBSolver,
    )
