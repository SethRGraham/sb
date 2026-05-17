import jax
import jax.numpy as jnp

from schrodinger_bridge.core.diffusion import (
    apply_diffusion,
    apply_diffusion_covariance,
    solve_diffusion_coefficient,
    solve_diffusion_covariance,
)
from schrodinger_bridge.core.problem import GaussianDistribution, ReferenceDynamics, SBProblem, TimeGrid
from schrodinger_bridge.solvers.doob import DoobConfig, DoobHTransformSolver
from schrodinger_bridge.solvers.fbsde import FBSDEConfig, FBSDESolver
from schrodinger_bridge.solvers.imf import IMFConfig, IMFSolver
from schrodinger_bridge.solvers.ipf import IPFConfig, IPFSolver
from schrodinger_bridge.solvers.ipf_dsb import IPFDSBConfig, IPFDSBSolver
from schrodinger_bridge.solvers.malliavin import MalliavinConfig, MalliavinScoreSolver
from schrodinger_bridge.solvers.mirror_descent_ipf import (
    MirrorDescentIPFConfig,
    MirrorDescentIPFSolver,
)
from schrodinger_bridge.solvers.rkhs import RKHSConfig, RKHSSolver


class MatrixReference(ReferenceDynamics):
    def __init__(self):
        self._sigma = jnp.array([[0.7, 0.2], [0.1, 1.1]])

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


class StateDependentMatrixReference(ReferenceDynamics):
    def __init__(self, scale=0.1):
        self.scale = scale

    def drift(self, x, t):
        return jnp.zeros_like(jnp.atleast_2d(x))

    def diffusion(self, x, t):
        x = jnp.atleast_2d(x)
        batch = x.shape[0]
        sigma = jnp.broadcast_to(jnp.eye(2), (batch, 2, 2))
        sigma = sigma.at[:, 0, 0].set(1.0 + self.scale * x[:, 0])
        sigma = sigma.at[:, 1, 1].set(1.0 + self.scale * x[:, 1])
        return sigma

    @property
    def is_time_homogeneous(self):
        return False

    @property
    def is_diffusion_scalar(self):
        return False

    @property
    def dim(self):
        return 2


def _make_problem():
    return SBProblem(
        reference=MatrixReference(),
        source=GaussianDistribution(dim=2),
        target=GaussianDistribution(mean=jnp.array([0.5, -0.2]), cov=0.8, dim=2),
        time_grid=TimeGrid(num_steps=3),
    )


def _sample_pair(problem, batch_size=3):
    return problem.sample_pair(jax.random.PRNGKey(11), batch_size)


def _assert_finite_shape(value, shape):
    assert value.shape == shape
    assert jnp.all(jnp.isfinite(value))


def test_diffusion_helpers_apply_and_solve_full_matrix():
    sigma = jnp.array([[0.7, 0.2], [0.1, 1.1]])
    vector = jnp.array([[1.0, -0.5], [0.25, 2.0]])
    cov = sigma @ sigma.T

    _assert_finite_shape(apply_diffusion(sigma, vector), vector.shape)
    assert jnp.allclose(apply_diffusion(sigma, vector), vector @ sigma.T)
    assert jnp.allclose(apply_diffusion_covariance(sigma, vector), vector @ cov.T)
    assert jnp.allclose(solve_diffusion_coefficient(sigma, vector) @ sigma.T, vector)
    assert jnp.allclose(
        apply_diffusion_covariance(sigma, solve_diffusion_covariance(sigma, vector)),
        vector,
        atol=1e-5,
    )


def test_malliavin_accepts_full_matrix_diffusion_rollout_and_drift():
    problem = _make_problem()
    solver = MalliavinScoreSolver(
        problem,
        MalliavinConfig(
            hidden_dims=(8,),
            time_embed_dim=8,
            reference_bank_size=4,
            reference_bank_refresh_every=10,
        ),
    )
    params = solver.init_params(jax.random.PRNGKey(0))
    x0, _ = _sample_pair(problem)

    paths, dB, jac = solver._simulate_reference_rollout(jax.random.PRNGKey(1), x0)
    targets = solver._estimate_bel_targets(paths, dB, jac)
    drift = solver.extract_drift(params)(x0, jnp.full((x0.shape[0],), 0.4))

    _assert_finite_shape(targets, (x0.shape[0], problem.time_grid.num_steps, problem.dim))
    _assert_finite_shape(drift, x0.shape)


def test_malliavin_diffusion_jacobian_is_opt_in_for_state_dependent_matrix():
    problem = SBProblem(
        reference=StateDependentMatrixReference(scale=0.2),
        source=GaussianDistribution(dim=2),
        target=GaussianDistribution(mean=jnp.array([0.5, -0.2]), cov=0.8, dim=2),
        time_grid=TimeGrid(num_steps=2),
    )
    x0 = jnp.array([[0.3, -0.4], [0.2, 0.1]])

    without_term = MalliavinScoreSolver(
        problem,
        MalliavinConfig(hidden_dims=(8,), time_embed_dim=8),
    )
    _, dB, jac_without = without_term._simulate_reference_rollout(
        jax.random.PRNGKey(21),
        x0,
    )

    with_term = MalliavinScoreSolver(
        problem,
        MalliavinConfig(
            hidden_dims=(8,),
            time_embed_dim=8,
            include_diffusion_jacobian=True,
        ),
    )
    _, dB_enabled, jac_with = with_term._simulate_reference_rollout(
        jax.random.PRNGKey(21),
        x0,
    )

    expected_extra = jnp.zeros_like(jac_with[:, 0])
    expected_extra = expected_extra.at[:, 0, 0].set(0.2 * dB[:, 0, 0])
    expected_extra = expected_extra.at[:, 1, 1].set(0.2 * dB[:, 0, 1])

    assert jnp.allclose(dB, dB_enabled)
    assert jnp.allclose(jac_without[:, 0], jnp.broadcast_to(jnp.eye(2), (2, 2, 2)))
    assert jnp.allclose(jac_with[:, 0], jac_without[:, 0] + expected_extra)


def test_kernel_and_matching_solvers_accept_full_matrix_bridge_noise():
    problem = _make_problem()
    x0, x1 = _sample_pair(problem)
    t = jnp.array([0.2, 0.5, 0.8])

    imf = IMFSolver(problem, IMFConfig(hidden_dims=(8,), time_embed_dim=8))
    x_imf, v_imf = imf._sample_conditional_path(jax.random.PRNGKey(2), x0, x1, t)
    _assert_finite_shape(x_imf, x0.shape)
    _assert_finite_shape(v_imf, x0.shape)

    md = MirrorDescentIPFSolver(
        problem,
        MirrorDescentIPFConfig(hidden_dims=(8,), num_md_iterations=2),
    )
    x_md, v_md = md._sample_bridge_path(jax.random.PRNGKey(3), x0, x1, t)
    _assert_finite_shape(x_md, x0.shape)
    _assert_finite_shape(v_md, x0.shape)

    rkhs = RKHSSolver(problem, RKHSConfig(num_inducing=6, num_time_points=3))
    rkhs_params = rkhs.init_params(jax.random.PRNGKey(4))
    coeffs = rkhs._estimate_score_at_time(
        jax.random.PRNGKey(5),
        0.5,
        x0,
        x1,
        rkhs_params["inducing_points"],
        rkhs_params["bandwidth"],
    )
    rkhs._inducing_points = rkhs_params["inducing_points"]
    rkhs._time_points = rkhs_params["time_points"]
    rkhs._coefficients = jnp.broadcast_to(
        coeffs[None, :, :],
        (rkhs.rkhs_config.num_time_points, coeffs.shape[0], coeffs.shape[1]),
    )
    rkhs._bandwidth = rkhs_params["bandwidth"]
    drift = rkhs.extract_drift(rkhs_params)(x0, 0.5)
    _assert_finite_shape(drift, x0.shape)


def test_discrete_and_sde_solvers_accept_full_matrix_diffusion():
    problem = _make_problem()
    x0, _ = _sample_pair(problem)

    ipf = IPFSolver(problem, IPFConfig(hidden_dims=(8,)))
    ipf_params = ipf.init_params(jax.random.PRNGKey(6))
    ipf_drift = ipf._get_forward_drift(ipf_params)(x0, 0.4)
    _assert_finite_shape(ipf_drift, x0.shape)

    dsb = IPFDSBSolver(
        problem,
        IPFDSBConfig(N=3, hidden_dims=(8,), time_embed_dim=8, inner_steps=1, batch_size=3),
    )
    dsb.init_params(jax.random.PRNGKey(7))
    step_noise = dsb._step_noise(jax.random.PRNGKey(8), x0, jnp.array(0))
    _assert_finite_shape(step_noise, x0.shape)

    fbsde = FBSDESolver(
        problem,
        FBSDEConfig(hidden_dims=(8,), time_embed_dim=8, num_stages=1, steps_per_stage=1),
    )
    fbsde_params = fbsde.init_params(jax.random.PRNGKey(9))
    traj = fbsde._simulate_forward_sde(fbsde_params, jax.random.PRNGKey(10), x0)
    fbsde_drift = fbsde.extract_drift(fbsde_params)(x0, 0.4)
    _assert_finite_shape(traj, (x0.shape[0], problem.time_grid.num_steps + 1, problem.dim))
    _assert_finite_shape(fbsde_drift, x0.shape)


def test_doob_kernel_accepts_full_matrix_diffusion():
    problem = _make_problem()
    solver = DoobHTransformSolver(
        problem,
        DoobConfig(method="kernel", num_inducing_points=8),
    )
    params = solver.init_params(jax.random.PRNGKey(12))
    solver._params = params
    solver._is_trained = True

    x0, _ = _sample_pair(problem)
    drift = solver.compute_drift(x0, 0.4)
    grad = solver.compute_log_h_gradient(x0, 0.4)

    _assert_finite_shape(drift, x0.shape)
    _assert_finite_shape(grad, x0.shape)
