"""Falsification tests for the conditional Malliavin adjoint foundation."""

from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from schrodinger_bridge import (
    BrownianMotion,
    GaussianDistribution,
    SBProblem,
    TimeGrid,
)
from schrodinger_bridge.core.types import Array
from schrodinger_bridge.network_factory import NetworkFactory
from schrodinger_bridge.networks import init_adam
from schrodinger_bridge.solvers.malliavin_adjoint import (
    AdjointRollout,
    BELCostateBatch,
    MalliavinAdjointConfig,
    MalliavinAdjointInnerSolver,
    ValueOnlyCost,
    assemble_bel_costate_labels,
    assemble_pinned_brownian_labels,
    simulate_additive_em_rollout,
    simulate_pinned_brownian_rollout,
    summarize_costate_labels,
)


def _dtype():
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


def _tol():
    return 2e-11 if jax.config.x64_enabled else 2e-5


def _all_rademacher_increments(num_steps: int, dim: int, dt: float):
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=num_steps * dim)))
    return jnp.asarray(signs.reshape((-1, num_steps, dim)) * math.sqrt(dt), dtype=_dtype())


def _linear_rollout(local: Array, sigma: Array, increments: Array, dt: float):
    """Construct the exact fixed-innovation linear EM rollout."""
    del dt
    batch_size, num_steps, dim = increments.shape
    x = jnp.zeros((batch_size, dim), dtype=increments.dtype)
    states = [x]
    for step in range(num_steps):
        x = x @ local.T + increments[:, step, :] @ sigma.T
        states.append(x)
    local_batch = jnp.broadcast_to(
        local,
        (batch_size, num_steps, dim, dim),
    )
    sigma_steps = jnp.broadcast_to(sigma, (num_steps, dim, dim))
    return AdjointRollout(
        states=jnp.stack(states, axis=1),
        innovations=increments,
        local_jacobians=local_batch,
        noise_matrices=sigma_steps,
        controls=jnp.zeros((batch_size, num_steps, dim), dtype=increments.dtype),
        times=jnp.linspace(0.0, num_steps * 0.2, num_steps + 1, dtype=increments.dtype),
        context=jnp.zeros((batch_size, 0), dtype=increments.dtype),
    )


def test_em_ou_linear_payoff_uses_arrival_flow_jacobian():
    """The obsolete J[j,k] convention gives a^(L-1), not a^L."""
    num_steps = 4
    dt = 0.2
    a = 0.8
    sigma = jnp.asarray([[1.3]], dtype=_dtype())
    increments = _all_rademacher_increments(num_steps, 1, dt)
    rollout = _linear_rollout(jnp.asarray([[a]], dtype=_dtype()), sigma, increments, dt)
    # _linear_rollout uses the same fixed dt chosen above.
    rollout = rollout._replace(times=jnp.linspace(0.0, num_steps * dt, num_steps + 1))
    batch_size = rollout.states.shape[0]
    labels = assemble_bel_costate_labels(
        rollout,
        anchors=jnp.zeros((batch_size,), dtype=jnp.int32),
        running_values=jnp.zeros((batch_size, num_steps + 1), dtype=_dtype()),
        terminal_values=rollout.states[:, -1, 0],
    )
    np.testing.assert_allclose(
        np.asarray(jnp.mean(labels.label[:, 0])),
        a**num_steps,
        rtol=_tol(),
        atol=_tol(),
    )
    assert not np.isclose(float(jnp.mean(labels.label[:, 0])), a ** (num_steps - 1))


def test_em_full_matrix_uses_sigma_inverse_transpose_orientation():
    dt = 0.2
    local = jnp.asarray([[0.8, 0.2], [-0.1, 0.9]], dtype=_dtype())
    sigma = jnp.asarray([[1.0, 0.4], [-0.2, 0.8]], dtype=_dtype())
    increments = _all_rademacher_increments(1, 2, dt)
    rollout = _linear_rollout(local, sigma, increments, dt)
    c = jnp.asarray([0.6, -1.1], dtype=_dtype())
    terminal = rollout.states[:, -1, :] @ c
    labels = assemble_bel_costate_labels(
        rollout,
        anchors=jnp.zeros((rollout.states.shape[0],), dtype=jnp.int32),
        running_values=jnp.zeros((rollout.states.shape[0], 2), dtype=_dtype()),
        terminal_values=terminal,
    )
    expected = local.T @ c
    np.testing.assert_allclose(
        np.asarray(jnp.mean(labels.label, axis=0)),
        np.asarray(expected),
        rtol=_tol(),
        atol=_tol(),
    )


def test_em_running_cost_uses_arrival_weight_not_terminal_weight():
    num_steps = 4
    dt = 0.2
    a = 0.75
    sigma = jnp.asarray([[0.9]], dtype=_dtype())
    increments = _all_rademacher_increments(num_steps, 1, dt)
    rollout = _linear_rollout(jnp.asarray([[a]], dtype=_dtype()), sigma, increments, dt)
    rollout = rollout._replace(times=jnp.linspace(0.0, num_steps * dt, num_steps + 1))
    running = jnp.zeros((rollout.states.shape[0], num_steps + 1), dtype=_dtype())
    arrival = 2
    running = running.at[:, arrival].set(rollout.states[:, arrival, 0])
    labels = assemble_bel_costate_labels(
        rollout,
        anchors=jnp.zeros((rollout.states.shape[0],), dtype=jnp.int32),
        running_values=running,
        terminal_values=jnp.zeros((rollout.states.shape[0],), dtype=_dtype()),
    )
    np.testing.assert_allclose(
        np.asarray(jnp.mean(labels.running_component[:, 0])),
        dt * a**arrival,
        rtol=_tol(),
        atol=_tol(),
    )


def test_singular_state_tangent_is_diagnostic_not_invalid_label():
    dt = 0.2
    increments = _all_rademacher_increments(1, 1, dt)
    rollout = _linear_rollout(
        jnp.asarray([[0.0]], dtype=_dtype()),
        jnp.asarray([[1.0]], dtype=_dtype()),
        increments,
        dt,
    )
    labels = assemble_bel_costate_labels(
        rollout,
        anchors=jnp.zeros((2,), dtype=jnp.int32),
        running_values=jnp.zeros((2, 2), dtype=_dtype()),
        terminal_values=rollout.states[:, -1, 0],
    )
    np.testing.assert_array_equal(np.asarray(labels.label), np.zeros((2, 1)))
    assert bool(jnp.all(labels.finite))
    assert np.isinf(np.asarray(labels.tangent_condition_number)).all()
    assert summarize_costate_labels(labels)["finite_fraction"] == 1.0


def test_label_assembly_stops_terminal_and_running_cost_gradients():
    dt = 0.2
    increments = _all_rademacher_increments(2, 1, dt)
    rollout = _linear_rollout(
        jnp.asarray([[0.9]], dtype=_dtype()),
        jnp.asarray([[1.1]], dtype=_dtype()),
        increments,
        dt,
    )
    rollout = rollout._replace(times=jnp.linspace(0.0, 0.4, 3))
    anchors = jnp.zeros((rollout.states.shape[0],), dtype=jnp.int32)

    def label_sum(scale):
        terminal = scale * rollout.states[:, -1, 0]
        running = jnp.zeros((rollout.states.shape[0], 3), dtype=_dtype())
        running = running.at[:, 1].set(scale * rollout.states[:, 1, 0])
        return jnp.sum(assemble_bel_costate_labels(rollout, anchors, running, terminal).label)

    assert float(jax.grad(label_sum)(jnp.asarray(1.0, dtype=_dtype()))) == 0.0
    assert float(label_sum(jnp.asarray(1.0, dtype=_dtype()))) != 0.0


def test_em_rollout_tangent_includes_state_feedback_and_jit_matches_eager():
    dtype = _dtype()
    times = jnp.linspace(0.0, 0.6, 4, dtype=dtype)
    x0 = jnp.asarray([[0.2], [-0.3]], dtype=dtype)
    context = jnp.asarray([[0.4], [0.1]], dtype=dtype)

    def drift(x, t, c):
        return (0.3 + 0.1 * t) * x + 0.2 * x**2 + 0.05 * c

    args = (jax.random.PRNGKey(4), x0, times, drift, jnp.asarray(0.7, dtype=dtype), context)
    eager = simulate_additive_em_rollout(*args)
    compiled = jax.jit(simulate_additive_em_rollout, static_argnums=3)(*args)
    np.testing.assert_allclose(
        np.asarray(compiled.states), np.asarray(eager.states), rtol=_tol(), atol=_tol()
    )
    np.testing.assert_allclose(
        np.asarray(compiled.local_jacobians),
        np.asarray(eager.local_jacobians),
        rtol=_tol(),
        atol=_tol(),
    )

    fixed_innovations = eager.innovations[0]

    def replay(initial):
        x = initial
        for step, t in enumerate(times[:-1]):
            x = (
                x
                + (times[1] - times[0]) * drift(x[None, :], t, context[:1])[0]
                + 0.7 * fixed_innovations[step]
            )
        return x

    full_tangent = jax.jacfwd(replay)(x0[0])
    scan_tangent = jnp.eye(1, dtype=dtype)
    for local in eager.local_jacobians[0]:
        scan_tangent = local @ scan_tangent
    np.testing.assert_allclose(
        np.asarray(scan_tangent), np.asarray(full_tangent), rtol=_tol(), atol=_tol()
    )


def test_pinned_brownian_rollout_is_reproducible_and_pins_both_endpoints():
    dtype = _dtype()
    x0 = jnp.asarray([[-1.0, 0.2], [0.3, -0.5]], dtype=dtype)
    endpoint = jnp.asarray([[1.0, -0.1], [-0.4, 0.9]], dtype=dtype)
    times = jnp.linspace(0.0, 1.0, 6, dtype=dtype)
    sigma = jnp.asarray([[0.7, 0.1], [-0.2, 0.6]], dtype=dtype)
    first = simulate_pinned_brownian_rollout(jax.random.PRNGKey(8), x0, endpoint, times, sigma)
    second = simulate_pinned_brownian_rollout(jax.random.PRNGKey(8), x0, endpoint, times, sigma)
    np.testing.assert_array_equal(np.asarray(first.states), np.asarray(second.states))
    np.testing.assert_allclose(np.asarray(first.states[:, 0]), np.asarray(x0))
    np.testing.assert_allclose(np.asarray(first.states[:, -1]), np.asarray(endpoint))
    assert first.innovations.shape == (2, 4, 2)
    assert first.local_jacobians.shape == (2, 4, 2, 2)


def test_controlled_pinned_tangent_matches_fixed_innovation_autodiff():
    dtype = _dtype()
    x0 = jnp.asarray([[0.2]], dtype=dtype)
    endpoint = jnp.asarray([[1.1]], dtype=dtype)
    times = jnp.linspace(0.0, 1.0, 5, dtype=dtype)
    sigma = jnp.asarray([[0.6]], dtype=dtype)

    def control(x, t, target):
        return 0.25 * x**2 + 0.1 * t * x + 0.05 * target

    rollout = simulate_pinned_brownian_rollout(
        jax.random.PRNGKey(12),
        x0,
        endpoint,
        times,
        sigma,
        control,
    )
    innovations = rollout.innovations[0]
    dt = times[1] - times[0]

    def replay(initial):
        x = initial[None, :]
        for index in range(innovations.shape[0]):
            rho = (times[-1] - times[index + 1]) / (times[-1] - times[index])
            gamma = jnp.sqrt(dt * rho) * sigma
            u = control(x, times[index], endpoint)
            x = rho * x + (1.0 - rho) * endpoint + jnp.sqrt(dt) * (u @ gamma.T)
            x = x + innovations[index][None, :] @ gamma.T
        return x[0]

    expected = jax.jacfwd(replay)(x0[0])
    actual = jnp.eye(1, dtype=dtype)
    for local in rollout.local_jacobians[0]:
        actual = local @ actual
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=_tol(), atol=_tol())


def test_pinned_zero_value_cost_has_zero_label_and_terminal_component():
    dtype = _dtype()
    x0 = jnp.asarray([[-1.0], [0.5]], dtype=dtype)
    endpoint = jnp.asarray([[1.0], [-0.25]], dtype=dtype)
    rollout = simulate_pinned_brownian_rollout(
        jax.random.PRNGKey(2),
        x0,
        endpoint,
        jnp.linspace(0.0, 1.0, 5, dtype=dtype),
        jnp.asarray([[0.8]], dtype=dtype),
    )
    labels = assemble_pinned_brownian_labels(
        rollout,
        anchors=jnp.asarray([0, 1], dtype=jnp.int32),
        running_values=jnp.zeros((2, 5), dtype=dtype),
        terminal_values=jnp.asarray([100.0, -100.0], dtype=dtype),
    )
    np.testing.assert_array_equal(np.asarray(labels.label), np.zeros((2, 1)))
    np.testing.assert_array_equal(np.asarray(labels.terminal_component), np.zeros((2, 1)))
    assert bool(jnp.all(labels.finite))


def test_pinned_linear_running_cost_matches_exact_discrete_costate():
    """A nonzero pinned calibration fixes Gamma, J-arrival, and dt scaling."""
    dtype = _dtype()
    dt = 0.5
    rho = 0.5
    sigma = jnp.asarray([[0.8]], dtype=dtype)
    gamma = jnp.sqrt(dt * rho) * sigma
    innovations = jnp.asarray([[[-1.0]], [[1.0]]], dtype=dtype)
    x0 = jnp.asarray([[0.2], [0.2]], dtype=dtype)
    endpoint = jnp.asarray([[1.0], [1.0]], dtype=dtype)
    interior = rho * x0 + (1.0 - rho) * endpoint + innovations[:, 0] @ gamma.T
    states = jnp.stack([x0, interior, endpoint], axis=1)
    rollout = AdjointRollout(
        states=states,
        innovations=innovations,
        local_jacobians=jnp.broadcast_to(
            jnp.asarray([[[rho]]], dtype=dtype),
            (2, 1, 1, 1),
        ),
        noise_matrices=gamma[None, :, :],
        controls=jnp.zeros((2, 1, 1), dtype=dtype),
        times=jnp.asarray([0.0, 0.5, 1.0], dtype=dtype),
        context=endpoint,
    )
    running = jnp.zeros((2, 3), dtype=dtype).at[:, 1].set(interior[:, 0])
    labels = assemble_pinned_brownian_labels(
        rollout,
        anchors=jnp.zeros((2,), dtype=jnp.int32),
        running_values=running,
    )
    # d/dx0 [dt E[X_1 | x0, endpoint]] = dt * rho.
    np.testing.assert_allclose(
        np.asarray(jnp.mean(labels.label[:, 0])),
        dt * rho,
        rtol=_tol(),
        atol=_tol(),
    )


class _ConstantFactory(NetworkFactory):
    def init(self, key, input_dim, output_dim):
        del key, input_dim
        return {"value": jnp.zeros((output_dim,), dtype=_dtype())}

    def forward(self, params, x, t):
        del t
        return jnp.broadcast_to(params["value"], (x.shape[0], params["value"].shape[0]))


class _StateFactory(NetworkFactory):
    def init(self, key, input_dim, output_dim):
        del key, input_dim, output_dim
        return {}

    def forward(self, params, x, t):
        del params, t
        return x[:, :1]


class _NaNPredictionFactory(NetworkFactory):
    def init(self, key, input_dim, output_dim):
        del key, input_dim
        return {"value": jnp.zeros((output_dim,), dtype=_dtype())}

    def forward(self, params, x, t):
        del t
        value = params["value"] * jnp.asarray(jnp.nan, dtype=x.dtype)
        return jnp.broadcast_to(value, (x.shape[0], value.shape[0]))


class _InfiniteGradientFactory(NetworkFactory):
    def init(self, key, input_dim, output_dim):
        del key, input_dim
        return {"value": jnp.zeros((output_dim,), dtype=_dtype())}

    def forward(self, params, x, t):
        del t
        value = jnp.sqrt(params["value"])
        return jnp.broadcast_to(value, (x.shape[0], value.shape[0]))


def _small_problem(dim=1, steps=4):
    return SBProblem(
        reference=BrownianMotion(sigma=0.7, dim=dim),
        source=GaussianDistribution(dim=dim),
        target=GaussianDistribution(mean=jnp.ones(dim), cov=0.5, dim=dim),
        time_grid=TimeGrid(num_steps=steps),
    )


def _training_batch(label=0.0, batch_size=2):
    labels = jnp.full((batch_size, 1), label, dtype=_dtype())
    return BELCostateBatch(
        anchor_index=jnp.zeros((batch_size,), dtype=jnp.int32),
        anchor_time=jnp.full((batch_size,), 0.25, dtype=_dtype()),
        anchor_state=jnp.zeros((batch_size, 1), dtype=_dtype()),
        context=jnp.ones((batch_size, 1), dtype=_dtype()),
        label=labels,
        terminal_component=jnp.zeros_like(labels),
        running_component=labels,
        direct_component=jnp.zeros_like(labels),
        terminal_weight=jnp.zeros_like(labels),
        tangent_condition_number=jnp.ones((batch_size,), dtype=_dtype()),
        finite=jnp.ones((batch_size,), dtype=bool),
    )


def test_inner_solver_trains_finitely_and_marks_control_as_not_endpoint_projected():
    def running(x, t, endpoint):
        del t
        return jnp.sum((x - 0.25 * endpoint) ** 2, axis=-1)

    solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(running_cost=running, identifier="quadratic_running"),
        MalliavinAdjointConfig(
            hidden_dims=(8,),
            time_embed_dim=4,
            training_steps=1,
            batch_size=8,
            minimum_remaining_steps=1,
        ),
    )
    result = solver.train(jax.random.PRNGKey(0), training_steps=1, batch_size=8)
    assert result.loss_history.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(result.loss_history)))
    assert float(result.final_metrics["finite_fraction"]) == 1.0
    proposal = solver.propose_control(
        jnp.asarray([[0.1]]),
        0.25,
        jnp.asarray([[1.0]]),
    )
    assert proposal.proposed_control.shape == (1, 1)
    assert not bool(proposal.endpoint_preserved)
    assert solver.status == "CONDITIONAL_MAM_FOUNDATION"
    assert not solver.endpoint_constrained_global_bridge


def test_pinned_inner_solver_rejects_terminal_cost_deferred_to_outer_coupling():
    with pytest.raises(ValueError, match="endpoint-coupling outer loop"):
        MalliavinAdjointInnerSolver(
            _small_problem(),
            ValueOnlyCost(terminal_cost=lambda x, endpoint: jnp.sum(x + endpoint, axis=-1)),
            MalliavinAdjointConfig(minimum_remaining_steps=1),
        )


def test_control_proposal_uses_cost_minimization_sign_and_trust_region():
    solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(
            network_factory=_ConstantFactory(),
            trust_region=0.25,
            minimum_remaining_steps=1,
        ),
    )
    params = {"value": jnp.asarray([2.0], dtype=_dtype())}
    proposal = solver.propose_control(
        jnp.asarray([[0.0]], dtype=_dtype()),
        0.2,
        jnp.asarray([[1.0]], dtype=_dtype()),
        params=params,
        current_control=jnp.asarray([[1.0]], dtype=_dtype()),
        rho=0.81,
        next_state=jnp.asarray([[0.0]], dtype=_dtype()),
        next_time=1.0 - 0.81 * (1.0 - 0.2),
    )
    # target = -sqrt(rho) Sigma^T p = -0.9 * 0.7 * 2
    np.testing.assert_allclose(np.asarray(proposal.target_control), [[-1.26]], atol=_tol())
    np.testing.assert_allclose(
        np.asarray(proposal.proposed_control),
        [[0.75 * 1.0 + 0.25 * -1.26]],
        atol=_tol(),
    )
    assert proposal.convention == "cost_minimization"
    assert proposal.coordinates == "brownian_noise_control"
    assert proposal.update_semantics == "one_sample_pinned_discrete"
    assert proposal.outer_projection_required


def test_discrete_control_proposal_uses_next_costate_not_current_costate():
    solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(
            network_factory=_StateFactory(),
            trust_region=1.0,
            minimum_remaining_steps=1,
        ),
    )
    with pytest.raises(ValueError, match="next_state"):
        solver.propose_control(
            jnp.asarray([[0.0]], dtype=_dtype()),
            0.2,
            jnp.asarray([[1.0]], dtype=_dtype()),
            params={},
            rho=0.25,
        )
    proposal = solver.propose_control(
        jnp.asarray([[0.0]], dtype=_dtype()),
        0.2,
        jnp.asarray([[1.0]], dtype=_dtype()),
        params={},
        rho=0.25,
        next_state=jnp.asarray([[2.0]], dtype=_dtype()),
        next_time=1.0 - 0.25 * (1.0 - 0.2),
    )
    # The next-state costate is 2, so -sqrt(.25) * .7 * 2 = -.7.
    np.testing.assert_allclose(np.asarray(proposal.target_control), [[-0.7]], atol=_tol())


def test_discrete_control_proposal_rejects_rho_inconsistent_with_times():
    solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(
            network_factory=_StateFactory(),
            minimum_remaining_steps=1,
        ),
    )
    with pytest.raises(ValueError, match="rho must agree"):
        solver.propose_control(
            jnp.asarray([[0.0]], dtype=_dtype()),
            0.2,
            jnp.asarray([[1.0]], dtype=_dtype()),
            params={},
            rho=0.5,
            next_state=jnp.asarray([[0.0]], dtype=_dtype()),
            next_time=0.3,
        )


@pytest.mark.parametrize(("time", "next_time"), [(-0.1, 0.45), (1.0, 0.9)])
def test_discrete_control_proposal_rejects_time_outside_bridge(time, next_time):
    solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(
            network_factory=_StateFactory(),
            minimum_remaining_steps=1,
        ),
    )
    with pytest.raises(ValueError, match="bridge interval"):
        solver.propose_control(
            jnp.asarray([[0.0]], dtype=_dtype()),
            time,
            jnp.asarray([[1.0]], dtype=_dtype()),
            params={},
            rho=0.5,
            next_state=jnp.asarray([[0.0]], dtype=_dtype()),
            next_time=next_time,
        )


@pytest.mark.parametrize(
    ("rho", "message"),
    [
        (jnp.asarray([[0.5], [0.5]]), "rho must be scalar or have shape"),
        (jnp.asarray([0.5, jnp.nan]), "rho must be finite"),
        (jnp.asarray([0.0, 0.5]), "0 < rho < 1"),
        (jnp.asarray([1.0, 0.5]), "0 < rho < 1"),
    ],
)
def test_discrete_control_proposal_rejects_invalid_rho(rho, message):
    solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(
            network_factory=_StateFactory(),
            minimum_remaining_steps=1,
        ),
    )
    with pytest.raises(ValueError, match=message):
        solver.propose_control(
            jnp.zeros((2, 1), dtype=_dtype()),
            jnp.asarray([0.2, 0.2], dtype=_dtype()),
            jnp.ones((2, 1), dtype=_dtype()),
            params={},
            rho=jnp.asarray(rho, dtype=_dtype()),
            next_state=jnp.zeros((2, 1), dtype=_dtype()),
            next_time=jnp.asarray([0.3, 0.4], dtype=_dtype()),
        )


@pytest.mark.parametrize(
    ("next_time", "message"),
    [
        ([[0.3], [0.4]], "next_time must be scalar or have shape"),
        ([0.3, math.nan], "time and next_time must be finite"),
        ([0.2, 0.4], "strictly greater than time"),
        ([0.1, 0.4], "strictly greater than time"),
        ([1.0, 0.4], "strictly before the terminal time"),
    ],
)
def test_discrete_control_proposal_rejects_invalid_next_time(next_time, message):
    solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(
            network_factory=_StateFactory(),
            minimum_remaining_steps=1,
        ),
    )
    with pytest.raises(ValueError, match=message):
        solver.propose_control(
            jnp.zeros((2, 1), dtype=_dtype()),
            jnp.asarray([0.2, 0.2], dtype=_dtype()),
            jnp.ones((2, 1), dtype=_dtype()),
            params={},
            rho=0.5,
            next_state=jnp.zeros((2, 1), dtype=_dtype()),
            next_time=jnp.asarray(next_time, dtype=_dtype()),
        )


def test_current_control_energy_has_explicit_immediate_derivative():
    gain = 1.5

    def control(x, t, endpoint):
        del t, endpoint
        return gain * x

    solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(minimum_remaining_steps=1),
        control_fn=control,
    )
    x0 = jnp.asarray([[0.4], [-0.2]], dtype=_dtype())
    endpoint = jnp.asarray([[1.0], [0.7]], dtype=_dtype())
    labels = solver.make_label_batch(jax.random.PRNGKey(17), x0, endpoint)
    anchor_states = labels.anchor_state[:, 0]
    expected = solver.problem.time_grid.dt * gain**2 * anchor_states
    np.testing.assert_allclose(
        np.asarray(labels.direct_component[:, 0]),
        np.asarray(expected),
        rtol=_tol(),
        atol=_tol(),
    )


def test_label_summary_keeps_nonfinite_fraction_and_reports_tail_energy():
    labels = jnp.asarray([[0.0], [1.0], [3.0], [jnp.nan]], dtype=_dtype())
    batch = BELCostateBatch(
        anchor_index=jnp.zeros((4,), dtype=jnp.int32),
        anchor_time=jnp.zeros((4,), dtype=_dtype()),
        anchor_state=jnp.zeros((4, 1), dtype=_dtype()),
        context=jnp.zeros((4, 1), dtype=_dtype()),
        label=labels,
        terminal_component=jnp.zeros_like(labels),
        running_component=labels,
        direct_component=jnp.zeros_like(labels),
        terminal_weight=jnp.zeros_like(labels),
        tangent_condition_number=jnp.ones((4,), dtype=_dtype()),
        finite=jnp.asarray([True, True, True, False]),
    )
    report = summarize_costate_labels(batch)
    assert report["count"] == 4
    assert report["finite_count"] == 3
    assert report["finite_fraction"] == 0.75
    assert report["norm_max"] == 3.0
    assert 0.0 < report["top_1_percent_centered_energy_share"] <= 1.0

    solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(
            network_factory=_ConstantFactory(),
            minimum_remaining_steps=1,
        ),
    )
    params = {"value": jnp.zeros((1,), dtype=_dtype())}
    loss, metrics = solver.loss(params, batch)
    assert bool(jnp.isfinite(loss))
    assert float(metrics["finite_fraction"]) == 0.75
    with pytest.raises(FloatingPointError, match="nonfinite"):
        solver.update_from_batch(params, init_adam(params), batch)


def test_training_update_rejects_nonfinite_parameters_and_predictions():
    batch = _training_batch()
    constant_solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(
            network_factory=_ConstantFactory(),
            minimum_remaining_steps=1,
        ),
    )
    nan_params = {"value": jnp.full((1,), jnp.nan, dtype=_dtype())}
    loss, metrics = constant_solver.loss(nan_params, batch)
    assert not bool(jnp.isfinite(loss))
    assert not bool(metrics["prediction_finite"])
    with pytest.raises(FloatingPointError, match="costate parameters"):
        constant_solver.update_from_batch(nan_params, init_adam(nan_params), batch)

    prediction_solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(
            network_factory=_NaNPredictionFactory(),
            minimum_remaining_steps=1,
        ),
    )
    finite_params = {"value": jnp.zeros((1,), dtype=_dtype())}
    with pytest.raises(FloatingPointError, match="model prediction"):
        prediction_solver.update_from_batch(
            finite_params,
            init_adam(finite_params),
            batch,
        )


def test_training_update_rejects_nonfinite_loss_and_gradient():
    zero_batch = _training_batch()
    constant_solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(
            network_factory=_ConstantFactory(),
            minimum_remaining_steps=1,
        ),
    )
    large = jnp.asarray(2.0 * np.sqrt(np.finfo(_dtype()).max), dtype=_dtype())
    large_params = {"value": jnp.reshape(large, (1,))}
    with pytest.raises(FloatingPointError, match="training loss"):
        constant_solver.update_from_batch(
            large_params,
            init_adam(large_params),
            zero_batch,
        )

    gradient_solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(
            network_factory=_InfiniteGradientFactory(),
            minimum_remaining_steps=1,
        ),
    )
    zero_params = {"value": jnp.zeros((1,), dtype=_dtype())}
    with pytest.raises(FloatingPointError, match="gradient"):
        gradient_solver.update_from_batch(
            zero_params,
            init_adam(zero_params),
            _training_batch(label=1.0),
        )


def test_training_update_rejects_nonfinite_updated_parameters(monkeypatch):
    import schrodinger_bridge.solvers.malliavin_adjoint as mam_module

    def corrupt_update(state, grads, params, **kwargs):
        del grads, kwargs
        corrupt = jax.tree_util.tree_map(lambda value: jnp.full_like(value, jnp.nan), params)
        return corrupt, state

    monkeypatch.setattr(mam_module, "adam_update", corrupt_update)
    solver = MalliavinAdjointInnerSolver(
        _small_problem(),
        ValueOnlyCost(),
        MalliavinAdjointConfig(
            network_factory=_ConstantFactory(),
            minimum_remaining_steps=1,
        ),
    )
    params = {"value": jnp.zeros((1,), dtype=_dtype())}
    with pytest.raises(FloatingPointError, match="updated costate parameters"):
        solver.update_from_batch(params, init_adam(params), _training_batch())


@pytest.mark.slow
def test_hard_pinned_running_cost_matches_conditional_gaussian_costate():
    dtype = _dtype()
    num_samples = 50_000
    dt = 0.5
    rho = 0.5
    sigma = 0.7
    gamma = math.sqrt(dt * rho) * sigma
    x0 = 0.1
    endpoint = 0.9
    threshold = 0.65
    innovations = jax.random.normal(
        jax.random.PRNGKey(27),
        (num_samples, 1, 1),
        dtype=dtype,
    )
    initial = jnp.full((num_samples, 1), x0, dtype=dtype)
    target = jnp.full((num_samples, 1), endpoint, dtype=dtype)
    interior = rho * initial + (1.0 - rho) * target + gamma * innovations[:, 0]
    rollout = AdjointRollout(
        states=jnp.stack([initial, interior, target], axis=1),
        innovations=innovations,
        local_jacobians=jnp.full((num_samples, 1, 1, 1), rho, dtype=dtype),
        noise_matrices=jnp.asarray([[[gamma]]], dtype=dtype),
        controls=jnp.zeros((num_samples, 1, 1), dtype=dtype),
        times=jnp.asarray([0.0, 0.5, 1.0], dtype=dtype),
        context=target,
    )
    running = jnp.zeros((num_samples, 3), dtype=dtype)
    running = running.at[:, 1].set((interior[:, 0] >= threshold).astype(dtype))
    labels = assemble_pinned_brownian_labels(
        rollout,
        anchors=jnp.zeros((num_samples,), dtype=jnp.int32),
        running_values=running,
    )
    samples = np.asarray(labels.label[:, 0])
    mean = float(samples.mean())
    se = float(samples.std(ddof=1) / math.sqrt(num_samples))
    conditional_mean = rho * x0 + (1.0 - rho) * endpoint
    z = (threshold - conditional_mean) / gamma
    truth = dt * rho * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) / gamma
    assert abs(mean - truth) <= 5.0 * se
    assert abs(mean - truth) / truth < 0.06


@pytest.mark.slow
def test_hard_brownian_terminal_bel_mean_matches_analytic_costate():
    dtype = _dtype()
    num_samples = 50_000
    sigma = 0.7
    threshold = 0.3
    dt = 1.0
    increments = jax.random.normal(
        jax.random.PRNGKey(91), (num_samples, 1, 1), dtype=dtype
    ) * math.sqrt(dt)
    rollout = _linear_rollout(
        jnp.asarray([[1.0]], dtype=dtype),
        jnp.asarray([[sigma]], dtype=dtype),
        increments,
        dt,
    )
    rollout = rollout._replace(times=jnp.asarray([0.0, 1.0], dtype=dtype))
    terminal = (rollout.states[:, -1, 0] >= threshold).astype(dtype)
    labels = assemble_bel_costate_labels(
        rollout,
        anchors=jnp.zeros((num_samples,), dtype=jnp.int32),
        running_values=jnp.zeros((num_samples, 2), dtype=dtype),
        terminal_values=terminal,
    )
    samples = np.asarray(labels.label[:, 0])
    mean = float(samples.mean())
    se = float(samples.std(ddof=1) / math.sqrt(num_samples))
    z = threshold / sigma
    truth = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) / sigma
    assert abs(mean - truth) <= 5.0 * se
    assert abs(mean - truth) / truth < 0.05
