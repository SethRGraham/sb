"""Gate-0 falsifiers for arrival-aware MAM and reverse-VJP labels.

These tests deliberately use fixed seeds and compare sample formulas.  They
validate the declared discrete algebra, not learned policy improvement or a
global endpoint-constrained bridge.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from schrodinger_bridge.solvers.malliavin_adjoint import (
    assemble_antithetic_direct_action_score,
    assemble_pinned_actor_targets,
    assemble_pinned_brownian_labels,
    assemble_pinned_brownian_labels_matrix_free,
    simulate_pinned_brownian_rollout,
    simulate_pinned_brownian_rollout_matrix_free,
)


def _dtype():
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


def _tol() -> float:
    return 2e-11 if jax.config.x64_enabled else 3e-5


def _zero_running(states, times, context):
    del times, context
    return jnp.zeros((states.shape[0],), dtype=states.dtype)


def _positive_halfspace(states, times, context):
    del times, context
    return (states[:, 0] > 0.0).astype(states.dtype)


def test_final_stochastic_hard_arrival_makes_old_zero_target_nonzero():
    """At n=N-2, p[N-1]=0 but the arrival score generally is not zero."""
    dtype = _dtype()
    result = assemble_pinned_actor_targets(
        jax.random.PRNGKey(701),
        state=jnp.zeros((1, 1), dtype=dtype),
        endpoint=jnp.zeros((1, 1), dtype=dtype),
        time=jnp.asarray(0.0, dtype=dtype),
        next_time=jnp.asarray(0.5, dtype=dtype),
        terminal_time=jnp.asarray(1.0, dtype=dtype),
        diffusion=jnp.asarray([[1.0]], dtype=dtype),
        current_control=jnp.zeros((1, 1), dtype=dtype),
        running_cost=_positive_halfspace,
        next_costate=None,
        num_antithetic=1,
    )

    dt = jnp.asarray(0.5, dtype=dtype)
    rho = jnp.asarray(0.5, dtype=dtype)
    gamma = jnp.sqrt(dt * rho)
    xi = result.innovation[0, 0, 0]
    plus = result.mean_state[0, 0] + gamma * xi
    minus = result.mean_state[0, 0] - gamma * xi
    ell_plus = (plus > 0.0).astype(dtype)
    ell_minus = (minus > 0.0).astype(dtype)
    expected_arrival = -jnp.sqrt(dt) * 0.5 * (ell_plus - ell_minus) * xi

    # The obsolete continuation-only target is exactly zero in this case.
    np.testing.assert_array_equal(
        np.asarray(result.continuation_component),
        np.zeros((1, 1)),
    )
    np.testing.assert_allclose(
        np.asarray(result.arrival_component[0, 0]),
        np.asarray(expected_arrival),
        rtol=_tol(),
        atol=_tol(),
    )
    assert float(result.target[0, 0]) < 0.0
    assert int(result.physical_oracle_queries) == 2
    assert bool(result.finite[0])


def test_two_step_target_is_arrival_plus_future_costate_continuation():
    """A hard cost at t1 and a smooth cost at t2 contribute separately."""
    dtype = _dtype()
    dt = 1.0 / 3.0
    rho_first = 2.0 / 3.0
    rho_second = 0.5
    future_slope = 1.2
    # If ell_2(x)=future_slope*x under zero control, then
    # p_1=d/dx_1 [dt E ell_2(X_2)|X_1]=dt*rho_second*future_slope.
    next_costate_value = dt * rho_second * future_slope

    def next_costate(states, times, context):
        del times, context
        return jnp.full_like(states, next_costate_value)

    sigma = jnp.asarray([[1.3]], dtype=dtype)
    result = assemble_pinned_actor_targets(
        jax.random.PRNGKey(702),
        state=jnp.zeros((1, 1), dtype=dtype),
        endpoint=jnp.zeros((1, 1), dtype=dtype),
        time=jnp.asarray(0.0, dtype=dtype),
        next_time=jnp.asarray(dt, dtype=dtype),
        terminal_time=jnp.asarray(1.0, dtype=dtype),
        diffusion=sigma,
        current_control=jnp.zeros((1, 1), dtype=dtype),
        running_cost=_positive_halfspace,
        next_costate=next_costate,
        num_antithetic=1,
    )

    expected_continuation = -math.sqrt(rho_first) * float(sigma[0, 0]) * next_costate_value
    np.testing.assert_allclose(
        np.asarray(result.continuation_component),
        [[expected_continuation]],
        rtol=_tol(),
        atol=_tol(),
    )
    assert abs(float(result.arrival_component[0, 0])) > _tol()
    np.testing.assert_allclose(
        np.asarray(result.target),
        np.asarray(result.continuation_component + result.arrival_component),
        rtol=_tol(),
        atol=_tol(),
    )


def test_actor_continuation_uses_rho_and_nonsymmetric_sigma_transpose_math():
    """Column Sigma.T p is row-vector p @ Sigma, not p @ Sigma.T."""
    dtype = _dtype()
    sigma = jnp.asarray([[1.0, 0.7], [-0.3, 1.2]], dtype=dtype)
    p_next = jnp.asarray([0.8, -1.1], dtype=dtype)
    time = 0.2
    next_time = 0.5
    terminal_time = 1.0
    rho = (terminal_time - next_time) / (terminal_time - time)

    def constant_costate(states, times, context):
        del times, context
        return jnp.broadcast_to(p_next, states.shape)

    result = assemble_pinned_actor_targets(
        jax.random.PRNGKey(703),
        state=jnp.asarray([[0.1, -0.2]], dtype=dtype),
        endpoint=jnp.asarray([[0.4, 0.3]], dtype=dtype),
        time=jnp.asarray(time, dtype=dtype),
        next_time=jnp.asarray(next_time, dtype=dtype),
        terminal_time=jnp.asarray(terminal_time, dtype=dtype),
        diffusion=sigma,
        current_control=jnp.zeros((1, 2), dtype=dtype),
        running_cost=_zero_running,
        next_costate=constant_costate,
        num_antithetic=2,
    )

    expected = -math.sqrt(rho) * (p_next @ sigma)
    wrong_orientation = -math.sqrt(rho) * (p_next @ sigma.T)
    np.testing.assert_allclose(
        np.asarray(result.continuation_component[0]),
        np.asarray(expected),
        rtol=_tol(),
        atol=_tol(),
    )
    assert not np.allclose(
        np.asarray(result.continuation_component[0]),
        np.asarray(wrong_orientation),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_array_equal(
        np.asarray(result.arrival_component),
        np.zeros((1, 2)),
    )


def test_complete_reverse_vjp_label_matches_dense_reference_samplewise():
    """Finite-sum reordering must not change any fixed-innovation label."""
    dtype = _dtype()
    key = jax.random.PRNGKey(704)
    times = jnp.linspace(0.0, 1.0, 5, dtype=dtype)
    x0 = jnp.asarray(
        [[-0.4, 0.2], [0.1, -0.3], [0.5, 0.4]],
        dtype=dtype,
    )
    endpoint = jnp.asarray(
        [[0.7, -0.1], [-0.2, 0.8], [0.3, -0.6]],
        dtype=dtype,
    )
    sigma = jnp.asarray([[1.0, 0.35], [-0.2, 0.85]], dtype=dtype)
    dense_rollout = simulate_pinned_brownian_rollout(
        key,
        x0,
        endpoint,
        times,
        sigma,
    )
    matrix_free_rollout = simulate_pinned_brownian_rollout_matrix_free(
        key,
        x0,
        endpoint,
        times,
        sigma,
    )
    np.testing.assert_allclose(
        np.asarray(matrix_free_rollout.states),
        np.asarray(dense_rollout.states),
        rtol=_tol(),
        atol=_tol(),
    )
    np.testing.assert_allclose(
        np.asarray(matrix_free_rollout.innovations),
        np.asarray(dense_rollout.innovations),
        rtol=0.0,
        atol=0.0,
    )

    running = 0.4 * (dense_rollout.states[..., 0] > 0.0).astype(dtype) + 0.3 * (
        dense_rollout.states[..., 1] < 0.1
    ).astype(dtype)
    anchors = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    immediate = jnp.asarray(
        [[0.07, -0.03], [-0.02, 0.05], [0.01, 0.04]],
        dtype=dtype,
    )
    dense = assemble_pinned_brownian_labels(
        dense_rollout,
        anchors,
        running_values=running,
        immediate_gradients=immediate,
    )
    reverse = assemble_pinned_brownian_labels_matrix_free(
        matrix_free_rollout,
        anchors,
        hard_running_values=running,
        control_fn=None,
        center_running=False,
        immediate_gradients=immediate,
    )

    np.testing.assert_allclose(
        np.asarray(reverse.running_component),
        np.asarray(dense.running_component),
        rtol=2 * _tol(),
        atol=2 * _tol(),
    )
    np.testing.assert_allclose(
        np.asarray(reverse.label),
        np.asarray(dense.label),
        rtol=2 * _tol(),
        atol=2 * _tol(),
    )
    np.testing.assert_array_equal(np.asarray(reverse.finite), np.ones((3,), dtype=bool))


def test_matrix_free_keeps_policy_jacobian_when_energy_is_excluded():
    """Objective bookkeeping must not alter the controlled transition VJP."""
    dtype = _dtype()
    key = jax.random.PRNGKey(705)
    times = jnp.linspace(0.0, 1.0, 5, dtype=dtype)
    x0 = jnp.asarray([[-0.3, 0.2], [0.4, -0.1]], dtype=dtype)
    endpoint = jnp.asarray([[0.8, -0.4], [-0.5, 0.7]], dtype=dtype)
    sigma = jnp.asarray([[0.9, 0.45], [-0.25, 1.1]], dtype=dtype)
    feedback = jnp.asarray([[0.4, -0.7], [0.6, 0.3]], dtype=dtype)

    def control(state, time, context):
        del time
        return 0.35 * jnp.tanh(state @ feedback.T + 0.1 * context)

    dense_rollout = simulate_pinned_brownian_rollout(key, x0, endpoint, times, sigma, control)
    matrix_free_rollout = simulate_pinned_brownian_rollout_matrix_free(
        key, x0, endpoint, times, sigma, control
    )
    running = 0.7 * (dense_rollout.states[..., 0] > -0.1).astype(dtype) + 0.2 * (
        dense_rollout.states[..., 1] < 0.25
    ).astype(dtype)
    anchors = jnp.asarray([0, 1], dtype=jnp.int32)
    dense = assemble_pinned_brownian_labels(
        dense_rollout,
        anchors,
        running_values=running,
        immediate_gradients=jnp.zeros_like(x0),
    )
    reverse = assemble_pinned_brownian_labels_matrix_free(
        matrix_free_rollout,
        anchors,
        hard_running_values=running,
        control_fn=control,
        include_control_energy=False,
        center_running=False,
    )
    np.testing.assert_allclose(
        np.asarray(reverse.label),
        np.asarray(dense.label),
        rtol=4 * _tol(),
        atol=4 * _tol(),
    )


def test_jitted_actor_target_marks_singular_diffusion_invalid():
    dtype = _dtype()

    @jax.jit
    def evaluate(key):
        return assemble_pinned_actor_targets(
            key,
            state=jnp.zeros((2, 1), dtype=dtype),
            endpoint=jnp.zeros((2, 1), dtype=dtype),
            time=jnp.asarray(0.0, dtype=dtype),
            next_time=jnp.asarray(0.5, dtype=dtype),
            terminal_time=jnp.asarray(1.0, dtype=dtype),
            diffusion=jnp.zeros((1, 1), dtype=dtype),
            current_control=jnp.zeros((2, 1), dtype=dtype),
            running_cost=_zero_running,
            next_costate=None,
        )

    result = evaluate(jax.random.PRNGKey(706))
    np.testing.assert_array_equal(np.asarray(result.finite), [False, False])


def test_direct_full_return_action_score_is_unbiased_linear_baseline():
    dtype = _dtype()
    batch_size = 65_536
    dt = jnp.asarray(0.2, dtype=dtype)
    rho = jnp.asarray(0.6, dtype=dtype)
    sigma = jnp.asarray(1.3, dtype=dtype)
    slope = jnp.asarray(0.7, dtype=dtype)
    xi = jax.random.normal(jax.random.PRNGKey(708), (batch_size, 1, 1), dtype=dtype)
    gamma = jnp.sqrt(dt * rho) * sigma
    plus_return = slope * gamma * xi[..., 0]
    minus_return = -slope * gamma * xi[..., 0]
    result = assemble_antithetic_direct_action_score(plus_return, minus_return, xi, dt)
    truth = -jnp.sqrt(rho) * sigma * slope
    assert abs(float(jnp.mean(result.target)) - float(truth)) < 0.01
    assert int(result.physical_return_queries) == 2 * batch_size
    assert bool(jnp.all(result.finite))


def test_matrix_free_nonfinite_rollout_fails_validity_mask():
    dtype = _dtype()
    times = jnp.linspace(0.0, 1.0, 4, dtype=dtype)
    x0 = jnp.zeros((2, 1), dtype=dtype)
    endpoint = jnp.ones((2, 1), dtype=dtype)

    def invalid_control(state, time, context):
        del time, context
        return jnp.full_like(state, jnp.inf)

    with pytest.raises(ValueError, match="nonfinite pinned rollout result"):
        simulate_pinned_brownian_rollout_matrix_free(
            jax.random.PRNGKey(707),
            x0,
            endpoint,
            times,
            jnp.asarray([[1.0]], dtype=dtype),
            invalid_control,
        )

    @jax.jit
    def compiled_rollout(key):
        return simulate_pinned_brownian_rollout_matrix_free(
            key,
            x0,
            endpoint,
            times,
            jnp.asarray([[1.0]], dtype=dtype),
            invalid_control,
        )

    rollout = compiled_rollout(jax.random.PRNGKey(707))
    assert bool(jnp.all(jnp.isnan(rollout.states)))
    labels = assemble_pinned_brownian_labels_matrix_free(
        rollout,
        anchors=jnp.asarray([0, 0], dtype=jnp.int32),
        hard_running_values=jnp.zeros((2, 4), dtype=dtype),
        control_fn=invalid_control,
        include_control_energy=False,
        center_running=False,
    )
    np.testing.assert_array_equal(np.asarray(labels.finite), [False, False])


def test_matrix_free_rejects_noise_factors_from_a_different_pinned_chain():
    dtype = _dtype()
    times = jnp.linspace(0.0, 1.0, 5, dtype=dtype)
    x0 = jnp.zeros((2, 1), dtype=dtype)
    endpoint = jnp.ones((2, 1), dtype=dtype)
    rollout = simulate_pinned_brownian_rollout_matrix_free(
        jax.random.PRNGKey(710),
        x0,
        endpoint,
        times,
        jnp.asarray([[0.8]], dtype=dtype),
    )
    inconsistent = rollout._replace(noise_matrices=rollout.noise_matrices.at[1].multiply(1.25))
    labels = assemble_pinned_brownian_labels_matrix_free(
        inconsistent,
        anchors=jnp.asarray([0, 1], dtype=jnp.int32),
        hard_running_values=jnp.zeros((2, 5), dtype=dtype),
        center_running=False,
    )
    np.testing.assert_array_equal(np.asarray(labels.finite), [False, False])


def test_matrix_free_rejects_a_control_callback_whose_recorded_values_differ():
    dtype = _dtype()
    times = jnp.linspace(0.0, 1.0, 4, dtype=dtype)
    rollout = simulate_pinned_brownian_rollout_matrix_free(
        jax.random.PRNGKey(715),
        jnp.asarray([[0.3], [-0.2]], dtype=dtype),
        jnp.asarray([[0.8], [0.6]], dtype=dtype),
        times,
        jnp.asarray([[0.9]], dtype=dtype),
    )

    def unrelated_control(state, time, endpoint):
        del time, endpoint
        return 5.0 * state

    def assemble(value):
        return assemble_pinned_brownian_labels_matrix_free(
            value,
            anchors=jnp.asarray([0, 0], dtype=jnp.int32),
            hard_running_values=value.states[..., 0] ** 2,
            control_fn=unrelated_control,
            include_control_energy=False,
            center_running=False,
        )

    eager = assemble(rollout)
    compiled = jax.jit(assemble)(rollout)
    np.testing.assert_array_equal(np.asarray(eager.finite), [False, False])
    np.testing.assert_array_equal(np.asarray(compiled.finite), [False, False])


def test_matrix_free_noise_chain_check_is_relative_at_tiny_physical_scale():
    dtype = _dtype()
    rollout = simulate_pinned_brownian_rollout_matrix_free(
        jax.random.PRNGKey(716),
        jnp.zeros((1, 1), dtype=dtype),
        jnp.ones((1, 1), dtype=dtype),
        jnp.linspace(0.0, 1.0, 5, dtype=dtype),
        jnp.asarray([[1.0e-6]], dtype=dtype),
    )
    inconsistent = rollout._replace(noise_matrices=rollout.noise_matrices.at[1].multiply(1.25))
    labels = assemble_pinned_brownian_labels_matrix_free(
        inconsistent,
        anchors=jnp.asarray([0], dtype=jnp.int32),
        hard_running_values=jnp.zeros((1, 5), dtype=dtype),
        include_control_energy=False,
        center_running=False,
    )
    np.testing.assert_array_equal(np.asarray(labels.finite), [False])


def test_matrix_free_assembler_applies_the_declared_diffusion_rcond():
    dtype = _dtype()
    rollout = simulate_pinned_brownian_rollout_matrix_free(
        jax.random.PRNGKey(717),
        jnp.zeros((1, 2), dtype=dtype),
        jnp.ones((1, 2), dtype=dtype),
        jnp.linspace(0.0, 1.0, 4, dtype=dtype),
        jnp.diag(jnp.asarray([1.0, 1.0e-6], dtype=dtype)),
        diffusion_rcond=1.0e-8,
    )
    labels = assemble_pinned_brownian_labels_matrix_free(
        rollout,
        anchors=jnp.asarray([0], dtype=jnp.int32),
        hard_running_values=jnp.zeros((1, 4), dtype=dtype),
        include_control_energy=False,
        center_running=False,
        diffusion_rcond=1.0e-4,
    )
    np.testing.assert_array_equal(np.asarray(labels.finite), [False])


def test_matrix_free_assembler_rejects_states_not_generated_by_recorded_transitions():
    dtype = _dtype()
    rollout = simulate_pinned_brownian_rollout_matrix_free(
        jax.random.PRNGKey(718),
        jnp.zeros((1, 1), dtype=dtype),
        jnp.ones((1, 1), dtype=dtype),
        jnp.linspace(0.0, 1.0, 4, dtype=dtype),
        0.8,
    )
    corrupted = rollout._replace(states=rollout.states.at[0, 1, 0].add(0.1))
    labels = assemble_pinned_brownian_labels_matrix_free(
        corrupted,
        anchors=jnp.asarray([0], dtype=jnp.int32),
        hard_running_values=jnp.zeros((1, 4), dtype=dtype),
        include_control_energy=False,
        center_running=False,
    )
    np.testing.assert_array_equal(np.asarray(labels.finite), [False])


def test_matrix_free_rejects_nonuniform_exported_time_grid():
    dtype = _dtype()
    rollout = simulate_pinned_brownian_rollout_matrix_free(
        jax.random.PRNGKey(712),
        jnp.zeros((1, 1), dtype=dtype),
        jnp.ones((1, 1), dtype=dtype),
        jnp.linspace(0.0, 1.0, 4, dtype=dtype),
        jnp.asarray([[0.8]], dtype=dtype),
    )
    nonuniform = rollout._replace(times=jnp.asarray([0.0, 0.2, 0.7, 1.0], dtype=dtype))
    with pytest.raises(ValueError, match="uniform"):
        assemble_pinned_brownian_labels_matrix_free(
            nonuniform,
            anchors=jnp.asarray([0], dtype=jnp.int32),
            hard_running_values=jnp.zeros((1, 4), dtype=dtype),
            center_running=False,
        )


def test_arrival_target_and_reverse_label_match_eager_jit_and_replay():
    dtype = _dtype()
    state = jnp.asarray([[0.15]], dtype=dtype)
    endpoint = jnp.asarray([[0.4]], dtype=dtype)

    def actor_target(key):
        return assemble_pinned_actor_targets(
            key,
            state=state,
            endpoint=endpoint,
            time=jnp.asarray(0.0, dtype=dtype),
            next_time=jnp.asarray(0.4, dtype=dtype),
            terminal_time=jnp.asarray(1.0, dtype=dtype),
            diffusion=jnp.asarray([[0.8]], dtype=dtype),
            current_control=jnp.zeros_like(state),
            running_cost=_positive_halfspace,
            next_costate=None,
            num_antithetic=2,
        )

    key = jax.random.PRNGKey(709)
    eager_target = actor_target(key)
    compiled_target = jax.jit(actor_target)(key)
    replay_target = actor_target(key)
    np.testing.assert_allclose(
        np.asarray(compiled_target.target),
        np.asarray(eager_target.target),
        rtol=_tol(),
        atol=_tol(),
    )
    np.testing.assert_array_equal(
        np.asarray(replay_target.innovation),
        np.asarray(eager_target.innovation),
    )
    assert not np.array_equal(
        np.asarray(actor_target(jax.random.PRNGKey(710)).innovation),
        np.asarray(eager_target.innovation),
    )

    times = jnp.linspace(0.0, 1.0, 4, dtype=dtype)
    rollout = simulate_pinned_brownian_rollout_matrix_free(
        jax.random.PRNGKey(711),
        jnp.asarray([[-0.2]], dtype=dtype),
        jnp.asarray([[0.7]], dtype=dtype),
        times,
        jnp.asarray([[0.9]], dtype=dtype),
    )
    running = (rollout.states[..., 0] > 0.0).astype(dtype)

    def reverse_label(value):
        return assemble_pinned_brownian_labels_matrix_free(
            value,
            anchors=jnp.asarray([0], dtype=jnp.int32),
            hard_running_values=running,
            include_control_energy=False,
            center_running=True,
        )

    eager_label = reverse_label(rollout)
    compiled_label = jax.jit(reverse_label)(rollout)
    np.testing.assert_allclose(
        np.asarray(compiled_label.label),
        np.asarray(eager_label.label),
        rtol=3 * _tol(),
        atol=3 * _tol(),
    )
    np.testing.assert_array_equal(np.asarray(compiled_label.finite), [True])


@pytest.mark.parametrize("invalid_next_time", [0.0, 1.0])
def test_actor_rho_boundary_fails_eager_and_marks_jitted_row_invalid(invalid_next_time):
    """rho=1 (zero step) and rho=0 (terminal arrival) are outside V1."""
    dtype = _dtype()

    def evaluate(next_time):
        return assemble_pinned_actor_targets(
            jax.random.PRNGKey(713),
            state=jnp.zeros((1, 1), dtype=dtype),
            endpoint=jnp.ones((1, 1), dtype=dtype),
            time=jnp.asarray(0.0, dtype=dtype),
            next_time=next_time,
            terminal_time=jnp.asarray(1.0, dtype=dtype),
            diffusion=jnp.ones((1, 1), dtype=dtype),
            current_control=jnp.zeros((1, 1), dtype=dtype),
            running_cost=_zero_running,
        )

    value = jnp.asarray(invalid_next_time, dtype=dtype)
    with pytest.raises(ValueError):
        evaluate(value)
    compiled = jax.jit(evaluate)(value)
    np.testing.assert_array_equal(np.asarray(compiled.finite), [False])


@pytest.mark.parametrize("field", ["state", "context", "control"])
def test_actor_nonfinite_inputs_fail_eager_and_jitted_validity(field):
    dtype = _dtype()

    def evaluate(state, endpoint, control):
        return assemble_pinned_actor_targets(
            jax.random.PRNGKey(714),
            state=state,
            endpoint=endpoint,
            time=jnp.asarray(0.0, dtype=dtype),
            next_time=jnp.asarray(0.5, dtype=dtype),
            terminal_time=jnp.asarray(1.0, dtype=dtype),
            diffusion=jnp.ones((1, 1), dtype=dtype),
            current_control=control,
            running_cost=_zero_running,
        )

    state = jnp.zeros((1, 1), dtype=dtype)
    endpoint = jnp.ones((1, 1), dtype=dtype)
    control = jnp.zeros((1, 1), dtype=dtype)
    if field == "state":
        state = state.at[0, 0].set(jnp.nan)
    elif field == "context":
        endpoint = endpoint.at[0, 0].set(jnp.nan)
    else:
        control = control.at[0, 0].set(jnp.nan)
    with pytest.raises(ValueError, match="actor input"):
        evaluate(state, endpoint, control)
    compiled = jax.jit(evaluate)(state, endpoint, control)
    np.testing.assert_array_equal(np.asarray(compiled.finite), [False])


@pytest.mark.parametrize("invalid_callback", ["cost", "costate"])
def test_actor_nonfinite_oracle_outputs_fail_closed(invalid_callback):
    dtype = _dtype()

    def nan_cost(states, times, context):
        del times, context
        return jnp.full((states.shape[0],), jnp.nan, dtype=states.dtype)

    def nan_costate(states, times, context):
        del times, context
        return jnp.full_like(states, jnp.nan)

    running = nan_cost if invalid_callback == "cost" else _zero_running
    costate = nan_costate if invalid_callback == "costate" else None

    def evaluate(key):
        return assemble_pinned_actor_targets(
            key,
            state=jnp.zeros((1, 1), dtype=dtype),
            endpoint=jnp.ones((1, 1), dtype=dtype),
            time=jnp.asarray(0.0, dtype=dtype),
            next_time=jnp.asarray(0.5, dtype=dtype),
            terminal_time=jnp.asarray(1.0, dtype=dtype),
            diffusion=jnp.ones((1, 1), dtype=dtype),
            current_control=jnp.zeros((1, 1), dtype=dtype),
            running_cost=running,
            next_costate=costate,
        )

    with pytest.raises(ValueError, match="nonfinite pinned actor target"):
        evaluate(jax.random.PRNGKey(715))
    compiled = jax.jit(evaluate)(jax.random.PRNGKey(715))
    np.testing.assert_array_equal(np.asarray(compiled.finite), [False])


@pytest.mark.parametrize("num_antithetic", [True, 1.5, 0])
def test_actor_requires_strict_positive_integer_antithetic_count(num_antithetic):
    dtype = _dtype()
    with pytest.raises(ValueError, match="positive integer"):
        assemble_pinned_actor_targets(
            jax.random.PRNGKey(716),
            state=jnp.zeros((1, 1), dtype=dtype),
            endpoint=jnp.ones((1, 1), dtype=dtype),
            time=0.0,
            next_time=0.5,
            terminal_time=1.0,
            diffusion=1.0,
            current_control=jnp.zeros((1, 1), dtype=dtype),
            running_cost=_zero_running,
            num_antithetic=num_antithetic,
        )


@pytest.mark.parametrize("rcond", [True, 0.0, -1.0, math.nan, math.inf])
def test_actor_requires_strict_finite_positive_diffusion_rcond(rcond):
    dtype = _dtype()
    with pytest.raises(ValueError, match="diffusion_rcond"):
        assemble_pinned_actor_targets(
            jax.random.PRNGKey(717),
            state=jnp.zeros((1, 1), dtype=dtype),
            endpoint=jnp.ones((1, 1), dtype=dtype),
            time=0.0,
            next_time=0.5,
            terminal_time=1.0,
            diffusion=1.0,
            current_control=jnp.zeros((1, 1), dtype=dtype),
            running_cost=_zero_running,
            diffusion_rcond=rcond,
        )
