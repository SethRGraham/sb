"""Focused tests for exact MAM work accounting and synchronized timing."""

import json
from itertools import permutations

import jax.numpy as jnp
import numpy as np
import pytest

import schrodinger_bridge.solvers.mam_accounting as accounting
from schrodinger_bridge.solvers.mam_accounting import (
    MAMWorkCounters,
    SynchronizedJaxTiming,
    completed_conditional_solve_work,
    conditional_costate_refresh_work,
    conditional_policy_iteration_work,
    global_half_iteration_work,
    merge_work_counters,
    time_jax_callable,
)


class _SequenceClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


def test_work_counter_identity_addition_and_exact_units():
    work = MAMWorkCounters(
        running_cost_oracle_evaluations=11,
        simulated_transitions=7,
        tangent_vjps=5,
        tangent_jvps=3,
        optimizer_examples=64,
        optimizer_updates=2,
        compile_time_ns=101,
        steady_state_time_ns=303,
        peak_device_memory_bytes=4096,
    )
    assert MAMWorkCounters.zero().merge(work) == work
    assert work + MAMWorkCounters.zero() == work

    updated = work.add(
        running_cost_oracle_evaluations=2,
        simulated_transitions=4,
        tangent_vjps=1,
        optimizer_examples=32,
        optimizer_updates=1,
        compile_time_ns=7,
        steady_state_time_ns=13,
        peak_device_memory_bytes=8192,
    )
    assert updated.running_cost_oracle_evaluations == 13
    assert updated.simulated_transitions == 11
    assert updated.tangent_vjps == 6
    assert updated.tangent_jvps == 3
    assert updated.optimizer_examples == 96
    assert updated.optimizer_updates == 3
    assert updated.compile_time_ns == 108
    assert updated.steady_state_time_ns == 316
    assert updated.peak_device_memory_bytes == 8192
    assert updated.compile_time_seconds == pytest.approx(108e-9)
    assert updated.steady_state_time_seconds == pytest.approx(316e-9)


def test_merge_is_associative_commutative_and_order_deterministic():
    records = (
        MAMWorkCounters(
            running_cost_oracle_evaluations=10,
            simulated_transitions=20,
            optimizer_examples=8,
            optimizer_updates=1,
            compile_time_ns=9,
            peak_device_memory_bytes=100,
        ),
        MAMWorkCounters(
            tangent_vjps=7,
            tangent_jvps=2,
            optimizer_examples=16,
            optimizer_updates=2,
            steady_state_time_ns=13,
            peak_device_memory_bytes=250,
        ),
        MAMWorkCounters(
            running_cost_oracle_evaluations=3,
            simulated_transitions=4,
            optimizer_examples=4,
            optimizer_updates=1,
            compile_time_ns=5,
            steady_state_time_ns=8,
            peak_device_memory_bytes=180,
        ),
    )
    reference = merge_work_counters(records)
    for ordering in permutations(records):
        assert merge_work_counters(ordering) == reference
    assert records[0].merge(records[1]).merge(records[2]) == records[0].merge(
        records[1].merge(records[2])
    )
    assert reference.running_cost_oracle_evaluations == 13
    assert reference.simulated_transitions == 24
    assert reference.tangent_vjps == 7
    assert reference.tangent_jvps == 2
    assert reference.optimizer_examples == 28
    assert reference.optimizer_updates == 4
    assert reference.compile_time_ns == 14
    assert reference.steady_state_time_ns == 21
    assert reference.peak_device_memory_bytes == 250


def test_unknown_peak_memory_is_explicit_and_propagates_except_through_identity():
    unknown = MAMWorkCounters(
        simulated_transitions=1,
        peak_device_memory_bytes=None,
    )
    known = MAMWorkCounters(
        simulated_transitions=2,
        peak_device_memory_bytes=2048,
    )
    assert MAMWorkCounters.zero().merge(unknown).peak_device_memory_bytes is None
    assert unknown.merge(MAMWorkCounters.zero()).peak_device_memory_bytes is None
    assert known.merge(unknown).peak_device_memory_bytes is None
    assert unknown.merge(known).peak_device_memory_bytes is None
    assert known.add(peak_device_memory_bytes=None).peak_device_memory_bytes is None


def test_work_counter_json_state_roundtrip_is_exact_and_strict():
    original = MAMWorkCounters(
        running_cost_oracle_evaluations=2**60 + 7,
        simulated_transitions=31,
        tangent_vjps=17,
        tangent_jvps=19,
        optimizer_examples=1024,
        optimizer_updates=8,
        compile_time_ns=987654321,
        steady_state_time_ns=123456789,
        peak_device_memory_bytes=None,
    )
    encoded = json.dumps(original.to_state(), sort_keys=True)
    restored = MAMWorkCounters.from_state(json.loads(encoded))
    assert restored == original

    missing = original.to_state()
    missing.pop("tangent_jvps")
    with pytest.raises(ValueError, match="schema mismatch"):
        MAMWorkCounters.from_state(missing)
    extra = original.to_state()
    extra["untracked_work"] = 1
    with pytest.raises(ValueError, match="schema mismatch"):
        MAMWorkCounters.from_state(extra)
    wrong_version = original.to_state()
    wrong_version["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported"):
        MAMWorkCounters.from_state(wrong_version)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"simulated_transitions": -1}, ValueError),
        ({"tangent_vjps": 1.5}, TypeError),
        ({"optimizer_examples": 1, "optimizer_updates": 2}, ValueError),
        ({"compile_time_ns": True}, TypeError),
        ({"peak_device_memory_bytes": -1}, ValueError),
    ],
)
def test_work_counters_fail_closed_on_negative_lossy_or_inconsistent_counts(kwargs, error):
    with pytest.raises(error):
        MAMWorkCounters(**kwargs)


def test_synchronized_timing_separates_compile_and_each_steady_run(monkeypatch):
    synchronization_events = []
    original_synchronize = accounting.synchronize_jax_result

    def recording_synchronize(value):
        synchronized = original_synchronize(value)
        synchronization_events.append(synchronized)
        return synchronized

    monkeypatch.setattr(accounting, "synchronize_jax_result", recording_synchronize)
    clock = _SequenceClock([100, 107, 200, 203, 300, 305, 400, 411])
    result, timing = time_jax_callable(
        lambda x, *, offset: {"value": x * x + offset},
        jnp.arange(4, dtype=jnp.float32),
        steady_state_runs=3,
        call_kwargs={"offset": jnp.asarray(1.0, dtype=jnp.float32)},
        peak_memory_probe=lambda: 12345,
        clock_ns=clock,
    )

    np.testing.assert_array_equal(result["value"], jnp.asarray([1.0, 2.0, 5.0, 10.0]))
    assert timing == SynchronizedJaxTiming(
        compile_time_ns=7,
        steady_state_run_times_ns=(3, 5, 11),
        peak_device_memory_bytes=12345,
    )
    assert timing.steady_state_runs == 3
    assert timing.steady_state_time_ns == 19
    assert timing.to_work_counters() == MAMWorkCounters(
        compile_time_ns=7,
        steady_state_time_ns=19,
        peak_device_memory_bytes=12345,
    )
    # One pre-timing input barrier plus one result barrier per execution.
    assert len(synchronization_events) == 4
    timing_state = json.loads(json.dumps(timing.to_state()))
    assert SynchronizedJaxTiming.from_state(timing_state) == timing


def test_timing_marks_unmeasured_peak_unknown_and_rejects_bad_clock_or_run_count():
    _, timing = time_jax_callable(
        lambda x: x + 1,
        jnp.asarray(2.0),
        clock_ns=_SequenceClock([10, 12, 20, 23]),
    )
    assert timing.peak_device_memory_bytes is None
    assert timing.to_work_counters().peak_device_memory_bytes is None

    with pytest.raises(ValueError, match="positive"):
        time_jax_callable(lambda x: x, jnp.asarray(1.0), steady_state_runs=0)
    with pytest.raises(RuntimeError, match="moved backwards"):
        time_jax_callable(
            lambda x: x,
            jnp.asarray(1.0),
            clock_ns=_SequenceClock([20, 10]),
        )


def test_synchronized_timing_record_rejects_inconsistent_values():
    with pytest.raises(ValueError, match="at least one"):
        SynchronizedJaxTiming(compile_time_ns=0, steady_state_run_times_ns=())
    with pytest.raises(ValueError, match="nonnegative"):
        SynchronizedJaxTiming(compile_time_ns=-1, steady_state_run_times_ns=(1,))
    with pytest.raises(TypeError, match="integer"):
        SynchronizedJaxTiming(compile_time_ns=1, steady_state_run_times_ns=(1.5,))
    corrupt = SynchronizedJaxTiming(
        compile_time_ns=1,
        steady_state_run_times_ns=(2,),
    ).to_state()
    corrupt["unexpected"] = 3
    with pytest.raises(ValueError, match="schema mismatch"):
        SynchronizedJaxTiming.from_state(corrupt)


def _policy_work(*, confirmation, accepted=False, oracle=True):
    return conditional_policy_iteration_work(
        num_steps=5,
        effective_batch_size=10,
        costate_steps=2,
        value_critic_training_steps=3,
        actor_field_training_steps=4,
        direct_score_diagnostic_size=2,
        acceptance_size=3,
        line_search_candidates=2,
        actor_confirmation_executed=confirmation,
        actor_update_accepted=accepted,
        running_cost_oracle_present=oracle,
    )


def test_conditional_policy_accounting_includes_masked_and_third_objective_work():
    selection_only = _policy_work(confirmation=False)
    confirmed = _policy_work(confirmation=True, accepted=True)

    assert selection_only == MAMWorkCounters(
        running_cost_oracle_evaluations=338,
        simulated_transitions=304,
        tangent_vjps=80,
        tangent_jvps=0,
        optimizer_examples=120,
        optimizer_updates=12,
        peak_device_memory_bytes=None,
    )
    assert confirmed == MAMWorkCounters(
        running_cost_oracle_evaluations=392,
        simulated_transitions=328,
        tangent_vjps=80,
        tangent_jvps=0,
        optimizer_examples=120,
        optimizer_updates=12,
        peak_device_memory_bytes=None,
    )
    # Confirmation adds two rollouts but three full objective evaluations: the
    # current objective is evaluated again for critic calibration.
    assert confirmed.simulated_transitions - selection_only.simulated_transitions == 2 * 3 * 4
    assert (
        confirmed.running_cost_oracle_evaluations - selection_only.running_cost_oracle_evaluations
        == 3 * 3 * 6
    )


def test_final_costate_refresh_accounts_only_disjoint_costate_training_work():
    refresh = conditional_costate_refresh_work(
        num_steps=5,
        effective_batch_size=10,
        costate_steps=2,
        running_cost_oracle_present=True,
    )
    assert refresh == MAMWorkCounters(
        running_cost_oracle_evaluations=120,
        simulated_transitions=80,
        tangent_vjps=80,
        tangent_jvps=0,
        optimizer_examples=20,
        optimizer_updates=2,
        peak_device_memory_bytes=None,
    )


def test_conditional_direct_diagnostic_uses_corrected_arrival_plus_suffix_count():
    without_diagnostic_rows = conditional_policy_iteration_work(
        num_steps=5,
        effective_batch_size=10,
        costate_steps=2,
        value_critic_training_steps=3,
        actor_field_training_steps=4,
        direct_score_diagnostic_size=3,
        acceptance_size=3,
        line_search_candidates=2,
        actor_confirmation_executed=False,
        actor_update_accepted=False,
        running_cost_oracle_present=True,
    )
    with_one_more_row = conditional_policy_iteration_work(
        num_steps=5,
        effective_batch_size=10,
        costate_steps=2,
        value_critic_training_steps=3,
        actor_field_training_steps=4,
        direct_score_diagnostic_size=4,
        acceptance_size=3,
        line_search_candidates=2,
        actor_confirmation_executed=False,
        actor_update_accepted=False,
        running_cost_oracle_present=True,
    )
    # One row adds a base rollout (S), two fully vectorized suffix scans (2S),
    # and its antithetic arrival pair (2 transitions).  Its oracle work is the
    # decomposed pair (2), suffix initial pair (2), and both S-step scans (2S).
    assert (
        with_one_more_row.simulated_transitions - without_diagnostic_rows.simulated_transitions
        == 3 * 4 + 2
    )
    assert (
        with_one_more_row.running_cost_oracle_evaluations
        - without_diagnostic_rows.running_cost_oracle_evaluations
        == 2 * (4 + 2)
    )


def test_absent_running_oracle_zeroes_only_oracle_counter():
    present = _policy_work(confirmation=True, oracle=True)
    absent = _policy_work(confirmation=True, oracle=False)
    assert absent.running_cost_oracle_evaluations == 0
    assert absent.simulated_transitions == present.simulated_transitions
    assert absent.tangent_vjps == present.tangent_vjps
    assert absent.optimizer_examples == present.optimizer_examples
    assert absent.optimizer_updates == present.optimizer_updates


def test_completed_conditional_solve_sums_actual_iterations_and_output_rollout():
    work = completed_conditional_solve_work(
        num_steps=5,
        effective_batch_size=10,
        costate_steps=2,
        value_critic_training_steps=3,
        actor_field_training_steps=4,
        direct_score_diagnostic_size=2,
        acceptance_size=3,
        line_search_candidates=2,
        pair_batch_size=7,
        policy_iterations_completed=2,
        actor_confirmation_executed=(False, True),
        actor_update_accepted=(False, True),
        final_costate_refresh_executed=True,
        running_cost_oracle_present=True,
    )
    assert work == MAMWorkCounters(
        running_cost_oracle_evaluations=850,
        simulated_transitions=740,
        tangent_vjps=240,
        tangent_jvps=0,
        optimizer_examples=260,
        optimizer_updates=26,
        peak_device_memory_bytes=None,
    )


def test_global_half_accounts_projection_validation_refresh_audit_and_optimizer():
    conditional = completed_conditional_solve_work(
        num_steps=5,
        effective_batch_size=10,
        costate_steps=2,
        value_critic_training_steps=3,
        actor_field_training_steps=4,
        direct_score_diagnostic_size=2,
        acceptance_size=3,
        line_search_candidates=2,
        pair_batch_size=7,
        policy_iterations_completed=2,
        actor_confirmation_executed=(False, True),
        actor_update_accepted=(False, True),
        final_costate_refresh_executed=True,
        running_cost_oracle_present=True,
    )
    work = global_half_iteration_work(
        conditional,
        num_steps=5,
        effective_batch_size=10,
        pair_batch_size=7,
        projection_field_training_steps=6,
        projection_validation_size=3,
        projection_line_search_candidates=2,
        projection_validation_replicates=2,
        globally_feasible_before_projection=True,
        projection_confirmation_executed=True,
        projection_update_accepted=False,
        audit_size=4,
        running_cost_oracle_present=True,
    )
    assert work == MAMWorkCounters(
        running_cost_oracle_evaluations=940,
        simulated_transitions=1040,
        tangent_vjps=240,
        tangent_jvps=0,
        optimizer_examples=320,
        optimizer_updates=32,
        peak_device_memory_bytes=None,
    )


def test_infeasible_projection_skips_objective_family_and_oracle():
    conditional = completed_conditional_solve_work(
        num_steps=5,
        effective_batch_size=10,
        costate_steps=2,
        value_critic_training_steps=3,
        actor_field_training_steps=0,
        direct_score_diagnostic_size=2,
        acceptance_size=3,
        line_search_candidates=2,
        pair_batch_size=7,
        policy_iterations_completed=1,
        actor_confirmation_executed=(False,),
        actor_update_accepted=(False,),
        final_costate_refresh_executed=False,
        running_cost_oracle_present=True,
    )
    half = global_half_iteration_work(
        conditional,
        num_steps=5,
        effective_batch_size=10,
        pair_batch_size=7,
        projection_field_training_steps=0,
        projection_validation_size=3,
        projection_line_search_candidates=2,
        projection_validation_replicates=2,
        globally_feasible_before_projection=False,
        projection_confirmation_executed=False,
        projection_update_accepted=False,
        audit_size=4,
        running_cost_oracle_present=True,
    )
    # Two independent endpoint-cloud replicates each evaluate current plus two
    # candidates: 2*3*3*5, followed by refresh 7*5 and two audits 2*4*5.
    assert half.simulated_transitions - conditional.simulated_transitions == 165
    assert half.running_cost_oracle_evaluations == conditional.running_cost_oracle_evaluations
    assert half.optimizer_updates == conditional.optimizer_updates


@pytest.mark.parametrize(
    "call",
    [
        lambda: conditional_policy_iteration_work(
            num_steps=2,
            effective_batch_size=10,
            costate_steps=2,
            value_critic_training_steps=3,
            actor_field_training_steps=0,
            direct_score_diagnostic_size=2,
            acceptance_size=3,
            line_search_candidates=2,
            actor_confirmation_executed=False,
            actor_update_accepted=False,
            running_cost_oracle_present=True,
        ),
        lambda: _policy_work(confirmation=False, accepted=True),
        lambda: completed_conditional_solve_work(
            num_steps=5,
            effective_batch_size=10,
            costate_steps=2,
            value_critic_training_steps=3,
            actor_field_training_steps=0,
            direct_score_diagnostic_size=2,
            acceptance_size=3,
            line_search_candidates=2,
            pair_batch_size=7,
            policy_iterations_completed=2,
            actor_confirmation_executed=(False,),
            actor_update_accepted=(False, False),
            final_costate_refresh_executed=False,
            running_cost_oracle_present=True,
        ),
        lambda: completed_conditional_solve_work(
            num_steps=5,
            effective_batch_size=10,
            costate_steps=2,
            value_critic_training_steps=3,
            actor_field_training_steps=0,
            direct_score_diagnostic_size=2,
            acceptance_size=3,
            line_search_candidates=2,
            pair_batch_size=7,
            policy_iterations_completed=1,
            actor_confirmation_executed=(False,),
            actor_update_accepted=(False,),
            final_costate_refresh_executed=True,
            running_cost_oracle_present=True,
        ),
        lambda: global_half_iteration_work(
            MAMWorkCounters(simulated_transitions=1),
            num_steps=5,
            effective_batch_size=10,
            pair_batch_size=7,
            projection_field_training_steps=0,
            projection_validation_size=3,
            projection_line_search_candidates=2,
            projection_validation_replicates=2,
            globally_feasible_before_projection=False,
            projection_confirmation_executed=False,
            projection_update_accepted=False,
            audit_size=4,
            running_cost_oracle_present=True,
        ),
    ],
)
def test_accounting_calculators_fail_closed_on_inconsistent_inputs(call):
    with pytest.raises((TypeError, ValueError)):
        call()


def test_accounting_calculators_preserve_unbounded_python_integer_counts():
    huge = conditional_policy_iteration_work(
        num_steps=5,
        effective_batch_size=2**62,
        costate_steps=2,
        value_critic_training_steps=3,
        actor_field_training_steps=4,
        direct_score_diagnostic_size=2,
        acceptance_size=3,
        line_search_candidates=2,
        actor_confirmation_executed=False,
        actor_update_accepted=False,
        running_cost_oracle_present=True,
    )
    assert isinstance(huge.simulated_transitions, int)
    assert huge.simulated_transitions > np.iinfo(np.int64).max
