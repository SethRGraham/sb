"""CPU-testable checks for single-host MAM execution infrastructure."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from schrodinger_bridge.solvers.mam_execution import (
    RNGDomain,
    RNGLedger,
    accumulate_gradient_sequence,
    accumulate_gradient_step,
    discover_device_topology,
    finalize_gradient_accumulator,
    initialize_gradient_accumulator,
    iter_cache_shards,
    make_cache_shard_plan,
    make_execution_plan,
    mean_gradient_replicas,
    prefetch_cache_shards,
    require_valid_gradients,
    resolve_local_devices,
    shard_microbatch_axis,
    static_microbatch_indices,
    take_scheduled_batches,
)


def _dtype():
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


def test_execution_plan_uses_declared_single_gpu_first_defaults():
    small = make_execution_plan(horizon=128, state_dim=64)
    large = make_execution_plan(horizon=128, state_dim=128)
    assert small.microbatch_size == 128
    assert small.accumulation_steps == 8
    assert small.device_count == 1
    assert small.per_device_microbatch_size == 128
    assert large.microbatch_size == 32
    assert large.accumulation_steps == 32
    assert large.production_dtype == "float32"
    assert large.matrix_free
    json.dumps(small.to_state())


def test_execution_plan_fails_closed_on_unmeasured_or_inconsistent_layouts():
    with pytest.raises(ValueError, match="no unmeasured default"):
        make_execution_plan(horizon=129, state_dim=64)
    with pytest.raises(ValueError, match="divisible"):
        make_execution_plan(
            horizon=16,
            state_dim=4,
            microbatch_size=30,
            effective_batch_size=128,
        )
    with pytest.raises(ValueError, match="at most two"):
        make_execution_plan(horizon=16, state_dim=4, device_count=3)


def test_static_microbatch_schedule_is_deterministic_complete_and_jittable():
    plan = make_execution_plan(
        horizon=8,
        state_dim=2,
        microbatch_size=4,
        effective_batch_size=8,
    )
    key = jax.random.PRNGKey(7)
    eager = static_microbatch_indices(key, num_examples=16, plan=plan)
    compiled = jax.jit(lambda rng: static_microbatch_indices(rng, num_examples=16, plan=plan))(key)
    assert eager.shape == (2, 2, 4)
    np.testing.assert_array_equal(eager, compiled)
    np.testing.assert_array_equal(np.sort(np.asarray(eager).ravel()), np.arange(16))
    with pytest.raises(ValueError, match="divisible"):
        static_microbatch_indices(key, num_examples=15, plan=plan)


def test_schedule_gather_and_device_axis_sharding_preserve_pytree_axes():
    plan = make_execution_plan(
        horizon=8,
        state_dim=2,
        microbatch_size=4,
        effective_batch_size=8,
        device_count=2,
    )
    indices = static_microbatch_indices(
        jax.random.PRNGKey(0), num_examples=8, plan=plan, shuffle=False
    )
    cache = {
        "x": jnp.arange(16, dtype=_dtype()).reshape(8, 2),
        "y": jnp.arange(8, dtype=_dtype()),
    }
    scheduled = take_scheduled_batches(cache, indices)
    assert scheduled["x"].shape == (1, 2, 4, 2)
    microbatch = jax.tree_util.tree_map(lambda value: value[0, 0], scheduled)
    sharded = shard_microbatch_axis(microbatch, plan)
    assert sharded["x"].shape == (2, 2, 2)
    assert sharded["y"].shape == (2, 2)


def test_gradient_accumulation_scan_matches_exact_mean_and_jit():
    sequence = {
        "w": jnp.asarray([[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]], dtype=_dtype()),
        "b": jnp.asarray([[2.0], [4.0], [6.0]], dtype=_dtype()),
    }
    eager = accumulate_gradient_sequence(sequence)
    compiled = jax.jit(accumulate_gradient_sequence)(sequence)
    expected = jax.tree_util.tree_map(lambda value: jnp.mean(value, axis=0), sequence)
    assert bool(eager.finite)
    assert bool(eager.count_matches)
    for actual, reference, jitted in zip(
        jax.tree_util.tree_leaves(eager.gradients),
        jax.tree_util.tree_leaves(expected),
        jax.tree_util.tree_leaves(compiled.gradients),
        strict=True,
    ):
        np.testing.assert_allclose(actual, reference)
        np.testing.assert_allclose(jitted, reference)


def test_gradient_accumulation_marks_nonfinite_and_incomplete_updates():
    example = {"w": jnp.ones((2,), dtype=_dtype())}
    initial = initialize_gradient_accumulator(example)
    one_step = accumulate_gradient_step(initial, example)
    incomplete = finalize_gradient_accumulator(one_step, expected_steps=2)
    assert not bool(incomplete.count_matches)
    assert np.isnan(np.asarray(incomplete.gradients["w"])).all()
    with pytest.raises(RuntimeError, match="incomplete"):
        require_valid_gradients(incomplete)

    bad = accumulate_gradient_step(initial, {"w": jnp.asarray([jnp.nan, 1.0])})
    nonfinite = finalize_gradient_accumulator(bad, expected_steps=1)
    assert not bool(nonfinite.finite)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        require_valid_gradients(nonfinite)


def test_gradient_finalization_rechecks_aggregate_overflow():
    """Finite microbatches can overflow while being accumulated."""
    maximum = jnp.asarray(jnp.finfo(_dtype()).max, dtype=_dtype())
    example = {"w": maximum[None]}
    accumulated = initialize_gradient_accumulator(example)
    accumulated = accumulate_gradient_step(accumulated, example)
    accumulated = accumulate_gradient_step(accumulated, example)
    assert bool(accumulated.finite)
    result = finalize_gradient_accumulator(accumulated, expected_steps=2)
    assert not bool(result.finite)
    assert np.isnan(np.asarray(result.gradients["w"])).all()
    with pytest.raises(FloatingPointError, match="nonfinite"):
        require_valid_gradients(result)


def test_reference_replica_mean_averages_only_leading_device_axis():
    replicas = {
        "w": jnp.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=_dtype()),
        "b": jnp.asarray([[2.0], [6.0]], dtype=_dtype()),
    }
    averaged = mean_gradient_replicas(replicas, device_count=2)
    np.testing.assert_allclose(averaged["w"], jnp.asarray([3.0, 5.0]))
    np.testing.assert_allclose(averaged["b"], jnp.asarray([4.0]))


def test_device_topology_defaults_to_one_and_round_trips_current_devices():
    topology = discover_device_topology()
    assert topology.selected_device_count == 1
    assert not topology.batch_data_parallel
    assert len(resolve_local_devices(topology)) == 1
    state = topology.to_state()
    assert state["selected_device_count"] == 1
    json.dumps(state)


def test_rng_ledger_domains_are_order_independent_and_serializable():
    initial = RNGLedger(root_seed=2026)
    costate_a, after_costate = initial.next(RNGDomain.COSTATE_FIT, 3, 0)
    actor_a, after_both_a = after_costate.next(RNGDomain.ACTOR_TARGET_FIT, 3, 0)

    actor_b, after_actor = initial.next(RNGDomain.ACTOR_TARGET_FIT, 3, 0)
    costate_b, after_both_b = after_actor.next(RNGDomain.COSTATE_FIT, 3, 0)
    np.testing.assert_array_equal(costate_a, costate_b)
    np.testing.assert_array_equal(actor_a, actor_b)
    assert not np.array_equal(np.asarray(costate_a), np.asarray(actor_a))

    restored = RNGLedger.from_state(json.loads(json.dumps(after_both_a.to_state())))
    key_a, next_a = after_both_a.next("confirmation", 4)
    key_b, next_b = restored.next("confirmation", 4)
    np.testing.assert_array_equal(key_a, key_b)
    assert next_a.to_state() == next_b.to_state()
    # Cross-domain consumption order changes no domain-local counter state.
    assert after_both_a.to_state() == after_both_b.to_state()


def test_rng_ledger_allocation_matches_individual_derivation_and_rejects_schema_drift():
    ledger = RNGLedger(root_seed=1)
    keys, advanced = ledger.allocate(RNGDomain.REPORTING, 3, 9)
    for index in range(3):
        np.testing.assert_array_equal(keys[index], ledger.key_for(RNGDomain.REPORTING, index, 9))
    assert advanced.to_state()["counters"]["reporting"] == 3
    corrupt = advanced.to_state()
    corrupt["counters"] = dict(corrupt["counters"])
    corrupt["counters"].pop("reporting")
    with pytest.raises(ValueError, match="domains"):
        RNGLedger.from_state(corrupt)


def test_cache_shards_retain_every_item_and_prefetch_on_cpu_device():
    cache = {
        "x": np.arange(24, dtype=np.float32).reshape(12, 2),
        "y": np.arange(12, dtype=np.float32),
    }
    plan = make_cache_shard_plan(num_items=12, shard_size=4)
    host_shards = list(iter_cache_shards(cache, plan))
    assert len(host_shards) == 3
    assert [(shard.start, shard.stop) for shard in host_shards] == [(0, 4), (4, 8), (8, 12)]
    prefetched = list(prefetch_cache_shards(iter(host_shards), buffer_size=2))
    reconstructed = np.concatenate([np.asarray(shard.data["x"]) for shard in prefetched])
    np.testing.assert_array_equal(reconstructed, cache["x"])
    assert all(isinstance(shard.data["x"], jax.Array) for shard in prefetched)


def test_cache_shard_plan_requires_explicit_partial_final_shape():
    with pytest.raises(ValueError, match="partial final shard"):
        make_cache_shard_plan(num_items=10, shard_size=4)
    plan = make_cache_shard_plan(
        num_items=10,
        shard_size=4,
        allow_partial_final_shard=True,
    )
    assert not plan.static_shapes
    assert plan.final_shard_size == 2
    shards = list(iter_cache_shards(np.arange(10), plan))
    assert [shard.stop - shard.start for shard in shards] == [4, 4, 2]
