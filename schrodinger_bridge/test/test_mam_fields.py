"""Focused tests for nonlinear MAM actor and endpoint-projection fields."""

from __future__ import annotations

import pickle
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from schrodinger_bridge.network_factory import NetworkFactory
from schrodinger_bridge.solvers.mam_fields import (
    MAMActorDataset,
    MAMActorField,
    MAMEndpointProjectorField,
    MAMFieldConfig,
    MAMProjectionDataset,
    actor_field_predict,
    endpoint_projector_field_predict,
)


class _TinyAffineFactory(NetworkFactory):
    """Fast exact factory for execution and checkpoint tests."""

    def init(self, key, input_dim, output_dim):
        key_w, key_t = jax.random.split(key)
        return {
            "weight": 0.05 * jax.random.normal(key_w, (input_dim, output_dim)),
            "time_weight": 0.05 * jax.random.normal(key_t, (output_dim,)),
            "bias": jnp.zeros((output_dim,)),
        }

    def forward(self, params, x, t):
        return x @ params["weight"] + t[:, None] * params["time_weight"] + params["bias"]


class _NonfiniteFactory(NetworkFactory):
    def init(self, key, input_dim, output_dim):
        del key, input_dim
        return {"bias": jnp.zeros((output_dim,))}

    def forward(self, params, x, t):
        del params, t
        return jnp.full((x.shape[0], 1), jnp.nan, dtype=x.dtype)


def _actor_dataset(row_count: int = 96) -> MAMActorDataset:
    key_x, key_t, key_y, key_s = jax.random.split(jax.random.PRNGKey(11), 4)
    states = jax.random.uniform(key_x, (row_count, 1), minval=-1.0, maxval=1.0)
    times = jax.random.uniform(key_t, (row_count,), minval=0.02, maxval=0.98)
    endpoints = jax.random.uniform(key_y, (row_count, 1), minval=-1.0, maxval=1.0)
    directions = jnp.where(
        jax.random.bernoulli(key_s, shape=(row_count,)),
        1.0,
        -1.0,
    )
    targets = (
        jnp.sin(2.2 * states[:, 0])
        + 0.45 * endpoints[:, 0] ** 2
        + 0.25 * times
        + 0.35 * directions
        + 0.2 * states[:, 0] * endpoints[:, 0]
    )[:, None]
    return MAMActorDataset(states, times, endpoints, directions, targets)


def _projection_dataset(row_count: int = 96) -> MAMProjectionDataset:
    key_x, key_t, key_s = jax.random.split(jax.random.PRNGKey(23), 3)
    states = jax.random.uniform(key_x, (row_count, 1), minval=-1.0, maxval=1.0)
    times = jax.random.uniform(key_t, (row_count,), minval=0.02, maxval=0.98)
    directions = jnp.where(
        jax.random.bernoulli(key_s, shape=(row_count,)),
        1.0,
        -1.0,
    )
    targets = (
        jnp.sin(1.7 * states[:, 0]) + directions * (1.0 - times) * (0.3 + 0.35 * states[:, 0] ** 2)
    )[:, None]
    return MAMProjectionDataset(states, times, directions, targets)


def _fast_config(**changes) -> MAMFieldConfig:
    defaults = {
        "hidden_dims": (32, 32),
        "time_embed_dim": 16,
        "learning_rate": 3e-3,
        "training_steps": 350,
        "microbatch_size": 16,
        "effective_batch_size": 64,
    }
    defaults.update(changes)
    return MAMFieldConfig(**defaults)


def test_actor_and_endpoint_projector_fit_nonlinear_targets():
    actor_data = _actor_dataset()
    actor = MAMActorField(1, 1, _fast_config())
    actor_state = actor.fit(jax.random.PRNGKey(101), actor_data)
    actor_prediction = actor.predict(
        actor_state,
        actor_data.states,
        actor_data.times,
        actor_data.endpoints,
        actor_data.directions,
    )
    actor_mse = float(jnp.mean((actor_prediction - actor_data.targets) ** 2))

    projection_data = _projection_dataset()
    projector = MAMEndpointProjectorField(1, 1, _fast_config())
    projector_state = projector.fit(jax.random.PRNGKey(202), projection_data)
    projection = projector.predict(
        projector_state,
        projection_data.states,
        projection_data.times,
        projection_data.directions,
    )
    projection_mse = float(jnp.mean((projection - projection_data.targets) ** 2))

    assert actor_mse < 5e-2
    assert projection_mse < 1.5e-2
    assert actor_prediction.dtype == jnp.float32
    assert projection.dtype == jnp.float32
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree_util.tree_leaves(actor_state.params))


def test_array_prediction_kernels_match_jit_and_preserve_float32():
    factory = _TinyAffineFactory()
    actor_params = factory.init(jax.random.PRNGKey(1), 4, 2)
    states = np.asarray([[0.2, -0.1], [0.4, 0.7]], dtype=np.float64)
    times = np.asarray([0.25, 0.75], dtype=np.float64)
    endpoints = np.asarray([[1.0], [-0.5]], dtype=np.float64)
    directions = np.asarray([1.0, -1.0], dtype=np.float64)

    eager_actor = actor_field_predict(
        factory,
        actor_params,
        states,
        times,
        endpoints,
        directions,
    )
    compiled_actor = jax.jit(
        lambda x, t, y, s: actor_field_predict(
            factory,
            actor_params,
            x,
            t,
            y,
            s,
        )
    )(states, times, endpoints, directions)
    np.testing.assert_allclose(compiled_actor.value, eager_actor.value, rtol=1e-6)
    np.testing.assert_array_equal(compiled_actor.finite, eager_actor.finite)
    assert compiled_actor.value.dtype == jnp.float32

    projector_params = factory.init(jax.random.PRNGKey(2), 3, 1)
    eager_projector = endpoint_projector_field_predict(
        factory,
        projector_params,
        states,
        times,
        directions,
        1,
    )
    compiled_projector = jax.jit(
        lambda x, t, s: endpoint_projector_field_predict(
            factory,
            projector_params,
            x,
            t,
            s,
            1,
        )
    )(states, times, directions)
    np.testing.assert_allclose(compiled_projector.value, eager_projector.value, rtol=1e-6)
    np.testing.assert_array_equal(compiled_projector.finite, eager_projector.finite)
    assert compiled_projector.value.dtype == jnp.float32


def test_deterministic_replay_resume_and_serializable_complete_state():
    config = _fast_config(
        training_steps=4,
        microbatch_size=4,
        effective_batch_size=8,
        network_factory=_TinyAffineFactory(),
    )
    data = _actor_dataset(row_count=24)
    actor = MAMActorField(1, 1, config)
    key = jax.random.PRNGKey(77)
    first = actor.fit(key, data)
    replay = actor.fit(key, data)

    first_leaves = jax.tree_util.tree_leaves(
        (first.params, first.optimizer, first.next_key, first.loss_history)
    )
    replay_leaves = jax.tree_util.tree_leaves(
        (replay.params, replay.optimizer, replay.next_key, replay.loss_history)
    )
    for left, right in zip(first_leaves, replay_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    initialized = actor.initialize(key)
    halfway = actor.train(initialized, data, steps=2)
    resumed = actor.train(halfway, data, steps=2)
    for left, right in zip(
        jax.tree_util.tree_leaves(first.params),
        jax.tree_util.tree_leaves(resumed.params),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    np.testing.assert_array_equal(first.loss_history, resumed.loss_history)

    payload = pickle.loads(pickle.dumps(first.to_state_dict()))
    restored = actor.load_state_dict(payload)
    assert restored.completed_steps == 4
    assert int(restored.optimizer.step) == 4
    np.testing.assert_array_equal(restored.next_key, first.next_key)
    np.testing.assert_array_equal(restored.loss_history, first.loss_history)
    before = actor.predict(
        first,
        data.states,
        data.times,
        data.endpoints,
        data.directions,
    )
    after = actor.predict(
        restored,
        data.states,
        data.times,
        data.endpoints,
        data.directions,
    )
    np.testing.assert_array_equal(before, after)


def test_static_microbatch_accumulation_matches_unsplit_effective_batch():
    # Every row is identical, so both sampled effective batches contain the
    # same eight examples.  Any difference would be caused by accumulation,
    # not by a different random draw.
    states = jnp.full((8, 1), 0.25)
    times = jnp.full((8,), 0.4)
    endpoints = jnp.full((8, 1), -0.5)
    directions = jnp.ones((8,))
    targets = jnp.full((8, 1), 0.75)
    data = MAMActorDataset(states, times, endpoints, directions, targets)
    common = {
        "hidden_dims": (4,),
        "time_embed_dim": 4,
        "learning_rate": 1e-2,
        "training_steps": 1,
        "effective_batch_size": 8,
        "network_factory": _TinyAffineFactory(),
    }
    unsplit = MAMActorField(1, 1, MAMFieldConfig(microbatch_size=8, **common))
    accumulated = MAMActorField(1, 1, MAMFieldConfig(microbatch_size=2, **common))
    key = jax.random.PRNGKey(19)
    state_unsplit = unsplit.fit(key, data)
    state_accumulated = accumulated.fit(key, data)

    for left, right in zip(
        jax.tree_util.tree_leaves(state_unsplit.params),
        jax.tree_util.tree_leaves(state_accumulated.params),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        state_unsplit.loss_history,
        state_accumulated.loss_history,
        rtol=2e-6,
        atol=2e-7,
    )


def test_nonfinite_data_factory_prediction_and_checkpoint_fail_closed():
    data = _actor_dataset(row_count=8)
    actor = MAMActorField(
        1,
        1,
        _fast_config(
            training_steps=1,
            microbatch_size=2,
            effective_batch_size=4,
            network_factory=_TinyAffineFactory(),
        ),
    )
    with pytest.raises(FloatingPointError, match="targets"):
        actor.fit(
            jax.random.PRNGKey(0),
            replace(data, targets=data.targets.at[0, 0].set(jnp.nan)),
        )
    with pytest.raises(ValueError, match=r"only -1 or \+1"):
        actor.fit(
            jax.random.PRNGKey(0),
            replace(data, directions=data.directions.at[0].set(0.0)),
        )
    with pytest.raises(TypeError, match="real floating dtype"):
        actor.fit(
            jax.random.PRNGKey(0),
            replace(data, states=data.states.astype(jnp.complex64)),
        )
    with pytest.raises(TypeError, match="real non-boolean"):
        actor.fit(
            jax.random.PRNGKey(0),
            replace(data, directions=data.directions > 0.0),
        )
    with pytest.raises(TypeError, match="real non-boolean"):
        actor.fit(
            jax.random.PRNGKey(0),
            replace(data, directions=data.directions.astype(jnp.complex64)),
        )

    bad_actor = MAMActorField(
        1,
        1,
        _fast_config(
            training_steps=1,
            microbatch_size=2,
            effective_batch_size=4,
            network_factory=_NonfiniteFactory(),
        ),
    )
    with pytest.raises(FloatingPointError, match="factory"):
        bad_actor.fit(jax.random.PRNGKey(0), data)

    state = actor.fit(jax.random.PRNGKey(3), data)
    payload = state.to_state_dict()
    first_key = next(iter(payload["params"]))
    payload["params"][first_key] = np.full_like(payload["params"][first_key], np.nan)
    with pytest.raises(FloatingPointError, match="train state"):
        actor.load_state_dict(payload)

    raw = actor_field_predict(
        actor.factory,
        state.params,
        data.states,
        data.times,
        data.endpoints,
        data.directions.at[0].set(0.0),
    )
    assert not bool(raw.finite[0])
    assert bool(jnp.all(jnp.isnan(raw.value[0])))
    with pytest.raises(FloatingPointError, match="states"):
        actor.predict(
            state,
            data.states.at[0, 0].set(jnp.inf),
            data.times,
            data.endpoints,
            data.directions,
        )


def test_config_and_shape_validation_are_fail_closed():
    with pytest.raises(ValueError, match="divisible"):
        MAMFieldConfig(microbatch_size=3, effective_batch_size=8)
    with pytest.raises(ValueError, match="training_steps"):
        MAMFieldConfig(training_steps=1.5)
    with pytest.raises(ValueError, match="positive"):
        MAMActorField(0)

    data = _projection_dataset(row_count=8)
    projector = MAMEndpointProjectorField(
        1,
        1,
        _fast_config(
            training_steps=1,
            microbatch_size=2,
            effective_batch_size=4,
            network_factory=_TinyAffineFactory(),
        ),
    )
    with pytest.raises(ValueError, match="targets"):
        projector.fit(
            jax.random.PRNGKey(1),
            replace(data, targets=jnp.zeros((8, 2))),
        )
