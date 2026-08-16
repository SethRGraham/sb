"""Tests for the cross-fitted scalar MAM value critic."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from schrodinger_bridge.network_factory import NetworkFactory
from schrodinger_bridge.solvers.mam_value_critic import (
    CRITIC_AUTODIFF_METHOD,
    CrossFittedValueCritic,
    FixedPolicyReturnDataset,
    ValueCriticConfig,
    cross_fitted_critic_autodiff,
    cross_fitted_value_predictions,
)


class _AffineScalarFactory(NetworkFactory):
    """Small exact factory whose context/time use is easy to falsify."""

    def init(self, key, input_dim, output_dim):
        if output_dim != 1:
            raise ValueError("affine test critic is scalar")
        return {
            "weight": 0.01 * jax.random.normal(key, (input_dim,)),
            "time_weight": jnp.asarray(0.0),
            "bias": jnp.asarray(0.0),
        }

    def forward(self, params, x, t):
        value = x @ params["weight"] + t * params["time_weight"] + params["bias"]
        return value[:, None]


class _NonfiniteFactory(NetworkFactory):
    def init(self, key, input_dim, output_dim):
        del key, input_dim, output_dim
        return {"value": jnp.asarray(0.0)}

    def forward(self, params, x, t):
        del params, t
        return jnp.full((x.shape[0], 1), jnp.nan, dtype=x.dtype)


class _InputIgnoringFactory(NetworkFactory):
    """Adversarial factory for checking validity independent of its output."""

    def init(self, key, input_dim, output_dim):
        del key, input_dim, output_dim
        return {"unused": jnp.asarray(0.0)}

    def forward(self, params, x, t):
        del params, t
        return jnp.full((x.shape[0], 1), 2.0, dtype=x.dtype)


def _manual_params(state_weight, context_weight, time_weight, bias=0.0):
    return {
        "weight": jnp.asarray([state_weight, context_weight], dtype=jnp.float32),
        "time_weight": jnp.asarray(time_weight, dtype=jnp.float32),
        "bias": jnp.asarray(bias, dtype=jnp.float32),
    }


def _dataset(row_count=32):
    states = jnp.linspace(-1.0, 1.0, row_count)[:, None]
    times = jnp.linspace(0.05, 0.95, row_count)
    endpoints = jnp.linspace(0.8, -0.4, row_count)[:, None]
    returns = 1.25 * states[:, 0] - 0.75 * endpoints[:, 0] + 0.4 * times + 0.2
    return FixedPolicyReturnDataset(
        states=states,
        times=times,
        endpoint_context=endpoints,
        returns=returns,
        policy_fingerprint="fixed-test-policy-v1",
    )


def test_opposite_fold_baseline_uses_context_and_time_and_is_stopped():
    factory = _AffineScalarFactory()
    params0 = _manual_params(1.0, 10.0, 100.0)
    params1 = _manual_params(2.0, 20.0, 200.0)
    states = jnp.asarray([[1.0], [2.0]], dtype=jnp.float32)
    times = jnp.asarray([0.1, 0.2], dtype=jnp.float32)
    context = jnp.asarray([[3.0], [4.0]], dtype=jnp.float32)
    folds = jnp.asarray([0, 1], dtype=jnp.int32)

    batch = cross_fitted_value_predictions(
        factory, (params0, params1), states, times, context, folds
    )
    # Fold-zero row uses model one; fold-one row uses model zero.
    np.testing.assert_allclose(np.asarray(batch.value), np.asarray([82.0, 62.0]))
    np.testing.assert_array_equal(np.asarray(batch.source_training_fold), [1, 0])
    assert bool(jnp.all(batch.finite))
    assert bool(batch.stopped)

    def summed_prediction(variable_states):
        return jnp.sum(
            cross_fitted_value_predictions(
                factory,
                (params0, params1),
                variable_states,
                times,
                context,
                folds,
            ).value
        )

    np.testing.assert_array_equal(np.asarray(jax.grad(summed_prediction)(states)), 0.0)
    compiled = jax.jit(
        lambda x: (
            cross_fitted_value_predictions(
                factory, (params0, params1), x, times, context, folds
            ).value
        )
    )(states)
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(batch.value))


def test_critic_autodiff_is_state_gradient_of_opposite_fold_only():
    factory = _AffineScalarFactory()
    params0 = _manual_params(1.0, 10.0, 100.0)
    params1 = _manual_params(2.0, 20.0, 200.0)
    states = jnp.asarray([[1.0], [2.0]], dtype=jnp.float32)
    times = jnp.asarray([0.1, 0.2], dtype=jnp.float32)
    context = jnp.asarray([[3.0], [4.0]], dtype=jnp.float32)
    folds = jnp.asarray([0, 1], dtype=jnp.int32)

    batch = cross_fitted_critic_autodiff(factory, (params0, params1), states, times, context, folds)
    np.testing.assert_allclose(np.asarray(batch.costate), np.asarray([[2.0], [1.0]]))
    np.testing.assert_array_equal(np.asarray(batch.source_training_fold), [1, 0])
    assert int(batch.method) == CRITIC_AUTODIFF_METHOD
    assert bool(jnp.all(batch.finite))


def test_fit_is_balanced_disjoint_reproducible_and_uses_accumulation():
    dataset = _dataset()
    critic = CrossFittedValueCritic(
        ValueCriticConfig(
            learning_rate=5e-2,
            training_steps=120,
            microbatch_size=4,
            effective_batch_size=16,
            network_factory=_AffineScalarFactory(),
        )
    )
    key = jax.random.PRNGKey(91)
    first = critic.fit(key, dataset)
    second = critic.fit(key, dataset)

    folds = np.asarray(first.row_fold)
    assert set(folds.tolist()) == {0, 1}
    assert np.count_nonzero(folds == 0) == np.count_nonzero(folds == 1) == 16
    assert first.loss_history.shape == (2, 120)
    assert int(first.final_metrics[0]["training_row_count"]) == 16
    assert int(first.final_metrics[1]["training_row_count"]) == 16
    assert float(jnp.max(first.loss_history[:, -1])) < 2e-3
    np.testing.assert_array_equal(np.asarray(first.row_fold), np.asarray(second.row_fold))
    for a, b in zip(
        jax.tree_util.tree_leaves(first.params_by_training_fold),
        jax.tree_util.tree_leaves(second.params_by_training_fold),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    baseline = critic.cross_fitted_baseline(first, dataset)
    comparison = critic.critic_autodiff_comparison(first, dataset)
    assert baseline.value.shape == (32,)
    assert comparison.costate.shape == (32, 1)
    assert bool(jnp.all(baseline.finite))
    assert bool(jnp.all(comparison.finite))


def test_explicit_fold_assignment_is_respected_and_must_have_both_folds():
    dataset = _dataset(row_count=6)
    critic = CrossFittedValueCritic(
        ValueCriticConfig(
            training_steps=1,
            microbatch_size=2,
            effective_batch_size=2,
            network_factory=_AffineScalarFactory(),
        )
    )
    assignment = jnp.asarray([0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    result = critic.fit(jax.random.PRNGKey(2), dataset, row_fold=assignment)
    np.testing.assert_array_equal(np.asarray(result.row_fold), np.asarray(assignment))

    with pytest.raises(ValueError, match="both cross-fitting folds"):
        critic.fit(
            jax.random.PRNGKey(2),
            dataset,
            row_fold=jnp.zeros((6,), dtype=jnp.int32),
        )
    with pytest.raises(TypeError, match="integer dtype"):
        critic.fit(
            jax.random.PRNGKey(2),
            dataset,
            row_fold=jnp.asarray([True, False, True, False, True, False]),
        )


@pytest.mark.parametrize(
    ("overrides", "exception", "match"),
    [
        ({"training_steps": True}, TypeError, "training_steps must be an integer"),
        ({"training_steps": 1.5}, TypeError, "training_steps must be an integer"),
        ({"microbatch_size": False}, TypeError, "microbatch_size must be an integer"),
        ({"effective_batch_size": 2.5}, TypeError, "effective_batch_size must be an integer"),
        ({"learning_rate": float("nan")}, ValueError, "learning_rate must be finite"),
        ({"learning_rate": float("inf")}, ValueError, "learning_rate must be finite"),
        ({"learning_rate": 0.0}, ValueError, "learning_rate must be greater than"),
        ({"learning_rate": -1e-3}, ValueError, "learning_rate must be greater than"),
        ({"weight_decay": float("nan")}, ValueError, "weight_decay must be finite"),
    ],
)
def test_value_critic_config_rejects_invalid_numeric_hyperparameters(overrides, exception, match):
    with pytest.raises(exception, match=match):
        ValueCriticConfig(**overrides)


def test_nonfinite_data_factory_and_crossfit_prediction_fail_closed():
    dataset = _dataset(row_count=8)
    critic = CrossFittedValueCritic(
        ValueCriticConfig(
            training_steps=1,
            microbatch_size=2,
            effective_batch_size=2,
            network_factory=_AffineScalarFactory(),
        )
    )
    bad_dataset = replace(dataset, returns=dataset.returns.at[0].set(jnp.nan))
    with pytest.raises(FloatingPointError, match="returns"):
        critic.fit(jax.random.PRNGKey(0), bad_dataset)

    bad_critic = CrossFittedValueCritic(
        ValueCriticConfig(
            training_steps=1,
            microbatch_size=2,
            effective_batch_size=2,
            network_factory=_NonfiniteFactory(),
        )
    )
    with pytest.raises(FloatingPointError, match="factory"):
        bad_critic.fit(jax.random.PRNGKey(0), dataset)

    params0 = _manual_params(1.0, 1.0, 1.0)
    params1 = _manual_params(1.0, 1.0, 1.0)
    invalid = cross_fitted_value_predictions(
        _AffineScalarFactory(),
        (params0, params1),
        dataset.states,
        dataset.times,
        dataset.endpoint_context,
        jnp.full((8,), 2, dtype=jnp.int32),
    )
    assert not bool(jnp.any(invalid.finite))
    assert bool(jnp.all(jnp.isnan(invalid.value)))


@pytest.mark.parametrize("field", ["state", "time", "context", "selected_params"])
def test_crossfit_kernel_rejects_nonfinite_inputs_even_when_factory_ignores_them(field):
    factory = _InputIgnoringFactory()
    params0 = {"unused": jnp.asarray(0.0, dtype=jnp.float32)}
    params1 = {"unused": jnp.asarray(0.0, dtype=jnp.float32)}
    states = jnp.zeros((2, 1), dtype=jnp.float32)
    times = jnp.zeros((2,), dtype=jnp.float32)
    context = jnp.zeros((2, 1), dtype=jnp.float32)
    folds = jnp.asarray([0, 1], dtype=jnp.int32)
    expected = np.asarray([True, True])
    if field == "state":
        states = states.at[0, 0].set(jnp.nan)
        expected[0] = False
    elif field == "time":
        times = times.at[0].set(jnp.nan)
        expected[0] = False
    elif field == "context":
        context = context.at[0, 0].set(jnp.nan)
        expected[0] = False
    else:
        # Fold-zero rows select params1, while fold-one rows select params0.
        params1 = {"unused": jnp.asarray(jnp.nan, dtype=jnp.float32)}
        expected[0] = False

    def evaluate(x, t, c):
        return cross_fitted_value_predictions(factory, (params0, params1), x, t, c, folds)

    eager = evaluate(states, times, context)
    compiled = jax.jit(evaluate)(states, times, context)
    np.testing.assert_array_equal(np.asarray(eager.finite), expected)
    np.testing.assert_array_equal(np.asarray(compiled.finite), expected)
    assert np.isnan(np.asarray(eager.value)[~expected]).all()

    autodiff = cross_fitted_critic_autodiff(
        factory,
        (params0, params1),
        states,
        times,
        context,
        folds,
    )
    np.testing.assert_array_equal(np.asarray(autodiff.finite), expected)


def test_result_refuses_changed_policy_provenance():
    dataset = _dataset(row_count=8)
    critic = CrossFittedValueCritic(
        ValueCriticConfig(
            training_steps=1,
            microbatch_size=2,
            effective_batch_size=2,
            network_factory=_AffineScalarFactory(),
        )
    )
    result = critic.fit(jax.random.PRNGKey(7), dataset)
    changed = replace(dataset, policy_fingerprint="different-policy")
    with pytest.raises(ValueError, match="policy fingerprint"):
        critic.cross_fitted_baseline(result, changed)

    reordered = replace(
        dataset,
        states=dataset.states[::-1],
        times=dataset.times[::-1],
        endpoint_context=dataset.endpoint_context[::-1],
        returns=dataset.returns[::-1],
    )
    with pytest.raises(ValueError, match="row order"):
        critic.cross_fitted_baseline(result, reordered)

    tampered_assignment = replace(result, row_fold=1 - result.row_fold)
    with pytest.raises(ValueError, match="assignment fingerprint"):
        critic.cross_fitted_baseline(tampered_assignment, dataset)


def test_fold_state_fingerprints_bind_model_adam_pairs_to_declared_fold():
    """A valid pair cannot be relabeled by exchanging the two tuple entries."""
    original = _dataset(row_count=8)
    # Checkpoint validation is deliberately float32 production semantics even
    # when this test is run with JAX x64 enabled.
    dataset = replace(
        original,
        states=original.states.astype(jnp.float32),
        times=original.times.astype(jnp.float32),
        endpoint_context=original.endpoint_context.astype(jnp.float32),
        returns=original.returns.astype(jnp.float32),
    )
    critic = CrossFittedValueCritic(
        ValueCriticConfig(
            training_steps=2,
            microbatch_size=2,
            effective_batch_size=2,
            network_factory=_AffineScalarFactory(),
        )
    )
    result = critic.fit(jax.random.PRNGKey(17), dataset)
    validation_kwargs = {"state_dim": 1, "context_dim": 1, "row_count": 8}

    # A fresh result remains checkpoint-valid and deterministic.
    critic.validate_result_state(result, **validation_kwargs)
    repeated = critic.fit(jax.random.PRNGKey(17), dataset)
    assert result.fold_provenance_fingerprints == repeated.fold_provenance_fingerprints
    assert result.fold_state_fingerprints == repeated.fold_state_fingerprints

    swapped_pairs = replace(
        result,
        params_by_training_fold=tuple(reversed(result.params_by_training_fold)),
        optimizer_by_training_fold=tuple(reversed(result.optimizer_by_training_fold)),
    )
    with pytest.raises(ValueError, match=r"fold 0 state fingerprint mismatch"):
        critic.validate_result_state(swapped_pairs, **validation_kwargs)
    # Prediction entry points use the same integrity check, rather than only
    # trusting it during an explicit checkpoint restore.
    with pytest.raises(ValueError, match=r"fold 0 state fingerprint mismatch"):
        critic.cross_fitted_baseline(swapped_pairs, dataset)

    swapped_everything = replace(
        swapped_pairs,
        fold_provenance_fingerprints=tuple(reversed(result.fold_provenance_fingerprints)),
        fold_state_fingerprints=tuple(reversed(result.fold_state_fingerprints)),
    )
    with pytest.raises(ValueError, match=r"fold 0 provenance fingerprint mismatch"):
        critic.validate_result_state(swapped_everything, **validation_kwargs)
