"""Falsification checks for the query-matched MAM Gate-A-v2 DEV module."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.malliavin_adjoint_matching import foundation_v2

jax.config.update("jax_enable_x64", True)


def _config(output_dir: Path) -> dict:
    return {
        "experiment": {
            "protocol": "mam_gate_a_v2_dev",
            "seed": 7,
            "dev_replicates": [0, 1, 2],
            "output_dir": str(output_dir),
            "intended_for_scientific_evidence": False,
        },
        "numerics": {"compute_dtype": "float64", "matmul_precision": "highest"},
        "dynamics": {
            "family": "brownian_threshold",
            "state_dim": 1,
            "horizon": 1.0,
            "steps": 8,
            "sigma": 0.7,
            "threshold": 0.4,
        },
        "anchors": {"minimum_remaining_steps": 2, "distribution": "uniform_discrete"},
        "costate_network": {
            "hidden_sizes": [8, 8],
            "activation": "silu",
            "time_embedding_dim": 4,
            "optimizer": "adam",
            "learning_rates": [0.001],
            "training_steps": 4,
            "batch_size": 4,
            "eval_every": 2,
        },
        "sampling": {
            "reward_query_budget": 16,
            "dev_contexts": 8,
            "state_range": [-0.5, 1.0],
        },
    }


def _sample(arm: str, query_budget: int, seed: int = 0):
    root = jax.random.PRNGKey(seed)
    return foundation_v2.sample_query_matched_dataset(
        jax.random.fold_in(root, 1),
        jax.random.fold_in(root, 2),
        arm,
        independent_second_noise_key=(jax.random.fold_in(root, 3) if arm == "IID2" else None),
        reward_query_budget=query_budget,
        sigma=0.7,
        horizon=1.0,
        threshold=0.4,
        time_steps=8,
        minimum_remaining_steps=2,
        state_minimum=-0.5,
        state_maximum=1.0,
        dtype=jnp.float64,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_explicit_noise_constructor_is_auditable_stopped_and_jittable() -> None:
    states = jnp.asarray([[0.0], [0.5]], dtype=jnp.float64)
    times = jnp.asarray([0.25, 0.5], dtype=jnp.float64)
    noise = jnp.asarray([[[1.0], [-1.0]], [[0.25], [-0.25]]], dtype=jnp.float64)
    kwargs = {"sigma": 0.7, "horizon": 1.0, "threshold": 0.4}
    eager = foundation_v2.hard_bel_from_noise(states, times, noise, **kwargs)
    compiled = jax.jit(
        lambda state, time, normal: foundation_v2.hard_bel_from_noise(state, time, normal, **kwargs)
    )(states, times, noise)
    np.testing.assert_array_equal(eager.rewards, compiled.rewards)
    np.testing.assert_allclose(
        eager.single_query_labels,
        compiled.single_query_labels,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    scale = 0.7 * np.sqrt(1.0 - np.asarray(times))
    expected = np.asarray(eager.rewards)[..., None] * np.asarray(noise) / scale[:, None, None]
    np.testing.assert_allclose(eager.single_query_labels, expected, rtol=1.0e-13, atol=0.0)
    assert eager.reward_query_count == 4

    reward_gradient = jax.grad(
        lambda shift: jnp.sum(
            foundation_v2.hard_bel_from_noise(states + shift, times, noise, **kwargs).rewards
        )
    )(jnp.asarray(0.0, dtype=jnp.float64))
    label_gradient = jax.grad(
        lambda shift: jnp.sum(
            foundation_v2.hard_bel_from_noise(
                states + shift, times, noise, **kwargs
            ).single_query_labels
        )
    )(jnp.asarray(0.0, dtype=jnp.float64))
    np.testing.assert_array_equal(reward_gradient, 0.0)
    np.testing.assert_array_equal(label_gradient, 0.0)


def test_query_matched_arm_algebra_replay_and_training_data_has_no_truth() -> None:
    raw = _sample("RAW1", 20, seed=11)
    raw_replay = _sample("RAW1", 20, seed=11)
    iid = _sample("IID2", 20, seed=12)
    antithetic = _sample("ANTITHETIC2", 20, seed=13)
    for field in fields(foundation_v2.QueryMatchedDataset):
        first = getattr(raw, field.name)
        replay = getattr(raw_replay, field.name)
        if isinstance(first, str):
            assert first == replay
        elif isinstance(first, int):
            assert first == replay
        else:
            np.testing.assert_array_equal(first, replay)

    assert raw.context_count == 20
    assert iid.context_count == antithetic.context_count == 10
    assert raw.reward_query_count == iid.reward_query_count == antithetic.reward_query_count == 20
    np.testing.assert_array_equal(
        antithetic.suffix_normals[:, 1, :], -antithetic.suffix_normals[:, 0, :]
    )
    for dataset in (raw, iid, antithetic):
        explicit = foundation_v2.hard_bel_from_noise(
            dataset.states,
            dataset.times,
            dataset.suffix_normals,
            sigma=0.7,
            horizon=1.0,
            threshold=0.4,
        )
        np.testing.assert_allclose(
            dataset.supervised_targets,
            jnp.mean(explicit.single_query_labels, axis=1),
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    training_fields = {field.name for field in fields(foundation_v2.QueryMatchedDataset)}
    assert "analytic_costates" not in training_fields
    assert "analytic_values" not in training_fields


@pytest.mark.parametrize("arm", ["IID2", "ANTITHETIC2"])
def test_two_query_arms_are_unbiased_against_analytic_costate(arm: str) -> None:
    dataset = _sample(arm, 100_000, seed=100 if arm == "IID2" else 101)
    truth = foundation_v2.hard_threshold_costate(
        dataset.states[:, 0],
        dataset.times,
        sigma=0.7,
        horizon=1.0,
        threshold=0.4,
    )[:, None]
    residual = np.asarray(dataset.supervised_targets - truth).reshape(-1)
    standard_error = residual.std(ddof=1) / np.sqrt(residual.size)
    assert abs(residual.mean()) <= 4.0 * standard_error
    assert np.all(np.isfinite(residual))


class _AffineLogitFactory:
    def forward(self, params, states, times):
        return params["slope"] * states + params["time"] * times[:, None] + params["bias"]


def test_critic_costate_is_sigmoid_logit_gradient_and_eager_jit_consistent() -> None:
    factory = _AffineLogitFactory()
    params = {
        "slope": jnp.asarray(2.0, dtype=jnp.float64),
        "time": jnp.asarray(-0.4, dtype=jnp.float64),
        "bias": jnp.asarray(0.1, dtype=jnp.float64),
    }
    states = jnp.asarray([[-1.0], [0.0], [0.7]], dtype=jnp.float64)
    times = jnp.asarray([0.0, 0.25, 0.5], dtype=jnp.float64)
    logits = factory.forward(params, states, times)
    probabilities = jax.nn.sigmoid(logits)
    expected = params["slope"] * probabilities * (1.0 - probabilities)
    eager = foundation_v2.critic_costate(factory, params, states, times)
    compiled = jax.jit(
        lambda value_params, x, t: foundation_v2.critic_costate(factory, value_params, x, t)
    )(params, states, times)
    np.testing.assert_allclose(eager, expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(compiled, expected, rtol=1.0e-13, atol=1.0e-13)

    state = jnp.asarray(0.2, dtype=jnp.float64)
    time = jnp.asarray(0.25, dtype=jnp.float64)
    autodiff_truth = jax.grad(
        lambda value: foundation_v2.hard_threshold_value(
            value,
            time,
            sigma=0.7,
            horizon=1.0,
            threshold=0.4,
        )
    )(state)
    analytic_truth = foundation_v2.hard_threshold_costate(
        state,
        time,
        sigma=0.7,
        horizon=1.0,
        threshold=0.4,
    )
    np.testing.assert_allclose(autodiff_truth, analytic_truth, rtol=1.0e-13, atol=1.0e-13)


def test_frozen_dev_key_domains_and_replicates_have_no_test_stream() -> None:
    assert foundation_v2.DEV_STREAM_IDS == {
        "training_contexts": 40_001,
        "base_noise": 40_002,
        "independent_second_noise": 40_003,
        "network_initialization": 40_004,
        "minibatch_schedule": 40_005,
        "validation_contexts": 40_006,
    }
    assert 42_001 not in foundation_v2.DEV_STREAM_IDS.values()
    assert all("final" not in name.lower() for name in foundation_v2.DEV_STREAM_IDS)
    assert all("final" not in name.lower() for name in foundation_v2.__all__)
    config = _config(Path("unused"))
    foundation_v2._validate_config(config)
    config["experiment"]["dev_replicates"] = [0, 1]
    with pytest.raises(ValueError, match="exactly"):
        foundation_v2._validate_config(config)


def test_tiny_dev_run_has_equal_q_shared_rewards_and_hashed_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "dev"
    config = _config(run_dir)
    result = foundation_v2.run_dev(config)
    assert result["status"] == foundation_v2.SMOKE_STATUS
    assert result["eligible_for_scientific_status"] is False
    assert result["eligible_to_lock_for_confirm"] is False
    assert result["dev_contract"]["matches_frozen_dev_contract"] is False
    accounting = result["query_accounting"]
    assert accounting["configured_reward_query_budget_Q_per_arm_per_replicate"] == 16
    assert accounting["all_arm_replicates_have_exactly_Q_logical_queries"] is True
    assert accounting["total_logical_reward_queries"] == 4 * 3 * 16
    assert accounting["physical_unique_reward_values_generated"] == 3 * 3 * 16
    assert accounting["value_critic_reuses_raw1_rewards"] is True
    assert accounting["dev_validation_reward_queries"] == 0

    budget = result["matched_budget"]
    assert budget["common_initialization_across_arms_within_replicate"] is True
    assert all(budget["raw1_and_value_critic_share_dataset_by_replicate"].values())
    assert budget["dev_used_for_training_loss"] is False
    for arm in foundation_v2.ALL_ARMS:
        selected = result["selection"][arm]
        assert selected["selection_rule"].startswith("minimum_mean_analytic_dev")
        assert set(selected["replicates"]) == {"0", "1", "2"}
        assert len(result["candidates"][arm]) == 1
        candidate = result["candidates"][arm][0]
        expected_mean = np.mean(
            [
                candidate["replicates"][str(rep)]["dev_costate_metrics"]["mean_squared_error"]
                for rep in (0, 1, 2)
            ]
        )
        np.testing.assert_allclose(
            candidate["mean_dev_costate_metrics"]["mean_squared_error"],
            expected_mean,
            rtol=0.0,
            atol=1.0e-15,
        )

    artifacts = {
        "resolved_config.json",
        "dev_results.json",
        "dev_run_manifest.json",
        "dev_training_metrics.csv",
        "dev_raw_samples.npz",
        "dev_checkpoint.npz",
    }
    assert artifacts.issubset({path.name for path in run_dir.iterdir()})
    manifest = json.loads((run_dir / "dev_run_manifest.json").read_text())
    assert manifest["status"] == foundation_v2.SMOKE_STATUS
    assert manifest["eligible_to_lock_for_confirm"] is False
    assert manifest["random_streams"]["domain_ids"] == foundation_v2.DEV_STREAM_IDS
    assert "docs/mam_gate_a_v2_protocol.md" in manifest["source_sha256"]
    assert "schrodinger_bridge/network_factory.py" in manifest["source_sha256"]
    assert "schrodinger_bridge/networks.py" in manifest["source_sha256"]
    assert "scripts/run_mam_foundation_v2_dev.py" in manifest["source_sha256"]
    for name, digest in manifest["artifact_sha256"].items():
        assert digest == _sha256(run_dir / name)

    with (run_dir / "dev_training_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "truth" not in " ".join(rows[0]).lower()
    restored = foundation_v2.load_dev_checkpoint(run_dir / "dev_checkpoint.npz", config)
    assert set(restored) == set(foundation_v2.ALL_ARMS)
    assert all(set(restored[arm]) == {0, 1, 2} for arm in foundation_v2.ALL_ARMS)
    with pytest.raises(ValueError, match="not eligible"):
        foundation_v2.load_dev_checkpoint(
            run_dir / "dev_checkpoint.npz",
            config,
            require_lockable=True,
        )

    wrong_config = json.loads(json.dumps(config))
    wrong_config["experiment"]["seed"] += 1
    with pytest.raises(ValueError, match="does not match"):
        foundation_v2.load_dev_checkpoint(run_dir / "dev_checkpoint.npz", wrong_config)

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        foundation_v2.run_dev(config)


def test_only_exact_frozen_dev_config_can_become_lockable() -> None:
    root = Path(__file__).resolve().parents[2]
    frozen = foundation_v2.load_config(
        root
        / "experiments"
        / "malliavin_adjoint_matching"
        / "configs"
        / "brownian_threshold_v2_dev.yaml"
    )
    assert foundation_v2._dev_contract(frozen)["matches_frozen_dev_contract"] is True
    changed = json.loads(json.dumps(frozen))
    changed["sampling"]["reward_query_budget"] //= 2
    comparison = foundation_v2._dev_contract(changed)
    assert comparison["matches_frozen_dev_contract"] is False
    assert any("reward_query_budget" in reason for reason in comparison["mismatches"])
