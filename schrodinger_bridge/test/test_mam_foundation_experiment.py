"""Focused checks for the Gate-A MAM foundation experiment."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.malliavin_adjoint_matching import foundation

jax.config.update("jax_enable_x64", True)


def _config(output_dir: Path) -> dict:
    return {
        "experiment": {
            "seed": 7,
            "output_dir": str(output_dir),
            "intended_for_scientific_evidence": False,
        },
        "numerics": {"compute_dtype": "float64", "matmul_precision": "highest"},
        "dynamics": {
            "family": "brownian_threshold",
            "state_dim": 1,
            "control_dim": 1,
            "horizon": 1.0,
            "steps": 8,
            "sigma": 0.7,
            "threshold": 0.4,
        },
        "anchors": {
            "samples_per_trajectory": 1,
            "minimum_remaining_steps": 2,
            "distribution": "uniform_discrete",
        },
        "labels": {
            "method": "bel_uniform",
            "include_terminal": True,
            "include_running": False,
            "stop_reward_gradient": True,
            "theorem_facing_clipping": False,
        },
        "costate_network": {
            "hidden_sizes": [8, 8],
            "activation": "silu",
            "time_embedding_dim": 4,
            "optimizer": "adam",
            "learning_rates": [0.001, 0.003],
            "training_steps": 4,
            "batch_size": 4,
            "eval_every": 2,
        },
        "sampling": {
            "calibration_samples": 256,
            "training_trajectories": 16,
            "validation_trajectories": 8,
            "analytic_evaluation_points": 4,
            "calibration_state": 0.1,
            "calibration_anchor_time": 0.25,
            "finite_difference_epsilon": 1.0e-5,
            "smoothing_temperatures": [0.2, 0.05],
            "running_calibration_steps": 2,
            "state_range": [-0.5, 1.0],
            "evaluation_time_points": 2,
            "evaluation_state_points": 2,
        },
        "acceptance": {
            "maximum_relative_l2": 1.0e9,
            "minimum_cosine": -1.0,
            "minimum_sign_agreement": 0.0,
            "mean_z_tolerance": 1.0e9,
            "minimum_finite_fraction": 1.0,
            "maximum_calibration_relative_error": 1.0e9,
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hard_threshold_truth_and_labels_are_analytic_and_replayable() -> None:
    sigma = 0.7
    horizon = 1.0
    threshold = 0.4
    state = jnp.asarray(threshold, dtype=jnp.float64)
    time_value = jnp.asarray(0.25, dtype=jnp.float64)
    expected = 1.0 / (sigma * np.sqrt(horizon - 0.25) * np.sqrt(2.0 * np.pi))
    actual = foundation.hard_threshold_truth(
        state,
        time_value,
        sigma=sigma,
        horizon=horizon,
        threshold=threshold,
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)
    assert bool(
        jnp.isnan(
            foundation.hard_threshold_truth(
                state,
                jnp.asarray(horizon),
                sigma=sigma,
                horizon=horizon,
                threshold=threshold,
            )
        )
    )

    kwargs = {
        "sigma": sigma,
        "horizon": horizon,
        "threshold": threshold,
        "time_steps": 8,
        "minimum_remaining_steps": 2,
        "state_minimum": -0.5,
        "state_maximum": 1.0,
        "dtype": jnp.float64,
    }
    first = foundation.sample_hard_threshold_labels(
        jax.random.PRNGKey(11),
        50_000,
        **kwargs,
    )
    replay = foundation.sample_hard_threshold_labels(
        jax.random.PRNGKey(11),
        50_000,
        **kwargs,
    )
    for observed, repeated in zip(first, replay, strict=True):
        np.testing.assert_array_equal(observed, repeated)
    scaled_times = np.asarray(first[1]) * 8 / horizon
    np.testing.assert_allclose(scaled_times, np.round(scaled_times), atol=1.0e-14)
    assert float(first[1].max()) <= 0.75
    labels = np.asarray(first[2][:, 0])
    truths = np.asarray(first[3][:, 0])
    differences = labels - truths
    standard_error = differences.std(ddof=1) / np.sqrt(differences.size)
    assert abs(differences.mean()) <= 4.0 * standard_error
    stopped_gradient = jax.grad(
        lambda shift: jnp.sum(
            foundation._stopped_hard_reward(
                jnp.asarray([0.3, 0.5]) + shift,
                threshold=threshold,
                dtype=jnp.float64,
            )
        )
    )(jnp.asarray(0.0))
    np.testing.assert_array_equal(stopped_gradient, 0.0)


def test_minibatch_schedule_is_common_random_and_covers_declared_dataset() -> None:
    kwargs = {"dataset_size": 13, "training_steps": 4, "batch_size": 4}
    first = foundation._minibatch_schedule(jax.random.PRNGKey(3), **kwargs)
    replay = foundation._minibatch_schedule(jax.random.PRNGKey(3), **kwargs)
    different = foundation._minibatch_schedule(jax.random.PRNGKey(4), **kwargs)
    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, different)
    np.testing.assert_array_equal(np.sort(np.asarray(first).reshape(-1)[:13]), np.arange(13))
    with pytest.raises(ValueError, match="expose every"):
        foundation._minibatch_schedule(
            jax.random.PRNGKey(0),
            dataset_size=17,
            training_steps=4,
            batch_size=4,
        )


def test_checkpoint_round_trip_validates_tree_shape_dtype_and_finiteness(
    tmp_path: Path,
) -> None:
    params = {
        "a": jnp.arange(6, dtype=jnp.float64).reshape(2, 3),
        "b": [jnp.asarray([1.5], dtype=jnp.float64)],
    }
    path = tmp_path / "checkpoint.npz"
    foundation._save_checkpoint(path, params, {"solver_status": "test"})
    restored = foundation.load_checkpoint_leaves(path, params)
    for expected, actual in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(restored),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

    wrong_dtype = jax.tree_util.tree_map(lambda value: value.astype(jnp.float32), params)
    with pytest.raises(ValueError, match="dtype"):
        foundation.load_checkpoint_leaves(path, wrong_dtype)
    nonfinite = deepcopy(params)
    nonfinite["a"] = nonfinite["a"].at[0, 0].set(jnp.nan)
    with pytest.raises(FloatingPointError, match="finite"):
        foundation._save_checkpoint(tmp_path / "bad.npz", nonfinite, {})


def test_foundation_run_keeps_tuning_disjoint_and_writes_hashed_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "foundation"
    config = _config(run_dir)
    results = foundation.run_foundation(config)

    assert results["status"] == "PASS_MAM_ANALYTIC_FOUNDATION_SMOKE_NOT_EVIDENCE"
    assert results["smooth_terminal"]["status"] == "PASS_CALIBRATION"
    assert results["smooth_terminal"]["implementation_equivalence"]["pass"] is True
    running = results["running_cost"]
    assert running["analytic_discrete_truth"] == {
        "terminal_component": 0.2,
        "running_component": 0.2,
        "sum": 0.4,
    }
    assert running["terminal_component"]["variance"] > 0.0
    regression = results["hard_threshold_costate_regression"]
    assert regression["heldout_used_for_selection"] is False
    assert regression["selection_rule"].startswith("minimum_disjoint_validation")
    assert regression["selected"]["evaluation_points"] == 4
    assert all("selection_relative_l2" in candidate for candidate in regression["candidates"])
    assert all("relative_l2" not in candidate for candidate in regression["candidates"])
    budget = regression["data_budget"]
    assert budget["unique_training_reward_queries"] == 16
    assert budget["unique_validation_reward_queries"] == 8
    assert budget["label_exposures_per_candidate"] == 16
    assert budget["common_training_dataset_across_candidates"] is True

    expected_artifacts = {
        "resolved_config.json",
        "results.json",
        "run_manifest.json",
        "training_metrics.csv",
        "raw_samples.npz",
        "checkpoint.npz",
    }
    assert expected_artifacts.issubset({path.name for path in run_dir.iterdir()})
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["heldout_used_for_selection"] is False
    assert manifest["environment"]["jaxlib"]
    assert manifest["profile"] == "smoke_not_evidence"
    assert manifest["reward_oracle_identifiers"]["hard_terminal"].endswith(".v1")
    for name, digest in manifest["artifact_sha256"].items():
        assert digest == _sha256(run_dir / name)
    assert (
        run_dir / "source_snapshot" / "experiments" / "malliavin_adjoint_matching" / "foundation.py"
    ).is_file()
    json.loads(
        (run_dir / "results.json").read_text(), parse_constant=lambda value: pytest.fail(value)
    )
    with np.load(run_dir / "raw_samples.npz", allow_pickle=False) as samples:
        assert samples["training_label"].shape == (16,)
        assert samples["validation_label"].shape == (8,)
        assert samples["evaluation_prediction"].shape == (4,)

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        foundation.run_foundation(config)


def test_invalid_evaluation_budget_is_rejected_before_writing(tmp_path: Path) -> None:
    config = _config(tmp_path / "invalid")
    config["sampling"]["analytic_evaluation_points"] = 5
    with pytest.raises(ValueError, match="must equal"):
        foundation.run_foundation(config)
    assert not Path(config["experiment"]["output_dir"]).exists()


def test_scientific_status_requires_the_exact_frozen_contract(tmp_path: Path) -> None:
    reduced = _config(tmp_path / "reduced")
    reduced["experiment"]["intended_for_scientific_evidence"] = True
    reduced_contract = foundation._scientific_contract(reduced)
    assert reduced_contract["requested"] is True
    assert reduced_contract["eligible_for_scientific_status"] is False
    assert reduced_contract["mismatches"]

    frozen = deepcopy(foundation._FROZEN_PRODUCTION_CONTRACT)
    frozen["experiment"]["output_dir"] = str(tmp_path / "production")
    foundation._validate_config(frozen)
    frozen_contract = foundation._scientific_contract(frozen)
    assert frozen_contract["eligible_for_scientific_status"] is True
    assert frozen_contract["mismatches"] == []


def test_overwrite_replaces_stale_pass_and_exception_writes_failure_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "failed"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text('{"status":"PASS_STALE"}\n')
    observed_status: list[str] = []

    def fail_after_running_manifest(*args, **kwargs):
        del args, kwargs
        observed_status.append(json.loads((run_dir / "run_manifest.json").read_text())["status"])
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(foundation, "_execute_foundation", fail_after_running_manifest)
    with pytest.raises(RuntimeError, match="deliberate"):
        foundation.run_foundation(_config(run_dir), overwrite=True)
    assert observed_status == ["RUNNING_INCOMPLETE"]
    failure = json.loads((run_dir / "run_manifest.json").read_text())
    assert failure["status"] == "FAILED_EXCEPTION"
    assert failure["failed_gate"] == "runtime_exception"
    assert failure["exception"]["type"] == "RuntimeError"
    assert "deliberate test failure" in failure["exception"]["message"]
