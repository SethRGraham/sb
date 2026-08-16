"""Falsification and artifact checks for the conditional MAM Gate-1 harness."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.mam_gate1 import conditional_benchmark as gate1


def _tiny_config(output_dir: Path) -> dict:
    config = gate1.load_config(
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "mam_gate1"
        / "configs"
        / "conditional_smoke.yaml"
    )
    config["experiment"]["output_dir"] = str(output_dir)
    config["evaluation"]["query_count"] = 2
    config["evaluation"]["objective_sample_size"] = 2
    config["evaluation"]["mam_antithetic_pairs"] = 1
    config["evaluation"]["direct_antithetic_pairs"] = 1
    config["evaluation"]["path_integral_pairs"] = 1
    config["evaluation"]["reference_path_integral_pairs"] = 2
    config["evaluation"]["path_integral_shard_size"] = 1
    return config


def test_discrete_lqg_reference_matches_one_step_quadratic_minimizer() -> None:
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    times = jnp.linspace(0.0, 1.0, 4, dtype=dtype)
    sigma = jnp.asarray([[0.7, 0.2], [-0.1, 0.9]], dtype=dtype)
    center = jnp.asarray([0.25, -0.4], dtype=dtype)
    policy = gate1.solve_discrete_lqg_policy(times, sigma, 1.7, center)
    state = jnp.asarray([[0.2, -0.7], [-0.4, 0.5]], dtype=dtype)
    endpoint = jnp.asarray([[1.0, 0.1], [0.8, -0.2]], dtype=dtype)
    final_anchor = jnp.full((2,), 1, dtype=jnp.int32)
    actual = policy.evaluate(state, endpoint, final_anchor)

    dt = times[1] - times[0]
    terminal = times[-1]
    rho = (terminal - times[2]) / (terminal - times[1])
    mean = rho * state + (1.0 - rho) * endpoint
    control_map = dt * jnp.sqrt(rho) * sigma
    q = dt * 1.7 * jnp.eye(2, dtype=dtype)
    r = -dt * 1.7 * center
    expected = -jax.vmap(
        lambda mu: jnp.linalg.solve(
            dt * jnp.eye(2, dtype=dtype) + control_map.T @ q @ control_map,
            control_map.T @ (q @ mu + r),
        )
    )(mean)
    tolerance = 2e-5 if dtype == jnp.float32 else 1e-11
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_grouped_direct_score_has_correct_final_arrival_sign_and_scale(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path / "unused")
    problem = gate1._make_problem(config)
    cost = gate1._make_cost(config, "smooth_lqg")
    times = jnp.asarray(problem.time_grid.times, dtype=jnp.float32)
    anchor = problem.time_grid.num_steps - 2
    states = jnp.asarray([[0.2, -0.4], [-0.7, 0.3]], dtype=jnp.float32)
    endpoints = jnp.asarray([[1.1, 0.1], [0.8, -0.2]], dtype=jnp.float32)
    anchors = jnp.full((states.shape[0],), anchor, dtype=jnp.int32)
    queries = gate1.EvaluationQueries(
        states=states,
        times=times[anchors],
        next_times=times[anchors + 1],
        endpoints=endpoints,
        anchors=anchors,
        dataset_sha256="analytic-test",
    )
    estimate, finite, _ = gate1._direct_full_return_target(
        jax.random.PRNGKey(91),
        queries,
        problem,
        cost,
        gate1._zero_control,
        100_000,
    )
    rho = (times[-1] - times[anchor + 1]) / (times[-1] - times[anchor])
    mean = rho * states + (1.0 - rho) * endpoints
    smooth = config["tasks"]["smooth_lqg"]
    center = jnp.asarray(smooth["center"], dtype=states.dtype)
    expected = (
        -problem.time_grid.dt
        * jnp.sqrt(rho)
        * config["problem"]["sigma"]
        * smooth["quadratic_weight"]
        * (mean - center)
    )
    assert np.all(np.asarray(finite))
    np.testing.assert_allclose(estimate, expected, rtol=0.02, atol=0.006)


def test_full_contract_has_five_locked_seeds_and_requested_thresholds() -> None:
    root = Path(__file__).resolve().parents[2]
    config = gate1.load_config(
        root / "experiments" / "mam_gate1" / "configs" / "conditional_full.yaml"
    )
    assert tuple(config["experiment"]["seeds"]) == gate1.FULL_GATE1_SEEDS
    assert config["gates"]["smooth_relative_l2"] == 0.05
    assert config["gates"]["smooth_cosine"] == 0.995
    assert config["gates"]["hard_boundary_relative_l2"] == 0.10
    assert config["gates"]["hard_boundary_cosine"] == 0.99
    assert config["gates"]["minimum_accepting_seeds"] == 4
    assert config["conditional"]["improvement_tolerance"] > 0.0
    assert config["gates"]["maximum_target_reference_relative_se"] == 0.02
    assert config["gates"]["maximum_path_integral_reference_relative_se"] == 0.05
    assert config["gates"]["maximum_target_reference_split_relative_l2"] == 0.08
    assert config["gates"]["maximum_path_integral_reference_split_relative_l2"] == 0.20
    assert gate1.full_contract_report(config)["matches_locked_full_contract"]

    modified = copy.deepcopy(config)
    modified["evaluation"]["reference_path_integral_pairs"] //= 2
    assert not gate1.full_contract_report(modified)["matches_locked_full_contract"]


def test_local_cpu_profile_is_non_scientific_and_preserves_locked_gates() -> None:
    root = Path(__file__).resolve().parents[2]
    local = gate1.load_config(
        root / "experiments" / "mam_gate1" / "configs" / "conditional_local_cpu.yaml"
    )
    full = gate1.load_config(
        root / "experiments" / "mam_gate1" / "configs" / "conditional_full.yaml"
    )

    assert local["experiment"]["profile"] == "smoke"
    assert local["experiment"]["intended_for_scientific_evidence"] is False
    assert local["experiment"]["seeds"] == [20260816]
    assert local["experiment"]["warmup"] is False
    assert not set(local["experiment"]["seeds"]) & set(gate1.FULL_GATE1_SEEDS)
    assert not gate1.full_contract_report(local)["matches_locked_full_contract"]
    assert local["problem"]["steps"] == 8
    assert local["conditional"]["actor_model"] == "nonlinear"
    assert local["conditional"]["policy_iterations"] == 2
    assert local["conditional"]["improvement_tolerance"] > 0.0
    assert (
        local["conditional"]["effective_batch_size"] % local["conditional"]["microbatch_size"] == 0
    )
    stochastic_steps = local["problem"]["steps"] - 1
    assert local["evaluation"]["query_count"] == 56
    assert local["evaluation"]["query_count"] >= stochastic_steps
    assert local["evaluation"]["query_count"] % stochastic_steps == 0
    assert local["evaluation"]["reference_direct_replicates"] % 2 == 0
    assert local["evaluation"]["reference_direct_antithetic_pairs"] == 128
    assert local["evaluation"]["reference_direct_replicates"] == 4
    assert local["evaluation"]["reference_path_integral_pairs"] == 256
    assert local["evaluation"]["reference_replicates"] % 2 == 0
    assert (
        local["evaluation"]["minimum_boundary_effective_queries"]
        <= local["evaluation"]["query_count"]
    )
    assert local["gates"] == full["gates"]
    assert local["tasks"] == full["tasks"]


def test_config_fails_closed_on_unknown_fields_and_fake_full_seed_list(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path / "out")
    config["evaluation"]["unreported_smoothing"] = 0.1
    with pytest.raises(ValueError, match="unexpected"):
        gate1.validate_config(config)

    config = _tiny_config(tmp_path / "out")
    config["experiment"]["profile"] = "full"
    config["experiment"]["intended_for_scientific_evidence"] = True
    with pytest.raises(ValueError, match="seeds"):
        gate1.validate_config(config)


@pytest.mark.slow
def test_actual_smoke_is_deterministic_and_writes_closed_schema(tmp_path: Path) -> None:
    first = gate1.run_benchmark(_tiny_config(tmp_path / "first"))
    second = gate1.run_benchmark(_tiny_config(tmp_path / "second"))
    assert first["status"] == second["status"] == gate1.SMOKE_STATUS
    assert first["scientific_evidence"] is False
    assert (
        first["manifest"]["scientific_payload_sha256"]
        == second["manifest"]["scientific_payload_sha256"]
    )
    for result in (first, second):
        output = Path(result["output_dir"])
        assert {path.name for path in output.iterdir()} == {
            "resolved_config.json",
            "results.json",
            "run_manifest.json",
        }
        payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
        assert set(payload["tasks"]) == {"smooth_lqg", "hard_disk"}
        for task in payload["tasks"].values():
            record = task["seeds"][0]
            assert set(record["arms"]) == set(gate1.REQUIRED_ARMS)
            assert all(arm["available"] for arm in record["arms"].values())
            assert record["evaluation_dataset_sha256"]
            assert "untouched_final_vs_zero" in record["acceptance"]
            assert record["references"]["current_policy_action_target"]["estimand"] == (
                "current_policy_action_target"
            )
            assert (
                "weighted_relative_standard_error"
                in record["references"]["current_policy_action_target"]
            )
            assert "split_half_relative_l2" in record["references"]["current_policy_action_target"]
            assert record["arms"]["path_integral"]["estimand"] == (
                "kl_relaxed_path_measure_desirability_control"
            )
            assert record["references"]["path_integral_desirability_control"][
                "not_a_discrete_mean_shift_optimality_reference"
            ]
        assert payload["tasks"]["smooth_lqg"]["seeds"][0]["references"]["discrete_optimal_control"][
            "available"
        ]
        assert not payload["tasks"]["hard_disk"]["seeds"][0]["references"][
            "discrete_optimal_control"
        ]["available"]
        assert payload["gate_summary"]["comparison_coverage"]["coverage_pass"] is False
        assert set(
            payload["gate_summary"]["comparison_coverage"]["unavailable_planned_arms"]
        ) == set(gate1.UNAVAILABLE_PLANNED_ARMS)
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["devices"]
        assert set(manifest["git"]) == {"start", "end", "clean_stable_revision"}
        assert set(manifest["rng_stream_ids"]) == set(gate1.STREAM_IDS)
        assert manifest["dependencies"]["jax"]
        assert manifest["timing"]["total_process_seconds"] >= 0.0
