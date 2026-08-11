"""Gate-A analytic foundation for Malliavin Adjoint Matching.

This experiment is deliberately independent of the future generalized-bridge
outer loop.  It tests three fixed-policy identities and then asks whether an
ordinary MSE regressor can recover the conditional hard-threshold costate from
uncensored value-only BEL labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import yaml

from schrodinger_bridge.network_factory import MLPFactory
from schrodinger_bridge.networks import adam_update, init_adam
from schrodinger_bridge.solvers.malliavin_adjoint import (
    assemble_bel_costate_labels,
    simulate_additive_em_rollout,
)

_STREAM_IDS = {
    "smooth_calibration": 10_001,
    "hard_calibration": 10_002,
    "running_calibration": 10_003,
    "network_initialization": 20_001,
    "training_dataset": 20_002,
    "validation_dataset": 20_003,
    "minibatch_schedule": 20_004,
}

_FROZEN_PRODUCTION_CONTRACT = {
    "experiment": {
        "seed": 0,
        "intended_for_scientific_evidence": True,
    },
    "numerics": {"compute_dtype": "float64", "matmul_precision": "highest"},
    "dynamics": {
        "family": "brownian_threshold",
        "state_dim": 1,
        "control_dim": 1,
        "horizon": 1.0,
        "steps": 64,
        "sigma": 0.7,
        "threshold": 0.8,
    },
    "anchors": {
        "samples_per_trajectory": 1,
        "minimum_remaining_steps": 8,
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
        "hidden_sizes": [64, 64],
        "activation": "silu",
        "time_embedding_dim": 16,
        "optimizer": "adam",
        "learning_rates": [0.0001, 0.0003, 0.001],
        "training_steps": 5000,
        "batch_size": 1024,
        "eval_every": 250,
    },
    "sampling": {
        "training_trajectories": 131072,
        "validation_trajectories": 32768,
        "analytic_evaluation_points": 2048,
        "calibration_samples": 131072,
        "calibration_state": 0.2,
        "calibration_anchor_time": 0.25,
        "finite_difference_epsilon": 0.0001,
        "smoothing_temperatures": [0.2, 0.1, 0.05, 0.025],
        "running_calibration_steps": 64,
        "state_range": [-1.0, 2.0],
        "evaluation_time_points": 32,
        "evaluation_state_points": 64,
    },
    "acceptance": {
        "maximum_relative_l2": 0.10,
        "minimum_cosine": 0.99,
        "minimum_sign_agreement": 0.99,
        "mean_z_tolerance": 3.0,
        "maximum_calibration_relative_error": 0.02,
        "minimum_finite_fraction": 1.0,
    },
}

_REWARD_ORACLE_IDENTIFIERS = {
    "smooth_terminal": "mam.gate_a.terminal.sin.v1",
    "hard_terminal": "mam.gate_a.terminal.indicator_x_ge_threshold.v1",
    "running": "mam.gate_a.running.right_endpoint_x_squared.v1",
    "running_terminal": "mam.gate_a.terminal.x_squared.v1",
    "costate_training": "mam.gate_a.terminal.indicator_x_ge_threshold.v1",
}


def _stream_key(master: jax.Array, name: str) -> jax.Array:
    """Derive a named, stable random stream without relying on split order."""
    return jax.random.fold_in(master, _STREAM_IDS[name])


def _require_finite(name: str, value: Any) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _scientific_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Compare all scientific semantics with the frozen production profile."""
    observed = json.loads(json.dumps(config))
    observed["experiment"].pop("output_dir", None)
    mismatches: list[str] = []

    def compare(expected: Any, actual: Any, path: str) -> None:
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key in sorted(expected.keys() - actual.keys()):
                mismatches.append(f"{path}.{key}:missing")
            for key in sorted(actual.keys() - expected.keys()):
                mismatches.append(f"{path}.{key}:unexpected")
            for key in sorted(expected.keys() & actual.keys()):
                compare(expected[key], actual[key], f"{path}.{key}")
            return
        if type(expected) is not type(actual) or expected != actual:
            mismatches.append(f"{path}:expected={expected!r},actual={actual!r}")

    compare(_FROZEN_PRODUCTION_CONTRACT, observed, "config")
    requested = bool(config["experiment"]["intended_for_scientific_evidence"])
    return {
        "requested": requested,
        "matches_frozen_production_contract": not mismatches,
        "eligible_for_scientific_status": requested and not mismatches,
        "operational_fields_excluded": ["experiment.output_dir"],
        "mismatches": mismatches,
        "contract_sha256": hashlib.sha256(
            json.dumps(
                _FROZEN_PRODUCTION_CONTRACT,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }


def _validate_config(config: dict[str, Any]) -> None:
    """Validate the complete Gate-A contract before creating artifacts."""
    for section in (
        "experiment",
        "numerics",
        "dynamics",
        "anchors",
        "labels",
        "costate_network",
        "sampling",
        "acceptance",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing mapping config section: {section}")

    experiment = config["experiment"]
    if isinstance(experiment.get("seed"), bool) or not isinstance(experiment.get("seed"), int):
        raise ValueError("experiment.seed must be an integer")
    if not str(experiment.get("output_dir", "")).strip():
        raise ValueError("experiment.output_dir must be nonempty")
    if not isinstance(experiment.get("intended_for_scientific_evidence"), bool):
        raise ValueError("experiment.intended_for_scientific_evidence must be boolean")

    numerics = config["numerics"]
    if str(numerics.get("compute_dtype", "float64")) not in {"float32", "float64"}:
        raise ValueError("numerics.compute_dtype must be float32 or float64")
    if str(numerics.get("matmul_precision", "highest")) not in {
        "default",
        "high",
        "highest",
        "bfloat16",
        "tensorfloat32",
        "float32",
    }:
        raise ValueError("numerics.matmul_precision is not recognized by JAX")

    dynamics = config["dynamics"]
    if int(dynamics.get("state_dim", 1)) != 1 or int(dynamics.get("control_dim", 1)) != 1:
        raise ValueError("Gate A currently requires one state and one control dimension")
    if str(dynamics.get("family", "brownian_threshold")) != "brownian_threshold":
        raise ValueError("Gate A currently supports only dynamics.family=brownian_threshold")
    horizon = _require_finite("dynamics.horizon", dynamics["horizon"])
    sigma = _require_finite("dynamics.sigma", dynamics["sigma"])
    _require_finite("dynamics.threshold", dynamics["threshold"])
    steps = int(dynamics["steps"])
    if horizon <= 0.0 or sigma <= 0.0 or steps < 1:
        raise ValueError("horizon, sigma, and steps must be positive")

    anchors = config["anchors"]
    minimum_remaining_steps = int(anchors["minimum_remaining_steps"])
    if not 1 <= minimum_remaining_steps <= steps:
        raise ValueError("anchors.minimum_remaining_steps must lie in [1, dynamics.steps]")
    if anchors.get("samples_per_trajectory") != 1:
        raise ValueError("Gate A requires anchors.samples_per_trajectory=1")
    if anchors.get("distribution") != "uniform_discrete":
        raise ValueError("Gate A requires anchors.distribution=uniform_discrete")

    labels = config["labels"]
    required_label_contract = {
        "method": "bel_uniform",
        "include_terminal": True,
        "include_running": False,
        "stop_reward_gradient": True,
        "theorem_facing_clipping": False,
    }
    for name, expected in required_label_contract.items():
        if labels.get(name) != expected:
            raise ValueError(f"Gate A requires labels.{name}={expected!r}")

    network = config["costate_network"]
    hidden_sizes = tuple(int(width) for width in network["hidden_sizes"])
    if not hidden_sizes or any(width < 1 for width in hidden_sizes):
        raise ValueError("costate_network.hidden_sizes must contain positive widths")
    if int(network["time_embedding_dim"]) < 2:
        raise ValueError("costate_network.time_embedding_dim must be at least two")
    if network.get("optimizer") != "adam":
        raise ValueError("Gate A currently requires costate_network.optimizer=adam")
    learning_rates = [_require_finite("learning rate", rate) for rate in network["learning_rates"]]
    if not learning_rates or any(rate <= 0.0 for rate in learning_rates):
        raise ValueError("costate_network.learning_rates must contain positive values")
    training_steps = int(network["training_steps"])
    batch_size = int(network["batch_size"])
    eval_every = int(network.get("eval_every", max(1, training_steps // 10)))
    if training_steps < 1 or batch_size < 1 or eval_every < 1:
        raise ValueError("training_steps, batch_size, and eval_every must be positive")

    sampling = config["sampling"]
    calibration_samples = int(sampling["calibration_samples"])
    training_trajectories = int(sampling["training_trajectories"])
    validation_trajectories = int(sampling["validation_trajectories"])
    if min(calibration_samples, training_trajectories, validation_trajectories) < 2:
        raise ValueError("calibration, training, and validation counts must be at least two")
    if training_steps * batch_size < training_trajectories:
        raise ValueError(
            "training_steps * batch_size must expose every declared training trajectory"
        )
    calibration_anchor = _require_finite(
        "sampling.calibration_anchor_time", sampling["calibration_anchor_time"]
    )
    if not 0.0 <= calibration_anchor < horizon:
        raise ValueError("sampling.calibration_anchor_time must lie in [0, horizon)")
    _require_finite("sampling.calibration_state", sampling["calibration_state"])
    epsilon = _require_finite(
        "sampling.finite_difference_epsilon", sampling["finite_difference_epsilon"]
    )
    if epsilon <= 0.0:
        raise ValueError("sampling.finite_difference_epsilon must be positive")
    temperatures = [
        _require_finite("smoothing temperature", value)
        for value in sampling["smoothing_temperatures"]
    ]
    if len(temperatures) < 2 or any(value <= 0.0 for value in temperatures):
        raise ValueError("sampling.smoothing_temperatures needs at least two positive values")
    state_range = list(sampling["state_range"])
    if len(state_range) != 2:
        raise ValueError("sampling.state_range must contain [minimum, maximum]")
    state_minimum = _require_finite("state range minimum", state_range[0])
    state_maximum = _require_finite("state range maximum", state_range[1])
    if state_minimum >= state_maximum:
        raise ValueError("sampling.state_range must be strictly increasing")
    if int(sampling["running_calibration_steps"]) < 1:
        raise ValueError("sampling.running_calibration_steps must be positive")
    time_points = int(sampling["evaluation_time_points"])
    state_points = int(sampling["evaluation_state_points"])
    analytic_points = int(sampling["analytic_evaluation_points"])
    if min(time_points, state_points) < 2 or analytic_points < 4:
        raise ValueError("analytic evaluation grids require at least two points per axis")
    if time_points * state_points != analytic_points:
        raise ValueError(
            "sampling.analytic_evaluation_points must equal "
            "evaluation_time_points * evaluation_state_points"
        )

    acceptance = config["acceptance"]
    nonnegative = (
        "mean_z_tolerance",
        "maximum_calibration_relative_error",
        "maximum_relative_l2",
    )
    for name in nonnegative:
        if _require_finite(f"acceptance.{name}", acceptance[name]) < 0.0:
            raise ValueError(f"acceptance.{name} must be nonnegative")
    minimum_finite = _require_finite(
        "acceptance.minimum_finite_fraction", acceptance["minimum_finite_fraction"]
    )
    minimum_cosine = _require_finite("acceptance.minimum_cosine", acceptance["minimum_cosine"])
    minimum_sign = _require_finite(
        "acceptance.minimum_sign_agreement", acceptance["minimum_sign_agreement"]
    )
    if not 0.0 <= minimum_finite <= 1.0:
        raise ValueError("acceptance.minimum_finite_fraction must lie in [0, 1]")
    if not -1.0 <= minimum_cosine <= 1.0:
        raise ValueError("acceptance.minimum_cosine must lie in [-1, 1]")
    if not 0.0 <= minimum_sign <= 1.0:
        raise ValueError("acceptance.minimum_sign_agreement must lie in [0, 1]")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML experiment configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("foundation config must be a YAML mapping")
    _validate_config(config)
    return config


def configure_numerics(config: dict[str, Any]) -> jnp.dtype:
    """Apply the declared precision before any compiled computation."""
    name = str(config["numerics"].get("compute_dtype", "float64"))
    if name not in {"float32", "float64"}:
        raise ValueError("numerics.compute_dtype must be float32 or float64")
    jax.config.update("jax_enable_x64", name == "float64")
    jax.config.update(
        "jax_default_matmul_precision",
        str(config["numerics"].get("matmul_precision", "highest")),
    )
    return jnp.float64 if name == "float64" else jnp.float32


def hard_threshold_truth(
    state: jax.Array,
    time_value: jax.Array,
    *,
    sigma: float,
    horizon: float,
    threshold: float,
) -> jax.Array:
    """Analytic Brownian costate for ``1{X_T >= threshold}`` at ``time < T``.

    The terminal-time derivative is distributional, so this function returns
    ``nan`` rather than clipping the remaining horizon when ``time >= T``.
    """
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    dtype = jnp.result_type(state, time_value, jnp.float32)
    state = jnp.asarray(state, dtype=dtype)
    time_value = jnp.asarray(time_value, dtype=dtype)
    remaining = horizon - time_value
    scale = sigma * jnp.sqrt(remaining)
    standardized = (threshold - state) / scale
    value = jnp.exp(-0.5 * standardized**2) / jnp.sqrt(2.0 * jnp.pi) / scale
    return jnp.where(remaining > 0.0, value, jnp.nan)


def _stopped_hard_reward(
    terminal: jax.Array,
    *,
    threshold: float,
    dtype: jnp.dtype,
) -> jax.Array:
    """Evaluate the hard oracle while making its no-gradient contract explicit."""
    return jax.lax.stop_gradient((terminal >= threshold).astype(dtype))


def sample_hard_threshold_labels(
    key: jax.Array,
    batch_size: int,
    *,
    sigma: float,
    horizon: float,
    threshold: float,
    time_steps: int,
    minimum_remaining_steps: int,
    state_minimum: float,
    state_maximum: float,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Draw exact one-suffix Brownian BEL labels at uniform discrete anchors.

    Returns state ``[B,1]``, time ``[B]``, stopped label ``[B,1]``, and
    analytic conditional target ``[B,1]``.  The base normal law is fixed and
    parameter independent; the reward value is never differentiated.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if time_steps < 1 or not 1 <= minimum_remaining_steps <= time_steps:
        raise ValueError("minimum_remaining_steps must lie in [1, time_steps]")
    if not (
        math.isfinite(state_minimum)
        and math.isfinite(state_maximum)
        and state_minimum < state_maximum
    ):
        raise ValueError("state bounds must be finite and strictly increasing")
    key_time, key_state, key_noise = jax.random.split(key, 3)
    maximum_anchor_index = time_steps - minimum_remaining_steps
    anchor_indices = jax.random.randint(
        key_time,
        (batch_size,),
        minval=0,
        maxval=maximum_anchor_index + 1,
        dtype=jnp.int32,
    )
    times = anchor_indices.astype(dtype) * jnp.asarray(horizon / time_steps, dtype=dtype)
    states = jax.random.uniform(
        key_state,
        (batch_size, 1),
        minval=state_minimum,
        maxval=state_maximum,
        dtype=dtype,
    )
    noise = jax.random.normal(key_noise, (batch_size, 1), dtype=dtype)
    remaining = horizon - times
    scale = sigma * jnp.sqrt(remaining)
    terminal = states[:, 0] + scale * noise[:, 0]
    reward = _stopped_hard_reward(terminal, threshold=threshold, dtype=dtype)
    labels = jax.lax.stop_gradient(reward[:, None] * noise / scale[:, None])
    truth = hard_threshold_truth(
        states[:, 0],
        times,
        sigma=sigma,
        horizon=horizon,
        threshold=threshold,
    )[:, None]
    return states, times, labels, truth


def _mean_summary(samples: np.ndarray, truth: float) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=float).reshape(-1)
    finite = np.isfinite(samples)
    values = samples[finite]
    if values.size < 2:
        raise ValueError("calibration requires at least two finite samples")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    difference = mean - truth
    z_score = (
        difference / standard_error if standard_error > 0 else math.copysign(math.inf, difference)
    )
    if difference == 0.0 and standard_error == 0.0:
        z_score = 0.0
    relative_error = abs(difference) / max(abs(truth), 1e-15)
    return {
        "samples": int(samples.size),
        "finite_samples": int(values.size),
        "effective_sample_size_after_filtering": int(values.size),
        "finite_fraction": float(finite.mean()),
        "mean": mean,
        "standard_error": standard_error,
        "truth": float(truth),
        "difference": float(difference),
        "absolute_z": float(abs(z_score)),
        "relative_error": float(relative_error),
        "variance": float(values.var(ddof=1)),
    }


def _tail_summary(samples: np.ndarray) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=float).reshape(-1)
    finite = np.isfinite(samples)
    values = samples[finite]
    centered = values - values.mean() if values.size else values
    energy = centered**2
    total = float(energy.sum())

    def share(fraction: float) -> float | None:
        if not values.size:
            return None
        if total == 0.0:
            return 0.0
        count = max(1, int(math.ceil(fraction * values.size)))
        return float(np.sort(energy)[-count:].sum() / total)

    norms = np.abs(values)
    return {
        "count": int(samples.size),
        "finite_fraction": float(finite.mean()) if samples.size else 0.0,
        "p50": float(np.quantile(norms, 0.50)) if values.size else None,
        "p95": float(np.quantile(norms, 0.95)) if values.size else None,
        "p99": float(np.quantile(norms, 0.99)) if values.size else None,
        "p999": float(np.quantile(norms, 0.999)) if values.size else None,
        "maximum": float(norms.max()) if values.size else None,
        "top_1_percent_centered_energy_share": share(0.01),
        "top_0_1_percent_centered_energy_share": share(0.001),
        "theorem_facing_clipping": False,
    }


def _smooth_label_components(
    noise: jax.Array,
    *,
    state: float,
    scale: float,
    finite_difference_epsilon: float,
) -> jax.Array:
    """Return columns ``[BEL, pathwise, CRN finite difference]``."""
    terminal = state + scale * noise
    payoff = jax.lax.stop_gradient(jnp.sin(terminal))
    bel = payoff * noise / scale
    pathwise = jnp.cos(terminal)
    finite_difference = (
        jnp.sin(terminal + finite_difference_epsilon)
        - jnp.sin(terminal - finite_difference_epsilon)
    ) / (2.0 * finite_difference_epsilon)
    return jnp.stack((bel, pathwise, finite_difference), axis=-1)


def _equivalence_metric(
    reference: jax.Array,
    candidate: jax.Array,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    reference_host = np.asarray(reference)
    candidate_host = np.asarray(candidate)
    absolute = np.abs(reference_host - candidate_host)
    scale = np.maximum(np.abs(reference_host), np.abs(candidate_host))
    return {
        "pass": bool(np.all(absolute <= atol + rtol * scale)),
        "maximum_absolute_difference": float(absolute.max(initial=0.0)),
        "rtol": rtol,
        "atol": atol,
        "compared_values": int(reference_host.size),
    }


def _smooth_terminal_calibration(
    key: jax.Array,
    *,
    samples: int,
    sigma: float,
    horizon: float,
    state: float,
    anchor_time: float,
    finite_difference_epsilon: float,
    dtype: jnp.dtype,
) -> dict[str, Any]:
    remaining = horizon - anchor_time
    scale = sigma * math.sqrt(remaining)
    noise_started = time.perf_counter()
    noise = jax.random.normal(key, (samples,), dtype=dtype)
    noise.block_until_ready()
    noise_seconds = time.perf_counter() - noise_started

    vectorized_started = time.perf_counter()
    components = _smooth_label_components(
        noise,
        state=state,
        scale=scale,
        finite_difference_epsilon=finite_difference_epsilon,
    )
    components.block_until_ready()
    vectorized_seconds = time.perf_counter() - vectorized_started

    diagnostic_samples = min(samples, 1_024)
    diagnostic_noise = noise[:diagnostic_samples]
    eager_reference = components[:diagnostic_samples]
    jit_started = time.perf_counter()
    jit_components = jax.jit(_smooth_label_components, static_argnames=("state", "scale"))(
        diagnostic_noise,
        state=state,
        scale=scale,
        finite_difference_epsilon=finite_difference_epsilon,
    )
    jit_components.block_until_ready()
    jit_seconds = time.perf_counter() - jit_started

    loop_started = time.perf_counter()
    loop_components = jax.lax.map(
        lambda one_noise: _smooth_label_components(
            one_noise[None],
            state=state,
            scale=scale,
            finite_difference_epsilon=finite_difference_epsilon,
        )[0],
        diagnostic_noise,
    )
    loop_components.block_until_ready()
    loop_seconds = time.perf_counter() - loop_started

    rtol, atol = (1.0e-10, 1.0e-10) if dtype == jnp.float64 else (2.0e-6, 2.0e-6)
    equivalence = {
        "diagnostic_samples": diagnostic_samples,
        "dtype": str(dtype),
        "eager_vs_jit": _equivalence_metric(
            eager_reference,
            jit_components,
            rtol=rtol,
            atol=atol,
        ),
        "vectorized_vs_lax_map_loop": _equivalence_metric(
            eager_reference,
            loop_components,
            rtol=rtol,
            atol=atol,
        ),
    }
    equivalence["pass"] = bool(
        equivalence["eager_vs_jit"]["pass"] and equivalence["vectorized_vs_lax_map_loop"]["pass"]
    )
    bel, pathwise, fd = (components[:, index] for index in range(3))
    truth = math.exp(-0.5 * scale**2) * math.cos(state)
    return {
        "payoff": "sin",
        "analytic_truth": float(truth),
        "bel": _mean_summary(np.asarray(bel), truth),
        "pathwise": _mean_summary(np.asarray(pathwise), truth),
        "common_random_finite_difference": _mean_summary(np.asarray(fd), truth),
        "paired_bel_minus_pathwise": _mean_summary(
            np.asarray(bel - pathwise),
            0.0,
        ),
        "paired_pathwise_minus_common_random_finite_difference": _mean_summary(
            np.asarray(pathwise - fd),
            0.0,
        ),
        "implementation_equivalence": equivalence,
        "phase_timing_seconds": {
            "noise_sampling": noise_seconds,
            "eager_vectorized_labels": vectorized_seconds,
            "jit_compile_and_first_execution": jit_seconds,
            "lax_map_loop_compile_and_first_execution": loop_seconds,
        },
    }


def _hard_terminal_calibration(
    key: jax.Array,
    *,
    samples: int,
    sigma: float,
    horizon: float,
    threshold: float,
    state: float,
    anchor_time: float,
    smoothing_temperatures: list[float],
    dtype: jnp.dtype,
) -> tuple[dict[str, Any], np.ndarray]:
    remaining = horizon - anchor_time
    scale = sigma * math.sqrt(remaining)
    noise_started = time.perf_counter()
    noise = jax.random.normal(key, (samples,), dtype=dtype)
    noise.block_until_ready()
    noise_seconds = time.perf_counter() - noise_started
    oracle_started = time.perf_counter()
    terminal = state + scale * noise
    reward = _stopped_hard_reward(terminal, threshold=threshold, dtype=dtype)
    bel = reward * noise / scale
    bel.block_until_ready()
    oracle_and_bel_seconds = time.perf_counter() - oracle_started
    truth = float(
        hard_threshold_truth(
            jnp.asarray(state, dtype=dtype),
            jnp.asarray(anchor_time, dtype=dtype),
            sigma=sigma,
            horizon=horizon,
            threshold=threshold,
        )
    )
    smoothing_started = time.perf_counter()
    smooth = {}
    for temperature in smoothing_temperatures:
        probability = jax.nn.sigmoid((terminal - threshold) / temperature)
        gradient = probability * (1.0 - probability) / temperature
        smooth[str(temperature)] = _mean_summary(np.asarray(gradient), truth)
    smoothing_seconds = time.perf_counter() - smoothing_started
    result = {
        "analytic_truth": truth,
        "bel": _mean_summary(np.asarray(bel), truth),
        "label_tails": _tail_summary(np.asarray(bel)),
        "unsmoothed_terminal_gradient": {
            "status": "INVALID_ZERO_OR_UNDEFINED_FOR_HARD_INDICATOR",
            "used": False,
        },
        "sigmoid_smoothing_ablation": smooth,
        "phase_timing_seconds": {
            "noise_sampling": noise_seconds,
            "hard_reward_oracle_and_bel_label": oracle_and_bel_seconds,
            "hard_reward_oracle": "UNAVAILABLE_MONOLITHIC_WITH_BEL_LABEL",
            "bel_label": "UNAVAILABLE_MONOLITHIC_WITH_HARD_REWARD_ORACLE",
            "sigmoid_smoothing_ablation": smoothing_seconds,
        },
    }
    return result, np.asarray(bel)


def _running_cost_calibration(
    key: jax.Array,
    *,
    samples: int,
    steps: int,
    sigma: float,
    horizon: float,
    state: float,
    dtype: jnp.dtype,
) -> tuple[dict[str, Any], np.ndarray]:
    x0 = jnp.full((samples, 1), state, dtype=dtype)
    times = jnp.linspace(0.0, horizon, steps + 1, dtype=dtype)

    def zero_drift(x: jax.Array, time_value: jax.Array, context: jax.Array) -> jax.Array:
        del time_value, context
        return jnp.zeros_like(x)

    rollout_started = time.perf_counter()
    rollout = simulate_additive_em_rollout(key, x0, times, zero_drift, sigma)
    rollout.states.block_until_ready()
    rollout_seconds = time.perf_counter() - rollout_started

    oracle_started = time.perf_counter()
    running = jax.lax.stop_gradient(rollout.states[:, :, 0] ** 2)
    terminal = jax.lax.stop_gradient(rollout.states[:, -1, 0] ** 2)
    terminal.block_until_ready()
    oracle_seconds = time.perf_counter() - oracle_started

    label_started = time.perf_counter()
    labels = assemble_bel_costate_labels(
        rollout,
        anchors=jnp.zeros((samples,), dtype=jnp.int32),
        running_values=running,
        terminal_values=terminal,
    )
    labels.label.block_until_ready()
    label_seconds = time.perf_counter() - label_started
    running_values = np.asarray(labels.running_component[:, 0])
    terminal_values = np.asarray(labels.terminal_component[:, 0])
    sum_values = np.asarray(labels.label[:, 0])
    terminal_truth = 2.0 * state
    running_truth = 2.0 * state * horizon
    sum_truth = terminal_truth + running_truth
    return {
        "running_cost": "right_endpoint_x_squared",
        "terminal_cost": "terminal_x_squared",
        "analytic_discrete_truth": {
            "terminal_component": float(terminal_truth),
            "running_component": float(running_truth),
            "sum": float(sum_truth),
        },
        "running_component": _mean_summary(running_values, running_truth),
        "terminal_component": _mean_summary(terminal_values, terminal_truth),
        "sum": _mean_summary(sum_values, sum_truth),
        "label_tails": {
            "running_component": _tail_summary(running_values),
            "terminal_component": _tail_summary(terminal_values),
            "sum": _tail_summary(sum_values),
        },
        "phase_timing_seconds": {
            "rollout_and_state_tangent": rollout_seconds,
            "state_tangent": "UNAVAILABLE_MONOLITHIC_WITH_ROLLOUT",
            "reward_oracle_evaluation": oracle_seconds,
            "bel_weight_and_label_assembly": label_seconds,
            "bel_weight": "UNAVAILABLE_MONOLITHIC_WITH_LABEL_ASSEMBLY",
        },
    }, sum_values


def _regression_metrics(
    factory: MLPFactory,
    params: Any,
    *,
    states: jax.Array,
    times: jax.Array,
    sigma: float,
    horizon: float,
    threshold: float,
) -> tuple[dict[str, Any], jax.Array, jax.Array]:
    states = jnp.asarray(states)
    times = jnp.asarray(times, dtype=states.dtype)
    if states.ndim != 2 or states.shape[-1] != 1 or times.shape != (states.shape[0],):
        raise ValueError("regression evaluation requires states [B,1] and times [B]")
    prediction = factory.forward(params, states, times).reshape(-1)
    truth = hard_threshold_truth(
        states[:, 0],
        times,
        sigma=sigma,
        horizon=horizon,
        threshold=threshold,
    )
    difference = prediction - truth
    relative_l2 = jnp.linalg.norm(difference) / jnp.maximum(jnp.linalg.norm(truth), 1e-15)
    cosine = jnp.vdot(prediction, truth) / jnp.maximum(
        jnp.linalg.norm(prediction) * jnp.linalg.norm(truth),
        1e-15,
    )
    sign_agreement = jnp.mean((prediction > 0.0) == (truth > 0.0))
    finite = jnp.isfinite(prediction) & jnp.isfinite(truth)
    metrics = {
        "relative_l2": float(relative_l2),
        "cosine": float(cosine),
        "sign_agreement": float(sign_agreement),
        "mean_squared_error": float(jnp.mean(difference**2)),
        "maximum_absolute_error": float(jnp.max(jnp.abs(difference))),
        "evaluation_points": int(prediction.size),
        "finite_fraction": float(jnp.mean(finite)),
    }
    return metrics, prediction, truth


def _regression_grid(
    factory: MLPFactory,
    params: Any,
    *,
    sigma: float,
    horizon: float,
    threshold: float,
    maximum_anchor_time: float,
    state_minimum: float,
    state_maximum: float,
    time_points: int,
    state_points: int,
    dtype: jnp.dtype,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate once on the untouched, deterministic Gate-A test grid."""
    grid_times = jnp.linspace(0.0, maximum_anchor_time, time_points, dtype=dtype)
    grid_states = jnp.linspace(state_minimum, state_maximum, state_points, dtype=dtype)
    time_grid, state_grid = jnp.meshgrid(grid_times, grid_states, indexing="ij")
    flat_time = time_grid.reshape(-1)
    flat_state = state_grid.reshape((-1, 1))
    metrics, prediction, truth = _regression_metrics(
        factory,
        params,
        states=flat_state,
        times=flat_time,
        sigma=sigma,
        horizon=horizon,
        threshold=threshold,
    )
    arrays = {
        "evaluation_time": np.asarray(flat_time),
        "evaluation_state": np.asarray(flat_state[:, 0]),
        "evaluation_prediction": np.asarray(prediction),
        "evaluation_truth": np.asarray(truth),
    }
    return metrics, arrays


def _minibatch_schedule(
    key: jax.Array,
    *,
    dataset_size: int,
    training_steps: int,
    batch_size: int,
) -> jax.Array:
    """Build an epoch-shuffled schedule that exposes every declared sample."""
    exposures = training_steps * batch_size
    if min(dataset_size, training_steps, batch_size) < 1:
        raise ValueError("dataset_size, training_steps, and batch_size must be positive")
    if exposures < dataset_size:
        raise ValueError("the schedule must expose every declared training trajectory")
    epochs = math.ceil(exposures / dataset_size)
    permutations = [
        np.asarray(jax.random.permutation(jax.random.fold_in(key, epoch), dataset_size))
        for epoch in range(epochs)
    ]
    flattened = np.concatenate(permutations)[:exposures]
    return jnp.asarray(flattened.reshape((training_steps, batch_size)), dtype=jnp.int32)


def _tree_all_finite(tree: Any) -> bool:
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(tree))


def _train_regressor(
    config: dict[str, Any],
    *,
    dtype: jnp.dtype,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    dynamics = config["dynamics"]
    network = config["costate_network"]
    sampling = config["sampling"]
    sigma = float(dynamics["sigma"])
    horizon = float(dynamics["horizon"])
    threshold = float(dynamics["threshold"])
    minimum_remaining_steps = int(config.get("anchors", {}).get("minimum_remaining_steps", 1))
    dynamics_steps = int(dynamics["steps"])
    dt = horizon / dynamics_steps
    maximum_anchor_time = horizon - minimum_remaining_steps * dt
    state_minimum, state_maximum = map(float, sampling["state_range"])
    batch_size = int(network["batch_size"])
    training_steps = int(network["training_steps"])
    eval_every = int(network.get("eval_every", max(1, training_steps // 10)))
    learning_rates = [float(value) for value in network["learning_rates"]]
    training_trajectories = int(sampling["training_trajectories"])
    validation_trajectories = int(sampling["validation_trajectories"])
    factory = MLPFactory(
        hidden_dims=tuple(int(width) for width in network["hidden_sizes"]),
        time_embed_dim=int(network["time_embedding_dim"]),
        activation=str(network.get("activation", "silu")),
    )
    master_key = jax.random.PRNGKey(int(config["experiment"]["seed"]))
    initialization_started = time.perf_counter()
    initial_params = factory.init(_stream_key(master_key, "network_initialization"), 1, 1)
    initial_params = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, dtype=dtype),
        initial_params,
    )
    for leaf in jax.tree_util.tree_leaves(initial_params):
        leaf.block_until_ready()
    initialization_seconds = time.perf_counter() - initialization_started

    dataset_started = time.perf_counter()
    train_state, train_time, train_label, train_truth = sample_hard_threshold_labels(
        _stream_key(master_key, "training_dataset"),
        training_trajectories,
        sigma=sigma,
        horizon=horizon,
        threshold=threshold,
        time_steps=dynamics_steps,
        minimum_remaining_steps=minimum_remaining_steps,
        state_minimum=state_minimum,
        state_maximum=state_maximum,
        dtype=dtype,
    )
    validation_state, validation_time, validation_label, validation_truth = (
        sample_hard_threshold_labels(
            _stream_key(master_key, "validation_dataset"),
            validation_trajectories,
            sigma=sigma,
            horizon=horizon,
            threshold=threshold,
            time_steps=dynamics_steps,
            minimum_remaining_steps=minimum_remaining_steps,
            state_minimum=state_minimum,
            state_maximum=state_maximum,
            dtype=dtype,
        )
    )
    validation_truth.block_until_ready()
    dataset_seconds = time.perf_counter() - dataset_started

    schedule_started = time.perf_counter()
    schedule = _minibatch_schedule(
        _stream_key(master_key, "minibatch_schedule"),
        dataset_size=training_trajectories,
        training_steps=training_steps,
        batch_size=batch_size,
    )
    schedule.block_until_ready()
    schedule_seconds = time.perf_counter() - schedule_started

    def loss_and_aux(
        params: Any,
        state: jax.Array,
        time_value: jax.Array,
        label: jax.Array,
        truth: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        prediction = factory.forward(params, state, time_value)
        loss = jnp.mean((prediction - label) ** 2)
        truth_mse = jnp.mean((prediction - truth) ** 2)
        return loss, (truth_mse, jnp.mean(jnp.abs(label)), jnp.max(jnp.abs(label)))

    @jax.jit
    def update(
        params: Any,
        opt_state: Any,
        state: jax.Array,
        time_value: jax.Array,
        label: jax.Array,
        truth: jax.Array,
        learning_rate: jax.Array,
    ):
        (loss, aux), gradients = jax.value_and_grad(loss_and_aux, has_aux=True)(
            params,
            state,
            time_value,
            label,
            truth,
        )
        params, opt_state = adam_update(
            opt_state,
            gradients,
            params,
            lr=learning_rate,
        )
        return params, opt_state, loss, aux

    all_rows: list[dict[str, Any]] = []
    candidate_results = []
    best_params = None
    best_selection: dict[str, Any] | None = None
    training_total_seconds = 0.0
    validation_total_seconds = 0.0
    for learning_rate in learning_rates:
        params = jax.tree_util.tree_map(jnp.array, initial_params)
        opt_state = init_adam(params)
        started = time.perf_counter()
        last_loss_value = math.nan
        for step in range(1, training_steps + 1):
            indices = schedule[step - 1]
            params, opt_state, loss, aux = update(
                params,
                opt_state,
                train_state[indices],
                train_time[indices],
                train_label[indices],
                train_truth[indices],
                jnp.asarray(learning_rate, dtype=dtype),
            )
            if step == 1 or step % eval_every == 0 or step == training_steps:
                truth_mse, mean_abs_label, max_abs_label = aux
                last_loss_value = float(loss)
                logged_values = (
                    last_loss_value,
                    float(truth_mse),
                    float(mean_abs_label),
                    float(max_abs_label),
                )
                if not all(math.isfinite(value) for value in logged_values):
                    raise FloatingPointError("nonfinite value during Gate-A costate training")
                all_rows.append(
                    {
                        "learning_rate": learning_rate,
                        "step": step,
                        "training_label_mse": logged_values[0],
                        "training_truth_mse": logged_values[1],
                        "mean_absolute_label": logged_values[2],
                        "maximum_absolute_label": logged_values[3],
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
        if not _tree_all_finite(params):
            raise FloatingPointError("nonfinite network parameters after Gate-A training")
        candidate_training_seconds = time.perf_counter() - started
        training_total_seconds += candidate_training_seconds
        validation_started = time.perf_counter()
        validation_metrics, validation_prediction, _ = _regression_metrics(
            factory,
            params,
            states=validation_state,
            times=validation_time,
            sigma=sigma,
            horizon=horizon,
            threshold=threshold,
        )
        validation_label_mse = float(
            jnp.mean((validation_prediction[:, None] - validation_label) ** 2)
        )
        candidate_validation_seconds = time.perf_counter() - validation_started
        validation_total_seconds += candidate_validation_seconds
        record = {
            "learning_rate": learning_rate,
            "final_training_label_mse": last_loss_value,
            "training_seconds": candidate_training_seconds,
            "selection_validation_seconds": candidate_validation_seconds,
            "selection_validation_label_mse": validation_label_mse,
            **{f"selection_{name}": value for name, value in validation_metrics.items()},
        }
        if not all(
            math.isfinite(float(value))
            for name, value in record.items()
            if name != "learning_rate" and name != "training_seconds"
        ):
            raise FloatingPointError("nonfinite validation metric during Gate-A training")
        candidate_results.append(record)
        if (
            best_selection is None
            or record["selection_mean_squared_error"]
            < best_selection["selection_mean_squared_error"]
        ):
            best_params = params
            best_selection = record
    assert best_params is not None and best_selection is not None
    heldout_started = time.perf_counter()
    heldout_metrics, best_arrays = _regression_grid(
        factory,
        best_params,
        sigma=sigma,
        horizon=horizon,
        threshold=threshold,
        maximum_anchor_time=maximum_anchor_time,
        state_minimum=state_minimum,
        state_maximum=state_maximum,
        time_points=int(sampling["evaluation_time_points"]),
        state_points=int(sampling["evaluation_state_points"]),
        dtype=dtype,
    )
    heldout_seconds = time.perf_counter() - heldout_started
    selected = {
        "learning_rate": best_selection["learning_rate"],
        **heldout_metrics,
    }
    raw_limit = 16_384
    best_arrays.update(
        {
            "training_time": np.asarray(train_time[:raw_limit]),
            "training_state": np.asarray(train_state[:raw_limit, 0]),
            "training_label": np.asarray(train_label[:raw_limit, 0]),
            "training_truth": np.asarray(train_truth[:raw_limit, 0]),
            "validation_time": np.asarray(validation_time[:raw_limit]),
            "validation_state": np.asarray(validation_state[:raw_limit, 0]),
            "validation_label": np.asarray(validation_label[:raw_limit, 0]),
            "validation_truth": np.asarray(validation_truth[:raw_limit, 0]),
        }
    )
    schedule_host = np.asarray(schedule)
    summary = {
        "selected_learning_rate": best_selection["learning_rate"],
        "selection_rule": "minimum_disjoint_validation_analytic_mse_gate_A_only",
        "heldout_used_for_selection": False,
        "selected": selected,
        "selected_validation": best_selection,
        "candidates": candidate_results,
        "maximum_anchor_time": maximum_anchor_time,
        "data_budget": {
            "unique_training_reward_queries": training_trajectories,
            "unique_validation_reward_queries": validation_trajectories,
            "candidate_count": len(learning_rates),
            "label_exposures_per_candidate": training_steps * batch_size,
            "total_label_exposures": training_steps * batch_size * len(learning_rates),
            "common_training_dataset_across_candidates": True,
            "common_minibatch_schedule_across_candidates": True,
            "common_initialization_across_candidates": True,
            "every_training_trajectory_exposed": True,
        },
        "training_label_tails": _tail_summary(np.asarray(train_label)),
        "validation_label_tails": _tail_summary(np.asarray(validation_label)),
        "minibatch_schedule_sha256": hashlib.sha256(schedule_host.tobytes()).hexdigest(),
        "phase_timing_seconds": {
            "network_initialization": initialization_seconds,
            "training_and_validation_dataset_generation": dataset_seconds,
            "minibatch_schedule_generation": schedule_seconds,
            "costate_training_all_candidates": training_total_seconds,
            "selection_validation_all_candidates": validation_total_seconds,
            "heldout_analytic_evaluation": heldout_seconds,
        },
    }
    return best_params, all_rows, summary, best_arrays


def _save_checkpoint(path: Path, params: Any, metadata: dict[str, Any]) -> None:
    leaves, tree = jax.tree_util.tree_flatten(params)
    arrays = {
        f"leaf_{index:04d}": np.asarray(jax.device_get(leaf)) for index, leaf in enumerate(leaves)
    }
    if not arrays or not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise FloatingPointError("checkpoint parameters must be nonempty and finite")
    payload = {
        "format_version": 1,
        "tree_structure": str(tree),
        "leaf_count": len(leaves),
        "leaf_shapes": [list(array.shape) for array in arrays.values()],
        "leaf_dtypes": [str(array.dtype) for array in arrays.values()],
        **metadata,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(payload, sort_keys=True))
    np.savez_compressed(path, **arrays)


def load_checkpoint_leaves(path: str | Path, template: Any) -> Any:
    """Restore a foundation checkpoint into an architecture-matched template."""
    template_leaves, tree = jax.tree_util.tree_flatten(template)
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("checkpoint is missing metadata_json")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if int(metadata.get("format_version", -1)) != 1:
            raise ValueError("unsupported checkpoint format version")
        if metadata.get("tree_structure") != str(tree):
            raise ValueError("checkpoint/template tree structure mismatch")
        if int(metadata["leaf_count"]) != len(template_leaves):
            raise ValueError("checkpoint/template leaf count mismatch")
        expected_names = [f"leaf_{index:04d}" for index in range(len(template_leaves))]
        if any(name not in archive.files for name in expected_names):
            raise ValueError("checkpoint is missing one or more parameter leaves")
        stored = [np.asarray(archive[name]) for name in expected_names]
    declared_shapes = [tuple(shape) for shape in metadata.get("leaf_shapes", [])]
    declared_dtypes = list(metadata.get("leaf_dtypes", []))
    if len(declared_shapes) != len(stored) or len(declared_dtypes) != len(stored):
        raise ValueError("checkpoint leaf metadata is incomplete")
    leaves = []
    for index, (array, template_leaf) in enumerate(zip(stored, template_leaves, strict=True)):
        if array.shape != tuple(template_leaf.shape) or array.shape != declared_shapes[index]:
            raise ValueError("checkpoint/template leaf shape mismatch")
        if (
            str(array.dtype) != declared_dtypes[index]
            or array.dtype != np.asarray(template_leaf).dtype
        ):
            raise ValueError("checkpoint/template leaf dtype mismatch")
        if not np.all(np.isfinite(array)):
            raise ValueError("checkpoint contains nonfinite parameter values")
        leaves.append(jnp.asarray(array))
    return jax.tree_util.tree_unflatten(tree, leaves)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    revision = command("rev-parse", "HEAD")
    status = command("status", "--porcelain")
    return {"revision": revision, "dirty": None if status is None else bool(status)}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer, jnp.integer)):
        return int(value)
    if isinstance(value, (np.floating, jnp.floating, float)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _snapshot_sources(run_dir: Path) -> dict[str, str]:
    """Copy the exact theorem-facing sources into the run artifact."""
    root = Path(__file__).resolve().parents[2]
    sources = (
        Path(__file__).resolve(),
        root / "schrodinger_bridge" / "solvers" / "malliavin_adjoint.py",
    )
    hashes: dict[str, str] = {}
    for source in sources:
        relative = source.relative_to(root)
        destination = run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        artifact_name = str(destination.relative_to(run_dir))
        hashes[artifact_name] = _sha256(destination)
        if hashes[artifact_name] != _sha256(source):
            raise OSError(f"source snapshot hash mismatch for {relative}")
    return hashes


def _running_manifest(
    config: dict[str, Any],
    *,
    run_dir: Path,
    started_at_utc: str,
    scientific_contract: dict[str, Any],
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return {
        "status": "RUNNING_INCOMPLETE",
        "failed_gate": None,
        "experiment": "malliavin_adjoint_matching_gate_a",
        "started_at_utc": started_at_utc,
        "output_dir": str(run_dir),
        "command": [sys.executable, *sys.argv],
        "seed": int(config["experiment"]["seed"]),
        "git": _git_state(root),
        "canonical_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "scientific_contract": scientific_contract,
        "reward_oracle_identifiers": _REWARD_ORACLE_IDENTIFIERS,
        "stale_artifacts_may_exist_during_overwrite": True,
    }


def _acceptance(
    config: dict[str, Any],
    smooth: dict[str, Any],
    hard: dict[str, Any],
    running: dict[str, Any],
    regression: dict[str, Any],
) -> tuple[bool, list[str], dict[str, str]]:
    thresholds = config["acceptance"]
    reasons: list[str] = []
    mean_z = float(thresholds["mean_z_tolerance"])
    maximum_mean_relative = float(thresholds["maximum_calibration_relative_error"])
    minimum_finite = float(thresholds["minimum_finite_fraction"])

    def check_mean(name: str, metric: dict[str, Any]) -> None:
        required = ("finite_fraction", "absolute_z", "relative_error")
        if any(not math.isfinite(float(metric[key])) for key in required):
            reasons.append(f"{name}:invalid_metric")
            return
        if metric["finite_fraction"] < minimum_finite:
            reasons.append(f"{name}:nonfinite")
        if metric["absolute_z"] > mean_z:
            reasons.append(f"{name}:mean_z")
        if metric["relative_error"] > maximum_mean_relative:
            reasons.append(f"{name}:relative_error")

    for name, metric in (
        ("smooth_bel", smooth["bel"]),
        ("smooth_pathwise", smooth["pathwise"]),
        ("smooth_common_random_finite_difference", smooth["common_random_finite_difference"]),
        ("hard_bel", hard["bel"]),
        ("running_component", running["running_component"]),
        ("running_terminal_component", running["terminal_component"]),
        ("running_sum", running["sum"]),
    ):
        check_mean(name, metric)

    paired = smooth["paired_bel_minus_pathwise"]
    if not math.isfinite(float(paired["finite_fraction"])) or not math.isfinite(
        float(paired["absolute_z"])
    ):
        reasons.append("smooth_bel_vs_pathwise:invalid_metric")
    else:
        if paired["finite_fraction"] < minimum_finite:
            reasons.append("smooth_bel_vs_pathwise:nonfinite")
        if paired["absolute_z"] > mean_z:
            reasons.append("smooth_bel_vs_pathwise:mean_z")

    equivalence = smooth["implementation_equivalence"]
    if not equivalence["eager_vs_jit"]["pass"]:
        reasons.append("smooth_labels:eager_vs_jit")
    if not equivalence["vectorized_vs_lax_map_loop"]["pass"]:
        reasons.append("smooth_labels:vectorized_vs_loop")

    for temperature, metric in hard["sigmoid_smoothing_ablation"].items():
        finite_fraction = float(metric["finite_fraction"])
        if not math.isfinite(finite_fraction) or finite_fraction < minimum_finite:
            reasons.append(f"hard_sigmoid_{temperature}:nonfinite")

    selected = regression["selected"]
    regression_metrics = (
        "finite_fraction",
        "relative_l2",
        "cosine",
        "sign_agreement",
    )
    if any(not math.isfinite(float(selected[name])) for name in regression_metrics):
        reasons.append("costate_regression:invalid_metric")
    else:
        if selected["finite_fraction"] < minimum_finite:
            reasons.append("costate_regression:nonfinite")
        if selected["relative_l2"] > float(thresholds["maximum_relative_l2"]):
            reasons.append("costate_regression:relative_l2")
        if selected["cosine"] < float(thresholds["minimum_cosine"]):
            reasons.append("costate_regression:cosine")
        if selected["sign_agreement"] < float(thresholds["minimum_sign_agreement"]):
            reasons.append("costate_regression:sign_agreement")

    statuses = {
        "smooth_terminal": (
            "FAIL_CALIBRATION"
            if any(reason.startswith("smooth_") for reason in reasons)
            else "PASS_CALIBRATION"
        ),
        "hard_threshold": (
            "FAIL_CALIBRATION"
            if any(reason.startswith("hard_") for reason in reasons)
            else "PASS_CALIBRATION"
        ),
        "running_cost": (
            "FAIL_CALIBRATION"
            if any(reason.startswith("running_") for reason in reasons)
            else "PASS_CALIBRATION"
        ),
        "costate_regression": (
            "FAIL_CALIBRATION"
            if any(reason.startswith("costate_regression") for reason in reasons)
            else "PASS_CALIBRATION"
        ),
    }
    return not reasons, reasons, statuses


def _execute_foundation(
    config: dict[str, Any],
    *,
    run_dir: Path,
    dtype: jnp.dtype,
    started_at_utc: str,
    scientific_contract: dict[str, Any],
    snapshot_hashes: dict[str, str],
) -> dict[str, Any]:
    """Execute validated Gate A after the durable RUNNING manifest is written."""
    dynamics = config["dynamics"]
    sampling = config["sampling"]
    calibration_samples = int(sampling["calibration_samples"])
    seed_value = int(config["experiment"]["seed"])
    master = jax.random.PRNGKey(seed_value)
    started = time.perf_counter()
    smooth_started = time.perf_counter()
    smooth = _smooth_terminal_calibration(
        _stream_key(master, "smooth_calibration"),
        samples=calibration_samples,
        sigma=float(dynamics["sigma"]),
        horizon=float(dynamics["horizon"]),
        state=float(sampling["calibration_state"]),
        anchor_time=float(sampling["calibration_anchor_time"]),
        finite_difference_epsilon=float(sampling["finite_difference_epsilon"]),
        dtype=dtype,
    )
    smooth_seconds = time.perf_counter() - smooth_started
    hard_started = time.perf_counter()
    hard, hard_labels = _hard_terminal_calibration(
        _stream_key(master, "hard_calibration"),
        samples=calibration_samples,
        sigma=float(dynamics["sigma"]),
        horizon=float(dynamics["horizon"]),
        threshold=float(dynamics["threshold"]),
        state=float(sampling["calibration_state"]),
        anchor_time=float(sampling["calibration_anchor_time"]),
        smoothing_temperatures=[float(value) for value in sampling["smoothing_temperatures"]],
        dtype=dtype,
    )
    hard_seconds = time.perf_counter() - hard_started
    running_started = time.perf_counter()
    running, running_labels = _running_cost_calibration(
        _stream_key(master, "running_calibration"),
        samples=calibration_samples,
        steps=int(sampling["running_calibration_steps"]),
        sigma=float(dynamics["sigma"]),
        horizon=float(dynamics["horizon"]),
        state=float(sampling["calibration_state"]),
        dtype=dtype,
    )
    running_seconds = time.perf_counter() - running_started
    calibration_seconds = time.perf_counter() - started

    training_started = time.perf_counter()
    params, training_rows, regression, grid_arrays = _train_regressor(config, dtype=dtype)
    training_seconds = time.perf_counter() - training_started
    accepted, reasons, component_statuses = _acceptance(
        config,
        smooth,
        hard,
        running,
        regression,
    )
    smooth["status"] = component_statuses["smooth_terminal"]
    hard["status"] = component_statuses["hard_threshold"]
    running["status"] = component_statuses["running_cost"]
    regression["status"] = component_statuses["costate_regression"]
    scientific = bool(scientific_contract["eligible_for_scientific_status"])
    if accepted and scientific:
        status = "PASS_MAM_ANALYTIC_FOUNDATION"
    elif accepted:
        status = "PASS_MAM_ANALYTIC_FOUNDATION_SMOKE_NOT_EVIDENCE"
    else:
        status = "FAIL_MAM_ANALYTIC_FOUNDATION"

    with (run_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training_rows[0]))
        writer.writeheader()
        writer.writerows(training_rows)
    np.savez_compressed(
        run_dir / "raw_samples.npz",
        hard_bel_labels=hard_labels[: min(16_384, hard_labels.size)],
        running_plus_terminal_bel_labels=running_labels[: min(16_384, running_labels.size)],
        **grid_arrays,
    )
    _save_checkpoint(
        run_dir / "checkpoint.npz",
        params,
        {
            "solver_status": "CONDITIONAL_MAM_FOUNDATION",
            "selected_learning_rate": regression["selected_learning_rate"],
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )
    results = {
        "status": status,
        "scientific_contract": scientific_contract,
        "intended_for_scientific_evidence": scientific_contract["requested"],
        "eligible_for_scientific_status": scientific,
        "claim_scope": (
            "generic/unpinned Brownian fixed-policy BEL costate foundation; "
            "not a conditional pinned solver and not a global bridge"
        ),
        "estimator": {
            "hard_terminal_label": "1{X_T>=c} (W_T-W_t)/(sigma*(T-t))",
            "generic_discrete_bel_weight": (
                "((m-k)*dt)^-1 sum_{j=k}^{m-1} (Sigma^+ J_{j+1,k})^T dW_j"
            ),
            "regression_loss": "ordinary_uncensored_mse",
            "approximation": "Euler-Maruyama only for the running-cost calibration",
        },
        "reward_oracle_identifiers": _REWARD_ORACLE_IDENTIFIERS,
        "smooth_terminal": smooth,
        "hard_threshold": hard,
        "running_cost": running,
        "hard_threshold_costate_regression": regression,
        "acceptance": {"pass": accepted, "failed_reasons": reasons, **config["acceptance"]},
        "timing_seconds": {
            "smooth_terminal_calibration": smooth_seconds,
            "hard_terminal_calibration": hard_seconds,
            "running_rollout_tangent_and_bel": running_seconds,
            "all_calibrations": calibration_seconds,
            "costate_training_and_evaluation": training_seconds,
            "costate_training_breakdown": regression["phase_timing_seconds"],
            "smooth_label_breakdown": smooth["phase_timing_seconds"],
            "hard_label_breakdown": hard["phase_timing_seconds"],
            "running_bel_breakdown": running["phase_timing_seconds"],
            "rollout": running["phase_timing_seconds"]["rollout_and_state_tangent"],
            "state_tangent": "UNAVAILABLE_MONOLITHIC_WITH_ROLLOUT",
            "bel_weight": "UNAVAILABLE_MONOLITHIC_WITH_LABEL_ASSEMBLY",
            "label_assembly": "UNAVAILABLE_MONOLITHIC_WITH_BEL_WEIGHT",
            "hamiltonian_target": "NOT_RUN_GATE_A",
            "policy_training": "NOT_RUN_GATE_A",
            "total": time.perf_counter() - started,
        },
        "reward_query_accounting": {
            "smooth_terminal_calibration": calibration_samples,
            "hard_terminal_calibration": calibration_samples,
            "running_cost_calibration": calibration_samples,
            "running_terminal_calibration": calibration_samples,
            **regression["data_budget"],
            "analytic_heldout_reward_queries": 0,
        },
        "output_dir": str(run_dir),
    }
    _write_json(run_dir / "results.json", results)

    root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__).resolve(),
        root / "schrodinger_bridge" / "solvers" / "malliavin_adjoint.py",
    ]
    manifest = {
        "status": status,
        "failed_gate": None if accepted else reasons[0],
        "experiment": "malliavin_adjoint_matching_gate_a",
        "profile": (
            "scientific_frozen_production"
            if scientific
            else (
                "scientific_requested_contract_mismatch_not_evidence"
                if scientific_contract["requested"]
                else "smoke_not_evidence"
            )
        ),
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "seed": seed_value,
        "git": _git_state(root),
        "source_sha256": {str(path.relative_to(root)): _sha256(path) for path in source_paths},
        "config_sha256": _sha256(run_dir / "resolved_config.json"),
        "artifact_sha256": {
            **{
                name: _sha256(run_dir / name)
                for name in (
                    "results.json",
                    "training_metrics.csv",
                    "raw_samples.npz",
                    "checkpoint.npz",
                )
            },
            **snapshot_hashes,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "matmul_precision": str(jax.config.jax_default_matmul_precision),
        },
        "theorem_assumptions": {
            "fixed_base_noise": True,
            "constant_full_rank_diffusion": True,
            "adapted_bel_weight": True,
            "reward_values_stopped": True,
            "reward_gradient_called": False,
            "theorem_facing_clipping": False,
            "global_endpoint_bridge_claim": False,
        },
        "scientific_contract": scientific_contract,
        "reward_oracle_identifiers": _REWARD_ORACLE_IDENTIFIERS,
        "phase_timing_seconds": results["timing_seconds"],
        "random_streams": {
            name: {"derivation": "jax.random.fold_in(master_key, stream_id)", "stream_id": value}
            for name, value in _STREAM_IDS.items()
        },
        "data_budget": regression["data_budget"],
        "heldout_used_for_selection": False,
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return results


def run_foundation(
    config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute Gate A with durable running/failure state and source snapshots."""
    config = json.loads(json.dumps(config))
    if seed is not None:
        config["experiment"]["seed"] = int(seed)
    if output_dir is not None:
        config["experiment"]["output_dir"] = str(output_dir)
    _validate_config(config)
    run_dir = Path(config["experiment"]["output_dir"])
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output already exists: {run_dir}; pass --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at_utc = datetime.now(UTC).isoformat()
    scientific_contract = _scientific_contract(config)
    running_manifest = _running_manifest(
        config,
        run_dir=run_dir,
        started_at_utc=started_at_utc,
        scientific_contract=scientific_contract,
    )
    # This is intentionally the first overwrite: an interrupted rerun can no
    # longer leave a stale PASS as the authoritative status.
    _write_json(run_dir / "run_manifest.json", running_manifest)

    snapshot_hashes: dict[str, str] = {}
    try:
        dtype = configure_numerics(config)
        _write_json(run_dir / "resolved_config.json", config)
        (run_dir / "figures").mkdir(exist_ok=True)
        snapshot_hashes = _snapshot_sources(run_dir)
        running_manifest.update(
            {
                "config_sha256": _sha256(run_dir / "resolved_config.json"),
                "source_snapshot_sha256": snapshot_hashes,
            }
        )
        _write_json(run_dir / "run_manifest.json", running_manifest)
        return _execute_foundation(
            config,
            run_dir=run_dir,
            dtype=dtype,
            started_at_utc=started_at_utc,
            scientific_contract=scientific_contract,
            snapshot_hashes=snapshot_hashes,
        )
    except Exception as error:
        failure_manifest = {
            **running_manifest,
            "status": "FAILED_EXCEPTION",
            "failed_gate": "runtime_exception",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "exception": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "source_snapshot_sha256": snapshot_hashes,
            "stale_artifacts_may_exist": True,
        }
        _write_json(run_dir / "run_manifest.json", failure_manifest)
        raise


__all__ = [
    "configure_numerics",
    "hard_threshold_truth",
    "load_checkpoint_leaves",
    "load_config",
    "run_foundation",
    "sample_hard_threshold_labels",
]
