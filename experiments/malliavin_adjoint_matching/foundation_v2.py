"""Query-matched DEV experiment for Malliavin Adjoint Matching Gate A v2.

This module intentionally implements only model development and hyperparameter
selection.  It has no scientific-pass status and no held-out test-set random
stream.  Every candidate is selected on one shared, analytic DEV context set.

For a Brownian suffix from ``(t, x)`` write

``X_T = x + sigma * sqrt(T - t) * Z`` and ``R = 1{X_T >= c}``.

The three direct costate arms regress stopped Bismut labels

``R Z / (sigma sqrt(T - t))``

using one query, two independent queries, or an antithetic pair.  The value
critic regresses the stopped scalar return from the exact RAW1 records and is
evaluated through ``grad_x V(t, x)``.  No training loss receives analytic
costate truth, and no theorem-facing label is clipped, filtered, or smoothed.
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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxlib
import numpy as np
import yaml

from schrodinger_bridge.network_factory import MLPFactory, NetworkFactory
from schrodinger_bridge.networks import adam_update, get_activation, init_adam

Arm = Literal["RAW1", "IID2", "ANTITHETIC2", "VALUE_CRITIC"]
SampledArm = Literal["RAW1", "IID2", "ANTITHETIC2"]

DIRECT_ARMS: tuple[SampledArm, ...] = ("RAW1", "IID2", "ANTITHETIC2")
ALL_ARMS: tuple[Arm, ...] = (*DIRECT_ARMS, "VALUE_CRITIC")
SUCCESS_STATUS = "COMPLETE_MAM_GATE_A_V2_DEV_NOT_EVIDENCE"
SMOKE_STATUS = "COMPLETE_MAM_GATE_A_V2_DEV_SMOKE_NOT_LOCKABLE"

_FROZEN_DEV_CONTRACT = {
    "experiment": {
        "protocol": "mam_gate_a_v2_dev",
        "seed": 20260811,
        "dev_replicates": [0, 1, 2],
        "intended_for_scientific_evidence": False,
    },
    "numerics": {"compute_dtype": "float64", "matmul_precision": "highest"},
    "dynamics": {
        "family": "brownian_threshold",
        "state_dim": 1,
        "horizon": 1.0,
        "steps": 64,
        "sigma": 0.7,
        "threshold": 0.8,
    },
    "anchors": {"minimum_remaining_steps": 8, "distribution": "uniform_discrete"},
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
        "reward_query_budget": 131072,
        "dev_contexts": 32768,
        "state_range": [-1.0, 2.0],
    },
}
SMOKE_STATUS = "COMPLETE_MAM_GATE_A_V2_DEV_SMOKE_NOT_LOCKABLE"

_FROZEN_DEV_CONTRACT = {
    "experiment": {
        "protocol": "mam_gate_a_v2_dev",
        "seed": 20_260_811,
        "dev_replicates": [0, 1, 2],
        "intended_for_scientific_evidence": False,
    },
    "numerics": {"compute_dtype": "float64", "matmul_precision": "highest"},
    "dynamics": {
        "family": "brownian_threshold",
        "state_dim": 1,
        "horizon": 1.0,
        "steps": 64,
        "sigma": 0.7,
        "threshold": 0.8,
    },
    "anchors": {"minimum_remaining_steps": 8, "distribution": "uniform_discrete"},
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
        "reward_query_budget": 131_072,
        "dev_contexts": 32_768,
        "state_range": [-1.0, 2.0],
    },
}

# Development-only streams.  A later evaluator must live in a separate module
# and own a separately frozen key namespace.
DEV_STREAM_IDS = {
    "training_contexts": 40_001,
    "base_noise": 40_002,
    "independent_second_noise": 40_003,
    "network_initialization": 40_004,
    "minibatch_schedule": 40_005,
    "validation_contexts": 40_006,
}
_ARM_STREAM_IDS = {
    "SHARED": 0,
    "RAW1": 1,
    "IID2": 2,
    "ANTITHETIC2": 3,
    "VALUE_CRITIC": 4,
}

_ALLOWED_CONFIG_KEYS = {
    "experiment": {
        "protocol",
        "seed",
        "dev_replicates",
        "output_dir",
        "intended_for_scientific_evidence",
    },
    "numerics": {"compute_dtype", "matmul_precision"},
    "dynamics": {"family", "state_dim", "horizon", "steps", "sigma", "threshold"},
    "anchors": {"minimum_remaining_steps", "distribution"},
    "costate_network": {
        "hidden_sizes",
        "activation",
        "time_embedding_dim",
        "optimizer",
        "learning_rates",
        "training_steps",
        "batch_size",
        "eval_every",
    },
    "sampling": {"reward_query_budget", "dev_contexts", "state_range"},
}


@dataclass(frozen=True)
class QueryMatchedDataset:
    """One query-matched training arm.

    Shapes are ``states[C,1]``, ``times[C]``, ``suffix_normals[C,M,1]``,
    ``rewards[C,M]``, and scalar-column targets ``[C,1]``.  Here ``M=1``
    for RAW1 and ``M=2`` for IID2/ANTITHETIC2.  The number of contexts is
    ``C=Q/M``, so every sampled arm has exactly ``Q`` logical reward calls.
    """

    arm: SampledArm
    states: jax.Array
    times: jax.Array
    suffix_normals: jax.Array
    rewards: jax.Array
    supervised_targets: jax.Array
    reward_query_count: int

    @property
    def context_count(self) -> int:
        return int(self.states.shape[0])


class DevContexts(NamedTuple):
    """Shared analytic contexts used only for candidate selection."""

    states: jax.Array
    times: jax.Array
    analytic_costates: jax.Array
    analytic_values: jax.Array


class HardBELQueries(NamedTuple):
    """Stopped hard rewards and single-query BEL labels from explicit noise."""

    rewards: jax.Array
    single_query_labels: jax.Array
    reward_query_count: int


def _stream_key(
    master: jax.Array,
    domain: str,
    replicate: int,
    arm: str,
) -> jax.Array:
    """Derive every stochastic object as root/domain/replicate/arm."""
    if domain not in DEV_STREAM_IDS:
        raise ValueError(f"unknown DEV key domain: {domain}")
    if arm not in _ARM_STREAM_IDS:
        raise ValueError(f"unknown DEV key arm: {arm}")
    key = jax.random.fold_in(master, DEV_STREAM_IDS[domain])
    key = jax.random.fold_in(key, int(replicate))
    return jax.random.fold_in(key, _ARM_STREAM_IDS[arm])


def _require_finite(name: str, value: Any) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _dev_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Compare a run with the immutable DEV-selection configuration."""
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

    compare(_FROZEN_DEV_CONTRACT, observed, "config")
    payload = json.dumps(
        _FROZEN_DEV_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "matches_frozen_dev_contract": not mismatches,
        "operational_fields_excluded": ["experiment.output_dir"],
        "mismatches": mismatches,
        "contract_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _require_exact_keys(config: dict[str, Any]) -> None:
    observed_sections = set(config)
    expected_sections = set(_ALLOWED_CONFIG_KEYS)
    if observed_sections != expected_sections:
        missing = sorted(expected_sections - observed_sections)
        unexpected = sorted(observed_sections - expected_sections)
        raise ValueError(f"invalid DEV config sections; missing={missing}, unexpected={unexpected}")
    for section, expected in _ALLOWED_CONFIG_KEYS.items():
        value = config[section]
        if not isinstance(value, dict):
            raise ValueError(f"config section {section} must be a mapping")
        observed = set(value)
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(f"invalid {section} keys; missing={missing}, unexpected={unexpected}")


def _validate_config(config: dict[str, Any]) -> None:
    """Validate the complete, closed DEV protocol before writing artifacts."""
    if not isinstance(config, dict):
        raise ValueError("DEV config must be a mapping")
    _require_exact_keys(config)

    experiment = config["experiment"]
    if experiment["protocol"] != "mam_gate_a_v2_dev":
        raise ValueError("experiment.protocol must be 'mam_gate_a_v2_dev'")
    if isinstance(experiment["seed"], bool) or not isinstance(experiment["seed"], int):
        raise ValueError("experiment.seed must be an integer")
    if experiment["dev_replicates"] != [0, 1, 2]:
        raise ValueError("experiment.dev_replicates must be exactly [0, 1, 2]")
    if not str(experiment["output_dir"]).strip():
        raise ValueError("experiment.output_dir must be nonempty")
    if experiment["intended_for_scientific_evidence"] is not False:
        raise ValueError("the DEV protocol cannot be intended for scientific evidence")

    numerics = config["numerics"]
    if numerics["compute_dtype"] not in {"float32", "float64"}:
        raise ValueError("numerics.compute_dtype must be float32 or float64")
    if numerics["matmul_precision"] not in {
        "default",
        "high",
        "highest",
        "bfloat16",
        "tensorfloat32",
        "float32",
    }:
        raise ValueError("numerics.matmul_precision is not recognized by JAX")

    dynamics = config["dynamics"]
    if dynamics["family"] != "brownian_threshold":
        raise ValueError("DEV supports only dynamics.family=brownian_threshold")
    if isinstance(dynamics["state_dim"], bool) or int(dynamics["state_dim"]) != 1:
        raise ValueError("DEV requires dynamics.state_dim=1")
    horizon = _require_finite("dynamics.horizon", dynamics["horizon"])
    sigma = _require_finite("dynamics.sigma", dynamics["sigma"])
    _require_finite("dynamics.threshold", dynamics["threshold"])
    steps = int(dynamics["steps"])
    if horizon <= 0.0 or sigma <= 0.0 or steps < 1:
        raise ValueError("horizon, sigma, and steps must be positive")

    anchors = config["anchors"]
    if anchors["distribution"] != "uniform_discrete":
        raise ValueError("DEV requires anchors.distribution=uniform_discrete")
    minimum_remaining = int(anchors["minimum_remaining_steps"])
    if not 1 <= minimum_remaining <= steps:
        raise ValueError("minimum_remaining_steps must lie in [1, dynamics.steps]")

    network = config["costate_network"]
    hidden_sizes = tuple(int(width) for width in network["hidden_sizes"])
    if not hidden_sizes or any(width < 1 for width in hidden_sizes):
        raise ValueError("costate_network.hidden_sizes must contain positive widths")
    if int(network["time_embedding_dim"]) < 2:
        raise ValueError("costate_network.time_embedding_dim must be at least two")
    get_activation(str(network["activation"]))
    if network["optimizer"] != "adam":
        raise ValueError("DEV currently requires costate_network.optimizer=adam")
    learning_rates = [
        _require_finite("learning rate", value) for value in network["learning_rates"]
    ]
    if not learning_rates or any(value <= 0.0 for value in learning_rates):
        raise ValueError("costate_network.learning_rates must contain positive values")
    if len(set(learning_rates)) != len(learning_rates):
        raise ValueError("costate_network.learning_rates must be unique")
    training_steps = int(network["training_steps"])
    batch_size = int(network["batch_size"])
    eval_every = int(network["eval_every"])
    if min(training_steps, batch_size, eval_every) < 1:
        raise ValueError("training_steps, batch_size, and eval_every must be positive")

    sampling = config["sampling"]
    query_budget = int(sampling["reward_query_budget"])
    dev_contexts = int(sampling["dev_contexts"])
    if query_budget < 2 or query_budget % 2:
        raise ValueError("sampling.reward_query_budget must be a positive even integer")
    if dev_contexts < 2:
        raise ValueError("sampling.dev_contexts must be at least two")
    if training_steps * batch_size < query_budget:
        raise ValueError("training_steps * batch_size must expose every RAW1 training context")
    state_range = sampling["state_range"]
    if not isinstance(state_range, list) or len(state_range) != 2:
        raise ValueError("sampling.state_range must be [minimum, maximum]")
    state_minimum = _require_finite("state range minimum", state_range[0])
    state_maximum = _require_finite("state range maximum", state_range[1])
    if state_minimum >= state_maximum:
        raise ValueError("sampling.state_range must be strictly increasing")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a Gate-A-v2 DEV YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate_config(config)
    return config


def configure_numerics(config: dict[str, Any]) -> jnp.dtype:
    """Apply the declared JAX precision before compiled work."""
    name = str(config["numerics"]["compute_dtype"])
    jax.config.update("jax_enable_x64", name == "float64")
    jax.config.update(
        "jax_default_matmul_precision",
        str(config["numerics"]["matmul_precision"]),
    )
    return jnp.float64 if name == "float64" else jnp.float32


def hard_threshold_value(
    state: jax.Array,
    time_value: jax.Array,
    *,
    sigma: float,
    horizon: float,
    threshold: float,
) -> jax.Array:
    """Analytic ``E[1{X_T>=threshold}|X_t=state]`` for Brownian motion."""
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    dtype = jnp.result_type(state, time_value, jnp.float32)
    state = jnp.asarray(state, dtype=dtype)
    time_value = jnp.asarray(time_value, dtype=dtype)
    remaining = jnp.asarray(horizon, dtype=dtype) - time_value
    standardized = (state - threshold) / (sigma * jnp.sqrt(remaining))
    value = jsp.special.ndtr(standardized)
    return jnp.where(remaining > 0.0, value, jnp.nan)


def hard_threshold_costate(
    state: jax.Array,
    time_value: jax.Array,
    *,
    sigma: float,
    horizon: float,
    threshold: float,
) -> jax.Array:
    """Analytic spatial derivative of :func:`hard_threshold_value`."""
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    dtype = jnp.result_type(state, time_value, jnp.float32)
    state = jnp.asarray(state, dtype=dtype)
    time_value = jnp.asarray(time_value, dtype=dtype)
    remaining = jnp.asarray(horizon, dtype=dtype) - time_value
    scale = jnp.asarray(sigma, dtype=dtype) * jnp.sqrt(remaining)
    standardized = (threshold - state) / scale
    density = jnp.exp(-0.5 * standardized**2) / jnp.sqrt(2.0 * jnp.pi)
    return jnp.where(remaining > 0.0, density / scale, jnp.nan)


def _stopped_reward(
    terminal_states: jax.Array,
    *,
    threshold: float,
    dtype: jnp.dtype,
) -> jax.Array:
    return jax.lax.stop_gradient((terminal_states >= threshold).astype(dtype))


def _sample_contexts(
    key: jax.Array,
    context_count: int,
    *,
    horizon: float,
    time_steps: int,
    minimum_remaining_steps: int,
    state_minimum: float,
    state_maximum: float,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    key_time, key_state = jax.random.split(key)
    maximum_anchor = time_steps - minimum_remaining_steps
    indices = jax.random.randint(
        key_time,
        (context_count,),
        minval=0,
        maxval=maximum_anchor + 1,
        dtype=jnp.int32,
    )
    times = indices.astype(dtype) * jnp.asarray(horizon / time_steps, dtype=dtype)
    states = jax.random.uniform(
        key_state,
        (context_count, 1),
        minval=state_minimum,
        maxval=state_maximum,
        dtype=dtype,
    )
    return states, times


def hard_bel_from_noise(
    states: jax.Array,
    times: jax.Array,
    suffix_normals: jax.Array,
    *,
    sigma: float,
    horizon: float,
    threshold: float,
) -> HardBELQueries:
    """Construct stopped hard rewards and BEL labels from explicit normals.

    This function contains no random sampling.  ``suffix_normals`` has shape
    ``[C,M,1]`` and therefore represents exactly ``C*M`` logical reward
    calls.  It is the audit boundary for RAW1, IID2, and antithetic algebra.
    """
    states = jnp.asarray(states)
    times = jnp.asarray(times, dtype=states.dtype)
    suffix_normals = jnp.asarray(suffix_normals, dtype=states.dtype)
    if states.ndim != 2 or states.shape[1] != 1:
        raise ValueError("states must have shape [contexts,1]")
    if times.shape != (states.shape[0],):
        raise ValueError("times must have shape [contexts]")
    if (
        suffix_normals.ndim != 3
        or suffix_normals.shape[:1] != states.shape[:1]
        or suffix_normals.shape[2] != 1
    ):
        raise ValueError("suffix_normals must have shape [contexts,queries,1]")
    if suffix_normals.shape[1] < 1:
        raise ValueError("each context needs at least one suffix normal")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    remaining = jnp.asarray(horizon, dtype=states.dtype) - times
    scale = jnp.asarray(sigma, dtype=states.dtype) * jnp.sqrt(remaining)
    terminal_states = states[:, None, :] + scale[:, None, None] * suffix_normals
    rewards = _stopped_reward(
        terminal_states[..., 0],
        threshold=threshold,
        dtype=states.dtype,
    )
    labels = jax.lax.stop_gradient(rewards[..., None] * suffix_normals / scale[:, None, None])
    return HardBELQueries(
        rewards=rewards,
        single_query_labels=labels,
        reward_query_count=int(states.shape[0] * suffix_normals.shape[1]),
    )


def sample_query_matched_dataset(
    context_key: jax.Array,
    base_noise_key: jax.Array,
    arm: SampledArm,
    *,
    independent_second_noise_key: jax.Array | None = None,
    reward_query_budget: int,
    sigma: float,
    horizon: float,
    threshold: float,
    time_steps: int,
    minimum_remaining_steps: int,
    state_minimum: float,
    state_maximum: float,
    dtype: jnp.dtype,
) -> QueryMatchedDataset:
    """Purely sample one costate arm with exactly ``reward_query_budget`` calls.

    RAW1 has ``Q`` contexts and one suffix per context.  IID2 and
    ANTITHETIC2 have ``Q/2`` contexts and two suffixes per context.  The
    returned target is the per-context average of the stopped single-query
    BEL labels.  No sample is clipped, filtered, or replaced.
    """
    if arm not in DIRECT_ARMS:
        raise ValueError(f"arm must be one of {DIRECT_ARMS}, got {arm!r}")
    if reward_query_budget < 2 or reward_query_budget % 2:
        raise ValueError("reward_query_budget must be a positive even integer")
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

    queries_per_context = 1 if arm == "RAW1" else 2
    context_count = reward_query_budget // queries_per_context
    states, times = _sample_contexts(
        context_key,
        context_count,
        horizon=horizon,
        time_steps=time_steps,
        minimum_remaining_steps=minimum_remaining_steps,
        state_minimum=state_minimum,
        state_maximum=state_maximum,
        dtype=dtype,
    )
    first_normal = jax.random.normal(base_noise_key, (context_count, 1, 1), dtype=dtype)
    if arm == "RAW1":
        suffix_normals = first_normal
    elif arm == "IID2":
        if independent_second_noise_key is None:
            raise ValueError("IID2 requires independent_second_noise_key")
        second_normal = jax.random.normal(
            independent_second_noise_key,
            (context_count, 1, 1),
            dtype=dtype,
        )
        suffix_normals = jnp.concatenate((first_normal, second_normal), axis=1)
    else:
        suffix_normals = jnp.concatenate((first_normal, -first_normal), axis=1)

    queries = hard_bel_from_noise(
        states,
        times,
        suffix_normals,
        sigma=sigma,
        horizon=horizon,
        threshold=threshold,
    )
    labels = jax.lax.stop_gradient(jnp.mean(queries.single_query_labels, axis=1))
    dataset = QueryMatchedDataset(
        arm=arm,
        states=states,
        times=times,
        suffix_normals=suffix_normals,
        rewards=queries.rewards,
        supervised_targets=labels,
        reward_query_count=queries.reward_query_count,
    )
    _validate_dataset(dataset)
    return dataset


def _validate_dataset(dataset: QueryMatchedDataset) -> None:
    context_count = dataset.context_count
    queries_per_context = 1 if dataset.arm == "RAW1" else 2
    expected = {
        "states": (context_count, 1),
        "times": (context_count,),
        "suffix_normals": (context_count, queries_per_context, 1),
        "rewards": (context_count, queries_per_context),
        "supervised_targets": (context_count, 1),
    }
    for name, shape in expected.items():
        value = getattr(dataset, name)
        if value.shape != shape:
            raise ValueError(f"{dataset.arm} {name} must have shape {shape}, got {value.shape}")
        if not bool(jnp.all(jnp.isfinite(value))):
            raise FloatingPointError(f"{dataset.arm} {name} contains nonfinite values")
    if dataset.reward_query_count != context_count * queries_per_context:
        raise ValueError("dataset reward query count does not match its tensor shape")
    if not bool(jnp.all((dataset.rewards == 0.0) | (dataset.rewards == 1.0))):
        raise ValueError("hard rewards must be binary")


def critic_costate(
    factory: NetworkFactory,
    params: Any,
    states: jax.Array,
    times: jax.Array,
) -> jax.Array:
    """Evaluate ``grad_x sigmoid(logit_psi(t,x))``, shape ``[B,d]``."""
    states = jnp.asarray(states)
    times = jnp.asarray(times, dtype=states.dtype)
    if states.ndim != 2 or times.shape != (states.shape[0],):
        raise ValueError("critic_costate requires states [B,d] and times [B]")

    def scalar_value(state: jax.Array, time_value: jax.Array) -> jax.Array:
        output = factory.forward(params, state[None, :], time_value[None])
        if output.shape != (1, 1):
            raise ValueError("value critic factory must return scalar output [B,1]")
        return jax.nn.sigmoid(output[0, 0])

    return jax.vmap(jax.grad(scalar_value))(states, times)


def _sample_dev_contexts(
    key: jax.Array,
    count: int,
    *,
    sigma: float,
    horizon: float,
    threshold: float,
    time_steps: int,
    minimum_remaining_steps: int,
    state_minimum: float,
    state_maximum: float,
    dtype: jnp.dtype,
) -> DevContexts:
    states, times = _sample_contexts(
        key,
        count,
        horizon=horizon,
        time_steps=time_steps,
        minimum_remaining_steps=minimum_remaining_steps,
        state_minimum=state_minimum,
        state_maximum=state_maximum,
        dtype=dtype,
    )
    costates = hard_threshold_costate(
        states[:, 0],
        times,
        sigma=sigma,
        horizon=horizon,
        threshold=threshold,
    )[:, None]
    values = hard_threshold_value(
        states[:, 0],
        times,
        sigma=sigma,
        horizon=horizon,
        threshold=threshold,
    )[:, None]
    for name, value in (
        ("DEV states", states),
        ("DEV times", times),
        ("DEV costates", costates),
        ("DEV values", values),
    ):
        if not bool(jnp.all(jnp.isfinite(value))):
            raise FloatingPointError(f"{name} contains nonfinite values")
    return DevContexts(states, times, costates, values)


def _minibatch_schedule(
    key: jax.Array,
    *,
    dataset_size: int,
    training_steps: int,
    batch_size: int,
) -> jax.Array:
    exposures = training_steps * batch_size
    if min(dataset_size, training_steps, batch_size) < 1:
        raise ValueError("dataset_size, training_steps, and batch_size must be positive")
    if exposures < dataset_size:
        raise ValueError("the schedule must expose every declared training context")
    epochs = math.ceil(exposures / dataset_size)
    permutations = [
        np.asarray(jax.random.permutation(jax.random.fold_in(key, epoch), dataset_size))
        for epoch in range(epochs)
    ]
    flattened = np.concatenate(permutations)[:exposures]
    return jnp.asarray(flattened.reshape((training_steps, batch_size)), dtype=jnp.int32)


def _hash_named_arrays(**arrays: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(jax.device_get(arrays[name])))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _dataset_hash(dataset: QueryMatchedDataset) -> str:
    return _hash_named_arrays(
        states=dataset.states,
        times=dataset.times,
        suffix_normals=dataset.suffix_normals,
        rewards=dataset.rewards,
        supervised_targets=dataset.supervised_targets,
    )


def _tree_hash(tree: Any) -> str:
    leaves = jax.tree_util.tree_leaves(tree)
    return _hash_named_arrays(**{f"leaf_{index:04d}": leaf for index, leaf in enumerate(leaves)})


def _tree_all_finite(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return bool(leaves) and all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def _tree_finite_flag(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(False)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def _tail_summary(values: jax.Array) -> dict[str, Any]:
    array = np.asarray(jax.device_get(values), dtype=float).reshape(-1)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise FloatingPointError("tail diagnostics require finite, nonempty values")
    centered_energy = (array - array.mean()) ** 2
    total = float(centered_energy.sum())

    def energy_share(fraction: float) -> float:
        if total == 0.0:
            return 0.0
        count = max(1, int(math.ceil(fraction * array.size)))
        return float(np.sort(centered_energy)[-count:].sum() / total)

    absolute = np.abs(array)
    return {
        "count": int(array.size),
        "p50": float(np.quantile(absolute, 0.50)),
        "p95": float(np.quantile(absolute, 0.95)),
        "p99": float(np.quantile(absolute, 0.99)),
        "p999": float(np.quantile(absolute, 0.999)),
        "maximum": float(absolute.max()),
        "top_1_percent_centered_energy_share": energy_share(0.01),
        "top_0_1_percent_centered_energy_share": energy_share(0.001),
    }


def _costate_metrics(prediction: jax.Array, truth: jax.Array) -> dict[str, Any]:
    prediction = jnp.asarray(prediction)
    truth = jnp.asarray(truth, dtype=prediction.dtype)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("costate prediction and truth must have equal shape [B,d]")
    if not bool(jnp.all(jnp.isfinite(prediction))) or not bool(jnp.all(jnp.isfinite(truth))):
        raise FloatingPointError("costate metrics received nonfinite values")
    difference = prediction - truth
    prediction_norm = jnp.linalg.norm(prediction)
    truth_norm = jnp.linalg.norm(truth)
    denominator = jnp.maximum(prediction_norm * truth_norm, 1.0e-15)
    return {
        "mean_squared_error": float(jnp.mean(difference**2)),
        "relative_l2": float(jnp.linalg.norm(difference) / jnp.maximum(truth_norm, 1.0e-15)),
        "cosine": float(jnp.vdot(prediction, truth) / denominator),
        "sign_agreement": float(jnp.mean((prediction > 0.0) == (truth > 0.0))),
        "maximum_absolute_error": float(jnp.max(jnp.abs(difference))),
        "evaluation_contexts": int(prediction.shape[0]),
        "finite_fraction": 1.0,
    }


def _make_factory(config: dict[str, Any]) -> MLPFactory:
    network = config["costate_network"]
    return MLPFactory(
        hidden_dims=tuple(int(width) for width in network["hidden_sizes"]),
        time_embed_dim=int(network["time_embedding_dim"]),
        activation=str(network["activation"]),
    )


def _predict_costate(
    arm: Arm,
    factory: NetworkFactory,
    params: Any,
    states: jax.Array,
    times: jax.Array,
) -> jax.Array:
    if arm == "VALUE_CRITIC":
        return critic_costate(factory, params, states, times)
    return factory.forward(params, states, times)


def _fit_arm_candidates(
    arm: Arm,
    *,
    factory: NetworkFactory,
    initial_params: dict[int, Any],
    datasets: dict[int, QueryMatchedDataset],
    supervised_targets: dict[int, jax.Array],
    schedules: dict[int, jax.Array],
    dev_contexts: dict[int, DevContexts],
    learning_rates: list[float],
    training_steps: int,
    eval_every: int,
    dtype: jnp.dtype,
) -> tuple[
    dict[int, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    replicate_ids = tuple(sorted(datasets))
    if replicate_ids != (0, 1, 2):
        raise ValueError("candidate fitting requires DEV replicates 0, 1, and 2")
    if not (
        set(initial_params)
        == set(supervised_targets)
        == set(schedules)
        == set(dev_contexts)
        == set(datasets)
    ):
        raise ValueError("all candidate inputs must contain identical replicate IDs")
    for replicate in replicate_ids:
        targets = supervised_targets[replicate]
        if targets.shape != (datasets[replicate].context_count, 1):
            raise ValueError("supervised targets must have shape [contexts,1]")
        if not bool(jnp.all(jnp.isfinite(targets))):
            raise FloatingPointError(f"{arm} supervised targets contain nonfinite values")

    def supervised_loss(
        params: Any,
        states: jax.Array,
        times: jax.Array,
        targets: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        prediction = factory.forward(params, states, times)
        if arm == "VALUE_CRITIC":
            # Stable Bernoulli cross-entropy for a scalar logit.  The hard
            # target remains stopped and the reward oracle is never traced.
            loss = jnp.mean(jax.nn.softplus(prediction) - targets * prediction)
        else:
            loss = jnp.mean((prediction - targets) ** 2)
        return loss, (jnp.mean(jnp.abs(targets)), jnp.max(jnp.abs(targets)))

    @jax.jit
    def update(
        params: Any,
        optimizer_state: Any,
        states: jax.Array,
        times: jax.Array,
        targets: jax.Array,
        learning_rate: jax.Array,
    ):
        (loss, aux), gradients = jax.value_and_grad(supervised_loss, has_aux=True)(
            params,
            states,
            times,
            targets,
        )
        updated_params, updated_optimizer = adam_update(
            optimizer_state,
            gradients,
            params,
            lr=learning_rate,
        )
        finite = (
            jnp.isfinite(loss)
            & _tree_finite_flag(gradients)
            & _tree_finite_flag(updated_params)
            & _tree_finite_flag(updated_optimizer)
        )
        return updated_params, updated_optimizer, loss, aux, finite

    candidate_records: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    best_params: dict[int, Any] | None = None
    best_record: dict[str, Any] | None = None
    for learning_rate in learning_rates:
        replicate_params: dict[int, Any] = {}
        replicate_records: dict[str, dict[str, Any]] = {}
        for replicate in replicate_ids:
            dataset = datasets[replicate]
            targets = supervised_targets[replicate]
            schedule = schedules[replicate]
            dev = dev_contexts[replicate]
            params = jax.tree_util.tree_map(jnp.array, initial_params[replicate])
            optimizer_state = init_adam(params)
            candidate_finite = jnp.asarray(True)
            started = time.perf_counter()
            last_loss = math.nan
            for step in range(1, training_steps + 1):
                indices = schedule[step - 1]
                params, optimizer_state, loss, aux, finite = update(
                    params,
                    optimizer_state,
                    dataset.states[indices],
                    dataset.times[indices],
                    targets[indices],
                    jnp.asarray(learning_rate, dtype=dtype),
                )
                candidate_finite = candidate_finite & finite
                if step == 1 or step % eval_every == 0 or step == training_steps:
                    mean_absolute_target, maximum_absolute_target = aux
                    logged = (
                        float(loss),
                        float(mean_absolute_target),
                        float(maximum_absolute_target),
                    )
                    if not all(math.isfinite(value) for value in logged):
                        raise FloatingPointError(f"nonfinite {arm} training metric")
                    last_loss = logged[0]
                    training_rows.append(
                        {
                            "arm": arm,
                            "replicate": replicate,
                            "learning_rate": learning_rate,
                            "step": step,
                            "training_supervised_loss": logged[0],
                            "mean_absolute_target": logged[1],
                            "maximum_absolute_target": logged[2],
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
            if not bool(candidate_finite):
                raise FloatingPointError(f"{arm} candidate encountered a nonfinite update")
            if not _tree_all_finite(params) or not _tree_all_finite(optimizer_state):
                raise FloatingPointError(f"{arm} candidate ended with nonfinite state")

            all_training_loss = float(
                supervised_loss(params, dataset.states, dataset.times, targets)[0]
            )
            dev_prediction = _predict_costate(
                arm,
                factory,
                params,
                dev.states,
                dev.times,
            )
            metrics = _costate_metrics(dev_prediction, dev.analytic_costates)
            replicate_record: dict[str, Any] = {
                "replicate": replicate,
                "last_minibatch_supervised_loss": last_loss,
                "training_supervised_loss": all_training_loss,
                "training_seconds": time.perf_counter() - started,
                "parameter_sha256": _tree_hash(params),
                "dev_costate_metrics": metrics,
            }
            if arm == "VALUE_CRITIC":
                critic_values = jax.nn.sigmoid(factory.forward(params, dev.states, dev.times))
                if not bool(jnp.all(jnp.isfinite(critic_values))):
                    raise FloatingPointError("value critic produced nonfinite probabilities")
                replicate_record["dev_value_metrics"] = {
                    "mean_squared_error": float(
                        jnp.mean((critic_values - dev.analytic_values) ** 2)
                    ),
                    "maximum_absolute_error": float(
                        jnp.max(jnp.abs(critic_values - dev.analytic_values))
                    ),
                }
            replicate_records[str(replicate)] = replicate_record
            replicate_params[replicate] = params

        metric_names = tuple(replicate_records["0"]["dev_costate_metrics"].keys())
        mean_metrics = {
            name: float(
                np.mean(
                    [
                        replicate_records[str(replicate)]["dev_costate_metrics"][name]
                        for replicate in replicate_ids
                    ]
                )
            )
            for name in metric_names
        }
        record = {
            "arm": arm,
            "learning_rate": learning_rate,
            "replicates": replicate_records,
            "mean_dev_costate_metrics": mean_metrics,
            "mean_training_supervised_loss": float(
                np.mean(
                    [
                        replicate_records[str(replicate)]["training_supervised_loss"]
                        for replicate in replicate_ids
                    ]
                )
            ),
        }
        if not all(math.isfinite(float(value)) for value in mean_metrics.values()):
            raise FloatingPointError(f"{arm} candidate produced a nonfinite record")
        candidate_records.append(record)
        candidate_key = (mean_metrics["mean_squared_error"], learning_rate)
        best_key = (
            None
            if best_record is None
            else (
                best_record["mean_dev_costate_metrics"]["mean_squared_error"],
                best_record["learning_rate"],
            )
        )
        if best_key is None or candidate_key < best_key:
            best_params = replicate_params
            best_record = record

    assert best_params is not None and best_record is not None
    selected = {
        "arm": arm,
        "selected_learning_rate": best_record["learning_rate"],
        "selection_rule": (
            "minimum_mean_analytic_dev_costate_mse_across_replicates_0_1_2;"
            "exact_tie_selects_smaller_learning_rate"
        ),
        "mean_training_supervised_loss": best_record["mean_training_supervised_loss"],
        "mean_dev_costate_metrics": best_record["mean_dev_costate_metrics"],
        "replicates": best_record["replicates"],
        "parameter_sha256": {
            str(replicate): _tree_hash(best_params[replicate]) for replicate in replicate_ids
        },
    }
    return best_params, selected, candidate_records, training_rows


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
        if not math.isfinite(converted):
            raise ValueError("cannot serialize a nonfinite floating-point value")
        return converted
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_checkpoint(path: Path, params: dict[str, Any], metadata: dict[str, Any]) -> None:
    leaves, tree = jax.tree_util.tree_flatten(params)
    arrays = {
        f"leaf_{index:04d}": np.asarray(jax.device_get(leaf)) for index, leaf in enumerate(leaves)
    }
    if not arrays or not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise FloatingPointError("DEV checkpoint parameters must be nonempty and finite")
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


def _initial_params(
    config: dict[str, Any],
    dtype: jnp.dtype,
    replicate: int,
) -> tuple[MLPFactory, Any]:
    factory = _make_factory(config)
    master = jax.random.PRNGKey(int(config["experiment"]["seed"]))
    params = factory.init(
        _stream_key(master, "network_initialization", replicate, "SHARED"),
        1,
        1,
    )
    params = jax.tree_util.tree_map(lambda value: jnp.asarray(value, dtype=dtype), params)
    if not _tree_all_finite(params):
        raise FloatingPointError("common network initialization is nonfinite")
    return factory, params


def load_dev_checkpoint(
    path: str | Path,
    config: dict[str, Any],
    *,
    require_lockable: bool = False,
) -> dict[str, Any]:
    """Restore and authenticate all selected DEV parameter trees.

    ``require_lockable=True`` is the confirmatory handoff contract: it rejects
    smoke/dirty runs in addition to checking the sibling config, manifest,
    checkpoint metadata, and parameter hashes.
    """
    _validate_config(config)
    path = Path(path)
    resolved_path = path.parent / "resolved_config.json"
    manifest_path = path.parent / "dev_run_manifest.json"
    if not resolved_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("DEV checkpoint requires sibling resolved config and manifest")
    resolved_config = json.loads(resolved_path.read_text(encoding="utf-8"))
    if resolved_config != config:
        raise ValueError("DEV checkpoint config does not match the supplied config")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_checkpoint_sha = manifest.get("artifact_sha256", {}).get(path.name)
    if expected_checkpoint_sha != _sha256(path):
        raise ValueError("DEV checkpoint SHA-256 does not match its manifest")
    dtype = jnp.float64 if config["numerics"]["compute_dtype"] == "float64" else jnp.float32
    initial = {
        replicate: _initial_params(config, dtype, replicate)[1]
        for replicate in config["experiment"]["dev_replicates"]
    }
    template = {
        arm: {
            replicate: jax.tree_util.tree_map(jnp.array, initial[replicate])
            for replicate in config["experiment"]["dev_replicates"]
        }
        for arm in ALL_ARMS
    }
    template_leaves, tree = jax.tree_util.tree_flatten(template)
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("DEV checkpoint is missing metadata_json")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if int(metadata.get("format_version", -1)) != 1:
            raise ValueError("unsupported DEV checkpoint format")
        if metadata.get("protocol") != "mam_gate_a_v2_dev":
            raise ValueError("DEV checkpoint protocol mismatch")
        if metadata.get("status") != manifest.get("status"):
            raise ValueError("DEV checkpoint status does not match its manifest")
        if bool(metadata.get("eligible_to_lock_for_confirm")) != bool(
            manifest.get("eligible_to_lock_for_confirm")
        ):
            raise ValueError("DEV checkpoint lock eligibility mismatch")
        if require_lockable and (
            metadata.get("status") != SUCCESS_STATUS
            or metadata.get("eligible_to_lock_for_confirm") is not True
        ):
            raise ValueError("DEV checkpoint is not eligible for CONFIRM locking")
        if metadata.get("config_sha256") != _sha256(resolved_path):
            raise ValueError("DEV checkpoint resolved-config SHA-256 mismatch")
        contract = _dev_contract(config)
        if metadata.get("dev_contract_sha256") != contract["contract_sha256"]:
            raise ValueError("DEV checkpoint frozen-contract SHA-256 mismatch")
        protocol_path = Path(__file__).resolve().parents[2] / "docs" / "mam_gate_a_v2_protocol.md"
        if metadata.get("protocol_sha256") != _sha256(protocol_path):
            raise ValueError("DEV checkpoint protocol-document SHA-256 mismatch")
        selected_learning_rates = metadata.get("selected_learning_rates")
        if not isinstance(selected_learning_rates, dict) or set(selected_learning_rates) != set(
            ALL_ARMS
        ):
            raise ValueError("DEV checkpoint selected-learning-rate metadata is malformed")
        allowed_rates = {float(value) for value in config["costate_network"]["learning_rates"]}
        if any(float(value) not in allowed_rates for value in selected_learning_rates.values()):
            raise ValueError("DEV checkpoint contains an undeclared selected learning rate")
        if metadata.get("tree_structure") != str(tree):
            raise ValueError("DEV checkpoint tree structure mismatch")
        if int(metadata.get("leaf_count", -1)) != len(template_leaves):
            raise ValueError("DEV checkpoint leaf count mismatch")
        leaves = []
        for index, expected in enumerate(template_leaves):
            name = f"leaf_{index:04d}"
            if name not in archive.files:
                raise ValueError(f"DEV checkpoint is missing {name}")
            array = np.asarray(archive[name])
            if array.shape != expected.shape or array.dtype != np.asarray(expected).dtype:
                raise ValueError(f"DEV checkpoint shape or dtype mismatch for {name}")
            if not np.all(np.isfinite(array)):
                raise FloatingPointError("DEV checkpoint contains nonfinite parameters")
            leaves.append(jnp.asarray(array))
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    observed_hashes = {
        arm: {
            str(replicate): _tree_hash(restored[arm][replicate])
            for replicate in config["experiment"]["dev_replicates"]
        }
        for arm in ALL_ARMS
    }
    if observed_hashes != metadata.get("parameter_sha256"):
        raise ValueError("DEV checkpoint parameter hashes do not match metadata")
    return restored


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


def _snapshot_source(run_dir: Path) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    sources = (
        Path(__file__).resolve(),
        root / "docs" / "mam_gate_a_v2_protocol.md",
        root / "schrodinger_bridge" / "network_factory.py",
        root / "schrodinger_bridge" / "networks.py",
        root / "scripts" / "run_mam_foundation_v2_dev.py",
    )
    hashes: dict[str, str] = {}
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"required DEV protocol source is missing: {source}")
        destination = run_dir / "source_snapshot" / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if _sha256(source) != _sha256(destination):
            raise OSError(f"DEV source snapshot hash mismatch for {source}")
        hashes[str(destination.relative_to(run_dir))] = _sha256(destination)
    return hashes


def _execute_dev(
    config: dict[str, Any],
    *,
    run_dir: Path,
    dtype: jnp.dtype,
    started_at_utc: str,
    source_snapshot_hashes: dict[str, str],
) -> dict[str, Any]:
    dynamics = config["dynamics"]
    anchors = config["anchors"]
    network = config["costate_network"]
    sampling = config["sampling"]
    replicate_ids = tuple(config["experiment"]["dev_replicates"])
    query_budget = int(sampling["reward_query_budget"])
    state_minimum, state_maximum = map(float, sampling["state_range"])
    common_sampling = {
        "reward_query_budget": query_budget,
        "sigma": float(dynamics["sigma"]),
        "horizon": float(dynamics["horizon"]),
        "threshold": float(dynamics["threshold"]),
        "time_steps": int(dynamics["steps"]),
        "minimum_remaining_steps": int(anchors["minimum_remaining_steps"]),
        "state_minimum": state_minimum,
        "state_maximum": state_maximum,
        "dtype": dtype,
    }
    master = jax.random.PRNGKey(int(config["experiment"]["seed"]))
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    git_state = _git_state(root)
    dev_contract = _dev_contract(config)
    eligible_to_lock = bool(
        dev_contract["matches_frozen_dev_contract"]
        and git_state["revision"]
        and git_state["dirty"] is False
    )
    terminal_status = SUCCESS_STATUS if eligible_to_lock else SMOKE_STATUS
    protocol_sha256 = _sha256(root / "docs" / "mam_gate_a_v2_protocol.md")

    data_started = time.perf_counter()
    datasets: dict[int, dict[SampledArm, QueryMatchedDataset]] = {}
    dev_contexts: dict[int, DevContexts] = {}
    for replicate in replicate_ids:
        replicate_datasets: dict[SampledArm, QueryMatchedDataset] = {}
        for arm in DIRECT_ARMS:
            second_key = (
                _stream_key(master, "independent_second_noise", replicate, arm)
                if arm == "IID2"
                else None
            )
            replicate_datasets[arm] = sample_query_matched_dataset(
                _stream_key(master, "training_contexts", replicate, arm),
                _stream_key(master, "base_noise", replicate, arm),
                arm,
                independent_second_noise_key=second_key,
                **common_sampling,
            )
        datasets[replicate] = replicate_datasets
        dev_contexts[replicate] = _sample_dev_contexts(
            _stream_key(master, "validation_contexts", replicate, "SHARED"),
            int(sampling["dev_contexts"]),
            sigma=float(dynamics["sigma"]),
            horizon=float(dynamics["horizon"]),
            threshold=float(dynamics["threshold"]),
            time_steps=int(dynamics["steps"]),
            minimum_remaining_steps=int(anchors["minimum_remaining_steps"]),
            state_minimum=state_minimum,
            state_maximum=state_maximum,
            dtype=dtype,
        )
    data_seconds = time.perf_counter() - data_started

    initialization_started = time.perf_counter()
    factory = _make_factory(config)
    common_initial = {
        replicate: _initial_params(config, dtype, replicate)[1] for replicate in replicate_ids
    }
    initial_hashes = {
        str(replicate): _tree_hash(common_initial[replicate]) for replicate in replicate_ids
    }
    initialization_seconds = time.perf_counter() - initialization_started

    schedule_started = time.perf_counter()
    training_steps = int(network["training_steps"])
    batch_size = int(network["batch_size"])
    schedules: dict[str, dict[int, jax.Array]] = {arm: {} for arm in ALL_ARMS}
    for arm in ALL_ARMS:
        context_count = query_budget if arm in {"RAW1", "VALUE_CRITIC"} else query_budget // 2
        for replicate in replicate_ids:
            schedules[arm][replicate] = _minibatch_schedule(
                _stream_key(master, "minibatch_schedule", replicate, arm),
                dataset_size=context_count,
                training_steps=training_steps,
                batch_size=batch_size,
            )
    schedule_hashes = {
        arm: {
            str(replicate): _hash_named_arrays(schedule=schedules[arm][replicate])
            for replicate in replicate_ids
        }
        for arm in ALL_ARMS
    }
    schedule_seconds = time.perf_counter() - schedule_started

    selected_params: dict[str, dict[int, Any]] = {}
    selected: dict[str, Any] = {}
    candidates: dict[str, list[dict[str, Any]]] = {}
    training_rows: list[dict[str, Any]] = []
    arm_timings: dict[str, float] = {}
    learning_rates = [float(value) for value in network["learning_rates"]]
    for arm in ALL_ARMS:
        arm_started = time.perf_counter()
        arm_datasets = {
            replicate: datasets[replicate]["RAW1" if arm in {"RAW1", "VALUE_CRITIC"} else arm]
            for replicate in replicate_ids
        }
        arm_targets = {
            replicate: jax.lax.stop_gradient(
                arm_datasets[replicate].rewards[:, :1]
                if arm == "VALUE_CRITIC"
                else arm_datasets[replicate].supervised_targets
            )
            for replicate in replicate_ids
        }
        params, arm_selected, arm_candidates, arm_rows = _fit_arm_candidates(
            arm,
            factory=factory,
            initial_params=common_initial,
            datasets=arm_datasets,
            supervised_targets=arm_targets,
            schedules=schedules[arm],
            dev_contexts=dev_contexts,
            learning_rates=learning_rates,
            training_steps=training_steps,
            eval_every=int(network["eval_every"]),
            dtype=dtype,
        )
        selected_params[arm] = params
        selected[arm] = arm_selected
        candidates[arm] = arm_candidates
        training_rows.extend(arm_rows)
        arm_timings[arm] = time.perf_counter() - arm_started

    if not all(
        _tree_all_finite(selected_params[arm][replicate])
        for arm in ALL_ARMS
        for replicate in replicate_ids
    ):
        raise FloatingPointError("one or more selected DEV parameter trees is nonfinite")

    parameter_hashes = {
        arm: {
            str(replicate): _tree_hash(selected_params[arm][replicate])
            for replicate in replicate_ids
        }
        for arm in ALL_ARMS
    }
    checkpoint_metadata = {
        "protocol": "mam_gate_a_v2_dev",
        "status": terminal_status,
        "eligible_to_lock_for_confirm": eligible_to_lock,
        "dev_replicates": list(replicate_ids),
        "selected_learning_rates": {
            arm: selected[arm]["selected_learning_rate"] for arm in ALL_ARMS
        },
        "parameter_sha256": parameter_hashes,
        "config_sha256": _sha256(run_dir / "resolved_config.json"),
        "dev_contract_sha256": dev_contract["contract_sha256"],
        "protocol_sha256": protocol_sha256,
        "git_revision": git_state["revision"],
    }
    _save_checkpoint(run_dir / "dev_checkpoint.npz", selected_params, checkpoint_metadata)

    with (run_dir / "dev_training_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training_rows[0]))
        writer.writeheader()
        writer.writerows(training_rows)

    raw_cap = min(16_384, query_budget)
    half_cap = min(16_384, query_budget // 2)
    raw_arrays: dict[str, np.ndarray] = {}
    for replicate in replicate_ids:
        for arm in DIRECT_ARMS:
            dataset = datasets[replicate][arm]
            cap = raw_cap if arm == "RAW1" else half_cap
            prefix = f"replicate_{replicate}_{arm.lower()}"
            raw_arrays[f"{prefix}_state"] = np.asarray(dataset.states[:cap, 0])
            raw_arrays[f"{prefix}_time"] = np.asarray(dataset.times[:cap])
            raw_arrays[f"{prefix}_normals"] = np.asarray(dataset.suffix_normals[:cap, :, 0])
            raw_arrays[f"{prefix}_rewards"] = np.asarray(dataset.rewards[:cap])
            raw_arrays[f"{prefix}_label"] = np.asarray(dataset.supervised_targets[:cap, 0])
        dev = dev_contexts[replicate]
        raw_arrays[f"replicate_{replicate}_dev_state"] = np.asarray(dev.states[:, 0])
        raw_arrays[f"replicate_{replicate}_dev_time"] = np.asarray(dev.times)
        raw_arrays[f"replicate_{replicate}_dev_costate_truth"] = np.asarray(
            dev.analytic_costates[:, 0]
        )
        raw_arrays[f"replicate_{replicate}_dev_value_truth"] = np.asarray(dev.analytic_values[:, 0])
    np.savez_compressed(run_dir / "dev_raw_samples.npz", **raw_arrays)

    dataset_hashes = {
        arm: {
            str(replicate): _dataset_hash(
                datasets[replicate]["RAW1" if arm == "VALUE_CRITIC" else arm]
            )
            for replicate in replicate_ids
        }
        for arm in ALL_ARMS
    }
    logical_queries = {
        arm: {str(replicate): query_budget for replicate in replicate_ids} for arm in ALL_ARMS
    }
    query_accounting = {
        "configured_reward_query_budget_Q_per_arm_per_replicate": query_budget,
        "dev_replicates": list(replicate_ids),
        "logical_reward_queries_by_arm_and_replicate": logical_queries,
        "all_arm_replicates_have_exactly_Q_logical_queries": all(
            value == query_budget
            for arm_counts in logical_queries.values()
            for value in arm_counts.values()
        ),
        "total_logical_reward_queries": 4 * len(replicate_ids) * query_budget,
        "physical_unique_reward_values_generated": 3 * len(replicate_ids) * query_budget,
        "value_critic_reuses_raw1_rewards": True,
        "incremental_physical_queries_for_value_critic": 0,
        "dev_validation_reward_queries": 0,
        "clipping": False,
        "filtering": False,
    }
    budget = {
        "common_network_architecture": True,
        "common_initialization_across_arms_within_replicate": True,
        "common_initialization_sha256_by_replicate": initial_hashes,
        "learning_rate_candidates": learning_rates,
        "candidate_count_per_arm": len(learning_rates),
        "training_steps_per_candidate_per_replicate": training_steps,
        "batch_size_per_candidate": batch_size,
        "label_exposures_per_candidate_per_replicate": training_steps * batch_size,
        "context_counts_per_replicate": {
            "RAW1": query_budget,
            "IID2": query_budget // 2,
            "ANTITHETIC2": query_budget // 2,
            "VALUE_CRITIC": query_budget,
        },
        "dataset_sha256": dataset_hashes,
        "schedule_sha256": schedule_hashes,
        "raw1_and_value_critic_share_dataset_by_replicate": {
            str(replicate): dataset_hashes["RAW1"][str(replicate)]
            == dataset_hashes["VALUE_CRITIC"][str(replicate)]
            for replicate in replicate_ids
        },
        "shared_dev_context_sha256_by_replicate": {
            str(replicate): _hash_named_arrays(
                states=dev_contexts[replicate].states,
                times=dev_contexts[replicate].times,
                analytic_costates=dev_contexts[replicate].analytic_costates,
                analytic_values=dev_contexts[replicate].analytic_values,
            )
            for replicate in replicate_ids
        },
        "dev_contexts_per_replicate": int(sampling["dev_contexts"]),
        "dev_used_for_training_loss": False,
    }
    target_tails = {
        arm: {
            str(replicate): _tail_summary(
                datasets[replicate]["RAW1"].rewards[:, :1]
                if arm == "VALUE_CRITIC"
                else datasets[replicate][arm].supervised_targets
            )
            for replicate in replicate_ids
        }
        for arm in ALL_ARMS
    }
    results = {
        "status": terminal_status,
        "protocol": "mam_gate_a_v2_dev",
        "eligible_for_scientific_status": False,
        "eligible_to_lock_for_confirm": eligible_to_lock,
        "dev_contract": dev_contract,
        "protocol_sha256": protocol_sha256,
        "git": git_state,
        "claim_scope": "three-replicate hyperparameter development only",
        "mathematical_targets": {
            "raw1": "R*Z/(sigma*sqrt(T-t))",
            "iid2": "mean_m R_m*Z_m/(sigma*sqrt(T-t)), independent Z_m",
            "antithetic2": "mean over Z and -Z of R(Z)*Z/(sigma*sqrt(T-t))",
            "value_critic": (
                "stable sigmoid cross-entropy on stopped R; evaluate grad_x sigmoid(logit)"
            ),
            "costate_truth_used_in_training_loss": False,
        },
        "selection": selected,
        "candidates": candidates,
        "training_target_tails": target_tails,
        "query_accounting": query_accounting,
        "matched_budget": budget,
        "checkpoint": {
            "path": str(run_dir / "dev_checkpoint.npz"),
            "parameter_sha256": parameter_hashes,
        },
        "timing_seconds": {
            "dataset_and_dev_generation": data_seconds,
            "network_initialization": initialization_seconds,
            "schedule_generation": schedule_seconds,
            "training_by_arm": arm_timings,
            "total": time.perf_counter() - started,
        },
        "output_dir": str(run_dir),
    }
    _write_json(run_dir / "dev_results.json", results)

    artifact_names = (
        "resolved_config.json",
        "dev_results.json",
        "dev_training_metrics.csv",
        "dev_raw_samples.npz",
        "dev_checkpoint.npz",
    )
    manifest = {
        "status": terminal_status,
        "protocol": "mam_gate_a_v2_dev",
        "eligible_for_scientific_status": False,
        "eligible_to_lock_for_confirm": eligible_to_lock,
        "dev_contract": dev_contract,
        "protocol_sha256": protocol_sha256,
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "git": git_state,
        "config_sha256": _sha256(run_dir / "resolved_config.json"),
        "source_sha256": {
            path.removeprefix("source_snapshot/"): digest
            for path, digest in source_snapshot_hashes.items()
        },
        "artifact_sha256": {
            **{name: _sha256(run_dir / name) for name in artifact_names},
            **source_snapshot_hashes,
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
        "random_streams": {
            "derivation": (
                "fold_in(fold_in(fold_in(PRNGKey(seed), domain_id), replicate_id), arm_id)"
            ),
            "domain_ids": DEV_STREAM_IDS,
            "arm_ids": _ARM_STREAM_IDS,
        },
        "query_accounting": query_accounting,
        "matched_budget": budget,
        "selection_context_only": "shared_analytic_DEV_by_replicate",
    }
    _write_json(run_dir / "dev_run_manifest.json", manifest)
    return results


def run_dev(
    config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run query-matched Gate-A-v2 development and return its JSON summary."""
    config = json.loads(json.dumps(config))
    if output_dir is not None:
        config["experiment"]["output_dir"] = str(output_dir)
    if seed is not None:
        config["experiment"]["seed"] = int(seed)
    _validate_config(config)
    run_dir = Path(config["experiment"]["output_dir"])
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output already exists: {run_dir}; pass --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at_utc = datetime.now(UTC).isoformat()
    running_manifest = {
        "status": "RUNNING_MAM_GATE_A_V2_DEV_INCOMPLETE",
        "protocol": "mam_gate_a_v2_dev",
        "eligible_for_scientific_status": False,
        "started_at_utc": started_at_utc,
        "output_dir": str(run_dir),
    }
    _write_json(run_dir / "dev_run_manifest.json", running_manifest)
    source_snapshot_hashes: dict[str, str] = {}
    try:
        dtype = configure_numerics(config)
        _write_json(run_dir / "resolved_config.json", config)
        source_snapshot_hashes = _snapshot_source(run_dir)
        return _execute_dev(
            config,
            run_dir=run_dir,
            dtype=dtype,
            started_at_utc=started_at_utc,
            source_snapshot_hashes=source_snapshot_hashes,
        )
    except Exception as error:
        failure = {
            **running_manifest,
            "status": "FAILED_MAM_GATE_A_V2_DEV_EXCEPTION",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "exception": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "source_snapshot_sha256": source_snapshot_hashes,
            "stale_artifacts_may_exist": True,
        }
        _write_json(run_dir / "dev_run_manifest.json", failure)
        raise


__all__ = [
    "ALL_ARMS",
    "DEV_STREAM_IDS",
    "DIRECT_ARMS",
    "QueryMatchedDataset",
    "SMOKE_STATUS",
    "SUCCESS_STATUS",
    "configure_numerics",
    "critic_costate",
    "hard_bel_from_noise",
    "hard_threshold_costate",
    "hard_threshold_value",
    "hard_bel_from_noise",
    "load_config",
    "load_dev_checkpoint",
    "run_dev",
    "sample_query_matched_dataset",
]
