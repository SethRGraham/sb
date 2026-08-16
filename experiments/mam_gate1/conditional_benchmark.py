r"""Gate-1 conditional benchmark for arrival-correct MAM.

The benchmark freezes the same endpoint-pinned chain as the MAM solver.  For
``n=0,...,N-2`` it uses

.. math::

   X_{n+1}=\rho_n X_n+(1-\rho_n)y
       +\Gamma_n(\sqrt{\Delta t}u_n+\xi_n),\qquad
   \Gamma_n=\sqrt{\Delta t\rho_n}\Sigma,

and measures the objective

.. math::

   \Delta t\sum_{n=0}^{N-2}
      \left(\tfrac12\|u_n\|^2+\ell_{n+1}(X_{n+1})\right).

Four deliberately distinct action estimators are evaluated on untouched
query states:

* ``mam_corrected`` uses the learned matrix-free MAM continuation costate and
  the exact one-step antithetic arrival correction;
* ``direct_full_return`` is the tangent-free antithetic score of the complete
  suffix return under the same frozen policy;
* ``critic_autodiff`` differentiates a scalar critic trained on a separate
  fixed-policy cache, then adds the same arrival correction; and
* ``path_integral`` is the self-normalized desirability-control estimator.

The first three are current-policy action targets and are evaluated against an
independent replicated high-sample direct score for that *same* frozen policy.
Path integral instead targets the KL-relaxed path-measure/desirability problem;
its finite-grid value is not claimed to be the optimum of the frozen Gaussian
mean-shift policy class.  It is compared only with an independent high-sample
path-integral reference.  The schema records these distinct estimands.

For an additional control diagnostic, smooth LQG policy actions are compared
with an exact finite-grid Riccati recursion.  No analogous discrete optimum is
claimed for the discontinuous disk task.  Every Monte Carlo reference reports
replicate standard error, and path integral also reports ESS.  No clipping,
filtering, or differentiable obstacle surrogate is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import yaml  # type: ignore[import-untyped]

from schrodinger_bridge.core.problem import BrownianMotion, GaussianDistribution, SBProblem
from schrodinger_bridge.core.types import Scalar, TimeGrid
from schrodinger_bridge.solvers.malliavin_adjoint import (
    MalliavinAdjointConfig,
    MalliavinAdjointInnerSolver,
    ValueOnlyCost,
    assemble_antithetic_direct_action_score,
    assemble_pinned_actor_targets,
    simulate_pinned_brownian_rollout_matrix_free,
)
from schrodinger_bridge.solvers.mam_acceptance import paired_objective_statistics
from schrodinger_bridge.solvers.mam_bridge import (
    ConditionalMAMConfig,
    EndpointPairBatch,
    MAMConditionalSolver,
    MAMExecutionConfig,
)
from schrodinger_bridge.solvers.mam_fields import MAMFieldConfig
from schrodinger_bridge.solvers.mam_path_integral import (
    PathIntegralConfig,
    estimate_pinned_path_integral_control,
    simulate_pinned_reference_suffix,
)
from schrodinger_bridge.solvers.mam_value_critic import ValueCriticConfig

FULL_GATE1_SEEDS = (31_415, 27_182, 16_180, 14_142, 17_320)
PROTOCOL = "mam_gate1_conditional_v1"
SCHEMA_VERSION = 1
SMOKE_STATUS = "COMPLETE_MAM_GATE1_SMOKE_NOT_SCIENTIFIC_EVIDENCE"
FULL_PASS_STATUS = "PASS_MAM_GATE1_CONDITIONAL"
FULL_FAIL_STATUS = "FAIL_MAM_GATE1_CONDITIONAL"
FULL_INELIGIBLE_STATUS = "INELIGIBLE_MAM_GATE1_FULL_UNCLEAN_OR_UNLOCKED"
REQUIRED_ARMS = (
    "mam_corrected",
    "direct_full_return",
    "critic_autodiff",
    "path_integral",
)
UNAVAILABLE_PLANNED_ARMS = {
    "joint_value_gradient_regression": (
        "no joint scalar-value/vector-gradient training API exists in the standalone sb "
        "repository; relabeling the scalar critic would be scientifically false"
    ),
    "smooth_hamiltonian_adjoint_matching": (
        "no separate smooth pathwise Hamiltonian-AM baseline is exposed; the benchmark "
        "will not differentiate the value-only oracle and call it that baseline"
    ),
}

# Every stochastic object has a stable domain.  The warm-up stream is outside
# the locked evaluation domains and is never included in scientific results.
STREAM_IDS = {
    "training_pairs": 10_001,
    "conditional_solver": 10_002,
    "evaluation_pairs": 10_003,
    "evaluation_rollout": 10_004,
    "mam_target": 10_005,
    "mam_final_policy_costate": 10_013,
    "critic_cache": 10_006,
    "critic_target": 10_007,
    "direct_target": 10_008,
    "target_reference": 10_014,
    "path_integral": 10_009,
    "path_integral_reference": 10_010,
    "objective_pairs": 10_011,
    "objective_noise": 10_012,
    "warmup": 90_000,
}


@dataclass(frozen=True)
class LQGPolicy:
    """Exact affine optimal noise-control policy on one pinned grid.

    Shapes are ``x_matrix, endpoint_matrix: [S,d,d]`` and
    ``bias: [S,d]``, where ``S=N-1`` is the number of stochastic transitions.
    The action is ``x_matrix[n] @ x + endpoint_matrix[n] @ y + bias[n]``.
    """

    x_matrix: jax.Array
    endpoint_matrix: jax.Array
    bias: jax.Array

    def evaluate(self, state: jax.Array, endpoint: jax.Array, anchor: jax.Array) -> jax.Array:
        state = jnp.asarray(state)
        endpoint = jnp.asarray(endpoint, dtype=state.dtype)
        anchor = jnp.asarray(anchor, dtype=jnp.int32)
        return (
            jnp.einsum("bij,bj->bi", self.x_matrix[anchor], state)
            + jnp.einsum("bij,bj->bi", self.endpoint_matrix[anchor], endpoint)
            + self.bias[anchor]
        )


class EvaluationQueries(NamedTuple):
    states: jax.Array
    times: jax.Array
    next_times: jax.Array
    endpoints: jax.Array
    anchors: jax.Array
    dataset_sha256: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(*arrays: jax.Array) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        host = np.ascontiguousarray(np.asarray(jax.device_get(array)))
        digest.update(str(host.dtype).encode("ascii"))
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(host.tobytes())
    return digest.hexdigest()


def _stream_key(seed: int, domain: str, *, task_index: int, replicate: int = 0) -> jax.Array:
    if domain not in STREAM_IDS:
        raise ValueError(f"unknown Gate-1 RNG domain: {domain}")
    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, STREAM_IDS[domain])
    key = jax.random.fold_in(key, task_index)
    return jax.random.fold_in(key, replicate)


def _require_exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"invalid {name} keys; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return cast(dict[str, Any], value)


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _finite_float(
    value: Any, name: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


_TOP_KEYS = {"experiment", "numerics", "problem", "conditional", "evaluation", "tasks", "gates"}
_SECTION_KEYS = {
    "experiment": {
        "protocol",
        "profile",
        "seeds",
        "output_dir",
        "intended_for_scientific_evidence",
        "warmup",
    },
    "numerics": {"compute_dtype", "matmul_precision"},
    "problem": {
        "horizon",
        "steps",
        "sigma",
        "source_mean",
        "target_mean",
        "marginal_variance",
        "pair_batch_size",
    },
    "conditional": {
        "costate_steps",
        "policy_iterations",
        "maximum_consecutive_rejections",
        "microbatch_size",
        "effective_batch_size",
        "acceptance_size",
        "direct_score_size",
        "actor_model",
        "actor_training_steps",
        "critic_training_steps",
        "hidden_sizes",
        "time_embedding_dim",
        "learning_rate",
        "actor_ridge",
        "line_search",
        "one_sided_z",
        "improvement_tolerance",
    },
    "evaluation": {
        "query_count",
        "objective_sample_size",
        "mam_antithetic_pairs",
        "direct_antithetic_pairs",
        "reference_direct_antithetic_pairs",
        "reference_direct_replicates",
        "path_integral_pairs",
        "reference_path_integral_pairs",
        "reference_replicates",
        "path_integral_shard_size",
        "minimum_ess_fraction",
        "boundary_bandwidth",
        "minimum_boundary_effective_queries",
    },
    "tasks": {"smooth_lqg", "hard_disk"},
    "gates": {
        "smooth_relative_l2",
        "smooth_cosine",
        "hard_boundary_relative_l2",
        "hard_boundary_cosine",
        "maximum_target_reference_relative_se",
        "maximum_path_integral_reference_relative_se",
        "maximum_target_reference_split_relative_l2",
        "maximum_path_integral_reference_split_relative_l2",
        "minimum_accepting_seeds",
        "required_seeds",
        "require_all_reference_queries_usable",
    },
}
_SMOOTH_KEYS = {"quadratic_weight", "center"}
_HARD_KEYS = {"disk_center", "disk_radius", "occupancy_penalty"}


def validate_config(config: dict[str, Any]) -> None:
    """Validate the closed benchmark schema before sampling or writing."""
    _require_exact_keys(config, _TOP_KEYS, "config")
    for section, keys in _SECTION_KEYS.items():
        _require_exact_keys(config[section], keys, section)
    _require_exact_keys(config["tasks"]["smooth_lqg"], _SMOOTH_KEYS, "tasks.smooth_lqg")
    _require_exact_keys(config["tasks"]["hard_disk"], _HARD_KEYS, "tasks.hard_disk")

    experiment = config["experiment"]
    if experiment["protocol"] != PROTOCOL:
        raise ValueError(f"experiment.protocol must be {PROTOCOL!r}")
    if experiment["profile"] not in {"smoke", "full"}:
        raise ValueError("experiment.profile must be 'smoke' or 'full'")
    if not isinstance(experiment["seeds"], list) or not experiment["seeds"]:
        raise ValueError("experiment.seeds must be a nonempty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in experiment["seeds"]):
        raise TypeError("every experiment seed must be an integer")
    if len(set(experiment["seeds"])) != len(experiment["seeds"]):
        raise ValueError("experiment seeds must be unique")
    if not isinstance(experiment["output_dir"], str) or not experiment["output_dir"].strip():
        raise ValueError("experiment.output_dir must be a nonempty path")
    if not isinstance(experiment["warmup"], bool):
        raise TypeError("experiment.warmup must be boolean")
    if not isinstance(experiment["intended_for_scientific_evidence"], bool):
        raise TypeError("intended_for_scientific_evidence must be boolean")
    if experiment["profile"] == "smoke" and experiment["intended_for_scientific_evidence"]:
        raise ValueError("a smoke run cannot be intended for scientific evidence")
    if experiment["profile"] == "full":
        if tuple(experiment["seeds"]) != FULL_GATE1_SEEDS:
            raise ValueError(f"full Gate-1 seeds must be exactly {list(FULL_GATE1_SEEDS)}")
        if experiment["intended_for_scientific_evidence"] is not True:
            raise ValueError("the full profile must declare scientific-evidence intent")

    numerics = config["numerics"]
    if numerics["compute_dtype"] != "float32":
        raise ValueError("MAMBridgeSolver production Gate 1 is fixed to float32")
    if numerics["matmul_precision"] not in {"default", "high", "highest"}:
        raise ValueError("unsupported JAX matmul precision")

    problem = config["problem"]
    _finite_float(problem["horizon"], "problem.horizon", positive=True)
    _positive_int(problem["steps"], "problem.steps", minimum=3)
    _finite_float(problem["sigma"], "problem.sigma", positive=True)
    for name in ("source_mean", "target_mean"):
        value = problem[name]
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"problem.{name} must be a two-vector")
        for index, scalar in enumerate(value):
            _finite_float(scalar, f"problem.{name}[{index}]")
    _finite_float(problem["marginal_variance"], "problem.marginal_variance", positive=True)
    _positive_int(problem["pair_batch_size"], "problem.pair_batch_size", minimum=2)

    conditional = config["conditional"]
    for name, minimum in (
        ("costate_steps", 1),
        ("policy_iterations", 1),
        ("maximum_consecutive_rejections", 1),
        ("microbatch_size", 2),
        ("effective_batch_size", 2),
        ("acceptance_size", 2),
        ("direct_score_size", 2),
        ("actor_training_steps", 1),
        ("critic_training_steps", 1),
        ("time_embedding_dim", 2),
    ):
        _positive_int(conditional[name], f"conditional.{name}", minimum)
    if conditional["effective_batch_size"] % conditional["microbatch_size"]:
        raise ValueError("effective_batch_size must be divisible by microbatch_size")
    if conditional["actor_model"] not in {"nonlinear", "affine_reference"}:
        raise ValueError("conditional.actor_model is invalid")
    if not isinstance(conditional["hidden_sizes"], list) or not conditional["hidden_sizes"]:
        raise ValueError("conditional.hidden_sizes must be nonempty")
    for width in conditional["hidden_sizes"]:
        _positive_int(width, "conditional.hidden_sizes entry")
    for name in ("learning_rate", "actor_ridge", "one_sided_z"):
        _finite_float(conditional[name], f"conditional.{name}", positive=True)
    _finite_float(
        conditional["improvement_tolerance"],
        "conditional.improvement_tolerance",
        nonnegative=True,
    )
    if not isinstance(conditional["line_search"], list) or not conditional["line_search"]:
        raise ValueError("conditional.line_search must be nonempty")
    line_search = [
        _finite_float(value, "conditional.line_search entry", positive=True)
        for value in conditional["line_search"]
    ]
    if any(value > 1.0 for value in line_search) or line_search != sorted(
        line_search, reverse=True
    ):
        raise ValueError("line_search must be descending and lie in (0,1]")

    evaluation = config["evaluation"]
    for name, minimum in (
        ("query_count", 2),
        ("objective_sample_size", 2),
        ("mam_antithetic_pairs", 1),
        ("direct_antithetic_pairs", 1),
        ("reference_direct_antithetic_pairs", 2),
        ("reference_direct_replicates", 2),
        ("path_integral_pairs", 1),
        ("reference_path_integral_pairs", 2),
        ("reference_replicates", 2),
        ("path_integral_shard_size", 1),
        ("minimum_boundary_effective_queries", 1),
    ):
        _positive_int(evaluation[name], f"evaluation.{name}", minimum)
    _finite_float(
        evaluation["minimum_ess_fraction"],
        "evaluation.minimum_ess_fraction",
        positive=True,
    )
    if evaluation["minimum_ess_fraction"] > 1.0:
        raise ValueError("minimum_ess_fraction must not exceed one")
    _finite_float(evaluation["boundary_bandwidth"], "evaluation.boundary_bandwidth", positive=True)

    smooth = config["tasks"]["smooth_lqg"]
    _finite_float(smooth["quadratic_weight"], "smooth.quadratic_weight", positive=True)
    hard = config["tasks"]["hard_disk"]
    _finite_float(hard["disk_radius"], "hard.disk_radius", positive=True)
    _finite_float(hard["occupancy_penalty"], "hard.occupancy_penalty", positive=True)
    for name, value in (
        ("smooth.center", smooth["center"]),
        ("hard.disk_center", hard["disk_center"]),
    ):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{name} must be a two-vector")
        for scalar in value:
            _finite_float(scalar, name)

    gates = config["gates"]
    for name in (
        "smooth_relative_l2",
        "smooth_cosine",
        "hard_boundary_relative_l2",
        "hard_boundary_cosine",
        "maximum_target_reference_relative_se",
        "maximum_path_integral_reference_relative_se",
        "maximum_target_reference_split_relative_l2",
        "maximum_path_integral_reference_split_relative_l2",
    ):
        _finite_float(gates[name], f"gates.{name}", nonnegative=True)
    _positive_int(gates["minimum_accepting_seeds"], "gates.minimum_accepting_seeds")
    _positive_int(gates["required_seeds"], "gates.required_seeds")
    if not isinstance(gates["require_all_reference_queries_usable"], bool):
        raise TypeError("gates.require_all_reference_queries_usable must be boolean")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a benchmark YAML file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    validate_config(value)
    return cast(dict[str, Any], value)


def _contract_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads(json.dumps(config)))
    payload["experiment"].pop("output_dir")
    return payload


def _full_contract_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "conditional_full.yaml"


def full_contract_report(config: dict[str, Any]) -> dict[str, Any]:
    """Compare a run with the checked-in full contract without self-reference."""
    expected = load_config(_full_contract_path())
    expected_payload = _contract_payload(expected)
    actual_payload = _contract_payload(config)
    return {
        "matches_locked_full_contract": actual_payload == expected_payload,
        "contract_sha256": _sha256_json(expected_payload),
        "observed_sha256": _sha256_json(actual_payload),
        "operational_fields_excluded": ["experiment.output_dir"],
    }


def solve_discrete_lqg_policy(
    times: jax.Array,
    diffusion: jax.Array | float,
    quadratic_weight: float,
    center: jax.Array,
) -> LQGPolicy:
    """Solve the exact discrete pinned LQG recursion.

    This is an exact reference for the declared finite chain, up to floating
    point linear solves.  It is not a continuous-time approximation.  The
    arrival cost is ``0.5 * quadratic_weight * ||x-center||^2``.
    """
    times = jnp.asarray(times)
    if times.ndim != 1 or times.shape[0] < 4:
        raise ValueError("LQG reference requires at least three time steps")
    if not jnp.issubdtype(times.dtype, jnp.floating):
        raise TypeError("times must have floating dtype")
    host_times = np.asarray(jax.device_get(times))
    increments = np.diff(host_times)
    if not np.all(np.isfinite(host_times)) or not np.all(increments > 0.0):
        raise ValueError("times must be finite and strictly increasing")
    if not np.allclose(increments, increments[0], rtol=1e-6, atol=1e-7):
        raise ValueError("LQG reference requires a uniform grid")
    center = jnp.asarray(center, dtype=times.dtype)
    if center.ndim != 1 or center.shape[0] < 1:
        raise ValueError("center must have shape [d]")
    dim = center.shape[0]
    sigma = jnp.asarray(diffusion, dtype=times.dtype)
    if sigma.ndim == 0:
        sigma = sigma * jnp.eye(dim, dtype=times.dtype)
    elif sigma.ndim == 1 and sigma.shape == (dim,):
        sigma = jnp.diag(sigma)
    if sigma.shape != (dim, dim):
        raise ValueError("diffusion must be scalar, diagonal [d], or square [d,d]")
    if not bool(jax.device_get(jnp.all(jnp.isfinite(sigma)))):
        raise ValueError("diffusion must be finite")
    if np.linalg.matrix_rank(np.asarray(jax.device_get(sigma))) != dim:
        raise ValueError("diffusion must be full rank")
    q = _finite_float(quadratic_weight, "quadratic_weight", positive=True)
    dt = times[1] - times[0]
    terminal = times[-1]
    stochastic_steps = times.shape[0] - 2
    p_next = jnp.zeros((dim, dim), dtype=times.dtype)
    endpoint_linear_next = jnp.zeros((dim, dim), dtype=times.dtype)
    r_next = jnp.zeros((dim,), dtype=times.dtype)
    x_matrices: list[jax.Array] = [jnp.zeros_like(p_next) for _ in range(stochastic_steps)]
    endpoint_matrices: list[jax.Array] = [jnp.zeros_like(p_next) for _ in range(stochastic_steps)]
    biases: list[jax.Array] = [jnp.zeros_like(r_next) for _ in range(stochastic_steps)]
    eye = jnp.eye(dim, dtype=times.dtype)
    arrival_q = dt * jnp.asarray(q, dtype=times.dtype) * eye
    arrival_r = -dt * jnp.asarray(q, dtype=times.dtype) * center

    for n in reversed(range(stochastic_steps)):
        rho = (terminal - times[n + 1]) / (terminal - times[n])
        a_matrix = rho * eye
        endpoint_map = (1.0 - rho) * eye
        control_map = dt * jnp.sqrt(rho) * sigma
        total_q = p_next + arrival_q
        total_endpoint_linear = endpoint_linear_next
        total_r = r_next + arrival_r
        control_hessian = dt * eye + control_map.T @ total_q @ control_map
        solve_left = jnp.linalg.solve(control_hessian, control_map.T)
        x_policy = -solve_left @ total_q @ a_matrix
        endpoint_policy = -solve_left @ (total_q @ endpoint_map + total_endpoint_linear)
        bias = -solve_left @ total_r
        x_matrices[n] = x_policy
        endpoint_matrices[n] = endpoint_policy
        biases[n] = bias

        projected = total_q @ control_map @ solve_left
        endpoint_linear_at_arrival = total_q @ endpoint_map + total_endpoint_linear
        p_current = a_matrix.T @ (total_q - projected @ total_q) @ a_matrix
        endpoint_linear_current = a_matrix.T @ (
            endpoint_linear_at_arrival - projected @ endpoint_linear_at_arrival
        )
        r_current = a_matrix.T @ (total_r - projected @ total_r)
        p_next = 0.5 * (p_current + p_current.T)
        endpoint_linear_next = endpoint_linear_current
        r_next = r_current

    return LQGPolicy(
        x_matrix=jnp.stack(x_matrices),
        endpoint_matrix=jnp.stack(endpoint_matrices),
        bias=jnp.stack(biases),
    )


def _make_problem(config: dict[str, Any]) -> SBProblem:
    problem = config["problem"]
    dtype = jnp.float32
    source_mean = jnp.asarray(problem["source_mean"], dtype=dtype)
    target_mean = jnp.asarray(problem["target_mean"], dtype=dtype)
    return SBProblem(
        reference=BrownianMotion(sigma=float(problem["sigma"]), dim=2),
        source=GaussianDistribution(
            mean=source_mean,
            cov=float(problem["marginal_variance"]),
            dim=2,
        ),
        target=GaussianDistribution(
            mean=target_mean,
            cov=float(problem["marginal_variance"]),
            dim=2,
        ),
        time_grid=TimeGrid(
            t0=0.0,
            t1=float(problem["horizon"]),
            num_steps=int(problem["steps"]),
        ),
        name="MAM Gate-1 conditional benchmark",
    )


def _brownian_sigma(problem: SBProblem) -> float:
    if type(problem.reference) is not BrownianMotion:
        raise TypeError("Gate 1 requires exact constant Brownian reference dynamics")
    return float(problem.reference.sigma)


def _running_cost(
    cost: ValueOnlyCost,
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    if cost.running_cost is None:
        raise ValueError("Gate-1 tasks require a running-cost callable")
    return cost.running_cost


def _make_cost(config: dict[str, Any], task: str) -> ValueOnlyCost:
    if task == "smooth_lqg":
        task_config = config["tasks"][task]
        center = jnp.asarray(task_config["center"], dtype=jnp.float32)
        weight = float(task_config["quadratic_weight"])

        def smooth_cost(x: jax.Array, time: jax.Array, context: jax.Array) -> jax.Array:
            del time, context
            return 0.5 * weight * jnp.sum((x - center) ** 2, axis=-1)

        return ValueOnlyCost(running_cost=smooth_cost, identifier="gate1_smooth_lqg")
    if task == "hard_disk":
        task_config = config["tasks"][task]
        center = jnp.asarray(task_config["disk_center"], dtype=jnp.float32)
        radius_squared = float(task_config["disk_radius"]) ** 2
        penalty = float(task_config["occupancy_penalty"])

        def hard_cost(x: jax.Array, time: jax.Array, context: jax.Array) -> jax.Array:
            del time, context
            occupied = jnp.sum((x - center) ** 2, axis=-1) <= radius_squared
            return penalty * occupied.astype(x.dtype)

        return ValueOnlyCost(running_cost=hard_cost, identifier="gate1_hard_binary_disk")
    raise ValueError(f"unknown Gate-1 task: {task}")


def _make_conditional_config(
    config: dict[str, Any],
) -> tuple[ConditionalMAMConfig, MAMExecutionConfig]:
    value = config["conditional"]
    hidden = tuple(int(width) for width in value["hidden_sizes"])
    microbatch = int(value["microbatch_size"])
    effective = int(value["effective_batch_size"])
    costate = MalliavinAdjointConfig(
        hidden_dims=hidden,
        time_embed_dim=int(value["time_embedding_dim"]),
        learning_rate=float(value["learning_rate"]),
        training_steps=int(value["costate_steps"]),
        batch_size=microbatch,
        minimum_remaining_steps=1,
        anchor_sampling="stratified",
        include_control_energy=True,
        matrix_free_labels=True,
        center_running_values=True,
    )
    critic = ValueCriticConfig(
        hidden_dims=hidden,
        time_embed_dim=int(value["time_embedding_dim"]),
        learning_rate=float(value["learning_rate"]),
        training_steps=int(value["critic_training_steps"]),
        microbatch_size=microbatch,
        effective_batch_size=effective,
    )
    actor_field = MAMFieldConfig(
        hidden_dims=hidden,
        time_embed_dim=int(value["time_embedding_dim"]),
        learning_rate=float(value["learning_rate"]),
        training_steps=int(value["actor_training_steps"]),
        microbatch_size=microbatch,
        effective_batch_size=effective,
    )
    conditional = ConditionalMAMConfig(
        costate=costate,
        costate_steps=int(value["costate_steps"]),
        batch_size=microbatch,
        value_critic=critic,
        train_value_critic=True,
        direct_score_diagnostic_size=int(value["direct_score_size"]),
        actor_model=str(value["actor_model"]),
        actor_field_config=actor_field,
        actor_ridge=float(value["actor_ridge"]),
        acceptance_size=int(value["acceptance_size"]),
        policy_iterations=int(value["policy_iterations"]),
        maximum_consecutive_rejections=int(value["maximum_consecutive_rejections"]),
        line_search=tuple(float(item) for item in value["line_search"]),
        one_sided_z=float(value["one_sided_z"]),
        improvement_tolerance=float(value["improvement_tolerance"]),
    )
    execution = MAMExecutionConfig(
        microbatch_size=microbatch,
        effective_batch_size=effective,
        allow_two_devices=False,
        production_dtype=jnp.float32,
    )
    return conditional, execution


def _sample_pairs(problem: SBProblem, key: jax.Array, size: int) -> EndpointPairBatch:
    source_key, target_key = jax.random.split(key)
    return EndpointPairBatch(
        source=jnp.asarray(problem.sample_source(source_key, size), dtype=jnp.float32),
        target=jnp.asarray(problem.sample_target(target_key, size), dtype=jnp.float32),
    )


def _zero_control(state: jax.Array, time: Scalar, endpoint: jax.Array) -> jax.Array:
    del time, endpoint
    return jnp.zeros_like(state)


def _make_evaluation_queries(
    problem: SBProblem,
    pair_key: jax.Array,
    rollout_key: jax.Array,
    size: int,
) -> EvaluationQueries:
    pairs = _sample_pairs(problem, pair_key, size)
    rollout = simulate_pinned_brownian_rollout_matrix_free(
        rollout_key,
        pairs.source,
        pairs.target,
        problem.time_grid.times,
        _brownian_sigma(problem),
        _zero_control,
    )
    stochastic_steps = problem.time_grid.num_steps - 1
    anchors = jnp.arange(size, dtype=jnp.int32) % stochastic_steps
    rows = jnp.arange(size, dtype=jnp.int32)
    states = rollout.states[rows, anchors]
    times = rollout.times[anchors]
    next_times = rollout.times[anchors + 1]
    digest = _array_sha256(states, times, next_times, pairs.target, anchors)
    return EvaluationQueries(states, times, next_times, pairs.target, anchors, digest)


def _evaluate_actor(actor: Any, queries: EvaluationQueries) -> jax.Array:
    """Evaluate a scalar-time actor on queries stratified over time."""
    output = jnp.zeros_like(queries.states)
    host_anchors = np.asarray(jax.device_get(queries.anchors))
    for anchor in range(int(host_anchors.max()) + 1):
        indices = np.flatnonzero(host_anchors == anchor)
        if indices.size == 0:
            continue
        index = jnp.asarray(indices, dtype=jnp.int32)
        prediction = actor(
            queries.states[index],
            queries.times[index][0],
            queries.endpoints[index],
        )
        output = output.at[index].set(prediction)
    return output


def _objective(rollout: Any, cost: ValueOnlyCost, dt: float) -> jax.Array:
    running = cost.running_values(rollout.states, rollout.times, rollout.context)
    potential = jnp.sum(running[:, 1:-1], axis=1)
    energy = 0.5 * jnp.sum(rollout.controls**2, axis=(1, 2))
    return jnp.asarray(dt, dtype=rollout.states.dtype) * (potential + energy)


def _heldout_objective_change(
    problem: SBProblem,
    cost: ValueOnlyCost,
    actor: Any,
    pair_key: jax.Array,
    noise_key: jax.Array,
    size: int,
    *,
    z_value: float,
    improvement_tolerance: float,
) -> dict[str, Any]:
    pairs = _sample_pairs(problem, pair_key, size)
    current = simulate_pinned_brownian_rollout_matrix_free(
        noise_key,
        pairs.source,
        pairs.target,
        problem.time_grid.times,
        _brownian_sigma(problem),
        _zero_control,
    )
    candidate = simulate_pinned_brownian_rollout_matrix_free(
        noise_key,
        pairs.source,
        pairs.target,
        problem.time_grid.times,
        _brownian_sigma(problem),
        actor,
    )
    stats = paired_objective_statistics(
        _objective(current, cost, problem.time_grid.dt),
        _objective(candidate, cost, problem.time_grid.dt),
        z_value=z_value,
        minimum_improvement=improvement_tolerance,
    )
    return {
        "comparison": "final_accepted_policy_minus_zero_policy",
        "stream": "untouched_paired_common_noise",
        "sample_count": int(jax.device_get(stats.sample_count)),
        "mean_delta": float(jax.device_get(stats.mean_delta)),
        "standard_error": float(jax.device_get(stats.standard_error)),
        "upper_confidence_bound": float(jax.device_get(stats.upper_confidence_bound)),
        "accepted": bool(jax.device_get(stats.accepted)),
        "confidence_method": "one_sided_normal_clt_approximation",
        "z_value": z_value,
        "minimum_improvement": improvement_tolerance,
    }


def _critic_ensemble_costate(critic: Any, result: Any) -> Any:
    """Differentiate both independently trained folds on untouched queries."""

    def costate(states: jax.Array, times: jax.Array, context: jax.Array) -> jax.Array:
        states = jnp.asarray(states)
        times = jnp.asarray(times, dtype=states.dtype)
        context = jnp.asarray(context, dtype=states.dtype)

        def one(params: Any, state: jax.Array, time: jax.Array, endpoint: jax.Array) -> jax.Array:
            def scalar(variable: jax.Array) -> jax.Array:
                inputs = jnp.concatenate([variable, endpoint])[None, :]
                output = critic.factory.forward(params, inputs, time[None])
                return jnp.asarray(output, dtype=variable.dtype)[0, 0]

            return jnp.asarray(jax.grad(scalar)(state))

        gradients = [
            jax.vmap(one, in_axes=(None, 0, 0, 0))(params, states, times, context)
            for params in result.params_by_training_fold
        ]
        value = 0.5 * (gradients[0] + gradients[1])
        return jax.lax.stop_gradient(value)

    return costate


def _direct_full_return_target(
    key: jax.Array,
    queries: EvaluationQueries,
    problem: SBProblem,
    cost: ValueOnlyCost,
    actor: Any,
    num_pairs: int,
) -> tuple[jax.Array, jax.Array, int]:
    """Estimate the current-policy full-return score, grouped by anchor."""
    result = jnp.zeros_like(queries.states)
    finite = jnp.zeros((queries.states.shape[0],), dtype=bool)
    oracle_queries = 0
    stochastic_steps = problem.time_grid.num_steps - 1
    for anchor in range(stochastic_steps):
        indices = np.flatnonzero(np.asarray(jax.device_get(queries.anchors)) == anchor)
        if indices.size == 0:
            continue
        index = jnp.asarray(indices, dtype=jnp.int32)
        suffix_key = jax.random.fold_in(key, anchor)
        samples = simulate_pinned_reference_suffix(
            suffix_key,
            queries.states[index],
            queries.endpoints[index],
            problem.time_grid.times[anchor:],
            _brownian_sigma(problem),
            _running_cost(cost),
            actor,
            config=PathIntegralConfig(
                num_samples=num_pairs,
                antithetic=True,
                minimum_ess_fraction=1e-12,
            ),
        )
        dt = jnp.asarray(problem.time_grid.dt, dtype=samples.states.dtype)
        returns = dt * jnp.sum(
            samples.running_values + 0.5 * jnp.sum(samples.reference_controls**2, axis=-1),
            axis=-1,
        )
        direct = assemble_antithetic_direct_action_score(
            returns[:, :num_pairs],
            returns[:, num_pairs:],
            samples.innovations[:, :num_pairs, 0, :],
            dt,
        )
        result = result.at[index].set(direct.target)
        finite = finite.at[index].set(direct.finite & samples.finite)
        oracle_queries += int(jax.device_get(samples.physical_query_count))
    return result, finite, oracle_queries


def _current_policy_target_reference(
    key: jax.Array,
    queries: EvaluationQueries,
    problem: SBProblem,
    cost: ValueOnlyCost,
    actor: Any,
    *,
    num_pairs: int,
    replicates: int,
    weights: jax.Array,
) -> tuple[jax.Array, jax.Array, dict[str, Any], int]:
    """Independent replicated reference for the current-policy estimand.

    Each replicate is a complete-suffix antithetic score average.  Unlike the
    path-integral ratio, this ordinary sample mean is unbiased in population
    under the declared fixed-policy score identity (subject to integrability).
    """
    estimates: list[jax.Array] = []
    valid_rows: list[jax.Array] = []
    oracle_queries = 0
    for replicate in range(replicates):
        estimate, finite, queries_used = _direct_full_return_target(
            jax.random.fold_in(key, replicate),
            queries,
            problem,
            cost,
            actor,
            num_pairs,
        )
        estimates.append(estimate)
        valid_rows.append(finite)
        oracle_queries += queries_used
    stacked = jnp.stack(estimates)
    all_valid = jnp.all(jnp.stack(valid_rows), axis=0)
    mean = jnp.mean(stacked, axis=0)
    standard_error = jnp.std(stacked, axis=0, ddof=1) / jnp.sqrt(replicates)
    precision = _reference_precision(stacked, mean, standard_error, weights, all_valid)
    return (
        mean,
        all_valid,
        {
            "kind": "independent_replicated_high_sample_direct_full_return_score",
            "estimand": "current_policy_action_target",
            "exact_in_population": True,
            "finite_sample_unbiased_under_integrability": True,
            "finite_sample_ratio_biased": False,
            "reference_replicates": replicates,
            "antithetic_pairs_per_replicate": num_pairs,
            "standard_error_rmse": float(jax.device_get(jnp.sqrt(jnp.mean(standard_error**2)))),
            **precision,
            "usable_fraction": float(jax.device_get(jnp.mean(all_valid.astype(jnp.float32)))),
        },
        oracle_queries,
    )


def _reference_precision(
    replicates: jax.Array,
    mean: jax.Array,
    standard_error: jax.Array,
    weights: jax.Array,
    valid: jax.Array,
) -> dict[str, float]:
    """Return boundary-aware precision and independent split stability."""
    weights = jnp.asarray(weights, dtype=mean.dtype)
    valid = jnp.asarray(valid, dtype=bool)
    safe_weights = jnp.where(valid, weights, 0.0)
    tiny = jnp.asarray(jnp.finfo(mean.dtype).tiny, dtype=mean.dtype)
    denominator = jnp.maximum(jnp.sum(safe_weights) * mean.shape[-1], tiny)
    mean_rms = jnp.sqrt(jnp.sum(safe_weights[:, None] * mean**2) / denominator)
    se_rms = jnp.sqrt(jnp.sum(safe_weights[:, None] * standard_error**2) / denominator)
    relative_se = se_rms / jnp.maximum(mean_rms, tiny)
    split = replicates.shape[0] // 2
    first = jnp.mean(replicates[:split], axis=0)
    second = jnp.mean(replicates[split:], axis=0)
    split_error = jnp.sum(safe_weights[:, None] * (first - second) ** 2)
    mean_energy = jnp.sum(safe_weights[:, None] * mean**2)
    split_relative = jnp.sqrt(split_error / jnp.maximum(mean_energy, tiny))
    return {
        "weighted_reference_rms": float(jax.device_get(mean_rms)),
        "weighted_standard_error_rms": float(jax.device_get(se_rms)),
        "weighted_relative_standard_error": float(jax.device_get(relative_se)),
        "split_half_relative_l2": float(jax.device_get(split_relative)),
    }


def _path_integral_target(
    key: jax.Array,
    queries: EvaluationQueries,
    problem: SBProblem,
    cost: ValueOnlyCost,
    *,
    num_pairs: int,
    minimum_ess_fraction: float,
    shard_size: int,
    replicate: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """Estimate optimal actions without materializing all queries at once."""
    result = jnp.zeros_like(queries.states)
    usable = jnp.zeros((queries.states.shape[0],), dtype=bool)
    ess_fraction = jnp.zeros((queries.states.shape[0],), dtype=queries.states.dtype)
    oracle_queries = 0
    host_anchors = np.asarray(jax.device_get(queries.anchors))
    stochastic_steps = problem.time_grid.num_steps - 1
    for anchor in range(stochastic_steps):
        anchor_indices = np.flatnonzero(host_anchors == anchor)
        for shard_index, offset in enumerate(range(0, anchor_indices.size, shard_size)):
            indices = anchor_indices[offset : offset + shard_size]
            if indices.size == 0:
                continue
            index = jnp.asarray(indices, dtype=jnp.int32)
            local_key = jax.random.fold_in(jax.random.fold_in(key, replicate), anchor)
            local_key = jax.random.fold_in(local_key, shard_index)
            samples, estimate = estimate_pinned_path_integral_control(
                local_key,
                queries.states[index],
                queries.endpoints[index],
                problem.time_grid.times[anchor:],
                _brownian_sigma(problem),
                _running_cost(cost),
                reference_control=None,
                config=PathIntegralConfig(
                    num_samples=num_pairs,
                    antithetic=True,
                    minimum_ess_fraction=minimum_ess_fraction,
                ),
            )
            result = result.at[index].set(estimate.raw_control_target)
            usable = usable.at[index].set(estimate.usable)
            ess_fraction = ess_fraction.at[index].set(estimate.ess_fraction)
            oracle_queries += int(jax.device_get(samples.physical_query_count))
    return result, usable, ess_fraction, oracle_queries


def _metric_sufficient_statistics(
    estimate: jax.Array,
    reference: jax.Array,
    weights: jax.Array,
    valid: jax.Array,
) -> dict[str, Any]:
    estimate = jnp.asarray(estimate)
    reference = jnp.asarray(reference, dtype=estimate.dtype)
    valid = jnp.asarray(valid, dtype=bool)
    weights = jnp.asarray(weights, dtype=estimate.dtype)
    finite = (
        valid
        & jnp.all(jnp.isfinite(estimate), axis=-1)
        & jnp.all(jnp.isfinite(reference), axis=-1)
        & jnp.isfinite(weights)
        & (weights >= 0.0)
    )
    safe_weights = jnp.where(finite, weights, 0.0)
    difference = jnp.where(finite[:, None], estimate - reference, 0.0)
    safe_estimate = jnp.where(finite[:, None], estimate, 0.0)
    safe_reference = jnp.where(finite[:, None], reference, 0.0)
    error_sq = jnp.sum(safe_weights[:, None] * difference**2)
    reference_sq = jnp.sum(safe_weights[:, None] * safe_reference**2)
    estimate_sq = jnp.sum(safe_weights[:, None] * safe_estimate**2)
    dot = jnp.sum(safe_weights[:, None] * safe_estimate * safe_reference)
    weight_sum = jnp.sum(safe_weights)
    valid_count = jnp.sum(finite)
    tiny = jnp.asarray(jnp.finfo(estimate.dtype).tiny, dtype=estimate.dtype)
    relative_l2 = jnp.sqrt(error_sq / jnp.maximum(reference_sq, tiny))
    rmse = jnp.sqrt(error_sq / jnp.maximum(weight_sum * estimate.shape[-1], tiny))
    cosine = dot / jnp.maximum(jnp.sqrt(estimate_sq * reference_sq), tiny)
    return {
        "rmse": float(jax.device_get(rmse)),
        "relative_l2": float(jax.device_get(relative_l2)),
        "cosine": float(jax.device_get(cosine)),
        "valid_count": int(jax.device_get(valid_count)),
        "total_count": int(estimate.shape[0]),
        "sufficient_statistics": {
            "weighted_squared_error_sum": float(jax.device_get(error_sq)),
            "weighted_reference_squared_sum": float(jax.device_get(reference_sq)),
            "weighted_estimate_squared_sum": float(jax.device_get(estimate_sq)),
            "weighted_dot_sum": float(jax.device_get(dot)),
            "weight_sum": float(jax.device_get(weight_sum)),
            "state_dimension": int(estimate.shape[-1]),
        },
    }


def _aggregate_metrics(seed_records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    stats = [record["arms"][arm]["metrics"]["sufficient_statistics"] for record in seed_records]
    return _aggregate_sufficient_statistics(stats)


def _aggregate_sufficient_statistics(stats: list[dict[str, Any]]) -> dict[str, Any]:
    error = sum(item["weighted_squared_error_sum"] for item in stats)
    reference = sum(item["weighted_reference_squared_sum"] for item in stats)
    estimate = sum(item["weighted_estimate_squared_sum"] for item in stats)
    dot = sum(item["weighted_dot_sum"] for item in stats)
    weight = sum(item["weight_sum"] for item in stats)
    dim = stats[0]["state_dimension"]
    tiny = np.finfo(np.float64).tiny
    return {
        "rmse": math.sqrt(error / max(weight * dim, tiny)),
        "relative_l2": math.sqrt(error / max(reference, tiny)),
        "cosine": dot / max(math.sqrt(estimate * reference), tiny),
        "seed_count": len(stats),
        "sufficient_statistics": {
            "weighted_squared_error_sum": error,
            "weighted_reference_squared_sum": reference,
            "weighted_estimate_squared_sum": estimate,
            "weighted_dot_sum": dot,
            "weight_sum": weight,
            "state_dimension": dim,
        },
    }


def _boundary_weights(config: dict[str, Any], states: jax.Array, task: str) -> jax.Array:
    if task != "hard_disk":
        return jnp.ones((states.shape[0],), dtype=states.dtype)
    hard = config["tasks"]["hard_disk"]
    center = jnp.asarray(hard["disk_center"], dtype=states.dtype)
    signed_distance = jnp.linalg.norm(states - center, axis=-1) - float(hard["disk_radius"])
    bandwidth = float(config["evaluation"]["boundary_bandwidth"])
    return jnp.exp(-0.5 * (signed_distance / bandwidth) ** 2)


def _discrete_optimal_reference(
    config: dict[str, Any],
    task: str,
    problem: SBProblem,
    cost: ValueOnlyCost,
    queries: EvaluationQueries,
    seed: int,
    task_index: int,
) -> tuple[jax.Array, jax.Array, dict[str, Any], int]:
    del seed, task_index, cost
    if task != "smooth_lqg":
        return (
            jnp.zeros_like(queries.states),
            jnp.zeros((queries.states.shape[0],), dtype=bool),
            {
                "available": False,
                "kind": None,
                "reason": (
                    "no trusted finite-grid optimal-control reference is implemented for "
                    "the discontinuous disk task"
                ),
                "usable_fraction": 0.0,
            },
            0,
        )
    smooth = config["tasks"]["smooth_lqg"]
    policy = solve_discrete_lqg_policy(
        jnp.asarray(problem.time_grid.times, dtype=jnp.float32),
        _brownian_sigma(problem),
        float(smooth["quadratic_weight"]),
        jnp.asarray(smooth["center"], dtype=jnp.float32),
    )
    action = policy.evaluate(queries.states, queries.endpoints, queries.anchors)
    return (
        action,
        jnp.ones((action.shape[0],), dtype=bool),
        {
            "available": True,
            "kind": "exact_discrete_lqg_riccati",
            "exact_for_declared_finite_chain": True,
            "finite_sample_biased": False,
            "standard_error_rmse": 0.0,
            "usable_fraction": 1.0,
        },
        0,
    )


def _path_integral_reference(
    config: dict[str, Any],
    problem: SBProblem,
    cost: ValueOnlyCost,
    queries: EvaluationQueries,
    seed: int,
    task_index: int,
    weights: jax.Array,
) -> tuple[jax.Array, jax.Array, dict[str, Any], int]:

    evaluation = config["evaluation"]
    replicates: list[jax.Array] = []
    replicate_usable: list[jax.Array] = []
    replicate_ess: list[jax.Array] = []
    query_count = 0
    base_key = _stream_key(seed, "path_integral_reference", task_index=task_index)
    for replicate in range(int(evaluation["reference_replicates"])):
        action, usable, ess, queries_used = _path_integral_target(
            base_key,
            queries,
            problem,
            cost,
            num_pairs=int(evaluation["reference_path_integral_pairs"]),
            minimum_ess_fraction=float(evaluation["minimum_ess_fraction"]),
            shard_size=int(evaluation["path_integral_shard_size"]),
            replicate=replicate,
        )
        replicates.append(action)
        replicate_usable.append(usable)
        replicate_ess.append(ess)
        query_count += queries_used
    stacked = jnp.stack(replicates)
    all_usable = jnp.all(jnp.stack(replicate_usable), axis=0)
    mean = jnp.mean(stacked, axis=0)
    standard_error = jnp.std(stacked, axis=0, ddof=1) / jnp.sqrt(len(replicates))
    ess = jnp.stack(replicate_ess)
    precision = _reference_precision(stacked, mean, standard_error, weights, all_usable)
    return (
        mean,
        all_usable,
        {
            "available": True,
            "kind": "independent_replicated_high_sample_path_integral_desirability_control",
            "estimand": "kl_relaxed_path_measure_desirability_control",
            "exact_for_declared_finite_chain": False,
            "exact_in_population_for_its_declared_path_integral_estimand": True,
            "not_a_discrete_mean_shift_optimality_reference": True,
            "finite_sample_biased": True,
            "reference_replicates": len(replicates),
            "antithetic_pairs_per_replicate": int(evaluation["reference_path_integral_pairs"]),
            "standard_error_rmse": float(jax.device_get(jnp.sqrt(jnp.mean(standard_error**2)))),
            **precision,
            "minimum_ess_fraction": float(jax.device_get(jnp.min(ess))),
            "median_ess_fraction": float(jax.device_get(jnp.median(ess))),
            "usable_fraction": float(jax.device_get(jnp.mean(all_usable.astype(jnp.float32)))),
        },
        query_count,
    )


def _run_seed_task(
    config: dict[str, Any],
    *,
    task: str,
    task_index: int,
    seed: int,
    warmup: bool = False,
) -> dict[str, Any]:
    problem = _make_problem(config)
    cost = _make_cost(config, task)
    conditional_config, execution = _make_conditional_config(config)
    pair_count = int(config["problem"]["pair_batch_size"])
    training_pairs = _sample_pairs(
        problem,
        _stream_key(seed, "training_pairs", task_index=task_index),
        pair_count,
    )
    solver = MAMConditionalSolver(problem, cost, conditional_config, execution)
    result = solver.solve(
        _stream_key(seed, "conditional_solver", task_index=task_index),
        training_pairs,
        "f",
    )
    jax.block_until_ready(result.paths)
    actor = solver._actor_fn(result.actor_params, "f")
    if not result.metrics["actor_costate_policy_aligned"]:
        raise AssertionError("conditional solve returned a costate for a different actor")
    directional_cost = solver._directional_cost("f")
    # The conditional solver now performs and accounts for a disjoint final
    # costate refresh whenever its last accepted actor differs from the policy
    # used for the preceding costate fit.  Reconstruct only the stateless
    # evaluator here; refitting again would double-count work and change the
    # benchmark estimand.
    final_inner = MalliavinAdjointInnerSolver(
        problem,
        directional_cost,
        mam_config=conditional_config.costate,
        control_fn=actor,
    )
    final_costate_params = result.costate_params
    evaluation = config["evaluation"]
    queries = _make_evaluation_queries(
        problem,
        _stream_key(seed, "evaluation_pairs", task_index=task_index),
        _stream_key(seed, "evaluation_rollout", task_index=task_index),
        int(evaluation["query_count"]),
    )
    weights = _boundary_weights(config, queries.states, task)
    weight_sum = jnp.sum(weights)
    boundary_effective = weight_sum**2 / jnp.sum(weights**2)

    discrete_optimal, discrete_optimal_valid, discrete_optimal_info, discrete_optimal_queries = (
        _discrete_optimal_reference(
            config,
            task,
            problem,
            cost,
            queries,
            seed,
            task_index,
        )
    )
    path_reference, path_reference_valid, path_reference_info, path_reference_queries = (
        _path_integral_reference(
            config,
            problem,
            cost,
            queries,
            seed,
            task_index,
            weights,
        )
    )
    target_reference, target_reference_valid, target_reference_info, target_reference_queries = (
        _current_policy_target_reference(
            _stream_key(seed, "target_reference", task_index=task_index),
            queries,
            problem,
            cost,
            actor,
            num_pairs=int(evaluation["reference_direct_antithetic_pairs"]),
            replicates=int(evaluation["reference_direct_replicates"]),
            weights=weights,
        )
    )

    current_query_action = _evaluate_actor(actor, queries)
    mam = final_inner.make_action_target_batch(
        _stream_key(seed, "mam_target", task_index=task_index),
        queries.states,
        queries.times,
        queries.endpoints,
        next_time=queries.next_times,
        params=final_costate_params,
        current_control=current_query_action,
        num_antithetic=int(evaluation["mam_antithetic_pairs"]),
    )

    direct_target, direct_valid, direct_queries = _direct_full_return_target(
        _stream_key(seed, "direct_target", task_index=task_index),
        queries,
        problem,
        cost,
        actor,
        int(evaluation["direct_antithetic_pairs"]),
    )

    critic_metrics, critic_dataset, critic = solver._fit_value_critic(
        _stream_key(seed, "critic_cache", task_index=task_index),
        training_pairs.source,
        training_pairs.target,
        "f",
        result.actor_params,
        directional_cost,
    )
    del critic_metrics
    critic_result = solver._value_critic_state["f"]
    critic_costate = _critic_ensemble_costate(critic, critic_result)
    penultimate = jnp.asarray(problem.time_grid.times[-2], dtype=queries.states.dtype)

    def critic_continuation(x: jax.Array, t: jax.Array, endpoint: jax.Array) -> jax.Array:
        value = critic_costate(x, t, endpoint)
        final = jnp.isclose(t, penultimate, rtol=1e-5, atol=1e-6)
        return jnp.asarray(jnp.where(final[:, None], 0.0, value))

    critic_target = assemble_pinned_actor_targets(
        _stream_key(seed, "critic_target", task_index=task_index),
        queries.states,
        queries.endpoints,
        queries.times,
        queries.next_times,
        problem.time_grid.t1,
        _brownian_sigma(problem),
        current_query_action,
        _running_cost(cost),
        critic_continuation,
        num_antithetic=int(evaluation["mam_antithetic_pairs"]),
    )

    path_target, path_usable, path_ess, path_queries = _path_integral_target(
        _stream_key(seed, "path_integral", task_index=task_index),
        queries,
        problem,
        cost,
        num_pairs=int(evaluation["path_integral_pairs"]),
        minimum_ess_fraction=float(evaluation["minimum_ess_fraction"]),
        shard_size=int(evaluation["path_integral_shard_size"]),
    )

    current_target_valid = target_reference_valid
    critic_cache_queries = int(execution.effective_batch_size) * (problem.time_grid.num_steps + 1)
    arms = {
        "mam_corrected": {
            "available": True,
            "estimand": "current_policy_action_target",
            "approximation": "learned_matrix_free_costate_plus_finite_antithetic_arrival",
            "metrics": _metric_sufficient_statistics(
                mam.target,
                target_reference,
                weights,
                current_target_valid & mam.finite,
            ),
            "physical_value_oracle_queries": int(jax.device_get(mam.physical_oracle_queries)),
            "theorem_facing_clipping": False,
        },
        "direct_full_return": {
            "available": True,
            "estimand": "current_policy_action_target",
            "approximation": "finite_antithetic_complete_suffix_score",
            "metrics": _metric_sufficient_statistics(
                direct_target,
                target_reference,
                weights,
                current_target_valid & direct_valid,
            ),
            "physical_value_oracle_queries": direct_queries,
            "tangent_free": True,
            "theorem_facing_clipping": False,
        },
        "critic_autodiff": {
            "available": True,
            "estimand": "current_policy_action_target",
            "approximation": "heldout_two_fold_value_gradient_plus_finite_antithetic_arrival",
            "metrics": _metric_sufficient_statistics(
                critic_target.target,
                target_reference,
                weights,
                current_target_valid & critic_target.finite,
            ),
            "physical_value_oracle_queries": int(
                jax.device_get(critic_target.physical_oracle_queries)
            ),
            "critic_training_cache_rows": int(critic_dataset.states.shape[0]),
            "critic_training_cache_physical_value_oracle_queries": critic_cache_queries,
            "critic_query_rows_are_untouched": True,
            "theorem_facing_clipping": False,
        },
        "path_integral": {
            "available": True,
            "estimand": "kl_relaxed_path_measure_desirability_control",
            "approximation": "finite_self_normalized_ratio_biased",
            "metrics": _metric_sufficient_statistics(
                path_target,
                path_reference,
                weights,
                path_reference_valid & path_usable,
            ),
            "physical_value_oracle_queries": path_queries,
            "minimum_ess_fraction": float(jax.device_get(jnp.min(path_ess))),
            "median_ess_fraction": float(jax.device_get(jnp.median(path_ess))),
            "usable_fraction": float(jax.device_get(jnp.mean(path_usable.astype(jnp.float32)))),
            "exact_in_population_for_declared_path_integral_estimand": True,
            "not_a_discrete_mean_shift_optimality_claim": True,
            "finite_sample_ratio_biased": True,
            "theorem_facing_clipping": False,
        },
    }
    missing = [arm for arm in REQUIRED_ARMS if not arms.get(arm, {}).get("available", False)]
    if missing:
        raise RuntimeError(f"required Gate-1 comparison arms are unavailable: {missing}")

    policy_action = current_query_action
    if bool(jax.device_get(jnp.all(discrete_optimal_valid))):
        policy_reference_record: dict[str, Any] = {
            "available": True,
            "reference": "discrete_optimal_control",
            "metrics": _metric_sufficient_statistics(
                policy_action,
                discrete_optimal,
                weights,
                discrete_optimal_valid,
            ),
        }
    else:
        policy_reference_record = {
            "available": False,
            "reference": "discrete_optimal_control",
            "reason": discrete_optimal_info["reason"],
        }
    objective = _heldout_objective_change(
        problem,
        cost,
        actor,
        _stream_key(seed, "objective_pairs", task_index=task_index),
        _stream_key(seed, "objective_noise", task_index=task_index),
        int(evaluation["objective_sample_size"]),
        z_value=float(config["conditional"]["one_sided_z"]),
        improvement_tolerance=float(config["conditional"]["improvement_tolerance"]),
    )
    history = result.metrics["actor_acceptance_history"]
    genuine = [
        item
        for item in history
        if item["actor_update_accepted"]
        and item["confirmation"] is not None
        and item["confirmation"]["mean_difference"] < 0.0
        and item["confirmation"]["upper_confidence_bound"]
        < -float(config["conditional"]["improvement_tolerance"])
    ]
    if warmup:
        return {"warmup_complete": True}
    return {
        "seed": seed,
        "task": task,
        "evaluation_dataset_sha256": queries.dataset_sha256,
        "references": {
            "current_policy_action_target": target_reference_info,
            "discrete_optimal_control": discrete_optimal_info,
            "path_integral_desirability_control": path_reference_info,
        },
        "reference_physical_value_oracle_queries": {
            "current_policy_action_target": target_reference_queries,
            "discrete_optimal_control": discrete_optimal_queries,
            "path_integral_desirability_control": path_reference_queries,
        },
        "boundary_weighting": {
            "kind": "uniform" if task == "smooth_lqg" else "gaussian_signed_distance",
            "bandwidth": None if task == "smooth_lqg" else evaluation["boundary_bandwidth"],
            "effective_query_count": float(jax.device_get(boundary_effective)),
        },
        "arms": arms,
        "accepted_policy_action_vs_reference": policy_reference_record,
        "acceptance": {
            "solver_accepted_any": bool(result.metrics["actor_update_accepted"]),
            "genuinely_improving_confirmation_count": len(genuine),
            "genuinely_improving_step_accepted": bool(genuine),
            "history": history,
            "untouched_final_vs_zero": objective,
        },
        "conditional_work_accounting": result.metrics["work_accounting"],
        "final_policy_costate_refresh": {
            "performed": bool(result.metrics["final_costate_refresh_executed"]),
            "uses_disjoint_solver_stream_when_performed": True,
            "warm_started_from_last_policy_iteration_when_performed": True,
            "training_steps": (
                int(conditional_config.costate_steps)
                if result.metrics["final_costate_refresh_executed"]
                else 0
            ),
            "final_loss": result.metrics["final_costate_refresh_loss"],
            "included_in_conditional_certified_work": True,
            "actor_costate_policy_aligned": True,
        },
        "physical_value_oracle_query_accounting": {
            "conditional_training_certified_requested_outputs": result.metrics["work_accounting"][
                "certified_counters"
            ]["running_cost_oracle_evaluations"],
            "evaluation_mam_arrival": arms["mam_corrected"]["physical_value_oracle_queries"],
            "evaluation_direct_suffix": direct_queries,
            "evaluation_critic_arrival": arms["critic_autodiff"]["physical_value_oracle_queries"],
            "extra_critic_training_cache": critic_cache_queries,
            "extra_final_policy_costate_refresh": 0,
            "evaluation_path_integral": path_queries,
            "current_policy_target_reference": target_reference_queries,
            "discrete_optimal_control_reference": discrete_optimal_queries,
            "path_integral_desirability_reference": path_reference_queries,
            "external_oracle_billing_certified": False,
        },
        "estimator_scope": {
            "mam_direct_critic_are_current_policy_targets": True,
            "path_integral_targets_kl_relaxed_desirability_control": True,
            "path_integral_not_claimed_discrete_mean_shift_optimal": True,
            "discrete_optimal_control_reference_available": task == "smooth_lqg",
        },
    }


def _gate_summary(config: dict[str, Any], tasks: dict[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    summaries: dict[str, Any] = {}
    all_pass = True
    for task, payload in tasks.items():
        seed_records = payload["seeds"]
        aggregate = _aggregate_metrics(seed_records, "mam_corrected")
        accepting = sum(
            record["acceptance"]["genuinely_improving_step_accepted"]
            and record["acceptance"]["untouched_final_vs_zero"]["accepted"]
            for record in seed_records
        )
        if task == "smooth_lqg":
            metric_pass = aggregate["relative_l2"] <= float(
                gates["smooth_relative_l2"]
            ) and aggregate["cosine"] >= float(gates["smooth_cosine"])
            policy_aggregate = _aggregate_sufficient_statistics(
                [
                    record["accepted_policy_action_vs_reference"]["metrics"][
                        "sufficient_statistics"
                    ]
                    for record in seed_records
                ]
            )
            policy_metric_pass = policy_aggregate["relative_l2"] <= float(
                gates["smooth_relative_l2"]
            ) and policy_aggregate["cosine"] >= float(gates["smooth_cosine"])
            boundary_pass = True
        else:
            metric_pass = aggregate["relative_l2"] <= float(
                gates["hard_boundary_relative_l2"]
            ) and aggregate["cosine"] >= float(gates["hard_boundary_cosine"])
            policy_aggregate = None
            policy_metric_pass = True
            boundary_pass = all(
                record["boundary_weighting"]["effective_query_count"]
                >= int(config["evaluation"]["minimum_boundary_effective_queries"])
                for record in seed_records
            )
        target_reference_pass = all(
            record["references"]["current_policy_action_target"]["usable_fraction"] == 1.0
            for record in seed_records
        )
        path_reference_pass = all(
            record["references"]["path_integral_desirability_control"]["usable_fraction"] == 1.0
            for record in seed_records
        )
        target_reference_precision_pass = all(
            record["references"]["current_policy_action_target"]["weighted_relative_standard_error"]
            <= float(gates["maximum_target_reference_relative_se"])
            and record["references"]["current_policy_action_target"]["split_half_relative_l2"]
            <= float(gates["maximum_target_reference_split_relative_l2"])
            for record in seed_records
        )
        path_reference_precision_pass = all(
            record["references"]["path_integral_desirability_control"][
                "weighted_relative_standard_error"
            ]
            <= float(gates["maximum_path_integral_reference_relative_se"])
            and record["references"]["path_integral_desirability_control"]["split_half_relative_l2"]
            <= float(gates["maximum_path_integral_reference_split_relative_l2"])
            for record in seed_records
        )
        reference_usable_pass = (target_reference_pass and path_reference_pass) or not gates[
            "require_all_reference_queries_usable"
        ]
        reference_precision_pass = target_reference_precision_pass and path_reference_precision_pass
        reference_pass = reference_usable_pass and reference_precision_pass
        seed_count_pass = len(seed_records) == int(gates["required_seeds"])
        acceptance_pass = accepting >= int(gates["minimum_accepting_seeds"])
        task_pass = (
            metric_pass
            and policy_metric_pass
            and boundary_pass
            and reference_pass
            and seed_count_pass
            and acceptance_pass
        )
        all_pass = all_pass and task_pass
        summaries[task] = {
            "mam_corrected_action_target": aggregate,
            "accepting_seed_count": accepting,
            "seed_count": len(seed_records),
            "metric_pass": metric_pass,
            "accepted_policy_metric": policy_aggregate,
            "accepted_policy_metric_pass": policy_metric_pass,
            "boundary_effective_count_pass": boundary_pass,
            "reference_pass": reference_pass,
            "reference_usable_pass": reference_usable_pass,
            "reference_precision_pass": reference_precision_pass,
            "current_policy_target_reference_usable_pass": target_reference_pass,
            "path_integral_reference_usable_pass": path_reference_pass,
            "current_policy_target_reference_precision_pass": target_reference_precision_pass,
            "path_integral_reference_precision_pass": path_reference_precision_pass,
            "seed_count_pass": seed_count_pass,
            "acceptance_pass": acceptance_pass,
            "task_pass": task_pass,
        }
    return {
        "tasks": summaries,
        "comparison_coverage": {
            "implemented_required_arms": list(REQUIRED_ARMS),
            "unavailable_planned_arms": UNAVAILABLE_PLANNED_ARMS,
            "coverage_pass": not UNAVAILABLE_PLANNED_ARMS,
        },
        "all_numerical_conditions_pass": all_pass,
        "all_gate_conditions_pass": all_pass and not UNAVAILABLE_PLANNED_ARMS,
        "gate_is_scientific_only_under_locked_full_contract": True,
    }


def _git_metadata(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _device_metadata() -> list[dict[str, Any]]:
    return [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
            "process_index": int(device.process_index),
        }
        for device in jax.devices()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_benchmark(
    config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run both conditional tasks and write a reproducible artifact bundle."""
    validate_config(config)
    resolved = json.loads(json.dumps(config))
    if output_dir is not None:
        resolved["experiment"]["output_dir"] = str(output_dir)
    validate_config(resolved)
    destination = Path(resolved["experiment"]["output_dir"]).expanduser().resolve()
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    jax.config.update("jax_default_matmul_precision", resolved["numerics"]["matmul_precision"])
    root = Path(__file__).resolve().parents[2]
    start_git = _git_metadata(root)
    started = datetime.now(timezone.utc)
    timing: dict[str, Any] = {"warmup_seconds": {}, "locked_seed_seconds": {}}
    task_results: dict[str, Any] = {}
    process_start = time.perf_counter()
    for task_index, task in enumerate(("smooth_lqg", "hard_disk")):
        if resolved["experiment"]["warmup"]:
            warmup_start = time.perf_counter()
            _run_seed_task(
                resolved,
                task=task,
                task_index=task_index,
                seed=STREAM_IDS["warmup"],
                warmup=True,
            )
            timing["warmup_seconds"][task] = time.perf_counter() - warmup_start
        records: list[dict[str, Any]] = []
        seed_timings: dict[str, float] = {}
        for seed in resolved["experiment"]["seeds"]:
            seed_start = time.perf_counter()
            record = _run_seed_task(
                resolved,
                task=task,
                task_index=task_index,
                seed=int(seed),
            )
            records.append(record)
            seed_timings[str(seed)] = time.perf_counter() - seed_start
        timing["locked_seed_seconds"][task] = seed_timings
        task_results[task] = {"seeds": records}

    gate = _gate_summary(resolved, task_results)
    contract = full_contract_report(resolved)
    end_git = _git_metadata(root)
    clean_reproducible_revision = (
        start_git["commit"] is not None
        and start_git["commit"] == end_git["commit"]
        and start_git["dirty"] is False
        and end_git["dirty"] is False
    )
    is_full = resolved["experiment"]["profile"] == "full"
    eligible_full = bool(
        is_full and contract["matches_locked_full_contract"] and clean_reproducible_revision
    )
    if eligible_full:
        status = FULL_PASS_STATUS if gate["all_gate_conditions_pass"] else FULL_FAIL_STATUS
    elif is_full:
        status = FULL_INELIGIBLE_STATUS
    else:
        status = SMOKE_STATUS
    results = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "scientific_evidence": eligible_full,
        "tasks": task_results,
        "gate_summary": gate,
        "full_contract": contract,
        "execution_eligibility": {
            "eligible_locked_full_run": eligible_full,
            "clean_stable_git_revision": clean_reproducible_revision,
            "start_git": start_git,
            "end_git": end_git,
        },
    }
    completed = datetime.now(timezone.utc)
    timing["total_process_seconds"] = time.perf_counter() - process_start
    timing["interpretation"] = {
        "warmup": "compile-plus-execute, excluded from locked scientific results",
        "locked_seeds": "steady-state after shape-matched warmup when warmup=true",
        "total": "wall clock including artifact-independent computation",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "config_sha256": _sha256_json(resolved),
        "scientific_payload_sha256": _sha256_json(results),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
        },
        "devices": _device_metadata(),
        "git": {
            "start": start_git,
            "end": end_git,
            "clean_stable_revision": clean_reproducible_revision,
        },
        "rng_stream_ids": STREAM_IDS,
        "timing": timing,
        "full_contract": contract,
        "limitations": [
            "hard-task current-policy truth is a finite direct-score sample mean",
            "path-integral diagnostics are self-normalized finite-sample ratio estimates",
            "normal/CLT acceptance bounds are nominal rather than distribution-free",
            "this conditional benchmark does not test global endpoint feasibility",
            "GPU memory and throughput are not inferred from CPU runs",
        ],
    }
    config_path = destination / "resolved_config.json"
    results_path = destination / "results.json"
    _write_json(config_path, resolved)
    _write_json(results_path, results)
    manifest["artifacts"] = {
        "resolved_config.json": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "results.json": hashlib.sha256(results_path.read_bytes()).hexdigest(),
    }
    _write_json(destination / "run_manifest.json", manifest)
    return {**results, "output_dir": str(destination), "manifest": manifest}


__all__ = [
    "FULL_GATE1_SEEDS",
    "FULL_FAIL_STATUS",
    "FULL_INELIGIBLE_STATUS",
    "FULL_PASS_STATUS",
    "LQGPolicy",
    "PROTOCOL",
    "REQUIRED_ARMS",
    "SCHEMA_VERSION",
    "SMOKE_STATUS",
    "UNAVAILABLE_PLANNED_ARMS",
    "full_contract_report",
    "load_config",
    "run_benchmark",
    "solve_discrete_lqg_policy",
    "validate_config",
]
