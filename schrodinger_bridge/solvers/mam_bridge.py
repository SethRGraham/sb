"""Experimental global bridge composition for Malliavin Adjoint Matching.

This module deliberately keeps the global solver separate from ``DSBMSolver``.
DSBM's analytic bridge-regression target is a baseline, whereas MAM first
solves endpoint-conditioned stochastic-control problems and then performs a
finite-grid Markov-style field projection.  Sharing those targets would
silently change both methods.

The implementation is a scientifically honest *reference composition*:

* conditional paths are exactly pinned under the declared discrete Brownian
  chain;
* the stopped antithetic actor label includes both the learned next-costate
  term and the transition's hard arrival potential; it equals the exact
  action estimand only when the supplied costate is exact;
* actor changes are accepted only on paired-noise streams disjoint from actor
  fitting, conditional on the fixed endpoint-pair cache;
* global endpoint satisfaction is empirical and is never inferred from exact
  conditional pinning; and
* the conditional/global samplers and costate labels use matrix-free arrays
  and ``lax.scan``.  Metadata still says production scalability is not yet
  validated until the declared one-GPU memory and timing gates are run.

Mathematical convention
-----------------------
For a fixed endpoint ``y`` and uniform grid, the stochastic transitions are

    X[n+1] = rho[n] X[n] + (1-rho[n]) y
             + Gamma[n] (sqrt(dt) u[n] + xi[n]),
    Gamma[n] = sqrt(dt rho[n]) Sigma.

The final transition is deterministic.  With right-endpoint quadrature, the
one-step action target is

    -sqrt(rho) Sigma.T E[p[n+1] | X[n], y]
    -sqrt(dt) E[(ell[n+1] - c) xi | X[n], y].

One antithetic pair estimates both expectations after plugging in the learned,
stopped costate.  The default actor and finite-grid Euler endpoint field are
nonlinear time-conditioned L2 regressors; ``affine_reference`` remains
available for small algebra tests.  The latter is a conditional-mean/Euler
projection, not an exact discrete Markov projection.  Both regressions are
finite-sample approximations, not theorems.  Conservative nonlinear updates
are represented as finite output-space mixtures because interpolating MLP
parameters would not interpolate their represented action or endpoint fields.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import marshal
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, Protocol, TypedDict, cast, runtime_checkable

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from ..core.problem import BrownianMotion, SBProblem
from ..core.types import (
    Array,
    DiagnosticReport,
    DriftFn,
    IntegratorType,
    Params,
    PRNGKey,
    RepresentationType,
    Scalar,
    SolverConfig,
    SolverResult,
    SolverType,
    TrainingConfig,
    TrajectoryBatch,
)
from ..networks import AdamState, adam_update, init_adam
from ..process import BridgeProcess
from . import mam_accounting as mam_accounting_module
from . import mam_fields as mam_fields_module
from .base import SBSolution, SBSolver
from .malliavin_adjoint import (
    MalliavinAdjointConfig,
    MalliavinAdjointInnerSolver,
    ValueOnlyCost,
    assemble_antithetic_direct_action_score,
    simulate_pinned_brownian_rollout_matrix_free,
)
from .mam_acceptance import paired_objective_statistics, select_line_search_candidate
from .mam_accounting import (
    MAMWorkCounters,
    completed_conditional_solve_work,
    global_half_iteration_work,
)
from .mam_diagnostics import (
    EndpointAuditConfig,
    EndpointMetrics,
    EndpointThresholds,
    ModeLabelFn,
    ModeProportionFn,
    NullCalibrationResult,
    audit_endpoint,
    calibrate_endpoint_thresholds,
)
from .mam_execution import (
    DeviceTopology,
    ExecutionPlan,
    RNGDomain,
    RNGLedger,
    discover_device_topology,
    make_execution_plan,
)
from .mam_fields import (
    MAMActorDataset,
    MAMActorField,
    MAMEndpointProjectorField,
    MAMFieldConfig,
    MAMFieldTrainState,
    MAMProjectionDataset,
    actor_field_predict,
    endpoint_projector_field_predict,
)
from .mam_value_critic import (
    CrossFittedValueCritic,
    CrossFittedValueCriticResult,
    FixedPolicyReturnDataset,
    ValueCriticConfig,
)

RunningPotentialFn = Callable[[Array, Array, Array], Array]

_CERTIFIED_WORK_FIELDS = (
    "running_cost_oracle_evaluations",
    "simulated_transitions",
    "tangent_vjps",
    "tangent_jvps",
    "optimizer_examples",
    "optimizer_updates",
)
_UNMEASURED_WORK_FIELDS = (
    "compile_time_ns",
    "steady_state_time_ns",
    "peak_device_memory_bytes",
)


def _require_integer(name: str, value: Any, *, minimum: int) -> None:
    """Reject bools and integer-looking floats at scientific config boundaries."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_finite_real(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    maximum: float | None = None,
) -> None:
    """Validate a scalar real without silently accepting booleans or NaNs."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    if maximum is not None and scalar > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def _validate_line_search(name: str, values: Any) -> None:
    if not isinstance(values, tuple) or not values:
        raise TypeError(f"{name} must be a nonempty tuple")
    for index, value in enumerate(values):
        _require_finite_real(f"{name}[{index}]", value, positive=True, maximum=1.0)
    if tuple(sorted(values, reverse=True)) != values:
        raise ValueError(f"{name} must be ordered from largest to smallest")


def _work_accounting_record(
    counters: MAMWorkCounters | None,
    *,
    scope: str,
    cumulative: MAMWorkCounters | None = None,
    uncertified_reason: str | None = None,
    derivation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return JSON-safe, deliberately narrow work-accounting metadata."""
    certified = counters is not None
    if certified and uncertified_reason is not None:
        raise ValueError("certified work counters cannot have an uncertified reason")
    if not certified and cumulative is not None:
        raise ValueError("uncertified work cannot have certified cumulative counters")
    if not isinstance(scope, str) or not scope:
        raise ValueError("work-accounting scope must be nonempty")
    if derivation is not None:
        try:
            json.dumps(derivation, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("work-accounting derivation must be finite JSON data") from exc
    return {
        "schema_version": 2,
        "scope": scope,
        "structural_counters_certified": certified,
        "certified_counters": None if counters is None else counters.to_state(),
        "cumulative_certified_counters": (None if cumulative is None else cumulative.to_state()),
        "certified_fields": list(_CERTIFIED_WORK_FIELDS) if certified else [],
        "unmeasured_fields": list(_UNMEASURED_WORK_FIELDS),
        "successful_completed_calls_only": True,
        "failed_attempt_work_included": False,
        "oracle_count_semantics": "requested_scalar_value_outputs",
        "external_oracle_billing_certified": False,
        "uncertified_reason": uncertified_reason,
        "derivation": derivation,
    }


@dataclass(frozen=True)
class ValueOnlyRunningPotential:
    """A JAX-evaluable, possibly discontinuous additive running potential.

    ``value(x, t, context)`` maps ``[B,d]``, ``[B]``, and ``[B,d]`` to
    ``[B]``.  V1 is a Markov state/time objective, so the adapter always
    supplies an all-zero reserved context.  Endpoint- or environment-dependent
    costs require a future explicit exogenous-context API; they must not
    silently optimize a different conditional and global objective.  Only
    values are required.  The adapter does not claim that a black-box host
    callback is differentiable or JIT-compatible.
    """

    value: RunningPotentialFn | None = None
    identifier: str = "zero_running_potential"

    def __post_init__(self) -> None:
        if self.value is not None and not callable(self.value):
            raise TypeError("running-potential value must be callable or None")
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise ValueError("running-potential identifier must be a nonempty string")

    def as_value_only_cost(self) -> ValueOnlyCost:
        if self.value is None:
            return ValueOnlyCost(identifier=self.identifier)
        raw_value = self.value

        def context_free_value(x: Array, t: Array, context: Array) -> Array:
            return raw_value(x, t, jnp.zeros_like(context))

        return ValueOnlyCost(running_cost=context_free_value, identifier=self.identifier)


@dataclass
class ConditionalMAMConfig:
    """Smallest complete conditional MAM policy-iteration configuration."""

    costate: MalliavinAdjointConfig = field(
        default_factory=lambda: MalliavinAdjointConfig(
            training_steps=32,
            batch_size=128,
            minimum_remaining_steps=1,
        )
    )
    costate_steps: int = 32
    batch_size: int = 128
    value_critic: ValueCriticConfig = field(
        default_factory=lambda: ValueCriticConfig(training_steps=32)
    )
    train_value_critic: bool = True
    direct_score_diagnostic_size: int = 128
    actor_model: str = "nonlinear"
    actor_field_config: MAMFieldConfig = field(
        default_factory=lambda: MAMFieldConfig(training_steps=128)
    )
    maximum_actor_components: int = 32
    actor_ridge: float = 1e-3
    acceptance_size: int = 256
    policy_iterations: int = 3
    maximum_consecutive_rejections: int = 3
    line_search: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    one_sided_z: float = 1.6448536269514722
    improvement_tolerance: float = 0.0

    def __post_init__(self) -> None:
        _require_integer("costate_steps", self.costate_steps, minimum=1)
        _require_integer("batch_size", self.batch_size, minimum=2)
        _require_integer(
            "direct_score_diagnostic_size",
            self.direct_score_diagnostic_size,
            minimum=2,
        )
        _require_integer(
            "maximum_actor_components",
            self.maximum_actor_components,
            minimum=1,
        )
        _require_integer("acceptance_size", self.acceptance_size, minimum=2)
        _require_integer("policy_iterations", self.policy_iterations, minimum=1)
        _require_integer(
            "maximum_consecutive_rejections",
            self.maximum_consecutive_rejections,
            minimum=1,
        )
        if self.actor_model not in {"nonlinear", "affine_reference"}:
            raise ValueError("actor_model must be 'nonlinear' or 'affine_reference'")
        if not isinstance(self.train_value_critic, bool):
            raise TypeError("train_value_critic must be bool")
        if not self.train_value_critic:
            raise ValueError(
                "the single-GPU MAM bridge protocol requires the cross-fitted value critic"
            )
        _require_finite_real("actor_ridge", self.actor_ridge, positive=True)
        _validate_line_search("line_search", self.line_search)
        _require_finite_real("one_sided_z", self.one_sided_z, positive=True)
        _require_finite_real(
            "improvement_tolerance",
            self.improvement_tolerance,
            nonnegative=True,
        )


@dataclass
class MarkovProjectionConfig:
    """Per-time finite endpoint-prediction regression configuration."""

    model: str = "nonlinear"
    field_config: MAMFieldConfig = field(default_factory=lambda: MAMFieldConfig(training_steps=256))
    maximum_components: int = 32
    ridge: float = 1e-3
    damping: float = 1.0
    skip_terminal_noise: bool = False
    validation_size: int = 256
    validation_projections: int = 32
    validation_replicates: int = 8
    line_search: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    one_sided_z: float = 1.6448536269514722
    feasible_endpoint_noninferiority: float = 0.0
    objective_improvement_tolerance: float = 0.0

    def __post_init__(self) -> None:
        _require_integer("maximum_components", self.maximum_components, minimum=1)
        _require_integer("validation_size", self.validation_size, minimum=2)
        _require_integer(
            "validation_projections",
            self.validation_projections,
            minimum=2,
        )
        _require_integer(
            "validation_replicates",
            self.validation_replicates,
            minimum=2,
        )
        if self.model not in {"nonlinear", "affine_reference"}:
            raise ValueError("projection model must be 'nonlinear' or 'affine_reference'")
        _require_finite_real("projection ridge", self.ridge, positive=True)
        _require_finite_real("projection damping", self.damping, positive=True, maximum=1.0)
        if not isinstance(self.skip_terminal_noise, bool):
            raise TypeError("skip_terminal_noise must be bool")
        if self.skip_terminal_noise:
            raise ValueError(
                "skipping terminal noise would make the audited sampler differ from the "
                "public Euler bridge process"
            )
        _validate_line_search("projection line_search", self.line_search)
        _require_finite_real("projection one_sided_z", self.one_sided_z, positive=True)
        _require_finite_real(
            "feasible_endpoint_noninferiority",
            self.feasible_endpoint_noninferiority,
            nonnegative=True,
        )
        _require_finite_real(
            "objective_improvement_tolerance",
            self.objective_improvement_tolerance,
            nonnegative=True,
        )


@dataclass
class MAMOuterLoopConfig:
    """Alternating reciprocal/Markov-projection loop."""

    num_iterations: int = 1
    directions: tuple[str, ...] = ("b", "f")
    cache_size: int = 1024
    audit_size: int = 512

    def __post_init__(self) -> None:
        _require_integer("num_iterations", self.num_iterations, minimum=1)
        _require_integer("cache_size", self.cache_size, minimum=2)
        _require_integer("audit_size", self.audit_size, minimum=2)
        if not isinstance(self.directions, tuple) or not self.directions:
            raise TypeError("directions must be a nonempty tuple")
        if any(not isinstance(d, str) or d not in {"f", "b"} for d in self.directions):
            raise ValueError("directions must contain only 'f' and 'b'")
        if len(self.directions) != 2 or set(self.directions) != {"f", "b"}:
            raise ValueError(
                "global MAM requires both forward and backward directions exactly once"
            )


@dataclass
class MAMExecutionConfig:
    """Single-device defaults; two-device execution is optional, never required."""

    microbatch_size: int | None = None
    effective_batch_size: int = 1024
    allow_two_devices: bool = False
    production_dtype: Any = jnp.float32

    def __post_init__(self) -> None:
        _require_integer("effective_batch_size", self.effective_batch_size, minimum=1)
        if not isinstance(self.allow_two_devices, bool):
            raise TypeError("allow_two_devices must be bool")
        if self.microbatch_size is not None:
            _require_integer("microbatch_size", self.microbatch_size, minimum=1)
            if self.effective_batch_size < self.microbatch_size:
                raise ValueError("effective_batch_size must be at least microbatch_size")
            if self.effective_batch_size % self.microbatch_size != 0:
                raise ValueError("effective_batch_size must be divisible by microbatch_size")
        if np.dtype(self.production_dtype) != np.dtype(np.float32):
            raise ValueError(
                "MAMBridgeSolver production execution is fixed to float32; "
                "use the pure kernels for float64 reference tests"
            )


@dataclass
class MAMBridgeConfig:
    """Configuration for the experimental global MAM bridge composition."""

    conditional: ConditionalMAMConfig = field(default_factory=ConditionalMAMConfig)
    projection: MarkovProjectionConfig = field(default_factory=MarkovProjectionConfig)
    outer: MAMOuterLoopConfig = field(default_factory=MAMOuterLoopConfig)
    execution: MAMExecutionConfig = field(default_factory=MAMExecutionConfig)
    audit: EndpointAuditConfig = field(default_factory=EndpointAuditConfig)

    def __post_init__(self) -> None:
        expected = {
            "conditional": (self.conditional, ConditionalMAMConfig),
            "projection": (self.projection, MarkovProjectionConfig),
            "outer": (self.outer, MAMOuterLoopConfig),
            "execution": (self.execution, MAMExecutionConfig),
            "audit": (self.audit, EndpointAuditConfig),
        }
        for name, (value, expected_type) in expected.items():
            if not isinstance(value, expected_type):
                raise TypeError(f"MAMBridgeConfig {name} must be {expected_type.__name__}")


def _validate_public_sample_count(num_samples: Any) -> int:
    _require_integer("num_samples", num_samples, minimum=1)
    return int(num_samples)


def _validated_public_start(
    start: Any,
    *,
    num_samples: int,
    dim: int,
    dtype: Any,
    name: str,
) -> Array:
    """Validate an explicit/sampled public start before entering a JAX scan."""

    try:
        host = np.asarray(jax.device_get(start))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an array") from exc
    expected_shape = (num_samples, dim)
    if host.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {host.shape}")
    expected_dtype = np.dtype(dtype)
    if host.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {host.dtype}")
    if not np.all(np.isfinite(host)):
        raise FloatingPointError(f"{name} contains nonfinite values")
    return jnp.asarray(start)


def _validate_public_paths(
    paths: Array,
    *,
    num_samples: int,
    num_steps: int,
    dim: int,
    dtype: Any,
) -> None:
    expected_shape = (num_samples, num_steps + 1, dim)
    if paths.shape != expected_shape:
        raise ValueError(f"sampled paths must have shape {expected_shape}, got {paths.shape}")
    expected_dtype = np.dtype(dtype)
    if np.dtype(paths.dtype) != expected_dtype:
        raise TypeError(f"sampled paths must have dtype {expected_dtype}, got {paths.dtype}")
    if not bool(jax.device_get(jnp.all(jnp.isfinite(paths)))):
        raise FloatingPointError("sampled paths contain nonfinite values")


class EndpointPairBatch(NamedTuple):
    """Current endpoint coupling, with arrays ``source,target: [B,d]``."""

    source: Array
    target: Array

    @property
    def batch_size(self) -> int:
        return int(self.source.shape[0])


@dataclass
class ConditionalMAMResult:
    """Output of one endpoint-conditioned policy-improvement stage.

    ``paths`` are in chronological orientation.  ``local_paths`` start at the
    constrained marginal for the requested direction.  Projection arrays are
    ordered by local elapsed time and have shapes ``[B,N,d]``, ``[N]``, and
    ``[B,N,d]`` respectively.
    """

    paths: Array
    local_paths: Array
    controls: Array
    projection_states: Array
    projection_times: Array
    endpoint_predictions: Array
    actor_params: _FieldParams
    costate_params: Params
    metrics: dict[str, Any]
    direction: str
    exact_conditional_endpoint: bool
    certified_work_counters: MAMWorkCounters | None = None
    status: str = "CONDITIONAL_MAM_MATRIX_FREE_REFERENCE"


@dataclass
class ProjectionResult:
    params: _FieldParams
    loss: Array
    finite: Array
    direction: str
    status: str = "FINITE_GRID_EULER_ENDPOINT_FIELD_REGRESSION"


@dataclass(frozen=True)
class _NonlinearFieldMixture:
    """Flat finite output-space mixture of complete neural train states.

    Missing mixture mass represents the exact zero field, so an empty mixture
    is a valid initial policy/projection.  Components are deduplicated by their
    prediction-parameter fingerprint.  We fail closed at the configured cap;
    silently pruning or parameter-averaging components would change the field
    that independent acceptance evaluated.
    """

    kind: str
    components: tuple[MAMFieldTrainState, ...]
    weights: Array


_FieldParams = Array | _NonlinearFieldMixture


class _CheckpointRollbackSnapshot(TypedDict):
    """Precisely typed in-memory state used for transactional restoration."""

    params: Params | None
    is_trained: bool
    conditional_solver: dict[str, Any]
    audit_history: list[dict[str, Any]]
    last_direction: str | None
    global_endpoint_pass: bool
    source_calibration: NullCalibrationResult | None
    target_calibration: NullCalibrationResult | None
    resume_pairs: EndpointPairBatch | None
    completed_half_iterations: int
    loss_history: list[float]
    last_metrics: dict[str, Any]
    rng_ledger: RNGLedger | None
    checkpoint_origin_device_topology: dict[str, Any] | None


def _direction_value(direction: str, dtype: Any = jnp.float32) -> Array:
    if direction == "f":
        return jnp.asarray(1.0, dtype=dtype)
    if direction == "b":
        return jnp.asarray(-1.0, dtype=dtype)
    raise ValueError("direction must be 'f' or 'b'")


def _projection_physical_times(
    problem: SBProblem,
    dtype: Any,
    direction: str,
) -> Array:
    """Return the one canonical preterminal projection grid in physical time."""
    local_times = jnp.asarray(problem.time_grid.times[:-1], dtype=dtype)
    if direction == "f":
        return local_times
    if direction == "b":
        terminal_sum = jnp.asarray(
            problem.time_grid.t0 + problem.time_grid.t1,
            dtype=dtype,
        )
        return terminal_sum - local_times
    raise ValueError("direction must be 'f' or 'b'")


def _prediction_parameter_fingerprint(state: MAMFieldTrainState) -> str:
    """Hash only values that determine predictions, not Adam/RNG state."""

    digest = hashlib.sha256()
    digest.update(state.field_kind.encode())
    digest.update(state.config_fingerprint.encode())
    digest.update(state.parameter_signature.encode())
    leaves, structure = jax.tree_util.tree_flatten(state.params)
    digest.update(str(structure).encode())
    for leaf in leaves:
        value = np.ascontiguousarray(jax.device_get(leaf))
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _mix_nonlinear_fields(
    left: _NonlinearFieldMixture,
    right: _NonlinearFieldMixture,
    left_weight: float,
    right_weight: float,
    *,
    maximum_components: int,
) -> _NonlinearFieldMixture:
    """Return the exact flattened output mixture with stable deduplication."""

    if left.kind != right.kind:
        raise ValueError("cannot mix nonlinear fields of different kinds")
    if not np.isfinite(left_weight) or not np.isfinite(right_weight):
        raise ValueError("field mixture weights must be finite")
    if left_weight < 0.0 or right_weight < 0.0:
        raise ValueError("field mixture weights must be nonnegative")
    entries: dict[str, tuple[MAMFieldTrainState, float]] = {}
    for mixture, scale in ((left, left_weight), (right, right_weight)):
        weights = np.asarray(jax.device_get(mixture.weights), dtype=np.float64)
        if weights.shape != (len(mixture.components),):
            raise ValueError("nonlinear field mixture weight shape mismatch")
        for component, weight in zip(mixture.components, weights, strict=True):
            combined_weight = scale * float(weight)
            if combined_weight == 0.0:
                continue
            fingerprint = _prediction_parameter_fingerprint(component)
            if fingerprint in entries:
                previous, previous_weight = entries[fingerprint]
                entries[fingerprint] = (previous, previous_weight + combined_weight)
            else:
                entries[fingerprint] = (component, combined_weight)
    if len(entries) > maximum_components:
        raise RuntimeError(
            "nonlinear field mixture reached its exact component cap; "
            "increase the cap or implement validated distillation"
        )
    ordered = [entries[key] for key in sorted(entries)]
    return _NonlinearFieldMixture(
        kind=left.kind,
        components=tuple(component for component, _ in ordered),
        weights=jnp.asarray([weight for _, weight in ordered], dtype=jnp.float32),
    )


def _single_component_mixture(
    kind: str,
    state: MAMFieldTrainState,
) -> _NonlinearFieldMixture:
    return _NonlinearFieldMixture(
        kind=kind,
        components=(state,),
        weights=jnp.ones((1,), dtype=jnp.float32),
    )


def _empty_mixture(kind: str) -> _NonlinearFieldMixture:
    return _NonlinearFieldMixture(
        kind=kind,
        components=(),
        weights=jnp.empty((0,), dtype=jnp.float32),
    )


@runtime_checkable
class ConditionalBridgeSolver(Protocol):
    """Narrow injection seam for a future matrix-free MAM conditional core."""

    status: str

    def solve(
        self,
        key: PRNGKey,
        endpoint_pairs: EndpointPairBatch,
        direction: str,
    ) -> ConditionalMAMResult: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


def _constant_diffusion(problem: SBProblem, dtype: Any) -> Array:
    dim = problem.dim
    probe = jnp.zeros((1, dim), dtype=dtype)
    sigma = jnp.asarray(problem.reference.diffusion(probe, problem.time_grid.t0), dtype=dtype)
    if sigma.ndim == 0:
        return sigma * jnp.eye(dim, dtype=dtype)
    if sigma.ndim == 1 and sigma.shape == (dim,):
        return jnp.diag(sigma)
    if sigma.shape == (dim, dim):
        return sigma
    raise ValueError("MAMBridgeSolver requires scalar, diagonal, or square diffusion")


def _ridge_per_time(features: Array, targets: Array, ridge: float) -> Array:
    """Fit independent affine maps at every time; shapes ``[B,N,f] -> [N,f,d]``."""
    if features.ndim != 3 or targets.ndim != 3:
        raise ValueError("features and targets must have shapes [batch,time,dim]")
    if features.shape[:2] != targets.shape[:2]:
        raise ValueError("features and targets must share batch and time axes")
    batch_size = features.shape[0]
    time_major_x = jnp.swapaxes(features, 0, 1)
    time_major_y = jnp.swapaxes(targets, 0, 1)
    eye = jnp.eye(features.shape[-1], dtype=features.dtype)

    def fit_one(x: Array, y: Array) -> Array:
        gram = (x.T @ x) / batch_size + ridge * eye
        rhs = (x.T @ y) / batch_size
        return jnp.asarray(jnp.linalg.solve(gram, rhs))

    return jax.vmap(fit_one)(time_major_x, time_major_y)


def _tree_all_finite(tree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def _linear_field(params: Array, state: Array, endpoint: Array, step: Array) -> Array:
    state = jnp.atleast_2d(state)
    endpoint = jnp.atleast_2d(endpoint)
    features = jnp.concatenate(
        [state, endpoint, jnp.ones((state.shape[0], 1), dtype=state.dtype)], axis=-1
    )
    selected = params[step]
    if selected.ndim == 2:
        return jnp.asarray(features @ selected)
    return jnp.asarray(jnp.einsum("bf,bfd->bd", features, selected))


class MAMConditionalSolver:
    """Matrix-free conditional MAM with an arrival-correct actor target.

    This class composes the currently verified BEL costate labels.  It never
    calls ``MalliavinAdjointInnerSolver.propose_control`` because that method's
    discrete branch does not own the arrival potential.  The production actor
    is a nonlinear time-conditioned regressor; ``affine_reference`` remains a
    small-problem algebra and checkpoint reference.
    """

    status = "CONDITIONAL_MAM_MATRIX_FREE_REFERENCE"

    def __init__(
        self,
        problem: SBProblem,
        value_cost: ValueOnlyCost,
        config: ConditionalMAMConfig,
        execution: MAMExecutionConfig | None = None,
    ):
        self.problem = problem
        self.value_cost = value_cost
        self.config = config
        requested_execution = execution or MAMExecutionConfig(
            microbatch_size=config.batch_size,
            effective_batch_size=config.batch_size,
            allow_two_devices=False,
        )
        execution_plan = make_execution_plan(
            horizon=problem.time_grid.num_steps,
            state_dim=problem.dim,
            microbatch_size=requested_execution.microbatch_size,
            effective_batch_size=requested_execution.effective_batch_size,
            device_count=1,
        )
        self.execution = MAMExecutionConfig(
            microbatch_size=execution_plan.microbatch_size,
            effective_batch_size=execution_plan.effective_batch_size,
            allow_two_devices=requested_execution.allow_two_devices,
            production_dtype=requested_execution.production_dtype,
        )
        if self.config.costate.minimum_remaining_steps != 1:
            raise ValueError(
                "arrival-correct actor fitting requires costate anchors through N-2; "
                "set minimum_remaining_steps=1"
            )
        if not self.config.costate.matrix_free_labels:
            raise ValueError("MAM bridge production requires matrix_free_labels=True")
        if not self.config.costate.include_control_energy:
            raise ValueError(
                "MAM bridge objective includes quadratic control energy; "
                "set include_control_energy=True"
            )
        if self.config.costate.anchor_sampling != "stratified":
            raise ValueError("MAM bridge production requires stratified anchor sampling")
        self._sigma = _constant_diffusion(problem, jnp.float32)
        actor_field_config = replace(
            self.config.actor_field_config,
            microbatch_size=execution_plan.microbatch_size,
            effective_batch_size=self.execution.effective_batch_size,
        )
        self._actor_field = (
            MAMActorField(problem.dim, problem.dim, actor_field_config)
            if self.config.actor_model == "nonlinear"
            else None
        )
        self._actor_params: dict[str, _FieldParams] = {}
        self._costate_params: dict[str, Params] = {}
        self._costate_opt_state: dict[str, AdamState] = {}
        self._costate_policy_fingerprint: dict[str, str] = {}
        self._value_critic_state: dict[str, CrossFittedValueCriticResult] = {}

    @property
    def num_steps(self) -> int:
        return self.problem.time_grid.num_steps

    @property
    def stochastic_steps(self) -> int:
        return self.num_steps - 1

    @property
    def microbatch_size(self) -> int:
        """Return the execution-plan-resolved (therefore nonoptional) size."""
        value = self.execution.microbatch_size
        if value is None:
            raise RuntimeError("conditional execution plan did not resolve a microbatch size")
        return value

    def _zero_actor_params(self, dtype: Any) -> _FieldParams:
        if self.config.actor_model == "nonlinear":
            return _empty_mixture("actor")
        feature_dim = 2 * self.problem.dim + 1
        return jnp.zeros((self.stochastic_steps, feature_dim, self.problem.dim), dtype=dtype)

    def _actor_fn(
        self,
        params: _FieldParams,
        direction: str,
    ) -> Callable[[Array, Scalar, Array], Array]:
        t0 = self.problem.time_grid.t0
        dt = self.problem.time_grid.dt
        maximum = self.stochastic_steps - 1
        direction_value = _direction_value(direction)

        def actor(state: Array, time: Scalar, endpoint: Array) -> Array:
            state = jnp.atleast_2d(jnp.asarray(state, dtype=jnp.float32))
            endpoint = jnp.atleast_2d(jnp.asarray(endpoint, dtype=state.dtype))
            batch_size = state.shape[0]
            if endpoint.shape[0] == 1 and batch_size != 1:
                endpoint = jnp.broadcast_to(endpoint, (batch_size, self.problem.dim))
            elif endpoint.shape != (batch_size, self.problem.dim):
                raise ValueError("actor endpoint must have shape [batch, state_dim]")
            time_value = jnp.asarray(time, dtype=state.dtype)
            if time_value.ndim == 0:
                time_value = jnp.full((batch_size,), time_value, dtype=state.dtype)
            elif time_value.shape != (batch_size,):
                raise ValueError("actor time must be scalar or have shape [batch]")
            grid_position = (time_value - jnp.asarray(t0, dtype=state.dtype)) / jnp.asarray(
                dt,
                dtype=state.dtype,
            )
            nearest = jnp.rint(grid_position)
            safe_step = jnp.clip(nearest.astype(jnp.int32), 0, maximum)
            actor_grid = jnp.asarray(
                self.problem.time_grid.times[: self.stochastic_steps],
                dtype=state.dtype,
            )
            nominal_time = actor_grid[safe_step]
            scale = jnp.maximum(
                jnp.maximum(
                    jnp.abs(jnp.asarray(dt, dtype=state.dtype)),
                    jnp.maximum(jnp.abs(time_value), jnp.abs(nominal_time)),
                ),
                jnp.asarray(jnp.finfo(state.dtype).tiny, dtype=state.dtype),
            )
            tolerance = 32.0 * jnp.finfo(state.dtype).eps * scale
            valid_time = (
                jnp.isfinite(time_value)
                & (nearest >= 0.0)
                & (nearest <= maximum)
                & (jnp.abs(time_value - nominal_time) <= tolerance)
            )
            if not isinstance(time_value, jax.core.Tracer):
                if not bool(np.all(np.asarray(jax.device_get(valid_time)))):
                    raise ValueError("actor time lies outside its exact departure grid")
            if self.config.actor_model == "nonlinear":
                if not isinstance(params, _NonlinearFieldMixture) or params.kind != "actor":
                    raise TypeError("nonlinear actor parameters must be an actor mixture")
                actor_field = self._actor_field
                if actor_field is None:
                    raise RuntimeError("nonlinear actor field was not initialized")
                directions = jnp.full((batch_size,), direction_value, dtype=state.dtype)
                value = jnp.zeros((batch_size, self.problem.dim), dtype=state.dtype)
                for weight, component in zip(
                    params.weights,
                    params.components,
                    strict=True,
                ):
                    prediction = actor_field_predict(
                        actor_field.factory,
                        component.params,
                        state,
                        nominal_time,
                        endpoint,
                        directions,
                    )
                    value = value + weight * prediction.value
                return jnp.where(valid_time[:, None], value, jnp.nan)
            value = _linear_field(cast(Array, params), state, endpoint, safe_step)
            return jnp.where(valid_time[:, None], value, jnp.nan)

        return actor

    def _directional_cost(self, direction: str) -> ValueOnlyCost:
        running_cost = self.value_cost.running_cost
        if direction == "f" or running_cost is None:
            return self.value_cost
        terminal_sum = self.problem.time_grid.t0 + self.problem.time_grid.t1

        def reverse_cost(x: Array, t: Array, context: Array) -> Array:
            return running_cost(x, terminal_sum - t, context)

        return ValueOnlyCost(
            running_cost=reverse_cost,
            identifier=f"{self.value_cost.identifier}:reverse_time",
        )

    def _local_endpoints(self, pairs: EndpointPairBatch, direction: str) -> tuple[Array, Array]:
        if pairs.source.shape != pairs.target.shape:
            raise ValueError("endpoint pair arrays must have matching shape")
        if pairs.source.ndim != 2 or pairs.source.shape[-1] != self.problem.dim:
            raise ValueError("endpoint pairs must have shape [batch, state_dim]")
        dtype = self.execution.production_dtype
        source = jnp.asarray(pairs.source, dtype=dtype)
        target = jnp.asarray(pairs.target, dtype=dtype)
        if not bool(jax.device_get(jnp.all(jnp.isfinite(source)) & jnp.all(jnp.isfinite(target)))):
            raise FloatingPointError("endpoint pairs must be finite")
        if direction == "f":
            return source, target
        if direction == "b":
            return target, source
        raise ValueError("direction must be 'f' or 'b'")

    def _train_costate(
        self,
        key: PRNGKey,
        pairs: EndpointPairBatch,
        direction: str,
        actor_params: _FieldParams,
    ) -> tuple[MalliavinAdjointInnerSolver, Params, Array]:
        start, endpoint = self._local_endpoints(pairs, direction)
        inner = MalliavinAdjointInnerSolver(
            self.problem,
            self._directional_cost(direction),
            mam_config=self.config.costate,
            control_fn=self._actor_fn(actor_params, direction),
        )
        if direction in self._costate_params:
            params = self._costate_params[direction]
            opt_state = self._costate_opt_state[direction]
        else:
            key, init_key = jax.random.split(key)
            params = inner.init_params(init_key)
            params = jax.tree_util.tree_map(
                lambda value: jnp.asarray(value, dtype=start.dtype),
                params,
            )
            opt_state = init_adam(params)
        losses: list[Array] = []
        microbatch_size = self.microbatch_size
        accumulation_steps = self.execution.effective_batch_size // microbatch_size
        loss_and_grad = jax.jit(jax.value_and_grad(inner.loss, has_aux=True))
        for _ in range(self.config.costate_steps):
            gradient_sum = jax.tree_util.tree_map(jnp.zeros_like, params)
            loss_sum = jnp.asarray(0.0, dtype=start.dtype)
            for _microbatch in range(accumulation_steps):
                key, index_key, label_key = jax.random.split(key, 3)
                indices = jax.random.randint(index_key, (microbatch_size,), 0, start.shape[0])
                labels = inner.make_label_batch(
                    label_key,
                    start[indices],
                    endpoint[indices],
                    # The inner solver's stopped, anchor-measurable hard-value
                    # baseline centers each instantaneous BEL residual.  A
                    # suffix-return critic estimates a different object and,
                    # although still mean-valid, injects pure noise when the
                    # hard running cost vanishes under a nonzero policy.
                    running_baseline_fn=None,
                )
                labels_finite = jnp.all(labels.finite) & jnp.all(jnp.isfinite(labels.label))
                if not bool(jax.device_get(labels_finite)):
                    raise FloatingPointError(
                        "nonfinite matrix-free costate label; refusing accumulated update"
                    )
                (loss, metrics), gradients = loss_and_grad(params, labels)
                update_finite = (
                    metrics["prediction_finite"]
                    & metrics["loss_finite"]
                    & _tree_all_finite(gradients)
                )
                if not bool(jax.device_get(update_finite)):
                    raise FloatingPointError(
                        "nonfinite costate loss/gradient; refusing accumulated update"
                    )
                gradient_sum = jax.tree_util.tree_map(
                    lambda total, value: total + value,
                    gradient_sum,
                    gradients,
                )
                loss_sum = loss_sum + loss
            inverse = jnp.asarray(1.0 / accumulation_steps, dtype=start.dtype)
            gradients = jax.tree_util.tree_map(
                lambda value, scale=inverse: scale * value, gradient_sum
            )
            params, opt_state = adam_update(
                opt_state,
                gradients,
                params,
                lr=self.config.costate.learning_rate,
            )
            if not bool(jax.device_get(_tree_all_finite((params, opt_state, gradients)))):
                raise FloatingPointError(
                    "nonfinite accumulated optimizer update; refusing costate step"
                )
            losses.append(loss_sum * inverse)
        self._costate_params[direction] = params
        self._costate_opt_state[direction] = opt_state
        self._costate_policy_fingerprint[direction] = self._policy_fingerprint(
            direction,
            actor_params,
        )
        inner._params = params
        inner._ema_params = params
        return inner, params, jnp.asarray(losses)

    @staticmethod
    def _policy_fingerprint(direction: str, actor_params: _FieldParams) -> str:
        """Hash the frozen actor whose returns populate a critic cache."""
        digest = hashlib.sha256()
        digest.update(direction.encode())
        if isinstance(actor_params, _NonlinearFieldMixture):
            digest.update(actor_params.kind.encode())
            weights = np.ascontiguousarray(jax.device_get(actor_params.weights))
            digest.update(weights.tobytes())
            for component in actor_params.components:
                digest.update(_prediction_parameter_fingerprint(component).encode())
        else:
            value = np.ascontiguousarray(jax.device_get(actor_params))
            digest.update(str(value.dtype).encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _parameter_tree_fingerprint(params: Params) -> str:
        """Hash the finite parameter values returned as a learned costate."""
        digest = hashlib.sha256()
        leaves, structure = jax.tree_util.tree_flatten(params)
        digest.update(str(structure).encode())
        for leaf in leaves:
            value = np.ascontiguousarray(jax.device_get(leaf))
            digest.update(str(value.dtype).encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
        return digest.hexdigest()

    def _make_value_critic_dataset(
        self,
        key: PRNGKey,
        start: Array,
        endpoint: Array,
        direction: str,
        actor_params: _FieldParams,
        directional_cost: ValueOnlyCost,
    ) -> FixedPolicyReturnDataset:
        """Build one frozen-policy return row per sampled path and anchor.

        The return uses the same left-open/right-arrival convention as the
        costate label: departure control energy begins at the anchor, while
        running potential begins at the next state.  Rows are generated in
        static microbatches and remain disjoint from costate/actor/acceptance
        streams through their caller-owned key.
        """
        microbatch_size = self.microbatch_size
        accumulation_steps = self.execution.effective_batch_size // microbatch_size
        states_out: list[Array] = []
        times_out: list[Array] = []
        contexts_out: list[Array] = []
        returns_out: list[Array] = []
        step_indices = jnp.arange(self.stochastic_steps, dtype=jnp.int32)
        for microbatch_index in range(accumulation_steps):
            key, index_key, rollout_key, anchor_key = jax.random.split(key, 4)
            indices = jax.random.randint(index_key, (microbatch_size,), 0, start.shape[0])
            batch_start = start[indices]
            batch_endpoint = endpoint[indices]
            rollout = simulate_pinned_brownian_rollout_matrix_free(
                rollout_key,
                batch_start,
                batch_endpoint,
                self.problem.time_grid.times,
                self._sigma,
                self._actor_fn(actor_params, direction),
                diffusion_rcond=self.config.costate.diffusion_rcond,
            )
            # A random offset rotates a complete stratified time pattern.
            offset = jax.random.randint(anchor_key, (), 0, self.stochastic_steps)
            anchors = (
                jnp.arange(microbatch_size, dtype=jnp.int32)
                + offset
                + microbatch_index * microbatch_size
            ) % self.stochastic_steps
            running = directional_cost.running_values(
                rollout.states,
                rollout.times,
                rollout.context,
            )
            energy = 0.5 * jnp.sum(rollout.controls**2, axis=-1)
            arrival_potential = running[:, 1 : self.stochastic_steps + 1]
            eligible = step_indices[None, :] >= anchors[:, None]
            dt = jnp.asarray(self.problem.time_grid.dt, dtype=rollout.states.dtype)
            returns = dt * jnp.sum(
                jnp.where(eligible, energy + arrival_potential, 0.0),
                axis=1,
            )
            row = jnp.arange(microbatch_size)
            states_out.append(rollout.states[row, anchors])
            times_out.append(rollout.times[anchors])
            contexts_out.append(batch_endpoint)
            returns_out.append(jax.lax.stop_gradient(returns))
        return FixedPolicyReturnDataset(
            states=jnp.concatenate(states_out, axis=0),
            times=jnp.concatenate(times_out, axis=0),
            endpoint_context=jnp.concatenate(contexts_out, axis=0),
            returns=jnp.concatenate(returns_out, axis=0),
            policy_fingerprint=self._policy_fingerprint(direction, actor_params),
        )

    def _fit_value_critic(
        self,
        key: PRNGKey,
        start: Array,
        endpoint: Array,
        direction: str,
        actor_params: _FieldParams,
        directional_cost: ValueOnlyCost,
    ) -> tuple[dict[str, Any], FixedPolicyReturnDataset, CrossFittedValueCritic]:
        """Train the mandatory critic before generating costate labels."""
        dataset_key, fit_key = jax.random.split(key)
        dataset = self._make_value_critic_dataset(
            dataset_key,
            start,
            endpoint,
            direction,
            actor_params,
            directional_cost,
        )
        critic_config = replace(
            self.config.value_critic,
            microbatch_size=self.microbatch_size,
            effective_batch_size=self.execution.effective_batch_size,
        )
        critic = CrossFittedValueCritic(critic_config)
        result = critic.fit(fit_key, dataset)
        baseline = critic.cross_fitted_baseline(result, dataset)
        if not bool(jax.device_get(jnp.all(jnp.isfinite(baseline.value)))):
            raise FloatingPointError("nonfinite cross-fitted value baseline")
        residual = baseline.value - dataset.returns
        self._value_critic_state[direction] = result
        return (
            {
                "trained": True,
                "cross_fitted": True,
                "rows": int(dataset.states.shape[0]),
                "training_steps": critic_config.training_steps,
                "return_rmse": float(jax.device_get(jnp.sqrt(jnp.mean(residual**2)))),
                "policy_fingerprint": dataset.policy_fingerprint,
                "used_for_costate_centering": False,
                "costate_centering": "stopped_anchor_hard_value",
                "used_for_direct_score_baseline": True,
                "used_for_acceptance_calibration": True,
                "diagnostic_not_ground_truth": True,
            },
            dataset,
            critic,
        )

    def _critic_baseline_fn(
        self,
        direction: str,
        critic: CrossFittedValueCritic,
    ) -> Callable[[Array, Array, Array], Array]:
        """Stopped ensemble baseline trained on an independent critic cache."""
        result = self._value_critic_state[direction]

        def baseline(states: Array, times: Array, context: Array) -> Array:
            inputs = jnp.concatenate([states, context], axis=-1)
            values = [
                jnp.asarray(
                    critic.factory.forward(params, inputs, times),
                    dtype=states.dtype,
                )[:, 0]
                for params in result.params_by_training_fold
            ]
            return jax.lax.stop_gradient(0.5 * (values[0] + values[1]))

        return baseline

    @staticmethod
    def _complete_value_critic_metrics(
        metrics: dict[str, Any],
        dataset: FixedPolicyReturnDataset,
        critic: CrossFittedValueCritic,
        result: CrossFittedValueCriticResult,
        inner: MalliavinAdjointInnerSolver,
        costate_params: Params,
    ) -> dict[str, Any]:
        comparison = critic.critic_autodiff_comparison(result, dataset)
        mam_costate = inner.extract_costate(costate_params)(
            dataset.states,
            dataset.times,
            dataset.endpoint_context,
        )
        if not bool(
            jax.device_get(jnp.all(comparison.finite) & jnp.all(jnp.isfinite(mam_costate)))
        ):
            raise FloatingPointError("nonfinite value-critic costate comparison")
        critic_delta = comparison.costate - jax.lax.stop_gradient(mam_costate)
        numerator = jnp.sum(comparison.costate * mam_costate)
        denominator = jnp.linalg.norm(comparison.costate) * jnp.linalg.norm(mam_costate)
        cosine = numerator / jnp.maximum(
            denominator,
            jnp.asarray(jnp.finfo(dataset.states.dtype).tiny, dtype=dataset.states.dtype),
        )
        return {
            **metrics,
            "critic_costate_norm": float(jax.device_get(jnp.sqrt(jnp.mean(comparison.costate**2)))),
            "mam_critic_costate_rmse": float(jax.device_get(jnp.sqrt(jnp.mean(critic_delta**2)))),
            "mam_critic_costate_cosine": float(jax.device_get(cosine)),
        }

    def _direct_action_score_diagnostic(
        self,
        key: PRNGKey,
        start: Array,
        endpoint: Array,
        direction: str,
        actor_params: _FieldParams,
        directional_cost: ValueOnlyCost,
        inner: MalliavinAdjointInnerSolver,
        costate_params: Params,
        critic_baseline_fn: Callable[[Array, Array, Array], Array],
    ) -> dict[str, Any]:
        """Compare MAM with a tangent-free antithetic full-return score.

        This diagnostic never selects or accepts an actor.  It uses a fresh
        rollout and suffix stream, so its disagreement with the decomposed MAM
        target is an auditable estimator/model diagnostic rather than a
        training loss.
        """
        size = self.config.direct_score_diagnostic_size
        index_key, rollout_key, arrival_key, suffix_key, anchor_key = jax.random.split(key, 5)
        indices = jax.random.randint(index_key, (size,), 0, start.shape[0])
        batch_start = start[indices]
        batch_endpoint = endpoint[indices]
        actor = self._actor_fn(actor_params, direction)
        rollout = simulate_pinned_brownian_rollout_matrix_free(
            rollout_key,
            batch_start,
            batch_endpoint,
            self.problem.time_grid.times,
            self._sigma,
            actor,
            diffusion_rcond=self.config.costate.diffusion_rcond,
        )
        offset = jax.random.randint(anchor_key, (), 0, self.stochastic_steps)
        anchors = (jnp.arange(size, dtype=jnp.int32) + offset) % self.stochastic_steps
        rows = jnp.arange(size)
        states = rollout.states[rows, anchors]
        times = rollout.times[anchors]
        next_times = rollout.times[anchors + 1]
        controls = rollout.controls[rows, anchors]
        decomposed = inner.make_action_target_batch(
            arrival_key,
            states,
            times,
            batch_endpoint,
            next_time=next_times,
            params=costate_params,
            current_control=controls,
            num_antithetic=1,
        )
        dt = jnp.asarray(self.problem.time_grid.dt, dtype=states.dtype)
        terminal = jnp.asarray(self.problem.time_grid.t1, dtype=states.dtype)
        rho = (terminal - next_times) / (terminal - times)
        gamma = jnp.sqrt(dt * rho)[:, None, None] * self._sigma[None, :, :]
        innovation = decomposed.innovation[:, 0, :]
        perturbation = jnp.einsum("bij,bj->bi", gamma, innovation)
        plus_state = decomposed.mean_state + perturbation
        minus_state = decomposed.mean_state - perturbation
        future_noise = jax.random.normal(
            suffix_key,
            (self.stochastic_steps, size, self.problem.dim),
            dtype=states.dtype,
        )

        def running_value(state: Array, time: Array) -> Array:
            if directional_cost.running_cost is None:
                return jnp.zeros((state.shape[0],), dtype=state.dtype)
            return jax.lax.stop_gradient(
                jnp.asarray(
                    directional_cost.running_cost(state, time, batch_endpoint),
                    dtype=state.dtype,
                ).reshape((state.shape[0],))
            )

        def suffix_return(initial_state: Array) -> tuple[Array, Array]:
            initial_cost = running_value(initial_state, next_times)
            initial_carry = (
                initial_state,
                dt * initial_cost,
                jnp.all(jnp.isfinite(initial_cost)),
            )

            @jax.checkpoint
            def scan_step(
                carry: tuple[Array, Array, Array],
                inputs: tuple[Array, Array],
            ) -> tuple[tuple[Array, Array, Array], None]:
                state, value, all_values_finite = carry
                step_index, step_noise = inputs
                step_time = rollout.times[step_index]
                step_next_time = rollout.times[step_index + 1]
                eligible = step_index >= anchors + 1
                step_control = actor(state, step_time, batch_endpoint)
                step_rho = (terminal - step_next_time) / (terminal - step_time)
                step_gamma = jnp.sqrt(dt * step_rho) * self._sigma
                proposal = (
                    step_rho * state
                    + (1.0 - step_rho) * batch_endpoint
                    + jnp.sqrt(dt) * (step_control @ step_gamma.T)
                    + step_noise @ step_gamma.T
                )
                arrival_cost = running_value(
                    proposal,
                    jnp.full((size,), step_next_time, dtype=states.dtype),
                )
                # The vectorized diagnostic evaluates every row at every loop
                # iteration before masking by its sampled anchor.  Account for
                # those physical oracle evaluations and fail closed on all of
                # them; a masked NaN must never disappear through jnp.where.
                all_values_finite = all_values_finite & jnp.all(jnp.isfinite(arrival_cost))
                increment = dt * (0.5 * jnp.sum(step_control**2, axis=-1) + arrival_cost)
                value = value + jnp.where(eligible, increment, 0.0)
                state = jnp.where(eligible[:, None], proposal, state)
                return (state, value, all_values_finite), None

            (_, value, all_values_finite), _ = jax.lax.scan(
                scan_step,
                initial_carry,
                (jnp.arange(self.stochastic_steps), future_noise),
            )
            return jax.lax.stop_gradient(value), all_values_finite

        positive_return, positive_values_finite = suffix_return(plus_state)
        negative_return, negative_values_finite = suffix_return(minus_state)
        score_baseline = critic_baseline_fn(states, times, batch_endpoint)
        baseline_finite = jnp.all(jnp.isfinite(score_baseline))
        direct = assemble_antithetic_direct_action_score(
            (positive_return - score_baseline)[:, None],
            (negative_return - score_baseline)[:, None],
            decomposed.innovation,
            dt,
        )
        finite = (
            jnp.all(decomposed.finite)
            & jnp.all(direct.finite)
            & positive_values_finite
            & negative_values_finite
            & baseline_finite
        )
        if not bool(jax.device_get(finite)):
            raise FloatingPointError("nonfinite direct action-score diagnostic")
        difference = direct.target - decomposed.target
        numerator = jnp.sum(direct.target * decomposed.target)
        denominator = jnp.linalg.norm(direct.target) * jnp.linalg.norm(decomposed.target)
        cosine = numerator / jnp.maximum(
            denominator,
            jnp.asarray(jnp.finfo(states.dtype).tiny, dtype=states.dtype),
        )
        oracle_queries = (
            0 if directional_cost.running_cost is None else 2 * size * (self.stochastic_steps + 2)
        )
        return {
            "enabled": True,
            "tangent_free": True,
            "policy_fingerprint": self._policy_fingerprint(direction, actor_params),
            "costate_parameter_fingerprint": self._parameter_tree_fingerprint(costate_params),
            "sample_count": size,
            "physical_suffix_return_queries": int(direct.physical_return_queries),
            "physical_value_oracle_queries": oracle_queries,
            "target_rmse_vs_mam": float(jax.device_get(jnp.sqrt(jnp.mean(difference**2)))),
            "target_cosine_vs_mam": float(jax.device_get(cosine)),
            "diagnostic_not_ground_truth": True,
            "used_for_actor_selection": False,
            "cross_fitted_value_baseline_subtracted": True,
        }

    def _actor_targets(
        self,
        key: PRNGKey,
        rollout: Any,
        endpoint: Array,
        inner: MalliavinAdjointInnerSolver,
        costate_params: Params,
        directional_cost: ValueOnlyCost,
    ) -> Array:
        """Return a stopped antithetic plug-in action label.

        The arrival correction is exact for the pinned transition, but the
        continuation term plugs in the learned, stopped costate.  Consequently
        this label equals the exact discrete action estimand only when that
        costate is exact.
        """
        del directional_cost
        batch_size = endpoint.shape[0]
        times = jnp.asarray(self.problem.time_grid.times, dtype=rollout.states.dtype)
        departures = times[: self.stochastic_steps]
        arrivals = times[1 : self.stochastic_steps + 1]
        states = rollout.states[:, : self.stochastic_steps, :]
        repeated_endpoints = jnp.broadcast_to(endpoint[:, None, :], states.shape)
        repeated_departures = jnp.broadcast_to(
            departures[None, :], (batch_size, self.stochastic_steps)
        )
        repeated_arrivals = jnp.broadcast_to(arrivals[None, :], (batch_size, self.stochastic_steps))
        flat_size = batch_size * self.stochastic_steps
        target_batch = inner.make_action_target_batch(
            key,
            states.reshape((flat_size, self.problem.dim)),
            repeated_departures.reshape((flat_size,)),
            repeated_endpoints.reshape((flat_size, self.problem.dim)),
            next_time=repeated_arrivals.reshape((flat_size,)),
            params=costate_params,
            current_control=rollout.controls.reshape((flat_size, self.problem.dim)),
            num_antithetic=1,
        )
        targets_finite = jnp.all(target_batch.finite) & jnp.all(jnp.isfinite(target_batch.target))
        if not bool(jax.device_get(targets_finite)):
            raise FloatingPointError("nonfinite arrival-correct actor target; refusing fit")
        return target_batch.target.reshape((batch_size, self.stochastic_steps, self.problem.dim))

    def _fit_actor(self, rollout: Any, endpoint: Array, targets: Array) -> Array:
        endpoint_steps = jnp.broadcast_to(
            endpoint[:, None, :],
            (endpoint.shape[0], self.stochastic_steps, endpoint.shape[-1]),
        )
        states = rollout.states[:, : self.stochastic_steps, :]
        ones = jnp.ones((*states.shape[:2], 1), dtype=states.dtype)
        features = jnp.concatenate([states, endpoint_steps, ones], axis=-1)
        return _ridge_per_time(features, targets, self.config.actor_ridge)

    def _fit_actor_streaming(
        self,
        key: PRNGKey,
        start: Array,
        endpoint: Array,
        direction: str,
        current: _FieldParams,
        inner: MalliavinAdjointInnerSolver,
        costate_params: Params,
        directional_cost: ValueOnlyCost,
    ) -> tuple[_FieldParams, Array]:
        """Fit an arrival-correct actor from independent static microbatches."""
        key, fit_key = jax.random.split(key)
        microbatch_size = self.microbatch_size
        accumulation_steps = self.execution.effective_batch_size // microbatch_size
        feature_dim = 2 * self.problem.dim + 1
        if self.config.actor_model == "affine_reference":
            gram: Array | None = jnp.zeros(
                (self.stochastic_steps, feature_dim, feature_dim),
                dtype=start.dtype,
            )
            rhs: Array | None = jnp.zeros(
                (self.stochastic_steps, feature_dim, self.problem.dim),
                dtype=start.dtype,
            )
        else:
            gram = None
            rhs = None
        nonlinear_states: list[Array] = []
        nonlinear_times: list[Array] = []
        nonlinear_endpoints: list[Array] = []
        nonlinear_targets: list[Array] = []
        target_norm_sum = jnp.asarray(0.0, dtype=start.dtype)
        for _microbatch in range(accumulation_steps):
            key, index_key, rollout_key, target_key = jax.random.split(key, 4)
            indices = jax.random.randint(index_key, (microbatch_size,), 0, start.shape[0])
            batch_start = start[indices]
            batch_endpoint = endpoint[indices]
            rollout = simulate_pinned_brownian_rollout_matrix_free(
                rollout_key,
                batch_start,
                batch_endpoint,
                self.problem.time_grid.times,
                self._sigma,
                self._actor_fn(current, direction),
                diffusion_rcond=self.config.costate.diffusion_rcond,
            )
            targets = self._actor_targets(
                target_key,
                rollout,
                batch_endpoint,
                inner,
                costate_params,
                directional_cost,
            )
            endpoint_steps = jnp.broadcast_to(
                batch_endpoint[:, None, :],
                (
                    microbatch_size,
                    self.stochastic_steps,
                    self.problem.dim,
                ),
            )
            states = rollout.states[:, : self.stochastic_steps, :]
            if self.config.actor_model == "affine_reference":
                assert gram is not None and rhs is not None
                ones = jnp.ones((*states.shape[:2], 1), dtype=states.dtype)
                features = jnp.concatenate([states, endpoint_steps, ones], axis=-1)
                gram = gram + jnp.einsum("bnf,bng->nfg", features, features)
                rhs = rhs + jnp.einsum("bnf,bnd->nfd", features, targets)
            else:
                departures = jnp.asarray(
                    self.problem.time_grid.times[: self.stochastic_steps],
                    dtype=states.dtype,
                )
                nonlinear_states.append(states.reshape((-1, self.problem.dim)))
                nonlinear_times.append(
                    jnp.broadcast_to(
                        departures[None, :],
                        (microbatch_size, self.stochastic_steps),
                    ).reshape((-1,))
                )
                nonlinear_endpoints.append(endpoint_steps.reshape((-1, self.problem.dim)))
                nonlinear_targets.append(
                    jax.lax.stop_gradient(targets.reshape((-1, self.problem.dim)))
                )
            target_norm_sum = target_norm_sum + jnp.sum(jnp.linalg.norm(targets, axis=-1))
        count = jnp.asarray(self.execution.effective_batch_size, dtype=start.dtype)
        if self.config.actor_model == "affine_reference":
            assert gram is not None and rhs is not None
            eye = jnp.eye(feature_dim, dtype=start.dtype)
            candidate: _FieldParams = jnp.asarray(
                jax.vmap(jnp.linalg.solve)(
                    gram / count + self.config.actor_ridge * eye[None, :, :],
                    rhs / count,
                )
            )
            if not bool(jax.device_get(_tree_all_finite(candidate))):
                raise FloatingPointError("nonfinite streaming actor fit; refusing proposal")
        else:
            assert self._actor_field is not None
            row_count = self.execution.effective_batch_size * self.stochastic_steps
            actor_dataset = MAMActorDataset(
                states=jnp.concatenate(nonlinear_states, axis=0),
                times=jnp.concatenate(nonlinear_times, axis=0),
                endpoints=jnp.concatenate(nonlinear_endpoints, axis=0),
                directions=jnp.full(
                    (row_count,),
                    _direction_value(direction, start.dtype),
                    dtype=start.dtype,
                ),
                targets=jnp.concatenate(nonlinear_targets, axis=0),
            )
            candidate_state = self._actor_field.fit(fit_key, actor_dataset)
            candidate = _single_component_mixture("actor", candidate_state)
        mean_target_norm = target_norm_sum / (
            count * jnp.asarray(self.stochastic_steps, dtype=start.dtype)
        )
        return candidate, mean_target_norm

    def _objective(self, rollout: Any, value_cost: ValueOnlyCost) -> Array:
        running = value_cost.running_values(rollout.states, rollout.times, rollout.context)
        dt = jnp.asarray(self.problem.time_grid.dt, dtype=rollout.states.dtype)
        potential = jnp.sum(running[:, 1:-1], axis=1)
        energy = 0.5 * jnp.sum(rollout.controls**2, axis=(1, 2))
        return dt * (potential + energy)

    def _mix_actor(
        self,
        current: _FieldParams,
        candidate: _FieldParams,
        eta: float,
    ) -> _FieldParams:
        if self.config.actor_model == "affine_reference":
            current_array = cast(Array, current)
            candidate_array = cast(Array, candidate)
            return (1.0 - eta) * current_array + eta * candidate_array
        if not isinstance(current, _NonlinearFieldMixture) or not isinstance(
            candidate, _NonlinearFieldMixture
        ):
            raise TypeError("nonlinear actor interpolation requires actor mixtures")
        return _mix_nonlinear_fields(
            current,
            candidate,
            1.0 - eta,
            eta,
            maximum_components=self.config.maximum_actor_components,
        )

    def _accept_actor(
        self,
        key: PRNGKey,
        start: Array,
        endpoint: Array,
        direction: str,
        current: _FieldParams,
        candidate: _FieldParams,
        value_cost: ValueOnlyCost,
        critic_baseline_fn: Callable[[Array, Array, Array], Array],
    ) -> tuple[_FieldParams, dict[str, Any]]:
        size = self.config.acceptance_size
        (
            selection_index_key,
            selection_rollout_key,
            confirmation_index_key,
            confirmation_rollout_key,
        ) = jax.random.split(key, 4)

        def evaluate_split(
            index_key: PRNGKey,
            rollout_key: PRNGKey,
            actor_params: _FieldParams,
        ) -> tuple[Array, Array, Array]:
            indices = jax.random.randint(index_key, (size,), 0, start.shape[0])
            split_start = start[indices]
            split_endpoint = endpoint[indices]
            rollout = simulate_pinned_brownian_rollout_matrix_free(
                rollout_key,
                split_start,
                split_endpoint,
                self.problem.time_grid.times,
                self._sigma,
                self._actor_fn(actor_params, direction),
                diffusion_rcond=self.config.costate.diffusion_rcond,
            )
            return self._objective(rollout, value_cost), split_start, split_endpoint

        current_selection, selection_start, selection_endpoint = evaluate_split(
            selection_index_key, selection_rollout_key, current
        )
        selection_times = jnp.full(
            (size,),
            self.problem.time_grid.t0,
            dtype=selection_start.dtype,
        )
        selection_baseline = critic_baseline_fn(
            selection_start,
            selection_times,
            selection_endpoint,
        )
        if not bool(jax.device_get(jnp.all(jnp.isfinite(selection_baseline)))):
            raise FloatingPointError("nonfinite critic baseline on actor selection split")
        selection_critic_rmse = float(
            jax.device_get(jnp.sqrt(jnp.mean((selection_baseline - current_selection) ** 2)))
        )
        proposals = [self._mix_actor(current, candidate, eta) for eta in self.config.line_search]
        candidate_selection = []
        for proposal in proposals:
            rollout = simulate_pinned_brownian_rollout_matrix_free(
                selection_rollout_key,
                selection_start,
                selection_endpoint,
                self.problem.time_grid.times,
                self._sigma,
                self._actor_fn(proposal, direction),
                diffusion_rcond=self.config.costate.diffusion_rcond,
            )
            candidate_selection.append(self._objective(rollout, value_cost))
        selection = select_line_search_candidate(
            jnp.asarray(self.config.line_search, dtype=start.dtype),
            current_selection,
            jnp.stack(candidate_selection),
            z_value=self.config.one_sided_z,
            minimum_improvement=self.config.improvement_tolerance,
        )
        selection_passed = bool(jax.device_get(selection.has_acceptable_candidate))
        selected_index = int(jax.device_get(selection.selected_index))
        selected_eta = (
            float(jax.device_get(selection.selected_step_size)) if selection_passed else 0.0
        )
        selection_records = []
        for index, eta in enumerate(self.config.line_search):
            selection_records.append(
                {
                    "eta": eta,
                    "mean_difference": float(
                        jax.device_get(selection.statistics.mean_delta[index])
                    ),
                    "standard_error": float(
                        jax.device_get(selection.statistics.standard_error[index])
                    ),
                    "upper_confidence_bound": float(
                        jax.device_get(selection.statistics.upper_confidence_bound[index])
                    ),
                    "accepted": bool(jax.device_get(selection.statistics.accepted[index])),
                }
            )
        if not selection_passed:
            return current, {
                "actor_update_accepted": False,
                "accepted_step_size": 0.0,
                "line_search": selection_records,
                "confirmation": None,
                "acceptance_independent_of_actor_fit": True,
                "acceptance_independence_scope": "conditional_on_fixed_endpoint_pair_cache",
                "selection_and_confirmation_streams_disjoint": True,
                "acceptance_uses_paired_common_noise": True,
                "critic_current_policy_rmse": selection_critic_rmse,
                "critic_calibration_split": "line_search_selection",
                "confidence_method": "normal_clt_approximation",
            }

        selected = proposals[selected_index]
        confirmation_indices = jax.random.randint(
            confirmation_index_key, (size,), 0, start.shape[0]
        )
        confirmation_start = start[confirmation_indices]
        confirmation_endpoint = endpoint[confirmation_indices]
        current_confirmation_rollout = simulate_pinned_brownian_rollout_matrix_free(
            confirmation_rollout_key,
            confirmation_start,
            confirmation_endpoint,
            self.problem.time_grid.times,
            self._sigma,
            self._actor_fn(current, direction),
            diffusion_rcond=self.config.costate.diffusion_rcond,
        )
        candidate_confirmation_rollout = simulate_pinned_brownian_rollout_matrix_free(
            confirmation_rollout_key,
            confirmation_start,
            confirmation_endpoint,
            self.problem.time_grid.times,
            self._sigma,
            self._actor_fn(selected, direction),
            diffusion_rcond=self.config.costate.diffusion_rcond,
        )
        confirmation = paired_objective_statistics(
            self._objective(current_confirmation_rollout, value_cost),
            self._objective(candidate_confirmation_rollout, value_cost),
            z_value=self.config.one_sided_z,
            minimum_improvement=self.config.improvement_tolerance,
        )
        confirmed = bool(jax.device_get(confirmation.accepted))
        confirmation_baseline = critic_baseline_fn(
            confirmation_start,
            jnp.full(
                (size,),
                self.problem.time_grid.t0,
                dtype=confirmation_start.dtype,
            ),
            confirmation_endpoint,
        )
        confirmation_current_objective = self._objective(current_confirmation_rollout, value_cost)
        if not bool(jax.device_get(jnp.all(jnp.isfinite(confirmation_baseline)))):
            raise FloatingPointError("nonfinite critic baseline on actor confirmation split")
        confirmation_critic_rmse = float(
            jax.device_get(
                jnp.sqrt(jnp.mean((confirmation_baseline - confirmation_current_objective) ** 2))
            )
        )
        accepted = selected if confirmed else current
        return accepted, {
            "actor_update_accepted": confirmed,
            "accepted_step_size": selected_eta if confirmed else 0.0,
            "line_search": selection_records,
            "confirmation": {
                "mean_difference": float(jax.device_get(confirmation.mean_delta)),
                "standard_error": float(jax.device_get(confirmation.standard_error)),
                "upper_confidence_bound": float(
                    jax.device_get(confirmation.upper_confidence_bound)
                ),
                "accepted": confirmed,
            },
            "acceptance_independent_of_actor_fit": True,
            "acceptance_independence_scope": "conditional_on_fixed_endpoint_pair_cache",
            "selection_and_confirmation_streams_disjoint": True,
            "acceptance_uses_paired_common_noise": True,
            "critic_current_policy_rmse": confirmation_critic_rmse,
            "critic_calibration_split": "untouched_confirmation",
            "confidence_method": "normal_clt_approximation",
        }

    def _projection_data(
        self,
        rollout: Any,
        endpoint: Array,
        direction: str,
    ) -> tuple[Array, Array, Array]:
        times = jnp.asarray(self.problem.time_grid.times, dtype=rollout.states.dtype)
        dt = jnp.asarray(self.problem.time_grid.dt, dtype=rollout.states.dtype)
        terminal = times[-1]
        departures = times[: self.stochastic_steps]
        arrivals = times[1 : self.stochastic_steps + 1]
        rho = (terminal - arrivals) / (terminal - departures)
        sigma_control = rollout.controls @ self._sigma.T
        controlled_predictions = endpoint[:, None, :] + (
            (terminal - departures)[None, :, None] * jnp.sqrt(rho)[None, :, None] * sigma_control
        )
        # The final deterministic pin supplies the last finite endpoint label.
        predictions = jnp.concatenate([controlled_predictions, endpoint[:, None, :]], axis=1)
        states = rollout.states[:, :-1, :]
        physical_times = _projection_physical_times(
            self.problem,
            rollout.states.dtype,
            direction,
        )
        del dt
        return states, physical_times, jax.lax.stop_gradient(predictions)

    def solve(
        self,
        key: PRNGKey,
        endpoint_pairs: EndpointPairBatch,
        direction: str,
    ) -> ConditionalMAMResult:
        start, endpoint = self._local_endpoints(endpoint_pairs, direction)
        dtype = start.dtype
        current = self._actor_params.get(direction, self._zero_actor_params(dtype))
        directional_cost = self._directional_cost(direction)
        iteration_root_key, final_costate_key, output_key = jax.random.split(key, 3)
        acceptance_history: list[dict[str, Any]] = []
        value_critic_history: list[dict[str, Any]] = []
        direct_score_history: list[dict[str, Any]] = []
        costate_history: list[Array] = []
        target_norm_history: list[Array] = []
        consecutive_rejections = 0
        stopped_after_rejections = False
        costate_params: Params | None = None
        for policy_iteration in range(self.config.policy_iterations):
            iteration_key = jax.random.fold_in(iteration_root_key, policy_iteration)
            costate_key, critic_key, direct_key, actor_key, accept_key = jax.random.split(
                iteration_key, 5
            )
            value_critic_metrics, critic_dataset, critic = self._fit_value_critic(
                critic_key,
                start,
                endpoint,
                direction,
                current,
                directional_cost,
            )
            critic_baseline_fn = self._critic_baseline_fn(direction, critic)
            inner, costate_params, costate_losses = self._train_costate(
                costate_key,
                endpoint_pairs,
                direction,
                current,
            )
            value_critic_metrics = self._complete_value_critic_metrics(
                value_critic_metrics,
                critic_dataset,
                critic,
                self._value_critic_state[direction],
                inner,
                costate_params,
            )
            value_critic_history.append(value_critic_metrics)
            direct_score_metrics = self._direct_action_score_diagnostic(
                direct_key,
                start,
                endpoint,
                direction,
                current,
                directional_cost,
                inner,
                costate_params,
                critic_baseline_fn,
            )
            direct_score_history.append(direct_score_metrics)
            candidate, target_norm = self._fit_actor_streaming(
                actor_key,
                start,
                endpoint,
                direction,
                current,
                inner,
                costate_params,
                directional_cost,
            )
            updated, acceptance = self._accept_actor(
                accept_key,
                start,
                endpoint,
                direction,
                current,
                candidate,
                directional_cost,
                critic_baseline_fn,
            )
            acceptance = {**acceptance, "policy_iteration": policy_iteration}
            acceptance_history.append(acceptance)
            costate_history.append(costate_losses)
            target_norm_history.append(target_norm)
            if acceptance["actor_update_accepted"]:
                current = updated
                consecutive_rejections = 0
            else:
                consecutive_rejections += 1
                if consecutive_rejections >= self.config.maximum_consecutive_rejections:
                    stopped_after_rejections = True
                    break
        assert costate_params is not None
        output_actor_fingerprint = self._policy_fingerprint(direction, current)
        pre_refresh_costate_policy_fingerprint = self._costate_policy_fingerprint[direction]
        final_costate_refresh_executed = (
            pre_refresh_costate_policy_fingerprint != output_actor_fingerprint
        )
        final_costate_refresh_losses: Array | None = None
        if final_costate_refresh_executed:
            if not bool(acceptance_history[-1]["actor_update_accepted"]):
                raise AssertionError(
                    "costate/output-actor mismatch without a final accepted actor update"
                )
            _, costate_params, final_costate_refresh_losses = self._train_costate(
                final_costate_key,
                endpoint_pairs,
                direction,
                current,
            )
        final_costate_policy_fingerprint = self._costate_policy_fingerprint[direction]
        if final_costate_policy_fingerprint != output_actor_fingerprint:
            raise AssertionError("stored costate is not conditioned on the output actor")
        costate_parameter_fingerprint = self._parameter_tree_fingerprint(costate_params)
        self._actor_params[direction] = current
        output_rollout = simulate_pinned_brownian_rollout_matrix_free(
            output_key,
            start,
            endpoint,
            self.problem.time_grid.times,
            self._sigma,
            self._actor_fn(current, direction),
            diffusion_rcond=self.config.costate.diffusion_rcond,
        )
        projection_states, projection_times, endpoint_predictions = self._projection_data(
            output_rollout, endpoint, direction
        )
        local_paths = output_rollout.states
        chronological = local_paths if direction == "f" else local_paths[:, ::-1, :]
        endpoint_error = jnp.max(jnp.abs(local_paths[:, -1, :] - endpoint))
        exact_endpoint = bool(jax.device_get(endpoint_error == 0))
        if not exact_endpoint:
            raise AssertionError("conditional pinned sampler did not preserve its endpoint")
        certified_work = completed_conditional_solve_work(
            num_steps=self.num_steps,
            effective_batch_size=self.execution.effective_batch_size,
            costate_steps=self.config.costate_steps,
            value_critic_training_steps=self.config.value_critic.training_steps,
            actor_field_training_steps=(
                self.config.actor_field_config.training_steps
                if self.config.actor_model == "nonlinear"
                else 0
            ),
            direct_score_diagnostic_size=self.config.direct_score_diagnostic_size,
            acceptance_size=self.config.acceptance_size,
            line_search_candidates=len(self.config.line_search),
            pair_batch_size=endpoint_pairs.batch_size,
            policy_iterations_completed=len(acceptance_history),
            actor_confirmation_executed=tuple(
                item["confirmation"] is not None for item in acceptance_history
            ),
            actor_update_accepted=tuple(
                bool(item["actor_update_accepted"]) for item in acceptance_history
            ),
            final_costate_refresh_executed=final_costate_refresh_executed,
            running_cost_oracle_present=directional_cost.running_cost is not None,
        )
        work_accounting = _work_accounting_record(
            certified_work,
            scope="successful_builtin_conditional_solve",
        )
        metrics: dict[str, Any] = {
            "actor_update_accepted": any(
                item["actor_update_accepted"] for item in acceptance_history
            ),
            "actor_acceptance_history": acceptance_history,
            "actor_iterations_completed": len(acceptance_history),
            "consecutive_rejections": consecutive_rejections,
            "stopped_after_consecutive_rejections": stopped_after_rejections,
            "costate_loss_final": float(
                jax.device_get(
                    costate_history[-1][-1]
                    if final_costate_refresh_losses is None
                    else final_costate_refresh_losses[-1]
                )
            ),
            "costate_loss_history": jnp.stack(costate_history),
            "final_costate_refresh_executed": final_costate_refresh_executed,
            "final_costate_refresh_loss": (
                None
                if final_costate_refresh_losses is None
                else float(jax.device_get(final_costate_refresh_losses[-1]))
            ),
            "output_actor_fingerprint": output_actor_fingerprint,
            "pre_refresh_costate_policy_fingerprint": (pre_refresh_costate_policy_fingerprint),
            "costate_policy_fingerprint": final_costate_policy_fingerprint,
            "costate_parameter_fingerprint": costate_parameter_fingerprint,
            "actor_costate_policy_aligned": True,
            "value_critic": value_critic_history[-1],
            "value_critic_history": value_critic_history,
            "value_critic_policy_fingerprint": value_critic_history[-1]["policy_fingerprint"],
            "value_critic_matches_output_actor": (
                value_critic_history[-1]["policy_fingerprint"] == output_actor_fingerprint
            ),
            "direct_action_score": direct_score_history[-1],
            "direct_action_score_history": direct_score_history,
            "direct_action_score_policy_fingerprint": direct_score_history[-1][
                "policy_fingerprint"
            ],
            "direct_action_score_matches_output_actor": (
                direct_score_history[-1]["policy_fingerprint"] == output_actor_fingerprint
            ),
            "direct_action_score_scope": "last_policy_iteration_before_actor_acceptance",
            "actor_target_norm": float(jax.device_get(jnp.mean(jnp.stack(target_norm_history)))),
            "gradient_accumulation_steps": (
                self.execution.effective_batch_size // self.microbatch_size
            ),
            "microbatch_size": self.microbatch_size,
            "effective_batch_size": self.execution.effective_batch_size,
            "exact_conditional_endpoint": True,
            "uses_arrival_correction": True,
            "actor_target_semantics": "stopped_antithetic_costate_plugin",
            "actor_target_exact_only_with_exact_costate": True,
            "matrix_free_costate_labels": bool(self.config.costate.matrix_free_labels),
            "matrix_free_transition_law_provenance": "fused_same_control_callback",
            "actor_model": self.config.actor_model,
            "actor_mixture_components": (
                len(current.components) if isinstance(current, _NonlinearFieldMixture) else 0
            ),
            "work_accounting": work_accounting,
        }
        return ConditionalMAMResult(
            paths=chronological,
            local_paths=local_paths,
            controls=output_rollout.controls,
            projection_states=projection_states,
            projection_times=projection_times,
            endpoint_predictions=endpoint_predictions,
            actor_params=current,
            costate_params=costate_params,
            metrics=metrics,
            direction=direction,
            exact_conditional_endpoint=True,
            certified_work_counters=certified_work,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "backend_status": self.status,
            "actor_params": self._actor_params,
            "costate_params": self._costate_params,
            "costate_opt_state": self._costate_opt_state,
            "costate_policy_fingerprint": self._costate_policy_fingerprint,
            "value_critic_state": self._value_critic_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schema_version",
            "backend_status",
            "actor_params",
            "costate_params",
            "costate_opt_state",
            "costate_policy_fingerprint",
            "value_critic_state",
        }
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("conditional MAM state fields do not match schema")
        if state["schema_version"] != 1 or state["backend_status"] != self.status:
            raise ValueError("conditional MAM state schema/status mismatch")
        self._actor_params = dict(state["actor_params"])
        self._costate_params = dict(state["costate_params"])
        self._costate_opt_state = dict(state["costate_opt_state"])
        self._costate_policy_fingerprint = dict(state["costate_policy_fingerprint"])
        self._value_critic_state = dict(state["value_critic_state"])

    def _validate_actor_checkpoint(self, value: _FieldParams) -> None:
        expected_dtype = np.dtype(self.execution.production_dtype)
        if self.config.actor_model == "affine_reference":
            actor = np.asarray(jax.device_get(value))
            expected_shape = (
                self.stochastic_steps,
                2 * self.problem.dim + 1,
                self.problem.dim,
            )
            if actor.shape != expected_shape or actor.dtype != expected_dtype:
                raise ValueError("conditional checkpoint actor shape/dtype mismatch")
            if not np.all(np.isfinite(actor)):
                raise FloatingPointError("conditional checkpoint actor is nonfinite")
            return
        if not isinstance(value, _NonlinearFieldMixture) or value.kind != "actor":
            raise TypeError("conditional nonlinear actor checkpoint has the wrong type")
        weights = np.asarray(jax.device_get(value.weights))
        if weights.shape != (len(value.components),) or weights.dtype != expected_dtype:
            raise ValueError("conditional nonlinear actor mixture weight mismatch")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise FloatingPointError("conditional nonlinear actor weights are invalid")
        if float(np.sum(weights, dtype=np.float64)) > 1.0 + 32.0 * np.finfo(np.float32).eps:
            raise ValueError("conditional nonlinear actor mixture mass exceeds one")
        if len(value.components) > self.config.maximum_actor_components:
            raise ValueError("conditional nonlinear actor exceeds its component cap")
        fingerprints = [_prediction_parameter_fingerprint(item) for item in value.components]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("conditional nonlinear actor is not canonically deduplicated")
        assert self._actor_field is not None
        for component in value.components:
            self._actor_field.validate_state(component)

    def validate_checkpoint_progress(self, expected_updates: Mapping[str, int]) -> None:
        """Validate trained components and exact optimizer progress by direction."""

        expected_dtype = np.dtype(self.execution.production_dtype)
        expected_directions = set(expected_updates)

        def validate_tree_like(
            name: str,
            value: Any,
            template: Any,
        ) -> None:
            if jax.tree_util.tree_structure(value) != jax.tree_util.tree_structure(template):
                raise ValueError(f"conditional checkpoint {name} tree mismatch")
            leaves = jax.tree_util.tree_leaves(value)
            expected_leaves = jax.tree_util.tree_leaves(template)
            for leaf, expected in zip(leaves, expected_leaves, strict=True):
                array = np.asarray(jax.device_get(leaf))
                expected_array = np.asarray(jax.device_get(expected))
                if array.shape != expected_array.shape or array.dtype != expected_dtype:
                    raise ValueError(f"conditional checkpoint {name} shape/dtype mismatch")
                if not np.all(np.isfinite(array)):
                    raise FloatingPointError(f"conditional checkpoint {name} is nonfinite")

        stores: tuple[tuple[str, Mapping[str, object]], ...] = (
            ("actor_params", self._actor_params),
            ("costate_params", self._costate_params),
            ("costate_opt_state", self._costate_opt_state),
            ("costate_policy_fingerprint", self._costate_policy_fingerprint),
            ("value_critic_state", self._value_critic_state),
        )
        for name, store in stores:
            if set(store) != expected_directions:
                raise ValueError(
                    f"conditional checkpoint {name} directions disagree with outer progress"
                )
        for direction in expected_directions:
            actor_params = self._actor_params[direction]
            self._validate_actor_checkpoint(actor_params)
            actor_fingerprint = self._policy_fingerprint(direction, actor_params)
            if self._costate_policy_fingerprint[direction] != actor_fingerprint:
                raise ValueError(
                    "conditional checkpoint costate policy fingerprint disagrees with actor"
                )
            inner = MalliavinAdjointInnerSolver(
                self.problem,
                self._directional_cost(direction),
                mam_config=self.config.costate,
                control_fn=self._actor_fn(actor_params, direction),
            )
            template = jax.tree_util.tree_map(
                lambda value: jnp.asarray(value, dtype=self.execution.production_dtype),
                inner.init_params(jax.random.PRNGKey(0)),
            )
            costate_params = self._costate_params[direction]
            validate_tree_like("costate parameters", costate_params, template)
            optimizer = self._costate_opt_state[direction]
            if not isinstance(optimizer, AdamState):
                raise TypeError("conditional checkpoint costate optimizer has the wrong type")
            validate_tree_like("costate Adam first moment", optimizer.m, template)
            validate_tree_like("costate Adam second moment", optimizer.v, template)
            optimizer_step = np.asarray(jax.device_get(optimizer.step))
            if optimizer_step.shape != () or not np.issubdtype(
                optimizer_step.dtype,
                np.integer,
            ):
                raise TypeError("conditional checkpoint costate Adam step must be an integer")
            step_value = int(optimizer_step)
            if step_value != expected_updates[direction]:
                raise ValueError(
                    "conditional checkpoint costate Adam step disagrees with exact "
                    "per-direction progress"
                )
            critic = self._value_critic_state[direction]
            critic_config = replace(
                self.config.value_critic,
                microbatch_size=self.microbatch_size,
                effective_batch_size=self.execution.effective_batch_size,
            )
            CrossFittedValueCritic(critic_config).validate_result_state(
                critic,
                state_dim=self.problem.dim,
                context_dim=self.problem.dim,
                row_count=self.execution.effective_batch_size,
            )


class MarkovProjector:
    """Finite-grid Euler endpoint-field regression with per-time targets.

    This estimates a conditional-mean field in the chosen regressor class.  It
    is used as a practical Markov projection step, but is not an exact discrete
    Markov projection at finite data, optimization, or model capacity.
    """

    def __init__(
        self,
        problem: SBProblem,
        config: MarkovProjectionConfig,
        execution: MAMExecutionConfig | None = None,
    ):
        self.problem = problem
        self.config = config
        field_config = config.field_config
        if execution is not None:
            resolved = make_execution_plan(
                horizon=problem.time_grid.num_steps,
                state_dim=problem.dim,
                microbatch_size=execution.microbatch_size,
                effective_batch_size=execution.effective_batch_size,
                device_count=1,
            )
            field_config = replace(
                config.field_config,
                microbatch_size=resolved.microbatch_size,
                effective_batch_size=resolved.effective_batch_size,
            )
        self._field = (
            MAMEndpointProjectorField(problem.dim, problem.dim, field_config)
            if config.model == "nonlinear"
            else None
        )
        self._physical_times = {
            direction: _projection_physical_times(problem, jnp.float32, direction)
            for direction in ("f", "b")
        }

    def init_params(self, dtype: Any = jnp.float32) -> _FieldParams:
        if self.config.model == "nonlinear":
            if np.dtype(dtype) != np.dtype(np.float32):
                raise ValueError("nonlinear projection parameters are fixed to float32")
            return _empty_mixture("endpoint_projector")
        return jnp.zeros(
            (self.problem.time_grid.num_steps, self.problem.dim + 1, self.problem.dim),
            dtype=dtype,
        )

    def fit(
        self,
        key: PRNGKey,
        controlled_paths: ConditionalMAMResult,
        direction: str,
    ) -> ProjectionResult:
        if direction != controlled_paths.direction:
            raise ValueError("projection direction must match conditional result")
        states = controlled_paths.projection_states
        targets = controlled_paths.endpoint_predictions
        if self.config.model == "affine_reference":
            ones = jnp.ones((*states.shape[:2], 1), dtype=states.dtype)
            features = jnp.concatenate([states, ones], axis=-1)
            params: _FieldParams = _ridge_per_time(features, targets, self.config.ridge)
            predictions = jnp.einsum("bnf,nfd->bnd", features, cast(Array, params))
        else:
            assert self._field is not None
            batch_size, num_steps, dim = states.shape
            times = jnp.broadcast_to(
                controlled_paths.projection_times[None, :],
                (batch_size, num_steps),
            )
            dataset = MAMProjectionDataset(
                states=states.reshape((-1, dim)),
                times=times.reshape((-1,)),
                directions=jnp.full(
                    (batch_size * num_steps,),
                    _direction_value(direction, states.dtype),
                    dtype=states.dtype,
                ),
                targets=jax.lax.stop_gradient(targets.reshape((-1, dim))),
            )
            train_state = self._field.fit(key, dataset)
            params = _single_component_mixture("endpoint_projector", train_state)
            predictions = self.predict(
                params,
                states.reshape((-1, dim)),
                jnp.tile(jnp.arange(num_steps, dtype=jnp.int32), batch_size),
                direction,
            ).reshape(targets.shape)
        loss = jnp.mean((predictions - targets) ** 2)
        finite = jnp.isfinite(loss) & _tree_all_finite(
            params.weights if isinstance(params, _NonlinearFieldMixture) else params
        )
        return ProjectionResult(params, loss, finite, direction)

    def predict(
        self,
        params: _FieldParams,
        state: Array,
        local_step: Array,
        direction: str,
    ) -> Array:
        if direction not in self._physical_times:
            raise ValueError("direction must be 'f' or 'b'")
        state = jnp.atleast_2d(state)
        raw_step = jnp.asarray(local_step)
        if not jnp.issubdtype(raw_step.dtype, jnp.integer):
            raise TypeError("local_step must have an integer dtype")
        if raw_step.ndim == 0:
            raw_step = jnp.full((state.shape[0],), raw_step, dtype=raw_step.dtype)
        elif raw_step.shape != (state.shape[0],):
            raise ValueError("local_step must be scalar or have shape [batch]")
        valid_step = (raw_step >= 0) & (raw_step < self.problem.time_grid.num_steps)
        if not isinstance(raw_step, jax.core.Tracer):
            host_valid = np.asarray(jax.device_get(valid_step))
            if not np.all(host_valid):
                raise ValueError("local_step lies outside the projection grid")
        safe_step = jnp.clip(
            raw_step,
            0,
            self.problem.time_grid.num_steps - 1,
        ).astype(jnp.int32)
        if self.config.model == "nonlinear":
            if not isinstance(params, _NonlinearFieldMixture):
                raise TypeError("nonlinear projection parameters must be a field mixture")
            if params.kind != "endpoint_projector":
                raise ValueError("nonlinear projection mixture has the wrong kind")
            assert self._field is not None
            physical_time = jnp.asarray(
                self._physical_times[direction][safe_step],
                dtype=state.dtype,
            )
            directions = jnp.full(
                (state.shape[0],),
                _direction_value(direction, state.dtype),
                dtype=state.dtype,
            )
            value = jnp.zeros((state.shape[0], self.problem.dim), dtype=state.dtype)
            for weight, component in zip(params.weights, params.components, strict=True):
                prediction = endpoint_projector_field_predict(
                    self._field.factory,
                    component.params,
                    state,
                    physical_time,
                    directions,
                    self.problem.dim,
                )
                value = value + weight * prediction.value
            return jnp.where(valid_step[:, None], value, jnp.nan)
        features = jnp.concatenate(
            [state, jnp.ones((state.shape[0], 1), dtype=state.dtype)], axis=-1
        )
        selected = jnp.asarray(params, dtype=state.dtype)[safe_step]
        value = jnp.einsum("bf,bfd->bd", features, selected)
        return jnp.where(valid_step[:, None], value, jnp.nan)

    def mix(
        self,
        left: _FieldParams,
        right: _FieldParams,
        left_weight: float,
        right_weight: float,
    ) -> _FieldParams:
        if self.config.model == "affine_reference":
            left_array = cast(Array, left)
            right_array = cast(Array, right)
            return left_weight * left_array + right_weight * right_array
        if not isinstance(left, _NonlinearFieldMixture) or not isinstance(
            right, _NonlinearFieldMixture
        ):
            raise TypeError("nonlinear projection interpolation requires mixtures")
        return _mix_nonlinear_fields(
            left,
            right,
            left_weight,
            right_weight,
            maximum_components=self.config.maximum_components,
        )

    def parameter_dtype(self, params: _FieldParams) -> Any:
        if self.config.model == "nonlinear":
            return jnp.float32
        return jnp.asarray(params).dtype

    def validate_params(self, params: _FieldParams) -> None:
        """Strictly validate runtime/checkpoint projection state."""

        expected_dtype = np.dtype(np.float32)
        if self.config.model == "affine_reference":
            array = np.asarray(jax.device_get(params))
            expected_shape = (
                self.problem.time_grid.num_steps,
                self.problem.dim + 1,
                self.problem.dim,
            )
            if array.shape != expected_shape or array.dtype != expected_dtype:
                raise ValueError("affine projection parameter shape/dtype mismatch")
            if not np.all(np.isfinite(array)):
                raise FloatingPointError("affine projection parameters are nonfinite")
            return
        if not isinstance(params, _NonlinearFieldMixture):
            raise TypeError("nonlinear projection checkpoint must contain a field mixture")
        if params.kind != "endpoint_projector":
            raise ValueError("nonlinear projection mixture has the wrong kind")
        weights = np.asarray(jax.device_get(params.weights))
        if weights.shape != (len(params.components),) or weights.dtype != expected_dtype:
            raise ValueError("nonlinear projection mixture weight shape/dtype mismatch")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise FloatingPointError("nonlinear projection mixture weights are invalid")
        if float(np.sum(weights, dtype=np.float64)) > 1.0 + 32.0 * np.finfo(np.float32).eps:
            raise ValueError("nonlinear projection mixture mass exceeds one")
        if len(params.components) > self.config.maximum_components:
            raise ValueError("nonlinear projection mixture exceeds its component cap")
        fingerprints = [_prediction_parameter_fingerprint(item) for item in params.components]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("nonlinear projection mixture is not canonically deduplicated")
        assert self._field is not None
        for component in params.components:
            self._field.validate_state(component)


@dataclass
class MAMBridgeProcess(BridgeProcess):
    """Exact-grid runtime for the audited discrete MAM Markov chain."""

    projector: MarkovProjector | None = None

    def _validate_runtime_params(self) -> None:
        if not isinstance(self.params, dict) or set(self.params) != {"F", "B"}:
            raise ValueError("MAM bridge runtime parameters must contain exactly F and B")
        if self.projector is not None:
            for direction in ("F", "B"):
                self.projector.validate_params(self.params[direction])
            return
        expected_shape = (
            self.time_grid.num_steps,
            self.problem.dim + 1,
            self.problem.dim,
        )
        for direction in ("F", "B"):
            array = np.asarray(jax.device_get(self.params[direction]))
            if array.shape != expected_shape or array.dtype != np.dtype(np.float32):
                raise ValueError("affine runtime parameter shape/dtype mismatch")
            if not np.all(np.isfinite(array)):
                raise FloatingPointError("affine runtime parameters are nonfinite")

    def _sample_local_chain(
        self,
        key: PRNGKey,
        start: Array,
        params: _FieldParams,
        direction: str,
    ) -> Array:
        """Replay the exact discrete chain used by ``MAMBridgeSolver``.

        This is intentionally implemented here instead of delegating to the
        generic continuous-process integrator.  Besides preventing alternate
        solvers/sub-grids, it fixes the random-key schedule, projection-head
        indexing, and production dtype to the audited training semantics.
        """
        start = jnp.atleast_2d(start)
        num_steps = self.time_grid.num_steps
        dt = jnp.asarray(self.time_grid.dt, dtype=start.dtype)
        sigma = _constant_diffusion(self.problem, start.dtype)
        step_keys = jax.random.split(key, num_steps)

        def step(state: Array, inputs: tuple[Array, PRNGKey]) -> tuple[Array, Array]:
            index, step_key = inputs
            innovation = jax.random.normal(step_key, state.shape, dtype=state.dtype)
            if self.projector is None:
                features = jnp.concatenate(
                    [state, jnp.ones((state.shape[0], 1), dtype=state.dtype)], axis=-1
                )
                endpoint_prediction = features @ jnp.asarray(params, dtype=state.dtype)[index]
            else:
                endpoint_prediction = self.projector.predict(
                    params,
                    state,
                    index,
                    direction,
                )
            remaining = jnp.asarray(num_steps - index, dtype=state.dtype) * dt
            proposal = state + dt * (endpoint_prediction - state) / remaining
            proposal = proposal + jnp.sqrt(dt) * (innovation @ sigma.T)
            return proposal, proposal

        _, states_after = jax.lax.scan(
            step,
            start,
            (jnp.arange(num_steps, dtype=jnp.int32), step_keys),
        )
        return jnp.concatenate([start[:, None, :], jnp.swapaxes(states_after, 0, 1)], axis=1)

    def sample_paths(
        self,
        key: PRNGKey,
        num_samples: int,
        x0: Array | None = None,
        direction: str = "forward",
        return_full: bool = True,
        integrator: Any | None = None,
    ) -> TrajectoryBatch | Array:
        num_samples = _validate_public_sample_count(num_samples)
        if self.backend != "native":
            raise ValueError("MAM bridge runtime supports only the audited native backend")
        active_integrator = integrator or self.integrator
        if active_integrator is None or active_integrator.type is not IntegratorType.EULER_MARUYAMA:
            raise ValueError("MAM bridge runtime supports only Euler-Maruyama on its exact grid")
        if direction not in {"forward", "backward"}:
            raise ValueError("direction must be 'forward' or 'backward'")
        self._validate_runtime_params()
        dtype = (
            jnp.asarray(self.params["F"]).dtype
            if self.projector is None
            else self.projector.parameter_dtype(self.params["F"])
        )
        start_key, path_key = jax.random.split(key)
        start = (
            self._resolve_initial_state(start_key, num_samples, None, direction)
            if x0 is None
            else x0
        )
        start = _validated_public_start(
            start,
            num_samples=num_samples,
            dim=self.problem.dim,
            dtype=dtype,
            name="public start",
        )
        head = "F" if direction == "forward" else "B"
        direction_code = "f" if direction == "forward" else "b"
        local_paths = self._sample_local_chain(
            path_key,
            start,
            self.params[head],
            direction_code,
        )
        paths = local_paths if direction == "forward" else local_paths[:, ::-1, :]
        _validate_public_paths(
            paths,
            num_samples=num_samples,
            num_steps=self.time_grid.num_steps,
            dim=self.problem.dim,
            dtype=dtype,
        )
        if not return_full:
            return paths[:, -1, :] if direction == "forward" else paths[:, 0, :]
        return TrajectoryBatch(
            paths=paths,
            times=jnp.asarray(self.time_grid.times, dtype=dtype),
        )

    def rollout_from(
        self,
        key: PRNGKey,
        x0: Array,
        t0: float | None = None,
        t1: float | None = None,
        direction: str = "forward",
        return_full: bool = True,
        integrator: Any | None = None,
    ) -> TrajectoryBatch | Array:
        if (t0 is not None and t0 != self.time_grid.t0) or (
            t1 is not None and t1 != self.time_grid.t1
        ):
            raise ValueError("MAM bridge runtime does not support unaudited sub-grid rollouts")
        active_integrator = integrator or self.integrator
        if active_integrator is None or active_integrator.type is not IntegratorType.EULER_MARUYAMA:
            raise ValueError("MAM bridge runtime supports only Euler-Maruyama on its exact grid")
        host_x0 = np.asarray(jax.device_get(x0))
        if host_x0.ndim != 2:
            raise ValueError("rollout_from x0 must have shape [batch, dimension]")
        return self.sample_paths(
            key,
            int(host_x0.shape[0]),
            x0=x0,
            direction=direction,
            return_full=return_full,
            integrator=active_integrator,
        )

    def reverse(self) -> BridgeProcess:
        raise ValueError(
            "MAM bridge reverse() is disabled; sample the audited reverse direction explicitly"
        )


@dataclass
class MAMBridgeSolution(SBSolution):
    """Solution container that cannot leave the audited discrete runtime."""

    initial_sampler: Callable[[PRNGKey, int], Array] | None = None
    terminal_sampler: Callable[[PRNGKey, int], Array] | None = None
    projector: MarkovProjector | None = None

    def as_process(
        self,
        integrator: Any | None = None,
        backend: str = "native",
    ) -> MAMBridgeProcess:
        if backend != "native":
            raise ValueError("MAM bridge solution does not support a continuous/diffrax backend")
        active_integrator = integrator or self._integrator
        if active_integrator is None or active_integrator.type is not IntegratorType.EULER_MARUYAMA:
            raise ValueError("MAM bridge solution requires the audited Euler-Maruyama integrator")
        return MAMBridgeProcess(
            problem=self.problem,
            solver_type=self.solver_type,
            representation_type=self.representation,
            params=self.params,
            forward_drift_fn=self.get_forward_drift(),
            backward_drift_fn=self._backward_drift,
            score_fn=self._score_fn,
            integrator=active_integrator,
            metadata=dict(self.metadata),
            backend="native",
            initial_sampler=self.initial_sampler,
            terminal_sampler=self.terminal_sampler,
            projector=self.projector,
        )


class MAMBridgeSolver(SBSolver):
    """Experimental generalized MAM bridge with explicit endpoint audits.

    ``solver_type`` remains ``MALLIAVIN`` until a public enum/registry entry is
    intentionally added after the endpoint gate.  The metadata ``status`` is
    the authoritative distinction between a conditional foundation, an
    unverified global composition, and an empirically endpoint-passing run.
    """

    def __init__(
        self,
        problem: SBProblem,
        running_potential: ValueOnlyRunningPotential | ValueOnlyCost,
        config: MAMBridgeConfig | None = None,
        solver_config: SolverConfig | None = None,
        *,
        conditional_solver: ConditionalBridgeSolver | None = None,
        source_mode_label_fn: ModeLabelFn | None = None,
        source_num_modes: int | None = None,
        source_mode_proportion_fn: ModeProportionFn | None = None,
        target_mode_label_fn: ModeLabelFn | None = None,
        target_num_modes: int | None = None,
        target_mode_proportion_fn: ModeProportionFn | None = None,
    ):
        self.mam_bridge_config = config or MAMBridgeConfig()
        if self.mam_bridge_config.execution.allow_two_devices:
            raise NotImplementedError(
                "two-device execution helpers exist, but the global training loop is not "
                "yet wired for pmap; set allow_two_devices=False"
            )
        if isinstance(running_potential, ValueOnlyRunningPotential):
            value_cost = running_potential.as_value_only_cost()
        elif isinstance(running_potential, ValueOnlyCost):
            if running_potential.running_cost is None:
                value_cost = running_potential
            else:
                raw_running_cost = running_potential.running_cost

                def context_free_cost(x: Array, t: Array, context: Array) -> Array:
                    return raw_running_cost(x, t, jnp.zeros_like(context))

                value_cost = ValueOnlyCost(
                    running_cost=context_free_cost,
                    terminal_cost=running_potential.terminal_cost,
                    identifier=running_potential.identifier,
                )
        else:
            raise TypeError("running_potential must be ValueOnlyRunningPotential or ValueOnlyCost")
        if value_cost.terminal_cost is not None:
            raise ValueError("pinned MAM global v1 accepts additive running costs only")
        self.running_potential = running_potential
        self.value_cost = value_cost
        self._validate_problem(problem)
        super().__init__(problem, config=solver_config)
        if self.integrator.type is not IntegratorType.EULER_MARUYAMA:
            raise ValueError("MAMBridgeSolver requires the native Euler-Maruyama integrator")
        self._sigma = _constant_diffusion(
            problem, self.mam_bridge_config.execution.production_dtype
        )
        self.projector = MarkovProjector(
            problem,
            self.mam_bridge_config.projection,
            self.mam_bridge_config.execution,
        )
        self.conditional_solver: ConditionalBridgeSolver = (
            conditional_solver
            or MAMConditionalSolver(
                problem,
                value_cost,
                self.mam_bridge_config.conditional,
                self.mam_bridge_config.execution,
            )
        )
        self._injected_conditional_scientific_fingerprint: str | None = None
        if type(self.conditional_solver) is not MAMConditionalSolver:
            self._injected_conditional_scientific_fingerprint = (
                self._read_injected_conditional_scientific_fingerprint()
            )
        initial_conditional_state = self.conditional_solver.state_dict()
        if (
            not isinstance(initial_conditional_state, dict)
            or initial_conditional_state.get("schema_version") != 1
            or initial_conditional_state.get("backend_status") != self.conditional_solver.status
        ):
            raise ValueError(
                "conditional solver state_dict must expose schema_version=1 and backend_status"
            )
        self._initial_conditional_state = copy.deepcopy(initial_conditional_state)
        self._audit_history: list[dict[str, Any]] = []
        self._last_direction: str | None = None
        self._global_endpoint_pass = False
        self._source_mode_label_fn = source_mode_label_fn
        self._source_num_modes = source_num_modes
        self._source_mode_proportion_fn = source_mode_proportion_fn
        self._target_mode_label_fn = target_mode_label_fn
        self._target_num_modes = target_num_modes
        self._target_mode_proportion_fn = target_mode_proportion_fn
        for name, label_fn, num_modes, proportion_fn in (
            (
                "source",
                source_mode_label_fn,
                source_num_modes,
                source_mode_proportion_fn,
            ),
            (
                "target",
                target_mode_label_fn,
                target_num_modes,
                target_mode_proportion_fn,
            ),
        ):
            if label_fn is not None and proportion_fn is not None:
                raise ValueError(
                    f"{name} mode audit accepts a label or proportion callback, not both"
                )
            if label_fn is None:
                if num_modes is not None:
                    raise ValueError(
                        f"{name}_num_modes is meaningful only with a mode-label callback"
                    )
            else:
                if num_modes is None:
                    raise ValueError(f"{name}_num_modes is required with a mode-label callback")
                _require_integer(f"{name}_num_modes", num_modes, minimum=1)
        self._source_calibration: NullCalibrationResult | None = None
        self._target_calibration: NullCalibrationResult | None = None
        self._device_topology: DeviceTopology = discover_device_topology(max_devices=1)
        self._execution_plan: ExecutionPlan = make_execution_plan(
            horizon=problem.time_grid.num_steps,
            state_dim=problem.dim,
            microbatch_size=self.mam_bridge_config.execution.microbatch_size,
            effective_batch_size=self.mam_bridge_config.execution.effective_batch_size,
            device_count=1,
        )
        self._rng_ledger: RNGLedger | None = None
        self._resume_pairs: EndpointPairBatch | None = None
        self._completed_half_iterations = 0
        self._loss_history: list[float] = []
        self._last_metrics: dict[str, Any] = {}
        self._checkpoint_origin_device_topology: dict[str, Any] | None = None

    def _validate_problem(self, problem: SBProblem) -> None:
        if type(problem.reference) is not BrownianMotion:
            raise ValueError(
                "MAMBridgeSolver v1 requires an exact BrownianMotion reference; "
                "state/time-dependent subclasses are unsupported"
            )
        if problem.reference.dim != problem.dim:
            raise ValueError("BrownianMotion reference dimension must match the bridge problem")
        if problem.time_grid.num_steps < 3:
            raise ValueError("MAMBridgeSolver requires at least three time steps")
        sigma = _constant_diffusion(problem, jnp.float32)
        host_sigma = np.asarray(jax.device_get(sigma))
        if host_sigma.shape != (problem.dim, problem.dim) or not np.all(np.isfinite(host_sigma)):
            raise ValueError("MAMBridgeSolver v1 requires finite square constant diffusion")
        singular = np.linalg.svd(host_sigma, compute_uv=False)
        if singular[-1] <= 1e-8 * singular[0]:
            raise ValueError("MAMBridgeSolver v1 requires full-rank diffusion")

    @property
    def solver_type(self) -> SolverType:
        return SolverType.MALLIAVIN

    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.DRIFT

    @property
    def status(self) -> str:
        if self._global_endpoint_pass:
            return "GLOBAL_ENDPOINT_AUDIT_PASSED"
        return "EXPERIMENTAL_GLOBAL_ENDPOINT_UNVERIFIED"

    def init_params(self, key: PRNGKey) -> Params:
        del key
        dtype = self.mam_bridge_config.execution.production_dtype
        return {
            "F": self.projector.init_params(dtype),
            "B": self.projector.init_params(dtype),
        }

    def _validate_public_projection_params(self, params: Params) -> None:
        if not isinstance(params, dict) or set(params) != {"F", "B"}:
            raise ValueError("MAM bridge parameters must contain exactly F and B")
        for direction in ("F", "B"):
            try:
                self.projector.validate_params(params[direction])
            except (TypeError, ValueError, FloatingPointError) as exc:
                raise type(exc)(
                    f"public {direction} projection parameters are invalid: {exc}"
                ) from exc

    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: Any,
        batch_size: int,
    ) -> tuple[Params, Any, dict[str, Scalar]]:
        del key, params, opt_state, batch_size
        raise NotImplementedError("MAMBridgeSolver uses an alternating conditional/projection loop")

    def _sample_source(self, key: PRNGKey, size: int) -> Array:
        return jnp.asarray(
            self.problem.sample_source(key, size),
            dtype=self.mam_bridge_config.execution.production_dtype,
        )

    def _sample_target(self, key: PRNGKey, size: int) -> Array:
        return jnp.asarray(
            self.problem.sample_target(key, size),
            dtype=self.mam_bridge_config.execution.production_dtype,
        )

    def _initial_pair_cache(self, key: PRNGKey, size: int) -> EndpointPairBatch:
        source_key, target_key, permutation_key = jax.random.split(key, 3)
        source = self._sample_source(source_key, size)
        target = self._sample_target(target_key, size)
        permutation = jax.random.permutation(permutation_key, size)
        return EndpointPairBatch(source, target[permutation])

    def _validate_pair_cache(self, pairs: Any) -> None:
        if not isinstance(pairs, EndpointPairBatch):
            raise TypeError("MAM pair cache must be an EndpointPairBatch")
        expected_shape = (self.mam_bridge_config.outer.cache_size, self.problem.dim)
        expected_dtype = np.dtype(self.mam_bridge_config.execution.production_dtype)
        for name, value in (("source", pairs.source), ("target", pairs.target)):
            if value.shape != expected_shape:
                raise ValueError(f"pair-cache {name} must have shape {expected_shape}")
            if np.dtype(value.dtype) != expected_dtype:
                raise TypeError(f"pair-cache {name} must have dtype {expected_dtype}")
            if not bool(jax.device_get(jnp.all(jnp.isfinite(value)))):
                raise FloatingPointError(f"pair-cache {name} contains nonfinite values")

    def _damped_projection(
        self,
        previous: _FieldParams,
        fitted: _FieldParams,
    ) -> _FieldParams:
        damping = self.mam_bridge_config.projection.damping
        return self.projector.mix(previous, fitted, 1.0 - damping, damping)

    def _projection_endpoint_scores(
        self,
        key: PRNGKey,
        params: Params,
        direction: str,
    ) -> Array:
        """Return one sliced-W1 statistic per independent cloud replicate.

        Directions within one generated/reference cloud reduce Monte Carlo
        error in that cloud's sliced-W1 statistic; they are not independent
        samples for the acceptance confidence interval.  Reusing ``key`` for
        current and candidate fields preserves paired common randomness at the
        cloud, reference, path-noise, and slicing-direction levels.
        """
        config = self.mam_bridge_config.projection
        replicate_keys = jax.random.split(key, config.validation_replicates)

        def one_replicate(replicate_key: PRNGKey) -> Array:
            path_key, reference_key, direction_key = jax.random.split(replicate_key, 3)
            trajectories = self._sample_direction(
                path_key,
                config.validation_size,
                params,
                direction,
            )
            if direction == "f":
                generated = trajectories.paths[:, -1]
                reference = self._sample_target(reference_key, config.validation_size)
            else:
                generated = trajectories.paths[:, 0]
                reference = self._sample_source(reference_key, config.validation_size)
            slicing_directions = jax.random.normal(
                direction_key,
                (config.validation_projections, self.problem.dim),
                dtype=generated.dtype,
            )
            slicing_directions /= jnp.maximum(
                jnp.linalg.norm(slicing_directions, axis=-1, keepdims=True),
                jnp.finfo(generated.dtype).tiny,
            )
            generated_sorted = jnp.sort(generated @ slicing_directions.T, axis=0)
            reference_sorted = jnp.sort(reference @ slicing_directions.T, axis=0)
            return jnp.mean(jnp.abs(generated_sorted - reference_sorted))

        return jnp.asarray(jax.lax.map(one_replicate, replicate_keys))

    def _projection_objective_samples(
        self,
        key: PRNGKey,
        params: Params,
        direction: str,
    ) -> Array:
        """Held-out generalized objective for an already feasible projection."""
        size = self.mam_bridge_config.projection.validation_size
        trajectories = self._sample_direction(key, size, params, direction)
        local_paths = trajectories.paths if direction == "f" else trajectories.paths[:, ::-1]
        times = jnp.asarray(self.problem.time_grid.times, dtype=local_paths.dtype)
        physical_times = (
            times
            if direction == "f"
            else self.problem.time_grid.t0 + self.problem.time_grid.t1 - times
        )
        context = jnp.zeros((size, self.problem.dim), dtype=local_paths.dtype)
        running = self.value_cost.running_values(local_paths, physical_times, context)
        dt = jnp.asarray(self.problem.time_grid.dt, dtype=local_paths.dtype)
        # The public Markov process has one noisy, controlled Euler transition
        # per grid interval.  Score every one of those controls.  This is
        # deliberately distinct from the endpoint-pinned conditional objective,
        # whose final transition is deterministic and therefore has N-1 controls.
        markov_steps = self.problem.time_grid.num_steps
        states = local_paths[:, :markov_steps]
        indices = jnp.arange(markov_steps)
        net_params = params[direction.upper()]

        def one_step(index: Array, state: Array) -> Array:
            prediction = self.projector.predict(net_params, state, index, direction)
            remaining = (self.problem.time_grid.num_steps - index) * dt
            drift = (prediction - state) / remaining
            return jnp.asarray(jnp.linalg.solve(self._sigma, drift.T).T)

        controls = jax.vmap(one_step, in_axes=(0, 1), out_axes=1)(indices, states)
        energy = 0.5 * jnp.sum(controls**2, axis=(1, 2))
        potential = jnp.sum(running[:, 1:-1], axis=1)
        return dt * (energy + potential)

    def _accept_projection(
        self,
        key: PRNGKey,
        params: Params,
        direction: str,
        fitted: _FieldParams,
    ) -> tuple[Params, dict[str, Any]]:
        """Select then independently confirm a damped finite-grid field update."""
        config = self.mam_bridge_config.projection
        selection_key, confirmation_key = jax.random.split(key)
        fitted = self._damped_projection(params[direction.upper()], fitted)
        proposals: list[Params] = []
        for eta in config.line_search:
            proposal = dict(params)
            proposal[direction.upper()] = self.projector.mix(
                params[direction.upper()],
                fitted,
                1.0 - eta,
                eta,
            )
            proposals.append(proposal)

        globally_feasible = bool(
            self._audit_history and self._audit_history[-1].get("global_endpoint_pass", False)
        )

        def select_on(split_key: PRNGKey) -> tuple[int | None, list[dict[str, Any]]]:
            endpoint_key, objective_key = jax.random.split(split_key)
            current_endpoint = self._projection_endpoint_scores(endpoint_key, params, direction)
            current_objective = (
                self._projection_objective_samples(objective_key, params, direction)
                if globally_feasible
                else None
            )
            records: list[dict[str, Any]] = []
            for index, (eta, proposal) in enumerate(
                zip(config.line_search, proposals, strict=True)
            ):
                endpoint_stats = paired_objective_statistics(
                    current_endpoint,
                    self._projection_endpoint_scores(endpoint_key, proposal, direction),
                    z_value=config.one_sided_z,
                )
                endpoint_upper = float(jax.device_get(endpoint_stats.upper_confidence_bound))
                endpoint_pass = (
                    endpoint_upper <= config.feasible_endpoint_noninferiority
                    if globally_feasible
                    else bool(jax.device_get(endpoint_stats.accepted))
                )
                objective_stats = None
                objective_pass = True
                objective_mean = 0.0
                if current_objective is not None:
                    objective_stats = paired_objective_statistics(
                        current_objective,
                        self._projection_objective_samples(objective_key, proposal, direction),
                        z_value=config.one_sided_z,
                        minimum_improvement=config.objective_improvement_tolerance,
                    )
                    objective_pass = bool(jax.device_get(objective_stats.accepted))
                    objective_mean = float(jax.device_get(objective_stats.mean_delta))
                records.append(
                    {
                        "index": index,
                        "eta": eta,
                        "endpoint_mean_delta": float(jax.device_get(endpoint_stats.mean_delta)),
                        "endpoint_upper_confidence_bound": endpoint_upper,
                        "endpoint_pass": endpoint_pass,
                        "objective_mean_delta": objective_mean,
                        "objective_upper_confidence_bound": (
                            None
                            if objective_stats is None
                            else float(jax.device_get(objective_stats.upper_confidence_bound))
                        ),
                        "objective_pass": objective_pass,
                        "accepted": endpoint_pass and objective_pass,
                    }
                )
            acceptable = [record for record in records if record["accepted"]]
            if not acceptable:
                return None, records
            score_name = "objective_mean_delta" if globally_feasible else "endpoint_mean_delta"
            selected = min(acceptable, key=lambda record: record[score_name])
            return int(selected["index"]), records

        selected_index, selection_records = select_on(selection_key)
        if selected_index is None:
            return params, {
                "projection_accepted": False,
                "selected_step_size": 0.0,
                "globally_feasible_before_projection": globally_feasible,
                "selection": selection_records,
                "confirmation": None,
                "independent_cloud_replicates": config.validation_replicates,
                "slicing_directions_per_cloud": config.validation_projections,
                "projection_statistical_unit": "independent_generated_reference_cloud",
                "confidence_method": "normal_clt_approximation",
            }
        # The confirmation split evaluates only the frozen selected candidate.
        endpoint_key, objective_key = jax.random.split(confirmation_key)
        selected = proposals[selected_index]
        current_endpoint = self._projection_endpoint_scores(endpoint_key, params, direction)
        candidate_endpoint = self._projection_endpoint_scores(endpoint_key, selected, direction)
        endpoint_stats = paired_objective_statistics(
            current_endpoint,
            candidate_endpoint,
            z_value=config.one_sided_z,
        )
        endpoint_upper = float(jax.device_get(endpoint_stats.upper_confidence_bound))
        endpoint_pass = (
            endpoint_upper <= config.feasible_endpoint_noninferiority
            if globally_feasible
            else bool(jax.device_get(endpoint_stats.accepted))
        )
        objective_pass = True
        objective_record = None
        if globally_feasible:
            objective_stats = paired_objective_statistics(
                self._projection_objective_samples(objective_key, params, direction),
                self._projection_objective_samples(objective_key, selected, direction),
                z_value=config.one_sided_z,
                minimum_improvement=config.objective_improvement_tolerance,
            )
            objective_pass = bool(jax.device_get(objective_stats.accepted))
            objective_record = {
                "mean_delta": float(jax.device_get(objective_stats.mean_delta)),
                "upper_confidence_bound": float(
                    jax.device_get(objective_stats.upper_confidence_bound)
                ),
                "passed": objective_pass,
            }
        confirmed = endpoint_pass and objective_pass
        return (selected if confirmed else params), {
            "projection_accepted": confirmed,
            "selected_step_size": (config.line_search[selected_index] if confirmed else 0.0),
            "globally_feasible_before_projection": globally_feasible,
            "selection": selection_records,
            "confirmation": {
                "endpoint_mean_delta": float(jax.device_get(endpoint_stats.mean_delta)),
                "endpoint_upper_confidence_bound": endpoint_upper,
                "endpoint_pass": endpoint_pass,
                "objective": objective_record,
                "accepted": confirmed,
            },
            "selection_and_confirmation_streams_disjoint": True,
            "paired_common_noise": True,
            "independent_cloud_replicates": config.validation_replicates,
            "slicing_directions_per_cloud": config.validation_projections,
            "projection_statistical_unit": "independent_generated_reference_cloud",
            "confidence_method": "normal_clt_approximation",
        }

    def _sample_local_chain(
        self,
        key: PRNGKey,
        start: Array,
        params: _FieldParams,
        direction: str,
    ) -> Array:
        start = jnp.atleast_2d(start)
        num_steps = self.problem.time_grid.num_steps
        dt = jnp.asarray(self.problem.time_grid.dt, dtype=start.dtype)
        sigma = jnp.asarray(self._sigma, dtype=start.dtype)
        step_keys = jax.random.split(key, num_steps)

        def step(state: Array, inputs: tuple[Array, PRNGKey]) -> tuple[Array, Array]:
            index, step_key = inputs
            innovation = jax.random.normal(step_key, state.shape, dtype=state.dtype)
            endpoint_prediction = self.projector.predict(params, state, index, direction)
            remaining = (num_steps - index) * dt
            drift = (endpoint_prediction - state) / remaining
            proposal = state + dt * drift
            proposal = proposal + jnp.sqrt(dt) * (innovation @ sigma.T)
            return proposal, proposal

        _, states_after = jax.lax.scan(step, start, (jnp.arange(num_steps), step_keys))
        return jnp.concatenate([start[:, None, :], jnp.swapaxes(states_after, 0, 1)], axis=1)

    def _sample_direction(
        self,
        key: PRNGKey,
        num_samples: int,
        params: Params,
        direction: str,
        start: Array | None = None,
    ) -> TrajectoryBatch:
        start_key, path_key = jax.random.split(key)
        if direction == "f":
            local_start = self._sample_source(start_key, num_samples) if start is None else start
            local_start = jnp.asarray(
                local_start, dtype=self.mam_bridge_config.execution.production_dtype
            )
            local = self._sample_local_chain(path_key, local_start, params["F"], "f")
            paths = local
        elif direction == "b":
            local_start = self._sample_target(start_key, num_samples) if start is None else start
            local_start = jnp.asarray(
                local_start, dtype=self.mam_bridge_config.execution.production_dtype
            )
            local = self._sample_local_chain(path_key, local_start, params["B"], "b")
            paths = local[:, ::-1, :]
        else:
            raise ValueError("direction must be 'f' or 'b'")
        return TrajectoryBatch(
            paths=paths,
            times=jnp.asarray(self.problem.time_grid.times, dtype=paths.dtype),
        )

    def _refresh_pairs(
        self,
        key: PRNGKey,
        params: Params,
        direction: str,
    ) -> EndpointPairBatch:
        size = self.mam_bridge_config.outer.cache_size
        if direction == "f":
            start_key, path_key = jax.random.split(key)
            source = self._sample_source(start_key, size)
            paths = self._sample_direction(path_key, size, params, "f", start=source)
            return EndpointPairBatch(source, paths.paths[:, -1, :])
        start_key, path_key = jax.random.split(key)
        target = self._sample_target(start_key, size)
        paths = self._sample_direction(path_key, size, params, "b", start=target)
        return EndpointPairBatch(paths.paths[:, 0, :], target)

    def _validate_conditional_result(
        self,
        result: ConditionalMAMResult,
        pairs: EndpointPairBatch,
        direction: str,
    ) -> None:
        """Fail closed on an injected conditional backend's public contract."""
        start = pairs.source if direction == "f" else pairs.target
        endpoint = pairs.target if direction == "f" else pairs.source
        expected_dtype = jnp.dtype(self.mam_bridge_config.execution.production_dtype)
        start = jnp.asarray(start, dtype=expected_dtype)
        endpoint = jnp.asarray(endpoint, dtype=expected_dtype)
        batch_size = start.shape[0]
        num_steps = self.problem.time_grid.num_steps
        dim = self.problem.dim
        expected_shapes = {
            "local_paths": (batch_size, num_steps + 1, dim),
            "paths": (batch_size, num_steps + 1, dim),
            "controls": (batch_size, num_steps - 1, dim),
            "projection_states": (batch_size, num_steps, dim),
            "projection_times": (num_steps,),
            "endpoint_predictions": (batch_size, num_steps, dim),
        }
        for name, shape in expected_shapes.items():
            value = jnp.asarray(getattr(result, name))
            if value.shape != shape:
                raise ValueError(
                    f"conditional backend returned {name} shape {value.shape}, expected {shape}"
                )
            if value.dtype != expected_dtype:
                raise TypeError(
                    f"conditional backend returned {name} dtype {value.dtype}, "
                    f"expected {expected_dtype}"
                )
            if not bool(jax.device_get(jnp.all(jnp.isfinite(value)))):
                raise FloatingPointError(f"conditional backend returned nonfinite {name}")
        if result.direction != direction:
            raise ValueError("conditional backend returned the wrong direction")
        if not result.exact_conditional_endpoint:
            raise ValueError("conditional backend did not certify exact endpoint pinning")
        work_record = result.metrics.get("work_accounting")
        if result.certified_work_counters is None:
            if isinstance(work_record, dict) and work_record.get(
                "structural_counters_certified", False
            ):
                raise ValueError(
                    "conditional backend claims certified work without supplying counters"
                )
        else:
            if not isinstance(result.certified_work_counters, MAMWorkCounters):
                raise TypeError("conditional certified_work_counters has the wrong type")
            if (
                not isinstance(work_record, dict)
                or work_record.get("structural_counters_certified") is not True
                or work_record.get("certified_counters")
                != result.certified_work_counters.to_state()
            ):
                raise ValueError(
                    "conditional certified counters disagree with its accounting metadata"
                )
        if not bool(jax.device_get(jnp.array_equal(result.local_paths[:, 0], start))):
            raise ValueError("conditional backend changed the supplied start points")
        if not bool(jax.device_get(jnp.array_equal(result.local_paths[:, -1], endpoint))):
            raise ValueError("conditional backend did not exactly pin the supplied endpoints")
        chronological = result.local_paths if direction == "f" else result.local_paths[:, ::-1]
        if not bool(jax.device_get(jnp.array_equal(result.paths, chronological))):
            raise ValueError("conditional backend path orientation is inconsistent")
        expected_projection_states = result.local_paths[:, :-1, :]
        if not bool(
            jax.device_get(jnp.array_equal(result.projection_states, expected_projection_states))
        ):
            raise ValueError("conditional backend projection states do not match local paths")
        expected_projection_times = _projection_physical_times(
            self.problem,
            expected_dtype,
            direction,
        )
        if not bool(
            jax.device_get(jnp.array_equal(result.projection_times, expected_projection_times))
        ):
            raise ValueError("conditional backend projection times do not match the exact grid")

        # Recompute the declared finite endpoint-prediction target instead of
        # trusting an injected backend's regression labels.  There are N-1
        # stochastic conditional controls and one final exact endpoint label.
        all_times = jnp.asarray(self.problem.time_grid.times, dtype=expected_dtype)
        departures = all_times[: num_steps - 1]
        arrivals = all_times[1:num_steps]
        terminal = all_times[-1]
        rho = (terminal - arrivals) / (terminal - departures)
        sigma_control = result.controls @ jnp.asarray(self._sigma, dtype=expected_dtype).T
        controlled_predictions = endpoint[:, None, :] + (
            (terminal - departures)[None, :, None] * jnp.sqrt(rho)[None, :, None] * sigma_control
        )
        expected_predictions = jnp.concatenate(
            [controlled_predictions, endpoint[:, None, :]],
            axis=1,
        )
        scale = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(expected_predictions)),
                jnp.max(jnp.abs(result.endpoint_predictions)),
            ),
            jnp.asarray(jnp.finfo(expected_dtype).tiny, dtype=expected_dtype),
        )
        tolerance = 64.0 * jnp.finfo(expected_dtype).eps * scale
        predictions_match = jnp.all(
            jnp.abs(result.endpoint_predictions - expected_predictions) <= tolerance
        )
        if not bool(jax.device_get(predictions_match)):
            raise ValueError(
                "conditional backend endpoint predictions do not match controls/endpoints"
            )

    def _audit_endpoints(self, key: PRNGKey, params: Params) -> dict[str, Any]:
        size = self.mam_bridge_config.outer.audit_size
        keys = jax.random.split(key, 12)
        if self._source_calibration is None:
            self._source_calibration = calibrate_endpoint_thresholds(
                keys[0],
                self._sample_source,
                size,
                self.mam_bridge_config.audit,
                mode_label_fn=self._source_mode_label_fn,
                num_modes=self._source_num_modes,
                mode_proportion_fn=self._source_mode_proportion_fn,
            )
        if self._target_calibration is None:
            self._target_calibration = calibrate_endpoint_thresholds(
                keys[1],
                self._sample_target,
                size,
                self.mam_bridge_config.audit,
                mode_label_fn=self._target_mode_label_fn,
                num_modes=self._target_num_modes,
                mode_proportion_fn=self._target_mode_proportion_fn,
            )
        forward = self._sample_direction(keys[10], size, params, "f")
        backward = self._sample_direction(keys[11], size, params, "b")
        source_reference_size = self._source_calibration.thresholds.reference_size
        target_reference_size = self._target_calibration.thresholds.reference_size
        forward_source = audit_endpoint(
            keys[6],
            forward.paths[:, 0],
            self._sample_source(keys[2], source_reference_size),
            self._source_calibration.thresholds,
            self.mam_bridge_config.audit,
            mode_label_fn=self._source_mode_label_fn,
            num_modes=self._source_num_modes,
            mode_proportion_fn=self._source_mode_proportion_fn,
        )
        forward_target = audit_endpoint(
            keys[7],
            forward.paths[:, -1],
            self._sample_target(keys[3], target_reference_size),
            self._target_calibration.thresholds,
            self.mam_bridge_config.audit,
            mode_label_fn=self._target_mode_label_fn,
            num_modes=self._target_num_modes,
            mode_proportion_fn=self._target_mode_proportion_fn,
        )
        backward_source = audit_endpoint(
            keys[8],
            backward.paths[:, 0],
            self._sample_source(keys[4], source_reference_size),
            self._source_calibration.thresholds,
            self.mam_bridge_config.audit,
            mode_label_fn=self._source_mode_label_fn,
            num_modes=self._source_num_modes,
            mode_proportion_fn=self._source_mode_proportion_fn,
        )
        backward_target = audit_endpoint(
            keys[9],
            backward.paths[:, -1],
            self._sample_target(keys[5], target_reference_size),
            self._target_calibration.thresholds,
            self.mam_bridge_config.audit,
            mode_label_fn=self._target_mode_label_fn,
            num_modes=self._target_num_modes,
            mode_proportion_fn=self._target_mode_proportion_fn,
        )
        all_results = (
            forward_source,
            forward_target,
            backward_source,
            backward_target,
        )
        global_pass = all(result.passed for result in all_results)
        finite = all(result.finite for result in all_results)
        return {
            "forward_source_mmd2": forward_source.metrics.mmd2,
            "forward_target_mmd2": forward_target.metrics.mmd2,
            "backward_source_mmd2": backward_source.metrics.mmd2,
            "backward_target_mmd2": backward_target.metrics.mmd2,
            "forward_target_sliced_wasserstein": (forward_target.metrics.sliced_wasserstein),
            "backward_source_sliced_wasserstein": (backward_source.metrics.sliced_wasserstein),
            "forward_target_sinkhorn_divergence": (forward_target.metrics.sinkhorn_divergence),
            "backward_source_sinkhorn_divergence": (backward_source.metrics.sinkhorn_divergence),
            "forward_target_mean_error": forward_target.metrics.mean_error,
            "backward_source_mean_error": backward_source.metrics.mean_error,
            "forward_target_covariance_error": (forward_target.metrics.covariance_error),
            "backward_source_covariance_error": (backward_source.metrics.covariance_error),
            "forward_endpoint_pass": (forward_source.passed and forward_target.passed),
            "backward_endpoint_pass": (backward_source.passed and backward_target.passed),
            "global_endpoint_pass": global_pass,
            "finite": finite,
            "endpoint_pass_is_empirical": True,
            "thresholds_null_calibrated": True,
            "source_null_calibration": asdict(self._source_calibration),
            "target_null_calibration": asdict(self._target_calibration),
            "forward_source_audit": asdict(forward_source),
            "forward_target_audit": asdict(forward_target),
            "backward_source_audit": asdict(backward_source),
            "backward_target_audit": asdict(backward_target),
        }

    @staticmethod
    def _seed_from_key(key: PRNGKey) -> int:
        raw = np.asarray(jax.device_get(key), dtype=np.uint32).reshape(-1)
        if raw.size != 2:
            raise ValueError("MAMBridgeSolver expects a conventional two-word JAX key")
        return ((int(raw[0]) * 0x9E3779B9) ^ int(raw[1])) & 0xFFFFFFFF

    @staticmethod
    def _pair_cache_hash(pairs: EndpointPairBatch) -> str:
        digest = hashlib.sha256()
        for value in (pairs.source, pairs.target):
            array = np.ascontiguousarray(jax.device_get(value))
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _callable_fingerprint(function: Any) -> str:
        """Stable, cycle-safe fingerprint for scientific callbacks.

        Bound methods include the declaring instance's type, source, and
        scientific state.  Hashing only ``method.__func__`` would make two
        obstacle/cost instances with different thresholds checkpoint-compatible.
        Plain Python functions also bind the safe values they actually read
        from module globals.  Modules are represented by stable identities;
        opaque referenced state fails closed instead of falling back to an
        address-bearing ``repr``.
        """
        digest = hashlib.sha256()
        active: set[int] = set()

        def enter(value: Any) -> bool:
            value_id = id(value)
            if value_id in active:
                digest.update(b"cycle")
                digest.update(type(value).__module__.encode())
                digest.update(type(value).__qualname__.encode())
                return False
            active.add(value_id)
            return True

        def leave(value: Any) -> None:
            active.remove(id(value))

        def scientific_object_state(value: Any) -> dict[str, Any] | None:
            if is_dataclass(value) and not isinstance(value, type):
                return {
                    field_info.name: getattr(value, field_info.name) for field_info in fields(value)
                }
            if hasattr(value, "__dict__"):
                return cast(dict[str, Any], vars(value))
            slot_state: dict[str, Any] = {}
            for base in reversed(type(value).__mro__):
                slots = getattr(base, "__slots__", ())
                if isinstance(slots, str):
                    slots = (slots,)
                for slot in slots:
                    if slot not in {"__dict__", "__weakref__"} and hasattr(value, slot):
                        slot_state[slot] = getattr(value, slot)
            return slot_state or None

        def safe_class_data(value: Any) -> bool:
            if value is None or isinstance(
                value,
                (bool, int, float, str, bytes, np.number, Path),
            ):
                return True
            if isinstance(value, np.ndarray) or (
                hasattr(value, "shape") and hasattr(value, "dtype")
            ):
                return True
            if is_dataclass(value) and not isinstance(value, type):
                return all(
                    safe_class_data(getattr(value, field_info.name)) for field_info in fields(value)
                )
            if isinstance(value, dict):
                return all(
                    safe_class_data(key) and safe_class_data(item) for key, item in value.items()
                )
            if isinstance(value, (tuple, list)):
                return all(safe_class_data(item) for item in value)
            return False

        def update_class(value: type[Any]) -> None:
            if not enter(value):
                return
            try:
                digest.update(b"class")
                digest.update(value.__module__.encode())
                digest.update(value.__qualname__.encode())
                try:
                    digest.update(inspect.getsource(value).encode())
                except (OSError, TypeError):
                    pass
                # Source binds implementation edits.  Safe class-level data
                # additionally binds runtime scientific constants without
                # traversing descriptors or framework internals.
                for attribute_name, attribute in sorted(vars(value).items()):
                    if attribute_name.startswith("__") or not safe_class_data(attribute):
                        continue
                    digest.update(b"class_state")
                    digest.update(attribute_name.encode())
                    update_safe_value(attribute, f"{value.__qualname__}.{attribute_name}")
            finally:
                leave(value)

        def update_object(value: Any) -> None:
            if not enter(value):
                return
            try:
                object_type = type(value)
                digest.update(b"object")
                update_class(object_type)
                update_safe_value(
                    scientific_object_state(value),
                    f"{object_type.__qualname__} instance state",
                )
            finally:
                leave(value)

        def update_callable(value: Any) -> None:
            if isinstance(value, type):
                update_class(value)
                return
            if not enter(value):
                return
            try:
                if isinstance(value, partial):
                    digest.update(b"partial")
                    update_callable(value.func)
                    update_safe_value(value.args, "partial positional arguments")
                    update_safe_value(value.keywords or {}, "partial keyword arguments")
                    return
                if inspect.ismethod(value) and value.__self__ is not None:
                    digest.update(b"bound_method")
                    update_callable(value.__func__)
                    if isinstance(value.__self__, type):
                        update_class(value.__self__)
                    else:
                        update_object(value.__self__)
                    return
                digest.update(b"callable")
                digest.update(type(value).__module__.encode())
                digest.update(type(value).__qualname__.encode())
                digest.update(str(getattr(value, "__module__", "")).encode())
                digest.update(str(getattr(value, "__qualname__", "")).encode())
                code = getattr(value, "__code__", None)
                if code is not None:
                    # marshal recursively serializes nested code objects without
                    # embedding their process-specific memory addresses.
                    digest.update(marshal.dumps(code))
                    global_scope = getattr(value, "__globals__", None)
                    if isinstance(global_scope, dict):
                        for global_name in sorted(set(code.co_names)):
                            if global_name not in global_scope:
                                continue
                            digest.update(b"referenced_global")
                            digest.update(global_name.encode())
                            update_safe_value(
                                global_scope[global_name],
                                f"referenced global {global_name!r}",
                            )
                else:
                    try:
                        digest.update(inspect.getsource(type(value)).encode())
                    except (OSError, TypeError):
                        pass
                update_safe_value(getattr(value, "__defaults__", None), "callable defaults")
                closure = getattr(value, "__closure__", None)
                if closure is not None:
                    update_safe_value(
                        tuple(cell.cell_contents for cell in closure),
                        "callable closure",
                    )
                state = scientific_object_state(value)
                if state:
                    update_safe_value(state, "callable instance state")
            finally:
                leave(value)

        def update_safe_value(value: Any, name: str) -> None:
            """Hash address-free scientific state or reject an opaque value."""
            if value is None:
                digest.update(b"none")
                return
            if isinstance(value, (bool, int, float, str, bytes, np.number)):
                if isinstance(
                    value, (float, np.floating, complex, np.complexfloating)
                ) and not bool(np.isfinite(value)):
                    raise ValueError(f"{name} contains a nonfinite scalar")
                digest.update(type(value).__name__.encode())
                digest.update(repr(value).encode())
                return
            if isinstance(value, Path):
                digest.update(b"path")
                digest.update(str(value).encode())
                return
            if isinstance(value, ModuleType):
                digest.update(b"module")
                digest.update(value.__name__.encode())
                return
            if isinstance(value, type):
                update_class(value)
                return
            if isinstance(value, np.ndarray) or (
                hasattr(value, "shape") and hasattr(value, "dtype")
            ):
                array = np.ascontiguousarray(jax.device_get(value))
                if array.dtype.hasobject:
                    raise TypeError(f"{name} contains an object-dtype array")
                if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} contains a nonfinite array")
                digest.update(b"array")
                digest.update(str(array.dtype).encode())
                digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
                digest.update(array.tobytes())
                return
            if is_dataclass(value) and not isinstance(value, type):
                if not enter(value):
                    return
                try:
                    digest.update(b"dataclass")
                    digest.update(type(value).__module__.encode())
                    digest.update(type(value).__qualname__.encode())
                    for field_info in fields(value):
                        digest.update(field_info.name.encode())
                        update_safe_value(
                            getattr(value, field_info.name),
                            f"{name}.{field_info.name}",
                        )
                finally:
                    leave(value)
                return
            if isinstance(value, dict):
                if not enter(value):
                    return
                try:
                    digest.update(b"dict")
                    for key in sorted(value, key=repr):
                        update_safe_value(key, f"{name} key")
                        update_safe_value(value[key], f"{name}[{key!r}]")
                finally:
                    leave(value)
                return
            if isinstance(value, (tuple, list)):
                if not enter(value):
                    return
                try:
                    digest.update(type(value).__name__.encode())
                    for index, item in enumerate(value):
                        update_safe_value(item, f"{name}[{index}]")
                finally:
                    leave(value)
                return
            if callable(value):
                module_name = str(getattr(value, "__module__", ""))
                if module_name == "builtins" or module_name.startswith(
                    ("jax", "jaxlib", "numpy", "scipy")
                ):
                    digest.update(b"external_callable")
                    digest.update(module_name.encode())
                    digest.update(str(getattr(value, "__qualname__", "")).encode())
                    return
                update_callable(value)
                return
            if hasattr(value, "__dict__") or any(
                hasattr(base, "__slots__") for base in type(value).__mro__
            ):
                update_object(value)
                return
            raise TypeError(
                f"{name} contains unsupported opaque state of type "
                f"{type(value).__module__}.{type(value).__qualname__}; capture scientific "
                "state in a closure, functools.partial, dataclass, or bound object"
            )

        if callable(function):
            update_callable(function)
        else:
            update_safe_value(function, "fingerprinted value")
        return digest.hexdigest()

    @staticmethod
    def _canonical_fingerprint_value(value: Any) -> Any:
        """Convert config state into stable, address-free JSON data."""
        active: set[int] = set()

        def canonical(item: Any) -> Any:
            if item is None or isinstance(item, (bool, int, str)):
                return item
            if isinstance(item, (float, np.floating)):
                return repr(float(item))
            if isinstance(item, np.integer):
                return int(item)
            if isinstance(item, np.ndarray) or (hasattr(item, "shape") and hasattr(item, "dtype")):
                array = np.ascontiguousarray(jax.device_get(item))
                return {
                    "array_dtype": str(array.dtype),
                    "array_shape": list(array.shape),
                    "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
            if isinstance(item, type):
                try:
                    dtype = np.dtype(item)
                except TypeError:
                    return {"type": f"{item.__module__}.{item.__qualname__}"}
                return {"dtype": dtype.str}

            item_id = id(item)
            if item_id in active:
                return {"cycle": f"{type(item).__module__}.{type(item).__qualname__}"}
            if is_dataclass(item):
                active.add(item_id)
                try:
                    return {
                        field_info.name: canonical(getattr(item, field_info.name))
                        for field_info in fields(item)
                    }
                finally:
                    active.remove(item_id)
            if isinstance(item, dict):
                active.add(item_id)
                try:
                    return {str(key): canonical(item[key]) for key in sorted(item, key=repr)}
                finally:
                    active.remove(item_id)
            if isinstance(item, (tuple, list)):
                active.add(item_id)
                try:
                    return [canonical(element) for element in item]
                finally:
                    active.remove(item_id)
            if callable(item):
                return {
                    "callable_type": f"{type(item).__module__}.{type(item).__qualname__}",
                    "callable_sha256": MAMBridgeSolver._callable_fingerprint(item),
                }
            if hasattr(item, "__dict__"):
                active.add(item_id)
                try:
                    object_type = f"{type(item).__module__}.{type(item).__qualname__}"
                    try:
                        source = inspect.getsource(type(item)).encode()
                    except (OSError, TypeError):
                        source = object_type.encode()
                    return {
                        "object_type": object_type,
                        "object_source_sha256": hashlib.sha256(source).hexdigest(),
                        "object_state": canonical(vars(item)),
                    }
                finally:
                    active.remove(item_id)
            return {
                "value_type": f"{type(item).__module__}.{type(item).__qualname__}",
                "value": str(item),
            }

        return canonical(value)

    @staticmethod
    def _dependency_versions() -> dict[str, str]:
        """Numerical dependency versions that define checkpoint semantics."""
        return {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
        }

    def _read_injected_conditional_scientific_fingerprint(self) -> str:
        """Read the immutable scientific identity required of injected cores."""
        method = getattr(self.conditional_solver, "scientific_fingerprint", None)
        if not callable(method):
            raise TypeError("injected conditional_solver must define scientific_fingerprint()")
        fingerprint = method()
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise TypeError(
                "conditional_solver.scientific_fingerprint() must return a nonempty string"
            )
        return fingerprint

    def _scientific_fingerprints(self) -> dict[str, str]:
        config_payload = json.dumps(
            self._canonical_fingerprint_value(self.mam_bridge_config),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        problem_payload = json.dumps(
            {
                "name": self.problem.name,
                "dim": self.problem.dim,
                "time_grid": self._canonical_fingerprint_value(self.problem.time_grid),
                "reference_type": (
                    f"{type(self.problem.reference).__module__}."
                    f"{type(self.problem.reference).__qualname__}"
                ),
                "reference": self._canonical_fingerprint_value(self.problem.reference.__dict__),
                "reference_drift_sha256": self._callable_fingerprint(self.problem.reference.drift),
                "reference_diffusion_sha256": self._callable_fingerprint(
                    self.problem.reference.diffusion
                ),
                "source_type": (
                    f"{type(self.problem.source).__module__}."
                    f"{type(self.problem.source).__qualname__}"
                ),
                "source": self._canonical_fingerprint_value(
                    getattr(self.problem.source, "__dict__", {})
                ),
                "source_sample_sha256": self._callable_fingerprint(self.problem.source.sample),
                "target_type": (
                    f"{type(self.problem.target).__module__}."
                    f"{type(self.problem.target).__qualname__}"
                ),
                "target": self._canonical_fingerprint_value(
                    getattr(self.problem.target, "__dict__", {})
                ),
                "target_sample_sha256": self._callable_fingerprint(self.problem.target.sample),
                "source_mode_label_sha256": self._callable_fingerprint(self._source_mode_label_fn),
                "source_mode_proportion_sha256": self._callable_fingerprint(
                    self._source_mode_proportion_fn
                ),
                "source_num_modes": self._source_num_modes,
                "target_mode_label_sha256": self._callable_fingerprint(self._target_mode_label_fn),
                "target_mode_proportion_sha256": self._callable_fingerprint(
                    self._target_mode_proportion_fn
                ),
                "target_num_modes": self._target_num_modes,
            },
            sort_keys=True,
        ).encode()
        implementation_modules = (
            inspect.getmodule(MAMBridgeSolver),
            mam_accounting_module,
            mam_fields_module,
            inspect.getmodule(MalliavinAdjointInnerSolver),
            inspect.getmodule(CrossFittedValueCritic),
            inspect.getmodule(paired_objective_statistics),
            inspect.getmodule(audit_endpoint),
            inspect.getmodule(RNGLedger),
            inspect.getmodule(adam_update),
        )
        if any(module is None for module in implementation_modules):
            raise RuntimeError("could not resolve a scientific implementation module")
        resolved_modules = tuple(cast(ModuleType, module) for module in implementation_modules)
        implementation_payload = "\n".join(
            inspect.getsource(module) for module in resolved_modules
        ).encode()
        conditional_type = type(self.conditional_solver)
        conditional_identifier = (
            f"{conditional_type.__module__}.{conditional_type.__qualname__}:"
            f"{getattr(self.conditional_solver, 'status', '')}"
        )
        try:
            conditional_source = inspect.getsource(conditional_type)
        except (OSError, TypeError):
            conditional_source = conditional_identifier
        if type(self.conditional_solver) is MAMConditionalSolver:
            conditional_instance_fingerprint = "builtin_bound_by_mam_config_and_implementation"
        else:
            conditional_instance_fingerprint = (
                self._read_injected_conditional_scientific_fingerprint()
            )
            if (
                conditional_instance_fingerprint
                != self._injected_conditional_scientific_fingerprint
            ):
                raise ValueError(
                    "injected conditional scientific fingerprint changed after construction"
                )
        conditional_payload = (
            conditional_source
            + "\nINSTANCE_SCIENTIFIC_FINGERPRINT="
            + conditional_instance_fingerprint
        ).encode()
        return {
            "config_sha256": hashlib.sha256(config_payload).hexdigest(),
            "problem_sha256": hashlib.sha256(problem_payload).hexdigest(),
            "cost_identifier": self.value_cost.identifier,
            "cost_sha256": self._callable_fingerprint(self.value_cost.running_cost),
            "implementation_sha256": hashlib.sha256(implementation_payload).hexdigest(),
            "conditional_backend": conditional_identifier,
            "conditional_backend_instance_sha256": hashlib.sha256(
                conditional_instance_fingerprint.encode()
            ).hexdigest(),
            "conditional_backend_sha256": hashlib.sha256(conditional_payload).hexdigest(),
        }

    def _conditional_work_from_derivation(
        self,
        derivation: Any,
    ) -> MAMWorkCounters:
        """Recompute built-in conditional work from checked runtime decisions."""

        expected_keys = {
            "policy_iterations_completed",
            "actor_confirmation_executed",
            "actor_update_accepted",
            "output_actor_fingerprint",
            "pre_refresh_costate_policy_fingerprint",
            "final_costate_refresh_executed",
        }
        if not isinstance(derivation, dict) or set(derivation) != expected_keys:
            raise ValueError("conditional work derivation schema mismatch")
        completed = derivation["policy_iterations_completed"]
        _require_integer("policy_iterations_completed", completed, minimum=1)
        if completed > self.mam_bridge_config.conditional.policy_iterations:
            raise ValueError("conditional work derivation exceeds configured policy iterations")
        confirmations = derivation["actor_confirmation_executed"]
        accepted = derivation["actor_update_accepted"]
        if not isinstance(confirmations, list) or not isinstance(accepted, list):
            raise TypeError("conditional work decision sequences must be JSON lists")
        if len(confirmations) != completed or len(accepted) != completed:
            raise ValueError("conditional work decision lengths disagree with completed iterations")
        if any(not isinstance(value, bool) for value in (*confirmations, *accepted)):
            raise TypeError("conditional work decisions must be bool values")
        if any(
            was_accepted and not confirmed
            for was_accepted, confirmed in zip(
                accepted,
                confirmations,
                strict=True,
            )
        ):
            raise ValueError("an actor update cannot be accepted without confirmation")
        output_actor_fingerprint = derivation["output_actor_fingerprint"]
        pre_refresh_fingerprint = derivation["pre_refresh_costate_policy_fingerprint"]
        for name, fingerprint in (
            ("output_actor_fingerprint", output_actor_fingerprint),
            ("pre_refresh_costate_policy_fingerprint", pre_refresh_fingerprint),
        ):
            if (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                raise ValueError(f"conditional work {name} must be a SHA-256 hex digest")
        refresh = derivation["final_costate_refresh_executed"]
        if not isinstance(refresh, bool):
            raise TypeError("final_costate_refresh_executed must be bool")
        if refresh != (output_actor_fingerprint != pre_refresh_fingerprint):
            raise ValueError("final costate refresh disagrees with actor/costate fingerprints")
        if refresh and not accepted[-1]:
            raise ValueError("final costate refresh requires an accepted last actor update")
        if completed < self.mam_bridge_config.conditional.policy_iterations:
            trailing_rejections = 0
            for was_accepted in reversed(accepted):
                if was_accepted:
                    break
                trailing_rejections += 1
            if trailing_rejections < (
                self.mam_bridge_config.conditional.maximum_consecutive_rejections
            ):
                raise ValueError("early conditional stop lacks the configured rejection run")
        config = self.mam_bridge_config.conditional
        return completed_conditional_solve_work(
            num_steps=self.problem.time_grid.num_steps,
            effective_batch_size=self._execution_plan.effective_batch_size,
            costate_steps=config.costate_steps,
            value_critic_training_steps=config.value_critic.training_steps,
            actor_field_training_steps=(
                config.actor_field_config.training_steps if config.actor_model == "nonlinear" else 0
            ),
            direct_score_diagnostic_size=config.direct_score_diagnostic_size,
            acceptance_size=config.acceptance_size,
            line_search_candidates=len(config.line_search),
            pair_batch_size=self.mam_bridge_config.outer.cache_size,
            policy_iterations_completed=completed,
            actor_confirmation_executed=tuple(confirmations),
            actor_update_accepted=tuple(accepted),
            final_costate_refresh_executed=refresh,
            running_cost_oracle_present=self.value_cost.running_cost is not None,
        )

    def _half_work_from_derivation(
        self,
        derivation: Any,
        *,
        history_index: int,
        globally_feasible_before_projection: bool,
    ) -> MAMWorkCounters:
        """Recompute one exact built-in half-iteration from active configuration."""

        expected_keys = {"half_iteration", "direction", "conditional", "global"}
        if not isinstance(derivation, dict) or set(derivation) != expected_keys:
            raise ValueError("global work derivation schema mismatch")
        half_iteration = derivation["half_iteration"]
        _require_integer("half_iteration", half_iteration, minimum=1)
        if half_iteration != history_index + 1:
            raise ValueError("work derivation half-iteration index mismatch")
        directions = self.mam_bridge_config.outer.directions
        expected_direction = directions[history_index % len(directions)]
        if derivation["direction"] != expected_direction:
            raise ValueError("work derivation direction disagrees with outer schedule")
        global_decisions = derivation["global"]
        expected_global_keys = {
            "projection_confirmation_executed",
            "projection_update_accepted",
        }
        if not isinstance(global_decisions, dict) or set(global_decisions) != expected_global_keys:
            raise ValueError("projection work derivation schema mismatch")
        confirmation = global_decisions["projection_confirmation_executed"]
        accepted = global_decisions["projection_update_accepted"]
        if not isinstance(confirmation, bool) or not isinstance(accepted, bool):
            raise TypeError("projection work decisions must be bool values")
        if accepted and not confirmation:
            raise ValueError("a projection update cannot be accepted without confirmation")
        conditional_work = self._conditional_work_from_derivation(derivation["conditional"])
        config = self.mam_bridge_config.projection
        return global_half_iteration_work(
            conditional_work,
            num_steps=self.problem.time_grid.num_steps,
            effective_batch_size=self._execution_plan.effective_batch_size,
            pair_batch_size=self.mam_bridge_config.outer.cache_size,
            projection_field_training_steps=(
                config.field_config.training_steps if config.model == "nonlinear" else 0
            ),
            projection_validation_size=config.validation_size,
            projection_line_search_candidates=len(config.line_search),
            projection_validation_replicates=config.validation_replicates,
            globally_feasible_before_projection=globally_feasible_before_projection,
            projection_confirmation_executed=confirmation,
            projection_update_accepted=accepted,
            audit_size=self.mam_bridge_config.outer.audit_size,
            running_cost_oracle_present=self.value_cost.running_cost is not None,
        )

    @staticmethod
    def _validate_endpoint_audit_payload(
        payload: Any,
        thresholds: EndpointThresholds,
        *,
        name: str,
    ) -> tuple[dict[str, Any], bool, bool]:
        expected_payload_keys = {
            "metrics",
            "thresholds",
            "metric_pass",
            "passed",
            "finite",
            "status",
        }
        if not isinstance(payload, dict) or set(payload) != expected_payload_keys:
            raise ValueError(f"checkpoint {name} endpoint audit schema mismatch")
        if payload["thresholds"] != asdict(thresholds):
            raise ValueError(f"checkpoint {name} endpoint thresholds disagree with calibration")
        metrics = payload["metrics"]
        expected_metric_keys = {
            "mmd2",
            "sliced_wasserstein",
            "sinkhorn_divergence",
            "mean_error",
            "covariance_error",
            "mode_proportion_l1",
            "sample_mode_proportions",
            "reference_mode_proportions",
            "sinkhorn_marginal_error",
            "sinkhorn_converged",
            "finite",
        }
        if not isinstance(metrics, dict) or set(metrics) != expected_metric_keys:
            raise ValueError(f"checkpoint {name} endpoint metrics schema mismatch")
        for field_name in (
            "mmd2",
            "sliced_wasserstein",
            "sinkhorn_divergence",
            "mean_error",
            "covariance_error",
            "sinkhorn_marginal_error",
        ):
            _require_finite_real(
                f"checkpoint {name} endpoint metric {field_name}",
                metrics[field_name],
            )
        if metrics["mode_proportion_l1"] is not None:
            _require_finite_real(
                f"checkpoint {name} endpoint metric mode_proportion_l1",
                metrics["mode_proportion_l1"],
                nonnegative=True,
            )
        if not isinstance(metrics["sinkhorn_converged"], bool) or not isinstance(
            metrics["finite"], bool
        ):
            raise TypeError(f"checkpoint {name} endpoint metric flags must be bool")
        expected_metric_pass = {
            "mmd2": metrics["mmd2"] <= thresholds.mmd2,
            "sliced_wasserstein": (metrics["sliced_wasserstein"] <= thresholds.sliced_wasserstein),
            "sinkhorn_divergence": (
                metrics["sinkhorn_divergence"] <= thresholds.sinkhorn_divergence
            ),
            "mean_error": metrics["mean_error"] <= thresholds.mean_error,
            "covariance_error": metrics["covariance_error"] <= thresholds.covariance_error,
        }
        if thresholds.mode_proportion_l1 is not None:
            expected_metric_pass["mode_proportion_l1"] = (
                metrics["mode_proportion_l1"] is not None
                and metrics["mode_proportion_l1"] <= thresholds.mode_proportion_l1
            )
        if payload["metric_pass"] != expected_metric_pass:
            raise ValueError(f"checkpoint {name} endpoint pass components are inconsistent")
        expected_finite = metrics["finite"] and thresholds.valid
        expected_passed = (
            expected_finite and metrics["sinkhorn_converged"] and all(expected_metric_pass.values())
        )
        if not isinstance(payload["finite"], bool) or payload["finite"] is not expected_finite:
            raise ValueError(f"checkpoint {name} endpoint finite flag is inconsistent")
        if not isinstance(payload["passed"], bool) or payload["passed"] is not expected_passed:
            raise ValueError(f"checkpoint {name} endpoint passed flag is inconsistent")
        expected_status = (
            "PASSED_EMPIRICAL_ENDPOINT_GATE" if expected_passed else "FAILED_ENDPOINT_GATE"
        )
        if payload["status"] != expected_status:
            raise ValueError(f"checkpoint {name} endpoint status is inconsistent")
        return metrics, expected_passed, expected_finite

    def _validate_global_audit_decisions(self, audit: dict[str, Any]) -> None:
        if not isinstance(self._source_calibration, NullCalibrationResult) or not isinstance(
            self._target_calibration, NullCalibrationResult
        ):
            raise TypeError("endpoint audit validation requires typed null calibrations")
        expected_keys = {
            "forward_source_mmd2",
            "forward_target_mmd2",
            "backward_source_mmd2",
            "backward_target_mmd2",
            "forward_target_sliced_wasserstein",
            "backward_source_sliced_wasserstein",
            "forward_target_sinkhorn_divergence",
            "backward_source_sinkhorn_divergence",
            "forward_target_mean_error",
            "backward_source_mean_error",
            "forward_target_covariance_error",
            "backward_source_covariance_error",
            "forward_endpoint_pass",
            "backward_endpoint_pass",
            "global_endpoint_pass",
            "finite",
            "endpoint_pass_is_empirical",
            "thresholds_null_calibrated",
            "source_null_calibration",
            "target_null_calibration",
            "forward_source_audit",
            "forward_target_audit",
            "backward_source_audit",
            "backward_target_audit",
            "work_accounting",
        }
        if set(audit) != expected_keys:
            raise ValueError("global endpoint audit schema mismatch")
        if audit["source_null_calibration"] != asdict(self._source_calibration) or audit[
            "target_null_calibration"
        ] != asdict(self._target_calibration):
            raise ValueError("global endpoint audit calibration payload is inconsistent")
        source_thresholds = self._source_calibration.thresholds
        target_thresholds = self._target_calibration.thresholds
        forward_source, fs_pass, fs_finite = self._validate_endpoint_audit_payload(
            audit["forward_source_audit"], source_thresholds, name="forward source"
        )
        forward_target, ft_pass, ft_finite = self._validate_endpoint_audit_payload(
            audit["forward_target_audit"], target_thresholds, name="forward target"
        )
        backward_source, bs_pass, bs_finite = self._validate_endpoint_audit_payload(
            audit["backward_source_audit"], source_thresholds, name="backward source"
        )
        backward_target, bt_pass, bt_finite = self._validate_endpoint_audit_payload(
            audit["backward_target_audit"], target_thresholds, name="backward target"
        )
        redundant_metrics = {
            "forward_source_mmd2": forward_source["mmd2"],
            "forward_target_mmd2": forward_target["mmd2"],
            "backward_source_mmd2": backward_source["mmd2"],
            "backward_target_mmd2": backward_target["mmd2"],
            "forward_target_sliced_wasserstein": forward_target["sliced_wasserstein"],
            "backward_source_sliced_wasserstein": backward_source["sliced_wasserstein"],
            "forward_target_sinkhorn_divergence": forward_target["sinkhorn_divergence"],
            "backward_source_sinkhorn_divergence": backward_source["sinkhorn_divergence"],
            "forward_target_mean_error": forward_target["mean_error"],
            "backward_source_mean_error": backward_source["mean_error"],
            "forward_target_covariance_error": forward_target["covariance_error"],
            "backward_source_covariance_error": backward_source["covariance_error"],
        }
        if any(audit[key] != value for key, value in redundant_metrics.items()):
            raise ValueError("global endpoint audit redundant metrics are inconsistent")
        expected_forward_pass = fs_pass and ft_pass
        expected_backward_pass = bs_pass and bt_pass
        expected_global_pass = expected_forward_pass and expected_backward_pass
        expected_finite = fs_finite and ft_finite and bs_finite and bt_finite
        expected_flags = {
            "forward_endpoint_pass": expected_forward_pass,
            "backward_endpoint_pass": expected_backward_pass,
            "global_endpoint_pass": expected_global_pass,
            "finite": expected_finite,
            "endpoint_pass_is_empirical": True,
            "thresholds_null_calibrated": True,
        }
        if any(audit[key] is not value for key, value in expected_flags.items()):
            raise ValueError("global endpoint audit decision flags are inconsistent")

    def _certified_total_from_audit_history(
        self,
        audit_history: list[dict[str, Any]],
    ) -> MAMWorkCounters | None:
        """Recompute certified work; never trust checkpointed counter magnitudes."""
        expected_fields = set(
            _work_accounting_record(
                None,
                scope="successful_global_half_iteration",
                uncertified_reason="placeholder",
            )
        )
        total = MAMWorkCounters.zero()
        all_certified = True
        exact_builtin = type(self.conditional_solver) is MAMConditionalSolver
        maximum_halves = self.mam_bridge_config.outer.num_iterations * len(
            self.mam_bridge_config.outer.directions
        )
        if len(audit_history) > maximum_halves:
            raise ValueError("audit history exceeds the configured outer schedule")
        for index, audit in enumerate(audit_history):
            if not isinstance(audit, dict):
                raise ValueError(f"audit history entry {index} must be a dictionary")
            self._validate_global_audit_decisions(audit)
            global_endpoint_pass = audit.get("global_endpoint_pass")
            if not isinstance(global_endpoint_pass, bool):
                raise TypeError("audit global_endpoint_pass must be bool")
            record = audit.get("work_accounting")
            if not isinstance(record, dict) or set(record) != expected_fields:
                raise ValueError(f"audit history entry {index} has invalid work accounting")
            if record.get("schema_version") != 2:
                raise ValueError("unsupported work-accounting audit schema")
            if record.get("scope") != "successful_global_half_iteration":
                raise ValueError("audit work-accounting scope mismatch")
            certified = record.get("structural_counters_certified")
            if not isinstance(certified, bool):
                raise TypeError("audit structural certification flag must be bool")
            if record.get("unmeasured_fields") != list(_UNMEASURED_WORK_FIELDS):
                raise ValueError("audit unmeasured work fields are inconsistent")
            if (
                record.get("successful_completed_calls_only") is not True
                or record.get("failed_attempt_work_included") is not False
                or record.get("oracle_count_semantics") != "requested_scalar_value_outputs"
                or record.get("external_oracle_billing_certified") is not False
            ):
                raise ValueError("audit work-accounting semantics are inconsistent")

            counter_state = record.get("certified_counters")
            derivation = record.get("derivation")
            if not isinstance(derivation, dict) or set(derivation) != {
                "half_iteration",
                "direction",
                "conditional",
                "global",
            }:
                raise ValueError("audit work accounting lacks a valid v2 derivation")
            _require_integer(
                "work derivation half_iteration",
                derivation.get("half_iteration"),
                minimum=1,
            )
            expected_direction = self.mam_bridge_config.outer.directions[
                index % len(self.mam_bridge_config.outer.directions)
            ]
            if derivation.get("half_iteration") != index + 1:
                raise ValueError("work derivation half-iteration index mismatch")
            if derivation.get("direction") != expected_direction:
                raise ValueError("work derivation direction disagrees with outer schedule")
            global_decisions = derivation.get("global")
            if not isinstance(global_decisions, dict) or set(global_decisions) != {
                "projection_confirmation_executed",
                "projection_update_accepted",
            }:
                raise ValueError("projection work derivation schema mismatch")
            for decision_name in (
                "projection_confirmation_executed",
                "projection_update_accepted",
            ):
                if not isinstance(global_decisions.get(decision_name), bool):
                    raise TypeError("projection work decisions must be bool values")
            if (
                global_decisions["projection_update_accepted"]
                and not global_decisions["projection_confirmation_executed"]
            ):
                raise ValueError("a projection update cannot be accepted without confirmation")

            expected_certified = exact_builtin
            if certified is not expected_certified:
                if certified:
                    raise ValueError("only the exact built-in MAM conditional may certify work")
                raise ValueError("built-in MAM work cannot be downgraded to uncertified")
            if certified:
                if record.get("certified_fields") != list(_CERTIFIED_WORK_FIELDS):
                    raise ValueError("audit certified work fields are inconsistent")
                if record.get("uncertified_reason") is not None:
                    raise ValueError("certified audit work has an uncertified reason")
                globally_feasible = bool(
                    index > 0 and audit_history[index - 1]["global_endpoint_pass"]
                )
                half_work = self._half_work_from_derivation(
                    derivation,
                    history_index=index,
                    globally_feasible_before_projection=globally_feasible,
                )
                saved_half_work = MAMWorkCounters.from_state(cast(Mapping[str, Any], counter_state))
                if saved_half_work != half_work:
                    raise ValueError("saved work counters disagree with exact recomputation")
            else:
                if counter_state is not None or record.get("certified_fields") != []:
                    raise ValueError("uncertified audit work contains certified counters")
                reason = record.get("uncertified_reason")
                if reason != "conditional_backend_is_not_exact_builtin_mam":
                    raise ValueError("uncertified injected-backend work reason is inconsistent")
                if derivation.get("conditional") is not None:
                    raise ValueError("injected-backend work cannot carry a certified derivation")
                half_work = None

            if all_certified and half_work is not None:
                total = total.merge(half_work)
                expected_cumulative = total.to_state()
            else:
                all_certified = False
                expected_cumulative = None
            if record.get("cumulative_certified_counters") != expected_cumulative:
                raise ValueError("audit cumulative work counters are inconsistent")
        return total if all_certified else None

    def _expected_costate_updates_by_direction(
        self,
        audit_history: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Derive exact persistent Adam steps from validated half-iteration records."""

        updates: dict[str, int] = {}
        directions = self.mam_bridge_config.outer.directions
        for index, audit in enumerate(audit_history):
            direction = directions[index % len(directions)]
            record = audit.get("work_accounting")
            derivation = record.get("derivation") if isinstance(record, dict) else None
            conditional = derivation.get("conditional") if isinstance(derivation, dict) else None
            if not isinstance(conditional, dict):
                raise ValueError("built-in checkpoint lacks conditional progress derivation")
            completed = conditional.get("policy_iterations_completed")
            _require_integer("policy_iterations_completed", completed, minimum=1)
            refresh = conditional.get("final_costate_refresh_executed")
            if not isinstance(refresh, bool):
                raise TypeError("final_costate_refresh_executed must be bool")
            updates[direction] = updates.get(direction, 0) + (
                (int(cast(int, completed)) + int(refresh))
                * self.mam_bridge_config.conditional.costate_steps
            )
        return updates

    def train(
        self,
        key: PRNGKey,
        training_config: TrainingConfig | None = None,
        callback: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> SolverResult:
        train_config = training_config or TrainingConfig(
            num_iterations=1,
            batch_size=self.mam_bridge_config.conditional.batch_size,
        )
        directions = self.mam_bridge_config.outer.directions
        total_half_iterations = self.mam_bridge_config.outer.num_iterations * len(directions)
        if self._completed_half_iterations > total_half_iterations:
            raise RuntimeError("MAM bridge progress exceeds the configured outer schedule")
        if self._completed_half_iterations == total_half_iterations and self._is_trained:
            raise RuntimeError(
                "MAM bridge outer schedule is already complete; construct a new solver for "
                "a fresh run"
            )
        resuming = (
            self._params is not None
            and self._resume_pairs is not None
            and self._rng_ledger is not None
            and 0 < self._completed_half_iterations <= total_half_iterations
        )
        if resuming:
            assert self._params is not None
            assert self._resume_pairs is not None
            assert self._rng_ledger is not None
            params = dict(self._params)
            pairs = self._resume_pairs
            ledger = self._rng_ledger
        else:
            self.conditional_solver.load_state_dict(copy.deepcopy(self._initial_conditional_state))
            key, init_key = jax.random.split(key)
            params = self.init_params(init_key)
            ledger = RNGLedger(self._seed_from_key(key))
            pair_key, ledger = ledger.next(RNGDomain.PAIR_CACHE)
            pairs = self._initial_pair_cache(pair_key, self.mam_bridge_config.outer.cache_size)
            self._completed_half_iterations = 0
            self._loss_history = []
            self._audit_history = []
            self._source_calibration = None
            self._target_calibration = None
            self._global_endpoint_pass = False
            self._last_direction = None
            self._last_metrics = {}

        self._validate_pair_cache(pairs)

        for half_index in range(self._completed_half_iterations, total_half_iterations):
            outer_iteration = half_index // len(directions)
            direction_index = half_index % len(directions)
            direction = directions[direction_index]
            half_iteration = half_index + 1
            conditional_key, ledger = ledger.next(
                RNGDomain.COSTATE_FIT, outer_iteration, direction_index
            )
            projection_key, ledger = ledger.next(
                RNGDomain.PROJECTION_FIT, outer_iteration, direction_index
            )
            projection_accept_key, ledger = ledger.next(
                RNGDomain.PROJECTION_EVALUATION,
                outer_iteration,
                direction_index,
            )
            refresh_key, ledger = ledger.next(
                RNGDomain.COUPLING_REFRESH, outer_iteration, direction_index
            )
            audit_key, ledger = ledger.next(RNGDomain.REPORTING, outer_iteration, direction_index)
            conditional_snapshot = copy.deepcopy(self.conditional_solver.state_dict())
            source_calibration_snapshot = self._source_calibration
            target_calibration_snapshot = self._target_calibration
            try:
                conditional = self.conditional_solver.solve(conditional_key, pairs, direction)
                self._validate_conditional_result(conditional, pairs, direction)
                projection = self.projector.fit(projection_key, conditional, direction)
                if not bool(jax.device_get(projection.finite)):
                    raise FloatingPointError("nonfinite Markov projection; refusing update")
                candidate_params, projection_acceptance = self._accept_projection(
                    projection_accept_key,
                    params,
                    direction,
                    projection.params,
                )
                candidate_pairs = self._refresh_pairs(refresh_key, candidate_params, direction)
                audit = self._audit_endpoints(audit_key, candidate_params)
                projection_confirmation = projection_acceptance["confirmation"] is not None
                projection_accepted = bool(projection_acceptance["projection_accepted"])
                exact_builtin = type(self.conditional_solver) is MAMConditionalSolver
                if exact_builtin:
                    acceptance_history = conditional.metrics.get("actor_acceptance_history")
                    if not isinstance(acceptance_history, list) or not acceptance_history:
                        raise ValueError(
                            "built-in conditional result lacks actor acceptance provenance"
                        )
                    conditional_derivation: dict[str, Any] | None = {
                        "policy_iterations_completed": len(acceptance_history),
                        "actor_confirmation_executed": [
                            item.get("confirmation") is not None for item in acceptance_history
                        ],
                        "actor_update_accepted": [
                            bool(item.get("actor_update_accepted")) for item in acceptance_history
                        ],
                        "output_actor_fingerprint": conditional.metrics.get(
                            "output_actor_fingerprint"
                        ),
                        "pre_refresh_costate_policy_fingerprint": conditional.metrics.get(
                            "pre_refresh_costate_policy_fingerprint"
                        ),
                        "final_costate_refresh_executed": conditional.metrics.get(
                            "final_costate_refresh_executed"
                        ),
                    }
                else:
                    conditional_derivation = None
                work_derivation = {
                    "half_iteration": half_iteration,
                    "direction": direction,
                    "conditional": conditional_derivation,
                    "global": {
                        "projection_confirmation_executed": projection_confirmation,
                        "projection_update_accepted": projection_accepted,
                    },
                }
                if not exact_builtin:
                    half_work = None
                    cumulative_work = None
                    uncertified_reason = "conditional_backend_is_not_exact_builtin_mam"
                else:
                    expected_conditional_work = self._conditional_work_from_derivation(
                        conditional_derivation
                    )
                    if conditional.certified_work_counters != expected_conditional_work:
                        raise ValueError(
                            "built-in conditional counters disagree with exact derivation"
                        )
                    globally_feasible = bool(
                        half_index > 0
                        and self._audit_history[half_index - 1]["global_endpoint_pass"]
                    )
                    if (
                        bool(projection_acceptance["globally_feasible_before_projection"])
                        != globally_feasible
                    ):
                        raise ValueError(
                            "projection feasibility branch disagrees with prior endpoint audit"
                        )
                    half_work = self._half_work_from_derivation(
                        work_derivation,
                        history_index=half_index,
                        globally_feasible_before_projection=globally_feasible,
                    )
                    prior_work = self._certified_total_from_audit_history(self._audit_history)
                    cumulative_work = None if prior_work is None else prior_work.merge(half_work)
                    uncertified_reason = None
                work_accounting = _work_accounting_record(
                    half_work,
                    scope="successful_global_half_iteration",
                    cumulative=cumulative_work,
                    uncertified_reason=uncertified_reason,
                    derivation=work_derivation,
                )
                audit = {**audit, "work_accounting": work_accounting}
            except Exception:
                self.conditional_solver.load_state_dict(conditional_snapshot)
                self._source_calibration = source_calibration_snapshot
                self._target_calibration = target_calibration_snapshot
                raise
            params = candidate_params
            pairs = candidate_pairs
            last_metrics = {
                "loss": projection.loss,
                "outer_iteration": outer_iteration,
                "half_iteration": half_iteration,
                "direction": direction,
                "conditional_status": conditional.status,
                "conditional": conditional.metrics,
                "projection_acceptance": projection_acceptance,
                "audit": audit,
                "work_accounting": work_accounting,
            }
            self._loss_history.append(float(jax.device_get(projection.loss)))
            self._audit_history.append(audit)
            self._last_direction = direction
            self._params = params
            self._resume_pairs = pairs
            self._rng_ledger = ledger
            self._completed_half_iterations = half_iteration
            self._last_metrics = last_metrics
            self._maybe_save_checkpoint(
                train_config,
                step=half_iteration,
                params=params,
                opt_state=None,
                loss_history=self._loss_history,
                metrics=last_metrics,
            )
            if callback is not None:
                callback(half_iteration, last_metrics)

        final_audit = self._audit_history[-1]
        self._global_endpoint_pass = bool(final_audit["global_endpoint_pass"])
        self._params = params
        self._is_trained = True
        certified_total = self._certified_total_from_audit_history(self._audit_history)
        if certified_total is None:
            final_work_accounting = _work_accounting_record(
                None,
                scope="complete_training_run",
                uncertified_reason="one_or_more_global_half_iterations_uncertified",
            )
        else:
            final_work_accounting = _work_accounting_record(
                certified_total,
                scope="complete_training_run",
                cumulative=certified_total,
            )
        forward_source = float(final_audit["forward_source_mmd2"])
        forward_target = float(final_audit["forward_target_mmd2"])
        diagnostics = DiagnosticReport(
            marginal_error_source=forward_source,
            marginal_error_target=forward_target,
            metadata={
                "backward_source_mmd2": final_audit["backward_source_mmd2"],
                "backward_target_mmd2": final_audit["backward_target_mmd2"],
                "audit_history": self._audit_history,
            },
        )
        metadata = {
            "converged": self._global_endpoint_pass,
            "status": self.status,
            "solver_type": self.solver_type.name,
            "algorithm": "MAM_GSBM_EXPERIMENTAL",
            "num_half_iterations": self._completed_half_iterations,
            "last_direction": self._last_direction,
            "conditional_status": self.conditional_solver.status,
            "conditional_endpoints_exact": True,
            "global_endpoints_empirically_audited": True,
            "global_endpoint_pass": self._global_endpoint_pass,
            "production_scalability_validated": False,
            "matrix_free_costate_labels": bool(
                type(self.conditional_solver) is MAMConditionalSolver
                and self.mam_bridge_config.conditional.costate.matrix_free_labels
            ),
            "matrix_free_costate_labels_backend_reported": (
                type(self.conditional_solver) is MAMConditionalSolver
            ),
            "markov_projection_exact": False,
            "markov_projection_semantics": ("finite_grid_euler_conditional_mean_field_regression"),
            "single_device_execution_path": True,
            "single_device_gpu_gate_passed": False,
            "multi_device_required": False,
            "two_device_execution_active": False,
            "two_device_acceleration_requested": bool(
                self.mam_bridge_config.execution.allow_two_devices
            ),
            "execution_plan": self._execution_plan.to_state(),
            "device_topology": self._device_topology.to_state(),
            "rng_ledger": ledger.to_state(),
            "pair_cache_sha256": self._pair_cache_hash(pairs),
            "fingerprints": self._scientific_fingerprints(),
            "dependencies": self._dependency_versions(),
            "final_audit": final_audit,
            "work_accounting": final_work_accounting,
        }
        final_checkpoint_path = self._maybe_save_checkpoint(
            train_config,
            step=self._completed_half_iterations,
            params=params,
            opt_state=None,
            loss_history=self._loss_history,
            metrics=self._last_metrics,
            final=True,
            metadata=metadata,
        )
        if final_checkpoint_path is not None:
            metadata["checkpoint_path"] = final_checkpoint_path
        return SolverResult(
            params=params,
            loss_history=jnp.asarray(self._loss_history),
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def solve(
        self,
        key: PRNGKey,
        training_config: TrainingConfig | None = None,
    ) -> MAMBridgeSolution:
        """Train and expose only the exact-grid runtime used by all audits."""
        result = self.train(key, training_config)
        metadata = dict(result.metadata)
        metadata["integrator_type"] = self.integrator.type.name
        metadata["runtime_semantics"] = "EXACT_GRID_NATIVE_EULER_ONLY"
        solution = MAMBridgeSolution(
            problem=self.problem,
            solver_type=self.solver_type,
            params=result.params,
            representation=self.representation_type,
            metadata=metadata,
            initial_sampler=self._sample_source,
            terminal_sampler=self._sample_target,
            projector=self.projector,
        )
        solution._integrator = self.integrator
        solution._forward_drift = self.extract_drift(result.params)
        solution._backward_drift = self.extract_backward_drift(result.params)
        return solution

    def sample(
        self,
        key: PRNGKey,
        num_samples: int,
        params: Params | None = None,
        x0: Array | None = None,
        direction: str = "forward",
    ) -> TrajectoryBatch:
        num_samples = _validate_public_sample_count(num_samples)
        active = params if params is not None else self._params
        if active is None:
            raise ValueError("Solver not trained. Call train() first or provide params.")
        self._validate_public_projection_params(active)
        if x0 is not None:
            x0 = _validated_public_start(
                x0,
                num_samples=num_samples,
                dim=self.problem.dim,
                dtype=self.mam_bridge_config.execution.production_dtype,
                name="public start",
            )
        if direction in {"forward", "f"}:
            trajectories = self._sample_direction(key, num_samples, active, "f", start=x0)
        elif direction in {"backward", "b"}:
            trajectories = self._sample_direction(key, num_samples, active, "b", start=x0)
        else:
            raise ValueError("direction must be 'forward'/'f' or 'backward'/'b'")
        _validate_public_paths(
            trajectories.paths,
            num_samples=num_samples,
            num_steps=self.problem.time_grid.num_steps,
            dim=self.problem.dim,
            dtype=self.mam_bridge_config.execution.production_dtype,
        )
        return trajectories

    def sample_backward(
        self,
        key: PRNGKey,
        num_samples: int,
        params: Params | None = None,
        xN: Array | None = None,
    ) -> TrajectoryBatch:
        return self.sample(key, num_samples, params=params, x0=xN, direction="b")

    def extract_drift(
        self,
        params: Params | None = None,
        *,
        direction: str = "forward",
    ) -> DriftFn:
        active = params if params is not None else self._params
        if active is None:
            raise ValueError("No Markov projection parameters available")
        self._validate_public_projection_params(active)
        if direction in {"forward", "f"}:
            direction_code = "f"
        elif direction in {"backward", "b"}:
            direction_code = "b"
        else:
            raise ValueError("direction must be 'forward'/'f' or 'backward'/'b'")
        net_params = active[direction_code.upper()]
        dt = self.problem.time_grid.dt
        num_steps = self.problem.time_grid.num_steps

        def drift(x: Array, t: Scalar) -> Array:
            x_array = jnp.asarray(x)
            if not jnp.issubdtype(x_array.dtype, jnp.floating):
                raise TypeError("MAM bridge drift state must have a floating dtype")
            unbatched = x_array.ndim == 1
            state = jnp.atleast_2d(x_array)
            time = jnp.asarray(t, dtype=state.dtype)
            if time.ndim != 0:
                raise ValueError("MAM bridge drift time must be a scalar")
            physical_grid = _projection_physical_times(
                self.problem,
                state.dtype,
                direction_code,
            )
            distances = jnp.abs(physical_grid - time)
            step = jnp.argmin(distances).astype(jnp.int32)
            nominal_time = physical_grid[step]
            scale = jnp.maximum(
                jnp.maximum(
                    jnp.abs(jnp.asarray(dt, dtype=state.dtype)),
                    jnp.maximum(jnp.abs(time), jnp.abs(nominal_time)),
                ),
                jnp.asarray(jnp.finfo(state.dtype).tiny, dtype=state.dtype),
            )
            tolerance = 32.0 * jnp.finfo(state.dtype).eps * scale
            valid_time = jnp.isfinite(time) & (jnp.abs(time - nominal_time) <= tolerance)
            if not isinstance(time, jax.core.Tracer):
                if not bool(np.asarray(jax.device_get(valid_time))):
                    raise ValueError("MAM bridge drift time lies outside its exact grid")
            prediction = self.projector.predict(net_params, state, step, direction_code)
            remaining = (num_steps - step).astype(state.dtype) * jnp.asarray(
                dt,
                dtype=state.dtype,
            )
            value = (prediction - state) / remaining
            value = jnp.where(valid_time, value, jnp.nan)
            return value[0] if unbatched else value

        return drift

    def extract_backward_drift(self, params: Params | None = None) -> DriftFn:
        return self.extract_drift(params, direction="backward")

    def _checkpoint_filename(self, config: TrainingConfig, step: int, final: bool) -> str:
        """Use a MAM-specific namespace despite the provisional solver enum."""
        suffix = "final" if final else f"step_{step:08d}"
        return f"{config.checkpoint_prefix}_mam_bridge_{suffix}.pkl"

    def _checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "conditional_solver": copy.deepcopy(self.conditional_solver.state_dict()),
            "audit_history": self._audit_history,
            "last_direction": self._last_direction,
            "global_endpoint_pass": self._global_endpoint_pass,
            "source_calibration": self._source_calibration,
            "target_calibration": self._target_calibration,
            "pair_cache": self._resume_pairs,
            "pair_cache_sha256": (
                None if self._resume_pairs is None else self._pair_cache_hash(self._resume_pairs)
            ),
            "completed_half_iterations": self._completed_half_iterations,
            "loss_history": self._loss_history,
            "last_metrics": self._last_metrics,
            "rng_ledger": (None if self._rng_ledger is None else self._rng_ledger.to_state()),
            "execution_plan": self._execution_plan.to_state(),
            "device_topology": self._device_topology.to_state(),
            "dependencies": self._dependency_versions(),
            "fingerprints": self._scientific_fingerprints(),
        }

    @staticmethod
    def _validated_device_topology_state(value: Any) -> dict[str, Any]:
        """Validate saved origin metadata without requiring current-device equality."""
        expected = {
            "platform",
            "device_kind",
            "available_local_device_count",
            "selected_device_count",
            "selected_device_ids",
            "process_count",
            "process_index",
            "batch_data_parallel",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("MAM bridge checkpoint device-topology schema mismatch")
        if not isinstance(value["platform"], str) or not isinstance(value["device_kind"], str):
            raise TypeError("checkpoint device platform and kind must be strings")
        selected_ids = value["selected_device_ids"]
        if not isinstance(selected_ids, list) or any(
            isinstance(device_id, bool) or not isinstance(device_id, (int, np.integer))
            for device_id in selected_ids
        ):
            raise TypeError("checkpoint selected_device_ids must be a list of integers")
        if not isinstance(value["batch_data_parallel"], bool):
            raise TypeError("checkpoint batch_data_parallel must be bool")
        integer_names = (
            "available_local_device_count",
            "selected_device_count",
            "process_count",
            "process_index",
        )
        if any(
            isinstance(value[name], bool) or not isinstance(value[name], (int, np.integer))
            for name in integer_names
        ):
            raise TypeError("checkpoint device-topology counts must be integers")
        topology = DeviceTopology(
            platform=value["platform"],
            device_kind=value["device_kind"],
            available_local_device_count=int(value["available_local_device_count"]),
            selected_device_count=int(value["selected_device_count"]),
            selected_device_ids=tuple(int(device_id) for device_id in selected_ids),
            process_count=int(value["process_count"]),
            process_index=int(value["process_index"]),
            batch_data_parallel=value["batch_data_parallel"],
        )
        return topology.to_state()

    def _validate_checkpoint_envelope(self, payload: Any) -> None:
        """Validate the strict v1 outer envelope after transactional restore."""
        expected = {
            "format_version",
            "solver_class",
            "solver_type",
            "representation_type",
            "step",
            "params",
            "opt_state",
            "loss_history",
            "metrics",
            "metadata",
            "training_config",
            "solver_state",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("MAM bridge checkpoint envelope schema mismatch")
        if (
            isinstance(payload["format_version"], bool)
            or not isinstance(payload["format_version"], (int, np.integer))
            or int(payload["format_version"]) != 1
        ):
            raise ValueError("unsupported MAM bridge checkpoint format version")
        expected_values = {
            "solver_class": type(self).__name__,
            "solver_type": self.solver_type.name,
            "representation_type": self.representation_type.name,
        }
        for name, expected_value in expected_values.items():
            if payload[name] != expected_value:
                raise ValueError(f"MAM bridge checkpoint {name} mismatch")
        step = payload["step"]
        if step is not None:
            _require_integer("checkpoint step", step, minimum=0)
        if not isinstance(payload["loss_history"], list):
            raise TypeError("checkpoint loss_history must be a list")
        if not isinstance(payload["metrics"], dict):
            raise TypeError("checkpoint metrics must be a dictionary")
        if not isinstance(payload["metadata"], dict):
            raise TypeError("checkpoint metadata must be a dictionary")
        if payload["training_config"] is not None and not isinstance(
            payload["training_config"], dict
        ):
            raise TypeError("checkpoint training_config must be a dictionary or None")
        if not isinstance(payload["solver_state"], dict):
            raise TypeError("checkpoint solver_state must be a dictionary")

    def _validate_final_checkpoint_metadata(self, metadata: Any) -> bool:
        """Validate generated final metadata; partial checkpoints remain resumable."""

        if not isinstance(metadata, dict):
            raise TypeError("MAM bridge checkpoint metadata must be a dictionary")
        if metadata.get("algorithm") != "MAM_GSBM_EXPERIMENTAL":
            return False
        required = {
            "algorithm",
            "num_half_iterations",
            "converged",
            "status",
            "global_endpoint_pass",
            "last_direction",
            "conditional_status",
            "execution_plan",
            "device_topology",
            "rng_ledger",
            "pair_cache_sha256",
            "fingerprints",
            "dependencies",
            "final_audit",
            "work_accounting",
        }
        # Manually saved or periodic checkpoints may carry a short descriptive
        # metadata dictionary.  It must never make a partial state inference-ready.
        if not required.issubset(metadata):
            return False
        completed = metadata["num_half_iterations"]
        _require_integer("checkpoint metadata num_half_iterations", completed, minimum=0)
        total_half_iterations = self.mam_bridge_config.outer.num_iterations * len(
            self.mam_bridge_config.outer.directions
        )
        if completed != self._completed_half_iterations or completed != total_half_iterations:
            raise ValueError("final checkpoint metadata disagrees with completed schedule")
        if not self._audit_history:
            raise ValueError("final checkpoint metadata requires a completed endpoint audit")
        expected_pass = self._audit_history[-1]["global_endpoint_pass"]
        for name in ("converged", "global_endpoint_pass"):
            if not isinstance(metadata[name], bool):
                raise TypeError(f"checkpoint metadata {name} must be bool")
            if metadata[name] is not expected_pass:
                raise ValueError(f"checkpoint metadata {name} disagrees with final audit")
        if self._global_endpoint_pass is not expected_pass:
            raise ValueError("checkpoint global endpoint state disagrees with final audit")
        expected_status = (
            "GLOBAL_ENDPOINT_AUDIT_PASSED"
            if expected_pass
            else "EXPERIMENTAL_GLOBAL_ENDPOINT_UNVERIFIED"
        )
        expected_values = {
            "status": expected_status,
            "last_direction": self._last_direction,
            "conditional_status": self.conditional_solver.status,
            "execution_plan": self._execution_plan.to_state(),
            "device_topology": (
                self._device_topology.to_state()
                if self._checkpoint_origin_device_topology is None
                else self._checkpoint_origin_device_topology
            ),
            "rng_ledger": None if self._rng_ledger is None else self._rng_ledger.to_state(),
            "pair_cache_sha256": (
                None if self._resume_pairs is None else self._pair_cache_hash(self._resume_pairs)
            ),
            "fingerprints": self._scientific_fingerprints(),
            "dependencies": self._dependency_versions(),
            "final_audit": self._audit_history[-1],
        }
        for name, expected in expected_values.items():
            if metadata[name] != expected:
                raise ValueError(f"checkpoint metadata {name} is inconsistent")
        certified_total = self._certified_total_from_audit_history(self._audit_history)
        expected_work = (
            _work_accounting_record(
                None,
                scope="complete_training_run",
                uncertified_reason="one_or_more_global_half_iterations_uncertified",
            )
            if certified_total is None
            else _work_accounting_record(
                certified_total,
                scope="complete_training_run",
                cumulative=certified_total,
            )
        )
        if metadata["work_accounting"] != expected_work:
            raise ValueError("checkpoint metadata work accounting is inconsistent")
        return True

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Restore a checkpoint transactionally after scientific validation.

        The base loader installs parameters before subclass validation.  MAM
        checkpoints carry stronger problem/cost/code invariants, so a rejected
        payload must leave this solver exactly as it was before the attempt.
        """
        snapshot: _CheckpointRollbackSnapshot = {
            "params": self._params,
            "is_trained": self._is_trained,
            "conditional_solver": copy.deepcopy(self.conditional_solver.state_dict()),
            "audit_history": self._audit_history,
            "last_direction": self._last_direction,
            "global_endpoint_pass": self._global_endpoint_pass,
            "source_calibration": self._source_calibration,
            "target_calibration": self._target_calibration,
            "resume_pairs": self._resume_pairs,
            "completed_half_iterations": self._completed_half_iterations,
            "loss_history": self._loss_history,
            "last_metrics": self._last_metrics,
            "rng_ledger": self._rng_ledger,
            "checkpoint_origin_device_topology": self._checkpoint_origin_device_topology,
        }
        try:
            payload = super().load_checkpoint(path)
            self._validate_checkpoint_envelope(payload)
            self._is_trained = self._validate_final_checkpoint_metadata(payload.get("metadata", {}))
            return payload
        except Exception:
            self._params = snapshot["params"]
            self._is_trained = snapshot["is_trained"]
            self.conditional_solver.load_state_dict(snapshot["conditional_solver"])
            self._audit_history = snapshot["audit_history"]
            self._last_direction = snapshot["last_direction"]
            self._global_endpoint_pass = snapshot["global_endpoint_pass"]
            self._source_calibration = snapshot["source_calibration"]
            self._target_calibration = snapshot["target_calibration"]
            self._resume_pairs = snapshot["resume_pairs"]
            self._completed_half_iterations = snapshot["completed_half_iterations"]
            self._loss_history = snapshot["loss_history"]
            self._last_metrics = snapshot["last_metrics"]
            self._rng_ledger = snapshot["rng_ledger"]
            self._checkpoint_origin_device_topology = snapshot["checkpoint_origin_device_topology"]
            raise

    def _validate_calibration_checkpoint(
        self,
        name: str,
        calibration: Any,
    ) -> None:
        if not isinstance(calibration, NullCalibrationResult):
            raise TypeError(f"checkpoint {name} calibration has the wrong type")
        if not isinstance(calibration.finite, bool) or not isinstance(
            calibration.sinkhorn_converged, bool
        ):
            raise TypeError(f"checkpoint {name} calibration flags must be bool")
        if not isinstance(calibration.status, str) or not calibration.status:
            raise ValueError(f"checkpoint {name} calibration status must be nonempty")
        thresholds = calibration.thresholds
        if not isinstance(thresholds, EndpointThresholds):
            raise TypeError(f"checkpoint {name} thresholds have the wrong type")
        for field_name in (
            "mmd2",
            "sliced_wasserstein",
            "sinkhorn_divergence",
            "mean_error",
            "covariance_error",
            "null_quantile",
        ):
            _require_finite_real(
                f"checkpoint {name} threshold {field_name}",
                getattr(thresholds, field_name),
            )
        if thresholds.mode_proportion_l1 is not None:
            _require_finite_real(
                f"checkpoint {name} threshold mode_proportion_l1",
                thresholds.mode_proportion_l1,
                nonnegative=True,
            )
        for field_name in ("generated_size", "reference_size", "null_replicates"):
            _require_integer(
                f"checkpoint {name} threshold {field_name}",
                getattr(thresholds, field_name),
                minimum=1,
            )
        if not isinstance(thresholds.valid, bool):
            raise TypeError(f"checkpoint {name} threshold valid flag must be bool")
        expected_reference_size = (
            self.mam_bridge_config.audit.reference_size or self.mam_bridge_config.outer.audit_size
        )
        if (
            thresholds.generated_size != self.mam_bridge_config.outer.audit_size
            or thresholds.reference_size != expected_reference_size
            or thresholds.null_replicates != self.mam_bridge_config.audit.null_replicates
            or thresholds.null_quantile != self.mam_bridge_config.audit.null_quantile
        ):
            raise ValueError(f"checkpoint {name} calibration disagrees with active audit config")
        expected_valid = calibration.finite and calibration.sinkhorn_converged
        if thresholds.valid != expected_valid:
            raise ValueError(f"checkpoint {name} calibration validity flags are inconsistent")
        expected_status = (
            "NULL_CALIBRATED" if thresholds.valid else "INVALID_NULL_CALIBRATION_FAIL_CLOSED"
        )
        if calibration.status != expected_status:
            raise ValueError(f"checkpoint {name} calibration status is inconsistent")
        if not isinstance(calibration.null_metrics, tuple) or len(calibration.null_metrics) != int(
            thresholds.null_replicates
        ):
            raise ValueError(f"checkpoint {name} null-metric count is inconsistent")
        for metric_index, metrics in enumerate(calibration.null_metrics):
            if not isinstance(metrics, EndpointMetrics):
                raise TypeError(f"checkpoint {name} null metric has the wrong type")
            for field_name in (
                "mmd2",
                "sliced_wasserstein",
                "sinkhorn_divergence",
                "mean_error",
                "covariance_error",
                "sinkhorn_marginal_error",
            ):
                _require_finite_real(
                    f"checkpoint {name} null metric {metric_index} {field_name}",
                    getattr(metrics, field_name),
                )
            if metrics.mode_proportion_l1 is not None:
                _require_finite_real(
                    f"checkpoint {name} null metric {metric_index} mode_proportion_l1",
                    metrics.mode_proportion_l1,
                    nonnegative=True,
                )
            for proportions_name in (
                "sample_mode_proportions",
                "reference_mode_proportions",
            ):
                proportions = getattr(metrics, proportions_name)
                if proportions is not None:
                    if not isinstance(proportions, tuple) or not proportions:
                        raise TypeError(
                            f"checkpoint {name} {proportions_name} must be a nonempty tuple"
                        )
                    for value in proportions:
                        _require_finite_real(
                            f"checkpoint {name} {proportions_name}",
                            value,
                            nonnegative=True,
                        )
            if not isinstance(metrics.sinkhorn_converged, bool) or not isinstance(
                metrics.finite, bool
            ):
                raise TypeError(f"checkpoint {name} null metric flags must be bool")
            if not metrics.finite:
                raise FloatingPointError(f"checkpoint {name} null metric is nonfinite")
        expected_finite = all(metric.finite for metric in calibration.null_metrics)
        expected_converged = all(metric.sinkhorn_converged for metric in calibration.null_metrics)
        if (
            calibration.finite is not expected_finite
            or calibration.sinkhorn_converged is not expected_converged
        ):
            raise ValueError(f"checkpoint {name} calibration aggregate flags are inconsistent")

        def calibrated_threshold(field_name: str, floor: float) -> float:
            values = [getattr(metric, field_name) for metric in calibration.null_metrics]
            quantile = float(
                np.quantile(
                    np.asarray(values),
                    self.mam_bridge_config.audit.null_quantile,
                    method="higher",
                )
            )
            return max(
                floor,
                self.mam_bridge_config.audit.null_threshold_scale * quantile,
            )

        floors = self.mam_bridge_config.audit.floors
        expected_threshold_values = {
            "mmd2": calibrated_threshold("mmd2", floors.mmd2),
            "sliced_wasserstein": calibrated_threshold(
                "sliced_wasserstein", floors.sliced_wasserstein
            ),
            "sinkhorn_divergence": calibrated_threshold(
                "sinkhorn_divergence", floors.sinkhorn_divergence
            ),
            "mean_error": calibrated_threshold("mean_error", floors.mean_error),
            "covariance_error": calibrated_threshold("covariance_error", floors.covariance_error),
        }
        mode_values = [metric.mode_proportion_l1 for metric in calibration.null_metrics]
        if all(value is None for value in mode_values):
            expected_mode_threshold = None
        elif any(value is None for value in mode_values):
            raise ValueError(f"checkpoint {name} mode metrics have inconsistent presence")
        else:
            expected_mode_threshold = max(
                floors.mode_proportion_l1,
                self.mam_bridge_config.audit.null_threshold_scale
                * float(
                    np.quantile(
                        np.asarray(mode_values, dtype=np.float64),
                        self.mam_bridge_config.audit.null_quantile,
                        method="higher",
                    )
                ),
            )
        for field_name, expected_value in expected_threshold_values.items():
            if getattr(thresholds, field_name) != expected_value:
                raise ValueError(
                    f"checkpoint {name} threshold {field_name} disagrees with null metrics"
                )
        if thresholds.mode_proportion_l1 != expected_mode_threshold:
            raise ValueError(f"checkpoint {name} mode threshold disagrees with null metrics")

    @staticmethod
    def _validate_finite_metrics_tree(name: str, tree: Any) -> None:
        for leaf in jax.tree_util.tree_leaves(tree):
            if leaf is None or isinstance(leaf, (str, bytes, bool)):
                continue
            try:
                array = np.asarray(jax.device_get(leaf))
            except (TypeError, ValueError):
                continue
            if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                raise FloatingPointError(f"checkpoint {name} contains nonfinite numeric values")

    def _restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        expected_state_fields = {
            "schema_version",
            "conditional_solver",
            "audit_history",
            "last_direction",
            "global_endpoint_pass",
            "source_calibration",
            "target_calibration",
            "pair_cache",
            "pair_cache_sha256",
            "completed_half_iterations",
            "loss_history",
            "last_metrics",
            "rng_ledger",
            "execution_plan",
            "device_topology",
            "dependencies",
            "fingerprints",
        }
        if not isinstance(state, dict) or set(state) != expected_state_fields:
            raise ValueError("MAM bridge checkpoint state schema mismatch")
        if state.get("schema_version") != 1:
            raise ValueError("unsupported or missing MAM bridge checkpoint schema")
        if state.get("fingerprints") != self._scientific_fingerprints():
            raise ValueError("MAM bridge checkpoint problem/config/cost fingerprint mismatch")
        if state.get("execution_plan") != self._execution_plan.to_state():
            raise ValueError("MAM bridge checkpoint execution-plan mismatch")
        if state.get("dependencies") != self._dependency_versions():
            raise ValueError("MAM bridge checkpoint dependency-version mismatch")
        self._checkpoint_origin_device_topology = self._validated_device_topology_state(
            state.get("device_topology")
        )
        if not isinstance(self._params, dict) or set(self._params) != {"F", "B"}:
            raise ValueError("MAM bridge checkpoint parameters must contain F and B")
        for direction, value in self._params.items():
            try:
                self.projector.validate_params(value)
            except (TypeError, ValueError, FloatingPointError) as exc:
                raise type(exc)(
                    f"MAM bridge checkpoint {direction} projection state is invalid: {exc}"
                ) from exc
        conditional_state = state.get("conditional_solver")
        if (
            not isinstance(conditional_state, dict)
            or conditional_state.get("schema_version") != 1
            or conditional_state.get("backend_status") != self.conditional_solver.status
        ):
            raise ValueError("MAM bridge checkpoint conditional-state schema/status mismatch")
        self.conditional_solver.load_state_dict(conditional_state)
        audit_history = state["audit_history"]
        if not isinstance(audit_history, list):
            raise TypeError("MAM bridge checkpoint audit_history must be a list")
        self._audit_history = audit_history
        self._last_direction = state.get("last_direction")
        global_endpoint_pass = state["global_endpoint_pass"]
        if not isinstance(global_endpoint_pass, bool):
            raise TypeError("MAM bridge checkpoint global_endpoint_pass must be bool")
        self._global_endpoint_pass = global_endpoint_pass
        self._source_calibration = state.get("source_calibration")
        self._target_calibration = state.get("target_calibration")
        self._resume_pairs = state.get("pair_cache")
        expected_pair_hash = state.get("pair_cache_sha256")
        if (self._resume_pairs is None) != (expected_pair_hash is None):
            raise ValueError("MAM bridge checkpoint pair-cache presence mismatch")
        if self._resume_pairs is not None and expected_pair_hash != self._pair_cache_hash(
            self._resume_pairs
        ):
            raise ValueError("MAM bridge checkpoint pair-cache hash mismatch")
        if self._resume_pairs is not None:
            self._validate_pair_cache(self._resume_pairs)
        completed_half_iterations = state["completed_half_iterations"]
        _require_integer(
            "checkpoint completed_half_iterations",
            completed_half_iterations,
            minimum=0,
        )
        self._completed_half_iterations = int(completed_half_iterations)
        loss_history = state["loss_history"]
        if not isinstance(loss_history, list):
            raise TypeError("MAM bridge checkpoint loss_history must be a list")
        self._loss_history = []
        for index, loss in enumerate(loss_history):
            _require_finite_real(f"checkpoint loss_history[{index}]", loss)
            self._loss_history.append(float(loss))
        last_metrics = state["last_metrics"]
        if not isinstance(last_metrics, dict):
            raise TypeError("MAM bridge checkpoint last_metrics must be a dictionary")
        self._last_metrics = last_metrics
        self._validate_finite_metrics_tree("last_metrics", self._last_metrics)
        self._validate_finite_metrics_tree("audit_history", self._audit_history)
        ledger_state = state.get("rng_ledger")
        self._rng_ledger = None if ledger_state is None else RNGLedger.from_state(ledger_state)
        directions = self.mam_bridge_config.outer.directions
        total_half_iterations = self.mam_bridge_config.outer.num_iterations * len(directions)
        completed = self._completed_half_iterations
        if not 0 <= completed <= total_half_iterations:
            raise ValueError("MAM bridge checkpoint half-iteration count is invalid")
        if len(self._loss_history) != completed or len(self._audit_history) != completed:
            raise ValueError("MAM bridge checkpoint histories are inconsistent")
        if completed > 0:
            self._validate_calibration_checkpoint("source", self._source_calibration)
            self._validate_calibration_checkpoint("target", self._target_calibration)
        self._certified_total_from_audit_history(self._audit_history)
        if completed == 0:
            if (
                self._global_endpoint_pass
                or any(
                    value is not None
                    for value in (
                        self._resume_pairs,
                        self._rng_ledger,
                        self._last_direction,
                        self._source_calibration,
                        self._target_calibration,
                    )
                )
                or self._last_metrics
            ):
                raise ValueError("empty MAM bridge checkpoint contains partial runtime state")
        else:
            last_audit_pass = self._audit_history[-1]["global_endpoint_pass"]
            if self._global_endpoint_pass and (
                completed != total_half_iterations or last_audit_pass is not True
            ):
                raise ValueError(
                    "checkpoint global endpoint status disagrees with validated progress/audit"
                )
            if type(self.conditional_solver) is MAMConditionalSolver:
                expected_updates = self._expected_costate_updates_by_direction(self._audit_history)
                self.conditional_solver.validate_checkpoint_progress(expected_updates)
            expected_direction = directions[(completed - 1) % len(directions)]
            if (
                self._resume_pairs is None
                or self._rng_ledger is None
                or self._source_calibration is None
                or self._target_calibration is None
                or not self._last_metrics
                or self._last_direction != expected_direction
            ):
                raise ValueError("MAM bridge checkpoint resume state is incomplete")
            last_half_iteration = self._last_metrics.get("half_iteration")
            _require_integer(
                "checkpoint last_metrics half_iteration",
                last_half_iteration,
                minimum=1,
            )
            if last_half_iteration != completed:
                raise ValueError("checkpoint last_metrics half iteration disagrees with progress")
            last_outer_iteration = self._last_metrics.get("outer_iteration")
            _require_integer(
                "checkpoint last_metrics outer_iteration",
                last_outer_iteration,
                minimum=0,
            )
            if last_outer_iteration != (completed - 1) // len(directions):
                raise ValueError("checkpoint last_metrics outer iteration disagrees with progress")
            if self._last_metrics.get("direction") != expected_direction:
                raise ValueError("checkpoint last_metrics direction disagrees with progress")
            if self._last_metrics.get("conditional_status") != self.conditional_solver.status:
                raise ValueError("checkpoint conditional status disagrees with active backend")
            last_loss_array = np.asarray(jax.device_get(self._last_metrics.get("loss")))
            if (
                last_loss_array.shape != ()
                or not np.issubdtype(last_loss_array.dtype, np.number)
                or np.issubdtype(last_loss_array.dtype, np.bool_)
                or not np.isfinite(last_loss_array).item()
            ):
                raise TypeError("checkpoint last_metrics loss must be a finite scalar")
            if float(last_loss_array) != self._loss_history[-1]:
                raise ValueError("checkpoint last loss disagrees with loss history")
            if (
                self._last_metrics.get("work_accounting")
                != self._audit_history[-1]["work_accounting"]
            ):
                raise ValueError(
                    "MAM bridge checkpoint work accounting disagrees with audit history"
                )
            if self._last_metrics.get("audit") != self._audit_history[-1]:
                raise ValueError("MAM bridge checkpoint last audit disagrees with audit history")
            last_derivation = self._audit_history[-1]["work_accounting"]["derivation"]
            projection_acceptance = self._last_metrics.get("projection_acceptance")
            if not isinstance(projection_acceptance, dict) or not isinstance(
                projection_acceptance.get("projection_accepted"), bool
            ):
                raise TypeError("checkpoint projection acceptance metadata is invalid")
            observed_projection_decisions = {
                "projection_confirmation_executed": (
                    projection_acceptance.get("confirmation") is not None
                ),
                "projection_update_accepted": projection_acceptance["projection_accepted"],
            }
            if observed_projection_decisions != last_derivation["global"]:
                raise ValueError("checkpoint projection decisions disagree with work derivation")
            expected_globally_feasible = bool(
                completed > 1 and self._audit_history[-2]["global_endpoint_pass"]
            )
            if (
                projection_acceptance.get("globally_feasible_before_projection")
                is not expected_globally_feasible
            ):
                raise ValueError(
                    "checkpoint projection feasibility branch disagrees with prior audit"
                )
            if type(self.conditional_solver) is MAMConditionalSolver:
                conditional_metrics = self._last_metrics.get("conditional")
                if not isinstance(conditional_metrics, dict):
                    raise TypeError("checkpoint conditional metrics are invalid")
                acceptance_history = conditional_metrics.get("actor_acceptance_history")
                if not isinstance(acceptance_history, list):
                    raise TypeError("checkpoint actor acceptance history is invalid")
                if any(
                    not isinstance(item, dict)
                    or not isinstance(item.get("actor_update_accepted"), bool)
                    for item in acceptance_history
                ):
                    raise TypeError("checkpoint actor acceptance decisions are invalid")
                observed_conditional_derivation = {
                    "policy_iterations_completed": len(acceptance_history),
                    "actor_confirmation_executed": [
                        item.get("confirmation") is not None for item in acceptance_history
                    ],
                    "actor_update_accepted": [
                        item.get("actor_update_accepted") for item in acceptance_history
                    ],
                    "output_actor_fingerprint": conditional_metrics.get("output_actor_fingerprint"),
                    "pre_refresh_costate_policy_fingerprint": conditional_metrics.get(
                        "pre_refresh_costate_policy_fingerprint"
                    ),
                    "final_costate_refresh_executed": conditional_metrics.get(
                        "final_costate_refresh_executed"
                    ),
                }
                if observed_conditional_derivation != last_derivation["conditional"]:
                    raise ValueError("checkpoint actor decisions disagree with work derivation")
                actor_params = self.conditional_solver._actor_params[expected_direction]
                expected_actor_fingerprint = self.conditional_solver._policy_fingerprint(
                    expected_direction,
                    actor_params,
                )
                expected_costate_policy_fingerprint = (
                    self.conditional_solver._costate_policy_fingerprint[expected_direction]
                )
                expected_costate_parameter_fingerprint = (
                    self.conditional_solver._parameter_tree_fingerprint(
                        self.conditional_solver._costate_params[expected_direction]
                    )
                )
                expected_provenance = {
                    "output_actor_fingerprint": expected_actor_fingerprint,
                    "costate_policy_fingerprint": expected_costate_policy_fingerprint,
                    "costate_parameter_fingerprint": expected_costate_parameter_fingerprint,
                    "actor_costate_policy_aligned": True,
                }
                for name, expected_value in expected_provenance.items():
                    if conditional_metrics.get(name) != expected_value:
                        raise ValueError(
                            f"checkpoint conditional {name} provenance is inconsistent"
                        )
                refresh = conditional_metrics.get("final_costate_refresh_executed")
                refresh_loss = conditional_metrics.get("final_costate_refresh_loss")
                if refresh is True:
                    _require_finite_real("checkpoint final_costate_refresh_loss", refresh_loss)
                elif refresh is False:
                    if refresh_loss is not None:
                        raise ValueError("checkpoint unexecuted final costate refresh has a loss")
                else:
                    raise TypeError("checkpoint final_costate_refresh_executed must be bool")
                critic_fingerprint = conditional_metrics.get("value_critic_policy_fingerprint")
                critic_matches = conditional_metrics.get("value_critic_matches_output_actor")
                if not isinstance(critic_fingerprint, str) or not isinstance(
                    critic_matches,
                    bool,
                ):
                    raise TypeError("checkpoint value-critic policy provenance is invalid")
                if critic_matches != (critic_fingerprint == expected_actor_fingerprint):
                    raise ValueError(
                        "checkpoint value-critic/output-actor provenance is inconsistent"
                    )
            counters = self._rng_ledger.to_state()["counters"]
            expected_counters = {
                domain.name.lower(): (
                    1
                    if domain is RNGDomain.PAIR_CACHE
                    else completed
                    if domain
                    in {
                        RNGDomain.COSTATE_FIT,
                        RNGDomain.PROJECTION_FIT,
                        RNGDomain.PROJECTION_EVALUATION,
                        RNGDomain.COUPLING_REFRESH,
                        RNGDomain.REPORTING,
                    }
                    else 0
                )
                for domain in RNGDomain
            }
            if counters != expected_counters:
                raise ValueError("MAM bridge checkpoint RNG counters disagree with progress")


__all__ = [
    "ConditionalBridgeSolver",
    "ConditionalMAMConfig",
    "ConditionalMAMResult",
    "EndpointPairBatch",
    "MAMBridgeConfig",
    "MAMBridgeSolver",
    "MAMConditionalSolver",
    "MAMExecutionConfig",
    "MAMOuterLoopConfig",
    "MarkovProjectionConfig",
    "MarkovProjector",
    "ProjectionResult",
    "ValueOnlyRunningPotential",
]
