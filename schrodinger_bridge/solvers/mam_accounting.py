"""Exact work accounting and synchronized timing for experimental MAM runs.

The scientific counters in :class:`MAMWorkCounters` are scalar, nonnegative
integers.  They count physical work rather than vectorized call sites:

* ``running_cost_oracle_evaluations`` counts individual state/time potential
  queries;
* ``simulated_transitions`` counts individual path transitions;
* ``tangent_vjps`` and ``tangent_jvps`` count individual tangent products;
* ``optimizer_examples`` counts examples consumed by optimizer updates; and
* ``optimizer_updates`` counts committed parameter updates.

Durations are stored as integer nanoseconds, so accounting merges are exact
and independent of reduction order.  Peak memory is a maximum rather than an
additive counter.  ``None`` means that at least one contributing phase did not
measure memory; it propagates through merges.  An all-zero record is the merge
identity because no work occurred whose peak could be unknown.

The timing helper explicitly lowers and compiles a JAX function, then times
synchronized executions separately.  This avoids reporting asynchronous
dispatch latency as execution time.  It does not claim that a host clock or a
device memory probe is exact hardware telemetry.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import jax
import numpy as np

_SCHEMA_VERSION = 1
_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "running_cost_oracle_evaluations",
        "simulated_transitions",
        "tangent_vjps",
        "tangent_jvps",
        "optimizer_examples",
        "optimizer_updates",
        "compile_time_ns",
        "steady_state_time_ns",
        "peak_device_memory_bytes",
    }
)
_TIMING_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "compile_time_ns",
        "steady_state_run_times_ns",
        "peak_device_memory_bytes",
    }
)
_UNSET = object()
ResultT = TypeVar("ResultT")


def _nonnegative_integer(name: str, value: Any) -> int:
    """Return ``value`` as a Python int, rejecting lossy coercions."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _optional_nonnegative_integer(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(name, value)


def _positive_integer(name: str, value: Any) -> int:
    result = _nonnegative_integer(name, value)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _boolean_sequence(name: str, values: Any) -> tuple[bool, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of bool values")
    return tuple(_boolean(f"{name}[{index}]", value) for index, value in enumerate(values))


@dataclass(frozen=True)
class MAMWorkCounters:
    """Immutable, JSON-safe physical-work counters for a MAM run.

    All additive fields use unbounded Python integers.  Thus ``merge`` is
    associative and commutative without integer overflow or floating-point
    reduction-order effects.  A positive optimizer-update count requires at
    least as many optimizer examples, but otherwise the work domains remain
    independent so that reference kernels can be accounted in isolation.
    """

    running_cost_oracle_evaluations: int = 0
    simulated_transitions: int = 0
    tangent_vjps: int = 0
    tangent_jvps: int = 0
    optimizer_examples: int = 0
    optimizer_updates: int = 0
    compile_time_ns: int = 0
    steady_state_time_ns: int = 0
    peak_device_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "running_cost_oracle_evaluations",
            "simulated_transitions",
            "tangent_vjps",
            "tangent_jvps",
            "optimizer_examples",
            "optimizer_updates",
            "compile_time_ns",
            "steady_state_time_ns",
        )
        for name in integer_fields:
            object.__setattr__(self, name, _nonnegative_integer(name, getattr(self, name)))
        object.__setattr__(
            self,
            "peak_device_memory_bytes",
            _optional_nonnegative_integer(
                "peak_device_memory_bytes", self.peak_device_memory_bytes
            ),
        )
        if self.optimizer_updates > self.optimizer_examples:
            raise ValueError("optimizer_updates cannot exceed optimizer_examples")

    @classmethod
    def zero(cls) -> MAMWorkCounters:
        """Return the exact additive identity representing no measured work."""
        return cls()

    @property
    def compile_time_seconds(self) -> float:
        """Compile/lowering duration in seconds for display only."""
        return self.compile_time_ns / 1_000_000_000

    @property
    def steady_state_time_seconds(self) -> float:
        """Synchronized execution duration in seconds for display only."""
        return self.steady_state_time_ns / 1_000_000_000

    def merge(self, other: MAMWorkCounters) -> MAMWorkCounters:
        """Merge disjoint work records exactly.

        Additive fields are summed.  Known peak-memory measurements are
        combined by ``max``; if either non-identity record has an unknown peak,
        the merged peak is unknown.  The no-work all-zero identity is
        therefore neutral for both known and unknown records.
        """
        if not isinstance(other, MAMWorkCounters):
            raise TypeError("other must be a MAMWorkCounters instance")
        if self == MAMWorkCounters.zero():
            return other
        if other == MAMWorkCounters.zero():
            return self
        if self.peak_device_memory_bytes is None or other.peak_device_memory_bytes is None:
            peak: int | None = None
        else:
            peak = max(self.peak_device_memory_bytes, other.peak_device_memory_bytes)
        return MAMWorkCounters(
            running_cost_oracle_evaluations=(
                self.running_cost_oracle_evaluations + other.running_cost_oracle_evaluations
            ),
            simulated_transitions=self.simulated_transitions + other.simulated_transitions,
            tangent_vjps=self.tangent_vjps + other.tangent_vjps,
            tangent_jvps=self.tangent_jvps + other.tangent_jvps,
            optimizer_examples=self.optimizer_examples + other.optimizer_examples,
            optimizer_updates=self.optimizer_updates + other.optimizer_updates,
            compile_time_ns=self.compile_time_ns + other.compile_time_ns,
            steady_state_time_ns=self.steady_state_time_ns + other.steady_state_time_ns,
            peak_device_memory_bytes=peak,
        )

    def add(
        self,
        *,
        running_cost_oracle_evaluations: int = 0,
        simulated_transitions: int = 0,
        tangent_vjps: int = 0,
        tangent_jvps: int = 0,
        optimizer_examples: int = 0,
        optimizer_updates: int = 0,
        compile_time_ns: int = 0,
        steady_state_time_ns: int = 0,
        peak_device_memory_bytes: int | None | object = _UNSET,
    ) -> MAMWorkCounters:
        """Return a new record with checked increments and an optional peak.

        ``peak_device_memory_bytes`` is an aggregate high-water observation,
        not an increment.  A known value is therefore allowed to resolve a
        previously unknown peak.  Passing ``None`` explicitly marks the new
        aggregate as unknown.  Omitting it preserves the current peak only
        when no work is added; otherwise the newly added work is conservatively
        treated as unmeasured.
        """
        increment = MAMWorkCounters(
            running_cost_oracle_evaluations=running_cost_oracle_evaluations,
            simulated_transitions=simulated_transitions,
            tangent_vjps=tangent_vjps,
            tangent_jvps=tangent_jvps,
            optimizer_examples=optimizer_examples,
            optimizer_updates=optimizer_updates,
            compile_time_ns=compile_time_ns,
            steady_state_time_ns=steady_state_time_ns,
            # Validate additive fields independently of memory semantics.
            peak_device_memory_bytes=0,
        )
        additive_work = any(
            (
                increment.running_cost_oracle_evaluations,
                increment.simulated_transitions,
                increment.tangent_vjps,
                increment.tangent_jvps,
                increment.optimizer_examples,
                increment.optimizer_updates,
                increment.compile_time_ns,
                increment.steady_state_time_ns,
            )
        )
        if peak_device_memory_bytes is _UNSET:
            peak = None if additive_work else self.peak_device_memory_bytes
        else:
            observation = _optional_nonnegative_integer(
                "peak_device_memory_bytes", peak_device_memory_bytes
            )
            if observation is None:
                peak = None
            elif self.peak_device_memory_bytes is None:
                peak = observation
            else:
                peak = max(self.peak_device_memory_bytes, observation)
        return MAMWorkCounters(
            running_cost_oracle_evaluations=(
                self.running_cost_oracle_evaluations + increment.running_cost_oracle_evaluations
            ),
            simulated_transitions=self.simulated_transitions + increment.simulated_transitions,
            tangent_vjps=self.tangent_vjps + increment.tangent_vjps,
            tangent_jvps=self.tangent_jvps + increment.tangent_jvps,
            optimizer_examples=self.optimizer_examples + increment.optimizer_examples,
            optimizer_updates=self.optimizer_updates + increment.optimizer_updates,
            compile_time_ns=self.compile_time_ns + increment.compile_time_ns,
            steady_state_time_ns=(self.steady_state_time_ns + increment.steady_state_time_ns),
            peak_device_memory_bytes=peak,
        )

    def __add__(self, other: MAMWorkCounters) -> MAMWorkCounters:
        return self.merge(other)

    def to_state(self) -> dict[str, int | None]:
        """Return an exact JSON-serializable checkpoint state."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "running_cost_oracle_evaluations": self.running_cost_oracle_evaluations,
            "simulated_transitions": self.simulated_transitions,
            "tangent_vjps": self.tangent_vjps,
            "tangent_jvps": self.tangent_jvps,
            "optimizer_examples": self.optimizer_examples,
            "optimizer_updates": self.optimizer_updates,
            "compile_time_ns": self.compile_time_ns,
            "steady_state_time_ns": self.steady_state_time_ns,
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> MAMWorkCounters:
        """Restore a record, rejecting schema drift and lossy values."""
        if not isinstance(state, Mapping):
            raise TypeError("work-counter state must be a mapping")
        fields = frozenset(state)
        if fields != _STATE_FIELDS:
            missing = sorted(_STATE_FIELDS - fields)
            extra = sorted(fields - _STATE_FIELDS)
            raise ValueError(
                f"work-counter state schema mismatch; missing={missing}, extra={extra}"
            )
        version = _nonnegative_integer("schema_version", state["schema_version"])
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported work-counter schema version {version}; expected {_SCHEMA_VERSION}"
            )
        return cls(
            running_cost_oracle_evaluations=state["running_cost_oracle_evaluations"],
            simulated_transitions=state["simulated_transitions"],
            tangent_vjps=state["tangent_vjps"],
            tangent_jvps=state["tangent_jvps"],
            optimizer_examples=state["optimizer_examples"],
            optimizer_updates=state["optimizer_updates"],
            compile_time_ns=state["compile_time_ns"],
            steady_state_time_ns=state["steady_state_time_ns"],
            peak_device_memory_bytes=state["peak_device_memory_bytes"],
        )


def merge_work_counters(records: Sequence[MAMWorkCounters]) -> MAMWorkCounters:
    """Return an exact deterministic reduction of disjoint work records."""
    aggregate = MAMWorkCounters.zero()
    for record in records:
        aggregate = aggregate.merge(record)
    return aggregate


def conditional_policy_iteration_work(
    *,
    num_steps: int,
    effective_batch_size: int,
    costate_steps: int,
    value_critic_training_steps: int,
    actor_field_training_steps: int,
    direct_score_diagnostic_size: int,
    acceptance_size: int,
    line_search_candidates: int,
    actor_confirmation_executed: bool,
    actor_update_accepted: bool,
    running_cost_oracle_present: bool,
) -> MAMWorkCounters:
    """Account one successful built-in conditional policy iteration.

    ``num_steps`` is the number ``N`` of global grid intervals.  The pinned
    conditional chain has ``S=N-1`` stochastic transitions and one exact
    deterministic pin.  Transition accounting includes every next-state
    proposal actually produced: base rollouts, both antithetic arrival
    branches, and every vectorized suffix proposal evaluated before masking.

    ``actor_field_training_steps`` is zero for the affine reference and the
    number of Adam updates for the nonlinear actor.  Confirmation work depends
    on whether confirmation executed, not whether it ultimately accepted the
    update.  An accepted update without confirmation is invalid.

    Oracle counts are scalar value-query equivalents.  They include complete
    ``[B,N+1]`` vectorized requests even when endpoints or pre-anchor values
    are masked later.  The direct diagnostic count includes both its
    decomposed antithetic arrival query and its two suffix returns.  Actor
    confirmation includes the third objective evaluation used for critic
    calibration.
    """
    steps = _positive_integer("num_steps", num_steps)
    if steps < 3:
        raise ValueError("num_steps must be at least three for MAM bridge V1")
    effective = _positive_integer("effective_batch_size", effective_batch_size)
    costate = _positive_integer("costate_steps", costate_steps)
    critic_steps = _positive_integer("value_critic_training_steps", value_critic_training_steps)
    actor_steps = _nonnegative_integer("actor_field_training_steps", actor_field_training_steps)
    diagnostic_size = _positive_integer(
        "direct_score_diagnostic_size", direct_score_diagnostic_size
    )
    if diagnostic_size < 2:
        raise ValueError("direct_score_diagnostic_size must be at least two")
    acceptance = _positive_integer("acceptance_size", acceptance_size)
    if acceptance < 2:
        raise ValueError("acceptance_size must be at least two")
    candidates = _positive_integer("line_search_candidates", line_search_candidates)
    confirmation = _boolean("actor_confirmation_executed", actor_confirmation_executed)
    accepted = _boolean("actor_update_accepted", actor_update_accepted)
    oracle_present = _boolean("running_cost_oracle_present", running_cost_oracle_present)
    if accepted and not confirmation:
        raise ValueError("an actor update cannot be accepted without confirmation")

    stochastic_steps = steps - 1
    confirmation_count = int(confirmation)
    simulated_transitions = (
        (costate + 4) * effective * stochastic_steps
        + diagnostic_size * (3 * stochastic_steps + 2)
        + (candidates + 1 + 2 * confirmation_count) * acceptance * stochastic_steps
    )
    if oracle_present:
        oracle_evaluations = (
            (costate + 1) * effective * (steps + 1)
            + 2 * effective * stochastic_steps
            + 2 * diagnostic_size * (stochastic_steps + 2)
            + (candidates + 1 + 3 * confirmation_count) * acceptance * (steps + 1)
        )
    else:
        oracle_evaluations = 0
    optimizer_updates = costate + 2 * critic_steps + actor_steps
    return MAMWorkCounters(
        running_cost_oracle_evaluations=oracle_evaluations,
        simulated_transitions=simulated_transitions,
        tangent_vjps=costate * effective * stochastic_steps,
        tangent_jvps=0,
        optimizer_examples=effective * optimizer_updates,
        optimizer_updates=optimizer_updates,
        peak_device_memory_bytes=None,
    )


def conditional_costate_refresh_work(
    *,
    num_steps: int,
    effective_batch_size: int,
    costate_steps: int,
    running_cost_oracle_present: bool,
) -> MAMWorkCounters:
    """Account a disjoint final costate fit under an accepted output actor.

    A policy iteration fits its costate before proposing an actor update.  If
    the last proposal is accepted, one additional costate fit is required so
    that the stored/returned costate is conditioned on the accepted actor.
    This phase contains only the matrix-free costate-label rollouts, VJPs, and
    Adam updates; it does not refit the critic or actor and does not perform an
    acceptance evaluation.
    """
    steps = _positive_integer("num_steps", num_steps)
    if steps < 3:
        raise ValueError("num_steps must be at least three for MAM bridge V1")
    effective = _positive_integer("effective_batch_size", effective_batch_size)
    costate = _positive_integer("costate_steps", costate_steps)
    oracle_present = _boolean("running_cost_oracle_present", running_cost_oracle_present)
    stochastic_steps = steps - 1
    return MAMWorkCounters(
        running_cost_oracle_evaluations=(
            costate * effective * (steps + 1) if oracle_present else 0
        ),
        simulated_transitions=costate * effective * stochastic_steps,
        tangent_vjps=costate * effective * stochastic_steps,
        tangent_jvps=0,
        optimizer_examples=costate * effective,
        optimizer_updates=costate,
        peak_device_memory_bytes=None,
    )


def completed_conditional_solve_work(
    *,
    num_steps: int,
    effective_batch_size: int,
    costate_steps: int,
    value_critic_training_steps: int,
    actor_field_training_steps: int,
    direct_score_diagnostic_size: int,
    acceptance_size: int,
    line_search_candidates: int,
    pair_batch_size: int,
    policy_iterations_completed: int,
    actor_confirmation_executed: Sequence[bool],
    actor_update_accepted: Sequence[bool],
    final_costate_refresh_executed: bool,
    running_cost_oracle_present: bool,
) -> MAMWorkCounters:
    """Account a completed built-in conditional solve.

    One policy-iteration record is created for every actual iteration, which
    may be fewer than configured after consecutive rejections.  The final
    output pinned rollout over all endpoint pairs is then added once.  The two
    decision sequences must exactly match ``policy_iterations_completed``.
    A final costate refresh may occur only after the last actor proposal was
    accepted.  The caller compares actor fingerprints to decide whether an
    accepted proposal actually changed the policy and therefore needs it.
    """
    steps = _positive_integer("num_steps", num_steps)
    if steps < 3:
        raise ValueError("num_steps must be at least three for MAM bridge V1")
    pair_count = _positive_integer("pair_batch_size", pair_batch_size)
    iterations = _positive_integer("policy_iterations_completed", policy_iterations_completed)
    confirmations = _boolean_sequence("actor_confirmation_executed", actor_confirmation_executed)
    acceptances = _boolean_sequence("actor_update_accepted", actor_update_accepted)
    if len(confirmations) != iterations or len(acceptances) != iterations:
        raise ValueError("actor decision sequence lengths must match policy_iterations_completed")
    refresh = _boolean("final_costate_refresh_executed", final_costate_refresh_executed)
    if refresh and not acceptances[-1]:
        raise ValueError("final costate refresh requires an accepted last actor update")
    oracle_present = _boolean("running_cost_oracle_present", running_cost_oracle_present)

    iterations_work = [
        conditional_policy_iteration_work(
            num_steps=steps,
            effective_batch_size=effective_batch_size,
            costate_steps=costate_steps,
            value_critic_training_steps=value_critic_training_steps,
            actor_field_training_steps=actor_field_training_steps,
            direct_score_diagnostic_size=direct_score_diagnostic_size,
            acceptance_size=acceptance_size,
            line_search_candidates=line_search_candidates,
            actor_confirmation_executed=confirmation,
            actor_update_accepted=accepted,
            running_cost_oracle_present=oracle_present,
        )
        for confirmation, accepted in zip(confirmations, acceptances, strict=True)
    ]
    aggregate = merge_work_counters(iterations_work)
    if refresh:
        aggregate = aggregate.merge(
            conditional_costate_refresh_work(
                num_steps=steps,
                effective_batch_size=effective_batch_size,
                costate_steps=costate_steps,
                running_cost_oracle_present=oracle_present,
            )
        )
    return aggregate.add(
        simulated_transitions=pair_count * (steps - 1),
        peak_device_memory_bytes=None,
    )


def global_half_iteration_work(
    conditional_work: MAMWorkCounters,
    *,
    num_steps: int,
    effective_batch_size: int,
    pair_batch_size: int,
    projection_field_training_steps: int,
    projection_validation_size: int,
    projection_line_search_candidates: int,
    projection_validation_replicates: int,
    globally_feasible_before_projection: bool,
    projection_confirmation_executed: bool,
    projection_update_accepted: bool,
    audit_size: int,
    running_cost_oracle_present: bool,
) -> MAMWorkCounters:
    """Account one successful global half-iteration including conditional work.

    Endpoint-score validation simulates one path cloud for each independent
    cloud replicate, current/candidate, and optional confirmation evaluation.
    Slicing directions within a cloud do not create additional transition or
    statistical samples.  Once the previous global audit is feasible, one
    additional generalized-objective cloud is evaluated per current/candidate
    and optional confirmation evaluation.  The coupling refresh and both
    directional audit rollouts execute regardless of the acceptance decision.
    """
    if not isinstance(conditional_work, MAMWorkCounters):
        raise TypeError("conditional_work must be a MAMWorkCounters instance")
    steps = _positive_integer("num_steps", num_steps)
    if steps < 3:
        raise ValueError("num_steps must be at least three for MAM bridge V1")
    effective = _positive_integer("effective_batch_size", effective_batch_size)
    pair_count = _positive_integer("pair_batch_size", pair_batch_size)
    projector_steps = _nonnegative_integer(
        "projection_field_training_steps", projection_field_training_steps
    )
    validation = _positive_integer("projection_validation_size", projection_validation_size)
    if validation < 2:
        raise ValueError("projection_validation_size must be at least two")
    candidates = _positive_integer(
        "projection_line_search_candidates", projection_line_search_candidates
    )
    replicates = _positive_integer(
        "projection_validation_replicates", projection_validation_replicates
    )
    if replicates < 2:
        raise ValueError("projection_validation_replicates must be at least two")
    feasible = _boolean("globally_feasible_before_projection", globally_feasible_before_projection)
    confirmation = _boolean("projection_confirmation_executed", projection_confirmation_executed)
    accepted = _boolean("projection_update_accepted", projection_update_accepted)
    audit = _positive_integer("audit_size", audit_size)
    if audit < 2:
        raise ValueError("audit_size must be at least two")
    oracle_present = _boolean("running_cost_oracle_present", running_cost_oracle_present)
    if accepted and not confirmation:
        raise ValueError("a projection update cannot be accepted without confirmation")
    if oracle_present != (conditional_work.running_cost_oracle_evaluations > 0):
        raise ValueError(
            "conditional_work oracle count is inconsistent with running_cost_oracle_present"
        )

    confirmation_count = int(confirmation)
    validation_rollouts = candidates + 1 + 2 * confirmation_count
    projection_transitions = (replicates + int(feasible)) * validation_rollouts * validation * steps
    refresh_transitions = pair_count * steps
    audit_transitions = 2 * audit * steps
    projection_oracle_evaluations = (
        int(oracle_present) * int(feasible) * validation_rollouts * validation * (steps + 1)
    )
    projection_work = MAMWorkCounters(
        running_cost_oracle_evaluations=projection_oracle_evaluations,
        simulated_transitions=(projection_transitions + refresh_transitions + audit_transitions),
        tangent_vjps=0,
        tangent_jvps=0,
        optimizer_examples=effective * projector_steps,
        optimizer_updates=projector_steps,
        peak_device_memory_bytes=None,
    )
    return conditional_work.merge(projection_work)


@dataclass(frozen=True)
class SynchronizedJaxTiming:
    """Timing metadata from one compile and synchronized steady executions."""

    compile_time_ns: int
    steady_state_run_times_ns: tuple[int, ...]
    peak_device_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compile_time_ns",
            _nonnegative_integer("compile_time_ns", self.compile_time_ns),
        )
        run_times = tuple(
            _nonnegative_integer(f"steady_state_run_times_ns[{index}]", value)
            for index, value in enumerate(self.steady_state_run_times_ns)
        )
        if not run_times:
            raise ValueError("at least one steady-state timing is required")
        object.__setattr__(self, "steady_state_run_times_ns", run_times)
        object.__setattr__(
            self,
            "peak_device_memory_bytes",
            _optional_nonnegative_integer(
                "peak_device_memory_bytes", self.peak_device_memory_bytes
            ),
        )

    @property
    def steady_state_runs(self) -> int:
        return len(self.steady_state_run_times_ns)

    @property
    def steady_state_time_ns(self) -> int:
        return sum(self.steady_state_run_times_ns)

    def to_work_counters(self) -> MAMWorkCounters:
        """Convert the timing record to a mergeable work record."""
        return MAMWorkCounters(
            compile_time_ns=self.compile_time_ns,
            steady_state_time_ns=self.steady_state_time_ns,
            peak_device_memory_bytes=self.peak_device_memory_bytes,
        )

    def to_state(self) -> dict[str, int | list[int] | None]:
        """Return JSON-safe per-run timing metadata."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "compile_time_ns": self.compile_time_ns,
            "steady_state_run_times_ns": list(self.steady_state_run_times_ns),
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> SynchronizedJaxTiming:
        """Restore timing metadata, rejecting schema or unit drift."""
        if not isinstance(state, Mapping):
            raise TypeError("synchronized-timing state must be a mapping")
        fields = frozenset(state)
        if fields != _TIMING_STATE_FIELDS:
            missing = sorted(_TIMING_STATE_FIELDS - fields)
            extra = sorted(fields - _TIMING_STATE_FIELDS)
            raise ValueError(
                f"synchronized-timing state schema mismatch; missing={missing}, extra={extra}"
            )
        version = _nonnegative_integer("schema_version", state["schema_version"])
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported synchronized-timing schema version {version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        run_times = state["steady_state_run_times_ns"]
        if isinstance(run_times, (str, bytes)) or not isinstance(run_times, Sequence):
            raise TypeError("steady_state_run_times_ns must be a sequence of integers")
        return cls(
            compile_time_ns=state["compile_time_ns"],
            steady_state_run_times_ns=tuple(run_times),
            peak_device_memory_bytes=state["peak_device_memory_bytes"],
        )


def synchronize_jax_result(result: ResultT) -> ResultT:
    """Synchronize every JAX leaf and return the original pytree."""
    return cast(ResultT, jax.block_until_ready(result))


def _elapsed_ns(start: Any, stop: Any, *, phase: str) -> int:
    start_ns = _nonnegative_integer(f"{phase} start time", start)
    stop_ns = _nonnegative_integer(f"{phase} stop time", stop)
    if stop_ns < start_ns:
        raise RuntimeError(f"the monotonic clock moved backwards during {phase}")
    return stop_ns - start_ns


def time_jax_callable(
    function: Callable[..., ResultT],
    *args: Any,
    steady_state_runs: int = 1,
    call_kwargs: Mapping[str, Any] | None = None,
    jit_options: Mapping[str, Any] | None = None,
    peak_memory_probe: Callable[[], int | None] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[ResultT, SynchronizedJaxTiming]:
    """Compile and synchronously time a pure JAX callable.

    Tracing, lowering, and executable compilation are included only in
    ``compile_time_ns``.  Each subsequent execution is individually delimited
    by the host clock and completed with :func:`jax.block_until_ready` before
    its stop timestamp.  Input transfers are synchronized before either phase
    and therefore excluded.

    ``peak_memory_probe`` is optional because portable JAX peak-memory
    telemetry is unavailable on some backends.  Omitting it records the peak
    explicitly as unknown instead of guessing from array sizes.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    runs = _nonnegative_integer("steady_state_runs", steady_state_runs)
    if runs == 0:
        raise ValueError("steady_state_runs must be positive")
    if not callable(clock_ns):
        raise TypeError("clock_ns must be callable")
    if peak_memory_probe is not None and not callable(peak_memory_probe):
        raise TypeError("peak_memory_probe must be callable or None")
    kwargs = {} if call_kwargs is None else dict(call_kwargs)
    options = {} if jit_options is None else dict(jit_options)

    # Exclude outstanding producer work and host-to-device transfers from both
    # compile and steady-state measurements.
    synchronize_jax_result((args, kwargs))

    compile_start = clock_ns()
    executable = jax.jit(function, **options).lower(*args, **kwargs).compile()
    compile_stop = clock_ns()
    compile_time = _elapsed_ns(compile_start, compile_stop, phase="compilation")

    run_times: list[int] = []
    result: Any = _UNSET
    for run_index in range(runs):
        run_start = clock_ns()
        result = executable(*args, **kwargs)
        result = synchronize_jax_result(result)
        run_stop = clock_ns()
        run_times.append(_elapsed_ns(run_start, run_stop, phase=f"steady-state run {run_index}"))

    peak = None if peak_memory_probe is None else peak_memory_probe()
    timing = SynchronizedJaxTiming(
        compile_time_ns=compile_time,
        steady_state_run_times_ns=tuple(run_times),
        peak_device_memory_bytes=peak,
    )
    # ``runs`` is positive, so this is established by construction even when
    # the compiled function legitimately returns ``None``.
    assert result is not _UNSET
    return cast(ResultT, result), timing


__all__ = [
    "MAMWorkCounters",
    "SynchronizedJaxTiming",
    "completed_conditional_solve_work",
    "conditional_costate_refresh_work",
    "conditional_policy_iteration_work",
    "global_half_iteration_work",
    "merge_work_counters",
    "synchronize_jax_result",
    "time_jax_callable",
]
