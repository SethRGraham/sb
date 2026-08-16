"""Single-host execution primitives for the experimental MAM bridge solver.

The helpers in this module make the one-device execution contract explicit:

* an effective batch is a fixed number of equal-sized microbatches;
* if two local devices are selected, only the microbatch axis is sharded;
* gradient accumulation is a pure JAX pytree reduction with an explicit
  finiteness result; and
* random streams are separated by stable, serializable domains.

This is execution infrastructure, not evidence that the MAM scalability gate
has passed.  In particular, discovering a GPU or constructing a valid plan
does not measure peak memory, throughput, or checkpoint parity.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..core.types import Array, PRNGKey

PyTree = Any


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_uint32(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0 or value > np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must lie in [0, 2**32 - 1]")
    return value


@dataclass(frozen=True)
class ExecutionPlan:
    """Static batch layout for one optimizer update.

    ``microbatch_size`` and ``effective_batch_size`` are global sizes.  With
    two devices, each microbatch is reshaped to
    ``[2, per_device_microbatch_size, ...]`` and parameters are expected to be
    replicated.  The public checkpoint format therefore need not depend on
    the number of devices.
    """

    horizon: int
    state_dim: int
    microbatch_size: int
    effective_batch_size: int
    accumulation_steps: int
    device_count: int
    per_device_microbatch_size: int
    production_dtype: str = "float32"
    matrix_free: bool = True

    def __post_init__(self) -> None:
        horizon = _positive_int("horizon", self.horizon)
        state_dim = _positive_int("state_dim", self.state_dim)
        microbatch = _positive_int("microbatch_size", self.microbatch_size)
        effective = _positive_int("effective_batch_size", self.effective_batch_size)
        accumulation = _positive_int("accumulation_steps", self.accumulation_steps)
        device_count = _positive_int("device_count", self.device_count)
        per_device = _positive_int("per_device_microbatch_size", self.per_device_microbatch_size)
        if device_count > 2:
            raise ValueError("MAM V1 supports at most two local data-parallel devices")
        if effective % microbatch != 0:
            raise ValueError("effective_batch_size must be divisible by microbatch_size")
        if accumulation != effective // microbatch:
            raise ValueError("accumulation_steps is inconsistent with the batch sizes")
        if microbatch % device_count != 0:
            raise ValueError("microbatch_size must be divisible by device_count")
        if per_device != microbatch // device_count:
            raise ValueError("per_device_microbatch_size is inconsistent with device_count")
        if self.production_dtype != "float32":
            raise ValueError("the MAM V1 production execution plan is fixed to float32")
        if not self.matrix_free:
            raise ValueError("production MAM execution must be matrix-free")
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "state_dim", state_dim)
        object.__setattr__(self, "microbatch_size", microbatch)
        object.__setattr__(self, "effective_batch_size", effective)
        object.__setattr__(self, "accumulation_steps", accumulation)
        object.__setattr__(self, "device_count", device_count)
        object.__setattr__(self, "per_device_microbatch_size", per_device)

    def to_state(self) -> dict[str, Any]:
        """Return JSON-serializable checkpoint metadata."""
        return {
            "horizon": self.horizon,
            "state_dim": self.state_dim,
            "microbatch_size": self.microbatch_size,
            "effective_batch_size": self.effective_batch_size,
            "accumulation_steps": self.accumulation_steps,
            "device_count": self.device_count,
            "per_device_microbatch_size": self.per_device_microbatch_size,
            "production_dtype": self.production_dtype,
            "matrix_free": self.matrix_free,
        }


def default_microbatch_size(*, horizon: int, state_dim: int) -> int:
    """Return the conservative V1 default inside the declared test envelope.

    The larger default is used through dimension 64.  Dimensions 65--128 use
    the conservative dimension-128 setting.  Outside ``N <= 128, d <= 128`` a
    caller must provide a measured, explicit microbatch size.
    """
    horizon = _positive_int("horizon", horizon)
    state_dim = _positive_int("state_dim", state_dim)
    if horizon > 128 or state_dim > 128:
        raise ValueError(
            "no unmeasured default exists outside N <= 128 and d <= 128; "
            "provide microbatch_size explicitly"
        )
    return 128 if state_dim <= 64 else 32


def make_execution_plan(
    *,
    horizon: int,
    state_dim: int,
    microbatch_size: int | None = None,
    effective_batch_size: int = 1024,
    device_count: int = 1,
) -> ExecutionPlan:
    """Validate and construct a static one- or two-device batch plan."""
    horizon = _positive_int("horizon", horizon)
    state_dim = _positive_int("state_dim", state_dim)
    effective_batch_size = _positive_int("effective_batch_size", effective_batch_size)
    device_count = _positive_int("device_count", device_count)
    if device_count > 2:
        raise ValueError("MAM V1 supports at most two local data-parallel devices")
    if microbatch_size is None:
        microbatch_size = default_microbatch_size(horizon=horizon, state_dim=state_dim)
    microbatch_size = _positive_int("microbatch_size", microbatch_size)
    if effective_batch_size % microbatch_size != 0:
        raise ValueError("effective_batch_size must be divisible by microbatch_size")
    if microbatch_size % device_count != 0:
        raise ValueError("microbatch_size must be divisible by device_count")
    return ExecutionPlan(
        horizon=horizon,
        state_dim=state_dim,
        microbatch_size=microbatch_size,
        effective_batch_size=effective_batch_size,
        accumulation_steps=effective_batch_size // microbatch_size,
        device_count=device_count,
        per_device_microbatch_size=microbatch_size // device_count,
    )


@dataclass(frozen=True)
class DeviceTopology:
    """Serializable metadata for the selected local JAX devices."""

    platform: str
    device_kind: str
    available_local_device_count: int
    selected_device_count: int
    selected_device_ids: tuple[int, ...]
    process_count: int
    process_index: int
    batch_data_parallel: bool

    def __post_init__(self) -> None:
        available = _positive_int("available_local_device_count", self.available_local_device_count)
        selected = _positive_int("selected_device_count", self.selected_device_count)
        if selected > 2:
            raise ValueError("MAM V1 supports at most two selected devices")
        if selected > available:
            raise ValueError("selected_device_count exceeds available local devices")
        if len(self.selected_device_ids) != selected:
            raise ValueError("selected_device_ids does not match selected_device_count")
        if len(set(self.selected_device_ids)) != selected:
            raise ValueError("selected_device_ids must be unique")
        if self.batch_data_parallel != (selected > 1):
            raise ValueError("batch_data_parallel must be true exactly when two devices are used")
        if not self.platform or not self.device_kind:
            raise ValueError("platform and device_kind must be nonempty")
        _positive_int("process_count", self.process_count)
        if self.process_index < 0 or self.process_index >= self.process_count:
            raise ValueError("process_index is outside [0, process_count)")

    def to_state(self) -> dict[str, Any]:
        """Return JSON-serializable device metadata."""
        return {
            "platform": self.platform,
            "device_kind": self.device_kind,
            "available_local_device_count": self.available_local_device_count,
            "selected_device_count": self.selected_device_count,
            "selected_device_ids": list(self.selected_device_ids),
            "process_count": self.process_count,
            "process_index": self.process_index,
            "batch_data_parallel": self.batch_data_parallel,
        }


def discover_device_topology(*, max_devices: int = 1) -> DeviceTopology:
    """Select at most ``max_devices`` homogeneous local devices.

    The default deliberately selects one device even if more are available.
    Passing ``max_devices=2`` opts into local batch data parallelism when two
    compatible devices exist; it does not require two devices to be present.
    """
    max_devices = _positive_int("max_devices", max_devices)
    if max_devices > 2:
        raise ValueError("max_devices must be one or two")
    devices = tuple(jax.local_devices())
    if not devices:
        raise RuntimeError("JAX reported no local devices")
    selected = devices[: min(max_devices, len(devices))]
    platforms = {device.platform for device in selected}
    kinds = {str(device.device_kind) for device in selected}
    if len(platforms) != 1 or len(kinds) != 1:
        raise RuntimeError("selected devices must have a homogeneous platform and device kind")
    return DeviceTopology(
        platform=selected[0].platform,
        device_kind=str(selected[0].device_kind),
        available_local_device_count=len(devices),
        selected_device_count=len(selected),
        selected_device_ids=tuple(int(device.id) for device in selected),
        process_count=int(jax.process_count()),
        process_index=int(jax.process_index()),
        batch_data_parallel=len(selected) > 1,
    )


def resolve_local_devices(topology: DeviceTopology) -> tuple[Any, ...]:
    """Resolve recorded device identities, failing on topology drift."""
    current = tuple(jax.local_devices())
    by_id = {int(device.id): device for device in current}
    try:
        selected = tuple(by_id[device_id] for device_id in topology.selected_device_ids)
    except KeyError as exc:
        raise RuntimeError("a recorded MAM device is unavailable") from exc
    for device in selected:
        if device.platform != topology.platform or str(device.device_kind) != topology.device_kind:
            raise RuntimeError("the current JAX device metadata differs from the checkpoint")
    return selected


def static_microbatch_indices(
    key: PRNGKey,
    *,
    num_examples: int,
    plan: ExecutionPlan,
    shuffle: bool = True,
) -> Array:
    """Return indices with shape ``[updates, accumulation, microbatch]``.

    ``num_examples`` must be an exact multiple of the effective batch.  This
    prevents examples from being silently dropped or repeated.  The returned
    schedule is deterministic for a fixed key and is compatible with ``jit``
    when the integer arguments and plan are closed-over static values.
    """
    num_examples = _positive_int("num_examples", num_examples)
    if num_examples % plan.effective_batch_size != 0:
        raise ValueError("num_examples must be divisible by effective_batch_size")
    indices = jnp.arange(num_examples, dtype=jnp.int32)
    if shuffle:
        indices = jax.random.permutation(key, indices)
    return indices.reshape(
        num_examples // plan.effective_batch_size,
        plan.accumulation_steps,
        plan.microbatch_size,
    )


def shard_microbatch_axis(tree: PyTree, plan: ExecutionPlan) -> PyTree:
    """Reshape global ``[microbatch, ...]`` leaves to ``[devices, local, ...]``."""
    leaves, structure = jax.tree_util.tree_flatten(tree)
    if not leaves:
        raise ValueError("microbatch pytree must have at least one leaf")
    reshaped = []
    for leaf in leaves:
        value = jnp.asarray(leaf)
        if value.ndim < 1 or value.shape[0] != plan.microbatch_size:
            raise ValueError("every microbatch leaf must have the configured leading size")
        reshaped.append(
            value.reshape(plan.device_count, plan.per_device_microbatch_size, *value.shape[1:])
        )
    return jax.tree_util.tree_unflatten(structure, reshaped)


def take_scheduled_batches(tree: PyTree, indices: Array) -> PyTree:
    """Gather a host/device cache using a static microbatch schedule."""
    indices = jnp.asarray(indices)
    if indices.ndim != 3 or not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError("indices must have shape [updates, accumulation, microbatch]")
    leaves, structure = jax.tree_util.tree_flatten(tree)
    if not leaves:
        raise ValueError("cache pytree must have at least one leaf")
    leading_sizes = set()
    gathered = []
    for leaf in leaves:
        value = jnp.asarray(leaf)
        if value.ndim < 1:
            raise ValueError("every cache leaf must have a leading example axis")
        leading_sizes.add(value.shape[0])
        gathered.append(jnp.take(value, indices, axis=0))
    if len(leading_sizes) != 1:
        raise ValueError("all cache leaves must have the same leading size")
    return jax.tree_util.tree_unflatten(structure, gathered)


class GradientAccumulator(NamedTuple):
    """JAX-pytree state for equal-size microbatch gradient accumulation."""

    total: PyTree
    count: Array  # type: ignore[assignment]  # NamedTuple intentionally exposes `.count`.
    finite: Array


class GradientAccumulationResult(NamedTuple):
    """Averaged gradient tree and validity diagnostics."""

    gradients: PyTree
    finite: Array
    count_matches: Array
    count: Array  # type: ignore[assignment]  # NamedTuple intentionally exposes `.count`.


def _validate_gradient_tree(tree: PyTree, *, name: str) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError(f"{name} must have at least one leaf")
    for leaf in leaves:
        dtype = jnp.asarray(leaf).dtype
        if not jnp.issubdtype(dtype, jnp.inexact):
            raise TypeError(f"all {name} leaves must have an inexact dtype")


def gradient_tree_is_finite(tree: PyTree) -> Array:
    """Return a scalar JAX boolean without synchronizing to the host."""
    _validate_gradient_tree(tree, name="gradient tree")
    leaves = jax.tree_util.tree_leaves(tree)
    flags = [jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves]
    return jnp.all(jnp.stack(flags))


def initialize_gradient_accumulator(example_gradients: PyTree) -> GradientAccumulator:
    """Create zero state with the same pytree structure as one gradient."""
    _validate_gradient_tree(example_gradients, name="example gradient")
    total = jax.tree_util.tree_map(jnp.zeros_like, example_gradients)
    return GradientAccumulator(
        total=total,
        count=jnp.asarray(0, dtype=jnp.int32),
        finite=jnp.asarray(True),
    )


def accumulate_gradient_step(
    accumulator: GradientAccumulator,
    gradients: PyTree,
) -> GradientAccumulator:
    """Add one equal-size microbatch gradient using a pure pytree operation."""
    if jax.tree_util.tree_structure(accumulator.total) != jax.tree_util.tree_structure(gradients):
        raise ValueError("gradient pytree structure changed during accumulation")
    _validate_gradient_tree(gradients, name="gradient")
    total = jax.tree_util.tree_map(jnp.add, accumulator.total, gradients)
    return GradientAccumulator(
        total=total,
        count=accumulator.count + jnp.asarray(1, dtype=jnp.int32),
        finite=accumulator.finite & gradient_tree_is_finite(gradients),
    )


def finalize_gradient_accumulator(
    accumulator: GradientAccumulator,
    *,
    expected_steps: int,
) -> GradientAccumulationResult:
    """Average a gradient accumulation and expose fail-closed validity flags.

    Invalid averages are replaced by NaNs.  A JIT caller must gate the update
    on both flags.  A host caller can use :func:`require_valid_gradients` to
    turn either flag into an exception.
    """
    expected_steps = _positive_int("expected_steps", expected_steps)
    count_matches = accumulator.count == jnp.asarray(expected_steps, dtype=jnp.int32)
    denominator = jnp.asarray(expected_steps, dtype=jnp.float32)

    def average(value: Array) -> Array:
        value = jnp.asarray(value)
        return value / denominator.astype(value.dtype)

    averages = jax.tree_util.tree_map(average, accumulator.total)
    aggregate_finite = gradient_tree_is_finite(accumulator.total) & gradient_tree_is_finite(
        averages
    )
    finite = accumulator.finite & aggregate_finite
    valid = finite & count_matches
    gradients = jax.tree_util.tree_map(
        lambda value: jnp.where(valid, value, jnp.full_like(value, jnp.nan)),
        averages,
    )
    return GradientAccumulationResult(
        gradients=gradients,
        finite=finite,
        count_matches=count_matches,
        count=accumulator.count,
    )


def accumulate_gradient_sequence(
    gradient_sequence: PyTree,
) -> GradientAccumulationResult:
    """Average a ``[accumulation_steps, ...]`` gradient pytree with ``scan``."""
    _validate_gradient_tree(gradient_sequence, name="gradient sequence")
    leaves = jax.tree_util.tree_leaves(gradient_sequence)
    if any(jnp.asarray(leaf).ndim < 1 for leaf in leaves):
        raise ValueError("gradient-sequence leaves must have a leading step axis")
    step_counts = {jnp.asarray(leaf).shape[0] for leaf in leaves}
    if len(step_counts) != 1:
        raise ValueError("gradient-sequence leaves must share a leading step count")
    steps = step_counts.pop()
    _positive_int("gradient step count", steps)
    example = jax.tree_util.tree_map(lambda value: value[0], gradient_sequence)
    initial = initialize_gradient_accumulator(example)

    def scan_step(
        accumulator: GradientAccumulator, gradients: PyTree
    ) -> tuple[GradientAccumulator, None]:
        return accumulate_gradient_step(accumulator, gradients), None

    accumulated, _ = jax.lax.scan(scan_step, initial, gradient_sequence)
    return finalize_gradient_accumulator(accumulated, expected_steps=steps)


def require_valid_gradients(result: GradientAccumulationResult) -> PyTree:
    """Synchronize validity flags and raise before applying an invalid update."""
    finite = bool(np.asarray(jax.device_get(result.finite)))
    count_matches = bool(np.asarray(jax.device_get(result.count_matches)))
    if not finite:
        raise FloatingPointError("a nonfinite microbatch gradient was accumulated")
    if not count_matches:
        raise RuntimeError("the gradient accumulation step count is incomplete")
    return result.gradients


def mean_gradient_replicas(gradient_replicas: PyTree, *, device_count: int) -> PyTree:
    """Reference average of a leading replica axis outside ``pmap``."""
    device_count = _positive_int("device_count", device_count)
    if device_count > 2:
        raise ValueError("MAM V1 supports at most two gradient replicas")
    _validate_gradient_tree(gradient_replicas, name="gradient replicas")

    def average(value: Array) -> Array:
        value = jnp.asarray(value)
        if value.ndim < 1 or value.shape[0] != device_count:
            raise ValueError("every gradient leaf must have the declared replica axis")
        return jnp.mean(value, axis=0)

    return jax.tree_util.tree_map(average, gradient_replicas)


def pmean_gradient_tree(gradients: PyTree, *, axis_name: str = "mam_batch") -> PyTree:
    """Average replicated gradients inside a caller-owned ``pmap``."""
    return jax.tree_util.tree_map(lambda value: jax.lax.pmean(value, axis_name), gradients)


class RNGDomain(IntEnum):
    """Stable, disjoint random-stream domains for MAM training and evaluation."""

    PAIR_CACHE = 101
    COSTATE_FIT = 211
    ACTOR_TARGET_FIT = 307
    LINE_SEARCH = 401
    CONFIRMATION = 503
    REPORTING = 601
    PROJECTION_FIT = 701
    PROJECTION_EVALUATION = 809
    COUPLING_REFRESH = 907


_RNG_DOMAINS = tuple(RNGDomain)
_RNG_DOMAIN_BY_NAME = {domain.name.lower(): domain for domain in _RNG_DOMAINS}
_RNG_SCHEMA_VERSION = 1


def _coerce_rng_domain(domain: RNGDomain | str) -> RNGDomain:
    if isinstance(domain, RNGDomain):
        return domain
    if isinstance(domain, str) and domain in _RNG_DOMAIN_BY_NAME:
        return _RNG_DOMAIN_BY_NAME[domain]
    valid = ", ".join(_RNG_DOMAIN_BY_NAME)
    raise ValueError(f"unknown RNG domain {domain!r}; expected one of: {valid}")


@dataclass(frozen=True)
class RNGLedger:
    """Immutable domain-local RNG counters with strict JSON serialization.

    Domain-local counters make one stream independent of how many keys were
    consumed from another stream.  ``coordinates`` can encode outer
    iteration or direction, but must be stable nonnegative uint32 integers.
    """

    root_seed: int
    counters: tuple[int, ...] = ()
    schema_version: int = _RNG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        seed = _nonnegative_uint32("root_seed", self.root_seed)
        if self.schema_version != _RNG_SCHEMA_VERSION:
            raise ValueError(f"unsupported RNG-ledger schema {self.schema_version}")
        counters = self.counters or (0,) * len(_RNG_DOMAINS)
        if len(counters) != len(_RNG_DOMAINS):
            raise ValueError("RNG ledger must contain one counter per domain")
        counters = tuple(
            _nonnegative_uint32(f"counter[{domain.name.lower()}]", value)
            for domain, value in zip(_RNG_DOMAINS, counters, strict=True)
        )
        object.__setattr__(self, "root_seed", seed)
        object.__setattr__(self, "counters", counters)

    def _domain_index(self, domain: RNGDomain | str) -> tuple[RNGDomain, int]:
        resolved = _coerce_rng_domain(domain)
        return resolved, _RNG_DOMAINS.index(resolved)

    def key_for(
        self,
        domain: RNGDomain | str,
        index: int,
        *coordinates: int,
    ) -> PRNGKey:
        """Derive a key without advancing the ledger."""
        resolved, _ = self._domain_index(domain)
        index = _nonnegative_uint32("RNG index", index)
        coordinates = tuple(
            _nonnegative_uint32(f"RNG coordinate {i}", coordinate)
            for i, coordinate in enumerate(coordinates)
        )
        key = jax.random.PRNGKey(self.root_seed)
        key = jax.random.fold_in(key, self.schema_version)
        key = jax.random.fold_in(key, int(resolved))
        key = jax.random.fold_in(key, index)
        for coordinate in coordinates:
            key = jax.random.fold_in(key, coordinate)
        return key

    def next(
        self,
        domain: RNGDomain | str,
        *coordinates: int,
    ) -> tuple[PRNGKey, RNGLedger]:
        """Return the next domain key and an advanced immutable ledger."""
        _, domain_index = self._domain_index(domain)
        counter = self.counters[domain_index]
        if counter == np.iinfo(np.uint32).max:
            raise OverflowError("RNG domain counter is exhausted")
        key = self.key_for(domain, counter, *coordinates)
        counters = list(self.counters)
        counters[domain_index] = counter + 1
        return key, RNGLedger(
            root_seed=self.root_seed,
            counters=tuple(counters),
            schema_version=self.schema_version,
        )

    def allocate(
        self,
        domain: RNGDomain | str,
        count: int,
        *coordinates: int,
    ) -> tuple[Array, RNGLedger]:
        """Allocate ``count`` consecutive keys from one domain."""
        count = _positive_int("RNG allocation count", count)
        _, domain_index = self._domain_index(domain)
        start = self.counters[domain_index]
        if start + count - 1 > np.iinfo(np.uint32).max:
            raise OverflowError("RNG domain counter allocation would overflow")
        keys = jnp.stack(
            [self.key_for(domain, start + offset, *coordinates) for offset in range(count)]
        )
        counters = list(self.counters)
        counters[domain_index] = start + count
        return keys, RNGLedger(
            root_seed=self.root_seed,
            counters=tuple(counters),
            schema_version=self.schema_version,
        )

    def to_state(self) -> dict[str, Any]:
        """Return a strict JSON-serializable checkpoint state."""
        return {
            "schema_version": self.schema_version,
            "root_seed": self.root_seed,
            "counters": {
                domain.name.lower(): self.counters[index]
                for index, domain in enumerate(_RNG_DOMAINS)
            },
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> RNGLedger:
        """Restore a ledger, rejecting missing or unknown fields."""
        if not isinstance(state, Mapping):
            raise TypeError("RNG ledger state must be a mapping")
        expected_fields = {"schema_version", "root_seed", "counters"}
        if set(state) != expected_fields:
            raise ValueError("RNG ledger state fields do not match the current schema")
        counter_state = state["counters"]
        if not isinstance(counter_state, Mapping):
            raise TypeError("RNG ledger counters must be a mapping")
        expected_domains = set(_RNG_DOMAIN_BY_NAME)
        if set(counter_state) != expected_domains:
            raise ValueError("RNG ledger domains do not match the current schema")
        counters = tuple(counter_state[domain.name.lower()] for domain in _RNG_DOMAINS)
        return cls(
            root_seed=state["root_seed"],
            counters=counters,
            schema_version=state["schema_version"],
        )


@dataclass(frozen=True)
class CacheShardPlan:
    """Static leading-axis shard layout for a host-resident pair cache."""

    num_items: int
    shard_size: int
    num_shards: int
    final_shard_size: int
    static_shapes: bool

    def __post_init__(self) -> None:
        num_items = _positive_int("num_items", self.num_items)
        shard_size = _positive_int("shard_size", self.shard_size)
        num_shards = _positive_int("num_shards", self.num_shards)
        final_size = _positive_int("final_shard_size", self.final_shard_size)
        expected_shards = (num_items + shard_size - 1) // shard_size
        expected_final = num_items - shard_size * (expected_shards - 1)
        if num_shards != expected_shards or final_size != expected_final:
            raise ValueError("cache shard metadata is inconsistent")
        if self.static_shapes != (final_size == shard_size):
            raise ValueError("static_shapes must describe whether every shard is full")


@dataclass(frozen=True)
class CacheShard:
    """One numbered cache shard, on host or prefetched to a JAX device."""

    index: int
    start: int
    stop: int
    data: PyTree


def make_cache_shard_plan(
    *,
    num_items: int,
    shard_size: int,
    allow_partial_final_shard: bool = False,
) -> CacheShardPlan:
    """Create a shard layout without silently dropping a remainder."""
    num_items = _positive_int("num_items", num_items)
    shard_size = _positive_int("shard_size", shard_size)
    remainder = num_items % shard_size
    if remainder and not allow_partial_final_shard:
        raise ValueError(
            "num_items must be divisible by shard_size for static shapes; explicitly "
            "allow a partial final shard to retain the remainder"
        )
    num_shards = (num_items + shard_size - 1) // shard_size
    final_size = remainder or shard_size
    return CacheShardPlan(
        num_items=num_items,
        shard_size=shard_size,
        num_shards=num_shards,
        final_shard_size=final_size,
        static_shapes=final_size == shard_size,
    )


def _tree_leading_size(tree: PyTree, *, name: str) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError(f"{name} must have at least one leaf")
    sizes = set()
    for leaf in leaves:
        value = np.asarray(leaf) if not isinstance(leaf, jax.Array) else leaf
        if value.ndim < 1:
            raise ValueError(f"every {name} leaf must have a leading item axis")
        sizes.add(value.shape[0])
    if len(sizes) != 1:
        raise ValueError(f"all {name} leaves must share their leading size")
    return int(sizes.pop())


def iter_cache_shards(cache: PyTree, plan: CacheShardPlan) -> Iterator[CacheShard]:
    """Stream NumPy, memmap, or JAX cache leaves without copying the full cache."""
    num_items = _tree_leading_size(cache, name="cache")
    if num_items != plan.num_items:
        raise ValueError("cache leading size differs from its shard plan")
    for index in range(plan.num_shards):
        start = index * plan.shard_size
        stop = min(start + plan.shard_size, plan.num_items)
        data = jax.tree_util.tree_map(
            lambda value, start=start, stop=stop: value[start:stop], cache
        )
        yield CacheShard(index=index, start=start, stop=stop, data=data)


def prefetch_cache_shards(
    shards: Iterable[CacheShard],
    *,
    device: Any | None = None,
    buffer_size: int = 2,
) -> Iterator[CacheShard]:
    """Bounded lookahead that initiates asynchronous ``device_put`` transfers.

    JAX device transfers may complete asynchronously.  The next transfer is
    enqueued before the current shard is yielded, so consumer computation can
    overlap it.  The helper deliberately owns no background thread and has a
    bounded device-memory footprint of ``buffer_size`` shards.
    """
    buffer_size = _positive_int("buffer_size", buffer_size)
    iterator = iter(shards)
    queue: deque[CacheShard] = deque()

    def transfer(shard: CacheShard) -> CacheShard:
        return CacheShard(
            index=shard.index,
            start=shard.start,
            stop=shard.stop,
            data=jax.device_put(shard.data, device=device),
        )

    for _ in range(buffer_size):
        try:
            queue.append(transfer(next(iterator)))
        except StopIteration:
            break
    while queue:
        current = queue.popleft()
        try:
            queue.append(transfer(next(iterator)))
        except StopIteration:
            pass
        yield current


__all__ = [
    "CacheShard",
    "CacheShardPlan",
    "DeviceTopology",
    "ExecutionPlan",
    "GradientAccumulationResult",
    "GradientAccumulator",
    "RNGDomain",
    "RNGLedger",
    "accumulate_gradient_sequence",
    "accumulate_gradient_step",
    "default_microbatch_size",
    "discover_device_topology",
    "finalize_gradient_accumulator",
    "gradient_tree_is_finite",
    "initialize_gradient_accumulator",
    "iter_cache_shards",
    "make_cache_shard_plan",
    "make_execution_plan",
    "mean_gradient_replicas",
    "pmean_gradient_tree",
    "prefetch_cache_shards",
    "require_valid_gradients",
    "resolve_local_devices",
    "shard_microbatch_axis",
    "static_microbatch_indices",
    "take_scheduled_batches",
]
