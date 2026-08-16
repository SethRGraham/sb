r"""Nonlinear, single-device-first regression fields for MAM bridges.

This module contains two deliberately small supervised-learning components:

``MAMActorField``
    fits the empirical L2 regression

    .. math::

        a_\theta(x,t,y,s) \simeq
        \mathbb E[\widehat u^\dagger\mid X=x,t,Y=y,S=s],

    where ``s`` is ``+1`` in the forward direction and ``-1`` backward.

``MAMEndpointProjectorField``
    fits the endpoint-free Markov projection

    .. math::

        m_\psi(x,t,s) \simeq
        \mathbb E[\widehat Y_t\mid X=x,t,S=s].

The projector therefore cannot leak the paired endpoint into its prediction.
Both fits are finite-sample neural approximations, not exact conditional
expectations.  They use pure JAX forward/update kernels, float32 production
arithmetic, equal-size static microbatches, and one Adam update per effective
batch.  Any nonfinite datum, gradient, parameter, optimizer state, or
prediction fails closed.

Shapes
------
``B`` is the number of cached examples, ``d`` the state dimension, and ``c``
the endpoint dimension (normally ``c=d``).

* actor state/context/target: ``[B,d]``, ``[B,c]``, ``[B,d]``;
* projector state/target: ``[B,d]``, ``[B,c]``;
* time/direction: ``[B]``, with direction entries exactly ``-1`` or ``+1``.

The serialized state dictionary contains parameters, both Adam moments and
step, the next PRNG key, completed-step counter, complete loss history,
dimensions, field kind, and a configuration/factory fingerprint.  Its leaves
are NumPy arrays and Python scalars, so it is safe to hand to the repository's
checkpoint layer or standard pickle without device-dependent objects.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import marshal
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..core.types import Array, Params, PRNGKey
from ..network_factory import MLPFactory, NetworkFactory, sanity_check
from ..networks import AdamState, adam_update, init_adam

FORWARD_DIRECTION = 1.0
BACKWARD_DIRECTION = -1.0
_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class MAMFieldConfig:
    """Shared optimization and architecture configuration.

    ``effective_batch_size / microbatch_size`` equal-size gradients are
    averaged before every Adam update.  Sampling is with replacement from the
    immutable caller-provided cache.  The current implementation materializes
    that complete training cache as JAX arrays; host-streamed sharding and
    asynchronous prefetch remain future scalability work.
    """

    hidden_dims: tuple[int, ...] = (128, 128)
    time_embed_dim: int = 32
    activation: str = "swish"
    learning_rate: float = 1e-3
    training_steps: int = 1_000
    microbatch_size: int = 128
    effective_batch_size: int = 1_024
    weight_decay: float = 0.0
    network_factory: NetworkFactory | None = None

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(
            isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1
            for width in self.hidden_dims
        ):
            raise ValueError("hidden_dims must contain positive widths")
        if (
            isinstance(self.time_embed_dim, bool)
            or not isinstance(self.time_embed_dim, (int, np.integer))
            or self.time_embed_dim < 2
        ):
            raise ValueError("time_embed_dim must be at least two")
        if not isinstance(self.activation, str) or not self.activation:
            raise ValueError("activation must be a nonempty string")
        if (
            isinstance(self.learning_rate, bool)
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite")
        if (
            isinstance(self.training_steps, bool)
            or not isinstance(self.training_steps, (int, np.integer))
            or self.training_steps < 1
        ):
            raise ValueError("training_steps must be positive")
        if (
            isinstance(self.microbatch_size, bool)
            or not isinstance(self.microbatch_size, (int, np.integer))
            or self.microbatch_size < 1
        ):
            raise ValueError("microbatch_size must be positive")
        if (
            isinstance(self.effective_batch_size, bool)
            or not isinstance(self.effective_batch_size, (int, np.integer))
            or self.effective_batch_size < 1
        ):
            raise ValueError("effective_batch_size must be positive")
        if self.effective_batch_size % self.microbatch_size != 0:
            raise ValueError("effective_batch_size must be divisible by microbatch_size")
        if (
            isinstance(self.weight_decay, bool)
            or not np.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be nonnegative and finite")
        if self.network_factory is not None and not isinstance(
            self.network_factory, NetworkFactory
        ):
            raise TypeError("network_factory must implement NetworkFactory")

    @property
    def accumulation_steps(self) -> int:
        """Number of static microbatches averaged per Adam update."""
        return self.effective_batch_size // self.microbatch_size


@dataclass(frozen=True)
class MAMActorDataset:
    """Frozen actor-target cache.

    ``targets`` are caller-computed arrival-correct MAM action labels.  This
    class does not derive or certify those labels; it only performs their L2
    regression.
    """

    states: Array
    times: Array
    endpoints: Array
    directions: Array
    targets: Array


@dataclass(frozen=True)
class MAMProjectionDataset:
    """Frozen endpoint-prediction cache for Markov projection.

    The inputs contain no paired endpoint.  ``targets`` are typically the
    finite endpoint prediction ``y + (T-t) sqrt(rho) Sigma u`` (or its
    time-reversed counterpart), computed by the caller.
    """

    states: Array
    times: Array
    directions: Array
    targets: Array


class MAMFieldPrediction(NamedTuple):
    """Array-only prediction result suitable for use under ``jax.jit``."""

    value: Array
    finite: Array


@dataclass(frozen=True)
class MAMFieldTrainState:
    """Complete deterministic training/checkpoint state."""

    params: Params
    optimizer: AdamState
    next_key: PRNGKey
    completed_steps: int
    loss_history: Array
    field_kind: str
    input_dim: int
    output_dim: int
    config_fingerprint: str
    parameter_signature: str

    def to_state_dict(self) -> dict[str, Any]:
        """Return a host-resident, pickle-serializable state dictionary."""

        return {
            "version": _CHECKPOINT_VERSION,
            "field_kind": self.field_kind,
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "config_fingerprint": self.config_fingerprint,
            "parameter_signature": self.parameter_signature,
            "completed_steps": int(self.completed_steps),
            "params": _tree_to_host(self.params),
            "optimizer": {
                "m": _tree_to_host(self.optimizer.m),
                "v": _tree_to_host(self.optimizer.v),
                "step": int(np.asarray(jax.device_get(self.optimizer.step))),
            },
            "next_key": np.asarray(jax.device_get(self.next_key)),
            "loss_history": np.asarray(jax.device_get(self.loss_history)),
        }


def _tree_to_host(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def _tree_to_float32(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), tree)


def _tree_all_finite(tree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(False)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(jnp.asarray(value))) for value in leaves]))


def _tree_is_float32(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return bool(leaves) and all(
        np.asarray(jax.device_get(value)).dtype == np.dtype(np.float32) for value in leaves
    )


def _parameter_signature(params: Params) -> str:
    """Hash pytree paths, shapes, and dtypes (but never parameter values)."""

    path_leaves, tree_definition = jax.tree_util.tree_flatten_with_path(params)
    payload = {
        "tree": str(tree_definition),
        "leaves": [
            {
                "path": jax.tree_util.keystr(path),
                "shape": tuple(int(size) for size in jnp.asarray(value).shape),
                "dtype": str(jnp.asarray(value).dtype),
            }
            for path, value in path_leaves
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    """Stable, intentionally conservative factory/config description."""
    active: set[int] = set()

    def type_name(item: Any) -> str:
        return f"{type(item).__module__}.{type(item).__qualname__}"

    def type_source_digest(item: Any) -> str:
        try:
            source = inspect.getsource(type(item)).encode()
        except (OSError, TypeError):
            source = type_name(item).encode()
        return hashlib.sha256(source).hexdigest()

    def canonical(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (str, bool, int, float)) or item is None:
            return item
        if isinstance(item, np.ndarray) or (hasattr(item, "shape") and hasattr(item, "dtype")):
            array = np.ascontiguousarray(jax.device_get(item))
            return {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }

        item_id = id(item)
        if item_id in active:
            return {"cycle": type_name(item)}
        active.add(item_id)
        try:
            if isinstance(item, partial):
                return {
                    "partial": canonical(item.func),
                    "args": canonical(item.args),
                    "keywords": canonical(item.keywords or {}),
                }
            if isinstance(item, NetworkFactory):
                return {
                    "type": type_name(item),
                    "type_source_sha256": type_source_digest(item),
                    "attributes": canonical(vars(item)) if hasattr(item, "__dict__") else {},
                }
            if dataclasses.is_dataclass(item):
                return {
                    "dataclass_type": type_name(item),
                    "fields": {
                        field.name: canonical(getattr(item, field.name))
                        for field in dataclasses.fields(item)
                    },
                }
            if isinstance(item, dict):
                return {
                    str(key): canonical(child)
                    for key, child in sorted(item.items(), key=lambda pair: str(pair[0]))
                }
            if isinstance(item, (tuple, list)):
                return [canonical(child) for child in item]
            if callable(item):
                code = getattr(item, "__code__", None)
                closure = getattr(item, "__closure__", None)
                if code is None:
                    call_method = type(item).__call__
                    code = getattr(call_method, "__code__", None)
                return {
                    "callable_type": type_name(item),
                    "module": str(getattr(item, "__module__", "")),
                    "qualname": str(getattr(item, "__qualname__", "")),
                    "code_sha256": (
                        None if code is None else hashlib.sha256(marshal.dumps(code)).hexdigest()
                    ),
                    "defaults": canonical(getattr(item, "__defaults__", None)),
                    "closure": (
                        None
                        if closure is None
                        else [canonical(cell.cell_contents) for cell in closure]
                    ),
                    "state": canonical(vars(item)) if hasattr(item, "__dict__") else {},
                }
            if hasattr(item, "__dict__"):
                return {
                    "object_type": type_name(item),
                    "type_source_sha256": type_source_digest(item),
                    "state": canonical(vars(item)),
                }
            raise TypeError(f"unsupported MAM field fingerprint value of type {type_name(item)}")
        finally:
            active.remove(item_id)

    return canonical(value)


def _configuration_fingerprint(
    config: MAMFieldConfig,
    factory: NetworkFactory,
    *,
    field_kind: str,
    input_dim: int,
    output_dim: int,
) -> str:
    payload = {
        "field_kind": field_kind,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dims": config.hidden_dims,
        "time_embed_dim": config.time_embed_dim,
        "activation": config.activation,
        "learning_rate": config.learning_rate,
        "microbatch_size": config.microbatch_size,
        "effective_batch_size": config.effective_batch_size,
        "weight_decay": config.weight_decay,
        # training_steps is deliberately absent: it is a run length, not a
        # change to either the model or optimizer semantics, so a checkpoint
        # can be resumed for a different number of steps.
        "factory": _jsonable(factory),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _direction_valid(directions: Array) -> Array:
    return (directions == FORWARD_DIRECTION) | (directions == BACKWARD_DIRECTION)


def _validated_directions(value: Array) -> Array:
    """Return real numeric direction labels without lossy coercions."""
    original = jnp.asarray(value)
    if original.ndim != 1:
        raise ValueError("directions must have shape [batch]")
    if jnp.issubdtype(original.dtype, jnp.bool_) or jnp.issubdtype(
        original.dtype, jnp.complexfloating
    ):
        raise TypeError("directions must have a real non-boolean numeric dtype")
    if not (
        jnp.issubdtype(original.dtype, jnp.integer) or jnp.issubdtype(original.dtype, jnp.floating)
    ):
        raise TypeError("directions must have a real non-boolean numeric dtype")
    return jnp.asarray(original, dtype=jnp.float32)


def _field_prediction(
    factory: NetworkFactory,
    params: Params,
    features: Array,
    times: Array,
    directions: Array,
    output_dim: int,
) -> MAMFieldPrediction:
    """Pure common forward kernel; invalid rows are replaced by NaNs."""

    features = jnp.asarray(features, dtype=jnp.float32)
    times = jnp.asarray(times, dtype=jnp.float32)
    directions = jnp.asarray(directions, dtype=jnp.float32)
    output = jnp.asarray(factory.forward(params, features, times), dtype=jnp.float32)
    expected = (features.shape[0], output_dim)
    if output.shape != expected:
        raise ValueError(f"field factory must return shape {expected}, got {output.shape}")
    row_finite = (
        jnp.all(jnp.isfinite(features), axis=-1)
        & jnp.isfinite(times)
        & _direction_valid(directions)
        & jnp.all(jnp.isfinite(output), axis=-1)
    )
    value = jnp.where(row_finite[:, None], output, jnp.full_like(output, jnp.nan))
    return MAMFieldPrediction(value=value, finite=row_finite)


def actor_field_predict(
    factory: NetworkFactory,
    params: Params,
    states: Array,
    times: Array,
    endpoints: Array,
    directions: Array,
) -> MAMFieldPrediction:
    """Pure actor prediction ``(x,t,endpoint,direction) -> action``.

    The factory receives ``[x, endpoint, direction]`` as its spatial input and
    ``time`` through its dedicated time input.  Close over ``factory`` when
    compiling this function with :func:`jax.jit`.
    """

    states = jnp.asarray(states, dtype=jnp.float32)
    times = jnp.asarray(times, dtype=jnp.float32)
    endpoints = jnp.asarray(endpoints, dtype=jnp.float32)
    directions = jnp.asarray(directions, dtype=jnp.float32)
    _validate_static_prediction_shapes(states, times, endpoints, directions)
    features = jnp.concatenate([states, endpoints, directions[:, None]], axis=-1)
    return _field_prediction(
        factory,
        params,
        features,
        times,
        directions,
        states.shape[1],
    )


def endpoint_projector_field_predict(
    factory: NetworkFactory,
    params: Params,
    states: Array,
    times: Array,
    directions: Array,
    endpoint_dim: int,
) -> MAMFieldPrediction:
    """Pure endpoint-free prediction ``(x,t,direction) -> endpoint``."""

    states = jnp.asarray(states, dtype=jnp.float32)
    times = jnp.asarray(times, dtype=jnp.float32)
    directions = jnp.asarray(directions, dtype=jnp.float32)
    if states.ndim != 2:
        raise ValueError("states must have shape [batch,state_dim]")
    if times.shape != (states.shape[0],):
        raise ValueError("times must have shape [batch]")
    if directions.shape != (states.shape[0],):
        raise ValueError("directions must have shape [batch]")
    if (
        isinstance(endpoint_dim, bool)
        or not isinstance(endpoint_dim, (int, np.integer))
        or endpoint_dim < 1
    ):
        raise ValueError("endpoint_dim must be positive")
    features = jnp.concatenate([states, directions[:, None]], axis=-1)
    return _field_prediction(
        factory,
        params,
        features,
        times,
        directions,
        endpoint_dim,
    )


def _validate_static_prediction_shapes(
    states: Array,
    times: Array,
    endpoints: Array,
    directions: Array,
) -> None:
    if states.ndim != 2:
        raise ValueError("states must have shape [batch,state_dim]")
    batch = states.shape[0]
    if times.shape != (batch,):
        raise ValueError("times must have shape [batch]")
    if endpoints.ndim != 2 or endpoints.shape[0] != batch:
        raise ValueError("endpoints must have shape [batch,endpoint_dim]")
    if directions.shape != (batch,):
        raise ValueError("directions must have shape [batch]")


class _MAMNonlinearField:
    """Shared training implementation; not part of the public API."""

    def __init__(
        self,
        *,
        field_kind: str,
        input_dim: int,
        output_dim: int,
        config: MAMFieldConfig | None,
    ) -> None:
        if (
            isinstance(input_dim, bool)
            or not isinstance(input_dim, (int, np.integer))
            or input_dim < 1
        ):
            raise ValueError("input_dim must be positive")
        if (
            isinstance(output_dim, bool)
            or not isinstance(output_dim, (int, np.integer))
            or output_dim < 1
        ):
            raise ValueError("output_dim must be positive")
        self.config = config or MAMFieldConfig()
        self.factory: NetworkFactory = self.config.network_factory or MLPFactory(
            hidden_dims=self.config.hidden_dims,
            time_embed_dim=self.config.time_embed_dim,
            activation=self.config.activation,
        )
        self.field_kind = field_kind
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.config_fingerprint = _configuration_fingerprint(
            self.config,
            self.factory,
            field_kind=field_kind,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        self._update = jax.jit(self._make_accumulated_update())

    def _make_accumulated_update(self):
        factory = self.factory
        output_dim = self.output_dim
        learning_rate = self.config.learning_rate
        weight_decay = self.config.weight_decay
        accumulation_steps = self.config.accumulation_steps

        def microbatch_loss(
            params: Params,
            features: Array,
            times: Array,
            directions: Array,
            targets: Array,
        ) -> tuple[Array, Array]:
            prediction = _field_prediction(
                factory,
                params,
                features,
                times,
                directions,
                output_dim,
            )
            stopped_targets = jax.lax.stop_gradient(targets)
            residual = prediction.value - stopped_targets
            loss = jnp.mean(residual**2)
            finite = (
                jnp.all(prediction.finite)
                & jnp.all(jnp.isfinite(stopped_targets))
                & jnp.isfinite(loss)
            )
            return loss, finite

        def update(
            params: Params,
            optimizer: AdamState,
            features: Array,
            times: Array,
            directions: Array,
            targets: Array,
        ) -> tuple[Params, AdamState, dict[str, Array]]:
            gradient_zero = jax.tree_util.tree_map(jnp.zeros_like, params)
            initial = (
                gradient_zero,
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(True),
            )

            def scan_step(carry, microbatch):
                gradient_sum, loss_sum, finite = carry
                batch_features, batch_times, batch_directions, batch_targets = microbatch
                (loss, batch_finite), gradients = jax.value_and_grad(microbatch_loss, has_aux=True)(
                    params,
                    batch_features,
                    batch_times,
                    batch_directions,
                    batch_targets,
                )
                gradient_sum = jax.tree_util.tree_map(jnp.add, gradient_sum, gradients)
                finite = finite & batch_finite & _tree_all_finite(gradients)
                return (gradient_sum, loss_sum + loss, finite), None

            (gradient_sum, loss_sum, finite), _ = jax.lax.scan(
                scan_step,
                initial,
                (features, times, directions, targets),
            )
            denominator = jnp.asarray(accumulation_steps, dtype=jnp.float32)
            gradients = jax.tree_util.tree_map(
                lambda gradient: gradient / denominator,
                gradient_sum,
            )
            candidate_params, candidate_optimizer = adam_update(
                optimizer,
                gradients,
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            valid_update = (
                finite
                & _tree_all_finite(params)
                & _tree_all_finite(optimizer)
                & _tree_all_finite(gradients)
                & _tree_all_finite(candidate_params)
                & _tree_all_finite(candidate_optimizer)
            )
            safe_params = jax.tree_util.tree_map(
                lambda candidate, old: jnp.where(valid_update, candidate, old),
                candidate_params,
                params,
            )
            safe_optimizer = jax.tree_util.tree_map(
                lambda candidate, old: jnp.where(valid_update, candidate, old),
                candidate_optimizer,
                optimizer,
            )
            return (
                safe_params,
                safe_optimizer,
                {
                    "loss": loss_sum / denominator,
                    "gradients_finite": _tree_all_finite(gradients),
                    "parameters_finite": _tree_all_finite(candidate_params),
                    "optimizer_finite": _tree_all_finite(candidate_optimizer),
                    "valid_update": valid_update,
                },
            )

        return update

    def initialize(self, key: PRNGKey) -> MAMFieldTrainState:
        """Initialize float32 parameters and a complete zero-step state."""

        self._validate_key(key)
        sanity_key, init_key, next_key = jax.random.split(key, 3)
        try:
            sanity_check(
                self.factory,
                sanity_key,
                self.input_dim,
                self.output_dim,
            )
        except AssertionError as exc:
            if "NaN/Inf" in str(exc):
                raise FloatingPointError("nonfinite MAM field factory") from exc
            raise ValueError(f"invalid MAM field factory: {exc}") from exc
        params = _tree_to_float32(self.factory.init(init_key, self.input_dim, self.output_dim))
        if not bool(np.asarray(jax.device_get(_tree_all_finite(params)))):
            raise FloatingPointError("MAM field initialization is nonfinite")
        state = MAMFieldTrainState(
            params=params,
            optimizer=init_adam(params),
            next_key=next_key,
            completed_steps=0,
            loss_history=jnp.empty((0,), dtype=jnp.float32),
            field_kind=self.field_kind,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config_fingerprint=self.config_fingerprint,
            parameter_signature=_parameter_signature(params),
        )
        self._validate_state(state)
        return state

    def _fit_features(
        self,
        key: PRNGKey,
        features: Array,
        times: Array,
        directions: Array,
        targets: Array,
        *,
        steps: int | None,
    ) -> MAMFieldTrainState:
        state = self.initialize(key)
        return self._train_features(
            state,
            features,
            times,
            directions,
            targets,
            steps=steps,
        )

    def _train_features(
        self,
        state: MAMFieldTrainState,
        features: Array,
        times: Array,
        directions: Array,
        targets: Array,
        *,
        steps: int | None,
    ) -> MAMFieldTrainState:
        self._validate_state(state)
        features, times, directions, targets = self._validate_training_arrays(
            features,
            times,
            directions,
            targets,
        )
        step_count = self.config.training_steps if steps is None else steps
        if isinstance(step_count, bool) or not isinstance(step_count, (int, np.integer)):
            raise TypeError("steps must be an integer")
        if step_count < 1:
            raise ValueError("steps must be positive")

        params = state.params
        optimizer = state.optimizer
        next_key = state.next_key
        losses: list[Array] = []
        sample_shape = (
            self.config.accumulation_steps,
            self.config.microbatch_size,
        )
        for local_step in range(int(step_count)):
            next_key, sampling_key = jax.random.split(next_key)
            indices = jax.random.randint(
                sampling_key,
                sample_shape,
                minval=0,
                maxval=features.shape[0],
            )
            params, optimizer, metrics = self._update(
                params,
                optimizer,
                features[indices],
                times[indices],
                directions[indices],
                targets[indices],
            )
            if not bool(np.asarray(jax.device_get(metrics["valid_update"]))):
                raise FloatingPointError(f"nonfinite MAM field update at local step {local_step}")
            losses.append(metrics["loss"])
        result = MAMFieldTrainState(
            params=params,
            optimizer=optimizer,
            next_key=next_key,
            completed_steps=state.completed_steps + int(step_count),
            loss_history=jnp.concatenate([state.loss_history, jnp.stack(losses)]),
            field_kind=self.field_kind,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config_fingerprint=self.config_fingerprint,
            parameter_signature=state.parameter_signature,
        )
        self._validate_state(result)
        return result

    def load_state_dict(self, payload: dict[str, Any]) -> MAMFieldTrainState:
        """Validate and restore a state produced by ``to_state_dict``."""

        if not isinstance(payload, dict):
            raise TypeError("MAM field checkpoint must be a dictionary")
        required = {
            "version",
            "field_kind",
            "input_dim",
            "output_dim",
            "config_fingerprint",
            "parameter_signature",
            "completed_steps",
            "params",
            "optimizer",
            "next_key",
            "loss_history",
        }
        if set(payload) != required:
            missing = sorted(required - set(payload))
            extra = sorted(set(payload) - required)
            raise ValueError(f"invalid MAM field checkpoint keys; missing={missing}, extra={extra}")
        if payload["version"] != _CHECKPOINT_VERSION:
            raise ValueError("unsupported MAM field checkpoint version")
        optimizer_payload = payload["optimizer"]
        if not isinstance(optimizer_payload, dict) or set(optimizer_payload) != {
            "m",
            "v",
            "step",
        }:
            raise ValueError("invalid MAM field optimizer checkpoint")
        completed_steps = payload["completed_steps"]
        optimizer_step = optimizer_payload["step"]
        for name, value in (
            ("completed_steps", completed_steps),
            ("optimizer step", optimizer_step),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")

        def require_float32_tree(name: str, tree: Any) -> None:
            leaves = jax.tree_util.tree_leaves(tree)
            if not leaves:
                raise ValueError(f"{name} tree must be nonempty")
            if any(np.asarray(value).dtype != np.dtype(np.float32) for value in leaves):
                raise TypeError(f"all {name} leaves must use float32")

        require_float32_tree("parameter", payload["params"])
        require_float32_tree("Adam first-moment", optimizer_payload["m"])
        require_float32_tree("Adam second-moment", optimizer_payload["v"])
        if np.asarray(payload["loss_history"]).dtype != np.dtype(np.float32):
            raise TypeError("loss_history must use float32")
        state = MAMFieldTrainState(
            params=jax.tree_util.tree_map(jnp.asarray, payload["params"]),
            optimizer=AdamState(
                m=jax.tree_util.tree_map(jnp.asarray, optimizer_payload["m"]),
                v=jax.tree_util.tree_map(jnp.asarray, optimizer_payload["v"]),
                step=int(optimizer_step),
            ),
            next_key=jnp.asarray(payload["next_key"]),
            completed_steps=int(completed_steps),
            loss_history=jnp.asarray(payload["loss_history"]),
            field_kind=str(payload["field_kind"]),
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
            config_fingerprint=str(payload["config_fingerprint"]),
            parameter_signature=str(payload["parameter_signature"]),
        )
        self._validate_state(state)
        return state

    def validate_state(self, state: MAMFieldTrainState) -> None:
        """Validate an in-memory train state without coercing any leaf."""
        self._validate_state(state)

    def _validate_training_arrays(
        self,
        features: Array,
        times: Array,
        directions: Array,
        targets: Array,
    ) -> tuple[Array, Array, Array, Array]:
        features = _validated_float_array("features", features, ndim=2)
        times = _validated_float_array("times", times, ndim=1)
        directions = _validated_directions(directions)
        targets = _validated_float_array("targets", targets, ndim=2)
        batch = features.shape[0]
        if batch < 1:
            raise ValueError("training cache must be nonempty")
        if features.shape[1] != self.input_dim:
            raise ValueError(f"features must have width {self.input_dim}, got {features.shape[1]}")
        if times.shape != (batch,):
            raise ValueError("times must have shape [batch]")
        if directions.shape != (batch,):
            raise ValueError("directions must have shape [batch]")
        if targets.shape != (batch, self.output_dim):
            raise ValueError(
                f"targets must have shape {(batch, self.output_dim)}, got {targets.shape}"
            )
        if not bool(np.asarray(jax.device_get(jnp.all(_direction_valid(directions))))):
            raise ValueError("directions must contain only -1 or +1")
        return features, times, directions, targets

    def _validate_state(self, state: MAMFieldTrainState) -> None:
        if not isinstance(state, MAMFieldTrainState):
            raise TypeError("state must be MAMFieldTrainState")
        if state.field_kind != self.field_kind:
            raise ValueError("field kind differs from this model")
        if state.input_dim != self.input_dim or state.output_dim != self.output_dim:
            raise ValueError("field dimensions differ from this model")
        if state.config_fingerprint != self.config_fingerprint:
            raise ValueError("field configuration/factory fingerprint mismatch")
        if state.parameter_signature != _parameter_signature(state.params):
            raise ValueError("field parameter structure/shape signature mismatch")
        if isinstance(state.completed_steps, bool) or state.completed_steps < 0:
            raise ValueError("completed_steps must be nonnegative")
        history = jnp.asarray(state.loss_history)
        if history.shape != (state.completed_steps,):
            raise ValueError("loss_history length must equal completed_steps")
        if history.dtype != jnp.float32:
            raise TypeError("loss_history must use float32")
        if not _tree_is_float32(state.params):
            raise TypeError("all MAM field parameters must use float32")
        if not _tree_is_float32(state.optimizer.m) or not _tree_is_float32(state.optimizer.v):
            raise TypeError("all MAM field optimizer moments must use float32")
        structure = jax.tree_util.tree_structure(state.params)
        if jax.tree_util.tree_structure(state.optimizer.m) != structure:
            raise ValueError("Adam first-moment tree differs from parameters")
        if jax.tree_util.tree_structure(state.optimizer.v) != structure:
            raise ValueError("Adam second-moment tree differs from parameters")
        for parameter, first, second in zip(
            jax.tree_util.tree_leaves(state.params),
            jax.tree_util.tree_leaves(state.optimizer.m),
            jax.tree_util.tree_leaves(state.optimizer.v),
            strict=True,
        ):
            if parameter.shape != first.shape or parameter.shape != second.shape:
                raise ValueError("Adam moment shape differs from its parameter")
        optimizer_step_array = np.asarray(jax.device_get(state.optimizer.step))
        if optimizer_step_array.shape != () or not np.issubdtype(
            optimizer_step_array.dtype,
            np.integer,
        ):
            raise TypeError("Adam step must be an integer scalar")
        optimizer_step = int(optimizer_step_array)
        if optimizer_step != state.completed_steps:
            raise ValueError("Adam step differs from completed_steps")
        self._validate_key(state.next_key)
        finite = (
            _tree_all_finite(state.params)
            & _tree_all_finite(state.optimizer)
            & jnp.all(jnp.isfinite(history))
        )
        if not bool(np.asarray(jax.device_get(finite))):
            raise FloatingPointError("MAM field train state contains a nonfinite value")

    @staticmethod
    def _validate_key(key: PRNGKey) -> None:
        try:
            jax.random.split(key)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid JAX PRNG key") from exc


def _validated_float_array(name: str, value: Array, *, ndim: int) -> Array:
    original = jnp.asarray(value)
    if original.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if not jnp.issubdtype(original.dtype, jnp.floating):
        raise TypeError(f"{name} must have a real floating dtype")
    cast = jnp.asarray(original, dtype=jnp.float32)
    if not bool(np.asarray(jax.device_get(jnp.all(jnp.isfinite(cast))))):
        raise FloatingPointError(f"{name} contains a nonfinite value")
    return cast


class MAMActorField(_MAMNonlinearField):
    """Nonlinear endpoint-conditioned action regressor.

    The field input has width ``state_dim + endpoint_dim + 1`` and the action
    output has width ``state_dim``.  L2 fitting learns only the conditional
    mean of the supplied action labels; label correctness remains the
    responsibility of the MAM costate/arrival-correction pipeline.
    """

    def __init__(
        self,
        state_dim: int,
        endpoint_dim: int | None = None,
        config: MAMFieldConfig | None = None,
    ) -> None:
        endpoint_dim = state_dim if endpoint_dim is None else endpoint_dim
        if (
            isinstance(state_dim, bool)
            or not isinstance(state_dim, (int, np.integer))
            or state_dim < 1
        ):
            raise ValueError("state_dim must be positive")
        if (
            isinstance(endpoint_dim, bool)
            or not isinstance(endpoint_dim, (int, np.integer))
            or endpoint_dim < 1
        ):
            raise ValueError("endpoint_dim must be positive")
        self.state_dim = int(state_dim)
        self.endpoint_dim = int(endpoint_dim)
        super().__init__(
            field_kind="actor",
            input_dim=self.state_dim + self.endpoint_dim + 1,
            output_dim=self.state_dim,
            config=config,
        )

    def fit(
        self,
        key: PRNGKey,
        dataset: MAMActorDataset,
        *,
        steps: int | None = None,
    ) -> MAMFieldTrainState:
        """Initialize and fit an actor to a frozen target cache."""

        features, times, directions, targets = self._actor_arrays(dataset)
        return self._fit_features(
            key,
            features,
            times,
            directions,
            targets,
            steps=steps,
        )

    def train(
        self,
        state: MAMFieldTrainState,
        dataset: MAMActorDataset,
        *,
        steps: int | None = None,
    ) -> MAMFieldTrainState:
        """Continue a fit using the state's recorded next PRNG key."""

        features, times, directions, targets = self._actor_arrays(dataset)
        return self._train_features(
            state,
            features,
            times,
            directions,
            targets,
            steps=steps,
        )

    def predict(
        self,
        state: MAMFieldTrainState,
        states: Array,
        times: Array,
        endpoints: Array,
        directions: Array,
    ) -> Array:
        """Return float32 actions, raising on any invalid/nonfinite row."""

        self._validate_state(state)
        states, times, endpoints, directions = self._validate_actor_inputs(
            states,
            times,
            endpoints,
            directions,
        )
        prediction = actor_field_predict(
            self.factory,
            state.params,
            states,
            times,
            endpoints,
            directions,
        )
        if not bool(np.asarray(jax.device_get(jnp.all(prediction.finite)))):
            raise FloatingPointError("actor prediction is nonfinite")
        return prediction.value

    def _actor_arrays(self, dataset: MAMActorDataset) -> tuple[Array, Array, Array, Array]:
        if not isinstance(dataset, MAMActorDataset):
            raise TypeError("dataset must be MAMActorDataset")
        states, times, endpoints, directions = self._validate_actor_inputs(
            dataset.states,
            dataset.times,
            dataset.endpoints,
            dataset.directions,
        )
        targets = _validated_float_array("targets", dataset.targets, ndim=2)
        if targets.shape != (states.shape[0], self.state_dim):
            raise ValueError(f"targets must have shape {(states.shape[0], self.state_dim)}")
        features = jnp.concatenate([states, endpoints, directions[:, None]], axis=-1)
        return features, times, directions, targets

    def _validate_actor_inputs(
        self,
        states: Array,
        times: Array,
        endpoints: Array,
        directions: Array,
    ) -> tuple[Array, Array, Array, Array]:
        states = _validated_float_array("states", states, ndim=2)
        times = _validated_float_array("times", times, ndim=1)
        endpoints = _validated_float_array("endpoints", endpoints, ndim=2)
        directions = _validated_directions(directions)
        batch = states.shape[0]
        if states.shape[1] != self.state_dim:
            raise ValueError(f"states must have width {self.state_dim}")
        if times.shape != (batch,):
            raise ValueError("times must have shape [batch]")
        if endpoints.shape != (batch, self.endpoint_dim):
            raise ValueError(f"endpoints must have shape {(batch, self.endpoint_dim)}")
        if directions.shape != (batch,):
            raise ValueError("directions must have shape [batch]")
        if not bool(np.asarray(jax.device_get(jnp.all(jnp.isfinite(directions))))):
            raise FloatingPointError("directions contain a nonfinite value")
        if not bool(np.asarray(jax.device_get(jnp.all(_direction_valid(directions))))):
            raise ValueError("directions must contain only -1 or +1")
        return states, times, endpoints, directions


class MAMEndpointProjectorField(_MAMNonlinearField):
    """Nonlinear endpoint-free Markov endpoint-prediction regressor."""

    def __init__(
        self,
        state_dim: int,
        endpoint_dim: int | None = None,
        config: MAMFieldConfig | None = None,
    ) -> None:
        endpoint_dim = state_dim if endpoint_dim is None else endpoint_dim
        if (
            isinstance(state_dim, bool)
            or not isinstance(state_dim, (int, np.integer))
            or state_dim < 1
        ):
            raise ValueError("state_dim must be positive")
        if (
            isinstance(endpoint_dim, bool)
            or not isinstance(endpoint_dim, (int, np.integer))
            or endpoint_dim < 1
        ):
            raise ValueError("endpoint_dim must be positive")
        self.state_dim = int(state_dim)
        self.endpoint_dim = int(endpoint_dim)
        super().__init__(
            field_kind="endpoint_projector",
            input_dim=self.state_dim + 1,
            output_dim=self.endpoint_dim,
            config=config,
        )

    def fit(
        self,
        key: PRNGKey,
        dataset: MAMProjectionDataset,
        *,
        steps: int | None = None,
    ) -> MAMFieldTrainState:
        """Initialize and fit an endpoint-free projector."""

        features, times, directions, targets = self._projection_arrays(dataset)
        return self._fit_features(
            key,
            features,
            times,
            directions,
            targets,
            steps=steps,
        )

    def train(
        self,
        state: MAMFieldTrainState,
        dataset: MAMProjectionDataset,
        *,
        steps: int | None = None,
    ) -> MAMFieldTrainState:
        """Continue a projector fit from a complete train state."""

        features, times, directions, targets = self._projection_arrays(dataset)
        return self._train_features(
            state,
            features,
            times,
            directions,
            targets,
            steps=steps,
        )

    def predict(
        self,
        state: MAMFieldTrainState,
        states: Array,
        times: Array,
        directions: Array,
    ) -> Array:
        """Return float32 endpoint predictions with no endpoint input."""

        self._validate_state(state)
        states, times, directions = self._validate_projection_inputs(
            states,
            times,
            directions,
        )
        prediction = endpoint_projector_field_predict(
            self.factory,
            state.params,
            states,
            times,
            directions,
            self.endpoint_dim,
        )
        if not bool(np.asarray(jax.device_get(jnp.all(prediction.finite)))):
            raise FloatingPointError("endpoint-projector prediction is nonfinite")
        return prediction.value

    def _projection_arrays(
        self, dataset: MAMProjectionDataset
    ) -> tuple[Array, Array, Array, Array]:
        if not isinstance(dataset, MAMProjectionDataset):
            raise TypeError("dataset must be MAMProjectionDataset")
        states, times, directions = self._validate_projection_inputs(
            dataset.states,
            dataset.times,
            dataset.directions,
        )
        targets = _validated_float_array("targets", dataset.targets, ndim=2)
        if targets.shape != (states.shape[0], self.endpoint_dim):
            raise ValueError(f"targets must have shape {(states.shape[0], self.endpoint_dim)}")
        features = jnp.concatenate([states, directions[:, None]], axis=-1)
        return features, times, directions, targets

    def _validate_projection_inputs(
        self,
        states: Array,
        times: Array,
        directions: Array,
    ) -> tuple[Array, Array, Array]:
        states = _validated_float_array("states", states, ndim=2)
        times = _validated_float_array("times", times, ndim=1)
        directions = _validated_directions(directions)
        batch = states.shape[0]
        if states.shape[1] != self.state_dim:
            raise ValueError(f"states must have width {self.state_dim}")
        if times.shape != (batch,):
            raise ValueError("times must have shape [batch]")
        if directions.shape != (batch,):
            raise ValueError("directions must have shape [batch]")
        if not bool(np.asarray(jax.device_get(jnp.all(jnp.isfinite(directions))))):
            raise FloatingPointError("directions contain a nonfinite value")
        if not bool(np.asarray(jax.device_get(jnp.all(_direction_valid(directions))))):
            raise ValueError("directions must contain only -1 or +1")
        return states, times, directions


__all__ = [
    "BACKWARD_DIRECTION",
    "FORWARD_DIRECTION",
    "MAMActorDataset",
    "MAMActorField",
    "MAMEndpointProjectorField",
    "MAMFieldConfig",
    "MAMFieldPrediction",
    "MAMFieldTrainState",
    "MAMProjectionDataset",
    "actor_field_predict",
    "endpoint_projector_field_predict",
]
