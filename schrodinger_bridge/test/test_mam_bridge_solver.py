"""Focused tests for the experimental global MAM bridge composition."""

from __future__ import annotations

import copy
import pickle
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import schrodinger_bridge.solvers.mam_bridge as mam_bridge_module
from schrodinger_bridge.core.problem import (
    BrownianMotion,
    GaussianDistribution,
    MixtureDistribution,
    SBProblem,
)
from schrodinger_bridge.core.types import SolverConfig, TimeGrid, TrainingConfig
from schrodinger_bridge.network_factory import NetworkFactory
from schrodinger_bridge.solvers.malliavin_adjoint import (
    ActionTargetBatch,
    MalliavinAdjointConfig,
)
from schrodinger_bridge.solvers.mam_accounting import (
    MAMWorkCounters,
    completed_conditional_solve_work,
)
from schrodinger_bridge.solvers.mam_bridge import (
    ConditionalMAMConfig,
    ConditionalMAMResult,
    EndpointPairBatch,
    MAMBridgeConfig,
    MAMBridgeSolution,
    MAMBridgeSolver,
    MAMConditionalSolver,
    MAMExecutionConfig,
    MAMOuterLoopConfig,
    MarkovProjectionConfig,
    MarkovProjector,
    ValueOnlyRunningPotential,
)
from schrodinger_bridge.solvers.mam_diagnostics import EndpointAuditConfig
from schrodinger_bridge.solvers.mam_fields import (
    MAMActorDataset,
    MAMActorField,
    MAMFieldConfig,
)
from schrodinger_bridge.solvers.mam_value_critic import ValueCriticConfig


def _problem(dim: int = 2, num_steps: int = 4) -> SBProblem:
    return SBProblem(
        reference=BrownianMotion(sigma=0.35, dim=dim),
        source=GaussianDistribution(mean=-jnp.ones((dim,)), cov=0.15, dim=dim),
        target=GaussianDistribution(mean=jnp.ones((dim,)), cov=0.15, dim=dim),
        time_grid=TimeGrid(num_steps=num_steps),
        name="MAM bridge Gaussian smoke",
    )


class _PinnedReferenceConditional:
    """Cheap exact-pinning backend used to isolate the global outer loop."""

    status = "TEST_PINNED_REFERENCE_NOT_MAM"

    def __init__(self, problem: SBProblem, scientific_variant: str = "v1"):
        self.problem = problem
        self.scientific_variant = scientific_variant
        self.calls: list[str] = []

    def solve(self, key, endpoint_pairs, direction):
        del key
        self.calls.append(direction)
        if direction == "f":
            start, endpoint = endpoint_pairs.source, endpoint_pairs.target
        else:
            start, endpoint = endpoint_pairs.target, endpoint_pairs.source
        local_fraction = jnp.linspace(
            0.0,
            1.0,
            self.problem.time_grid.num_steps + 1,
            dtype=start.dtype,
        )
        local_paths = (1.0 - local_fraction)[None, :, None] * start[:, None, :] + local_fraction[
            None, :, None
        ] * endpoint[:, None, :]
        paths = local_paths if direction == "f" else local_paths[:, ::-1]
        projection_states = local_paths[:, :-1]
        predictions = jnp.broadcast_to(endpoint[:, None, :], projection_states.shape)
        times = jnp.asarray(self.problem.time_grid.times[:-1], dtype=start.dtype)
        if direction == "b":
            times = self.problem.time_grid.t1 - times
        controls = jnp.zeros(
            (
                start.shape[0],
                self.problem.time_grid.num_steps - 1,
                self.problem.dim,
            ),
            dtype=start.dtype,
        )
        return ConditionalMAMResult(
            paths=paths,
            local_paths=local_paths,
            controls=controls,
            projection_states=projection_states,
            projection_times=times,
            endpoint_predictions=predictions,
            actor_params={},
            costate_params={},
            metrics={"exact_conditional_endpoint": True},
            direction=direction,
            exact_conditional_endpoint=True,
            status=self.status,
        )

    def state_dict(self):
        return {
            "schema_version": 1,
            "backend_status": self.status,
            "calls": tuple(self.calls),
        }

    def load_state_dict(self, state):
        if set(state) != {"schema_version", "backend_status", "calls"}:
            raise ValueError("test conditional state schema mismatch")
        if state["schema_version"] != 1 or state["backend_status"] != self.status:
            raise ValueError("test conditional state status mismatch")
        self.calls = list(state["calls"])

    def scientific_fingerprint(self):
        return (
            f"{type(self).__module__}.{type(self).__qualname__}:"
            f"deterministic_linear_pin:{self.scientific_variant}"
        )


class _DifferentPinnedReferenceConditional(_PinnedReferenceConditional):
    """Semantically distinct injected backend used for resume rejection."""

    status = "TEST_DIFFERENT_PINNED_REFERENCE"


class _UnpinnedReferenceConditional(_PinnedReferenceConditional):
    status = "TEST_UNPINNED_REFERENCE"

    def solve(self, key, endpoint_pairs, direction):
        result = super().solve(key, endpoint_pairs, direction)
        return replace(result, exact_conditional_endpoint=False)


class _FreshScalarFactory(NetworkFactory):
    """Equivalent fresh instances intentionally retain object-identity reprs."""

    def init(self, key, input_dim, output_dim):
        return {"weight": jax.random.normal(key, (input_dim, output_dim), dtype=jnp.float32)}

    def forward(self, params, x, t):
        del t
        return x @ params["weight"]


class _TimeEchoFactory(NetworkFactory):
    """One-dimensional test field whose prediction exposes its exact time input."""

    def init(self, key, input_dim, output_dim):
        del key, input_dim
        return {"scale": jnp.ones((output_dim,), dtype=jnp.float32)}

    def forward(self, params, x, t):
        del x
        return jnp.asarray(t)[:, None] * params["scale"][None, :]


class _CallableStateFactory(NetworkFactory):
    def __init__(self, scale):
        self._activation = lambda value: scale * value

    def init(self, key, input_dim, output_dim):
        return {"weight": jax.random.normal(key, (input_dim, output_dim), dtype=jnp.float32)}

    def forward(self, params, x, t):
        del t
        return self._activation(x @ params["weight"])


class _ThresholdRunningCost:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.cycle = self

    def value(self, states, times, context):
        del times, context
        return (states[:, 0] > self.threshold).astype(states.dtype)


class _StateDependentBrownian(BrownianMotion):
    def diffusion(self, x, t):
        del t
        return self.sigma * (1.0 + 0.1 * jnp.mean(x))


_GLOBAL_RUNNING_SCALE = 1.0
_OPAQUE_FINGERPRINT_STATE = object()


def _global_scale_running_cost(states, times, context):
    del times, context
    return _GLOBAL_RUNNING_SCALE * jnp.sum(states**2, axis=-1)


def _opaque_global_running_cost(states, times, context):
    del times, context
    if _OPAQUE_FINGERPRINT_STATE is None:
        return jnp.zeros((states.shape[0],), dtype=states.dtype)
    return jnp.sum(states**2, axis=-1)


class _ClassShiftGaussian(GaussianDistribution):
    class_shift = 0.0

    def sample(self, key, num_samples):
        return super().sample(key, num_samples) + type(self).class_shift


def _nested_fingerprint_callback(states, times, context):
    del times, context
    transforms = tuple((lambda value, shift=shift: value + shift) for shift in (0.0, 1.0))
    return sum(transform(states[:, 0]) for transform in transforms)


def _tiny_config(**outer_overrides) -> MAMBridgeConfig:
    outer = MAMOuterLoopConfig(
        num_iterations=1,
        cache_size=16,
        audit_size=16,
    )
    outer = replace(outer, **outer_overrides)
    return MAMBridgeConfig(
        conditional=ConditionalMAMConfig(actor_model="affine_reference"),
        projection=MarkovProjectionConfig(
            model="affine_reference",
            ridge=1e-4,
            validation_size=16,
            validation_projections=4,
            validation_replicates=2,
            line_search=(1.0, 0.5),
        ),
        outer=outer,
        execution=MAMExecutionConfig(
            microbatch_size=4,
            effective_batch_size=4,
        ),
        audit=EndpointAuditConfig(
            num_projections=2,
            sinkhorn_iterations=20,
            sinkhorn_tolerance=5e-2,
            sinkhorn_max_samples=16,
            null_replicates=2,
        ),
    )


def test_global_config_requires_both_directions():
    with pytest.raises(ValueError, match="both forward and backward"):
        MAMOuterLoopConfig(directions=("f",))
    with pytest.raises(ValueError, match="exactly once"):
        MAMOuterLoopConfig(directions=("f", "f", "b"))
    with pytest.raises(ValueError, match="validation_replicates"):
        MarkovProjectionConfig(validation_replicates=1)


def test_v1_problem_rejects_brownian_subclasses_and_nonfinite_diffusion():
    base = _problem(dim=1, num_steps=3)
    state_dependent = replace(
        base,
        reference=_StateDependentBrownian(sigma=0.35, dim=1),
    )
    with pytest.raises(ValueError, match="subclasses"):
        MAMBridgeSolver(state_dependent, ValueOnlyRunningPotential(), _tiny_config())
    nonfinite = replace(base, reference=BrownianMotion(sigma=np.nan, dim=1))
    with pytest.raises(ValueError, match="finite"):
        MAMBridgeSolver(nonfinite, ValueOnlyRunningPotential(), _tiny_config())


def test_public_config_wrappers_validate_semantic_types():
    with pytest.raises(TypeError, match="callable"):
        ValueOnlyRunningPotential(value=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonempty"):
        ValueOnlyRunningPotential(identifier="  ")
    with pytest.raises(TypeError, match="conditional"):
        MAMBridgeConfig(conditional={})  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [True, 2.0, 2.5, np.float64(2.0)])
def test_solver_mode_count_requires_a_strict_integer(invalid):
    problem = _problem(dim=1, num_steps=3)

    def labels(samples):
        return (samples[:, 0] >= 0.0).astype(jnp.int32)

    with pytest.raises(TypeError, match="source_num_modes.*integer"):
        MAMBridgeSolver(
            problem,
            ValueOnlyRunningPotential(),
            _tiny_config(),
            source_mode_label_fn=labels,
            source_num_modes=invalid,
        )


def test_solver_mode_count_rejects_nonpositive_and_ignored_values():
    problem = _problem(dim=1, num_steps=3)

    def labels(samples):
        return (samples[:, 0] >= 0.0).astype(jnp.int32)

    with pytest.raises(ValueError, match="source_num_modes.*at least 1"):
        MAMBridgeSolver(
            problem,
            ValueOnlyRunningPotential(),
            _tiny_config(),
            source_mode_label_fn=labels,
            source_num_modes=0,
        )
    with pytest.raises(ValueError, match="only with a mode-label"):
        MAMBridgeSolver(
            problem,
            ValueOnlyRunningPotential(),
            _tiny_config(),
            source_num_modes=2,
        )

    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        source_mode_label_fn=labels,
        source_num_modes=np.int32(2),
    )
    assert solver._source_num_modes == np.int32(2)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: ConditionalMAMConfig(costate_steps=1.5), "integer"),
        (lambda: ConditionalMAMConfig(batch_size=True), "integer"),
        (lambda: ConditionalMAMConfig(actor_ridge=np.nan), "finite"),
        (lambda: ConditionalMAMConfig(one_sided_z=np.inf), "finite"),
        (lambda: ConditionalMAMConfig(train_value_critic=1), "bool"),
        (lambda: ConditionalMAMConfig(line_search=(1.0, np.nan)), "finite"),
        (lambda: MarkovProjectionConfig(maximum_components=1.5), "integer"),
        (lambda: MarkovProjectionConfig(validation_size=True), "integer"),
        (lambda: MarkovProjectionConfig(ridge=np.nan), "finite"),
        (lambda: MarkovProjectionConfig(damping=np.inf), "finite"),
        (lambda: MarkovProjectionConfig(skip_terminal_noise=0), "bool"),
        (lambda: MAMOuterLoopConfig(num_iterations=1.5), "integer"),
        (lambda: MAMOuterLoopConfig(cache_size=True), "integer"),
        (lambda: MAMExecutionConfig(effective_batch_size=1.5), "integer"),
        (lambda: MAMExecutionConfig(microbatch_size=True), "integer"),
        (lambda: MAMExecutionConfig(allow_two_devices=1), "bool"),
    ],
)
def test_bridge_configs_reject_lossy_or_nonfinite_numeric_values(factory, match):
    with pytest.raises((TypeError, ValueError), match=match):
        factory()


def test_projection_ci_sample_size_is_independent_cloud_count_not_direction_count():
    problem = _problem(dim=1, num_steps=3)
    base = _tiny_config()
    sparse_directions = replace(
        base,
        projection=replace(
            base.projection,
            validation_size=4,
            validation_projections=2,
            validation_replicates=3,
        ),
    )
    many_directions = replace(
        sparse_directions,
        projection=replace(
            sparse_directions.projection,
            validation_projections=17,
        ),
    )
    key = jax.random.PRNGKey(701)
    sparse_solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        sparse_directions,
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    many_solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        many_directions,
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    sparse_params = sparse_solver.init_params(jax.random.PRNGKey(702))
    many_params = many_solver.init_params(jax.random.PRNGKey(702))
    sparse_scores = sparse_solver._projection_endpoint_scores(
        key,
        sparse_params,
        "f",
    )
    repeated_scores = sparse_solver._projection_endpoint_scores(
        key,
        sparse_params,
        "f",
    )
    many_scores = many_solver._projection_endpoint_scores(key, many_params, "f")

    assert sparse_scores.shape == (3,)
    assert many_scores.shape == (3,)
    assert jnp.array_equal(sparse_scores, repeated_scores)


def test_nonlinear_models_are_default_and_inherit_resolved_execution_batches():
    assert ConditionalMAMConfig().actor_model == "nonlinear"
    assert MarkovProjectionConfig().model == "nonlinear"
    problem = _problem(dim=128, num_steps=3)
    execution = MAMExecutionConfig(microbatch_size=None, effective_batch_size=1024)
    conditional = MAMConditionalSolver(
        problem,
        ValueOnlyRunningPotential().as_value_only_cost(),
        ConditionalMAMConfig(),
        execution,
    )
    projector = MarkovProjector(problem, MarkovProjectionConfig(), execution)
    assert conditional._actor_field.config.microbatch_size == 32
    assert conditional._actor_field.config.effective_batch_size == 1024
    assert projector._field.config.microbatch_size == 32
    assert projector._field.config.effective_batch_size == 1024


def test_private_callable_factory_state_changes_field_and_outer_fingerprints():
    first_field_config = MAMFieldConfig(network_factory=_CallableStateFactory(1.0))
    second_field_config = replace(
        first_field_config,
        network_factory=_CallableStateFactory(2.0),
    )
    assert (
        MAMActorField(1, 1, first_field_config).config_fingerprint
        != MAMActorField(1, 1, second_field_config).config_fingerprint
    )

    problem = _problem(dim=1, num_steps=3)
    first_config = replace(
        _tiny_config(),
        conditional=replace(
            _tiny_config().conditional,
            actor_model="nonlinear",
            actor_field_config=first_field_config,
        ),
    )
    second_config = replace(
        first_config,
        conditional=replace(
            first_config.conditional,
            actor_field_config=second_field_config,
        ),
    )
    first_solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        first_config,
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    second_solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        second_config,
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    assert (
        first_solver._scientific_fingerprints()["config_sha256"]
        != second_solver._scientific_fingerprints()["config_sha256"]
    )


def test_conditional_actor_preserves_heterogeneous_row_times_eager_and_jit():
    problem = replace(
        _problem(dim=1, num_steps=4),
        time_grid=TimeGrid(t0=0.001, t1=0.101, num_steps=4),
    )
    affine = MAMConditionalSolver(
        problem,
        ValueOnlyRunningPotential().as_value_only_cost(),
        ConditionalMAMConfig(actor_model="affine_reference"),
        MAMExecutionConfig(microbatch_size=4, effective_batch_size=4),
    )
    params = jnp.asarray(affine._zero_actor_params(jnp.float32))
    params = params.at[:, -1, 0].set(jnp.asarray([3.0, 5.0, 7.0], dtype=jnp.float32))
    actor = affine._actor_fn(params, "f")
    states = jnp.zeros((3, 1), dtype=jnp.float32)
    endpoints = jnp.zeros_like(states)
    row_times = jnp.asarray(problem.time_grid.times[:3], dtype=jnp.float32)
    expected = jnp.asarray([[3.0], [5.0], [7.0]], dtype=jnp.float32)
    assert jnp.array_equal(actor(states, row_times, endpoints), expected)
    assert jnp.array_equal(jax.jit(actor)(states, row_times, endpoints), expected)

    with pytest.raises(ValueError, match="exact departure grid"):
        actor(states, row_times.at[1].add(0.1 * problem.time_grid.dt), endpoints)
    compiled_invalid = jax.jit(actor)(
        states,
        row_times.at[1].add(0.1 * problem.time_grid.dt),
        endpoints,
    )
    assert jnp.all(jnp.isfinite(compiled_invalid[jnp.asarray([0, 2])]))
    assert jnp.isnan(compiled_invalid[1, 0])

    nonlinear_config = ConditionalMAMConfig(
        actor_model="nonlinear",
        actor_field_config=MAMFieldConfig(
            hidden_dims=(4,),
            time_embed_dim=2,
            training_steps=1,
            microbatch_size=4,
            effective_batch_size=4,
            network_factory=_TimeEchoFactory(),
        ),
    )
    nonlinear = MAMConditionalSolver(
        problem,
        ValueOnlyRunningPotential().as_value_only_cost(),
        nonlinear_config,
        MAMExecutionConfig(microbatch_size=4, effective_batch_size=4),
    )
    assert nonlinear._actor_field is not None
    component = nonlinear._actor_field.initialize(jax.random.PRNGKey(722))
    mixture = mam_bridge_module._single_component_mixture("actor", component)
    nonlinear_actor = nonlinear._actor_fn(mixture, "f")
    assert jnp.allclose(
        nonlinear_actor(states, row_times, endpoints),
        row_times[:, None],
    )
    assert jnp.allclose(
        jax.jit(nonlinear_actor)(states, row_times, endpoints),
        row_times[:, None],
    )


def test_conditional_actor_rejects_short_horizon_off_grid_times():
    problem = replace(
        _problem(dim=1, num_steps=4),
        time_grid=TimeGrid(t0=0.0, t1=1.0e-7, num_steps=4),
    )
    conditional = MAMConditionalSolver(
        problem,
        ValueOnlyRunningPotential().as_value_only_cost(),
        ConditionalMAMConfig(actor_model="affine_reference"),
        MAMExecutionConfig(microbatch_size=2, effective_batch_size=4),
    )
    actor = conditional._actor_fn(conditional._zero_actor_params(jnp.float32), "f")
    states = jnp.zeros((1, 1), dtype=jnp.float32)
    endpoints = jnp.zeros_like(states)
    grid = jnp.asarray(problem.time_grid.times, dtype=jnp.float32)
    off_grid = 0.5 * (grid[0] + grid[1])

    assert jnp.all(jnp.isfinite(actor(states, grid[0], endpoints)))
    with pytest.raises(ValueError, match="exact departure grid"):
        actor(states, off_grid, endpoints)
    assert jnp.all(jnp.isnan(jax.jit(actor)(states, off_grid, endpoints)))


def test_nonlinear_projector_uses_precast_nonbinary_grid_and_checks_eager_steps():
    problem = replace(
        _problem(dim=1, num_steps=128),
        time_grid=TimeGrid(t0=0.001, t1=0.101, num_steps=128),
    )
    config = MarkovProjectionConfig(
        model="nonlinear",
        field_config=MAMFieldConfig(
            hidden_dims=(4,),
            time_embed_dim=2,
            training_steps=1,
            microbatch_size=2,
            effective_batch_size=4,
            network_factory=_TimeEchoFactory(),
        ),
        validation_size=2,
        validation_projections=2,
        validation_replicates=2,
        line_search=(1.0,),
    )
    projector = MarkovProjector(problem, config)
    local_times = jnp.asarray(problem.time_grid.times[:-1], dtype=jnp.float32)
    physical_times = (
        jnp.asarray(
            problem.time_grid.t0 + problem.time_grid.t1,
            dtype=jnp.float32,
        )
        - local_times
    )
    reconstructed = jnp.asarray(problem.time_grid.t1, dtype=jnp.float32) - jnp.arange(
        problem.time_grid.num_steps,
        dtype=jnp.float32,
    ) * jnp.asarray(problem.time_grid.dt, dtype=jnp.float32)
    assert not jnp.array_equal(physical_times, reconstructed)

    batch_size = 2
    states = jnp.zeros((batch_size, problem.time_grid.num_steps, 1), dtype=jnp.float32)
    targets = jnp.broadcast_to(
        physical_times[None, :, None],
        states.shape,
    )
    local_paths = jnp.zeros(
        (batch_size, problem.time_grid.num_steps + 1, 1),
        dtype=jnp.float32,
    )
    controlled = ConditionalMAMResult(
        paths=local_paths[:, ::-1],
        local_paths=local_paths,
        controls=jnp.zeros(
            (batch_size, problem.time_grid.num_steps - 1, 1),
            dtype=jnp.float32,
        ),
        projection_states=states,
        projection_times=physical_times,
        endpoint_predictions=targets,
        actor_params={},
        costate_params={},
        metrics={},
        direction="b",
        exact_conditional_endpoint=True,
    )
    fitted = projector.fit(jax.random.PRNGKey(703), controlled, "b")
    predicted = projector.predict(
        fitted.params,
        jnp.zeros((problem.time_grid.num_steps, 1), dtype=jnp.float32),
        jnp.arange(problem.time_grid.num_steps, dtype=jnp.int32),
        "b",
    )[:, 0]
    assert jnp.array_equal(predicted, physical_times)
    assert float(fitted.loss) == 0.0
    with pytest.raises(ValueError, match="outside the projection grid"):
        projector.predict(
            fitted.params,
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.asarray(problem.time_grid.num_steps),
            "b",
        )


@pytest.mark.parametrize("model", ["affine_reference", "nonlinear"])
def test_projector_step_validation_is_common_and_jit_fail_closed(model):
    problem = _problem(dim=1, num_steps=3)
    projector = MarkovProjector(
        problem,
        MarkovProjectionConfig(
            model=model,
            validation_size=2,
            validation_projections=2,
            validation_replicates=2,
        ),
    )
    params = projector.init_params(jnp.float32)
    states = jnp.zeros((2, 1), dtype=jnp.float32)
    valid = projector.predict(params, states, jnp.asarray([0, 2], dtype=jnp.int32), "f")
    assert valid.shape == (2, 1)
    assert jnp.all(jnp.isfinite(valid))

    with pytest.raises(ValueError, match="outside the projection grid"):
        projector.predict(params, states, jnp.asarray(3, dtype=jnp.int32), "f")
    with pytest.raises(TypeError, match="integer dtype"):
        projector.predict(params, states, jnp.asarray(1.0, dtype=jnp.float32), "f")
    with pytest.raises(ValueError, match="direction"):
        projector.predict(params, states, jnp.asarray(0, dtype=jnp.int32), "sideways")

    traced_predict = jax.jit(lambda step: projector.predict(params, states, step, "f"))
    traced_invalid = traced_predict(jnp.asarray(-1, dtype=jnp.int32))
    assert traced_invalid.shape == (2, 1)
    assert not jnp.any(jnp.isfinite(traced_invalid))


def test_nonlinear_actor_checkpoint_rejects_uncoerced_float64_component():
    problem = _problem(dim=1, num_steps=3)
    field_config = MAMFieldConfig(
        hidden_dims=(4,),
        time_embed_dim=2,
        training_steps=1,
        microbatch_size=2,
        effective_batch_size=4,
    )
    conditional = MAMConditionalSolver(
        problem,
        ValueOnlyRunningPotential().as_value_only_cost(),
        ConditionalMAMConfig(
            actor_model="nonlinear",
            actor_field_config=field_config,
        ),
        MAMExecutionConfig(microbatch_size=2, effective_batch_size=4),
    )
    assert conditional._actor_field is not None
    dataset = MAMActorDataset(
        states=jnp.zeros((4, 1), dtype=jnp.float32),
        times=jnp.linspace(0.0, 0.5, 4, dtype=jnp.float32),
        endpoints=jnp.ones((4, 1), dtype=jnp.float32),
        directions=jnp.ones((4,), dtype=jnp.float32),
        targets=jnp.zeros((4, 1), dtype=jnp.float32),
    )
    state = conditional._actor_field.fit(jax.random.PRNGKey(704), dataset)
    mixture = mam_bridge_module._single_component_mixture("actor", state)
    conditional._validate_actor_checkpoint(mixture)
    bad_state = replace(
        state,
        params=jax.tree_util.tree_map(
            lambda value: np.asarray(value, dtype=np.float64),
            state.params,
        ),
    )
    bad_mixture = replace(mixture, components=(bad_state,))
    with pytest.raises((TypeError, ValueError), match="float32|signature"):
        conditional._validate_actor_checkpoint(bad_mixture)


def test_global_execution_fails_closed_on_unwired_or_nonproduction_modes():
    with pytest.raises(ValueError, match="fixed to float32"):
        MAMExecutionConfig(production_dtype=jnp.float64)
    problem = _problem(dim=1)
    config = replace(
        _tiny_config(),
        execution=MAMExecutionConfig(
            microbatch_size=4,
            effective_batch_size=4,
            allow_two_devices=True,
        ),
    )
    with pytest.raises(NotImplementedError, match="two-device"):
        MAMBridgeSolver(problem, ValueOnlyRunningPotential(), config)

    for invalid_costate, message in (
        (MalliavinAdjointConfig(minimum_remaining_steps=1, matrix_free_labels=False), "matrix"),
        (
            MalliavinAdjointConfig(
                minimum_remaining_steps=1,
                include_control_energy=False,
            ),
            "control energy",
        ),
        (
            MalliavinAdjointConfig(
                minimum_remaining_steps=1,
                anchor_sampling="iid_uniform",
            ),
            "stratified",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            MAMConditionalSolver(
                problem,
                ValueOnlyRunningPotential().as_value_only_cost(),
                ConditionalMAMConfig(costate=invalid_costate),
                MAMExecutionConfig(microbatch_size=4, effective_batch_size=4),
            )


def test_initial_coupling_is_seeded_random_permutation():
    problem = _problem(dim=1)
    backend = _PinnedReferenceConditional(problem)
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        SolverConfig(verbose=0),
        conditional_solver=backend,
    )
    first = solver._initial_pair_cache(jax.random.PRNGKey(2), 16)
    second = solver._initial_pair_cache(jax.random.PRNGKey(2), 16)
    assert jnp.array_equal(first.source, second.source)
    assert jnp.array_equal(first.target, second.target)
    # Independent marginal draws plus an explicit permutation must not silently
    # become the row-wise identity coupling.
    unpermuted_target = jnp.asarray(
        problem.sample_target(jax.random.split(jax.random.PRNGKey(2), 3)[1], 16),
        dtype=first.target.dtype,
    )
    assert not jnp.array_equal(first.target, unpermuted_target)


def test_outer_loop_rejects_and_rolls_back_unpinned_conditional_backend():
    problem = _problem(dim=1)
    backend = _UnpinnedReferenceConditional(problem)
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        SolverConfig(verbose=0),
        conditional_solver=backend,
    )
    with pytest.raises(ValueError, match="exact endpoint pinning"):
        solver.train(jax.random.PRNGKey(6))
    assert backend.calls == []
    assert solver._completed_half_iterations == 0


def test_markov_projection_recovers_finite_affine_endpoint_targets():
    problem = _problem(dim=2, num_steps=3)
    projector = MarkovProjector(
        problem,
        MarkovProjectionConfig(model="affine_reference", ridge=1e-8),
    )
    key = jax.random.PRNGKey(7)
    states = jax.random.normal(key, (64, 3, 2))
    matrix = jnp.array([[1.2, -0.4], [0.3, 0.8]])
    bias = jnp.array([0.5, -0.25])
    targets = states @ matrix + bias
    local_paths = jnp.concatenate([states, targets[:, -1:, :]], axis=1)
    result = ConditionalMAMResult(
        paths=local_paths,
        local_paths=local_paths,
        controls=jnp.zeros((64, 2, 2)),
        projection_states=states,
        projection_times=problem.time_grid.times[:-1],
        endpoint_predictions=targets,
        actor_params={},
        costate_params={},
        metrics={},
        direction="f",
        exact_conditional_endpoint=True,
    )
    fitted = projector.fit(jax.random.PRNGKey(8), result, "f")
    features = jnp.concatenate([states, jnp.ones((64, 3, 1))], axis=-1)
    predictions = jnp.einsum("bnf,nfd->bnd", features, fitted.params)
    assert bool(fitted.finite)
    assert float(fitted.loss) < 1e-10
    assert jnp.allclose(predictions, targets, atol=2e-5, rtol=2e-5)


def test_global_gaussian_zero_cost_smoke_and_honest_status(tmp_path):
    problem = _problem(dim=1, num_steps=3)
    backend = _PinnedReferenceConditional(problem)
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        SolverConfig(verbose=0),
        conditional_solver=backend,
    )
    result = solver.train(jax.random.PRNGKey(0))

    assert backend.calls == ["b", "f"]
    assert result.loss_history.shape == (2,)
    assert jnp.all(jnp.isfinite(result.loss_history))
    assert result.metadata["conditional_endpoints_exact"] is True
    assert result.metadata["global_endpoints_empirically_audited"] is True
    assert result.metadata["matrix_free_costate_labels"] is False
    assert result.metadata["matrix_free_costate_labels_backend_reported"] is False
    assert result.metadata["markov_projection_exact"] is False
    assert result.metadata["markov_projection_semantics"] == (
        "finite_grid_euler_conditional_mean_field_regression"
    )
    # Exact conditional pins are not allowed to masquerade as a passed global
    # marginal test.  The deliberately tiny audit and strict thresholds fail.
    assert result.metadata["global_endpoint_pass"] is False
    assert result.metadata["status"] == "EXPERIMENTAL_GLOBAL_ENDPOINT_UNVERIFIED"
    assert result.converged is False
    assert solver._last_metrics["projection_acceptance"]["independent_cloud_replicates"] == 2
    final_work = result.metadata["work_accounting"]
    assert final_work["structural_counters_certified"] is False
    assert final_work["certified_counters"] is None
    assert final_work["cumulative_certified_counters"] is None
    assert final_work["failed_attempt_work_included"] is False
    assert final_work["external_oracle_billing_certified"] is False
    assert final_work["uncertified_reason"] == "one_or_more_global_half_iterations_uncertified"
    for audit in result.diagnostics.metadata["audit_history"]:
        half_work = audit["work_accounting"]
        assert half_work["structural_counters_certified"] is False
        assert half_work["certified_counters"] is None
        assert half_work["cumulative_certified_counters"] is None
        assert half_work["uncertified_reason"] == "conditional_backend_is_not_exact_builtin_mam"
        assert half_work["schema_version"] == 2
        assert half_work["derivation"]["conditional"] is None

    forward = solver.sample(jax.random.PRNGKey(11), 5)
    backward = solver.sample_backward(jax.random.PRNGKey(12), 5)
    assert forward.paths.shape == (5, 4, 1)
    assert backward.paths.shape == (5, 4, 1)
    assert jnp.all(jnp.isfinite(forward.paths))
    assert jnp.all(jnp.isfinite(backward.paths))
    assert jnp.array_equal(forward.paths[:, 0], forward.source_samples)
    assert jnp.array_equal(backward.paths[:, -1], backward.target_samples)

    checkpoint = tmp_path / "injected-uncertified.pkl"
    solver.save_checkpoint(
        checkpoint,
        params=result.params,
        step=2,
        loss_history=list(np.asarray(result.loss_history)),
        metrics=solver._last_metrics,
        metadata={"algorithm": "MAM_GSBM_EXPERIMENTAL", "num_half_iterations": 2},
    )
    with checkpoint.open("rb") as handle:
        flipped = pickle.load(handle)
    flipped_state = flipped["solver_state"]
    flipped_record = flipped_state["audit_history"][-1]["work_accounting"]
    flipped_record["structural_counters_certified"] = True
    flipped_record["certified_counters"] = MAMWorkCounters.zero().to_state()
    flipped_record["certified_fields"] = list(mam_bridge_module._CERTIFIED_WORK_FIELDS)
    flipped_record["uncertified_reason"] = None
    flipped_state["last_metrics"]["work_accounting"] = copy.deepcopy(flipped_record)
    flipped_state["last_metrics"]["audit"]["work_accounting"] = copy.deepcopy(flipped_record)
    flipped_path = tmp_path / "injected-forged-certified.pkl"
    with flipped_path.open("wb") as handle:
        pickle.dump(flipped, handle, protocol=pickle.HIGHEST_PROTOCOL)
    fresh = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    with pytest.raises(ValueError, match="only the exact built-in"):
        fresh.load_checkpoint(flipped_path)


def test_conditional_projection_cache_is_recomputed_and_cannot_be_poisoned():
    problem = _problem(dim=1, num_steps=3)
    backend = _PinnedReferenceConditional(problem)
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        SolverConfig(verbose=0),
        conditional_solver=backend,
    )
    pair_key, solve_key = jax.random.split(jax.random.PRNGKey(301))
    pairs = solver._initial_pair_cache(pair_key, 8)
    result = backend.solve(solve_key, pairs, "f")
    solver._validate_conditional_result(result, pairs, "f")

    with pytest.raises(ValueError, match="projection states"):
        solver._validate_conditional_result(
            replace(
                result,
                projection_states=result.projection_states.at[0, 0, 0].add(0.1),
            ),
            pairs,
            "f",
        )
    with pytest.raises(ValueError, match="projection times"):
        solver._validate_conditional_result(
            replace(
                result,
                projection_times=result.projection_times.at[0].add(0.1),
            ),
            pairs,
            "f",
        )
    with pytest.raises(ValueError, match="endpoint predictions"):
        solver._validate_conditional_result(
            replace(
                result,
                endpoint_predictions=result.endpoint_predictions.at[0, 0, 0].add(0.1),
            ),
            pairs,
            "f",
        )

    zero_pairs = EndpointPairBatch(
        source=jnp.zeros((8, 1), dtype=jnp.float32),
        target=jnp.zeros((8, 1), dtype=jnp.float32),
    )
    zero_result = backend.solve(solve_key, zero_pairs, "f")
    solver._validate_conditional_result(zero_result, zero_pairs, "f")
    with pytest.raises(ValueError, match="endpoint predictions"):
        solver._validate_conditional_result(
            replace(
                zero_result,
                endpoint_predictions=zero_result.endpoint_predictions.at[0, 0, 0].add(1.0e-7),
            ),
            zero_pairs,
            "f",
        )


def test_conditional_actor_targets_reject_false_finite_mask():
    problem = _problem(dim=1, num_steps=3)
    conditional = MAMConditionalSolver(
        problem,
        ValueOnlyRunningPotential().as_value_only_cost(),
        ConditionalMAMConfig(actor_model="affine_reference"),
        MAMExecutionConfig(microbatch_size=2, effective_batch_size=4),
    )
    batch_size = 2
    endpoint = jnp.zeros((batch_size, 1), dtype=jnp.float32)
    rollout = SimpleNamespace(
        states=jnp.zeros((batch_size, 4, 1), dtype=jnp.float32),
        controls=jnp.zeros((batch_size, 2, 1), dtype=jnp.float32),
    )

    class InvalidInner:
        def make_action_target_batch(self, key, states, times, context, **kwargs):
            del key, times, context, kwargs
            rows = states.shape[0]
            target = jnp.zeros((rows, 1), dtype=states.dtype)
            return ActionTargetBatch(
                target=target,
                continuation_component=target,
                arrival_component=target,
                mean_state=target,
                innovation=jnp.zeros((rows, 1, 1), dtype=states.dtype),
                finite=jnp.zeros((rows,), dtype=bool),
                physical_oracle_queries=jnp.asarray(0, dtype=jnp.int32),
                estimator=jnp.asarray(0, dtype=jnp.int32),
            )

    with pytest.raises(FloatingPointError, match="actor target"):
        conditional._actor_targets(
            jax.random.PRNGKey(722),
            rollout,
            endpoint,
            InvalidInner(),
            {},
            ValueOnlyRunningPotential().as_value_only_cost(),
        )


def test_public_solution_samples_the_same_audited_discrete_process():
    from schrodinger_bridge.integrators import Heun

    problem = _problem(dim=1, num_steps=3)
    backend = _PinnedReferenceConditional(problem)
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        SolverConfig(verbose=0),
        conditional_solver=backend,
    )
    solution = solver.solve(jax.random.PRNGKey(41))
    sample_key = jax.random.PRNGKey(42)
    direct_forward = solver.sample(sample_key, 6)
    public_forward = solution.sample_trajectories(sample_key, 6)
    direct_backward = solver.sample_backward(sample_key, 6)
    public_backward = solution.as_process().sample_paths(
        sample_key,
        6,
        direction="backward",
    )
    assert jnp.allclose(direct_forward.paths, public_forward.paths, atol=2e-7, rtol=2e-7)
    assert jnp.allclose(direct_backward.paths, public_backward.paths, atol=2e-7, rtol=2e-7)
    with pytest.raises(ValueError, match="exact|Euler"):
        solution.sample_trajectories(sample_key, 2, integrator=Heun())
    with pytest.raises(ValueError, match="continuous|diffrax"):
        solution.as_process(backend="diffrax")
    process = solution.as_process()
    with pytest.raises(ValueError, match="sub-grid"):
        process.rollout_from(sample_key, direct_forward.paths[:, 0], t0=0.2, t1=0.8)
    with pytest.raises(ValueError, match="Euler"):
        process.rollout_from(
            sample_key,
            direct_forward.paths[:, 0],
            integrator=Heun(),
        )
    with pytest.raises(ValueError, match="disabled"):
        process.reverse()


def test_public_sampling_and_drift_fail_closed_on_invalid_starts_params_and_paths():
    problem = _problem(dim=1, num_steps=3)
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    params = solver.init_params(jax.random.PRNGKey(710))
    valid_start = jnp.zeros((2, 1), dtype=jnp.float32)
    valid = solver.sample(
        jax.random.PRNGKey(711),
        2,
        params=params,
        x0=valid_start,
    )
    assert jnp.all(jnp.isfinite(valid.paths))

    with pytest.raises(ValueError, match="shape"):
        solver.sample(
            jax.random.PRNGKey(711),
            2,
            params=params,
            x0=jnp.zeros((1, 1), dtype=jnp.float32),
        )
    with pytest.raises(TypeError, match="dtype"):
        solver.sample(
            jax.random.PRNGKey(711),
            2,
            params=params,
            x0=np.zeros((2, 1), dtype=np.float64),
        )
    with pytest.raises(FloatingPointError, match="nonfinite"):
        solver.sample(
            jax.random.PRNGKey(711),
            2,
            params=params,
            x0=jnp.asarray([[jnp.nan], [0.0]], dtype=jnp.float32),
        )

    bad_params = dict(params)
    bad_params["F"] = bad_params["F"].at[0, 0, 0].set(jnp.nan)
    with pytest.raises(FloatingPointError, match="projection parameters"):
        solver.sample(jax.random.PRNGKey(711), 2, params=bad_params, x0=valid_start)
    with pytest.raises(FloatingPointError, match="projection parameters"):
        solver.extract_drift(bad_params)

    solution = MAMBridgeSolution(
        problem=problem,
        solver_type=solver.solver_type,
        params=params,
        representation=solver.representation_type,
        initial_sampler=solver._sample_source,
        terminal_sampler=solver._sample_target,
        projector=solver.projector,
    )
    solution._integrator = solver.integrator
    solution._forward_drift = solver.extract_drift(params)
    solution._backward_drift = solver.extract_backward_drift(params)
    process = solution.as_process()
    with pytest.raises(FloatingPointError, match="nonfinite"):
        process.sample_paths(
            jax.random.PRNGKey(712),
            2,
            x0=jnp.asarray([[0.0], [jnp.inf]], dtype=jnp.float32),
        )
    process.params = bad_params
    with pytest.raises(FloatingPointError, match="parameters"):
        process.sample_paths(jax.random.PRNGKey(712), 2, x0=valid_start)

    overflow_params = dict(params)
    maximum = jnp.asarray(jnp.finfo(jnp.float32).max, dtype=jnp.float32)
    overflow_params["F"] = overflow_params["F"].at[:, 0, 0].set(maximum)
    with pytest.raises(FloatingPointError, match="sampled paths"):
        solver.sample(
            jax.random.PRNGKey(713),
            2,
            params=overflow_params,
            x0=jnp.ones((2, 1), dtype=jnp.float32),
        )


def test_nonlinear_projector_samples_in_public_runtime_and_validates_checkpoint(tmp_path):
    problem = _problem(dim=1, num_steps=3)
    projection = MarkovProjectionConfig(
        model="nonlinear",
        field_config=MAMFieldConfig(
            hidden_dims=(8,),
            time_embed_dim=4,
            learning_rate=5e-3,
            training_steps=12,
            microbatch_size=99,
            effective_batch_size=99,
        ),
        maximum_components=4,
        validation_size=8,
        validation_projections=2,
        validation_replicates=2,
        line_search=(1.0,),
    )
    config = replace(_tiny_config(), projection=projection)
    backend = _PinnedReferenceConditional(problem)
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        config,
        SolverConfig(verbose=0),
        conditional_solver=backend,
    )
    # The standalone field config's deliberately incompatible 99/99 batch is
    # overridden by the solver's resolved execution contract.
    assert solver.projector._field.config.microbatch_size == 4
    assert solver.projector._field.config.effective_batch_size == 4

    pair_key, forward_key, backward_key, fit_f_key, fit_b_key = jax.random.split(
        jax.random.PRNGKey(411), 5
    )
    pairs = solver._initial_pair_cache(pair_key, 16)
    forward_conditional = backend.solve(forward_key, pairs, "f")
    backward_conditional = backend.solve(backward_key, pairs, "b")
    solver._validate_conditional_result(forward_conditional, pairs, "f")
    solver._validate_conditional_result(backward_conditional, pairs, "b")
    fitted_forward = solver.projector.fit(fit_f_key, forward_conditional, "f")
    fitted_backward = solver.projector.fit(fit_b_key, backward_conditional, "b")
    assert bool(fitted_forward.finite) and bool(fitted_backward.finite)
    assert len(fitted_forward.params.components) == 1
    assert len(fitted_backward.params.components) == 1
    assert fitted_forward.params.components[0].params

    params = solver.init_params(jax.random.PRNGKey(412))
    params["F"] = fitted_forward.params
    params["B"] = fitted_backward.params
    solver.projector.validate_params(params["F"])
    solver.projector.validate_params(params["B"])
    forward_component = fitted_forward.params.components[0]
    bad_forward_component = replace(
        forward_component,
        optimizer=replace(
            forward_component.optimizer,
            m=jax.tree_util.tree_map(
                lambda value: np.asarray(value, dtype=np.float64),
                forward_component.optimizer.m,
            ),
        ),
    )
    bad_forward = replace(
        fitted_forward.params,
        components=(bad_forward_component,),
    )
    with pytest.raises(TypeError, match="float32"):
        solver.projector.validate_params(bad_forward)
    solver._params = params
    solver._is_trained = True
    sample_key = jax.random.PRNGKey(413)
    direct = solver.sample(sample_key, 5, params=params)
    runtime = MAMBridgeSolution(
        problem=problem,
        solver_type=solver.solver_type,
        params=params,
        representation=solver.representation_type,
        initial_sampler=solver._sample_source,
        terminal_sampler=solver._sample_target,
        projector=solver.projector,
    )
    runtime._integrator = solver.integrator
    runtime._forward_drift = solver.extract_drift(params)
    runtime._backward_drift = solver.extract_backward_drift(params)
    public = runtime.sample_trajectories(sample_key, 5)
    assert public.paths.dtype == jnp.float32
    assert jnp.allclose(public.paths, direct.paths, atol=2e-6, rtol=2e-6)

    # Use a fresh injected backend in the checkpointing solver so the
    # zero-progress conditional state is internally consistent.
    checkpoint_solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        config,
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    checkpoint_solver._params = params
    checkpoint = tmp_path / "nonlinear_projection.pkl"
    checkpoint_solver.save_checkpoint(
        checkpoint,
        params=params,
        step=0,
        metadata={"algorithm": "MAM_GSBM_EXPERIMENTAL", "num_half_iterations": 0},
    )
    restored = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        config,
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    restored.load_checkpoint(checkpoint)
    restored.projector.validate_params(restored._params["F"])
    restored_paths = restored.sample(sample_key, 5).paths
    assert jnp.allclose(restored_paths, direct.paths, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_public_runtime_uses_exact_projection_heads_on_long_grid(direction):
    problem = _problem(dim=1, num_steps=100)
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    params = solver.init_params(jax.random.PRNGKey(50))
    # Distinct time-head biases make an off-by-one lookup immediately visible.
    head_biases = jnp.arange(100, dtype=params["F"].dtype)[:, None]
    params["F"] = params["F"].at[:, -1, :].set(head_biases)
    params["B"] = params["B"].at[:, -1, :].set(-head_biases)
    solver._params = params
    solver._is_trained = True

    from schrodinger_bridge.solvers.mam_bridge import MAMBridgeSolution

    runtime = MAMBridgeSolution(
        problem=problem,
        solver_type=solver.solver_type,
        params=params,
        representation=solver.representation_type,
        initial_sampler=solver._sample_source,
        terminal_sampler=solver._sample_target,
    )
    runtime._integrator = solver.integrator
    runtime._forward_drift = solver.extract_drift(params)
    runtime._backward_drift = solver.extract_backward_drift(params)
    sample_key = jax.random.PRNGKey(51)
    if direction == "forward":
        direct = solver.sample(sample_key, 3, params=params)
        public = runtime.sample_trajectories(sample_key, 3)
    else:
        direct = solver.sample_backward(sample_key, 3, params=params)
        public = runtime.as_process().sample_paths(sample_key, 3, direction="backward")
    assert jnp.allclose(direct.paths, public.paths, atol=3e-5, rtol=3e-6)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_extracted_drift_rejects_off_grid_times_eager_and_poison_jit(direction):
    problem = _problem(dim=1, num_steps=4)
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    params = solver.init_params(jax.random.PRNGKey(723))
    drift = solver.extract_drift(params, direction=direction)
    state = jnp.zeros((1,), dtype=jnp.float32)
    local_grid = jnp.asarray(problem.time_grid.times[:-1], dtype=jnp.float32)
    grid = (
        local_grid
        if direction == "forward"
        else jnp.asarray(problem.time_grid.t0 + problem.time_grid.t1, dtype=jnp.float32)
        - local_grid
    )
    values = jax.vmap(lambda time: jax.jit(drift)(state, time))(grid)
    assert jnp.all(jnp.isfinite(values))

    off_grid = 0.5 * (grid[0] + grid[1])
    outside = (
        jnp.asarray(problem.time_grid.t1, dtype=jnp.float32)
        if direction == "forward"
        else jnp.asarray(problem.time_grid.t0, dtype=jnp.float32)
    )
    for invalid_time in (off_grid, outside, jnp.asarray(jnp.nan, dtype=jnp.float32)):
        with pytest.raises(ValueError, match="outside its exact grid"):
            drift(state, invalid_time)
        assert jnp.all(jnp.isnan(jax.jit(drift)(state, invalid_time)))
    with pytest.raises(ValueError, match="scalar"):
        drift(state, grid[:2])


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_extracted_drift_rejects_short_horizon_off_grid_times(direction):
    problem = replace(
        _problem(dim=1, num_steps=4),
        time_grid=TimeGrid(t0=0.0, t1=1.0e-7, num_steps=4),
    )
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    params = solver.init_params(jax.random.PRNGKey(724))
    drift = solver.extract_drift(params, direction=direction)
    state = jnp.zeros((1,), dtype=jnp.float32)
    local_grid = jnp.asarray(problem.time_grid.times[:-1], dtype=jnp.float32)
    grid = (
        local_grid
        if direction == "forward"
        else jnp.asarray(problem.time_grid.t0 + problem.time_grid.t1, dtype=jnp.float32)
        - local_grid
    )
    off_grid = 0.5 * (grid[0] + grid[1])

    assert jnp.all(jnp.isfinite(drift(state, grid[0])))
    with pytest.raises(ValueError, match="outside its exact grid"):
        drift(state, off_grid)
    assert jnp.all(jnp.isnan(jax.jit(drift)(state, off_grid)))


def test_projection_objective_charges_the_last_global_control():
    problem = _problem(dim=1, num_steps=3)
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    current = solver.init_params(jax.random.PRNGKey(52))
    candidate = dict(current)
    candidate["F"] = candidate["F"].at[-1, -1, 0].set(5.0)
    key = jax.random.PRNGKey(53)
    current_paths = solver._sample_direction(key, 8, current, "f").paths
    candidate_paths = solver._sample_direction(key, 8, candidate, "f").paths
    current_objective = solver._projection_objective_samples(key, current, "f")
    candidate_objective = solver._projection_objective_samples(key, candidate, "f")
    assert not jnp.allclose(current_paths[:, -1], candidate_paths[:, -1])
    assert not jnp.allclose(candidate_objective, current_objective)


def test_partial_running_cost_arguments_change_scientific_fingerprint():
    def scaled_cost(states, times, context, *, scale):
        del times, context
        return scale * jnp.sum(states**2, axis=-1)

    first = partial(scaled_cost, scale=1.0)
    second = partial(scaled_cost, scale=9.0)
    assert MAMBridgeSolver._callable_fingerprint(first) != MAMBridgeSolver._callable_fingerprint(
        second
    )


def test_referenced_global_cost_state_changes_scientific_fingerprint():
    global _GLOBAL_RUNNING_SCALE

    original = _GLOBAL_RUNNING_SCALE
    try:
        _GLOBAL_RUNNING_SCALE = 1.0
        first_problem = _problem(dim=1, num_steps=3)
        first = MAMBridgeSolver(
            first_problem,
            ValueOnlyRunningPotential(_global_scale_running_cost, identifier="global-scale"),
            _tiny_config(),
            conditional_solver=_PinnedReferenceConditional(first_problem),
        )
        first_hash = first._scientific_fingerprints()["cost_sha256"]
        _GLOBAL_RUNNING_SCALE = 2.0
        second_problem = _problem(dim=1, num_steps=3)
        second = MAMBridgeSolver(
            second_problem,
            ValueOnlyRunningPotential(_global_scale_running_cost, identifier="global-scale"),
            _tiny_config(),
            conditional_solver=_PinnedReferenceConditional(second_problem),
        )
        assert second._scientific_fingerprints()["cost_sha256"] != first_hash
    finally:
        _GLOBAL_RUNNING_SCALE = original


def test_sampler_class_state_and_implementation_are_problem_fingerprinted():
    original = _ClassShiftGaussian.class_shift
    try:
        _ClassShiftGaussian.class_shift = 0.0
        base = _problem(dim=1, num_steps=3)
        first_problem = replace(
            base,
            source=_ClassShiftGaussian(mean=-jnp.ones((1,)), cov=0.15, dim=1),
        )
        first = MAMBridgeSolver(
            first_problem,
            ValueOnlyRunningPotential(),
            _tiny_config(),
            conditional_solver=_PinnedReferenceConditional(first_problem),
        )
        first_hash = first._scientific_fingerprints()["problem_sha256"]
        _ClassShiftGaussian.class_shift = 3.0
        second_problem = replace(
            base,
            source=_ClassShiftGaussian(mean=-jnp.ones((1,)), cov=0.15, dim=1),
        )
        second = MAMBridgeSolver(
            second_problem,
            ValueOnlyRunningPotential(),
            _tiny_config(),
            conditional_solver=_PinnedReferenceConditional(second_problem),
        )
        assert second._scientific_fingerprints()["problem_sha256"] != first_hash
    finally:
        _ClassShiftGaussian.class_shift = original


def test_opaque_referenced_global_fingerprint_fails_closed():
    with pytest.raises(TypeError, match="unsupported opaque state"):
        MAMBridgeSolver._callable_fingerprint(_opaque_global_running_cost)


def test_bound_running_cost_fingerprint_binds_instance_state_and_checkpoint(tmp_path):
    problem = _problem(dim=1, num_steps=3)
    first_cost = _ThresholdRunningCost(0.25)
    equivalent_cost = _ThresholdRunningCost(0.25)
    changed_cost = _ThresholdRunningCost(0.75)
    assert MAMBridgeSolver._callable_fingerprint(
        first_cost.value
    ) == MAMBridgeSolver._callable_fingerprint(equivalent_cost.value)
    assert MAMBridgeSolver._callable_fingerprint(
        first_cost.value
    ) != MAMBridgeSolver._callable_fingerprint(changed_cost.value)

    first = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(first_cost.value, identifier="threshold"),
        _tiny_config(),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    equivalent = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(equivalent_cost.value, identifier="threshold"),
        _tiny_config(),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    changed = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(changed_cost.value, identifier="threshold"),
        _tiny_config(),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    assert first._scientific_fingerprints() == equivalent._scientific_fingerprints()
    assert (
        first._scientific_fingerprints()["cost_sha256"]
        != changed._scientific_fingerprints()["cost_sha256"]
    )
    params = first.init_params(jax.random.PRNGKey(720))
    checkpoint = tmp_path / "bound-cost.pkl"
    first.save_checkpoint(
        checkpoint,
        params=params,
        step=0,
        metadata={"algorithm": "MAM_GSBM_EXPERIMENTAL", "num_half_iterations": 0},
    )
    equivalent.load_checkpoint(checkpoint)
    assert not equivalent._is_trained
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        changed.load_checkpoint(checkpoint)


def test_injected_conditional_instance_fingerprint_binds_checkpoint(tmp_path):
    problem = _problem(dim=1, num_steps=3)
    first = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        conditional_solver=_PinnedReferenceConditional(problem, "variant-a"),
    )
    equivalent = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        conditional_solver=_PinnedReferenceConditional(problem, "variant-a"),
    )
    changed = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        conditional_solver=_PinnedReferenceConditional(problem, "variant-b"),
    )
    assert first._scientific_fingerprints() == equivalent._scientific_fingerprints()
    assert (
        first._scientific_fingerprints()["conditional_backend_instance_sha256"]
        != changed._scientific_fingerprints()["conditional_backend_instance_sha256"]
    )
    checkpoint = tmp_path / "conditional-instance-fingerprint.pkl"
    first.save_checkpoint(checkpoint, params=first.init_params(jax.random.PRNGKey(721)), step=0)
    equivalent.load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        changed.load_checkpoint(checkpoint)


def test_injected_conditional_without_stable_fingerprint_fails_closed():
    problem = _problem(dim=1, num_steps=3)
    backend = _PinnedReferenceConditional(problem)
    backend.scientific_fingerprint = None
    with pytest.raises(TypeError, match="scientific_fingerprint"):
        MAMBridgeSolver(
            problem,
            ValueOnlyRunningPotential(),
            _tiny_config(),
            conditional_solver=backend,
        )


def test_nested_code_callback_fingerprint_is_stable_across_fresh_interpreter():
    expected = MAMBridgeSolver._callable_fingerprint(_nested_fingerprint_callback)
    test_path = Path(__file__).resolve()
    module_name = _nested_fingerprint_callback.__module__
    script = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location({module_name!r}, {str(test_path)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(module.MAMBridgeSolver._callable_fingerprint(module._nested_fingerprint_callback))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=test_path.parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip().splitlines()[-1] == expected


def test_equivalent_fresh_network_factories_have_stable_config_fingerprint():
    problem = _problem(dim=1, num_steps=3)

    def make_config():
        conditional = ConditionalMAMConfig(
            costate=MalliavinAdjointConfig(network_factory=_FreshScalarFactory()),
            value_critic=ValueCriticConfig(network_factory=_FreshScalarFactory()),
        )
        return replace(_tiny_config(), conditional=conditional)

    first = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        make_config(),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    second = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        make_config(),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    assert (
        first._scientific_fingerprints()["config_sha256"]
        == second._scientific_fingerprints()["config_sha256"]
    )


def test_equivalent_fresh_mixture_problems_have_stable_problem_fingerprint():
    def make_problem():
        source = MixtureDistribution(
            [
                GaussianDistribution(mean=jnp.asarray([-1.0]), cov=0.2, dim=1),
                GaussianDistribution(mean=jnp.asarray([0.5]), cov=0.3, dim=1),
            ],
            weights=jnp.asarray([0.4, 0.6]),
        )
        target = MixtureDistribution(
            [
                GaussianDistribution(mean=jnp.asarray([1.0]), cov=0.2, dim=1),
                GaussianDistribution(mean=jnp.asarray([-0.5]), cov=0.3, dim=1),
            ],
            weights=jnp.asarray([0.4, 0.6]),
        )
        return SBProblem(
            reference=BrownianMotion(sigma=0.35, dim=1),
            source=source,
            target=target,
            time_grid=TimeGrid(num_steps=3),
            name="stable mixture fingerprint",
        )

    first_problem = make_problem()
    second_problem = make_problem()
    first = MAMBridgeSolver(
        first_problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        conditional_solver=_PinnedReferenceConditional(first_problem),
    )
    second = MAMBridgeSolver(
        second_problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        conditional_solver=_PinnedReferenceConditional(second_problem),
    )
    assert (
        first._scientific_fingerprints()["problem_sha256"]
        == second._scientific_fingerprints()["problem_sha256"]
    )


def test_equivalent_nested_mixtures_have_stable_problem_fingerprint():
    def nested_mixture(shift):
        inner = MixtureDistribution(
            [
                GaussianDistribution(mean=jnp.asarray([shift - 0.5]), cov=0.2, dim=1),
                GaussianDistribution(mean=jnp.asarray([shift + 0.5]), cov=0.2, dim=1),
            ]
        )
        return MixtureDistribution(
            [inner, GaussianDistribution(mean=jnp.asarray([shift]), cov=0.4, dim=1)]
        )

    def make_problem():
        return SBProblem(
            reference=BrownianMotion(sigma=0.35, dim=1),
            source=nested_mixture(-1.0),
            target=nested_mixture(1.0),
            time_grid=TimeGrid(num_steps=3),
            name="stable nested mixture fingerprint",
        )

    first_problem = make_problem()
    second_problem = make_problem()
    first = MAMBridgeSolver(
        first_problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        conditional_solver=_PinnedReferenceConditional(first_problem),
    )
    second = MAMBridgeSolver(
        second_problem,
        ValueOnlyRunningPotential(),
        _tiny_config(),
        conditional_solver=_PinnedReferenceConditional(second_problem),
    )
    assert (
        first._scientific_fingerprints()["problem_sha256"]
        == second._scientific_fingerprints()["problem_sha256"]
    )


def test_non_brownian_reference_fails_closed():
    from schrodinger_bridge.core.problem import OrnsteinUhlenbeck

    problem = SBProblem(
        reference=OrnsteinUhlenbeck(dim=1),
        source=GaussianDistribution(dim=1),
        target=GaussianDistribution(dim=1),
        time_grid=TimeGrid(num_steps=3),
    )
    with pytest.raises(ValueError, match="BrownianMotion"):
        MAMBridgeSolver(problem, ValueOnlyRunningPotential(), _tiny_config())


def test_real_conditional_solver_uses_accumulation_and_exact_pins():
    problem = replace(
        _problem(dim=1, num_steps=3),
        time_grid=TimeGrid(t0=0.001, t1=0.101, num_steps=3),
    )

    def hard_running(states, times, context):
        del times, context
        return (states[:, 0] > 0.0).astype(states.dtype)

    potential = ValueOnlyRunningPotential(hard_running, identifier="halfspace")
    conditional_config = ConditionalMAMConfig(
        actor_model="affine_reference",
        costate=MalliavinAdjointConfig(
            hidden_dims=(8,),
            time_embed_dim=4,
            training_steps=1,
            batch_size=4,
            minimum_remaining_steps=1,
        ),
        costate_steps=1,
        batch_size=4,
        acceptance_size=4,
        direct_score_diagnostic_size=4,
        policy_iterations=1,
        line_search=(0.125,),
        value_critic=ValueCriticConfig(
            hidden_dims=(8,),
            time_embed_dim=4,
            training_steps=1,
            microbatch_size=2,
            effective_batch_size=4,
        ),
    )
    execution = MAMExecutionConfig(
        microbatch_size=2,
        effective_batch_size=4,
        allow_two_devices=False,
    )
    solver = MAMConditionalSolver(
        problem,
        potential.as_value_only_cost(),
        conditional_config,
        execution,
    )
    source_key, target_key, solve_key = jax.random.split(jax.random.PRNGKey(19), 3)
    pairs = EndpointPairBatch(
        problem.sample_source(source_key, 8),
        problem.sample_target(target_key, 8),
    )
    result = solver.solve(solve_key, pairs, "b")
    assert result.exact_conditional_endpoint
    assert result.local_paths.dtype == jnp.float32
    assert jnp.array_equal(result.local_paths[:, -1], pairs.source.astype(jnp.float32))
    local_times = jnp.asarray(problem.time_grid.times[:-1], dtype=jnp.float32)
    expected_backward_times = (
        jnp.asarray(
            problem.time_grid.t0 + problem.time_grid.t1,
            dtype=jnp.float32,
        )
        - local_times
    )
    assert jnp.array_equal(result.projection_times, expected_backward_times)
    assert result.metrics["uses_arrival_correction"] is True
    assert result.metrics["matrix_free_costate_labels"] is True
    assert result.metrics["gradient_accumulation_steps"] == 2
    assert result.metrics["actor_iterations_completed"] == 1
    assert result.metrics["value_critic"]["trained"] is True
    assert result.metrics["value_critic"]["cross_fitted"] is True
    assert result.metrics["value_critic"]["used_for_costate_centering"] is False
    assert result.metrics["value_critic"]["costate_centering"] == "stopped_anchor_hard_value"
    assert result.metrics["value_critic"]["rows"] == 4
    assert result.metrics["direct_action_score"]["tangent_free"] is True
    assert result.metrics["direct_action_score"]["used_for_actor_selection"] is False
    assert result.metrics["direct_action_score"]["physical_value_oracle_queries"] == 32
    assert result.metrics["actor_acceptance_history"][0][
        "selection_and_confirmation_streams_disjoint"
    ]
    assert result.metrics["actor_acceptance_history"][0]["acceptance_independence_scope"] == (
        "conditional_on_fixed_endpoint_pair_cache"
    )
    assert (
        result.metrics["output_actor_fingerprint"] == result.metrics["costate_policy_fingerprint"]
    )
    assert result.metrics["actor_costate_policy_aligned"] is True
    assert result.metrics["final_costate_refresh_executed"] == (
        result.metrics["output_actor_fingerprint"]
        != result.metrics["pre_refresh_costate_policy_fingerprint"]
    )
    assert result.metrics["value_critic_matches_output_actor"] == (
        result.metrics["value_critic_policy_fingerprint"]
        == result.metrics["output_actor_fingerprint"]
    )
    expected_work = completed_conditional_solve_work(
        num_steps=problem.time_grid.num_steps,
        effective_batch_size=execution.effective_batch_size,
        costate_steps=conditional_config.costate_steps,
        value_critic_training_steps=conditional_config.value_critic.training_steps,
        actor_field_training_steps=0,
        direct_score_diagnostic_size=conditional_config.direct_score_diagnostic_size,
        acceptance_size=conditional_config.acceptance_size,
        line_search_candidates=len(conditional_config.line_search),
        pair_batch_size=pairs.batch_size,
        policy_iterations_completed=1,
        actor_confirmation_executed=(
            result.metrics["actor_acceptance_history"][0]["confirmation"] is not None,
        ),
        actor_update_accepted=(
            bool(result.metrics["actor_acceptance_history"][0]["actor_update_accepted"]),
        ),
        final_costate_refresh_executed=result.metrics["final_costate_refresh_executed"],
        running_cost_oracle_present=True,
    )
    assert result.certified_work_counters == expected_work
    work_record = result.metrics["work_accounting"]
    assert work_record["structural_counters_certified"] is True
    assert work_record["certified_counters"] == expected_work.to_state()
    assert work_record["cumulative_certified_counters"] is None
    assert work_record["unmeasured_fields"] == [
        "compile_time_ns",
        "steady_state_time_ns",
        "peak_device_memory_bytes",
    ]
    validator = MAMBridgeSolver(
        problem,
        potential,
        replace(
            _tiny_config(),
            conditional=conditional_config,
            execution=execution,
        ),
        conditional_solver=solver,
    )
    validator._validate_conditional_result(result, pairs, "b")
    expected_updates = {
        "b": (
            conditional_config.policy_iterations
            + int(result.metrics["final_costate_refresh_executed"])
        )
        * conditional_config.costate_steps
    }
    solver.validate_checkpoint_progress(expected_updates)
    saved_costate = solver._costate_params.pop("b")
    with pytest.raises(ValueError, match="directions disagree"):
        solver.validate_checkpoint_progress(expected_updates)
    solver._costate_params["b"] = saved_costate
    solver._costate_params["b"] = jnp.asarray(0.0, dtype=jnp.float32)
    with pytest.raises(ValueError, match="costate parameters tree mismatch"):
        solver.validate_checkpoint_progress(expected_updates)
    solver._costate_params["b"] = saved_costate
    saved_critic = solver._value_critic_state["b"]
    solver._value_critic_state["b"] = replace(
        saved_critic,
        row_fold=1 - saved_critic.row_fold,
    )
    with pytest.raises(ValueError, match="assignment fingerprint"):
        solver.validate_checkpoint_progress(expected_updates)
    solver._value_critic_state["b"] = saved_critic


@pytest.mark.parametrize("accepted", [False, True])
def test_final_costate_refresh_tracks_output_actor_on_disjoint_key(monkeypatch, accepted):
    problem = _problem(dim=1, num_steps=3)
    config = ConditionalMAMConfig(
        actor_model="affine_reference",
        costate=MalliavinAdjointConfig(
            hidden_dims=(8,),
            time_embed_dim=4,
            training_steps=1,
            batch_size=4,
            minimum_remaining_steps=1,
        ),
        costate_steps=1,
        batch_size=4,
        acceptance_size=4,
        direct_score_diagnostic_size=4,
        policy_iterations=1,
        line_search=(0.5,),
        value_critic=ValueCriticConfig(
            hidden_dims=(8,),
            time_embed_dim=4,
            training_steps=1,
            microbatch_size=2,
            effective_batch_size=4,
        ),
    )
    execution = MAMExecutionConfig(microbatch_size=2, effective_batch_size=4)
    solver = MAMConditionalSolver(
        problem,
        ValueOnlyRunningPotential().as_value_only_cost(),
        config,
        execution,
    )
    source_key, target_key, solve_key = jax.random.split(jax.random.PRNGKey(725), 3)
    pairs = EndpointPairBatch(
        problem.sample_source(source_key, 8),
        problem.sample_target(target_key, 8),
    )
    original_train_costate = solver._train_costate
    costate_keys = []

    def recorded_train_costate(key, endpoint_pairs, direction, actor_params):
        costate_keys.append(np.asarray(jax.device_get(key)).copy())
        return original_train_costate(key, endpoint_pairs, direction, actor_params)

    def changed_actor_fit(
        key,
        start,
        endpoint,
        direction,
        current,
        inner,
        costate_params,
        directional_cost,
    ):
        del key, start, endpoint, direction, inner, costate_params, directional_cost
        return jnp.asarray(current).at[:, -1, :].add(0.25), jnp.asarray(1.0)

    def forced_accept(
        key,
        start,
        endpoint,
        direction,
        current,
        candidate,
        directional_cost,
        critic_baseline_fn,
    ):
        del key, start, endpoint, direction, directional_cost, critic_baseline_fn
        return (candidate if accepted else current), {
            "actor_update_accepted": accepted,
            "accepted_step_size": 0.5 if accepted else 0.0,
            "line_search": [],
            "confirmation": {"accepted": True} if accepted else None,
            "acceptance_independent_of_actor_fit": True,
            "acceptance_independence_scope": "conditional_on_fixed_endpoint_pair_cache",
            "selection_and_confirmation_streams_disjoint": True,
            "acceptance_uses_paired_common_noise": True,
            "confidence_method": "forced_test_decision",
        }

    monkeypatch.setattr(solver, "_train_costate", recorded_train_costate)
    monkeypatch.setattr(solver, "_fit_actor_streaming", changed_actor_fit)
    monkeypatch.setattr(solver, "_accept_actor", forced_accept)
    result = solver.solve(solve_key, pairs, "f")

    assert result.metrics["final_costate_refresh_executed"] is accepted
    assert len(costate_keys) == 1 + int(accepted)
    if accepted:
        assert not np.array_equal(costate_keys[0], costate_keys[1])
        assert result.metrics["final_costate_refresh_loss"] is not None
        assert result.metrics["value_critic_matches_output_actor"] is False
        assert result.metrics["direct_action_score_matches_output_actor"] is False
    else:
        assert result.metrics["final_costate_refresh_loss"] is None
        assert result.metrics["value_critic_matches_output_actor"] is True
        assert result.metrics["direct_action_score_matches_output_actor"] is True
    assert (
        result.metrics["direct_action_score"]["policy_fingerprint"]
        == result.metrics["direct_action_score_policy_fingerprint"]
    )
    assert (
        result.metrics["direct_action_score_scope"]
        == "last_policy_iteration_before_actor_acceptance"
    )
    assert (
        result.metrics["output_actor_fingerprint"] == result.metrics["costate_policy_fingerprint"]
    )
    expected_step = config.costate_steps * (1 + int(accepted))
    assert int(solver._costate_opt_state["f"].step) == expected_step


def test_zero_hard_cost_nonzero_policy_uses_anchor_hard_centering(monkeypatch):
    problem = _problem(dim=1, num_steps=3)
    config = ConditionalMAMConfig(
        actor_model="affine_reference",
        costate=MalliavinAdjointConfig(
            hidden_dims=(8,),
            time_embed_dim=4,
            training_steps=1,
            batch_size=4,
            minimum_remaining_steps=1,
        ),
        costate_steps=1,
        batch_size=4,
        acceptance_size=4,
        direct_score_diagnostic_size=4,
        policy_iterations=1,
        value_critic=ValueCriticConfig(
            hidden_dims=(8,),
            time_embed_dim=4,
            training_steps=1,
            microbatch_size=2,
            effective_batch_size=4,
        ),
    )
    execution = MAMExecutionConfig(microbatch_size=2, effective_batch_size=4)
    conditional = MAMConditionalSolver(
        problem,
        ValueOnlyRunningPotential().as_value_only_cost(),
        config,
        execution,
    )
    actor = conditional._zero_actor_params(jnp.float32)
    actor = actor.at[:, 0, 0].set(0.2)
    actor = actor.at[:, -1, 0].set(0.5)
    source_key, target_key, train_key = jax.random.split(jax.random.PRNGKey(721), 3)
    pairs = EndpointPairBatch(
        problem.sample_source(source_key, 4).astype(jnp.float32),
        problem.sample_target(target_key, 4).astype(jnp.float32),
    )
    observed_default_labels = []
    observed_zero_baseline_labels = []
    observed_wrong_suffix_labels = []
    observed_direct_components = []
    original = mam_bridge_module.MalliavinAdjointInnerSolver.make_label_batch

    def checked_make_label(inner, *args, **kwargs):
        assert kwargs.get("running_baseline_fn") is None
        labels = original(inner, *args, **kwargs)
        zero_baseline = original(
            inner,
            *args,
            **{
                **kwargs,
                "running_baseline_fn": lambda states, times, context: jnp.zeros(
                    (states.shape[0],), dtype=states.dtype
                ),
            },
        )
        wrong_suffix_baseline = original(
            inner,
            *args,
            **{
                **kwargs,
                "running_baseline_fn": lambda states, times, context: jnp.ones(
                    (states.shape[0],), dtype=states.dtype
                ),
            },
        )
        observed_default_labels.append(labels.label)
        observed_zero_baseline_labels.append(zero_baseline.label)
        observed_wrong_suffix_labels.append(wrong_suffix_baseline.label)
        observed_direct_components.append(labels.direct_component)
        return labels

    monkeypatch.setattr(
        mam_bridge_module.MalliavinAdjointInnerSolver,
        "make_label_batch",
        checked_make_label,
    )
    _, _, losses = conditional._train_costate(train_key, pairs, "f", actor)
    assert observed_default_labels
    assert all(
        jnp.array_equal(default, zero)
        for default, zero in zip(
            observed_default_labels,
            observed_zero_baseline_labels,
            strict=True,
        )
    )
    assert any(
        not jnp.array_equal(default, wrong)
        for default, wrong in zip(
            observed_default_labels,
            observed_wrong_suffix_labels,
            strict=True,
        )
    )
    assert any(bool(jnp.any(jnp.abs(value) > 0.0)) for value in observed_direct_components)
    assert jnp.all(jnp.isfinite(losses))


def test_builtin_global_run_aggregates_certified_half_work_exactly(tmp_path):
    problem = _problem(dim=1, num_steps=3)

    def hard_running(states, times, context):
        del times, context
        return (states[:, 0] > 0.0).astype(states.dtype)

    conditional = ConditionalMAMConfig(
        actor_model="affine_reference",
        costate=MalliavinAdjointConfig(
            hidden_dims=(8,),
            time_embed_dim=4,
            training_steps=1,
            batch_size=4,
            minimum_remaining_steps=1,
        ),
        costate_steps=1,
        batch_size=4,
        acceptance_size=4,
        direct_score_diagnostic_size=4,
        policy_iterations=1,
        line_search=(0.125,),
        value_critic=ValueCriticConfig(
            hidden_dims=(8,),
            time_embed_dim=4,
            training_steps=1,
            microbatch_size=2,
            effective_batch_size=4,
        ),
    )
    base = _tiny_config(cache_size=4, audit_size=4)
    config = replace(
        base,
        conditional=conditional,
        projection=replace(
            base.projection,
            validation_size=4,
            validation_projections=2,
            validation_replicates=2,
            line_search=(0.5,),
        ),
    )
    potential = ValueOnlyRunningPotential(hard_running, identifier="halfspace")
    solver = MAMBridgeSolver(
        problem,
        potential,
        config,
        SolverConfig(verbose=0),
    )
    result = solver.train(jax.random.PRNGKey(705))

    total = MAMWorkCounters.zero()
    history = result.diagnostics.metadata["audit_history"]
    assert len(history) == 2
    for audit in history:
        record = audit["work_accounting"]
        assert record["structural_counters_certified"] is True
        half = MAMWorkCounters.from_state(record["certified_counters"])
        total = total.merge(half)
        assert record["cumulative_certified_counters"] == total.to_state()
        assert record["failed_attempt_work_included"] is False
        assert record["external_oracle_billing_certified"] is False
    final = result.metadata["work_accounting"]
    assert final["structural_counters_certified"] is True
    assert final["certified_counters"] == total.to_state()
    assert final["cumulative_certified_counters"] == total.to_state()
    assert total.running_cost_oracle_evaluations > 0
    assert total.simulated_transitions > 0
    assert total.tangent_vjps > 0
    assert total.compile_time_ns == 0
    assert total.steady_state_time_ns == 0
    assert total.peak_device_memory_bytes is None

    checkpoint = tmp_path / "certified-work.pkl"
    solver.save_checkpoint(
        checkpoint,
        params=result.params,
        step=2,
        loss_history=list(np.asarray(result.loss_history)),
        metrics=solver._last_metrics,
        metadata=result.metadata,
    )
    with checkpoint.open("rb") as handle:
        original_payload = pickle.load(handle)

    def assert_rejected(payload, filename, match):
        path = tmp_path / filename
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        fresh = MAMBridgeSolver(problem, potential, config, SolverConfig(verbose=0))
        with pytest.raises((TypeError, ValueError), match=match):
            fresh.load_checkpoint(path)

    baseline = MAMBridgeSolver(problem, potential, config, SolverConfig(verbose=0))
    baseline.load_checkpoint(checkpoint)
    assert baseline._is_trained

    forged = copy.deepcopy(original_payload)
    forged_state = forged["solver_state"]
    forged_record = forged_state["audit_history"][-1]["work_accounting"]
    forged_half = MAMWorkCounters.from_state(forged_record["certified_counters"]).add(
        running_cost_oracle_evaluations=10**12
    )
    prior_total = MAMWorkCounters.from_state(
        forged_state["audit_history"][-2]["work_accounting"]["cumulative_certified_counters"]
    )
    forged_record["certified_counters"] = forged_half.to_state()
    forged_record["cumulative_certified_counters"] = prior_total.merge(forged_half).to_state()
    forged_state["last_metrics"]["work_accounting"] = copy.deepcopy(forged_record)
    forged_state["last_metrics"]["audit"]["work_accounting"] = copy.deepcopy(forged_record)
    assert_rejected(forged, "forged-counters.pkl", "exact recomputation")

    malformed = copy.deepcopy(original_payload)
    malformed_state = malformed["solver_state"]
    malformed_record = malformed_state["audit_history"][-1]["work_accounting"]
    malformed_record["derivation"]["conditional"]["policy_iterations_completed"] = 2
    malformed_state["last_metrics"]["work_accounting"] = copy.deepcopy(malformed_record)
    malformed_state["last_metrics"]["audit"]["work_accounting"] = copy.deepcopy(malformed_record)
    assert_rejected(malformed, "malformed-decisions.pkl", "policy iterations")

    impossible_acceptance = copy.deepcopy(original_payload)
    impossible_state = impossible_acceptance["solver_state"]
    impossible_record = impossible_state["audit_history"][-1]["work_accounting"]
    impossible_record["derivation"]["global"]["projection_update_accepted"] = True
    impossible_record["derivation"]["global"]["projection_confirmation_executed"] = False
    impossible_state["last_metrics"]["work_accounting"] = copy.deepcopy(impossible_record)
    impossible_state["last_metrics"]["audit"]["work_accounting"] = copy.deepcopy(impossible_record)
    assert_rejected(
        impossible_acceptance,
        "impossible-acceptance.pkl",
        "without confirmation",
    )

    legacy = copy.deepcopy(original_payload)
    legacy_state = legacy["solver_state"]
    legacy_record = legacy_state["audit_history"][-1]["work_accounting"]
    legacy_record["schema_version"] = 1
    legacy_record.pop("derivation")
    legacy_state["last_metrics"]["work_accounting"] = copy.deepcopy(legacy_record)
    legacy_state["last_metrics"]["audit"]["work_accounting"] = copy.deepcopy(legacy_record)
    assert_rejected(legacy, "legacy-certified.pkl", "schema|accounting")

    fractional_progress = copy.deepcopy(original_payload)
    fractional_progress["solver_state"]["completed_half_iterations"] = 1.9
    assert_rejected(fractional_progress, "fractional-progress.pkl", "integer")

    truthy_string = copy.deepcopy(original_payload)
    truthy_string["solver_state"]["global_endpoint_pass"] = "yes"
    assert_rejected(truthy_string, "truthy-global-pass.pkl", "bool")

    flipped_outer_pass = copy.deepcopy(original_payload)
    flipped_outer_pass["solver_state"]["global_endpoint_pass"] = not flipped_outer_pass[
        "solver_state"
    ]["global_endpoint_pass"]
    assert_rejected(
        flipped_outer_pass,
        "flipped-outer-global-pass.pkl",
        "global endpoint",
    )

    flipped_metadata_pass = copy.deepcopy(original_payload)
    flipped_metadata_pass["metadata"]["global_endpoint_pass"] = not flipped_metadata_pass[
        "metadata"
    ]["global_endpoint_pass"]
    assert_rejected(
        flipped_metadata_pass,
        "flipped-metadata-global-pass.pkl",
        "metadata global_endpoint_pass",
    )

    wrong_metadata_status = copy.deepcopy(original_payload)
    wrong_metadata_status["metadata"]["status"] = "GLOBAL_ENDPOINT_AUDIT_PASSED"
    if original_payload["metadata"]["status"] == "GLOBAL_ENDPOINT_AUDIT_PASSED":
        wrong_metadata_status["metadata"]["status"] = "EXPERIMENTAL_GLOBAL_ENDPOINT_UNVERIFIED"
    assert_rejected(
        wrong_metadata_status,
        "wrong-metadata-status.pkl",
        "metadata status",
    )

    wrong_metadata_topology = copy.deepcopy(original_payload)
    wrong_metadata_topology["metadata"]["device_topology"]["device_kind"] += "-forged"
    assert_rejected(
        wrong_metadata_topology,
        "wrong-metadata-topology.pkl",
        "device_topology",
    )

    wrong_costate_policy = copy.deepcopy(original_payload)
    conditional_state = wrong_costate_policy["solver_state"]["conditional_solver"]
    direction = next(iter(conditional_state["costate_policy_fingerprint"]))
    conditional_state["costate_policy_fingerprint"][direction] = "0" * 64
    assert_rejected(
        wrong_costate_policy,
        "wrong-costate-policy.pkl",
        "costate policy fingerprint",
    )

    wrong_optimizer_step = copy.deepcopy(original_payload)
    optimizer_store = wrong_optimizer_step["solver_state"]["conditional_solver"][
        "costate_opt_state"
    ]
    direction = next(iter(optimizer_store))
    optimizer = optimizer_store[direction]
    optimizer_store[direction] = replace(
        optimizer,
        step=optimizer.step + config.conditional.costate_steps,
    )
    assert_rejected(
        wrong_optimizer_step,
        "wrong-costate-adam-step.pkl",
        "exact per-direction progress",
    )

    nonfinite_loss = copy.deepcopy(original_payload)
    nonfinite_loss["solver_state"]["loss_history"][-1] = float("nan")
    assert_rejected(nonfinite_loss, "nonfinite-loss.pkl", "finite")

    malformed_calibration = copy.deepcopy(original_payload)
    malformed_calibration["solver_state"]["source_calibration"] = {"finite": True}
    assert_rejected(malformed_calibration, "malformed-calibration.pkl", "wrong type")

    inflated_threshold = copy.deepcopy(original_payload)
    inflated_state = inflated_threshold["solver_state"]
    source_calibration = inflated_state["source_calibration"]
    inflated_state["source_calibration"] = replace(
        source_calibration,
        thresholds=replace(
            source_calibration.thresholds,
            mmd2=source_calibration.thresholds.mmd2 + 1.0,
        ),
    )
    assert_rejected(
        inflated_threshold,
        "inflated-threshold.pkl",
        "disagrees with null metrics",
    )

    flipped_global_audit = copy.deepcopy(original_payload)
    flipped_global_state = flipped_global_audit["solver_state"]
    flipped_audit = flipped_global_state["audit_history"][-1]
    flipped_audit["global_endpoint_pass"] = not flipped_audit["global_endpoint_pass"]
    flipped_global_state["last_metrics"]["audit"] = copy.deepcopy(flipped_audit)
    assert_rejected(
        flipped_global_audit,
        "flipped-global-audit.pkl",
        "decision flags",
    )


def _assert_tree_equal(left, right):
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        assert jnp.array_equal(left_leaf, right_leaf)


def test_checkpoint_envelope_and_origin_topology_fail_closed_but_are_portable(tmp_path):
    problem = _problem(dim=1, num_steps=3)
    config = _tiny_config()
    backend = _PinnedReferenceConditional(problem)
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        config,
        SolverConfig(verbose=0),
        conditional_solver=backend,
    )
    checkpoint = tmp_path / "checkpoint-envelope.pkl"
    solver.save_checkpoint(
        checkpoint,
        params=solver.init_params(jax.random.PRNGKey(724)),
        step=0,
    )
    with checkpoint.open("rb") as handle:
        original = pickle.load(handle)

    def load_payload(payload, name):
        path = tmp_path / name
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        fresh = MAMBridgeSolver(
            problem,
            ValueOnlyRunningPotential(),
            config,
            SolverConfig(verbose=0),
            conditional_solver=_PinnedReferenceConditional(problem),
        )
        return fresh, path

    wrong_version = copy.deepcopy(original)
    wrong_version["format_version"] = "1"
    fresh, path = load_payload(wrong_version, "wrong-envelope-version.pkl")
    with pytest.raises(ValueError, match="format version"):
        fresh.load_checkpoint(path)
    assert fresh._params is None and fresh._is_trained is False

    wrong_type = copy.deepcopy(original)
    wrong_type["solver_type"] = "SCORE"
    fresh, path = load_payload(wrong_type, "wrong-envelope-solver.pkl")
    with pytest.raises(ValueError, match="solver_type mismatch"):
        fresh.load_checkpoint(path)
    assert fresh._params is None and fresh._is_trained is False

    extra_field = copy.deepcopy(original)
    extra_field["unexpected"] = True
    fresh, path = load_payload(extra_field, "extra-envelope-field.pkl")
    with pytest.raises(ValueError, match="envelope schema"):
        fresh.load_checkpoint(path)

    malformed_topology = copy.deepcopy(original)
    malformed_topology["solver_state"]["device_topology"]["selected_device_ids"] = []
    fresh, path = load_payload(malformed_topology, "malformed-topology.pkl")
    with pytest.raises(ValueError, match="selected_device_ids"):
        fresh.load_checkpoint(path)

    portable = copy.deepcopy(original)
    origin = {
        "platform": "recorded-accelerator",
        "device_kind": "recorded-device-kind",
        "available_local_device_count": 2,
        "selected_device_count": 1,
        "selected_device_ids": [99],
        "process_count": 2,
        "process_index": 1,
        "batch_data_parallel": False,
    }
    portable["solver_state"]["device_topology"] = origin
    fresh, path = load_payload(portable, "portable-origin-topology.pkl")
    fresh.load_checkpoint(path)
    assert fresh._checkpoint_origin_device_topology == origin
    assert fresh._device_topology.to_state() != origin


def test_interrupted_checkpoint_resume_matches_uninterrupted_run(tmp_path):
    problem = _problem(dim=1, num_steps=3)
    config = _tiny_config(num_iterations=2)
    training = TrainingConfig(
        num_iterations=1,
        batch_size=4,
        checkpoint_every=1,
        checkpoint_dir=str(tmp_path),
        save_final_checkpoint=True,
    )

    uninterrupted = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        config,
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    expected = uninterrupted.train(jax.random.PRNGKey(31))
    with pytest.raises(RuntimeError, match="already complete"):
        uninterrupted.train(jax.random.PRNGKey(31))

    interrupted = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        config,
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )

    class ExpectedInterruption(RuntimeError):
        pass

    def interrupt_after_first_half(step, metrics):
        del metrics
        if step == 1:
            raise ExpectedInterruption

    with pytest.raises(ExpectedInterruption):
        interrupted.train(
            jax.random.PRNGKey(31),
            training,
            callback=interrupt_after_first_half,
        )

    checkpoint = tmp_path / "checkpoint_mam_bridge_step_00000001.pkl"
    assert checkpoint.exists()
    resumed = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        config,
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    payload = resumed.load_checkpoint(checkpoint)
    actual = resumed.train(jax.random.PRNGKey(999), training)

    def changed_zero_cost(states, times, context):
        del times, context
        return jnp.zeros((states.shape[0],), dtype=states.dtype)

    incompatible = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(changed_zero_cost, identifier="zero_running_potential"),
        config,
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        incompatible.load_checkpoint(checkpoint)
    assert incompatible._params is None
    assert incompatible._is_trained is False
    with pytest.raises(ValueError, match="not trained"):
        incompatible.sample(jax.random.PRNGKey(32), 1)

    backend_incompatible = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        config,
        SolverConfig(verbose=0),
        conditional_solver=_DifferentPinnedReferenceConditional(problem),
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        backend_incompatible.load_checkpoint(checkpoint)
    assert backend_incompatible._params is None
    assert backend_incompatible._is_trained is False

    assert payload["step"] == 1
    assert actual.metadata["checkpoint_path"].endswith("checkpoint_mam_bridge_final.pkl")
    assert actual.metadata["rng_ledger"] == expected.metadata["rng_ledger"]
    assert actual.metadata["pair_cache_sha256"] == expected.metadata["pair_cache_sha256"]
    assert jnp.array_equal(actual.loss_history, expected.loss_history)
    _assert_tree_equal(actual.params, expected.params)


def test_last_half_callback_failure_can_finalize_or_resume(tmp_path):
    problem = _problem(dim=1, num_steps=3)
    config = _tiny_config()
    training = TrainingConfig(
        num_iterations=1,
        batch_size=4,
        checkpoint_every=1,
        checkpoint_dir=str(tmp_path),
        save_final_checkpoint=True,
    )
    solver = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        config,
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )

    class FinalCallbackFailure(RuntimeError):
        pass

    def fail_after_final_half(step, metrics):
        del metrics
        if step == 2:
            raise FinalCallbackFailure

    with pytest.raises(FinalCallbackFailure):
        solver.train(jax.random.PRNGKey(61), training, callback=fail_after_final_half)
    assert solver._completed_half_iterations == 2
    assert solver._is_trained is False
    finalized = solver.train(jax.random.PRNGKey(999), training)
    assert finalized.metadata["num_half_iterations"] == 2

    step_checkpoint = tmp_path / "checkpoint_mam_bridge_step_00000002.pkl"
    fresh = MAMBridgeSolver(
        problem,
        ValueOnlyRunningPotential(),
        config,
        SolverConfig(verbose=0),
        conditional_solver=_PinnedReferenceConditional(problem),
    )
    fresh.load_checkpoint(step_checkpoint)
    assert fresh._is_trained is False
    resumed = fresh.train(jax.random.PRNGKey(123), training)
    _assert_tree_equal(resumed.params, finalized.params)
