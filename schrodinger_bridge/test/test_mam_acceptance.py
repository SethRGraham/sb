"""Tests for independent paired MAM update acceptance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from schrodinger_bridge.solvers.mam_acceptance import (
    paired_objective_statistics,
    select_line_search_candidate,
)


def _dtype():
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


def test_paired_objective_accepts_only_negative_upper_confidence_bound():
    current = jnp.asarray([1.0, 1.2, 0.8, 1.1], dtype=_dtype())
    improving = current - jnp.asarray([0.2, 0.1, 0.3, 0.2], dtype=_dtype())
    worsening = current + 0.1
    accepted = paired_objective_statistics(current, improving, z_value=1.645)
    rejected = paired_objective_statistics(current, worsening, z_value=1.645)
    assert bool(accepted.accepted)
    assert float(accepted.upper_confidence_bound) < 0.0
    assert not bool(rejected.accepted)


def test_line_search_selects_best_acceptable_candidate_and_is_jittable():
    current = jnp.ones((8,), dtype=_dtype())
    candidates = jnp.stack(
        [
            current + 0.1,
            current - 0.05,
            current - 0.2,
        ]
    )
    steps = jnp.asarray([1.0, 0.5, 0.25], dtype=_dtype())
    eager = select_line_search_candidate(steps, current, candidates)
    compiled = jax.jit(select_line_search_candidate)(steps, current, candidates)
    assert bool(eager.has_acceptable_candidate)
    assert int(eager.selected_index) == 2
    np.testing.assert_array_equal(
        np.asarray(compiled.selected_index),
        np.asarray(eager.selected_index),
    )


def test_line_search_explicitly_reports_when_nothing_passes():
    current = jnp.ones((4,), dtype=_dtype())
    candidates = jnp.stack([current + 0.1, current + 0.2])
    selection = select_line_search_candidate(
        jnp.asarray([1.0, 0.5], dtype=_dtype()),
        current,
        candidates,
    )
    assert not bool(selection.has_acceptable_candidate)
    assert int(selection.selected_index) == 0


def test_acceptance_contract_rejects_invalid_statistics_inputs():
    current = jnp.ones((4,), dtype=_dtype())
    with pytest.raises(ValueError, match="z_value"):
        paired_objective_statistics(current, current, z_value=-1.0)
    with pytest.raises(ValueError, match="minimum_improvement"):
        paired_objective_statistics(current, current, minimum_improvement=-0.1)
    with pytest.raises(TypeError, match="floating"):
        paired_objective_statistics(jnp.ones((4,), dtype=jnp.int32), jnp.ones((4,)))
    with pytest.raises(ValueError, match="current_objective"):
        select_line_search_candidate(
            jnp.asarray([1.0], dtype=_dtype()),
            current[:3],
            current[None, :],
        )
