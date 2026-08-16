"""Uncertainty-aware acceptance utilities for MAM policy updates.

The functions in this module operate on already evaluated paired objectives.
They never reuse training labels and make no claim that Monte Carlo acceptance
is a theorem of monotone improvement.  Common random numbers reduce the
variance of the paired difference; an untouched confirmation set is still
required after selecting a line-search candidate.  The reported bound is the
usual normal/CLT approximation, not a distribution-free finite-sample bound;
heavy-tailed hard-cost differences require calibration or a robust alternative.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ..core.types import Array


class PairedObjectiveStatistics(NamedTuple):
    """Statistics for candidate minus current objective values."""

    mean_delta: Array
    standard_error: Array
    upper_confidence_bound: Array
    accepted: Array
    sample_count: Array


class LineSearchSelection(NamedTuple):
    """One candidate selected on a disposable line-search split."""

    selected_index: Array
    selected_step_size: Array
    has_acceptable_candidate: Array
    statistics: PairedObjectiveStatistics


def paired_objective_statistics(
    current_objective: Array,
    candidate_objective: Array,
    *,
    z_value: float = 1.6448536269514722,
    minimum_improvement: float = 0.0,
) -> PairedObjectiveStatistics:
    """Compute a one-sided normal-approximation paired decision.

    Inputs have trailing sample axis ``B`` and may contain leading candidate
    axes.  Acceptance requires

    ``mean(candidate-current) + z * SE < -minimum_improvement``.

    With the default ``z``, this is conventionally called a one-sided 95%
    upper bound only under an adequate normal approximation.  It is not a
    nonasymptotic guarantee for skewed or heavy-tailed paired differences.
    """
    current = jnp.asarray(current_objective)
    candidate = jnp.asarray(candidate_objective, dtype=current.dtype)
    if not jnp.issubdtype(current.dtype, jnp.inexact):
        raise TypeError("objective arrays must have a floating dtype")
    if not np.isfinite(z_value) or z_value < 0.0:
        raise ValueError("z_value must be finite and nonnegative")
    if not np.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if current.ndim < 1 or candidate.ndim < 1:
        raise ValueError("objective arrays must have a sample axis")
    try:
        difference = candidate - current
    except TypeError as exc:
        raise ValueError("candidate and current objectives must be broadcast-compatible") from exc
    sample_count = difference.shape[-1]
    if sample_count < 2:
        raise ValueError("paired acceptance requires at least two samples")
    finite = jnp.all(jnp.isfinite(difference), axis=-1)
    mean_delta = jnp.mean(difference, axis=-1)
    centered = difference - mean_delta[..., None]
    variance = jnp.sum(centered**2, axis=-1) / (sample_count - 1)
    standard_error = jnp.sqrt(variance / sample_count)
    upper = mean_delta + jnp.asarray(z_value, dtype=difference.dtype) * standard_error
    accepted = finite & (upper < -jnp.asarray(minimum_improvement, dtype=difference.dtype))
    return PairedObjectiveStatistics(
        mean_delta=mean_delta,
        standard_error=standard_error,
        upper_confidence_bound=upper,
        accepted=accepted,
        sample_count=jnp.asarray(sample_count, dtype=jnp.int32),
    )


def select_line_search_candidate(
    step_sizes: Array,
    current_objective: Array,
    candidate_objectives: Array,
    *,
    z_value: float = 1.6448536269514722,
    minimum_improvement: float = 0.0,
) -> LineSearchSelection:
    """Select the best statistically acceptable candidate on one split.

    ``candidate_objectives`` has shape ``[K,B]`` and ``step_sizes`` has shape
    ``[K]``.  The selected candidate must be confirmed on an independent split
    before it is committed.
    """
    steps = jnp.asarray(step_sizes)
    candidates = jnp.asarray(candidate_objectives)
    current = jnp.asarray(current_objective)
    if steps.ndim != 1:
        raise ValueError("step_sizes must have shape [num_candidates]")
    if candidates.ndim != 2 or candidates.shape[0] != steps.shape[0]:
        raise ValueError("candidate_objectives must have shape [num_candidates, samples]")
    if current.ndim != 1 or current.shape[0] != candidates.shape[1]:
        raise ValueError("current_objective must have shape [samples]")
    stats = paired_objective_statistics(
        current[None, :],
        candidates,
        z_value=z_value,
        minimum_improvement=minimum_improvement,
    )
    acceptable_score = jnp.where(stats.accepted, stats.mean_delta, jnp.inf)
    index = jnp.argmin(acceptable_score)
    has_candidate = jnp.any(stats.accepted)
    # Index zero is a harmless placeholder when no candidate passes.  The
    # explicit boolean prevents callers from silently accepting it.
    index = jnp.where(has_candidate, index, 0)
    return LineSearchSelection(
        selected_index=index,
        selected_step_size=steps[index],
        has_acceptable_candidate=has_candidate,
        statistics=stats,
    )


__all__ = [
    "LineSearchSelection",
    "PairedObjectiveStatistics",
    "paired_objective_statistics",
    "select_line_search_candidate",
]
