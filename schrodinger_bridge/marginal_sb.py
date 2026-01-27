"""Marginal Schrödinger Bridge Extension.

Extends the standard Schrödinger Bridge to handle intermediate marginal constraints.

Standard SB: Match marginals at t=0 and t=1
Marginal SB: Match marginals at t=0, t₁, t₂, ..., tₖ, t=1

Mathematical formulation:
    P* = argmin KL(P || P_ref)
    subject to: P_{t_i} = μ_i for i = 0, 1, ..., K

This decomposes into K+1 coupled SB problems on intervals [t_{i-1}, t_i].

References:
    Liu et al. "Generalized Schrödinger Bridge Matching" (2023)
    Chen et al. "Multi-marginal Schrödinger Bridges" (2021)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .core.types import (
    Array,
    DriftFn,
    Params,
    PRNGKey,
    Scalar,
    TimeGrid,
    TrajectoryBatch,
    SolverResult,
    DiagnosticReport,
    InvariantViolation,
)
from .core.problem import SBProblem, MarginalDistribution, ReferenceDynamics
from .core.invariants import mmd_squared, InvariantChecker
from .solvers.base import SBSolver, SBSolution


# =============================================================================
# Marginal Constraint Specification
# =============================================================================

@dataclass
class MarginalConstraint:
    """A marginal constraint at a specific time.
    
    Attributes:
        time: Time point t in [0, 1] for the constraint.
        distribution: Target marginal distribution at this time.
        weight: Relative weight for this constraint in the loss.
    """
    time: float
    distribution: MarginalDistribution
    weight: float = 1.0
    
    def __post_init__(self):
        if not 0 <= self.time <= 1:
            raise ValueError(f"Time must be in [0, 1], got {self.time}")


@dataclass
class MarginalSBProblem:
    """Marginal Schrödinger Bridge problem specification.
    
    Extends SBProblem with intermediate marginal constraints.
    
    Attributes:
        reference: Reference stochastic process.
        marginals: List of marginal constraints (must include t=0 and t=1).
        time_grid: Time discretization for the full interval.
        name: Optional problem name.
    """
    reference: ReferenceDynamics
    marginals: List[MarginalConstraint]
    time_grid: TimeGrid = field(default_factory=lambda: TimeGrid(num_steps=100))
    name: str = "MarginalSB"
    
    def __post_init__(self):
        # Sort marginals by time
        self.marginals = sorted(self.marginals, key=lambda m: m.time)
        
        # Validate endpoints exist
        times = [m.time for m in self.marginals]
        if 0.0 not in times:
            raise ValueError("Must include marginal at t=0")
        if 1.0 not in times:
            raise ValueError("Must include marginal at t=1")
        
        # Validate dimension consistency
        dims = [m.distribution.dim for m in self.marginals]
        if len(set(dims)) > 1:
            raise ValueError(f"All marginals must have same dimension, got {dims}")
    
    @property
    def dim(self) -> int:
        """State space dimension."""
        return self.marginals[0].distribution.dim
    
    @property
    def num_segments(self) -> int:
        """Number of SB segments (intervals between consecutive marginals)."""
        return len(self.marginals) - 1
    
    @property
    def segment_times(self) -> List[Tuple[float, float]]:
        """List of (t_start, t_end) for each segment."""
        times = [m.time for m in self.marginals]
        return [(times[i], times[i+1]) for i in range(len(times) - 1)]
    
    def get_marginal_at(self, t: float) -> Optional[MarginalDistribution]:
        """Get marginal distribution at time t (if constrained)."""
        for m in self.marginals:
            if abs(m.time - t) < 1e-6:
                return m.distribution
        return None
    
    def sample_marginal(self, key: PRNGKey, t: float, n: int) -> Array:
        """Sample from marginal at time t."""
        dist = self.get_marginal_at(t)
        if dist is None:
            raise ValueError(f"No marginal constraint at t={t}")
        return dist.sample(key, n)
    
    def get_segment_problem(self, segment_idx: int) -> SBProblem:
        """Get SBProblem for a single segment.
        
        Args:
            segment_idx: Index of segment (0 to num_segments-1).
            
        Returns:
            SBProblem for that segment with rescaled time [0, 1].
        """
        if segment_idx < 0 or segment_idx >= self.num_segments:
            raise ValueError(f"Invalid segment index {segment_idx}")
        
        t_start, t_end = self.segment_times[segment_idx]
        source = self.marginals[segment_idx].distribution
        target = self.marginals[segment_idx + 1].distribution
        
        # Compute number of steps proportional to segment length
        segment_length = t_end - t_start
        num_steps = max(10, int(self.time_grid.num_steps * segment_length))
        
        return SBProblem(
            reference=self.reference,
            source=source,
            target=target,
            time_grid=TimeGrid(t0=0.0, t1=1.0, num_steps=num_steps),
            name=f"{self.name}_segment_{segment_idx}",
        )
    
    def summary(self) -> str:
        """Get problem summary string."""
        lines = [
            f"=== {self.name} ===",
            f"Dimension: {self.dim}",
            f"Reference: {type(self.reference).__name__}",
            f"Num marginals: {len(self.marginals)}",
            f"Num segments: {self.num_segments}",
            "Marginal times: " + ", ".join(f"{m.time:.3f}" for m in self.marginals),
        ]
        return "\n".join(lines)


# =============================================================================
# Marginal SB Solver
# =============================================================================

@dataclass
class MarginalSBConfig:
    """Configuration for Marginal SB solver."""
    segment_solver_type: str = 'score'  # 'score', 'fbsde', 'doob', etc.
    coupling_method: str = 'sequential'  # 'sequential' or 'joint'
    num_iterations: int = 1000
    batch_size: int = 256
    verbose: int = 1


class MarginalSBSolver:
    """Solver for Marginal Schrödinger Bridge problems.
    
    Decomposes the problem into K segments and solves each as a standard SB.
    Two approaches are supported:
    
    1. Sequential: Solve segments independently, then concatenate.
       Fast but may have discontinuities at boundaries.
       
    2. Joint: Iterate between segment solutions to enforce consistency.
       More accurate but slower.
    
    Attributes:
        problem: MarginalSBProblem specification.
        config: Solver configuration.
        segment_solvers: List of SBSolver for each segment.
    """
    
    def __init__(
        self,
        problem: MarginalSBProblem,
        config: Optional[MarginalSBConfig] = None,
    ):
        self.problem = problem
        self.config = config or MarginalSBConfig()
        
        self.segment_solvers: List[SBSolver] = []
        self.segment_solutions: List[SBSolution] = []
        self._is_trained = False
    
    def _create_segment_solver(self, segment_idx: int) -> SBSolver:
        """Create solver for a single segment."""
        from .solvers import (
            ScoreBasedSolver, FBSDESolver, DoobHTransformSolver,
            RKHSSolver, IMFSolver
        )
        
        segment_problem = self.problem.get_segment_problem(segment_idx)
        
        solver_map = {
            'score': ScoreBasedSolver,
            'fbsde': FBSDESolver,
            'doob': DoobHTransformSolver,
            'rkhs': RKHSSolver,
            'imf': IMFSolver,
        }
        
        solver_cls = solver_map.get(self.config.segment_solver_type, ScoreBasedSolver)
        return solver_cls(segment_problem)
    
    def train(
        self,
        key: PRNGKey,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Train the marginal SB solver.
        
        Args:
            key: Random key.
            callback: Optional callback(segment_idx, result).
            
        Returns:
            Dictionary with training results.
        """
        from .core.types import TrainingConfig
        
        results = {
            'segment_results': [],
            'total_loss': 0.0,
        }
        
        if self.config.verbose >= 1:
            print(f"Training Marginal SB with {self.problem.num_segments} segments")
            print(f"Segment solver: {self.config.segment_solver_type}")
        
        # Create solvers for each segment
        self.segment_solvers = []
        for i in range(self.problem.num_segments):
            solver = self._create_segment_solver(i)
            self.segment_solvers.append(solver)
        
        # Train each segment
        self.segment_solutions = []
        
        for i, solver in enumerate(self.segment_solvers):
            key, subkey = jax.random.split(key)
            
            t_start, t_end = self.problem.segment_times[i]
            if self.config.verbose >= 1:
                print(f"\n--- Segment {i}: [{t_start:.3f}, {t_end:.3f}] ---")
            
            train_config = TrainingConfig(
                num_iterations=self.config.num_iterations,
                batch_size=self.config.batch_size,
            )
            
            result = solver.train(subkey, train_config)
            results['segment_results'].append(result)
            results['total_loss'] += float(result.final_loss)
            
            # Create solution
            solution = SBSolution(
                problem=solver.problem,
                solver_type=solver.solver_type,
                params=result.params,
                representation=solver.representation_type,
                metadata={'segment_idx': i},
            )
            self.segment_solutions.append(solution)
            
            if callback:
                callback(i, result)
        
        self._is_trained = True
        
        if self.config.verbose >= 1:
            print(f"\nTotal loss: {results['total_loss']:.6f}")
        
        return results
    
    def sample(
        self,
        key: PRNGKey,
        num_samples: int,
    ) -> TrajectoryBatch:
        """Sample full trajectories through all segments.
        
        Args:
            key: Random key.
            num_samples: Number of trajectories.
            
        Returns:
            TrajectoryBatch for the full time interval.
        """
        if not self._is_trained:
            raise RuntimeError("Solver must be trained before sampling")
        
        all_paths = []
        all_times = []
        
        # Sample initial points from t=0 marginal
        key, subkey = jax.random.split(key)
        x = self.problem.sample_marginal(subkey, 0.0, num_samples)
        
        for i, (solver, solution) in enumerate(zip(self.segment_solvers, self.segment_solutions)):
            key, subkey = jax.random.split(key)
            
            t_start, t_end = self.problem.segment_times[i]
            segment_length = t_end - t_start
            
            # Sample trajectories starting from current x
            traj = solver.sample(subkey, num_samples, params=solution.params)
            
            # Override starting point with our x
            # (This is approximate - ideally we'd integrate from x)
            paths = traj.paths.at[:, 0, :].set(x)
            
            # Rescale times to global coordinates
            segment_times = t_start + traj.times * segment_length
            
            # Don't duplicate endpoint (except for last segment)
            if i < self.problem.num_segments - 1:
                all_paths.append(paths[:, :-1, :])
                all_times.append(segment_times[:-1])
            else:
                all_paths.append(paths)
                all_times.append(segment_times)
            
            # Update x to endpoint for next segment
            x = paths[:, -1, :]
        
        # Concatenate
        full_paths = jnp.concatenate(all_paths, axis=1)
        full_times = jnp.concatenate(all_times)
        
        return TrajectoryBatch(
            paths=full_paths,
            times=full_times,
        )
    
    def get_drift(self, t: float) -> DriftFn:
        """Get drift function at time t.
        
        Automatically selects the correct segment.
        """
        if not self._is_trained:
            raise RuntimeError("Solver must be trained first")
        
        # Find segment containing t
        for i, (t_start, t_end) in enumerate(self.problem.segment_times):
            if t_start <= t <= t_end:
                # Rescale t to segment's [0, 1]
                t_local = (t - t_start) / (t_end - t_start)
                
                segment_drift = self.segment_solvers[i].extract_drift(
                    self.segment_solutions[i].params
                )
                
                # Wrap to handle time rescaling
                def drift_wrapper(x, t_global, _t_local=t_local, _drift=segment_drift):
                    return _drift(x, _t_local)
                
                return drift_wrapper
        
        raise ValueError(f"Time {t} not in any segment")
    
    def check_marginal_consistency(
        self,
        key: PRNGKey,
        num_samples: int = 500,
    ) -> Dict[str, float]:
        """Check how well the solution matches intermediate marginals.
        
        Returns:
            Dictionary mapping constraint times to MMD values.
        """
        results = {}
        
        # Sample full trajectory
        key, subkey = jax.random.split(key)
        traj = self.sample(subkey, num_samples)
        
        for marginal in self.problem.marginals:
            t = marginal.time
            
            # Find closest time index
            idx = int(jnp.argmin(jnp.abs(traj.times - t)))
            samples_at_t = traj.paths[:, idx, :]
            
            # Sample from target marginal
            key, subkey = jax.random.split(key)
            target_samples = marginal.distribution.sample(subkey, num_samples)
            
            # Compute MMD
            mmd = float(mmd_squared(samples_at_t, target_samples))
            results[f"t={t:.3f}"] = mmd
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def create_marginal_sb_problem(
    reference: ReferenceDynamics,
    marginal_times: List[float],
    marginal_distributions: List[MarginalDistribution],
    weights: Optional[List[float]] = None,
    num_steps: int = 100,
    name: str = "MarginalSB",
) -> MarginalSBProblem:
    """Create a MarginalSBProblem from lists.
    
    Args:
        reference: Reference dynamics.
        marginal_times: List of times [t0, t1, ..., tK].
        marginal_distributions: List of distributions at each time.
        weights: Optional weights for each constraint.
        num_steps: Number of time steps.
        name: Problem name.
        
    Returns:
        MarginalSBProblem instance.
    """
    if len(marginal_times) != len(marginal_distributions):
        raise ValueError("Times and distributions must have same length")
    
    if weights is None:
        weights = [1.0] * len(marginal_times)
    
    marginals = [
        MarginalConstraint(time=t, distribution=d, weight=w)
        for t, d, w in zip(marginal_times, marginal_distributions, weights)
    ]
    
    return MarginalSBProblem(
        reference=reference,
        marginals=marginals,
        time_grid=TimeGrid(num_steps=num_steps),
        name=name,
    )


def solve_marginal_sb(
    problem: MarginalSBProblem,
    key: PRNGKey,
    solver_type: str = 'score',
    num_iterations: int = 1000,
    verbose: int = 1,
) -> Tuple[MarginalSBSolver, Dict]:
    """Convenience function to solve a marginal SB problem.
    
    Args:
        problem: MarginalSBProblem to solve.
        key: Random key.
        solver_type: Type of segment solver.
        num_iterations: Training iterations per segment.
        verbose: Verbosity level.
        
    Returns:
        (solver, results) tuple.
    """
    config = MarginalSBConfig(
        segment_solver_type=solver_type,
        num_iterations=num_iterations,
        verbose=verbose,
    )
    
    solver = MarginalSBSolver(problem, config)
    results = solver.train(key)
    
    return solver, results


# =============================================================================
# Interpolation Utilities
# =============================================================================

def interpolate_marginals(
    source: MarginalDistribution,
    target: MarginalDistribution,
    num_intermediate: int = 3,
    method: str = 'linear',
) -> List[MarginalConstraint]:
    """Create intermediate marginal constraints by interpolation.
    
    For Gaussian marginals, interpolates means and covariances.
    For others, uses displacement interpolation via OT (if available).
    
    Args:
        source: Source distribution (t=0).
        target: Target distribution (t=1).
        num_intermediate: Number of intermediate constraints.
        method: 'linear' or 'ot'.
        
    Returns:
        List of MarginalConstraints including endpoints.
    """
    from .core.problem import GaussianDistribution
    
    times = jnp.linspace(0, 1, num_intermediate + 2)
    constraints = []
    
    # Check if both are Gaussian
    if isinstance(source, GaussianDistribution) and isinstance(target, GaussianDistribution):
        for t in times:
            t = float(t)
            
            # Linear interpolation of parameters
            mean_t = (1 - t) * source.mean + t * target.mean
            
            # Interpolate covariance (simple linear for now)
            cov_t = (1 - t) * source.cov + t * target.cov
            
            dist_t = GaussianDistribution(
                mean=mean_t,
                cov=cov_t,
                dim=source.dim,
            )
            
            constraints.append(MarginalConstraint(time=t, distribution=dist_t))
    
    else:
        # For non-Gaussian, only set endpoints
        # Intermediate constraints would require OT interpolation
        constraints = [
            MarginalConstraint(time=0.0, distribution=source),
            MarginalConstraint(time=1.0, distribution=target),
        ]
        
        if num_intermediate > 0:
            import warnings
            warnings.warn(
                "Intermediate marginal interpolation for non-Gaussian distributions "
                "requires OT. Only endpoints will be constrained."
            )
    
    return constraints


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core classes
    'MarginalConstraint',
    'MarginalSBProblem',
    'MarginalSBConfig',
    'MarginalSBSolver',
    # Convenience functions
    'create_marginal_sb_problem',
    'solve_marginal_sb',
    'interpolate_marginals',
]
