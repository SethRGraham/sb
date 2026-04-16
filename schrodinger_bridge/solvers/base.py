"""Base classes for Schrödinger Bridge solvers.

This module defines the abstract interface that all SB solvers must implement,
ensuring interoperability and consistent behavior.

Key principles:
1. Solvers don't assume a specific representation
2. Continuous time API with internal discretization
3. Comprehensive diagnostics
4. Support for both neural and non-neural methods
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp

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
    TimeGrid,
    TrajectoryBatch,
    TrainingConfig,
)
from ..core.problem import SBProblem
from ..core.invariants import InvariantChecker
from ..integrators import Integrator, create_integrator
from ..process import BridgeProcess


# =============================================================================
# Solution Container
# =============================================================================

@dataclass
class SBSolution:
    """Container for a trained Schrödinger Bridge solution.
    
    Provides unified interface for sampling and density evaluation
    regardless of the underlying solver representation.
    
    Attributes:
        problem: The SB problem being solved.
        solver_type: Type of solver used.
        params: Learned parameters (solver-specific).
        representation: Type of representation used.
        metadata: Additional solver-specific data.
    """
    problem: SBProblem
    solver_type: SolverType
    params: Any
    representation: RepresentationType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Cached function handles
    _integrator: Optional[Integrator] = None
    _forward_drift: Optional[DriftFn] = None
    _backward_drift: Optional[DriftFn] = None
    _score_fn: Optional[Callable] = None
    
    def get_forward_drift(self) -> DriftFn:
        """Get the forward SDE drift function.
        
        For SB: b*(x,t) = b_ref(x,t) + σ²(t) ∇log ψ(x,t)
        
        Returns:
            Drift function b*(x, t).
        """
        if self._forward_drift is None:
            raise ValueError("Forward drift not set. Call solver.extract_drift() first.")
        return self._forward_drift
    
    def get_backward_drift(self) -> DriftFn:
        """Get the backward (reverse-time) SDE drift.
        
        Returns:
            Backward drift function.
        """
        if self._backward_drift is None:
            raise ValueError("Backward drift not set.")
        return self._backward_drift

    def as_process(
        self,
        integrator: Optional[Integrator] = None,
        backend: str = "native",
    ) -> BridgeProcess:
        """Materialize the solved bridge as a runtime process object."""
        if integrator is None:
            integrator = self._integrator
        if integrator is None:
            integrator_name = self.metadata.get("integrator_type")
            if integrator_name is not None:
                try:
                    integrator = create_integrator(IntegratorType[integrator_name])
                except (KeyError, ValueError):
                    integrator = None

        return BridgeProcess(
            problem=self.problem,
            solver_type=self.solver_type,
            representation_type=self.representation,
            params=self.params,
            forward_drift_fn=self.get_forward_drift(),
            backward_drift_fn=self._backward_drift,
            score_fn=self._score_fn,
            integrator=integrator,
            metadata=dict(self.metadata),
            backend=backend,
        )
    
    def sample_trajectories(
        self,
        key: PRNGKey,
        num_samples: int,
        return_full: bool = True,
        integrator: Optional[Integrator] = None,
    ) -> Union[TrajectoryBatch, Array]:
        """Sample trajectories from the learned bridge.
        
        Args:
            key: JAX random key.
            num_samples: Number of trajectories.
            return_full: If True, return full trajectories.
            integrator: Integrator to use (default: Euler-Maruyama).
            
        Returns:
            Batch of trajectories.
        """
        return self.as_process(integrator=integrator).sample_paths(
            key,
            num_samples,
            return_full=return_full,
        )
    
    def sample_endpoint(self, key: PRNGKey, num_samples: int) -> Array:
        """Sample from the transported distribution at t=1.
        
        Args:
            key: JAX random key.
            num_samples: Number of samples.
            
        Returns:
            Samples at t=1, shape [num_samples, dim].
        """
        return self.as_process().sample_endpoint(key, num_samples)
    
    def evaluate_at_time(
        self,
        key: PRNGKey,
        t: Scalar,
        num_samples: int,
    ) -> Array:
        """Sample from the marginal distribution at time t.
        
        Args:
            key: JAX random key.
            t: Time point.
            num_samples: Number of samples.
            
        Returns:
            Samples at time t.
        """
        return self.as_process().sample_marginal(key, t, num_samples)


# =============================================================================
# Solver Base Class
# =============================================================================

class SBSolver(abc.ABC):
    """Abstract base class for Schrödinger Bridge solvers.
    
    All solver implementations must inherit from this class and implement
    the required abstract methods.
    
    Attributes:
        problem: The SB problem to solve.
        config: Solver configuration.
        integrator: Time integrator.
        invariant_checker: Diagnostics helper.
    """
    
    def __init__(
        self,
        problem: SBProblem,
        config: Optional[SolverConfig] = None,
        integrator_type: IntegratorType = IntegratorType.EULER_MARUYAMA,
    ):
        """Initialize solver.
        
        Args:
            problem: SB problem specification.
            config: Solver configuration.
            integrator_type: Type of time integrator.
        """
        self.problem = problem
        self.config = config or SolverConfig(time_grid=problem.time_grid)
        self.integrator = create_integrator(integrator_type)
        self.invariant_checker = InvariantChecker()
        
        # State
        self._params: Optional[Params] = None
        self._is_trained: bool = False
    
    @property
    @abc.abstractmethod
    def solver_type(self) -> SolverType:
        """Return the solver type."""
        pass
    
    @property
    @abc.abstractmethod
    def representation_type(self) -> RepresentationType:
        """Return the representation type used by this solver."""
        pass
    
    @property
    def is_neural(self) -> bool:
        """Whether this solver uses neural networks."""
        return self.representation_type in {
            RepresentationType.SCORE,
            RepresentationType.CONTROL,
        }
    
    @abc.abstractmethod
    def init_params(self, key: PRNGKey) -> Params:
        """Initialize solver parameters.
        
        Args:
            key: JAX random key.
            
        Returns:
            Initial parameters.
        """
        pass
    
    @abc.abstractmethod
    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: Any,
        batch_size: int,
    ) -> Tuple[Params, Any, Dict[str, Scalar]]:
        """Perform a single training step.
        
        Args:
            key: JAX random key.
            params: Current parameters.
            opt_state: Optimizer state.
            batch_size: Batch size.
            
        Returns:
            (new_params, new_opt_state, metrics)
        """
        pass
    
    @abc.abstractmethod
    def extract_drift(self, params: Params) -> DriftFn:
        """Extract the forward drift function from parameters.
        
        The drift incorporates the learned bridge correction:
            b*(x,t) = b_ref(x,t) + σ²(t) × (learned term)
        
        Args:
            params: Trained parameters.
            
        Returns:
            Forward drift function.
        """
        pass
    
    def train(
        self,
        key: PRNGKey,
        training_config: Optional[TrainingConfig] = None,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> SolverResult:
        """Train the solver.
        
        Args:
            key: JAX random key.
            training_config: Training configuration.
            callback: Optional callback called each iteration.
            
        Returns:
            Training result with final parameters and diagnostics.
        """
        config = training_config or TrainingConfig()
        
        # Initialize
        k1, k2 = jax.random.split(key)
        params = self.init_params(k1)
        opt_state = self._init_optimizer(params)
        
        loss_history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for step in range(config.num_iterations):
            k2, step_key = jax.random.split(k2)
            
            params, opt_state, metrics = self.train_step(
                step_key, params, opt_state, config.batch_size
            )
            
            loss = metrics.get('loss', 0.0)
            loss_history.append(float(loss))
            
            # Logging
            if self.config.verbose >= 1 and step % config.eval_every == 0:
                print(f"Step {step}: loss = {loss:.6f}")
            
            # Callback
            if callback is not None:
                callback(step, metrics)
            
            # Early stopping
            if loss < best_loss - config.min_delta:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config.patience:
                if self.config.verbose >= 1:
                    print(f"Early stopping at step {step}")
                break
        
        # Store trained params
        self._params = params
        self._is_trained = True
        
        # Run diagnostics
        diagnostics = self._run_diagnostics(key, params)
        
        return SolverResult(
            params=params,
            loss_history=jnp.array(loss_history),
            diagnostics=diagnostics,
            metadata={
                'converged': patience_counter < config.patience,
                'final_step': step,
                'solver_type': self.solver_type.name,
            },
        )
    
    def solve(
        self,
        key: PRNGKey,
        training_config: Optional[TrainingConfig] = None,
    ) -> SBSolution:
        """Train solver and return solution object.
        
        Args:
            key: JAX random key.
            training_config: Training configuration.
            
        Returns:
            Solution object for sampling and evaluation.
        """
        result = self.train(key, training_config)
        metadata = dict(result.metadata)
        metadata.setdefault('integrator_type', self.integrator.type.name)
        
        # Create solution
        solution = SBSolution(
            problem=self.problem,
            solver_type=self.solver_type,
            params=result.params,
            representation=self.representation_type,
            metadata=metadata,
        )
        
        # Set drift functions
        solution._integrator = self.integrator
        solution._forward_drift = self.extract_drift(result.params)
        if hasattr(self, 'extract_backward_drift'):
            try:
                solution._backward_drift = self.extract_backward_drift(result.params)
            except (AttributeError, NotImplementedError, ValueError, KeyError):
                pass
        if hasattr(self, 'extract_score'):
            try:
                solution._score_fn = self.extract_score(result.params)
            except (AttributeError, NotImplementedError, ValueError, KeyError):
                pass
        elif hasattr(self, 'get_score_fn'):
            try:
                solution._score_fn = self.get_score_fn(result.params)
            except (AttributeError, NotImplementedError, ValueError, KeyError, TypeError):
                pass
        
        return solution
    
    def sample(
        self,
        key: PRNGKey,
        num_samples: int,
        params: Optional[Params] = None,
        x0: Optional[Array] = None,
    ) -> TrajectoryBatch:
        """Sample trajectories using current or provided parameters.
        
        Args:
            key: JAX random key.
            num_samples: Number of trajectories.
            params: Parameters to use (uses trained if None).
            x0: Initial points (samples from source if None).
            
        Returns:
            Batch of trajectories.
        """
        if params is None:
            if not self._is_trained:
                raise ValueError("Solver not trained. Call train() first or provide params.")
            params = self._params
        
        k1, k2 = jax.random.split(key)
        
        if x0 is None:
            x0 = self.problem.sample_source(k1, num_samples)
        
        drift = self.extract_drift(params)
        diffusion = self.problem.sigma
        
        return self.integrator.integrate(
            k2, x0, self.problem.time_grid, drift, diffusion, True
        )
    
    def _init_optimizer(self, params: Params) -> Any:
        """Initialize optimizer state.
        
        Override in subclasses for custom optimizers.
        """
        from ..networks import init_adam
        return init_adam(params)
    
    def _run_diagnostics(
        self,
        key: PRNGKey,
        params: Params,
    ) -> DiagnosticReport:
        """Run diagnostics on trained model."""
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Sample trajectories
        trajectories = self.sample(k1, 500, params)
        
        # Reference samples
        source_ref = self.problem.sample_source(k2, 500)
        target_ref = self.problem.sample_target(k3, 500)
        
        return self.invariant_checker.check_all(
            trajectories, source_ref, target_ref, key
        )


# =============================================================================
# Representation Protocol
# =============================================================================

class Representation(abc.ABC):
    """Abstract base class for SB representations.
    
    Different solvers use different internal representations:
    - Score: ∇log p_t
    - Control: Optimal control u(x,t)
    - Potential: Schrödinger potentials ψ, ψ̂
    - Kernel: RKHS embedding
    """
    
    @property
    @abc.abstractmethod
    def type(self) -> RepresentationType:
        pass
    
    @abc.abstractmethod
    def to_drift(
        self,
        reference_drift: DriftFn,
        diffusion: Callable[[Scalar], Scalar],
    ) -> DriftFn:
        """Convert representation to drift function.
        
        Args:
            reference_drift: Reference SDE drift.
            diffusion: Diffusion coefficient function.
            
        Returns:
            Forward drift for the bridge SDE.
        """
        pass


class ScoreRepresentation(Representation):
    """Score function representation: ∇log p_t(x)."""
    
    def __init__(self, score_fn: Callable[[Array, Scalar], Array]):
        self.score_fn = score_fn
    
    @property
    def type(self) -> RepresentationType:
        return RepresentationType.SCORE
    
    def to_drift(
        self,
        reference_drift: DriftFn,
        diffusion: Callable[[Scalar], Scalar],
    ) -> DriftFn:
        def drift(x: Array, t: Scalar) -> Array:
            sigma = diffusion(t)
            return reference_drift(x, t) + sigma ** 2 * self.score_fn(x, t)
        return drift


class ControlRepresentation(Representation):
    """Optimal control representation: u(x,t)."""
    
    def __init__(self, control_fn: Callable[[Array, Scalar], Array]):
        self.control_fn = control_fn
    
    @property
    def type(self) -> RepresentationType:
        return RepresentationType.CONTROL
    
    def to_drift(
        self,
        reference_drift: DriftFn,
        diffusion: Callable[[Scalar], Scalar],
    ) -> DriftFn:
        def drift(x: Array, t: Scalar) -> Array:
            sigma = diffusion(t)
            return reference_drift(x, t) + sigma * self.control_fn(x, t)
        return drift


class PotentialRepresentation(Representation):
    """Schrödinger potential representation: ψ(x,t)."""
    
    def __init__(
        self,
        potential_fn: Callable[[Array, Scalar], Array],
        potential_grad_fn: Callable[[Array, Scalar], Array],
    ):
        self.potential_fn = potential_fn
        self.potential_grad_fn = potential_grad_fn
    
    @property
    def type(self) -> RepresentationType:
        return RepresentationType.POTENTIAL
    
    def to_drift(
        self,
        reference_drift: DriftFn,
        diffusion: Callable[[Scalar], Scalar],
    ) -> DriftFn:
        def drift(x: Array, t: Scalar) -> Array:
            sigma = diffusion(t)
            grad_log_psi = self.potential_grad_fn(x, t)  # ∇log ψ
            return reference_drift(x, t) + sigma ** 2 * grad_log_psi
        return drift


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'BridgeProcess',
    'SBSolution',
    'SBSolver',
    'Representation',
    'ScoreRepresentation',
    'ControlRepresentation',
    'PotentialRepresentation',
]
