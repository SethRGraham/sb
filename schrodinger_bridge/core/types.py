"""Core type definitions for Schrödinger Bridge library.

This module provides foundational type aliases, protocols, and dataclasses
used throughout the library. All components reference these shared types
to ensure consistency and interoperability.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import jax
import jax.numpy as jnp

# =============================================================================
# Fundamental Type Aliases
# =============================================================================

Array = jnp.ndarray
PRNGKey = Any  # jax.random.PRNGKey
Scalar = Union[float, Array]
PyTree = Any  # JAX pytree

# Function signatures
DriftFn = Callable[[Array, Scalar], Array]
DiffusionFn = Callable[[Scalar], Union[Scalar, Array]]
ScoreFn = Callable[[Array, Scalar], Array]
ControlFn = Callable[[Array, Scalar], Array]
PotentialFn = Callable[[Array, Scalar], Array]
DensityFn = Callable[[Array, Scalar], Array]

# Parameter types
Params = Dict[str, Any]
OptState = Any


# =============================================================================
# Enumerations
# =============================================================================

class DeviceType(Enum):
    """Supported compute devices."""
    CPU = auto()
    GPU = auto()
    TPU = auto()
    
    @classmethod
    def from_string(cls, s: str) -> 'DeviceType':
        return cls[s.upper()]
    
    @classmethod
    def detect(cls) -> 'DeviceType':
        """Detect current JAX backend."""
        devices = jax.devices()
        if devices:
            platform = devices[0].platform
            if platform == 'gpu':
                return cls.GPU
            elif platform == 'tpu':
                return cls.TPU
        return cls.CPU


class RepresentationType(Enum):
    """Types of SB representations."""
    PARTICLE = auto()      # Sample-based
    SCORE = auto()         # ∇log p_t
    CONTROL = auto()       # Optimal control u(x,t)
    DENSITY = auto()       # p(x,t) directly
    KERNEL = auto()        # RKHS representation
    POTENTIAL = auto()     # Schrödinger potentials ψ, ψ̂


class IntegratorType(Enum):
    """Types of time integrators."""
    EULER_MARUYAMA = auto()
    HEUN = auto()
    MILSTEIN = auto()
    ADAPTIVE = auto()
    SPECTRAL = auto()


class SolverType(Enum):
    """Available solver methods."""
    IPF = auto()           # Iterative Proportional Fitting
    IMF = auto()           # Iterative Markovian Fitting
    SCORE_BASED = auto()   # Score matching
    DOOB = auto()          # Doob h-transform
    RKHS = auto()          # Kernel-based
    FBSDE = auto()         # Forward-Backward SDE


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass(frozen=True)
class TimeGrid:
    """Immutable time discretization specification.
    
    Attributes:
        t0: Initial time.
        t1: Terminal time.
        num_steps: Number of discretization steps.
    """
    t0: float = 0.0
    t1: float = 1.0
    num_steps: int = 100
    
    @property
    def dt(self) -> float:
        """Time step size."""
        return (self.t1 - self.t0) / self.num_steps
    
    @property
    def times(self) -> Array:
        """Array of time points."""
        return jnp.linspace(self.t0, self.t1, self.num_steps + 1)
    
    def __post_init__(self):
        if self.t1 <= self.t0:
            raise ValueError(f"t1 ({self.t1}) must be greater than t0 ({self.t0})")
        if self.num_steps < 1:
            raise ValueError(f"num_steps ({self.num_steps}) must be positive")


@dataclass
class SDECoefficients:
    """Coefficients defining an SDE: dX = b(X,t)dt + σ(X,t)dW.
    
    Attributes:
        drift: Drift function b(x, t) -> dx/dt contribution.
        diffusion: Diffusion function σ(x, t) or σ(t).
        is_diffusion_scalar: Whether diffusion is state-independent.
    """
    drift: DriftFn
    diffusion: DiffusionFn
    is_diffusion_scalar: bool = True
    
    def __call__(self, x: Array, t: Scalar) -> Tuple[Array, Union[Scalar, Array]]:
        """Evaluate both coefficients."""
        return self.drift(x, t), self.diffusion(t)


@dataclass
class InvariantViolation:
    """Record of a violated SB invariant.
    
    Attributes:
        name: Name of the invariant.
        expected: Expected value or bound.
        actual: Actual computed value.
        severity: 'warning' or 'error'.
        message: Human-readable description.
    """
    name: str
    expected: Any
    actual: Any
    severity: str  # 'warning' or 'error'
    message: str
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.name}: {self.message}"


@dataclass
class DiagnosticReport:
    """Collection of diagnostic measurements.
    
    Attributes:
        mass_conservation: Mass conservation error over time.
        marginal_error_source: Error at source marginal.
        marginal_error_target: Error at target marginal.
        kl_evolution: KL divergence evolution.
        entropy_evolution: Entropy evolution.
        violations: List of invariant violations.
        metadata: Additional diagnostic data.
    """
    mass_conservation: Optional[Array] = None
    marginal_error_source: Optional[float] = None
    marginal_error_target: Optional[float] = None
    kl_evolution: Optional[Array] = None
    entropy_evolution: Optional[Array] = None
    violations: List[InvariantViolation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_errors(self) -> bool:
        return any(v.severity == 'error' for v in self.violations)
    
    @property
    def has_warnings(self) -> bool:
        return any(v.severity == 'warning' for v in self.violations)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=== Diagnostic Report ==="]
        
        if self.mass_conservation is not None:
            max_err = float(jnp.max(jnp.abs(self.mass_conservation)))
            lines.append(f"Mass conservation max error: {max_err:.2e}")
        
        if self.marginal_error_source is not None:
            lines.append(f"Source marginal error: {self.marginal_error_source:.2e}")
        
        if self.marginal_error_target is not None:
            lines.append(f"Target marginal error: {self.marginal_error_target:.2e}")
        
        if self.violations:
            lines.append(f"\nViolations ({len(self.violations)}):")
            for v in self.violations:
                lines.append(f"  {v}")
        
        return "\n".join(lines)


# =============================================================================
# Protocol Definitions (Interfaces)
# =============================================================================

@runtime_checkable
class Sampler(Protocol):
    """Protocol for sampling from distributions."""
    
    def sample(self, key: PRNGKey, num_samples: int) -> Array:
        """Draw samples from the distribution.
        
        Args:
            key: JAX random key.
            num_samples: Number of samples to draw.
            
        Returns:
            Array of shape [num_samples, dim].
        """
        ...
    
    @property
    def dim(self) -> int:
        """Dimension of samples."""
        ...


@runtime_checkable  
class DensityEvaluator(Protocol):
    """Protocol for evaluating probability densities."""
    
    def log_prob(self, x: Array, t: Scalar) -> Array:
        """Compute log probability density.
        
        Args:
            x: Points to evaluate, shape [batch, dim].
            t: Time.
            
        Returns:
            Log probabilities, shape [batch].
        """
        ...
    
    def prob(self, x: Array, t: Scalar) -> Array:
        """Compute probability density."""
        return jnp.exp(self.log_prob(x, t))


@runtime_checkable
class Representation(Protocol):
    """Protocol for SB representations."""
    
    @property
    def type(self) -> RepresentationType:
        """Return the representation type."""
        ...
    
    def to_drift(self, reference_drift: DriftFn, sigma: DiffusionFn) -> DriftFn:
        """Convert representation to a drift function for sampling."""
        ...


# =============================================================================
# Result Containers
# =============================================================================

class TrajectoryBatch(NamedTuple):
    """Batch of sampled trajectories.
    
    Attributes:
        paths: Trajectories of shape [batch, time, dim].
        times: Time points of shape [time].
        log_weights: Optional importance weights, shape [batch].
    """
    paths: Array
    times: Array
    log_weights: Optional[Array] = None
    
    @property
    def batch_size(self) -> int:
        return self.paths.shape[0]
    
    @property
    def num_times(self) -> int:
        return self.paths.shape[1]
    
    @property
    def dim(self) -> int:
        return self.paths.shape[2]
    
    def at_time(self, t_idx: int) -> Array:
        """Get all samples at a specific time index."""
        return self.paths[:, t_idx, :]
    
    @property
    def source_samples(self) -> Array:
        """Samples at t=0."""
        return self.paths[:, 0, :]
    
    @property
    def target_samples(self) -> Array:
        """Samples at t=1."""
        return self.paths[:, -1, :]


class SolverResult(NamedTuple):
    """Result from training a Schrödinger Bridge solver.
    
    Attributes:
        params: Learned parameters (solver-specific).
        loss_history: Training loss over iterations.
        diagnostics: Diagnostic report.
        metadata: Additional solver-specific data.
    """
    params: Any
    loss_history: Array
    diagnostics: DiagnosticReport
    metadata: Dict[str, Any]
    
    @property
    def final_loss(self) -> float:
        if len(self.loss_history) > 0:
            return float(self.loss_history[-1])
        return float('nan')
    
    @property
    def converged(self) -> bool:
        return self.metadata.get('converged', False)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class SolverConfig:
    """Base configuration for all solvers.
    
    Attributes:
        time_grid: Time discretization.
        device: Compute device type.
        dtype: JAX dtype for computations.
        seed: Random seed.
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
    """
    time_grid: TimeGrid = field(default_factory=TimeGrid)
    device: DeviceType = field(default_factory=DeviceType.detect)
    dtype: Any = jnp.float32
    seed: int = 42
    verbose: int = 1
    
    def __post_init__(self):
        # Validate device availability
        detected = DeviceType.detect()
        if self.device == DeviceType.GPU and detected != DeviceType.GPU:
            import warnings
            warnings.warn(f"GPU requested but not available. Using {detected.name}.")
            self.device = detected


@dataclass
class NetworkConfig:
    """Configuration for neural networks.
    
    Attributes:
        hidden_dims: Tuple of hidden layer dimensions.
        activation: Activation function name.
        time_embed_dim: Dimension of time embedding.
        dropout_rate: Dropout rate (0 = no dropout).
        use_layer_norm: Whether to use layer normalization.
    """
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    activation: str = 'swish'
    time_embed_dim: int = 64
    dropout_rate: float = 0.0
    use_layer_norm: bool = True


@dataclass
class OptimizerConfig:
    """Configuration for optimization.
    
    Attributes:
        learning_rate: Base learning rate.
        weight_decay: L2 regularization.
        max_grad_norm: Gradient clipping threshold.
        warmup_steps: Linear warmup steps.
        schedule: Learning rate schedule ('constant', 'cosine', 'linear').
    """
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    schedule: str = 'constant'


@dataclass
class TrainingConfig:
    """Configuration for training loop.
    
    Attributes:
        num_iterations: Total training iterations.
        batch_size: Batch size.
        eval_every: Evaluation frequency.
        checkpoint_every: Checkpoint frequency.
        patience: Early stopping patience.
        min_delta: Minimum improvement for early stopping.
    """
    num_iterations: int = 10000
    batch_size: int = 256
    eval_every: int = 100
    checkpoint_every: int = 1000
    patience: int = 20
    min_delta: float = 1e-6


# =============================================================================
# Exception Classes
# =============================================================================

class SchrodingerBridgeError(Exception):
    """Base exception for Schrödinger Bridge errors."""
    pass


class InvariantError(SchrodingerBridgeError):
    """Raised when a critical invariant is violated."""
    
    def __init__(self, violation: InvariantViolation):
        self.violation = violation
        super().__init__(str(violation))


class ConvergenceError(SchrodingerBridgeError):
    """Raised when a solver fails to converge."""
    
    def __init__(self, message: str, iterations: int, final_loss: float):
        self.iterations = iterations
        self.final_loss = final_loss
        super().__init__(f"{message} (iterations={iterations}, loss={final_loss:.2e})")


class DimensionError(SchrodingerBridgeError):
    """Raised for dimension mismatches."""
    
    def __init__(self, expected: int, got: int, context: str = ""):
        super().__init__(f"Dimension mismatch: expected {expected}, got {got}. {context}")


class ConfigurationError(SchrodingerBridgeError):
    """Raised for invalid configurations."""
    pass
