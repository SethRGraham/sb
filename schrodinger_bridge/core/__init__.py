"""Core module for Schrödinger Bridge library.

Provides foundational types, problem definitions, and invariant checking.
"""

from .types import (
    # Type aliases
    Array,
    PRNGKey,
    Scalar,
    PyTree,
    DriftFn,
    DiffusionFn,
    ScoreFn,
    ControlFn,
    PotentialFn,
    DensityFn,
    Params,
    OptState,
    # Enums
    DeviceType,
    RepresentationType,
    IntegratorType,
    SolverType,
    # Data structures
    TimeGrid,
    SDECoefficients,
    InvariantViolation,
    DiagnosticReport,
    TrajectoryBatch,
    SolverResult,
    # Configs
    SolverConfig,
    NetworkConfig,
    OptimizerConfig,
    TrainingConfig,
    # Protocols
    Sampler,
    DensityEvaluator,
    Representation,
    # Exceptions
    SchrodingerBridgeError,
    InvariantError,
    ConvergenceError,
    DimensionError,
    ConfigurationError,
)

from .problem import (
    # Reference dynamics
    ReferenceDynamics,
    BrownianMotion,
    OrnsteinUhlenbeck,
    VarianceExploding,
    VariancePreserving,
    # Marginal distributions
    MarginalDistribution,
    GaussianDistribution,
    EmpiricalDistribution,
    MixtureDistribution,
    TwoMoonsDistribution,
    SwissRollDistribution,
    # Problem definition
    SBProblem,
    # Factories
    create_gaussian_to_gaussian,
    create_gaussian_to_moons,
    create_moons_to_moons,
)

from .invariants import (
    InvariantThresholds,
    InvariantChecker,
    mmd_squared,
    sliced_wasserstein,
    estimate_entropy,
    check_continuity_equation,
    quick_marginal_check,
    quick_trajectory_check,
)

__all__ = [
    # Types
    'Array', 'PRNGKey', 'Scalar', 'PyTree',
    'DriftFn', 'DiffusionFn', 'ScoreFn', 'ControlFn', 'PotentialFn', 'DensityFn',
    'Params', 'OptState',
    # Enums
    'DeviceType', 'RepresentationType', 'IntegratorType', 'SolverType',
    # Data structures
    'TimeGrid', 'SDECoefficients', 'InvariantViolation', 'DiagnosticReport',
    'TrajectoryBatch', 'SolverResult',
    # Configs
    'SolverConfig', 'NetworkConfig', 'OptimizerConfig', 'TrainingConfig',
    # Protocols
    'Sampler', 'DensityEvaluator', 'Representation',
    # Exceptions
    'SchrodingerBridgeError', 'InvariantError', 'ConvergenceError',
    'DimensionError', 'ConfigurationError',
    # Reference dynamics
    'ReferenceDynamics', 'BrownianMotion', 'OrnsteinUhlenbeck',
    'VarianceExploding', 'VariancePreserving',
    # Marginals
    'MarginalDistribution', 'GaussianDistribution', 'EmpiricalDistribution',
    'MixtureDistribution', 'TwoMoonsDistribution', 'SwissRollDistribution',
    # Problem
    'SBProblem',
    'create_gaussian_to_gaussian', 'create_gaussian_to_moons', 'create_moons_to_moons',
    # Invariants
    'InvariantThresholds', 'InvariantChecker',
    'mmd_squared', 'sliced_wasserstein', 'estimate_entropy',
    'check_continuity_equation', 'quick_marginal_check', 'quick_trajectory_check',
]
