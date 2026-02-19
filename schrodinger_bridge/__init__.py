"""Schrödinger Bridge Library for JAX.

A production-grade implementation of continuous-time Schrödinger Bridges
with multiple fundamentally distinct solver methods.

The Schrödinger Bridge problem finds the stochastic process P* that minimizes:
    
    P* = argmin_{P} KL(P || P_ref)
    
subject to marginal constraints P_0 = μ_0 and P_1 = μ_1.

Key Features:
=============
- Multiple solver methods: IPF, IMF, Score-Based, Doob h-transform, RKHS, FBSDE
- Distinct representations: Score, Control, Potential, Kernel, Particle
- Comprehensive diagnostics: Mass conservation, marginal consistency, KL evolution
- Continuous time API with flexible internal discretization
- Support for CPU, GPU, and TPU (via JAX)
- Non-neural options for interpretable solutions (Doob, RKHS)
- Visualization with GIF export

Quick Start:
============
    >>> import jax
    >>> from schrodinger_bridge import (
    ...     SBProblem, BrownianMotion, GaussianDistribution, TwoMoonsDistribution,
    ...     TimeGrid, ScoreBasedSolver, create_transport_gif
    ... )
    >>> 
    >>> # Define problem
    >>> problem = SBProblem(
    ...     reference=BrownianMotion(sigma=0.5, dim=2),
    ...     source=GaussianDistribution(dim=2),
    ...     target=TwoMoonsDistribution(),
    ...     time_grid=TimeGrid(num_steps=50),
    ... )
    >>> 
    >>> # Solve
    >>> solver = ScoreBasedSolver(problem)
    >>> result = solver.train(jax.random.PRNGKey(0))
    >>> 
    >>> # Sample and visualize
    >>> trajectories = solver.sample(jax.random.PRNGKey(1), num_samples=100)
    >>> create_transport_gif(trajectories, save_path="transport.gif")

Available Solvers:
==================
Neural Network Based:
- ScoreBasedSolver: Denoising score matching (fastest to train)
- FBSDESolver: Forward-Backward SDE / stochastic optimal control
- IMFSolver: Iterative Markovian Fitting (simulation-free)
- IPFSolver: Iterative Proportional Fitting with neural parameterization

Non-Neural Network:
- DoobHTransformSolver: Analytical for Gaussians, kernel-based otherwise
- RKHSSolver: Pure kernel methods (no neural networks)

Architecture:
=============
    SBProblem
     ├─ reference: ReferenceDynamics (Brownian, OU, VP-SDE, VE-SDE)
     ├─ source: MarginalDistribution
     ├─ target: MarginalDistribution  
     └─ time_grid: TimeGrid
    
    SBSolver
     ├─ representation: Score | Control | Potential | Kernel
     ├─ integrator: EulerMaruyama | Heun | Adaptive | Spectral
     └─ diagnostics: InvariantChecker
    
    SBSolution
     ├─ sample_trajectories()
     ├─ get_forward_drift()
     └─ evaluate_at_time()

Author: Built with Claude (Anthropic)
"""

__version__ = '0.1.0'

# =============================================================================
# Core Types and Structures
# =============================================================================

from .core.types import (
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
    Params,
    # Enums
    DeviceType,
    RepresentationType,
    IntegratorType,
    SolverType,
    # Data structures
    TimeGrid,
    SDECoefficients,
    TrajectoryBatch,
    SolverResult,
    # Diagnostics
    InvariantViolation,
    DiagnosticReport,
    # Configs
    SolverConfig,
    NetworkConfig,
    OptimizerConfig,
    TrainingConfig,
    # Exceptions
    SchrodingerBridgeError,
    InvariantError,
    ConvergenceError,
    DimensionError,
    ConfigurationError,
)

# =============================================================================
# Problem Definition
# =============================================================================

from .core.problem import (
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
    # Problem container
    SBProblem,
    # Factory functions
    create_gaussian_to_gaussian,
    create_gaussian_to_moons,
    create_moons_to_moons,
)

# =============================================================================
# Invariant Checking
# =============================================================================

from .core.invariants import (
    InvariantThresholds,
    InvariantChecker,
    mmd_squared,
    sliced_wasserstein,
    estimate_entropy,
    quick_marginal_check,
    quick_trajectory_check,
)

# =============================================================================
# Integrators
# =============================================================================

from .integrators import (
    Integrator,
    StepResult,
    EulerMaruyama,
    Heun,
    Milstein,
    AdaptiveIntegrator,
    AdaptiveConfig,
    SpectralIntegrator,
    sample_brownian_bridge,
    create_integrator,
)

# =============================================================================
# Neural Networks
# =============================================================================

from .networks import (
    # Embeddings
    sinusoidal_embedding,
    random_fourier_features,
    # Network building
    TimeConditionedMLPConfig,
    init_time_conditioned_mlp,
    time_conditioned_mlp_forward,
    # Specialized networks
    init_score_network,
    score_network_forward,
    init_potential_network,
    potential_network_forward,
    potential_network_gradient,
    # ICNN
    init_icnn_params,
    icnn_forward,
    icnn_gradient,
    # Optimizer
    AdamState,
    init_adam,
    adam_update,
    # Factory convenience
    create_default_factory,
)

# =============================================================================
# Network Factory
# =============================================================================

from .network_factory import (
    NetworkFactory,
    MLPFactory,
    UNetFactory,
    TransformerFactory,
    CustomFactory,
    sanity_check,
)

# =============================================================================
# Kernel Methods
# =============================================================================

from .kernels import (
    # Kernels
    gaussian_kernel,
    laplacian_kernel,
    matern_kernel,
    polynomial_kernel,
    imq_kernel,
    # Gradients
    gaussian_kernel_gradient,
    gaussian_kernel_laplacian,
    # Bandwidth selection
    median_heuristic,
    silverman_bandwidth,
    # KDE and embeddings
    KernelDensityEstimate,
    fit_kde,
    KernelMeanEmbedding,
    # Regression
    kernel_ridge_regression,
    kernel_score_estimation,
)

# =============================================================================
# Solvers
# =============================================================================

from .solvers import (
    # Base classes
    SBSolver,
    SBSolution,
    Representation,
    ScoreRepresentation,
    ControlRepresentation,
    PotentialRepresentation,
    # Neural solvers
    ScoreBasedSolver,
    ScoreBasedConfig,
    FBSDESolver,
    FBSDEConfig,
    FBSDESolution,
    IMFSolver,
    IMFConfig,
    # Non-neural solvers
    RKHSSolver,
    RKHSConfig,
    DoobHTransformSolver,
    DoobConfig,
    IPFSolver,
    IPFConfig,
)

# =============================================================================
# Visualization
# =============================================================================

from .visualization import (
    VisualizationConfig,
    plot_marginals,
    plot_trajectories,
    plot_diagnostics,
    create_transport_gif,
    create_comparison_gif,
    plot_velocity_field,
)

# =============================================================================
# Device Utilities
# =============================================================================

from .devices import (
    DeviceKind,
    DeviceInfo,
    get_device_info,
    print_device_info,
    place_on_device,
    get_default_device,
    ensure_on_device,
    clear_cache,
    estimate_memory_usage,
    check_memory_for_batch,
    shard_batch,
    unshard_batch,
    pmap_with_devices,
    jit_with_device,
    process_in_batches,
    split_key_for_devices,
)

# =============================================================================
# OTT-JAX Integration
# =============================================================================

from .ott_integration import (
    is_ott_available,
    OTConfig,
    compute_ot_coupling,
    compute_ot_cost,
    compute_sinkhorn_divergence,
    sinkhorn_coupling_fallback,
    get_ot_paired_samples,
    get_ot_barycentric_interpolation,
    OTCoupledSampler,
    create_ot_coupled_sampler,
    ot_loss,
    sinkhorn_loss,
)

# =============================================================================
# Marginal Schrödinger Bridge Extension
# =============================================================================

from .marginal_sb import (
    MarginalConstraint,
    MarginalSBProblem,
    MarginalSBConfig,
    MarginalSBSolver,
    create_marginal_sb_problem,
    solve_marginal_sb,
    interpolate_marginals,
)

# =============================================================================
# Marginal Schrödinger Bridge
# =============================================================================

from .marginal_sb import (
    MarginalConstraint,
    MarginalSBProblem,
    MarginalSBConfig,
    MarginalSBSolver,
    create_marginal_sb_problem,
    solve_marginal_sb,
    interpolate_marginals,
)

# =============================================================================
# Convenience Functions
# =============================================================================

def list_solvers() -> dict:
    """List available solvers and their properties."""
    return {
        'ScoreBasedSolver': {
            'representation': 'score',
            'neural': True,
            'description': 'Denoising score matching on bridge paths',
        },
        'FBSDESolver': {
            'representation': 'control',
            'neural': True,
            'description': 'Forward-Backward SDE / optimal control',
        },
        'IMFSolver': {
            'representation': 'score',
            'neural': True,
            'description': 'Iterative Markovian Fitting (simulation-free)',
        },
        'IPFSolver': {
            'representation': 'score',
            'neural': True,
            'description': 'Iterative Proportional Fitting (Sinkhorn)',
        },
        'DoobHTransformSolver': {
            'representation': 'potential',
            'neural': False,
            'description': 'Doob h-transform (analytical/kernel)',
        },
        'RKHSSolver': {
            'representation': 'kernel',
            'neural': False,
            'description': 'RKHS kernel methods (non-parametric)',
        },
    }


def get_solver(
    name: str,
    problem: SBProblem,
    **kwargs,
) -> SBSolver:
    """Get solver by name.
    
    Args:
        name: Solver name (e.g., 'score', 'fbsde', 'doob', 'rkhs').
        problem: SB problem specification.
        **kwargs: Additional solver arguments.
        
    Returns:
        Solver instance.
    """
    name = name.lower()
    
    solvers = {
        'score': ScoreBasedSolver,
        'score_based': ScoreBasedSolver,
        'fbsde': FBSDESolver,
        'imf': IMFSolver,
        'ipf': IPFSolver,
        'doob': DoobHTransformSolver,
        'rkhs': RKHSSolver,
        'kernel': RKHSSolver,
    }
    
    if name not in solvers:
        raise ValueError(f"Unknown solver: {name}. Available: {list(solvers.keys())}")
    
    return solvers[name](problem, **kwargs)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Version
    '__version__',
    # Types
    'Array', 'PRNGKey', 'Scalar', 'PyTree',
    'DriftFn', 'DiffusionFn', 'ScoreFn', 'ControlFn', 'PotentialFn', 'Params',
    # Enums
    'DeviceType', 'RepresentationType', 'IntegratorType', 'SolverType',
    # Data structures
    'TimeGrid', 'SDECoefficients', 'TrajectoryBatch', 'SolverResult',
    'InvariantViolation', 'DiagnosticReport',
    # Configs
    'SolverConfig', 'NetworkConfig', 'OptimizerConfig', 'TrainingConfig',
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
    'quick_marginal_check', 'quick_trajectory_check',
    # Integrators
    'Integrator', 'StepResult', 'EulerMaruyama', 'Heun', 'Milstein',
    'AdaptiveIntegrator', 'AdaptiveConfig', 'SpectralIntegrator',
    'sample_brownian_bridge', 'create_integrator',
    # Networks
    'sinusoidal_embedding', 'random_fourier_features',
    'TimeConditionedMLPConfig', 'init_time_conditioned_mlp', 'time_conditioned_mlp_forward',
    'init_score_network', 'score_network_forward',
    'init_potential_network', 'potential_network_forward', 'potential_network_gradient',
    'init_icnn_params', 'icnn_forward', 'icnn_gradient',
    'AdamState', 'init_adam', 'adam_update',
    'create_default_factory',
    # Network Factory
    'NetworkFactory', 'MLPFactory', 'UNetFactory', 'TransformerFactory',
    'CustomFactory', 'sanity_check',
    # Kernels
    'gaussian_kernel', 'laplacian_kernel', 'matern_kernel', 'polynomial_kernel', 'imq_kernel',
    'gaussian_kernel_gradient', 'gaussian_kernel_laplacian',
    'median_heuristic', 'silverman_bandwidth',
    'KernelDensityEstimate', 'fit_kde', 'KernelMeanEmbedding',
    'kernel_ridge_regression', 'kernel_score_estimation',
    # Solvers
    'SBSolver', 'SBSolution',
    'Representation', 'ScoreRepresentation', 'ControlRepresentation', 'PotentialRepresentation',
    'ScoreBasedSolver', 'ScoreBasedConfig',
    'FBSDESolver', 'FBSDEConfig', 'FBSDESolution',
    'IMFSolver', 'IMFConfig',
    'RKHSSolver', 'RKHSConfig',
    'DoobHTransformSolver', 'DoobConfig',
    'IPFSolver', 'IPFConfig',
    # Visualization
    'VisualizationConfig',
    'plot_marginals', 'plot_trajectories', 'plot_diagnostics',
    'create_transport_gif', 'create_comparison_gif', 'plot_velocity_field',
    # Convenience
    'list_solvers', 'get_solver',
]
