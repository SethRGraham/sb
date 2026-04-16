"""Koopman-Accelerated Schrödinger Bridge Solvers.

This subpackage provides solvers that leverage Koopman operator theory,
Dynamic Mode Decomposition (DMD), and sparse identification (SINDy/HyperSINDy)
to accelerate Schrödinger Bridge computation.

Key Mathematical Insight:
========================
The Schrödinger potential ψ(x,t) satisfies the backward Kolmogorov equation:
    ∂_t ψ + b_ref · ∇ψ + (σ²/2) Δψ = 0

This is precisely the equation whose solutions are related to Koopman eigenfunctions
for the reference SDE. By approximating these eigenfunctions via data-driven methods
(EDMD, gEDMD), we can warm-start or directly compute SB solutions.

Available Solvers:
=================
- EDMDWarmStartSolver: Use EDMD to warm-start neural SB solvers
- GEDMDSolver: Generator EDMD for direct SDE drift identification  
- HyperSINDySolver: Sparse stochastic dynamics with hypernetworks
- KoopmanHybridSolver: Combines all methods in a multi-stage pipeline

References:
==========
- Williams et al. (2015) "A Data-Driven Approximation of the Koopman Operator"
- Klus et al. (2020) "Data-driven approximation of the Koopman generator"
- Jacobs et al. (2023) "HyperSINDy: Deep Generative Modeling of Stochastic Dynamics"
- Chen et al. (2016) "Optimal Transport and Schrödinger Bridges: A Control Viewpoint"
"""

from .dictionary import (
    Dictionary,
    PolynomialDictionary,
    FourierDictionary,
    RBFDictionary,
    HermiteDictionary,
    CompositeDictionary,
    build_adaptive_dictionary,
)

from .edmd import (
    EDMDResult,
    edmd,
    extended_dmd,
    kernel_edmd,
    compute_koopman_modes,
)

from .optdmd import (
    OptDMDResult,
    BagOptDMDResult,
    standard_dmd,
    optdmd,
    bagging_optdmd,
    forward_backward_dmd,
    tls_dmd,
    optdmd_from_trajectories,
)

from .empirical import (
    EmpiricalReferenceDynamics,
    KernelDriftModel,
    LocalLinearDriftModel,
    create_empirical_reference,
    fit_drift_model,
)

from .discrete_ipf import (
    DiscreteIPFConfig,
    DiscreteIPFResult,
    BridgePathResult,
    DiscreteIPFKoopmanSolver,
    sinkhorn_log_domain,
    sinkhorn_standard,
    build_koopman_kernel,
    create_discrete_ipf_solver,
)

from .gedmd import (
    GEDMDResult,
    gedmd,
    gedmd_sde_identification,
    extract_drift_diffusion,
)

from .hypersindy import (
    HyperSINDyConfig,
    HyperSINDyParams,
    init_hypersindy,
    hypersindy_forward,
    hypersindy_loss,
    train_hypersindy,
    SparsityMask,
)

from .solvers import (
    EDMDWarmStartSolver,
    EDMDWarmStartConfig,
    GEDMDSolver,
    GEDMDConfig,
    HyperSINDySolver,
    HyperSINDySolverConfig,
    KoopmanHybridSolver,
    KoopmanHybridConfig,
)

__all__ = [
    # Dictionary
    'Dictionary',
    'PolynomialDictionary',
    'FourierDictionary',
    'RBFDictionary',
    'HermiteDictionary',
    'CompositeDictionary',
    'build_adaptive_dictionary',
    # EDMD
    'EDMDResult',
    'edmd',
    'extended_dmd',
    'kernel_edmd',
    'compute_koopman_modes',
    # optDMD (noise-robust)
    'OptDMDResult',
    'BagOptDMDResult',
    'standard_dmd',
    'optdmd',
    'bagging_optdmd',
    'forward_backward_dmd',
    'tls_dmd',
    'optdmd_from_trajectories',
    # Empirical reference
    'EmpiricalReferenceDynamics',
    'KernelDriftModel',
    'LocalLinearDriftModel',
    'create_empirical_reference',
    'fit_drift_model',
    # Discrete-time IPF-Koopman
    'DiscreteIPFConfig',
    'DiscreteIPFResult',
    'BridgePathResult',
    'DiscreteIPFKoopmanSolver',
    'sinkhorn_log_domain',
    'sinkhorn_standard',
    'build_koopman_kernel',
    'create_discrete_ipf_solver',
    # gEDMD
    'GEDMDResult',
    'gedmd',
    'gedmd_sde_identification',
    'extract_drift_diffusion',
    # HyperSINDy
    'HyperSINDyConfig',
    'HyperSINDyParams',
    'init_hypersindy',
    'hypersindy_forward',
    'hypersindy_loss',
    'train_hypersindy',
    'SparsityMask',
    # Solvers
    'EDMDWarmStartSolver',
    'EDMDWarmStartConfig',
    'GEDMDSolver',
    'GEDMDConfig',
    'HyperSINDySolver',
    'HyperSINDySolverConfig',
    'KoopmanHybridSolver',
    'KoopmanHybridConfig',
]
