"""Schrödinger Bridge Solvers.

This package provides multiple fundamentally distinct approaches to solving
the Schrödinger Bridge problem:

Neural Network Based:
- ScoreBasedSolver: Denoising score matching
- IPFDSBSolver: DSB-style IPF with forward/backward mean maps
- DSBMSolver: Diffusion Schrodinger Bridge Matching
- FBSDESolver: Forward-Backward SDE / stochastic optimal control
- IMFSolver: Iterative Markovian Fitting

Non-Neural Network:
- DoobHTransformSolver: Doob h-transform (potential-based)
- RKHSSolver: Kernel methods (RKHS representation)
- IPFSolver: Iterative Proportional Fitting (Sinkhorn)

Each solver implements the SBSolver interface but uses a distinct
internal representation (score, drift, control, potential, kernel).
"""

from .base import (
    BridgeProcess,
    SBSolver,
    SBSolution,
    Representation,
    ScoreRepresentation,
    ControlRepresentation,
    PotentialRepresentation,
)

from .score_based import ScoreBasedSolver, ScoreBasedConfig
from .ipf_dsb import IPFDSBSolver, IPFDSBConfig, DSBSolver, DSBConfig
from .dsbm import DSBMSolver, DSBMConfig
from .malliavin import MalliavinScoreSolver, MalliavinBridgeSolver, MalliavinConfig
from .fbsde import FBSDESolver, FBSDEConfig, FBSDESolution
from .imf import IMFSolver, IMFConfig
from .rkhs import RKHSSolver, RKHSConfig

# Optional solvers may depend on extras that are not installed.
try:
    from .doob import DoobHTransformSolver, DoobConfig
except ImportError:
    DoobHTransformSolver = None
    DoobConfig = None

try:
    from .ipf import IPFSolver, IPFConfig
except ImportError:
    IPFSolver = None
    IPFConfig = None

__all__ = [
    # Base classes
    'BridgeProcess',
    'SBSolver',
    'SBSolution',
    'Representation',
    'ScoreRepresentation',
    'ControlRepresentation',
    'PotentialRepresentation',
    # Neural solvers
    'ScoreBasedSolver',
    'ScoreBasedConfig',
    'IPFDSBSolver',
    'IPFDSBConfig',
    'DSBSolver',
    'DSBConfig',
    'DSBMSolver',
    'DSBMConfig',
    'MalliavinScoreSolver',
    'MalliavinBridgeSolver',
    'MalliavinConfig',
    'FBSDESolver',
    'FBSDEConfig',
    'FBSDESolution',
    'IMFSolver',
    'IMFConfig',
    # Non-neural solvers
    'RKHSSolver',
    'RKHSConfig',
    'DoobHTransformSolver',
    'DoobConfig',
    'IPFSolver',
    'IPFConfig',
]
