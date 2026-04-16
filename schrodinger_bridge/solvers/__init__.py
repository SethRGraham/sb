"""Schrödinger Bridge Solvers.

This package provides multiple fundamentally distinct approaches to solving
the Schrödinger Bridge problem:

Neural Network Based:
- ScoreBasedSolver: Denoising score matching
- FBSDESolver: Forward-Backward SDE / stochastic optimal control
- IMFSolver: Iterative Markovian Fitting

Non-Neural Network:
- DoobHTransformSolver: Doob h-transform (potential-based)
- RKHSSolver: Kernel methods (RKHS representation)
- IPFSolver: Iterative Proportional Fitting (Sinkhorn)

Each solver implements the SBSolver interface but uses a distinct
internal representation (score, control, potential, kernel).
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
from .malliavin import MalliavinScoreSolver, MalliavinBridgeSolver, MalliavinConfig
from .fbsde import FBSDESolver, FBSDEConfig, FBSDESolution
from .imf import IMFSolver, IMFConfig
from .rkhs import RKHSSolver, RKHSConfig

# Import existing solvers from transcript
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
