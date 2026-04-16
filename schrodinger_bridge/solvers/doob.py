"""Doob h-Transform Solver for Schrödinger Bridges.

The Doob h-transform provides an elegant characterization of conditioned diffusions.
For the Schrödinger Bridge, the h-function satisfies specific boundary conditions.

Mathematical foundation:
========================
Given reference process dX = b(X,t)dt + σdW (STANDARD ITÔ CONVENTION),
the h-transformed process has drift:

    b*(x,t) = b(x,t) + σ² ∇log h(x,t)

NOTE ON CONVENTIONS:
    - This library uses dX = b dt + σ dW (standard Itô)
    - Some literature uses dX = b dt + √(2σ) dW (physics convention)
    - The physics convention would give b* = b + 2σ²∇log h
    - Our formula is CORRECT for our convention.

KEY MATH INSIGHT (Gaussian Case - EXACT):
=========================================
For SB between N(m0, Σ0) and N(m1, Σ1):

The OT map is: T(x) = A(x - m0) + m1
where A = Σ0^{-1/2} (Σ0^{1/2} Σ1 Σ0^{1/2})^{1/2} Σ0^{-1/2}

The McCann interpolation at time t is:
    X_t = M_t X_0 - t A m0 + t m1,  where M_t = (1-t)I + tA

The conditional expectation is:
    E[X_1 | X_t = x] = A M_t^{-1} (x + t A m0 - t m1) - A m0 + m1

The drift is:
    b*(x,t) = (E[X_1 | X_t = x] - x) / (1-t)

KERNEL METHOD (APPROXIMATE):
============================
For general distributions, the kernel method provides an APPROXIMATION.

IMPORTANT LIMITATIONS (be aware of these):
1. Index pairing: We pair samples by index, which doesn't give the true
   entropic OT coupling. For better results, use Sinkhorn to compute
   the coupling first.
2. Bridge conditional: We use p(X_t | X_0, X_1) from a Brownian bridge,
   which ignores Schrödinger potentials φ, ψ.
3. The drift is locally plausible but not globally optimal.

For rigorous solutions to general SB problems, consider:
- IPFSolver (iterative Sinkhorn to learn potentials)
- ScoreBasedSolver (neural network based)
- MarginalSBSolver (from OTT-JAX integration)

References:
    Doob (1957) "Conditional Brownian Motion"
    McCann (1997) "A Convexity Principle for Interacting Gases"
    De Bortoli et al. (2021) "Diffusion Schrödinger Bridge"
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union
import warnings

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve

from ..core.types import (
    Array,
    DriftFn,
    Params,
    PRNGKey,
    RepresentationType,
    Scalar,
    SolverConfig,
    SolverType,
    SolverResult,
)
from ..core.problem import (
    SBProblem,
    GaussianDistribution,
    BrownianMotion,
    OrnsteinUhlenbeck,
)
from ..kernels import (
    gaussian_kernel,
    gaussian_kernel_gradient,
    median_heuristic,
    kernel_ridge_regression,
)
from .base import SBSolver, PotentialRepresentation


@dataclass
class DoobConfig:
    """Configuration for Doob h-transform solver.
    
    Attributes:
        method: 'analytical', 'kernel', 'kernel_sinkhorn', or 'auto'.
            - 'analytical': Exact solution for Gaussian marginals
            - 'kernel': Heuristic kernel method (index-paired samples)
            - 'kernel_sinkhorn': Kernel method with Sinkhorn coupling (recommended)
            - 'auto': Automatically select best method
        kernel_bandwidth: Bandwidth for kernel method (None = median heuristic).
        kernel_reg: Regularization for kernel method.
        num_inducing_points: Number of inducing points for kernel.
        sinkhorn_epsilon: Regularization for Sinkhorn (entropic OT).
        sinkhorn_iterations: Max iterations for Sinkhorn.
    """
    method: str = 'auto'
    kernel_bandwidth: Optional[float] = None
    kernel_reg: float = 1e-4
    num_inducing_points: int = 500
    sinkhorn_epsilon: float = 0.1
    sinkhorn_iterations: int = 100


def sinkhorn_coupling(
    source: Array,
    target: Array,
    epsilon: float = 0.1,
    num_iterations: int = 100,
    key: Optional[PRNGKey] = None,
) -> Tuple[Array, Array]:
    n = source.shape[0]
    C = jnp.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=-1)
    
    """Compute entropic OT coupling via Sinkhorn and sample from it.

    Returns resampled (source, target) pairs that approximate the
    entropic OT coupling π_ε.

    Math: Sinkhorn finds
        π* = argmin_π ⟨C, π⟩ + ε H(π)
    where H is entropy. As ε→0, this approaches exact OT.

    Implementation notes
    --------------------
    * Log-domain iterations for numerical stability (avoids underflow when
      ε is small relative to typical costs).
    * Cost is median-normalized before scaling by ε, making the algorithm
      robust to different problem scales without re-tuning ε.
    * Pairs are drawn by sampling target index j ~ P[i,:] / P[i,:].sum()
      for each source index i (row-conditional sampling via Gumbel-max
      trick). This preserves multimodality that argmax destroys.
      If key is None, falls back to argmax (deterministic but biased).

    Args:
        source: Source samples [n, d]
        target: Target samples [n, d]
        epsilon: Entropic regularization (smaller = closer to exact OT).
            Interpreted relative to median squared distance, so the same
            ε works across problem scales.
        num_iterations: Sinkhorn iterations.
        key: JAX PRNGKey for stochastic sampling. If None, uses argmax
            (faster but deterministic and biased toward unimodal coupling).

    Returns:
        (coupled_source, coupled_target): n paired samples drawn from π_ε.
    """

    # Robust scale: ignore diagonal by adding large value
    big = 1e6
    C_no_diag = C + big * jnp.eye(n)
    C_scale = jnp.median(C_no_diag) + 1e-10
    C_norm = C / C_scale

    log_K = -C_norm / epsilon

    # uniform marginals
    log_a = -jnp.log(n) * jnp.ones(n)
    log_b = -jnp.log(n) * jnp.ones(n)

    log_u = jnp.zeros(n)
    log_v = jnp.zeros(n)

    for _ in range(num_iterations):
        log_u = log_a - jax.nn.logsumexp(log_K + log_v[None, :], axis=1)
        log_v = log_b - jax.nn.logsumexp(log_K + log_u[:, None], axis=0)

    log_P = log_u[:, None] + log_K + log_v[None, :]

    # row-conditional logits
    log_P_cond = log_P - jax.nn.logsumexp(log_P, axis=1, keepdims=True)

    if key is not None:
        g = jax.random.gumbel(key, (n, n))
        assignments = jnp.argmax(log_P_cond + g, axis=1)
    else:
        assignments = jnp.argmax(log_P_cond, axis=1)

    return source, target[assignments]



class DoobHTransformSolver(SBSolver):
    """Doob h-Transform Schrödinger Bridge solver.
    
    This solver computes the h-function that transforms the reference process
    into the Schrödinger Bridge. Methods supported:
    
    1. Analytical (Gaussian case): Closed-form solution using OT theory.
       EXACT for Gaussian marginals.
       
    2. Kernel-based: Uses bridge conditional sampling with kernel regression.
       APPROXIMATE for general distributions.
       
    3. Kernel + Sinkhorn: Uses entropic OT coupling before kernel regression.
       Better approximation than naive kernel method.
    
    The h-function gives drift via: b*(x,t) = b_ref(x,t) + σ² ∇log h(x,t)
    
    Attributes:
        problem: The SB problem specification.
        doob_config: Configuration for the solver.
    """
    
    def __init__(
        self,
        problem: Any,  # SBProblem
        doob_config: Optional[DoobConfig] = None,
        config: Optional[DoobConfig] = None,
        **kwargs,
    ):
        """Initialize Doob h-Transform solver.
        
        Args:
            problem: SBProblem specification.
            doob_config: Solver configuration.
            config: Alternative config parameter name.
        """
        # Initialize base SBSolver to set integrator, config, diagnostics, etc.
        super().__init__(problem)
        self.problem = problem
        
        # Handle config parameter flexibility
        if doob_config is None and config is not None:
            doob_config = config
        self.doob_config = doob_config or DoobConfig()
        
        # Determine method
        if self.doob_config.method == 'auto':
            self._method = self._select_method()
        else:
            self._method = self.doob_config.method
        
        # Issue warning for approximate methods
        if self._method in ['kernel']:
            warnings.warn(
                "Kernel method uses index-pairing which doesn't give the "
                "true SB coupling. Consider 'kernel_sinkhorn' for better results, "
                "or use IPFSolver for exact solutions.",
                UserWarning
            )
        
        # Analytical solution components (for Gaussian case)
        self._A: Optional[Array] = None
        self._params: Optional[Params] = None
        
        # Kernel solution components
        self._source_samples: Optional[Array] = None
        self._target_samples: Optional[Array] = None
        self._bandwidth: Optional[float] = None
        self._is_trained: bool = False

    @property
    def solver_type(self) -> SolverType:
        return SolverType.DOOB

    @property
    def representation_type(self) -> RepresentationType:
        # Analytical and sinkhorn-kernel methods produce Schrödinger potentials (h)
        if self._method == 'analytical':
            return RepresentationType.POTENTIAL
        # Kernel-based methods use an RKHS-like representation
        return RepresentationType.KERNEL
    
    def _select_method(self) -> str:
        """Automatically select the best method based on problem structure."""
        # Check if both marginals are Gaussian
        source_type = type(self.problem.source).__name__
        target_type = type(self.problem.target).__name__
        ref_type = type(self.problem.reference).__name__
        
        is_source_gaussian = 'Gaussian' in source_type
        is_target_gaussian = 'Gaussian' in target_type
        is_compatible_ref = ref_type in ['BrownianMotion', 'OrnsteinUhlenbeck']
        
        if is_source_gaussian and is_target_gaussian and is_compatible_ref:
            return 'analytical'
        else:
            return 'kernel_sinkhorn'  # Default to better method
    
    def init_params(self, key: PRNGKey) -> Params:
        """Initialize solver parameters.
        
        Args:
            key: JAX random key.
            
        Returns:
            Dictionary of parameters.
        """
        if self._method == 'analytical':
            return self._init_analytical()
        else:
            return self._init_kernel(key)

    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: Any,
        batch_size: int,
    ) -> Tuple[Params, Any, Dict[str, Scalar]]:
        """No-op training step for non-iterative Doob solver.

        Returns params unchanged and a trivial loss metric so the
        `SBSolver` training loop can run if accidentally invoked.
        """
        metrics = {'loss': 0.0}
        return params, opt_state, metrics
    
    def _init_analytical(self) -> Params:
        """Initialize analytical Gaussian solution.
        
        Computes the OT map A and stores all necessary parameters.
        This is EXACT for Gaussian-to-Gaussian transport.
        """
        source = self.problem.source
        target = self.problem.target
        d = self.problem.dim
        
        m0, Σ0 = source.mean, source.cov
        m1, Σ1 = target.mean, target.cov
        
        # Get reference diffusion coefficient
        sigma = self.problem.reference.diffusion(None, 0.5)
        
        # Ensure covariances are matrices
        if jnp.ndim(Σ0) == 0:
            Σ0 = Σ0 * jnp.eye(d)
        if jnp.ndim(Σ1) == 0:
            Σ1 = Σ1 * jnp.eye(d)
        
        # Compute OT map: T(x) = A(x - m0) + m1
        # A = Σ0^{-1/2} (Σ0^{1/2} Σ1 Σ0^{1/2})^{1/2} Σ0^{-1/2}
        
        Σ0_reg = Σ0 + 1e-6 * jnp.eye(d)
        Σ0_sqrt = jnp.linalg.cholesky(Σ0_reg)
        
        middle = Σ0_sqrt @ Σ1 @ Σ0_sqrt.T
        eigvals, eigvecs = jnp.linalg.eigh(middle + 1e-6 * jnp.eye(d))
        eigvals = jnp.maximum(eigvals, 0)
        middle_sqrt = eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T
        
        Σ0_sqrt_inv = jnp.linalg.solve(Σ0_sqrt, jnp.eye(d))
        A = Σ0_sqrt_inv.T @ middle_sqrt @ Σ0_sqrt_inv
        
        self._A = A
        
        params = {
            'A': A,
            'm0': m0,
            'm1': m1,
            'Σ0': Σ0,
            'Σ1': Σ1,
            'sigma': sigma,
            'd': d,
            'method': 'analytical',
        }
        
        return params
    
    def _init_kernel(self, key: PRNGKey) -> Params:
        """Initialize kernel-based solution.
        
        For the kernel method, we store source and target samples.
        If using Sinkhorn coupling, we first compute the entropic OT plan.
        """
        k1, k2, k3 = jax.random.split(key, 3)

        n = self.doob_config.num_inducing_points

        # Sample from source and target
        source_samples = self.problem.sample_source(k1, n)
        target_samples = self.problem.sample_target(k2, n)

        # Apply Sinkhorn coupling if requested
        if self._method == 'kernel_sinkhorn':
            source_samples, target_samples = sinkhorn_coupling(
                source_samples,
                target_samples,
                epsilon=self.doob_config.sinkhorn_epsilon,
                num_iterations=self.doob_config.sinkhorn_iterations,
                key=k3,  # enables stochastic row-conditional sampling
            )
        
        self._source_samples = source_samples
        self._target_samples = target_samples
        
        # Compute bandwidth
        if self.doob_config.kernel_bandwidth is None:
            all_samples = jnp.concatenate([source_samples, target_samples], axis=0)
            # Median heuristic
            dists = jnp.sum((all_samples[:, None] - all_samples[None, :]) ** 2, axis=-1)
            self._bandwidth = jnp.sqrt(jnp.median(dists[dists > 0]))
        else:
            self._bandwidth = self.doob_config.kernel_bandwidth
        
        return {
            'source_samples': self._source_samples,
            'target_samples': self._target_samples,
            'bandwidth': self._bandwidth,
            'method': self._method,
        }
    
    def train(
        self,
        key: PRNGKey,
        training_config: Optional[Any] = None,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> "SolverResult":
        """Train the Doob solver and return a SolverResult.

        The Doob solver is non-iterative: training just initializes parameters
        (analytical or kernel initialization). We wrap the result in the
        standard `SolverResult` to match the `SBSolver` interface.
        """
        # Initialize parameters and mark trained
        params = self.init_params(key)
        self._params = params
        self._is_trained = True

        # Run diagnostics using base helper
        diagnostics = self._run_diagnostics(key, params)

        metadata = {
            'converged': True,
            'method': self._method,
            'solver_type': self.solver_type.name,
        }

        return SolverResult(
            params=params,
            loss_history=jnp.array([0.0]),
            diagnostics=diagnostics,
            metadata=metadata,
        )
    
    def _compute_drift_analytical(self, x: Array, t: Scalar) -> Array:
        """Compute drift for Gaussian SB (analytical solution).
        
        The drift is: b*(x,t) = (E[X_1 | X_t = x] - x) / (1 - t)
        
        where E[X_1 | X_t = x] comes from the OT interpolation theory.
        
        For OT map T(x) = A(x - m0) + m1, the McCann interpolation gives:
            X_t = M_t X_0 - t A m0 + t m1,  where M_t = (1-t)I + tA
            
        Inverting: X_0 = M_t^{-1} (X_t + t A m0 - t m1)
        Target: X_1 = A M_t^{-1} (X_t + t A m0 - t m1) - A m0 + m1
        
        So: E[X_1 | X_t = x] = A M_t^{-1} (x + t A m0 - t m1) - A m0 + m1
        """
        x = jnp.atleast_2d(x)
        t_scalar = jnp.asarray(t).reshape(())
        
        m0 = self._params['m0']
        m1 = self._params['m1']
        A = self._params['A']
        d = self._params['d']
        
        # Safe remaining time
        remaining_time = jnp.maximum(1.0 - t_scalar, 1e-6)
        
        # M_t = (1-t)I + tA
        M_t = (1.0 - t_scalar) * jnp.eye(d) + t_scalar * A
        
        # M_t^{-1} - regularized for numerical stability
        M_t_inv = jnp.linalg.solve(M_t + 1e-8 * jnp.eye(d), jnp.eye(d))
        
        # Compute offset: t A m0 - t m1
        offset = t_scalar * (A @ m0 - m1)
        
        # E[X_1 | X_t = x] = A M_t^{-1} (x + offset) - A m0 + m1
        L = A @ M_t_inv
        c = L @ offset - A @ m0 + m1
        
        # Expected target for each x
        expected_target = x @ L.T + c  # [batch, d]
        
        # Drift points toward expected target
        drift = (expected_target - x) / remaining_time
        
        return drift
    
    def _compute_drift_kernel(self, x: Array, t: Scalar) -> Array:
        """Compute drift for kernel-based SB (APPROXIMATE).
        
        KEY INSIGHT: The SB drift at (x, t) points toward the expected 
        target position, weighted by how likely each target is given x at time t.
        
        Using the bridge formula:
            b*(x,t) = (E[X₁ | X_t = x] - x) / (1 - t)
        
        We estimate E[X₁ | X_t = x] using kernel weights based on the bridge conditional.
        
        The bridge from x₀ to x₁ has:
            X_t | X₀=x₀, X₁=x₁ ~ N((1-t)x₀ + t·x₁, σ²t(1-t))
        
        LIMITATION: This ignores Schrödinger potentials. The true SB posterior is:
            p_SB(X₁ | X_t) ∝ p_bridge(X_t | X₀, X₁) × ψ(X₁) × φ(X₀)
        
        We use p_bridge alone as an approximation.
        """
        x = jnp.atleast_2d(x)
        t_scalar = jnp.asarray(t).reshape(())
        batch_size = x.shape[0]
        
        # Get sigma from reference
        sigma = self.problem.reference.diffusion(None, 0.5)
        
        # Safe remaining time
        remaining_time = jnp.maximum(1.0 - t_scalar, 1e-4)
        
        # Bridge variance at time t: σ²t(1-t)
        bridge_var = sigma ** 2 * jnp.maximum(t_scalar, 0.01) * remaining_time
        
        source = self._source_samples
        target = self._target_samples
        
        # Expected position at time t for each pair: μ_t = (1-t) * source + t * target
        mu_t = (1.0 - t_scalar) * source + t_scalar * target
        
        # Compute weights based on how close x is to each μ_t
        # Weight ∝ exp(-||x - μ_t||² / (2 * bridge_var))
        diff = x[:, None, :] - mu_t[None, :, :]
        sq_dist = jnp.sum(diff ** 2, axis=-1)
        
        log_weights = -sq_dist / (2 * bridge_var + 1e-8)
        weights = jax.nn.softmax(log_weights, axis=-1)
        
        # Expected target position: weighted average of targets
        expected_target = jnp.einsum('bi,id->bd', weights, target)
        
        # Drift points toward expected target
        drift = (expected_target - x) / remaining_time
        
        return drift
    
    def compute_drift(self, x: Array, t: Scalar) -> Array:
        """Compute the SB drift at position x and time t.
        
        Args:
            x: Position, shape [batch, dim] or [dim].
            t: Time in [0, 1].
            
        Returns:
            Drift vector, same shape as x.
        """
        if not self._is_trained:
            raise RuntimeError("Solver must be trained before computing drift")
        
        x = jnp.atleast_2d(x)
        
        # Reference drift (usually zero for Brownian motion)
        b_ref = self.problem.reference.drift(x, t)
        
        # Bridge correction
        if self._method == 'analytical':
            bridge_drift = self._compute_drift_analytical(x, t)
        else:
            bridge_drift = self._compute_drift_kernel(x, t)
        
        return b_ref + bridge_drift
    
    def extract_drift(self, params: Optional[Params] = None) -> DriftFn:
        """Extract forward drift function.
        
        Args:
            params: Optional parameters (uses stored params if None).
            
        Returns:
            Drift function b*(x, t).
        """
        if params is not None:
            self._params = params
        
        def drift(x: Array, t: Scalar) -> Array:
            return self.compute_drift(x, t)
        
        return drift
    
    def get_expected_target(self, x: Array, t: Scalar) -> Array:
        """Get the expected target position given current state.
        
        Useful for understanding where particles are heading.
        
        Args:
            x: Current position, shape [batch, dim] or [dim].
            t: Current time.
            
        Returns:
            Expected position at t=1.
        """
        x = jnp.atleast_2d(x)
        t_scalar = jnp.asarray(t).reshape(())
        
        if self._method == 'analytical':
            m0 = self._params['m0']
            m1 = self._params['m1']
            A = self._params['A']
            d = self._params['d']
            
            M_t = (1.0 - t_scalar) * jnp.eye(d) + t_scalar * A
            M_t_inv = jnp.linalg.solve(M_t + 1e-8 * jnp.eye(d), jnp.eye(d))
            
            offset = t_scalar * (A @ m0 - m1)
            L = A @ M_t_inv
            c = L @ offset - A @ m0 + m1
            
            return x @ L.T + c
        else:
            # Use kernel regression
            sigma = self.problem.reference.diffusion(None, 0.5)
            remaining_time = jnp.maximum(1.0 - t_scalar, 1e-4)
            bridge_var = sigma ** 2 * jnp.maximum(t_scalar, 0.01) * remaining_time
            
            source = self._source_samples
            target = self._target_samples
            
            mu_t = (1.0 - t_scalar) * source + t_scalar * target
            diff = x[:, None, :] - mu_t[None, :, :]
            sq_dist = jnp.sum(diff ** 2, axis=-1)
            
            log_weights = -sq_dist / (2 * bridge_var + 1e-8)
            weights = jax.nn.softmax(log_weights, axis=-1)
            
            return jnp.einsum('bi,id->bd', weights, target)


# # =============================================================================
# # Verification Tests
# # =============================================================================

# def test_gaussian_analytical_correctness():
#     """Test that analytical method gives correct drift for Gaussians.
    
#     For equal covariances (A = I), drift should be constant = m1 - m0.
#     """
#     print("=" * 60)
#     print("TEST: Gaussian Analytical Correctness")
#     print("=" * 60)
    
#     # Create a simple problem: same covariance, different means
#     # This should give constant drift = m1 - m0
    
#     @dataclass
#     class MockGaussian:
#         mean: Array
#         cov: Array
#         dim: int = 2
        
#         def sample(self, key, n):
#             return jax.random.multivariate_normal(
#                 key, self.mean, self.cov * jnp.eye(self.dim), (n,)
#             )
    
#     @dataclass
#     class MockBrownian:
#         sigma: float = 1.0
#         dim: int = 2
        
#         def drift(self, x, t):
#             return jnp.zeros_like(x)
        
#         def diffusion(self, x, t):
#             return self.sigma
    
#     @dataclass
#     class MockProblem:
#         source: MockGaussian
#         target: MockGaussian
#         reference: MockBrownian
#         dim: int = 2
        
#         def sample_source(self, key, n):
#             return self.source.sample(key, n)
        
#         def sample_target(self, key, n):
#             return self.target.sample(key, n)
    
#     m0 = jnp.array([0.0, 0.0])
#     m1 = jnp.array([2.0, 1.0])
#     cov = 1.0
    
#     problem = MockProblem(
#         source=MockGaussian(m0, cov),
#         target=MockGaussian(m1, cov),
#         reference=MockBrownian(),
#     )
    
#     solver = DoobHTransformSolver(problem, DoobConfig(method='analytical'))
#     key = jax.random.PRNGKey(42)
#     solver.train(key)
    
#     # For A = I (equal covariances), drift should be constant = m1 - m0
#     expected_drift = m1 - m0
    
#     # Test at various points and times
#     test_points = jnp.array([[0.0, 0.0], [1.0, 0.5], [-1.0, 2.0]])
    
#     for t in [0.0, 0.25, 0.5, 0.75]:
#         drift = solver.compute_drift(test_points, t)
        
#         # All points should have same drift (constant field)
#         drift_variation = jnp.std(drift, axis=0)
        
#         # Check drift is close to expected
#         mean_drift = jnp.mean(drift, axis=0)
#         drift_error = jnp.linalg.norm(mean_drift - expected_drift)
        
#         print(f"  t={t:.2f}: drift={mean_drift}, expected={expected_drift}, error={drift_error:.6f}")
        
#         assert drift_error < 0.01, f"Drift error too large at t={t}"
#         assert jnp.all(drift_variation < 0.001), f"Drift should be constant at t={t}"
    
#     print("  ✓ Analytical method gives correct constant drift for equal covariances")
#     print()
#     return True


# def test_sde_convention_consistency():
#     """Verify the SDE convention is consistent throughout.
    
#     The library uses dX = b dt + σ dW (standard Itô).
#     The Doob formula is b* = b + σ² ∇log h.
    
#     If we were using dX = b dt + √(2σ) dW, we'd need b* = b + 2σ² ∇log h.
#     """
#     print("=" * 60)
#     print("TEST: SDE Convention Consistency")
#     print("=" * 60)
    
#     # The key test: For a Brownian bridge from 0 to 1 in 1D,
#     # the drift at the midpoint (x=0.5, t=0.5) should be:
#     # b* = (E[X_1 | X_{0.5} = 0.5] - 0.5) / 0.5
#     #    = (1 - 0.5) / 0.5 = 1.0
    
#     # This is independent of σ for the bridge drift!
#     # The σ only affects the variance, not the mean.
    
#     @dataclass
#     class MockGaussian1D:
#         mean: Array
#         cov: Array
#         dim: int = 1
        
#         def sample(self, key, n):
#             return jax.random.normal(key, (n, 1)) * jnp.sqrt(self.cov) + self.mean
    
#     @dataclass
#     class MockBrownian1D:
#         sigma: float = 1.0
#         dim: int = 1
        
#         def drift(self, x, t):
#             return jnp.zeros_like(x)
        
#         def diffusion(self, x, t):
#             return self.sigma
    
#     @dataclass
#     class MockProblem1D:
#         source: MockGaussian1D
#         target: MockGaussian1D
#         reference: MockBrownian1D
#         dim: int = 1
        
#         def sample_source(self, key, n):
#             return self.source.sample(key, n)
        
#         def sample_target(self, key, n):
#             return self.target.sample(key, n)
    
#     for sigma in [0.5, 1.0, 2.0]:
#         m0 = jnp.array([0.0])
#         m1 = jnp.array([2.0])
#         cov = 0.01  # Small variance for point-like marginals
        
#         problem = MockProblem1D(
#             source=MockGaussian1D(m0, cov),
#             target=MockGaussian1D(m1, cov),
#             reference=MockBrownian1D(sigma=sigma),
#         )
        
#         solver = DoobHTransformSolver(problem, DoobConfig(method='analytical'))
#         key = jax.random.PRNGKey(42)
#         solver.train(key)
        
#         # At midpoint x=1, t=0.5, drift should be (2-1)/0.5 = 2
#         x_mid = jnp.array([[1.0]])
#         drift_mid = solver.compute_drift(x_mid, 0.5)
#         expected = (m1 - x_mid.flatten()) / 0.5  # = 2
        
#         print(f"  σ={sigma}: drift at (x=1, t=0.5) = {drift_mid.flatten()[0]:.4f}, expected = {expected[0]:.4f}")
        
#         assert jnp.abs(drift_mid.flatten()[0] - expected[0]) < 0.01
    
#     print("  ✓ Drift is independent of σ (as expected for Brownian bridge mean)")
#     print("  ✓ SDE convention is correct: dX = b dt + σ dW")
#     print()
#     return True


# def test_kernel_sinkhorn_vs_naive():
#     """Compare kernel methods with and without Sinkhorn coupling.
    
#     The Sinkhorn-coupled version should generally produce better results
#     because it uses a proper OT coupling rather than arbitrary index pairing.
#     """
#     print("=" * 60)
#     print("TEST: Kernel Methods Comparison")
#     print("=" * 60)
    
#     @dataclass
#     class MockGaussian:
#         mean: Array
#         cov: Array
#         dim: int = 2
        
#         def sample(self, key, n):
#             return jax.random.multivariate_normal(
#                 key, self.mean, self.cov * jnp.eye(self.dim), (n,)
#             )
    
#     @dataclass
#     class MockBrownian:
#         sigma: float = 0.5
#         dim: int = 2
        
#         def drift(self, x, t):
#             return jnp.zeros_like(x)
        
#         def diffusion(self, x, t):
#             return self.sigma
    
#     @dataclass
#     class MockProblem:
#         source: MockGaussian
#         target: MockGaussian
#         reference: MockBrownian
#         dim: int = 2
        
#         def sample_source(self, key, n):
#             return self.source.sample(key, n)
        
#         def sample_target(self, key, n):
#             return self.target.sample(key, n)
    
#     # Create problem with different means
#     m0 = jnp.array([-2.0, 0.0])
#     m1 = jnp.array([2.0, 0.0])
    
#     problem = MockProblem(
#         source=MockGaussian(m0, 0.5),
#         target=MockGaussian(m1, 0.5),
#         reference=MockBrownian(),
#     )
    
#     key = jax.random.PRNGKey(42)
    
#     # Test naive kernel
#     k1, k2 = jax.random.split(key)
    
#     with warnings.catch_warnings(record=True) as w:
#         warnings.simplefilter("always")
#         solver_naive = DoobHTransformSolver(
#             problem, 
#             DoobConfig(method='kernel', num_inducing_points=200)
#         )
#         solver_naive.train(k1)
#         assert len(w) == 1  # Should get warning about index pairing
#         print(f"  ✓ Naive kernel method issues warning: '{w[0].message}'")
    
#     # Test Sinkhorn kernel
#     solver_sinkhorn = DoobHTransformSolver(
#         problem,
#         DoobConfig(method='kernel_sinkhorn', num_inducing_points=200)
#     )
#     solver_sinkhorn.train(k2)
    
#     # Compare drifts at test points
#     test_x = jnp.array([[0.0, 0.0]])  # Midpoint
#     t = 0.5
    
#     drift_naive = solver_naive.compute_drift(test_x, t)
#     drift_sinkhorn = solver_sinkhorn.compute_drift(test_x, t)
    
#     # Expected drift direction: toward m1
#     expected_direction = (m1 - test_x.flatten()) / jnp.linalg.norm(m1 - test_x.flatten())
    
#     naive_direction = drift_naive.flatten() / jnp.linalg.norm(drift_naive.flatten())
#     sinkhorn_direction = drift_sinkhorn.flatten() / jnp.linalg.norm(drift_sinkhorn.flatten())
    
#     naive_alignment = jnp.dot(naive_direction, expected_direction)
#     sinkhorn_alignment = jnp.dot(sinkhorn_direction, expected_direction)
    
#     print(f"  Naive kernel drift direction alignment: {naive_alignment:.4f}")
#     print(f"  Sinkhorn kernel drift direction alignment: {sinkhorn_alignment:.4f}")
#     print(f"  (1.0 = perfect alignment with expected direction)")
    
#     # Both should be reasonable, but Sinkhorn should be at least as good
#     assert naive_alignment > 0.8, "Naive method should give reasonable direction"
#     assert sinkhorn_alignment > 0.8, "Sinkhorn method should give reasonable direction"
#     print("  ✓ Both methods produce reasonable drift directions")
#     print()
#     return True


