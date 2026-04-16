"""Schrödinger Bridge Problem Definition.

This module defines the mathematical structure of a Schrödinger Bridge problem:

    P* = argmin_{P} KL(P || P_ref)
    
subject to:
    P_0 = mu_0  (source marginal constraint)
    P_1 = mu_1  (target marginal constraint)

The problem consists of:
1. Reference dynamics (the prior stochastic process)
2. Marginal constraints (source and target distributions)
3. Cost structure (implicitly KL divergence from reference)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from .types import (
    Array,
    DensityEvaluator,
    DiffusionFn,
    DriftFn,
    PRNGKey,
    Sampler,
    Scalar,
    SDECoefficients,
    TimeGrid,
)


# Reference Dynamics
# -----------------------------------------------------------------------------
class ReferenceDynamics(abc.ABC):
    """Abstract base class for reference SDE dynamics.
    
    The reference process defines the prior over paths:
        dX_t = b(X_t, t) dt + sigma(X_t, t) dW_t
    
    Different reference processes lead to different computational strategies.
    """
    
    @abc.abstractmethod
    def drift(self, x: Array, t: Scalar) -> Array:
        """Compute drift coefficient b(x, t).
        
        Args:
            x: State, shape [batch, dim] or [dim].
            t: Time.
            
        Returns:
            Drift, same shape as x.
        """
        pass
    
    @abc.abstractmethod
    def diffusion(self, x: Array, t: Scalar) -> Union[Scalar, Array]:
        """Compute diffusion coefficient sigma(x, t).
        
        Args:
            x: State (may be ignored for scalar diffusion).
            t: Time.
            
        Returns:
            Diffusion coefficient (scalar or matrix).
        """
        pass
    
    @property
    @abc.abstractmethod
    def is_time_homogeneous(self) -> bool:
        """Whether coefficients are independent of time."""
        pass
    
    @property
    def is_diffusion_scalar(self) -> bool:
        """Whether diffusion is state-independent and scalar."""
        return True
    
    def to_sde_coefficients(self) -> SDECoefficients:
        """Convert to SDECoefficients container."""
        return SDECoefficients(
            drift=self.drift,
            diffusion=lambda t: self.diffusion(None, t),
            is_diffusion_scalar=self.is_diffusion_scalar,
        )
    
    def reverse_drift(
        self, 
        x: Array, 
        t: Scalar, 
        score: Callable[[Array, Scalar], Array],
    ) -> Array:
        """Compute drift of time-reversed process.
        
        For the reverse SDE: dX = [-b + sigma^2 grad log p_t] dt + sigma dW̃
        
        Args:
            x: State.
            t: Time.
            score: Score function grad log p_t(x).
            
        Returns:
            Reverse drift.
        """
        sigma = self.diffusion(x, t)
        sigma_sq = sigma ** 2 if self.is_diffusion_scalar else jnp.dot(sigma, sigma.T)
        return -self.drift(x, t) + sigma_sq * score(x, t)


class BrownianMotion(ReferenceDynamics):
    """Standard Brownian motion: dX_t = sigma dW_t.
    
    Attributes:
        sigma: Diffusion coefficient (scalar).
        dim: Dimension of the process.
    """
    
    def __init__(self, sigma: float = 1.0, dim: int = 2):
        self.sigma = sigma
        self._dim = dim
    
    def drift(self, x: Array, t: Scalar) -> Array:
        return jnp.zeros_like(x)
    
    def diffusion(self, x: Array, t: Scalar) -> Scalar:
        return self.sigma
    
    @property
    def is_time_homogeneous(self) -> bool:
        return True
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def transition_mean(self, x0: Array, t: Scalar) -> Array:
        """Mean of X_t | X_0 = x0."""
        return x0
    
    def transition_std(self, t: Scalar) -> float:
        """Standard deviation of X_t | X_0."""
        return self.sigma * jnp.sqrt(t)


class OrnsteinUhlenbeck(ReferenceDynamics):
    """Ornstein-Uhlenbeck process: dX_t = -theta(X_t - mu) dt + sigma dW_t.
    
    Attributes:
        theta: Mean reversion rate.
        mu: Long-term mean.
        sigma: Diffusion coefficient.
        dim: Dimension of the process.
    """
    
    def __init__(
        self,
        theta: float = 1.0,
        mu: Optional[Array] = None,
        sigma: float = 1.0,
        dim: int = 2,
    ):
        self.theta = theta
        self.mu = mu if mu is not None else jnp.zeros(dim)
        self.sigma = sigma
        self._dim = dim
    
    def drift(self, x: Array, t: Scalar) -> Array:
        return -self.theta * (x - self.mu)
    
    def diffusion(self, x: Array, t: Scalar) -> Scalar:
        return self.sigma
    
    @property
    def is_time_homogeneous(self) -> bool:
        return True
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def transition_mean(self, x0: Array, t: Scalar) -> Array:
        """Mean of X_t | X_0 = x0."""
        decay = jnp.exp(-self.theta * t)
        return self.mu + decay * (x0 - self.mu)
    
    def transition_std(self, t: Scalar) -> float:
        """Standard deviation of X_t | X_0."""
        return self.sigma * jnp.sqrt((1 - jnp.exp(-2 * self.theta * t)) / (2 * self.theta))
    
    def stationary_std(self) -> float:
        """Standard deviation of stationary distribution."""
        return self.sigma / jnp.sqrt(2 * self.theta)


class VarianceExploding(ReferenceDynamics):
    """Variance Exploding SDE: dX_t = sigma(t) dW_t.
    
    Common in diffusion models. sigma(t) increases over time.
    
    Attributes:
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        dim: Dimension.
    """
    
    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        dim: int = 2,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._dim = dim
    
    def drift(self, x: Array, t: Scalar) -> Array:
        return jnp.zeros_like(x)
    
    def diffusion(self, x: Array, t: Scalar) -> Scalar:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    @property
    def is_time_homogeneous(self) -> bool:
        return False
    
    @property
    def dim(self) -> int:
        return self._dim


class VariancePreserving(ReferenceDynamics):
    """Variance Preserving SDE: dX_t = -1/2 beta(t) X_t dt + sqrtbeta(t) dW_t.
    
    Also known as VP-SDE. Common in DDPM-style diffusion.
    
    Attributes:
        beta_min: Minimum noise schedule value.
        beta_max: Maximum noise schedule value.
        dim: Dimension.
    """
    
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        dim: int = 2,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self._dim = dim
    
    def beta(self, t: Scalar) -> Scalar:
        """Noise schedule beta(t)."""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def drift(self, x: Array, t: Scalar) -> Array:
        return -0.5 * self.beta(t) * x
    
    def diffusion(self, x: Array, t: Scalar) -> Scalar:
        return jnp.sqrt(self.beta(t))
    
    @property
    def is_time_homogeneous(self) -> bool:
        return False
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def alpha_bar(self, t: Scalar) -> Scalar:
        """Cumulative signal retention: ᾱ(t) = exp(-integral_0ᵗ beta(s) ds)."""
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return jnp.exp(-0.5 * integral)


# Marginal Distributions
# -----------------------------------------------------------------------------
class MarginalDistribution(abc.ABC):
    """Abstract base class for marginal distributions.
    
    A marginal distribution must support:
    1. Sampling
    2. (Optionally) Density evaluation
    """
    
    @abc.abstractmethod
    def sample(self, key: PRNGKey, num_samples: int) -> Array:
        """Draw samples from the distribution.
        
        Args:
            key: JAX random key.
            num_samples: Number of samples.
            
        Returns:
            Samples of shape [num_samples, dim].
        """
        pass
    
    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimension of the distribution."""
        pass
    
    def log_prob(self, x: Array) -> Array:
        """Compute log probability (if available).
        
        Args:
            x: Points, shape [batch, dim].
            
        Returns:
            Log probabilities, shape [batch].
            
        Raises:
            NotImplementedError: If density is not available.
        """
        raise NotImplementedError("Log probability not available for this distribution.")
    
    @property
    def has_density(self) -> bool:
        """Whether log_prob is implemented."""
        try:
            # Try to compute log_prob for a dummy point
            self.log_prob(jnp.zeros((1, self.dim)))
            return True
        except NotImplementedError:
            return False


class GaussianDistribution(MarginalDistribution):
    """Multivariate Gaussian distribution.
    
    Attributes:
        mean: Mean vector, shape [dim].
        cov: Covariance matrix, shape [dim, dim], or variance (scalar).
    """
    
    def __init__(
        self,
        mean: Optional[Array] = None,
        cov: Optional[Union[Array, float]] = None,
        dim: int = 2,
    ):
        self._dim = dim
        self.mean = mean if mean is not None else jnp.zeros(dim)
        
        if cov is None:
            self.cov = jnp.eye(dim)
        elif isinstance(cov, (int, float)):
            self.cov = float(cov) * jnp.eye(dim)
        else:
            self.cov = jnp.asarray(cov)
        
        # Precompute for sampling and density
        self._L = jnp.linalg.cholesky(self.cov)
        self._log_det = jnp.linalg.slogdet(self.cov)[1]
        self._cov_inv = jnp.linalg.inv(self.cov)
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def sample(self, key: PRNGKey, num_samples: int) -> Array:
        z = jax.random.normal(key, (num_samples, self._dim))
        return self.mean + z @ self._L.T
    
    def log_prob(self, x: Array) -> Array:
        x = jnp.atleast_2d(x)
        diff = x - self.mean
        mahal = jnp.sum(diff @ self._cov_inv * diff, axis=-1)
        return -0.5 * (self._dim * jnp.log(2 * jnp.pi) + self._log_det + mahal)


class EmpiricalDistribution(MarginalDistribution):
    """Distribution defined by samples (empirical measure).
    
    Useful when marginals are given as data points.
    
    Attributes:
        samples: Data points, shape [n, dim].
        bandwidth: Bandwidth for optional KDE density.
    """
    
    def __init__(self, samples: Array, bandwidth: Optional[float] = None):
        self.samples = jnp.asarray(samples)
        self._dim = self.samples.shape[1]
        self.bandwidth = bandwidth
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def num_samples(self) -> int:
        return self.samples.shape[0]
    
    def sample(self, key: PRNGKey, num_samples: int) -> Array:
        """Sample with replacement from empirical distribution."""
        indices = jax.random.randint(key, (num_samples,), 0, self.num_samples)
        return self.samples[indices]
    
    def log_prob(self, x: Array) -> Array:
        """Kernel density estimate (if bandwidth specified)."""
        if self.bandwidth is None:
            raise NotImplementedError(
                "Set bandwidth to enable KDE density estimation."
            )
        
        x = jnp.atleast_2d(x)
        # Gaussian KDE
        diffs = x[:, None, :] - self.samples[None, :, :]  # [batch, n, dim]
        sq_dists = jnp.sum(diffs ** 2, axis=-1)  # [batch, n]
        log_kernels = -0.5 * sq_dists / self.bandwidth ** 2
        log_normalization = -0.5 * self._dim * jnp.log(2 * jnp.pi * self.bandwidth ** 2)
        
        # Log-sum-exp for stability
        log_density = jax.scipy.special.logsumexp(
            log_kernels + log_normalization, axis=-1
        ) - jnp.log(self.num_samples)
        
        return log_density


class MixtureDistribution(MarginalDistribution):
    """Mixture of distributions.
    
    Attributes:
        components: List of component distributions.
        weights: Mixture weights (must sum to 1).
    """
    
    def __init__(
        self,
        components: list[MarginalDistribution],
        weights: Optional[Array] = None,
    ):
        if len(components) == 0:
            raise ValueError("Must provide at least one component.")
        
        self.components = components
        self._dim = components[0].dim
        
        # Validate dimensions match
        for i, c in enumerate(components):
            if c.dim != self._dim:
                raise ValueError(
                    f"Component {i} has dim {c.dim}, expected {self._dim}."
                )
        
        if weights is None:
            self.weights = jnp.ones(len(components)) / len(components)
        else:
            self.weights = jnp.asarray(weights)
            if not jnp.isclose(self.weights.sum(), 1.0):
                raise ValueError("Weights must sum to 1.")
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def sample(self, key: PRNGKey, num_samples: int) -> Array:
        k1, k2 = jax.random.split(key)
        
        # Sample component assignments
        assignments = jax.random.choice(
            k1, len(self.components), shape=(num_samples,), p=self.weights
        )
        
        # Sample from each component
        keys = jax.random.split(k2, len(self.components))
        all_samples = jnp.stack([
            c.sample(k, num_samples) for c, k in zip(self.components, keys)
        ])  # [num_components, num_samples, dim]
        
        # Select based on assignments
        return all_samples[assignments, jnp.arange(num_samples)]
    
    def log_prob(self, x: Array) -> Array:
        """Log probability via log-sum-exp over components."""
        log_probs = jnp.stack([c.log_prob(x) for c in self.components])  # [K, batch]
        log_weights = jnp.log(self.weights)[:, None]  # [K, 1]
        return jax.scipy.special.logsumexp(log_probs + log_weights, axis=0)


class TwoMoonsDistribution(MarginalDistribution):
    """Two moons (crescent) distribution.
    
    A classic benchmark for density estimation and transport.
    
    Attributes:
        noise: Noise level.
        offset: Offset between moons.
    """
    
    def __init__(self, noise: float = 0.05, offset: float = 0.5):
        self.noise = noise
        self.offset = offset
        self._dim = 2
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def sample(self, key: PRNGKey, num_samples: int) -> Array:
        k1, k2, k3 = jax.random.split(key, 3)
        
        n_upper = num_samples // 2
        n_lower = num_samples - n_upper
        
        # Upper moon
        theta_upper = jax.random.uniform(k1, (n_upper,)) * jnp.pi
        upper = jnp.stack([jnp.cos(theta_upper), jnp.sin(theta_upper)], axis=-1)
        
        # Lower moon (flipped and shifted)
        theta_lower = jax.random.uniform(k2, (n_lower,)) * jnp.pi
        lower = jnp.stack([
            1.0 - jnp.cos(theta_lower),
            self.offset - jnp.sin(theta_lower)
        ], axis=-1)
        
        samples = jnp.concatenate([upper, lower], axis=0)
        
        # Add noise
        noise = jax.random.normal(k3, samples.shape) * self.noise
        return samples + noise


class SwissRollDistribution(MarginalDistribution):
    """Swiss roll distribution (3D or 2D projection)."""
    
    def __init__(self, noise: float = 0.1, project_2d: bool = True):
        self.noise = noise
        self.project_2d = project_2d
        self._dim = 2 if project_2d else 3
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def sample(self, key: PRNGKey, num_samples: int) -> Array:
        k1, k2, k3 = jax.random.split(key, 3)
        
        t = 1.5 * jnp.pi * (1 + 2 * jax.random.uniform(k1, (num_samples,)))
        x = t * jnp.cos(t)
        y = jax.random.uniform(k2, (num_samples,)) * 10
        z = t * jnp.sin(t)
        
        if self.project_2d:
            samples = jnp.stack([x, z], axis=-1) / 10.0
        else:
            samples = jnp.stack([x, y, z], axis=-1) / 10.0
        
        noise = jax.random.normal(k3, samples.shape) * self.noise
        return samples + noise


# Schrödinger Bridge Problem
# -----------------------------------------------------------------------------
@dataclass
class SBProblem:
    """Complete specification of a Schrödinger Bridge problem.
    
    The SB problem is:
        P* = argmin_{P} KL(P || P_ref)
        s.t. P_0 = mu_0, P_1 = mu_1
    
    Attributes:
        reference: Reference dynamics (prior SDE).
        source: Source marginal distribution mu_0.
        target: Target marginal distribution mu_1.
        time_grid: Time discretization.
        name: Optional name for the problem.
    """
    reference: ReferenceDynamics
    source: MarginalDistribution
    target: MarginalDistribution
    time_grid: TimeGrid = field(default_factory=TimeGrid)
    name: str = "SBProblem"
    
    def __post_init__(self):
        # Validate dimensions
        if self.source.dim != self.target.dim:
            raise ValueError(
                f"Source dim ({self.source.dim}) != target dim ({self.target.dim})"
            )
        if hasattr(self.reference, 'dim') and self.reference.dim != self.source.dim:
            raise ValueError(
                f"Reference dim ({self.reference.dim}) != marginal dim ({self.source.dim})"
            )
    
    @property
    def dim(self) -> int:
        """Dimension of the state space."""
        return self.source.dim
    
    @property
    def sigma(self) -> DiffusionFn:
        """Diffusion coefficient function."""
        return lambda x, t: self.reference.diffusion(x, t)
    
    def sample_source(self, key: PRNGKey, num_samples: int) -> Array:
        """Sample from source distribution."""
        return self.source.sample(key, num_samples)
    
    def sample_target(self, key: PRNGKey, num_samples: int) -> Array:
        """Sample from target distribution."""
        return self.target.sample(key, num_samples)
    
    def sample_pair(
        self, key: PRNGKey, num_samples: int
    ) -> Tuple[Array, Array]:
        """Sample matched pairs from source and target.
        
        Note: This returns independent samples. For coupled samples,
        use an OT solver to compute a coupling first.
        """
        k1, k2 = jax.random.split(key)
        return self.sample_source(k1, num_samples), self.sample_target(k2, num_samples)
    
    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"=== {self.name} ===",
            f"Dimension: {self.dim}",
            f"Reference: {type(self.reference).__name__}",
            f"Source: {type(self.source).__name__}",
            f"Target: {type(self.target).__name__}",
            f"Time: [{self.time_grid.t0}, {self.time_grid.t1}]",
            f"Steps: {self.time_grid.num_steps}",
        ]
        return "\n".join(lines)


# Factory Functions
# -----------------------------------------------------------------------------
def create_gaussian_to_gaussian(
    source_mean: Array,
    source_cov: Union[Array, float],
    target_mean: Array,
    target_cov: Union[Array, float],
    sigma: float = 1.0,
    time_grid: Optional[TimeGrid] = None,
) -> SBProblem:
    """Create Gaussian-to-Gaussian SB problem.
    
    This is a canonical test case with known analytical solution.
    """
    dim = len(source_mean)
    return SBProblem(
        reference=BrownianMotion(sigma=sigma, dim=dim),
        source=GaussianDistribution(source_mean, source_cov, dim=dim),
        target=GaussianDistribution(target_mean, target_cov, dim=dim),
        time_grid=time_grid or TimeGrid(),
        name="Gaussian-to-Gaussian",
    )


def create_gaussian_to_moons(
    sigma: float = 0.5,
    noise: float = 0.05,
    time_grid: Optional[TimeGrid] = None,
) -> SBProblem:
    """Create Gaussian-to-TwoMoons SB problem.
    
    Common benchmark for generative modeling.
    """
    return SBProblem(
        reference=BrownianMotion(sigma=sigma, dim=2),
        source=GaussianDistribution(dim=2),
        target=TwoMoonsDistribution(noise=noise),
        time_grid=time_grid or TimeGrid(),
        name="Gaussian-to-TwoMoons",
    )


def create_moons_to_moons(
    sigma: float = 0.5,
    noise: float = 0.05,
    time_grid: Optional[TimeGrid] = None,
) -> SBProblem:
    """Create TwoMoons-to-TwoMoons SB problem.
    
    Data-to-data transport benchmark.
    """
    return SBProblem(
        reference=BrownianMotion(sigma=sigma, dim=2),
        source=TwoMoonsDistribution(noise=noise),
        target=TwoMoonsDistribution(noise=noise, offset=-0.5),  # Shifted
        time_grid=time_grid or TimeGrid(),
        name="TwoMoons-to-TwoMoons",
    )
