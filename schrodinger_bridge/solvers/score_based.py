"""Score-Based Schrödinger Bridge Solver.

This solver learns the score function grad log p_t(x) using denoising score matching.
The bridge drift is then: b*(x,t) = b_ref(x,t) + sigma^2(t) grad log p_t(x)

Key insight: For the SB, we can use conditional score matching on bridge paths
connecting source-target sample pairs.

Reference:
    De Bortoli et al. "Diffusion Schrödinger Bridge with Applications to 
    Score-Based Generative Modeling" (NeurIPS 2021)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ..core.types import (
    Array,
    DriftFn,
    Params,
    PRNGKey,
    RepresentationType,
    Scalar,
    SolverConfig,
    SolverType,
    TrainingConfig,
)
from ..core.problem import SBProblem
from ..networks import (
    init_adam,
    adam_update,
    AdamState,
)
from ..network_factory import NetworkFactory, MLPFactory, sanity_check
from .base import SBSolver, ScoreRepresentation


@dataclass
class ScoreBasedConfig:
    """Configuration for score-based solver.
    
    Attributes:
        hidden_dims: Hidden layer dimensions.
        time_embed_dim: Time embedding dimension.
        learning_rate: Learning rate.
        ema_decay: EMA decay for parameters.
        use_bridge_matching: Use bridge-based score matching.
        weight_by_sigma: Weight loss by noise level.
        use_ot_coupling: Use optimal transport coupling for sample pairing.
        ot_regularization: Entropic regularization for OT coupling.
    """
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    time_embed_dim: int = 64
    learning_rate: float = 1e-4
    ema_decay: float = 0.999
    use_bridge_matching: bool = True
    weight_by_sigma: bool = True
    use_ot_coupling: bool = False
    ot_regularization: float = 0.1
    network_factory: Optional[NetworkFactory] = None


class ScoreBasedSolver(SBSolver):
    """Score-based Schrödinger Bridge solver.
    
    Learns the score function grad log p_t(x) directly using denoising score matching.
    The key insight is that for SB, we can construct training pairs using
    Brownian bridge samples between source-target pairs.
    
    Training objective (bridge score matching):
        L = E_{x0~mu0, x1~mu1, t~U[0,1], xt~Bridge(x0,x1,t)} [||s_theta(xt,t) - grad log p(xt|x0,x1)||^2]
    
    The conditional score grad log p(xt|x0,x1) has closed form for Brownian bridge.
    """
    
    def __init__(
        self,
        problem: SBProblem,
        sb_config: Optional[ScoreBasedConfig] = None,
        config: Optional[Union[ScoreBasedConfig, SolverConfig]] = None,
        solver_config: Optional[SolverConfig] = None,
        **kwargs,
    ):
        """Initialize Score-Based solver.
        
        Args:
            problem: SB problem specification.
            sb_config: Score-based specific configuration.
            config: Can be either ScoreBasedConfig or SolverConfig (for convenience).
            solver_config: Base solver configuration (explicit).
            **kwargs: Additional arguments for base class.
        
        Examples:
            # All these work:
            solver = ScoreBasedSolver(problem, sb_config=ScoreBasedConfig(...))
            solver = ScoreBasedSolver(problem, config=ScoreBasedConfig(...))
            solver = ScoreBasedSolver(problem, ScoreBasedConfig(...))
        """
        # Handle config parameter flexibility
        if sb_config is None and config is not None:
            if isinstance(config, ScoreBasedConfig):
                sb_config = config
                config = None
        
        # Filter kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if not isinstance(v, ScoreBasedConfig)}
        
        # Determine base class config
        base_config = None
        if solver_config is not None:
            base_config = solver_config
        elif config is not None and isinstance(config, SolverConfig):
            base_config = config
        
        if base_config is not None:
            filtered_kwargs['config'] = base_config
            
        super().__init__(problem, **filtered_kwargs)
        self.sb_config = sb_config or ScoreBasedConfig()
        self._ema_params: Optional[Params] = None

        # Resolve network factory: custom if provided, else default MLP
        self._factory: NetworkFactory = self.sb_config.network_factory or MLPFactory(
            hidden_dims=self.sb_config.hidden_dims,
            time_embed_dim=self.sb_config.time_embed_dim,
        )
    
    @property
    def solver_type(self) -> SolverType:
        return SolverType.SCORE_BASED
    
    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.SCORE
    
    def init_params(self, key: PRNGKey) -> Params:
        """Initialize score network parameters."""
        params = self._factory.init(key, self.problem.dim, self.problem.dim)
        sanity_check(self._factory, key, self.problem.dim, self.problem.dim)
        return params
    
    def _compute_ot_coupling(
        self,
        x0: Array,
        x1: Array,
        reg: float = 0.1,
    ) -> Array:
        """Compute OT coupling using Sinkhorn algorithm.
        
        Args:
            x0: Source samples [batch, dim]
            x1: Target samples [batch, dim]
            reg: Entropic regularization
            
        Returns:
            Coupling indices to reorder x1
        """
        batch_size = x0.shape[0]
        
        # Cost matrix: C[i,j] = ||x0[i] - x1[j]||^2
        C = jnp.sum((x0[:, None, :] - x1[None, :, :]) ** 2, axis=-1)
        
        # Sinkhorn iterations
        K = jnp.exp(-C / reg)
        u = jnp.ones(batch_size)
        v = jnp.ones(batch_size)
        
        for _ in range(50):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.T @ u + 1e-8)
        
        # Get coupling matrix and find best matches
        P = u[:, None] * K * v[None, :]
        coupling = jnp.argmax(P, axis=1)
        
        return coupling
    
    def _sample_bridge_point(
        self,
        key: PRNGKey,
        x0: Array,
        x1: Array,
        t: Array,
    ) -> Tuple[Array, Array]:
        """Sample point from Brownian bridge and compute true score.
        
        Brownian bridge from x0 to x1:
            x_t = (1-t)x0 + t*x1 + sigma*sqrt(t(1-t))*z
        
        The conditional score is:
            grad log p(x_t|x0,x1) = -(x_t - mu_t) / sigma_t^2
        
        where mu_t = (1-t)x0 + t*x1 and sigma_t = sigma*sqrt(t(1-t))
        
        Returns:
            (x_t, true_score)
        """
        batch_size = x0.shape[0]
        sigma = self.problem.reference.diffusion(None, t[0])
        
        # Bridge statistics
        t_col = t[:, None]  # [batch, 1]
        bridge_mean = (1 - t_col) * x0 + t_col * x1
        bridge_std = sigma * jnp.sqrt(t_col * (1 - t_col) + 1e-6)
        
        # Sample
        z = jax.random.normal(key, x0.shape)
        x_t = bridge_mean + bridge_std * z
        
        # True score: -(x_t - mean) / var
        bridge_var = bridge_std ** 2 + 1e-8
        true_score = -(x_t - bridge_mean) / bridge_var
        
        return x_t, true_score
    
    def _loss_fn(
        self,
        params: Params,
        key: PRNGKey,
        x0: Array,
        x1: Array,
    ) -> Tuple[Scalar, Dict[str, Scalar]]:
        """Compute score matching loss."""
        batch_size = x0.shape[0]
        k1, k2 = jax.random.split(key)
        
        # Sample random times (avoid boundaries)
        t = jax.random.uniform(k1, (batch_size,), minval=0.01, maxval=0.99)
        
        # Sample bridge points and get true score
        x_t, true_score = self._sample_bridge_point(k2, x0, x1, t)
        
        # Predicted score
        pred_score = self._factory.forward(params, x_t, t)
        
        # MSE loss
        diff = pred_score - true_score
        
        if self.sb_config.weight_by_sigma:
            # Weight by noise level for stability
            sigma = self.problem.reference.diffusion(None, t[0])
            bridge_std = sigma * jnp.sqrt(t * (1 - t) + 1e-6)
            weights = bridge_std ** 2
            loss = jnp.mean(weights[:, None] * diff ** 2)
        else:
            loss = jnp.mean(diff ** 2)
        
        metrics = {
            'loss': loss,
            'score_norm': jnp.mean(jnp.linalg.norm(pred_score, axis=-1)),
        }
        
        return loss, metrics
    
    def train_step(
        self,
        key: PRNGKey,
        params: Params,
        opt_state: AdamState,
        batch_size: int,
    ) -> Tuple[Params, AdamState, Dict[str, Scalar]]:
        """Perform one training step."""
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Sample source and target
        x0 = self.problem.sample_source(k1, batch_size)
        x1 = self.problem.sample_target(k2, batch_size)
        
        # Optional OT coupling for better sample pairing
        if self.sb_config.use_ot_coupling:
            coupling = self._compute_ot_coupling(
                x0, x1, self.sb_config.ot_regularization
            )
            x1 = x1[coupling]
        
        # Compute loss and gradients
        (loss, metrics), grads = jax.value_and_grad(
            self._loss_fn, has_aux=True
        )(params, k3, x0, x1)
        
        # Update parameters
        new_params, new_opt_state = adam_update(
            opt_state, grads, params,
            lr=self.sb_config.learning_rate,
        )
        
        # Update EMA
        if self._ema_params is not None:
            self._ema_params = jax.tree_util.tree_map(
                lambda ema, new: self.sb_config.ema_decay * ema + (1 - self.sb_config.ema_decay) * new,
                self._ema_params, new_params,
            )
        else:
            self._ema_params = new_params
        
        return new_params, new_opt_state, metrics
    
    def _init_optimizer(self, params: Params) -> AdamState:
        """Initialize Adam optimizer."""
        self._ema_params = params  # Initialize EMA
        return init_adam(params)
    
    def extract_drift(self, params: Params) -> DriftFn:
        """Extract forward drift from learned score."""
        # Use EMA params if available
        score_params = self._ema_params if self._ema_params is not None else params
        factory = self._factory

        def drift(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))

            # Reference drift
            b_ref = self.problem.reference.drift(x, t)

            # Score contribution
            sigma = self.problem.reference.diffusion(x, t)
            score = factory.forward(score_params, x, t_arr)

            return b_ref + sigma ** 2 * score

        return drift
    
    def get_score_fn(self, params: Optional[Params] = None) -> Callable:
        """Get the learned score function."""
        if params is None:
            params = self._ema_params if self._ema_params is not None else self._params
        factory = self._factory

        def score(x: Array, t: Scalar) -> Array:
            x = jnp.atleast_2d(x)
            t_arr = jnp.atleast_1d(t)
            if t_arr.shape[0] == 1:
                t_arr = jnp.broadcast_to(t_arr, (x.shape[0],))
            return factory.forward(params, x, t_arr)

        return score
