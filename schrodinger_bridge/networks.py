"""Neural Network Implementations for Schrödinger Bridge solvers.

Pure JAX implementations of neural networks for parameterizing
drift, score, control, and potential functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .core.types import Array, NetworkConfig, PRNGKey, Params, Scalar


# Activation Functions
def swish(x: Array) -> Array:
    """Swish activation: x * sigmoid(x)."""
    return x * jax.nn.sigmoid(x)


def gelu(x: Array) -> Array:
    """GELU activation."""
    return jax.nn.gelu(x)


def get_activation(name: str) -> Callable[[Array], Array]:
    """Get activation function by name."""
    activations = {
        'relu': jax.nn.relu,
        'tanh': jnp.tanh,
        'sigmoid': jax.nn.sigmoid,
        'swish': swish,
        'silu': swish,  # Alias
        'gelu': gelu,
        'softplus': jax.nn.softplus,
        'elu': jax.nn.elu,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]


# Weight Initialization
def xavier_uniform(key: PRNGKey, shape: Tuple[int, ...]) -> Array:
    """Xavier/Glorot uniform initialization."""
    fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def xavier_normal(key: PRNGKey, shape: Tuple[int, ...]) -> Array:
    """Xavier/Glorot normal initialization."""
    fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return jax.random.normal(key, shape) * std


def kaiming_uniform(key: PRNGKey, shape: Tuple[int, ...]) -> Array:
    """Kaiming/He uniform initialization."""
    fan_in = shape[0]
    limit = math.sqrt(3.0 / fan_in)
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def init_linear_params(
    key: PRNGKey,
    in_dim: int,
    out_dim: int,
    use_bias: bool = True,
) -> Dict[str, Array]:
    """Initialize parameters for a linear layer.
    
    Args:
        key: JAX random key.
        in_dim: Input dimension.
        out_dim: Output dimension.
        use_bias: Whether to include bias.
        
    Returns:
        Dictionary with 'w' (weight) and optionally 'b' (bias).
    """
    k1, k2 = jax.random.split(key)
    params = {
        'w': xavier_uniform(k1, (in_dim, out_dim)),
    }
    if use_bias:
        params['b'] = jnp.zeros(out_dim)
    return params


# Time Embedding
def sinusoidal_embedding(t: Array, dim: int, max_period: float = 10000.0) -> Array:
    """Sinusoidal positional embedding for time.
    
    Following transformer-style positional encoding.
    
    Args:
        t: Time values, shape [...].
        dim: Embedding dimension.
        max_period: Maximum period.
        
    Returns:
        Embeddings, shape [..., dim].
    """
    t = jnp.atleast_1d(t)
    half_dim = dim // 2
    frequency_index = jnp.arange(half_dim, dtype=t.dtype)
    log_period = jnp.asarray(math.log(max_period), dtype=t.dtype)
    freqs = jnp.exp(-log_period * frequency_index / jnp.asarray(half_dim, dtype=t.dtype))
    args = t[..., None] * freqs
    embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    
    # Handle odd dimensions
    if dim % 2 == 1:
        embedding = jnp.concatenate([
            embedding, jnp.zeros_like(embedding[..., :1])
        ], axis=-1)
    
    return embedding


def random_fourier_features(
    t: Array,
    dim: int,
    scale: float = 16.0,
    key: Optional[PRNGKey] = None,
) -> Array:
    """Random Fourier features for time embedding.
    
    Args:
        t: Time values.
        dim: Output dimension.
        scale: Frequency scale.
        key: Random key (uses fixed if None).
        
    Returns:
        Features, shape [..., dim].
    """
    t = jnp.atleast_1d(t)
    half_dim = dim // 2
    
    if key is None:
        # Fixed frequencies for reproducibility
        freqs = jnp.linspace(0.1, scale, half_dim)
    else:
        freqs = jax.random.normal(key, (half_dim,)) * scale
    
    args = 2 * jnp.pi * t[..., None] * freqs
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


# MLP Building Blocks
def linear_forward(params: Dict[str, Array], x: Array) -> Array:
    """Apply linear transformation: y = xW + b."""
    y = x @ params['w']
    if 'b' in params:
        y = y + params['b']
    return y


def mlp_forward(
    params: List[Dict[str, Array]],
    x: Array,
    activation: Callable[[Array], Array] = swish,
    final_activation: bool = False,
) -> Array:
    """Forward pass through MLP.
    
    Args:
        params: List of layer parameters.
        x: Input, shape [batch, in_dim].
        activation: Activation function.
        final_activation: Apply activation to final layer.
        
    Returns:
        Output, shape [batch, out_dim].
    """
    for i, layer_params in enumerate(params):
        x = linear_forward(layer_params, x)
        # Apply activation except possibly last layer
        if i < len(params) - 1 or final_activation:
            x = activation(x)
    return x


def init_mlp_params(
    key: PRNGKey,
    layer_dims: Sequence[int],
    use_bias: bool = True,
) -> List[Dict[str, Array]]:
    """Initialize MLP parameters.
    
    Args:
        key: JAX random key.
        layer_dims: Sequence of layer dimensions [in, h1, h2, ..., out].
        use_bias: Whether to use biases.
        
    Returns:
        List of layer parameter dictionaries.
    """
    params = []
    keys = jax.random.split(key, len(layer_dims) - 1)
    
    for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        params.append(init_linear_params(keys[i], in_dim, out_dim, use_bias))
    
    return params


# Time-Conditioned MLP
@dataclass
class TimeConditionedMLPConfig:
    """Configuration for time-conditioned MLP."""
    input_dim: int
    output_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    time_embed_dim: int = 64
    activation: str = 'swish'
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    output_scale: float = 1.0


def init_time_conditioned_mlp(
    key: PRNGKey,
    config: TimeConditionedMLPConfig,
) -> Params:
    """Initialize time-conditioned MLP parameters.
    
    Architecture:
    1. Time embedding (sinusoidal)
    2. Time embedding projection
    3. Main MLP with time conditioning injected at each layer
    
    Args:
        key: JAX random key.
        config: Network configuration.
        
    Returns:
        Parameter dictionary.
    """
    keys = jax.random.split(key, 10)
    
    params = {}
    
    # Time embedding projection: time_embed_dim -> first hidden dim
    params['time_proj'] = init_mlp_params(
        keys[0],
        [config.time_embed_dim, config.hidden_dims[0], config.hidden_dims[0]],
    )
    
    # Main network layers
    # Input projection
    params['input_proj'] = init_linear_params(
        keys[1], config.input_dim, config.hidden_dims[0]
    )
    
    # Hidden layers with time injection
    params['hidden'] = []
    for i, (h_in, h_out) in enumerate(zip(
        config.hidden_dims[:-1], config.hidden_dims[1:]
    )):
        layer_params = {
            'main': init_linear_params(keys[2 + i], h_in, h_out),
            'time': init_linear_params(keys[2 + i + len(config.hidden_dims)], h_in, h_out),
        }
        params['hidden'].append(layer_params)
    
    # Output projection
    params['output_proj'] = init_linear_params(
        keys[9], config.hidden_dims[-1], config.output_dim
    )
    
    return params


# Store config separately from params - global cache
_NETWORK_CONFIGS = {}


def register_network_config(params_id: int, config: TimeConditionedMLPConfig):
    """Register a config for a params dict."""
    _NETWORK_CONFIGS[params_id] = config


def get_network_config(params: Params) -> TimeConditionedMLPConfig:
    """Get config for params, or use defaults."""
    params_id = id(params)
    if params_id in _NETWORK_CONFIGS:
        return _NETWORK_CONFIGS[params_id]
    # Return default config
    return TimeConditionedMLPConfig(
        input_dim=2, output_dim=2, hidden_dims=(256, 256, 256),
        activation='swish', time_embed_dim=64, output_scale=1.0
    )


def time_conditioned_mlp_forward(
    params: Params,
    x: Array,
    t: Array,
    train: bool = False,
    key: Optional[PRNGKey] = None,
    activation: str = 'swish',
    time_embed_dim: int = 64,
    output_scale: float = 1.0,
) -> Array:
    """Forward pass through time-conditioned MLP.
    
    Args:
        params: Network parameters.
        x: Input, shape [batch, input_dim].
        t: Time, shape [batch] or scalar.
        train: Training mode (for dropout).
        key: Random key for dropout.
        activation: Activation function name.
        time_embed_dim: Time embedding dimension.
        output_scale: Output scaling factor.
        
    Returns:
        Output, shape [batch, output_dim].
    """
    act_fn = get_activation(activation)
    
    # Ensure proper shapes
    x = jnp.atleast_2d(x)
    t = jnp.atleast_1d(t)
    
    # Broadcast t if needed
    if t.shape[0] == 1 and x.shape[0] > 1:
        t = jnp.broadcast_to(t, (x.shape[0],))
    
    # Time embedding
    t_emb = sinusoidal_embedding(t, time_embed_dim)
    t_emb = mlp_forward(params['time_proj'], t_emb, act_fn)
    
    # Input projection
    h = linear_forward(params['input_proj'], x)
    h = act_fn(h)
    
    # Add time embedding
    h = h + t_emb
    
    # Hidden layers with time conditioning
    for layer_params in params['hidden']:
        h_main = linear_forward(layer_params['main'], h)
        h_time = linear_forward(layer_params['time'], t_emb)
        h = act_fn(h_main + h_time)
    
    # Output
    out = linear_forward(params['output_proj'], h)
    
    return out * output_scale


# Score Network (for score-based methods)
def init_score_network(
    key: PRNGKey,
    dim: int,
    hidden_dims: Tuple[int, ...] = (256, 256, 256),
    time_embed_dim: int = 64,
) -> Params:
    """Initialize score network.
    
    The score network estimates grad log p_t(x).
    """
    config = TimeConditionedMLPConfig(
        input_dim=dim,
        output_dim=dim,
        hidden_dims=hidden_dims,
        time_embed_dim=time_embed_dim,
        output_scale=1.0,
    )
    return init_time_conditioned_mlp(key, config)


def score_network_forward(
    params: Params,
    x: Array,
    t: Array,
    sigma_fn: Optional[Callable[[Array], Array]] = None,
) -> Array:
    """Compute score estimate with optional preconditioning.
    
    Args:
        params: Network parameters.
        x: Input points.
        t: Time.
        sigma_fn: Optional noise schedule for preconditioning.
        
    Returns:
        Score estimate grad log p_t(x).
    """
    score = time_conditioned_mlp_forward(params, x, t)
    
    # Optional preconditioning by 1/sigma(t)
    if sigma_fn is not None:
        sigma = sigma_fn(t)
        if sigma.ndim == 0:
            sigma = sigma * jnp.ones(x.shape[0])
        score = score / sigma[:, None]
    
    return score


# Potential Network (for Doob h-transform)
def init_potential_network(
    key: PRNGKey,
    dim: int,
    hidden_dims: Tuple[int, ...] = (256, 256, 256),
    time_embed_dim: int = 64,
) -> Params:
    """Initialize potential network.
    
    Outputs scalar potential psi(x, t) whose gradient gives drift correction.
    """
    config = TimeConditionedMLPConfig(
        input_dim=dim,
        output_dim=1,
        hidden_dims=hidden_dims,
        time_embed_dim=time_embed_dim,
    )
    return init_time_conditioned_mlp(key, config)


def potential_network_forward(params: Params, x: Array, t: Array) -> Array:
    """Evaluate potential psi(x, t).
    
    Returns:
        Potential values, shape [batch].
    """
    return time_conditioned_mlp_forward(params, x, t).squeeze(-1)


def potential_network_gradient(params: Params, x: Array, t: Array) -> Array:
    """Compute grad_x psi(x, t).
    
    Returns:
        Gradient, shape [batch, dim].
    """
    def potential_single(x_single, t_single):
        return potential_network_forward(params, x_single[None], t_single[None])[0]
    
    grad_fn = jax.vmap(jax.grad(potential_single, argnums=0))
    return grad_fn(x, t)


# Input Convex Neural Network (ICNN)
def init_icnn_params(
    key: PRNGKey,
    input_dim: int,
    hidden_dims: Tuple[int, ...] = (256, 256, 256),
) -> Params:
    """Initialize Input Convex Neural Network.
    
    ICNN architecture ensures the output is convex in the input x.
    Key constraint: weights for z (hidden) path must be non-negative.
    
    Args:
        key: Random key.
        input_dim: Input dimension.
        hidden_dims: Hidden layer dimensions.
        
    Returns:
        Parameter dictionary.
    """
    keys = jax.random.split(key, 2 * len(hidden_dims) + 2)
    
    params = {
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'Wz': [],  # Weights for z path (will be constrained non-negative)
        'Wx': [],  # Weights for x skip connection
        'bz': [],  # Biases
    }
    
    # First layer: only x input
    params['Wx_first'] = xavier_uniform(keys[0], (input_dim, hidden_dims[0]))
    params['b_first'] = jnp.zeros(hidden_dims[0])
    
    # Hidden layers
    key_idx = 1
    dims = list(hidden_dims)
    for i in range(len(dims) - 1):
        params['Wz'].append(xavier_uniform(keys[key_idx], (dims[i], dims[i+1])))
        params['Wx'].append(xavier_uniform(keys[key_idx + 1], (input_dim, dims[i+1])))
        params['bz'].append(jnp.zeros(dims[i+1]))
        key_idx += 2
    
    # Output layer
    params['Wz_out'] = xavier_uniform(keys[key_idx], (dims[-1], 1))
    params['Wx_out'] = xavier_uniform(keys[key_idx + 1], (input_dim, 1))
    params['b_out'] = jnp.zeros(1)
    
    return params


def icnn_forward(params: Params, x: Array) -> Array:
    """Forward pass through ICNN.
    
    Ensures convexity by:
    1. Using softplus on Wz weights to make them non-negative
    2. Using convex non-decreasing activation (softplus)
    
    Args:
        params: ICNN parameters.
        x: Input, shape [batch, dim].
        
    Returns:
        Convex potential, shape [batch].
    """
    x = jnp.atleast_2d(x)
    
    # First layer
    z = x @ params['Wx_first'] + params['b_first']
    z = jax.nn.softplus(z)
    
    # Hidden layers
    for Wz, Wx, bz in zip(params['Wz'], params['Wx'], params['bz']):
        # Ensure Wz is non-negative via softplus
        Wz_pos = jax.nn.softplus(Wz)
        z = z @ Wz_pos + x @ Wx + bz
        z = jax.nn.softplus(z)
    
    # Output layer
    Wz_out_pos = jax.nn.softplus(params['Wz_out'])
    out = z @ Wz_out_pos + x @ params['Wx_out'] + params['b_out']
    
    return out.squeeze(-1)


def icnn_gradient(params: Params, x: Array) -> Array:
    """Compute gradient of ICNN (the transport map).
    
    For Brenier potential, grad phi gives the optimal transport map.
    """
    grad_fn = jax.vmap(jax.grad(lambda x_: icnn_forward(params, x_[None])[0]))
    return grad_fn(x)


# Simple Optimizer (Pure JAX)
@dataclass
class AdamState:
    """Adam optimizer state."""
    m: Params  # First moment
    v: Params  # Second moment
    step: int

    def tree_flatten(self):
        return (self.m, self.v, jnp.asarray(self.step)), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        m, v, step = children
        return cls(m=m, v=v, step=step)


jax.tree_util.register_pytree_node_class(AdamState)


def init_adam(params: Params) -> AdamState:
    """Initialize Adam optimizer state."""
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(m=m, v=v, step=0)


def adam_update(
    state: AdamState,
    grads: Params,
    params: Params,
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> Tuple[Params, AdamState]:
    """Perform Adam update."""
    step = state.step + 1
    
    # Update moments
    m = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1 - beta1) * g,
        state.m, grads
    )
    v = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1 - beta2) * g ** 2,
        state.v, grads
    )
    
    # Bias correction
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta1 ** step), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta2 ** step), v)
    
    # Update parameters
    def update_param(p, m_h, v_h):
        update = lr * m_h / (jnp.sqrt(v_h) + eps)
        if weight_decay > 0:
            update = update + weight_decay * lr * p
        return p - update
    
    new_params = jax.tree_util.tree_map(update_param, params, m_hat, v_hat)
    new_state = AdamState(m=m, v=v, step=step)
    
    return new_params, new_state


def create_default_factory(
    hidden_dims: Tuple[int, ...] = (256, 256, 256),
    time_embed_dim: int = 64,
    activation: str = 'swish',
):
    """Create the default MLP factory.

    Convenience function for users who want to explicitly create the default
    factory without importing from network_factory.
    """
    from .network_factory import MLPFactory
    return MLPFactory(
        hidden_dims=hidden_dims,
        time_embed_dim=time_embed_dim,
        activation=activation,
    )


# Module Exports
__all__ = [
    # Activation
    'swish', 'gelu', 'get_activation',
    # Initialization
    'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
    'init_linear_params', 'init_mlp_params',
    # Embeddings
    'sinusoidal_embedding', 'random_fourier_features',
    # MLP
    'linear_forward', 'mlp_forward',
    # Time-conditioned networks
    'TimeConditionedMLPConfig',
    'init_time_conditioned_mlp', 'time_conditioned_mlp_forward',
    # Score network
    'init_score_network', 'score_network_forward',
    # Potential network
    'init_potential_network', 'potential_network_forward', 'potential_network_gradient',
    # ICNN
    'init_icnn_params', 'icnn_forward', 'icnn_gradient',
    # Optimizer
    'AdamState', 'init_adam', 'adam_update',
    # Factory convenience
    'create_default_factory',
]
