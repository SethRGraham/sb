"""Network Factory Protocol for Custom Architectures.

Drop this file into: schrodinger_bridge/network_factory.py

Core math
---------
Every SB solver learns a correction f(x,t) to the reference drift:

    b*(x,t) = b_ref(x,t) + sigma^2(t) * f(x,t)

The network factory abstracts HOW f is parameterized. The solver only ever
calls init() and forward(). JAX autodiff handles grad_theta f automatically
regardless of architecture.

Main math takeaway:
    f : R^d x [0,1] -> R^d'  is just a function. Whether parameterized
    by an MLP, U-Net, or transformer is invisible to the solver. All that
    matters is the (params, x, t) -> output contract.

Solver-specific output semantics (what f MEANS to each solver):
    Solver            | f(x,t) represents        | output_dim
    ------------------+--------------------------+------------
    ScoreBasedSolver  | grad log p_t(x)  (score) | D
    FBSDESolver (Z)   | Z(x,t)       (control)   | D
    FBSDESolver (Y)   | Y(x,t)       (value fn)  | 1
    IMFSolver         | v(x,t)       (velocity)  | D
    IPFSolver         | drift correction          | D

    The factory doesn't know or care about these semantics -- the solver
    sets input_dim and output_dim explicitly when calling factory.init().
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Union

import jax
import jax.numpy as jnp

# Type aliases (matching core/types.py)
Array = jnp.ndarray
PRNGKey = jnp.ndarray
Params = Any
Scalar = Union[float, Array]


# =============================================================================
# The Protocol
# =============================================================================

class NetworkFactory(abc.ABC):
    """Abstract interface for pluggable network architectures.

    Subclass this to plug any architecture into any SB solver.
    The solver calls exactly two methods:

        params = factory.init(key, input_dim, output_dim)
        output = factory.forward(params, x, t)

    Contract:
        - x is ALWAYS [batch, input_dim]  (flat vectors)
        - t is ALWAYS [batch]             (one scalar per sample)
        - output is ALWAYS [batch, output_dim]
        - params is a JAX pytree (dicts, lists, arrays)

    The solver sets input_dim and output_dim explicitly. Do NOT assume
    output_dim == input_dim; the FBSDE value network has output_dim=1.

    Example subclass::

        class MyFancyNet(NetworkFactory):
            def __init__(self, width: int = 128):
                self.width = width

            def init(self, key, input_dim, output_dim):
                return my_custom_init(key, input_dim, output_dim, self.width)

            def forward(self, params, x, t):
                return my_custom_forward(params, x, t)
    """

    @abc.abstractmethod
    def init(self, key: PRNGKey, input_dim: int, output_dim: int) -> Params:
        """Initialize network parameters.

        Args:
            key: JAX random key.
            input_dim: Flattened input dim (e.g. 784 for 28x28 images).
            output_dim: Output dim. Set by solver: D for score/control, 1 for value.

        Returns:
            JAX pytree of parameters.
        """
        ...

    @abc.abstractmethod
    def forward(self, params: Params, x: Array, t: Array) -> Array:
        """Forward pass.

        Args:
            params: Parameters from init().
            x: Input states, shape [batch, input_dim].
            t: Time values, shape [batch].

        Returns:
            Output, shape [batch, output_dim].
        """
        ...

    def has_spatial_structure(self) -> bool:
        """Whether this factory reshapes flat vectors to spatial tensors internally."""
        return False


# =============================================================================
# Sanity Harness -- catches 80% of failures before training
# =============================================================================

def sanity_check(
    factory: NetworkFactory,
    key: PRNGKey,
    input_dim: int,
    output_dim: int,
    batch_size: int = 4,
) -> None:
    """Run pre-training sanity checks on a factory.

    Catches shape mismatches, NaN forward passes, and broken gradients
    BEFORE you waste GPU time on a training run.

    Raises:
        AssertionError with descriptive message on failure.
    """
    k1, k2 = jax.random.split(key)

    # 1. Init produces valid params
    params = factory.init(k1, input_dim, output_dim)
    leaves = jax.tree_util.tree_leaves(params)
    assert len(leaves) > 0, "init() returned empty params"
    assert all(jnp.all(jnp.isfinite(l)) for l in leaves), \
        "init() produced NaN/Inf in parameters"

    # 2. Forward shape check
    x = jax.random.normal(k2, (batch_size, input_dim))
    t = jnp.linspace(0.01, 0.99, batch_size)
    y = factory.forward(params, x, t)
    assert y.shape == (batch_size, output_dim), \
        f"forward() shape: expected {(batch_size, output_dim)}, got {y.shape}"
    assert jnp.all(jnp.isfinite(y)), "forward() produced NaN/Inf"

    # 3. Gradient check
    def dummy_loss(p):
        return jnp.mean(factory.forward(p, x, t) ** 2)

    g = jax.grad(dummy_loss)(params)
    g_leaves = jax.tree_util.tree_leaves(g)
    assert all(jnp.all(jnp.isfinite(gl)) for gl in g_leaves), \
        "jax.grad produced NaN/Inf -- check forward() for non-differentiable ops"

    # 4. Single-sample forward should not crash
    y1 = factory.forward(params, x[:1], t[:1])
    assert y1.shape == (1, output_dim), \
        f"Single-sample shape: expected {(1, output_dim)}, got {y1.shape}"


# =============================================================================
# Built-in: MLP Factory (wraps existing networks.py)
# =============================================================================

@dataclass
class MLPFactory(NetworkFactory):
    """Default MLP factory -- wraps the existing time-conditioned MLP.

    This is what every solver uses today. Making it explicit as a factory
    means it can be swapped out without touching solver internals.
    """
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    time_embed_dim: int = 64
    activation: str = 'swish'

    def init(self, key: PRNGKey, input_dim: int, output_dim: int) -> Params:
        from .networks import init_time_conditioned_mlp, TimeConditionedMLPConfig
        config = TimeConditionedMLPConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.hidden_dims,
            time_embed_dim=self.time_embed_dim,
            activation=self.activation,
        )
        return init_time_conditioned_mlp(key, config)

    def forward(self, params: Params, x: Array, t: Array) -> Array:
        from .networks import time_conditioned_mlp_forward
        return time_conditioned_mlp_forward(
            params, x, t,
            activation=self.activation,
            time_embed_dim=self.time_embed_dim,
        )


# =============================================================================
# Built-in: U-Net Factory (2D and 3D spatial data)
# =============================================================================

def _sinusoidal_embedding(t: Array, dim: int, max_period: float = 10000.0) -> Array:
    """Sinusoidal time embedding."""
    t = jnp.atleast_1d(t)
    half_dim = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half_dim) / half_dim)
    args = t[..., None] * freqs
    embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2 == 1:
        embedding = jnp.concatenate([
            embedding, jnp.zeros_like(embedding[..., :1])
        ], axis=-1)
    return embedding


def _conv2d_init(key, in_c, out_c, kernel=3):
    fan_in = in_c * kernel * kernel
    std = math.sqrt(2.0 / fan_in)
    k1, _ = jax.random.split(key)
    return {
        'w': jax.random.normal(k1, (kernel, kernel, in_c, out_c)) * std,
        'b': jnp.zeros(out_c),
    }


def _conv2d_forward(params, x):
    out = jax.lax.conv_general_dilated(
        x, params['w'], window_strides=(1, 1), padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return out + params['b']


def _conv3d_init(key, in_c, out_c, kernel=3):
    fan_in = in_c * kernel ** 3
    std = math.sqrt(2.0 / fan_in)
    k1, _ = jax.random.split(key)
    return {
        'w': jax.random.normal(k1, (kernel, kernel, kernel, in_c, out_c)) * std,
        'b': jnp.zeros(out_c),
    }


def _conv3d_forward(params, x):
    out = jax.lax.conv_general_dilated(
        x, params['w'], window_strides=(1, 1, 1), padding='SAME',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    return out + params['b']


def _groupnorm(x, num_groups=8, eps=1e-5):
    """Group normalization -- works for 4D (NHWC) and 5D (NDHWC)."""
    shape = x.shape
    N, C = shape[0], shape[-1]
    spatial = shape[1:-1]
    g = min(num_groups, C)
    x = x.reshape(N, *spatial, g, C // g)
    reduce_axes = tuple(range(1, len(spatial) + 1)) + (len(spatial) + 2,)
    mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
    var = jnp.var(x, axis=reduce_axes, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    return x.reshape(N, *spatial, C)


@dataclass
class UNetFactory(NetworkFactory):
    """U-Net factory for 2D or 3D spatial data.

    Solver passes flat [B, dim]. This factory reshapes internally, runs
    the U-Net, then flattens back. SBProblem stays unchanged.

    Args:
        spatial_shape: (H, W, C) for 2D or (D, H, W, C) for 3D.
        channels: Feature channels at each resolution level.
        time_embed_dim: Time embedding dimension.
    """
    spatial_shape: Tuple[int, ...] = (28, 28, 1)
    channels: Tuple[int, ...] = (32, 64, 128)
    time_embed_dim: int = 64

    @property
    def _ndim(self):
        return len(self.spatial_shape) - 1

    @property
    def _C(self):
        return self.spatial_shape[-1]

    def _flat_dim(self):
        r = 1
        for s in self.spatial_shape:
            r *= s
        return r

    def has_spatial_structure(self):
        return True

    def _ci(self, key, in_c, out_c, kernel=3):
        return _conv2d_init(key, in_c, out_c, kernel) if self._ndim == 2 \
            else _conv3d_init(key, in_c, out_c, kernel)

    def _cf(self, params, x):
        return _conv2d_forward(params, x) if self._ndim == 2 \
            else _conv3d_forward(params, x)

    def init(self, key, input_dim, output_dim):
        assert input_dim == self._flat_dim(), (
            f"input_dim={input_dim} != prod(spatial_shape)={self._flat_dim()}")

        keys = jax.random.split(key, 30)
        ki = iter(range(30))
        from .networks import init_mlp_params

        params = {'spatial_shape': self.spatial_shape, 'output_dim': output_dim}

        params['time_mlp'] = init_mlp_params(
            keys[next(ki)], [self.time_embed_dim, self.channels[0]*4, self.channels[0]])

        # Encoder
        params['encoder'] = []
        in_c = self._C
        for out_c in self.channels:
            params['encoder'].append({
                'conv1': self._ci(keys[next(ki)], in_c, out_c),
                'conv2': self._ci(keys[next(ki)], out_c, out_c),
                'time_proj': self._ci(keys[next(ki)], self.channels[0], out_c, kernel=1),
            })
            in_c = out_c

        # Bottleneck
        params['bottleneck'] = {
            'conv1': self._ci(keys[next(ki)], in_c, in_c),
            'conv2': self._ci(keys[next(ki)], in_c, in_c),
        }

        # Decoder
        params['decoder'] = []
        for lvl in range(len(self.channels) - 1, -1, -1):
            out_c = self.channels[lvl]
            params['decoder'].append({
                'conv1': self._ci(keys[next(ki)], in_c + out_c, out_c),
                'conv2': self._ci(keys[next(ki)], out_c, out_c),
            })
            in_c = out_c

        out_C = output_dim // (self._flat_dim() // self._C) if output_dim != input_dim else self._C
        params['out_conv'] = self._ci(keys[next(ki)], self.channels[0], out_C, kernel=1)
        return params

    def forward(self, params, x, t):
        from .networks import mlp_forward, swish
        batch = x.shape[0]
        ss = params['spatial_shape']
        ndim = len(ss) - 1

        h = x.reshape(batch, *ss)

        t_emb = _sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = mlp_forward(params['time_mlp'], t_emb, swish)
        t_spatial = t_emb.reshape(batch, *([1]*ndim), -1)

        skips = []
        for lp in params['encoder']:
            h = swish(_groupnorm(self._cf(lp['conv1'], h)))
            tp = self._cf(lp['time_proj'], t_spatial)
            h = h + jnp.broadcast_to(tp, h.shape)
            h = swish(_groupnorm(self._cf(lp['conv2'], h)))
            skips.append(h)
            if all(s > 2 for s in h.shape[1:-1]):
                win = (1,) + (2,)*ndim + (1,)
                h = jax.lax.reduce_window(h, 0.0, jax.lax.add, win, win, 'SAME') / (2**ndim)

        h = swish(_groupnorm(self._cf(params['bottleneck']['conv1'], h)))
        h = swish(_groupnorm(self._cf(params['bottleneck']['conv2'], h)))

        for i, lp in enumerate(params['decoder']):
            skip = skips[len(skips) - 1 - i]
            if h.shape[1:-1] != skip.shape[1:-1]:
                h = jax.image.resize(h, skip.shape, method='nearest')
            h = jnp.concatenate([h, skip], axis=-1)
            h = swish(_groupnorm(self._cf(lp['conv1'], h)))
            h = swish(_groupnorm(self._cf(lp['conv2'], h)))

        h = self._cf(params['out_conv'], h)
        od = params['output_dim']
        return h.reshape(batch, -1)[:, :od]


# =============================================================================
# Built-in: Transformer Factory
# =============================================================================

@dataclass
class TransformerFactory(NetworkFactory):
    """Transformer factory for sequence-structured data.

    Treats input as tokens with self-attention + time conditioning.
    input_dim must equal token_dim * num_tokens.
    """
    token_dim: int = 1
    num_tokens: int = 10
    num_heads: int = 4
    num_layers: int = 2
    hidden_dim: int = 128
    time_embed_dim: int = 64

    def init(self, key, input_dim, output_dim):
        assert input_dim == self.token_dim * self.num_tokens
        from .networks import init_linear_params, init_mlp_params, xavier_uniform

        keys = jax.random.split(key, 4 * self.num_layers + 10)
        ki = iter(range(len(keys)))
        d = self.hidden_dim

        params = {
            'num_tokens': self.num_tokens,
            'token_dim': self.token_dim,
            'output_dim': output_dim,
        }

        params['input_proj'] = init_linear_params(keys[next(ki)], self.token_dim, d)
        params['time_mlp'] = init_mlp_params(keys[next(ki)], [self.time_embed_dim, d*2, d])

        params['layers'] = []
        for _ in range(self.num_layers):
            params['layers'].append({
                'Wq': xavier_uniform(keys[next(ki)], (d, d)),
                'Wk': xavier_uniform(keys[next(ki)], (d, d)),
                'Wv': xavier_uniform(keys[next(ki)], (d, d)),
                'Wo': init_linear_params(keys[next(ki)], d, d),
                'ffn': init_mlp_params(keys[next(ki)], [d, d*2, d]),
            })

        out_token_dim = output_dim // self.num_tokens
        params['output_proj'] = init_linear_params(keys[next(ki)], d, out_token_dim)
        params['_out_token_dim'] = out_token_dim
        return params

    def forward(self, params, x, t):
        from .networks import linear_forward, mlp_forward, swish

        B = x.shape[0]
        n = params['num_tokens']
        d = params['layers'][0]['Wq'].shape[0]
        nh = self.num_heads
        hd = d // nh

        tokens = x.reshape(B, n, params['token_dim'])
        h = jax.vmap(lambda tok: linear_forward(params['input_proj'], tok))(tokens)

        te = _sinusoidal_embedding(t, self.time_embed_dim)
        te = mlp_forward(params['time_mlp'], te, swish)
        h = h + te[:, None, :]

        for layer in params['layers']:
            res = h
            Q = (h @ layer['Wq']).reshape(B, n, nh, hd).transpose(0, 2, 1, 3)
            K = (h @ layer['Wk']).reshape(B, n, nh, hd).transpose(0, 2, 1, 3)
            V = (h @ layer['Wv']).reshape(B, n, nh, hd).transpose(0, 2, 1, 3)

            attn = jax.nn.softmax((Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(hd), axis=-1)
            out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, n, d)
            out = jax.vmap(lambda o: linear_forward(layer['Wo'], o))(out)
            h = res + out
            h = h + jax.vmap(lambda tok: mlp_forward(layer['ffn'], tok, swish))(h)

        out = jax.vmap(lambda tok: linear_forward(params['output_proj'], tok))(h)
        od = params['output_dim']
        return out.reshape(B, -1)[:, :od]


# =============================================================================
# Escape hatch: wrap any (init_fn, forward_fn) pair
# =============================================================================

@dataclass
class CustomFactory(NetworkFactory):
    """Wrap arbitrary init/forward functions as a NetworkFactory.

    Example::

        factory = CustomFactory(
            init_fn=lambda key, d_in, d_out: my_init(key, d_in, d_out),
            forward_fn=lambda params, x, t: my_forward(params, x, t),
        )
    """
    init_fn: Callable = None
    forward_fn: Callable = None

    def init(self, key, input_dim, output_dim):
        return self.init_fn(key, input_dim, output_dim)

    def forward(self, params, x, t):
        return self.forward_fn(params, x, t)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'NetworkFactory', 'MLPFactory', 'UNetFactory',
    'TransformerFactory', 'CustomFactory', 'sanity_check',
]
