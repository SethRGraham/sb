"""
JAX Integration Tests for NetworkFactory.

Drop into: tests/test_network_factory.py
Requires: pip install pytest jax jaxlib

Run: pytest tests/test_network_factory.py -v

Tests the full pipeline:
  - Factory init / forward / shapes for all built-in factories
  - jax.grad works through every factory
  - jax.jit works with every factory
  - Sanity harness catches bad factories
  - Mock solver integration (ScoreBased, FBSDE, IMF, IPF patterns)
  - 2D and 3D spatial data
  - Variable batch sizes
  - Backward compatibility (no factory = default MLP)
"""

import pytest
import jax
import jax.numpy as jnp
import numpy.testing as npt

from schrodinger_bridge.network_factory import (
    NetworkFactory,
    MLPFactory,
    UNetFactory,
    TransformerFactory,
    CustomFactory,
    sanity_check,
)
from schrodinger_bridge.networks import (
    init_mlp_params,
    mlp_forward,
    sinusoidal_embedding,
    swish,
    init_linear_params,
    linear_forward,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def mlp_factory():
    return MLPFactory(hidden_dims=(32, 32), time_embed_dim=16)


@pytest.fixture
def unet_factory_2d():
    return UNetFactory(spatial_shape=(8, 8, 1), channels=(4, 8), time_embed_dim=8)


@pytest.fixture
def unet_factory_3d():
    return UNetFactory(spatial_shape=(8, 8, 8, 1), channels=(4,), time_embed_dim=8)


@pytest.fixture
def transformer_factory():
    return TransformerFactory(
        token_dim=1, num_tokens=4,
        num_heads=2, num_layers=1, hidden_dim=8, time_embed_dim=8,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Basic Shape Contract Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMLPFactory:

    def test_shape_square(self, key, mlp_factory):
        """output_dim == input_dim (score, control, velocity)."""
        params = mlp_factory.init(key, 10, 10)
        x = jax.random.normal(key, (8, 10))
        t = jnp.linspace(0.01, 0.99, 8)
        y = mlp_factory.forward(params, x, t)
        assert y.shape == (8, 10)

    def test_shape_scalar_output(self, key, mlp_factory):
        """output_dim == 1 (FBSDE value function Y)."""
        params = mlp_factory.init(key, 10, 1)
        x = jax.random.normal(key, (8, 10))
        t = jnp.linspace(0.01, 0.99, 8)
        y = mlp_factory.forward(params, x, t)
        assert y.shape == (8, 1)

    def test_not_spatial(self, mlp_factory):
        assert not mlp_factory.has_spatial_structure()

    @pytest.mark.parametrize("dim", [1, 2, 10, 50, 100])
    def test_various_dims(self, key, mlp_factory, dim):
        params = mlp_factory.init(key, dim, dim)
        x = jax.random.normal(key, (4, dim))
        t = jnp.ones(4) * 0.5
        y = mlp_factory.forward(params, x, t)
        assert y.shape == (4, dim)
        assert jnp.all(jnp.isfinite(y))


class TestUNetFactory2D:

    def test_shape(self, key, unet_factory_2d):
        dim = 8 * 8 * 1
        params = unet_factory_2d.init(key, dim, dim)
        x = jax.random.normal(key, (4, dim))
        t = jnp.linspace(0.1, 0.9, 4)
        y = unet_factory_2d.forward(params, x, t)
        assert y.shape == (4, dim)
        assert jnp.all(jnp.isfinite(y))

    def test_is_spatial(self, unet_factory_2d):
        assert unet_factory_2d.has_spatial_structure()

    def test_rejects_wrong_dim(self, key, unet_factory_2d):
        with pytest.raises(AssertionError, match="input_dim"):
            unet_factory_2d.init(key, 100, 100)


class TestUNetFactory3D:

    def test_shape(self, key, unet_factory_3d):
        dim = 8 * 8 * 8 * 1
        params = unet_factory_3d.init(key, dim, dim)
        x = jax.random.normal(key, (2, dim))
        t = jnp.array([0.3, 0.7])
        y = unet_factory_3d.forward(params, x, t)
        assert y.shape == (2, dim)
        assert jnp.all(jnp.isfinite(y))


class TestTransformerFactory:

    def test_shape(self, key, transformer_factory):
        dim = 4  # 4 tokens × 1 dim
        params = transformer_factory.init(key, dim, dim)
        x = jax.random.normal(key, (8, dim))
        t = jnp.linspace(0.1, 0.9, 8)
        y = transformer_factory.forward(params, x, t)
        assert y.shape == (8, dim)
        assert jnp.all(jnp.isfinite(y))

    def test_rejects_wrong_dim(self, key, transformer_factory):
        with pytest.raises(AssertionError):
            transformer_factory.init(key, 7, 7)  # 7 != 4*1


class TestCustomFactory:

    def test_wraps_functions(self, key):
        factory = CustomFactory(
            init_fn=lambda k, d_in, d_out: {
                'w': jax.random.normal(k, (d_in + 16, d_out)) * 0.01
            },
            forward_fn=lambda p, x, t: jnp.concatenate(
                [x, sinusoidal_embedding(t, 16)], axis=-1
            ) @ p['w'],
        )
        params = factory.init(key, 3, 3)
        y = factory.forward(params, jnp.ones((2, 3)), jnp.array([0.5, 0.5]))
        assert y.shape == (2, 3)


# ═══════════════════════════════════════════════════════════════════════════
# JAX Autodiff Tests (the critical thing)
# ═══════════════════════════════════════════════════════════════════════════

class TestGradient:
    """Verify jax.grad works through every factory — this is the whole point."""

    def _check_grad(self, factory, key, input_dim, output_dim):
        params = factory.init(key, input_dim, output_dim)
        x = jax.random.normal(key, (4, input_dim))
        t = jnp.linspace(0.1, 0.9, 4)

        def loss(p):
            return jnp.mean(factory.forward(p, x, t) ** 2)

        grads = jax.grad(loss)(params)
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0, "No gradient leaves"
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves), \
            "NaN/Inf in gradients"

    def test_mlp_grad(self, key, mlp_factory):
        self._check_grad(mlp_factory, key, 5, 5)

    def test_mlp_grad_scalar_output(self, key, mlp_factory):
        self._check_grad(mlp_factory, key, 5, 1)

    def test_unet_grad(self, key, unet_factory_2d):
        self._check_grad(unet_factory_2d, key, 64, 64)

    def test_unet_3d_grad(self, key, unet_factory_3d):
        self._check_grad(unet_factory_3d, key, 512, 512)

    def test_transformer_grad(self, key, transformer_factory):
        self._check_grad(transformer_factory, key, 4, 4)

    def test_custom_grad(self, key):
        factory = CustomFactory(
            init_fn=lambda k, d_in, d_out: {
                'w': jax.random.normal(k, (d_in + 8, d_out)) * 0.01
            },
            forward_fn=lambda p, x, t: jnp.concatenate(
                [x, sinusoidal_embedding(t, 8)], axis=-1
            ) @ p['w'],
        )
        self._check_grad(factory, key, 3, 3)


# ═══════════════════════════════════════════════════════════════════════════
# JIT Compilation Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestJIT:
    """Verify factories work under jax.jit (catches tracing issues)."""

    def test_mlp_jit(self, key, mlp_factory):
        params = mlp_factory.init(key, 5, 5)
        x = jax.random.normal(key, (4, 5))
        t = jnp.ones(4) * 0.5

        @jax.jit
        def f(p, x, t):
            return mlp_factory.forward(p, x, t)

        y = f(params, x, t)
        assert y.shape == (4, 5)
        assert jnp.all(jnp.isfinite(y))

    def test_unet_jit(self, key, unet_factory_2d):
        dim = 64
        params = unet_factory_2d.init(key, dim, dim)
        x = jax.random.normal(key, (2, dim))
        t = jnp.array([0.3, 0.7])

        @jax.jit
        def f(p, x, t):
            return unet_factory_2d.forward(p, x, t)

        y = f(params, x, t)
        assert y.shape == (2, dim)

    def test_jit_grad_combined(self, key, mlp_factory):
        """JIT around a grad computation — the real solver pattern."""
        params = mlp_factory.init(key, 5, 5)
        x = jax.random.normal(key, (4, 5))
        t = jnp.ones(4) * 0.5

        @jax.jit
        def train_step(p):
            def loss(p):
                return jnp.mean(mlp_factory.forward(p, x, t) ** 2)
            g = jax.grad(loss)(p)
            return g

        grads = train_step(params)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads))


# ═══════════════════════════════════════════════════════════════════════════
# Sanity Harness Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSanityCheck:

    def test_passes_for_good_factory(self, key, mlp_factory):
        sanity_check(mlp_factory, key, 5, 5)  # should not raise

    def test_catches_nan_params(self, key):
        class NaNFactory(NetworkFactory):
            def init(self, key, d_in, d_out):
                return {'w': jnp.full((d_in, d_out), jnp.nan)}
            def forward(self, params, x, t):
                return x @ params['w']

        with pytest.raises(AssertionError, match="NaN"):
            sanity_check(NaNFactory(), key, 3, 3)

    def test_catches_wrong_shape(self, key):
        class BadShape(NetworkFactory):
            def init(self, key, d_in, d_out):
                return {'w': jax.random.normal(key, (d_in, d_in))}  # wrong: d_in not d_out
            def forward(self, params, x, t):
                return x @ params['w']  # returns [B, d_in] not [B, d_out]

        with pytest.raises(AssertionError, match="shape"):
            sanity_check(BadShape(), key, 3, 5)  # 3 != 5


# ═══════════════════════════════════════════════════════════════════════════
# Solver Integration Tests (mock the real solver pattern)
# ═══════════════════════════════════════════════════════════════════════════

class TestSolverIntegration:
    """Test the exact pattern each solver uses."""

    def test_score_based_pattern(self, key, mlp_factory):
        """ScoreBasedSolver: 1 network, R^d → R^d."""
        dim = 5
        factory = mlp_factory
        k1, k2, k3 = jax.random.split(key, 3)

        # init_params
        params = factory.init(k1, dim, dim)
        sanity_check(factory, k2, dim, dim)

        # _loss_fn
        x = jax.random.normal(k3, (8, dim))
        t = jax.random.uniform(k3, (8,), minval=0.01, maxval=0.99)
        pred_score = factory.forward(params, x, t)
        target_score = jax.random.normal(k3, (8, dim))
        loss = jnp.mean((pred_score - target_score) ** 2)
        assert jnp.isfinite(loss)

        # extract_drift
        def drift(x, t_scalar):
            x = jnp.atleast_2d(x)
            t_arr = jnp.broadcast_to(jnp.atleast_1d(t_scalar), (x.shape[0],))
            sigma = 0.5
            score = factory.forward(params, x, t_arr)
            return sigma ** 2 * score

        d = drift(jnp.ones((2, dim)), 0.5)
        assert d.shape == (2, dim)

    def test_fbsde_pattern(self, key):
        """FBSDESolver: 2 factories, Z: R^d→R^d, Y: R^d→R^1."""
        dim = 5
        z_factory = MLPFactory(hidden_dims=(16,), time_embed_dim=8)
        y_factory = MLPFactory(hidden_dims=(16,), time_embed_dim=8)
        k1, k2, k3 = jax.random.split(key, 3)

        z_params = z_factory.init(k1, dim, dim)
        y_params = y_factory.init(k2, dim, 1)
        params = {'z': z_params, 'y': y_params}

        x = jax.random.normal(k3, (4, dim))
        t = jnp.ones(4) * 0.5

        z_out = z_factory.forward(params['z'], x, t)
        assert z_out.shape == (4, dim), f"Z: {z_out.shape}"

        y_out = y_factory.forward(params['y'], x, t).squeeze(-1)
        assert y_out.shape == (4,), f"Y: {y_out.shape}"

        # Grad through both
        def loss(p):
            z = z_factory.forward(p['z'], x, t)
            y = y_factory.forward(p['y'], x, t)
            return jnp.mean(z ** 2) + jnp.mean(y ** 2)

        g = jax.grad(loss)(params)
        assert 'z' in g and 'y' in g

    def test_imf_pattern(self, key):
        """IMFSolver: 2 networks from same factory, both R^d→R^d."""
        dim = 5
        factory = MLPFactory(hidden_dims=(16,), time_embed_dim=8)
        k1, k2, k3 = jax.random.split(key, 3)

        fwd_params = factory.init(k1, dim, dim)
        bwd_params = factory.init(k2, dim, dim)
        params = {'forward': fwd_params, 'backward': bwd_params}

        x = jax.random.normal(k3, (4, dim))
        t = jnp.ones(4) * 0.5

        fwd_v = factory.forward(params['forward'], x, t)
        bwd_v = factory.forward(params['backward'], x, t)
        assert fwd_v.shape == bwd_v.shape == (4, dim)

    def test_ipf_pattern(self, key):
        """IPFSolver: 2 networks, forward + backward drift correction."""
        dim = 5
        factory = MLPFactory(hidden_dims=(16,), time_embed_dim=8)
        k1, k2, k3 = jax.random.split(key, 3)

        fwd_params = factory.init(k1, dim, dim)
        bwd_params = factory.init(k2, dim, dim)

        x = jax.random.normal(k3, (4, dim))
        t_arr = jnp.ones(4) * 0.5
        sigma = 0.5

        # Forward drift: b_ref + σ² · correction
        correction = factory.forward(fwd_params, x, t_arr)
        fwd_drift = sigma ** 2 * correction
        assert fwd_drift.shape == (4, dim)

        # Backward drift: -b_ref + σ² · correction
        bwd_correction = factory.forward(bwd_params, x, t_arr)
        bwd_drift = sigma ** 2 * bwd_correction
        assert bwd_drift.shape == (4, dim)


# ═══════════════════════════════════════════════════════════════════════════
# User Subclass Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestUserSubclass:

    def test_fourier_mlp(self, key):
        """User subclasses NetworkFactory for custom architecture."""

        class FourierMLPFactory(NetworkFactory):
            def __init__(self, fourier_dim=32, scale=5.0):
                self.fourier_dim = fourier_dim
                self.scale = scale

            def init(self, key, input_dim, output_dim):
                k1, k2 = jax.random.split(key)
                return {
                    'B': jax.random.normal(k1, (input_dim, self.fourier_dim)) * self.scale,
                    'mlp': init_mlp_params(k2, [2 * self.fourier_dim + 16, 32, output_dim]),
                }

            def forward(self, params, x, t):
                proj = x @ params['B']
                fourier = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
                t_emb = sinusoidal_embedding(t, 16)
                h = jnp.concatenate([fourier, t_emb], axis=-1)
                return mlp_forward(params['mlp'], h, swish)

        factory = FourierMLPFactory(fourier_dim=32)

        # Sanity check
        sanity_check(factory, key, input_dim=3, output_dim=3)

        # Gradient check
        params = factory.init(key, 3, 3)
        x = jax.random.normal(key, (4, 3))
        t = jnp.ones(4) * 0.5

        def loss(p):
            return jnp.mean(factory.forward(p, x, t) ** 2)

        g = jax.grad(loss)(params)
        assert all(jnp.all(jnp.isfinite(gl)) for gl in jax.tree_util.tree_leaves(g))

    def test_abc_not_instantiable(self):
        with pytest.raises(TypeError):
            NetworkFactory()


# ═══════════════════════════════════════════════════════════════════════════
# Batch Size Robustness
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchSizes:

    @pytest.mark.parametrize("batch_size", [1, 2, 7, 32, 128])
    def test_mlp_batch_sizes(self, key, mlp_factory, batch_size):
        params = mlp_factory.init(key, 5, 5)
        x = jax.random.normal(key, (batch_size, 5))
        t = jnp.linspace(0.01, 0.99, batch_size)
        y = mlp_factory.forward(params, x, t)
        assert y.shape == (batch_size, 5)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_unet_batch_sizes(self, key, unet_factory_2d, batch_size):
        dim = 64
        params = unet_factory_2d.init(key, dim, dim)
        x = jax.random.normal(key, (batch_size, dim))
        t = jnp.linspace(0.01, 0.99, batch_size)
        y = unet_factory_2d.forward(params, x, t)
        assert y.shape == (batch_size, dim)


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Factory Interchangeability
# ═══════════════════════════════════════════════════════════════════════════

class TestInterchangeability:
    """Verify that different factories can be swapped into the same solver pattern."""

    def _run_solver_pattern(self, factory, dim, key):
        """The universal solver integration pattern."""
        k1, k2, k3 = jax.random.split(key, 3)

        # init
        params = factory.init(k1, dim, dim)

        # forward
        x = jax.random.normal(k2, (4, dim))
        t = jnp.linspace(0.1, 0.9, 4)
        y = factory.forward(params, x, t)
        assert y.shape == (4, dim)

        # grad
        def loss(p):
            return jnp.mean(factory.forward(p, x, t) ** 2)
        g = jax.grad(loss)(params)
        assert all(jnp.all(jnp.isfinite(gl)) for gl in jax.tree_util.tree_leaves(g))

        return True

    def test_mlp_swappable(self, key):
        self._run_solver_pattern(MLPFactory(hidden_dims=(16,), time_embed_dim=8), 8, key)

    def test_unet_swappable(self, key):
        self._run_solver_pattern(
            UNetFactory(spatial_shape=(8, 8, 1), channels=(4,), time_embed_dim=8),
            64, key
        )

    def test_transformer_swappable(self, key):
        self._run_solver_pattern(
            TransformerFactory(token_dim=2, num_tokens=4, num_heads=2,
                             num_layers=1, hidden_dim=8, time_embed_dim=8),
            8, key
        )
