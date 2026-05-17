"""Diffusion tensor utilities shared by solvers and integrators."""

from __future__ import annotations

from typing import Tuple, Union

import jax
import jax.numpy as jnp

from .types import Array, Scalar


def apply_diffusion(
    sigma: Union[Scalar, Array],
    vector: Array,
    *,
    is_scalar_diffusion: bool = False,
    matrix_is_covariance: bool = False,
) -> Array:
    """Apply a diffusion coefficient to a batch of vectors.

    Supports scalar, diagonal, per-example diagonal, full matrix, and
    per-example full matrix diffusion coefficients. Square matrices are treated
    as volatility coefficients by default. Set ``matrix_is_covariance=True`` to
    interpret them as covariance matrices and apply their Cholesky factor.
    """
    sigma = jnp.asarray(sigma)
    vector = jnp.atleast_2d(vector)
    batch_size, dim = vector.shape

    if sigma.ndim == 0:
        return sigma * vector

    if sigma.ndim == 1:
        if is_scalar_diffusion and sigma.shape[0] == batch_size:
            return sigma[:, None] * vector
        if sigma.shape[0] == dim:
            return sigma[None, :] * vector
        if sigma.shape[0] == batch_size:
            return sigma[:, None] * vector
        if sigma.shape[0] == 1:
            return sigma.reshape(()) * vector

    if sigma.ndim == 2:
        if sigma.shape == (dim, dim):
            if matrix_is_covariance:
                sigma = jnp.linalg.cholesky(
                    sigma + 1e-8 * jnp.eye(dim, dtype=sigma.dtype)
                )
            return vector @ sigma.T
        if sigma.shape == (batch_size, dim):
            return sigma * vector
        if sigma.shape == (1, dim):
            return sigma * vector

    if sigma.ndim == 3 and sigma.shape[-2:] == (dim, dim):
        if matrix_is_covariance:
            eye = jnp.eye(dim, dtype=sigma.dtype)
            sigma = jax.vmap(lambda a: jnp.linalg.cholesky(a + 1e-8 * eye))(sigma)
        return jnp.einsum("bij,bj->bi", sigma, vector)

    raise ValueError(
        f"Unsupported diffusion shape {sigma.shape}; expected scalar, "
        "[dim], [batch], [batch, dim], [dim, dim], or [batch, dim, dim]."
    )


def diffusion_covariance_diag(
    sigma: Union[Scalar, Array],
    vector_shape: Tuple[int, int],
    *,
    is_scalar_diffusion: bool = False,
    matrix_is_covariance: bool = False,
) -> Array:
    """Return ``diag(a)`` with ``a = sigma sigma^T`` as ``[batch, dim]``."""
    sigma = jnp.asarray(sigma)
    batch_size, dim = vector_shape

    if sigma.ndim == 0:
        return jnp.full((batch_size, dim), sigma ** 2)

    if sigma.ndim == 1:
        if is_scalar_diffusion and sigma.shape[0] == batch_size:
            return jnp.broadcast_to((sigma ** 2)[:, None], (batch_size, dim))
        if sigma.shape[0] == dim:
            return jnp.broadcast_to((sigma ** 2)[None, :], (batch_size, dim))
        if sigma.shape[0] == batch_size:
            return jnp.broadcast_to((sigma ** 2)[:, None], (batch_size, dim))
        if sigma.shape[0] == 1:
            return jnp.full((batch_size, dim), sigma.reshape(()) ** 2)

    if sigma.ndim == 2:
        if sigma.shape == (dim, dim):
            cov = sigma if matrix_is_covariance else sigma @ sigma.T
            return jnp.broadcast_to(jnp.diag(cov)[None, :], (batch_size, dim))
        if sigma.shape == (batch_size, dim):
            return sigma ** 2
        if sigma.shape == (1, dim):
            return jnp.broadcast_to(sigma ** 2, (batch_size, dim))

    if sigma.ndim == 3 and sigma.shape[-2:] == (dim, dim):
        cov = sigma if matrix_is_covariance else sigma @ jnp.swapaxes(sigma, -1, -2)
        return jnp.diagonal(cov, axis1=-2, axis2=-1)

    raise ValueError(
        f"Unsupported diffusion shape {sigma.shape}; expected scalar, "
        "[dim], [batch], [batch, dim], [dim, dim], or [batch, dim, dim]."
    )


def apply_diffusion_covariance(
    sigma: Union[Scalar, Array],
    vector: Array,
    *,
    is_scalar_diffusion: bool = False,
    matrix_is_covariance: bool = False,
) -> Array:
    """Apply the diffusion covariance ``a = sigma sigma^T`` to vectors."""
    sigma = jnp.asarray(sigma)
    vector = jnp.atleast_2d(vector)
    batch_size, dim = vector.shape

    if sigma.ndim == 2 and sigma.shape == (dim, dim):
        cov = sigma if matrix_is_covariance else sigma @ sigma.T
        return vector @ cov.T

    if sigma.ndim == 2 and sigma.shape == (batch_size, dim):
        return (sigma ** 2) * vector

    if sigma.ndim <= 2:
        return diffusion_covariance_diag(
            sigma,
            vector.shape,
            is_scalar_diffusion=is_scalar_diffusion,
            matrix_is_covariance=matrix_is_covariance,
        ) * vector

    if sigma.ndim == 3 and sigma.shape[-2:] == (dim, dim):
        cov = sigma if matrix_is_covariance else sigma @ jnp.swapaxes(sigma, -1, -2)
        return jnp.einsum("bij,bj->bi", cov, vector)

    raise ValueError(
        f"Unsupported diffusion shape {sigma.shape}; expected scalar, "
        "[dim], [batch], [batch, dim], [dim, dim], or [batch, dim, dim]."
    )


def solve_diffusion_covariance(
    sigma: Union[Scalar, Array],
    vector: Array,
    *,
    is_scalar_diffusion: bool = False,
    matrix_is_covariance: bool = False,
) -> Array:
    """Solve ``a y = vector`` where ``a = sigma sigma^T``."""
    sigma = jnp.asarray(sigma)
    vector = jnp.atleast_2d(vector)
    batch_size, dim = vector.shape

    if sigma.ndim == 2 and sigma.shape == (dim, dim):
        cov = sigma if matrix_is_covariance else sigma @ sigma.T
        cov = cov + 1e-8 * jnp.eye(dim, dtype=cov.dtype)
        return jnp.linalg.solve(cov, vector.T).T

    if sigma.ndim == 2 and sigma.shape == (batch_size, dim):
        return vector / (sigma ** 2 + 1e-8)

    if sigma.ndim <= 2:
        diag = diffusion_covariance_diag(
            sigma,
            vector.shape,
            is_scalar_diffusion=is_scalar_diffusion,
            matrix_is_covariance=matrix_is_covariance,
        )
        return vector / (diag + 1e-8)

    if sigma.ndim == 3 and sigma.shape[-2:] == (dim, dim):
        cov = sigma if matrix_is_covariance else sigma @ jnp.swapaxes(sigma, -1, -2)
        eye = jnp.eye(dim, dtype=cov.dtype)
        return jax.vmap(lambda a, v: jnp.linalg.solve(a + 1e-8 * eye, v))(cov, vector)

    raise ValueError(
        f"Unsupported diffusion shape {sigma.shape}; expected scalar, "
        "[dim], [batch], [batch, dim], [dim, dim], or [batch, dim, dim]."
    )


def solve_diffusion_coefficient(
    sigma: Union[Scalar, Array],
    vector: Array,
    *,
    is_scalar_diffusion: bool = False,
    matrix_is_covariance: bool = False,
) -> Array:
    """Solve ``sigma y = vector`` for BEL/Malliavin noise coordinates."""
    sigma = jnp.asarray(sigma)
    vector = jnp.atleast_2d(vector)
    batch_size, dim = vector.shape

    if matrix_is_covariance and sigma.ndim >= 2 and sigma.shape[-2:] == (dim, dim):
        if sigma.ndim == 2:
            sigma = jnp.linalg.cholesky(
                sigma + 1e-8 * jnp.eye(dim, dtype=sigma.dtype)
            )
        else:
            eye = jnp.eye(dim, dtype=sigma.dtype)
            sigma = jax.vmap(lambda a: jnp.linalg.cholesky(a + 1e-8 * eye))(sigma)

    if sigma.ndim == 0:
        return vector / (sigma + 1e-8)

    if sigma.ndim == 1:
        if is_scalar_diffusion and sigma.shape[0] == batch_size:
            return vector / (sigma[:, None] + 1e-8)
        if sigma.shape[0] == dim:
            return vector / (sigma[None, :] + 1e-8)
        if sigma.shape[0] == batch_size:
            return vector / (sigma[:, None] + 1e-8)
        if sigma.shape[0] == 1:
            return vector / (sigma.reshape(()) + 1e-8)

    if sigma.ndim == 2:
        if sigma.shape == (dim, dim):
            return jnp.linalg.solve(
                sigma + 1e-8 * jnp.eye(dim, dtype=sigma.dtype),
                vector.T,
            ).T
        if sigma.shape == (batch_size, dim):
            return vector / (sigma + 1e-8)
        if sigma.shape == (1, dim):
            return vector / (sigma + 1e-8)

    if sigma.ndim == 3 and sigma.shape[-2:] == (dim, dim):
        eye = jnp.eye(dim, dtype=sigma.dtype)
        return jax.vmap(lambda s, v: jnp.linalg.solve(s + 1e-8 * eye, v))(sigma, vector)

    raise ValueError(
        f"Unsupported diffusion shape {sigma.shape}; expected scalar, "
        "[dim], [batch], [batch, dim], [dim, dim], or [batch, dim, dim]."
    )


def diffusion_quadratic_form(
    sigma: Union[Scalar, Array],
    vector: Array,
    *,
    is_scalar_diffusion: bool = False,
    matrix_is_covariance: bool = False,
) -> Array:
    """Return ``vector^T a^{-1} vector`` for each batch element."""
    solved = solve_diffusion_covariance(
        sigma,
        vector,
        is_scalar_diffusion=is_scalar_diffusion,
        matrix_is_covariance=matrix_is_covariance,
    )
    return jnp.sum(jnp.atleast_2d(vector) * solved, axis=-1)


__all__ = [
    "apply_diffusion",
    "apply_diffusion_covariance",
    "diffusion_covariance_diag",
    "diffusion_quadratic_form",
    "solve_diffusion_coefficient",
    "solve_diffusion_covariance",
]
