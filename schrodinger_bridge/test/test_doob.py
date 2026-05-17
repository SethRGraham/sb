#!/usr/bin/env python3
"""Quick test for Doob h-transform solver."""

import jax
import jax.numpy as jnp

print("Testing Doob h-Transform Solver")
print("=" * 50)

from schrodinger_bridge import (
    SBProblem,
    BrownianMotion,
    GaussianDistribution,
    TwoMoonsDistribution,
    TimeGrid,
    DoobHTransformSolver,
    DoobConfig,
    print_device_info,
)

# Device info
print_device_info()
print()

KEY = jax.random.PRNGKey(42)

# Test 1: Gaussian-to-Gaussian (analytical method)
print("Test 1: Gaussian-to-Gaussian (Analytical)")
print("-" * 50)

problem_g2g = SBProblem(
    reference=BrownianMotion(sigma=0.5, dim=2),
    source=GaussianDistribution(mean=jnp.array([-2.0, 0.0]), cov=0.3, dim=2),
    target=GaussianDistribution(mean=jnp.array([2.0, 0.0]), cov=0.5, dim=2),
    time_grid=TimeGrid(num_steps=30),
    name="G2G-Analytical",
)

doob_analytical = DoobHTransformSolver(problem_g2g)
print(f"  Selected method: {doob_analytical._method}")
print(f"  Is neural: {doob_analytical.is_neural}")

k1, KEY = jax.random.split(KEY)
result = doob_analytical.train(k1)
print(f"  Converged: {result.converged}")

# Sample
k2, KEY = jax.random.split(KEY)
traj = doob_analytical.sample(k2, num_samples=100)
print(f"  Trajectory shape: {traj.paths.shape}")
print(f"  Start mean: {traj.paths[:, 0, :].mean(axis=0)}")
print(f"  End mean: {traj.paths[:, -1, :].mean(axis=0)}")
print("✓ Analytical Doob test passed!")

# Test 2: Gaussian-to-Moons (kernel method)
print("\nTest 2: Gaussian-to-TwoMoons (Kernel)")
print("-" * 50)

problem_g2m = SBProblem(
    reference=BrownianMotion(sigma=0.5, dim=2),
    source=GaussianDistribution(dim=2),
    target=TwoMoonsDistribution(noise=0.05),
    time_grid=TimeGrid(num_steps=30),
    name="G2M-Kernel",
)

config = DoobConfig(
    method='kernel',
    num_inducing_points=200,
    kernel_reg=1e-3,
)
doob_kernel = DoobHTransformSolver(problem_g2m, config=config)
print(f"  Selected method: {doob_kernel._method}")

k3, KEY = jax.random.split(KEY)
result = doob_kernel.train(k3)
print(f"  Converged: {result.converged}")

k4, KEY = jax.random.split(KEY)
traj = doob_kernel.sample(k4, num_samples=100)
print(f"  Trajectory shape: {traj.paths.shape}")
print(f"  End mean: {traj.paths[:, -1, :].mean(axis=0)}")
print("✓ Kernel Doob test passed!")

print("\n" + "=" * 50)
print("All Doob tests passed!")
