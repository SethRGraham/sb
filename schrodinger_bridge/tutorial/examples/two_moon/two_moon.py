"""
Test file for Doob h-Transform Solver

This tests BOTH the analytical method (Gaussian→Gaussian) 
and the kernel method (Gaussian→TwoMoons).
"""

import jax
import jax.numpy as jnp
from schrodinger_bridge import (
    SBProblem,
    BrownianMotion,
    GaussianDistribution,
    TwoMoonsDistribution,
    TimeGrid,
    create_transport_gif,
    mmd_squared,
)
from schrodinger_bridge.solvers import DoobHTransformSolver, DoobConfig

# Local configuration (haven't done a pip install -e yet)
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # -> schrodinger_bridge_library




def test_gaussian_to_gaussian():
    """
    Gaussian → Gaussian with ANALYTICAL method (should be perfect).
    
    This is the ideal case for Doob - closed-form solution exists.
    """
    print("="*70)
    print("TEST 1: GAUSSIAN → GAUSSIAN (ANALYTICAL)")
    print("="*70 + "\n")
    
    source = GaussianDistribution(
        mean=jnp.array([-2.0, 0.0]),
        cov=jnp.array([[0.3, 0.0], [0.0, 0.3]]),
        dim=2
    )
    
    target = GaussianDistribution(
        mean=jnp.array([2.0, 0.0]),
        cov=jnp.array([[0.3, 0.0], [0.0, 0.3]]),
        dim=2
    )
    
    problem = SBProblem(
        reference=BrownianMotion(sigma=0.8, dim=2),
        source=source,
        target=target,
        time_grid=TimeGrid(t0=0.0, t1=1.0, num_steps=100),
        name="Gaussian→Gaussian",
    )
    
    print(problem.summary())
    
    key = jax.random.PRNGKey(42)
    
    # Use 'auto' or 'analytical' - both should pick analytical for Gaussians
    config = DoobConfig(method='auto')
    solver = DoobHTransformSolver(problem, config=config)
    
    print(f"\nSelected method: {solver._method}")
    print("Computing solution...")
    
    start = time.time()
    result = solver.train(key)
    elapsed = time.time() - start
    print(f"✓ Done in {elapsed:.4f} seconds")
    
    # Sample trajectories
    print("\nSampling 2000 trajectories...")
    trajectories = solver.sample(key, num_samples=2000)
    
    # Check endpoint quality
    endpoints = trajectories.paths[:, -1, :]
    true_target = target.sample(jax.random.PRNGKey(999), 2000)
    mmd = mmd_squared(endpoints, true_target)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print('='*70)
    print(f"Target MMD²: {mmd:.6f}")
    
    # For analytical Gaussian, should be very good
    if mmd < 0.05:
        print("  ✓ EXCELLENT!")
    elif mmd < 0.15:
        print("  ✓ Good")
    else:
        print("  ⚠ Check implementation")
    
    endpoint_mean = jnp.mean(endpoints, axis=0)
    target_mean = jnp.mean(true_target, axis=0)
    print(f"\nEndpoint mean: {endpoint_mean}")
    print(f"Target mean:   {target_mean}")
    print(f"Mean error:    {jnp.linalg.norm(endpoint_mean - target_mean):.6f}")
    
    # Create visualization
    create_transport_gif(
        trajectories,
        source_samples=source.sample(jax.random.PRNGKey(1), 300),
        target_samples=target.sample(jax.random.PRNGKey(2), 300),
        save_path="doob_gaussian_analytical.gif",
    )
    print("\n✓ Saved: doob_gaussian_analytical.gif")
    
    return mmd


def test_gaussian_to_twomoons():
    """
    Gaussian → TwoMoons with KERNEL method.
    
    Non-Gaussian target requires kernel approximation.
    """
    print("\n\n" + "="*70)
    print("TEST 2: GAUSSIAN → TWOMOONS (KERNEL)")
    print("="*70 + "\n")
    
    source = GaussianDistribution(dim=2)
    target = TwoMoonsDistribution(noise=0.05, offset=0.5)
    
    problem = SBProblem(
        reference=BrownianMotion(sigma=1.0, dim=2),
        source=source,
        target=target,
        time_grid=TimeGrid(t0=0.0, t1=1.0, num_steps=100),
        name="Gaussian→TwoMoons",
    )
    
    print(problem.summary())
    
    key = jax.random.PRNGKey(123)
    
    # Use 'auto' or 'kernel' - both should pick kernel for non-Gaussian
    config = DoobConfig(
        method='auto',  # Will auto-select 'kernel' since target is not Gaussian
        num_inducing_points=1000,
    )
    
    solver = DoobHTransformSolver(problem, config=config)
    
    print(f"\nSelected method: {solver._method}")
    print("Computing solution...")
    
    start = time.time()
    result = solver.train(key)
    elapsed = time.time() - start
    print(f"✓ Done in {elapsed:.4f} seconds")
    
    # Sample trajectories
    print("\nSampling 2000 trajectories...")
    trajectories = solver.sample(key, num_samples=2000)
    
    # Check endpoint quality
    endpoints = trajectories.paths[:, -1, :]
    true_target = target.sample(jax.random.PRNGKey(999), 2000)
    mmd = mmd_squared(endpoints, true_target)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print('='*70)
    print(f"Target MMD²: {mmd:.6f}")
    
    # Kernel method won't be as good as analytical
    if mmd < 0.10:
        print("  ✓ EXCELLENT for kernel method!")
    elif mmd < 0.20:
        print("  ✓ Good for kernel method")
    else:
        print("  ⚠ Acceptable (kernel is approximate)")
    
    # Create visualization
    create_transport_gif(
        trajectories,
        source_samples=source.sample(jax.random.PRNGKey(1), 300),
        target_samples=target.sample(jax.random.PRNGKey(2), 300),
        save_path="doob_twomoons_kernel.gif",
    )
    print("\n✓ Saved: doob_twomoons_kernel.gif")
    
    return mmd


def main():
    print("\n" + "="*70)
    print("DOOB H-TRANSFORM SOLVER TESTS")
    print("="*70 + "\n")
    
    # Test both methods
    mmd_analytical = test_gaussian_to_gaussian()
    mmd_kernel = test_gaussian_to_twomoons()
    
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Gaussian → Gaussian (Analytical):
  MMD² = {mmd_analytical:.6f}
  Expected: < 0.05 (closed-form solution)
  
Gaussian → TwoMoons (Kernel):
  MMD² = {mmd_kernel:.6f}
  Expected: < 0.20 (approximate method)

Key insight:
  - Analytical works perfectly for Gaussian-to-Gaussian
  - Kernel is approximate but should capture the structure
  - If analytical gives bad results, check the drift formula
  - If kernel gives bad results, check time-dependence
""")


if __name__ == "__main__":
    main()