#!/usr/bin/env python3
"""
Quick Test Suite for Schrödinger Bridge Library
================================================

Tests all solvers with minimal iterations to verify they work.
"""

import sys
sys.path.insert(0, '/home/claude')

import time
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp

NUM_ITERS = 100
BATCH_SIZE = 128

print("=" * 70)
print("SCHRÖDINGER BRIDGE LIBRARY - COMPREHENSIVE TEST")
print("=" * 70)

# Imports
try:
    from schrodinger_bridge import (
        SBProblem, TimeGrid,
        BrownianMotion, GaussianDistribution, TwoMoonsDistribution,
        ScoreBasedSolver, ScoreBasedConfig,
        FBSDESolver, FBSDEConfig,
        DoobHTransformSolver, DoobConfig,
        RKHSSolver, RKHSConfig,
        IMFSolver, IMFConfig,
        print_device_info,
        quick_marginal_check,
        TrainingConfig,
        MarginalSBProblem, MarginalConstraint, MarginalSBSolver, MarginalSBConfig,
        is_ott_available, compute_ot_coupling, OTConfig,
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print_device_info()
print()

KEY = jax.random.PRNGKey(42)

# Problems
problem = SBProblem(
    reference=BrownianMotion(sigma=0.5, dim=2),
    source=GaussianDistribution(dim=2),
    target=TwoMoonsDistribution(noise=0.05),
    time_grid=TimeGrid(num_steps=30),
)

problem_g2g = SBProblem(
    reference=BrownianMotion(sigma=0.5, dim=2),
    source=GaussianDistribution(mean=jnp.array([-2.0, 0.0]), cov=0.3, dim=2),
    target=GaussianDistribution(mean=jnp.array([2.0, 0.0]), cov=0.5, dim=2),
    time_grid=TimeGrid(num_steps=30),
)

key, k1, k2 = jax.random.split(KEY, 3)
target_samples = problem.sample_target(k1, 500)

results = {}

def test_solver(name, solver_cls, prob, key, **kwargs):
    print(f"\n{'─' * 50}")
    print(f"Testing: {name}")
    result = {'name': name, 'success': False}
    try:
        start = time.time()
        solver = solver_cls(prob, **kwargs)
        print(f"  Repr: {solver.representation_type.name}, Neural: {solver.is_neural}")
        
        k1, k2 = jax.random.split(key)
        train_config = TrainingConfig(num_iterations=NUM_ITERS, batch_size=BATCH_SIZE)
        train_result = solver.train(k1, train_config)
        
        traj = solver.sample(k2, num_samples=100)
        mmd = quick_marginal_check(traj.paths[:, -1, :], target_samples)
        
        result['loss'] = float(train_result.final_loss)
        result['mmd'] = float(mmd)
        result['time'] = time.time() - start
        result['success'] = True
        print(f"  ✓ Loss={result['loss']:.4f}, MMD={result['mmd']:.4f}, Time={result['time']:.1f}s")
    except Exception as e:
        result['error'] = str(e)
        print(f"  ✗ Error: {e}")
    return result

# Test solvers
key, subkey = jax.random.split(key)
results['score'] = test_solver("Score-Based", ScoreBasedSolver, problem, subkey,
    config=ScoreBasedConfig(hidden_dims=(64, 64), learning_rate=1e-3))

key, subkey = jax.random.split(key)
results['fbsde'] = test_solver("FBSDE", FBSDESolver, problem, subkey,
    fbsde_config=FBSDEConfig(hidden_dims=(64, 64), learning_rate=1e-3, method='soc'))

key, subkey = jax.random.split(key)
results['doob_analytical'] = test_solver("Doob-Analytical", DoobHTransformSolver, problem_g2g, subkey,
    config=DoobConfig(method='analytical'))

key, subkey = jax.random.split(key)
results['doob_kernel'] = test_solver("Doob-Kernel", DoobHTransformSolver, problem, subkey,
    config=DoobConfig(method='kernel', num_inducing_points=150))

key, subkey = jax.random.split(key)
results['rkhs'] = test_solver("RKHS", RKHSSolver, problem, subkey,
    rkhs_config=RKHSConfig(num_inducing=150, num_time_points=10))

key, subkey = jax.random.split(key)
results['imf'] = test_solver("IMF", IMFSolver, problem, subkey,
    imf_config=IMFConfig(hidden_dims=(64, 64), num_imf_iterations=2, steps_per_iteration=50))

# Test Marginal SB
print(f"\n{'─' * 50}")
print("Testing: Marginal-SB")
try:
    marginal_problem = MarginalSBProblem(
        reference=BrownianMotion(sigma=0.5, dim=2),
        marginals=[
            MarginalConstraint(0.0, GaussianDistribution(mean=jnp.array([-2., 0.]), cov=0.3, dim=2)),
            MarginalConstraint(0.5, GaussianDistribution(mean=jnp.array([0., 1.]), cov=0.4, dim=2)),
            MarginalConstraint(1.0, GaussianDistribution(mean=jnp.array([2., 0.]), cov=0.5, dim=2)),
        ],
        time_grid=TimeGrid(num_steps=50),
    )
    
    config = MarginalSBConfig(segment_solver_type='doob', num_iterations=50, verbose=0)
    key, subkey = jax.random.split(key)
    solver = MarginalSBSolver(marginal_problem, config)
    start = time.time()
    solver.train(subkey)
    
    key, subkey = jax.random.split(key)
    traj = solver.sample(subkey, 50)
    
    results['marginal_sb'] = {'name': 'Marginal-SB', 'success': True, 'time': time.time() - start}
    print(f"  ✓ Shape={traj.paths.shape}, Time={results['marginal_sb']['time']:.1f}s")
except Exception as e:
    results['marginal_sb'] = {'name': 'Marginal-SB', 'success': False, 'error': str(e)}
    print(f"  ✗ Error: {e}")

# Test OTT
print(f"\n{'─' * 50}")
print("Testing: OTT-Integration")
try:
    key, k1, k2 = jax.random.split(key, 3)
    x, y = jax.random.normal(k1, (30, 2)), jax.random.normal(k2, (30, 2)) + 2.0
    P, info = compute_ot_coupling(x, y, OTConfig(epsilon=0.5))
    results['ott'] = {'name': 'OTT', 'success': True}
    print(f"  ✓ OTT available={is_ott_available()}, Cost={info['cost']:.4f}")
except Exception as e:
    results['ott'] = {'name': 'OTT', 'success': False, 'error': str(e)}
    print(f"  ✗ Error: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
passed = sum(1 for r in results.values() if r.get('success'))
failed = len(results) - passed
print(f"\n{passed}/{len(results)} tests passed")

for name, r in results.items():
    status = "✓" if r.get('success') else "✗"
    print(f"  {status} {r.get('name', name)}")

if failed == 0:
    print("\nALL TESTS PASSED! ✓")
else:
    print(f"\n{failed} TESTS FAILED")
