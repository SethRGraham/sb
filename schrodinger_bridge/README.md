# Schrödinger Bridge Library

A production-grade implementation of continuous-time Schrödinger Bridges in JAX with multiple solver methods and specialized support for quantitative finance applications.

## Features

### Multiple Solver Methods
- **Neural Network Based:**
  - `ScoreBasedSolver`: Denoising score matching (fastest to train)
  - `FBSDESolver`: Forward-Backward SDE / stochastic optimal control
  - `IMFSolver`: Iterative Markovian Fitting (simulation-free)
  - `IPFSolver`: Iterative Proportional Fitting with neural parameterization

- **Non-Neural Methods:**
  - `DoobHTransformSolver`: Analytical for Gaussians, kernel-based otherwise
  - `RKHSSolver`: Pure kernel methods (no neural networks)

### Specialized Extensions
- **Marginal Schrödinger Bridge**: Match marginals at multiple intermediate times
- **Martingale Schrödinger Bridge**: Enforce martingale constraints for risk-neutral pricing in finance

### Key Capabilities
- Continuous time API with flexible internal discretization
- Comprehensive diagnostics: mass conservation, marginal consistency, KL evolution
- Support for CPU, GPU, and TPU (via JAX)
- Visualization with GIF export
- Integration with OTT-JAX for optimal transport

## Installation

### Basic Installation
```bash
pip install schrodinger-bridge
```

### With Optional Dependencies
```bash
# With OTT-JAX for optimal transport
pip install schrodinger-bridge[ott]

# With development tools
pip install schrodinger-bridge[dev]

# With everything
pip install schrodinger-bridge[all]
```

### From Source
```bash
git clone https://github.com/yourusername/schrodinger-bridge.git
cd schrodinger-bridge
pip install -e .
```

## Quick Start

```python
import jax
from schrodinger_bridge import (
    SBProblem, BrownianMotion, GaussianDistribution, TwoMoonsDistribution,
    TimeGrid, ScoreBasedSolver, create_transport_gif
)

# Define problem
problem = SBProblem(
    reference=BrownianMotion(sigma=0.5, dim=2),
    source=GaussianDistribution(dim=2),
    target=TwoMoonsDistribution(),
    time_grid=TimeGrid(num_steps=50),
)

# Solve
solver = ScoreBasedSolver(problem)
result = solver.train(jax.random.PRNGKey(0))

# Sample and visualize
trajectories = solver.sample(jax.random.PRNGKey(1), num_samples=100)
create_transport_gif(trajectories, save_path="transport.gif")
```

## Quantitative Finance Example

```python
from schrodinger_bridge import (
    MartingaleSBSolver, ForwardCurve, extract_risk_neutral_density
)

# Extract risk-neutral densities from option prices
densities = [
    extract_risk_neutral_density(strikes_i, prices_i, spot, rate, T_i)
    for T_i in maturities
]

# Create martingale problem with forward curve
forward_curve = ForwardCurve(spot=100.0, rate=0.05)
problem = create_martingale_sb_problem(
    densities=densities,
    times=maturities,
    forward_curve=forward_curve,
)

# Solve with martingale constraint
solver = MartingaleSBSolver(problem)
result = solver.train(key)

# Generate risk-neutral paths
paths = solver.sample(key, num_paths=10000)
```

## Documentation

Full documentation is available at [https://schrodinger-bridge.readthedocs.io](https://schrodinger-bridge.readthedocs.io)

## Mathematical Background

The Schrödinger Bridge problem finds the stochastic process P* that minimizes:

```
P* = argmin_{P} KL(P || P_ref)
```

subject to marginal constraints P_0 = μ_0 and P_1 = μ_1.

For quantitative finance applications, the martingale extension additionally enforces:

```
E[X_{t_{i+1}} | X_{t_i}] = F(t_i, t_{i+1})
```

where F is the forward price, ensuring no-arbitrage conditions.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{schrodinger_bridge_2025,
  title = {Schrödinger Bridge Library: Production-Grade Implementation in JAX},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/schrodinger-bridge}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Acknowledgments

Built with Claude (Anthropic) for quantitative finance applications.
