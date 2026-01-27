# Schrödinger Bridge Tutorial Preview

This Marimo tutorial (`schrodinger_bridge_tutorial.py`) includes:

## Interactive Components

### Sliders
- **Diffusion σ** (0.1 - 1.0): Controls stochasticity of paths
- **Number of samples** (50 - 500): Particle count
- **Time steps** (10 - 100): Discretization resolution

### Mermaid Diagrams

1. **Architecture Diagram**: Shows Source → SB Process → Target flow
2. **Solver Comparison**: Neural vs Non-Neural solver tree
3. **Decision Flowchart**: Which solver to use when
4. **OT vs SB Comparison**: Key differences visualized

### Visualizations
- Transport evolution at t = 0, 0.33, 0.67, 1.0
- Individual trajectory plots with color coding
- Source (blue) and Target (red) markers

## Mathematical Content

### Core Equation (the main takeaway!)
```
b*(x,t) = b_ref(x,t) + σ² ∇log ψ(x,t)
```

### Topics Covered
1. What is a Schrödinger Bridge
2. The KL minimization formulation
3. Why Gaussians have closed-form solutions
4. Score-based learning objective
5. FBSDE optimal control formulation
6. Doob h-transform derivation
7. RKHS kernel expansion
8. Connection to Optimal Transport
9. The σ → 0 limit

### Code Examples
- Basic usage with ScoreBasedSolver
- Gaussian-to-Gaussian with DoobHTransformSolver
- Multi-marginal with MarginalSBSolver

## Running the Tutorial

```bash
# Install marimo
pip install marimo jax jaxlib numpy matplotlib

# Run the tutorial
marimo run schrodinger_bridge_tutorial.py

# Or edit interactively
marimo edit schrodinger_bridge_tutorial.py
```

The sliders update the visualizations in real-time!
