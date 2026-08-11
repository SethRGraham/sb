# Malliavin Adjoint Matching: novelty boundary

The intended contribution is not the isolated combination of Malliavin
calculus and Adjoint Matching. Each ingredient, and several close combinations,
already exists. The table below is a research-positioning checklist, not a
completed literature review: exact citations, theorem scopes, contemporaneous
papers, and priority dates must be verified before making a novelty claim.

| Nearest work | Capability already plausibly covered | Question that remains for MAM |
|---|---|---|
| Diffusion Schrödinger Bridge Matching | Alternating conditional bridge/Markov fitting between endpoint marginals | Can a value-only conditional-SOC inner solver be inserted while retaining bridge feasibility and competitive compute? |
| Generalized Schrödinger Bridge Matching | Running state costs, conditional SOC, and value-based path-importance resampling | Can direct, on-policy BEL costate regression outperform path reweighting at equal oracle-query and wall-clock budgets? |
| Conditioning Diffusions Using Malliavin Calculus | Neural BEL/Malliavin regression for terminal conditioning under a fixed reference process | Does learning the on-policy running-cost costate enable a distinct control update, rather than merely another conditioning estimator? |
| Adjoint Schrödinger Bridge Sampler / Adjoint Matching | Adjoint-based sampling or matching with differentiable energies/potentials | Can stopped value-only labels replace unavailable potential derivatives while recovering the same smooth-problem update? |
| Twisted Schrödinger Bridge Matching | Bridge matching with continuous or discrete path potentials | Can nonsmooth Markov potentials be handled without reward-smoothing bias, while controlling BEL, discretization, and regression error? |
| Value critic followed by autodiff | Learns a differentiable value approximation from returns, then differentiates the critic | Do direct BEL labels improve held-out costate error, reward-query efficiency, or update stability? |

The defensible **research hypothesis**, not a current result, is:

> For fixed-policy conditional stochastic-control problems with nonsmooth
> value-only Markov costs, amortized on-policy BEL regression can recover the
> state costate without differentiating the cost values and can support useful
> Hamiltonian control updates without exponential path-importance weights.

This wording does not claim that BEL labels have low variance. They can have
severe tails, especially near terminal time, and may be less practical than a
critic or a path-integral baseline.

## Evidence boundary

The current implementation addresses only discrete label construction,
endpoint-pinned conditional sampling, and Gate-A analytic regression. Even a
scientific `PASS_MAM_ANALYTIC_FOUNDATION` would establish only that the
hard-threshold Brownian costate can be recovered under the declared finite-step
protocol. It would not establish Hamiltonian policy improvement, equivalence to
Adjoint Matching, superiority to a baseline, or a valid global bridge.

The hypothesis is weakened or falsified as a standalone method claim if any of
the following holds under predeclared, equal budgets:

- the analytic BEL mean or conditional hard-threshold regression fails;
- MAM does not recover classical pathwise Adjoint Matching on smooth LQG;
- a value critic followed by autodiff matches costate accuracy and update
  quality using no more oracle queries or wall-clock time;
- the GSBM path-integral baseline matches or exceeds objective improvement
  without prohibitive weight degeneracy;
- BEL label tails make stable learning impractical without target-changing
  clipping; or
- the outer reciprocal/Markov projection fails the endpoint and objective
  diagnostics.

The first implementation is therefore deliberately narrower than the paper
hypothesis. No finished global-bridge claim is permitted until the outer loop
exists, both endpoint marginals pass predeclared diagnostics, and the learned
control improves the declared generalized-bridge objective against strong
baselines.
