# Malliavin Adjoint Matching: theory worklist

This file separates algebra implemented in code, standard results that still
need citations, empirical checks, and open paper claims. An implementation test
is not a proof, and a standard theorem is not available for use in the paper
until its assumptions are matched to this solver.

## Status labels

- **Proved in repository:** a complete written argument is present and audited.
- **Cited theorem:** an exact source and theorem are recorded, and its
  assumptions are mapped to the implementation.
- **Derived, proof pending:** the formula has a local derivation and tests, but
  no complete paper-ready proof.
- **Empirically tested:** a named experiment/test checks a finite instance.
- **Conjectured/open:** neither proof nor sufficient evidence exists.

At this stage, no end-to-end MAM policy-improvement or global-bridge theorem is
proved in the repository.

## Claim ledger

| Target | Current status | Acceptance needed |
|---|---|---|
| Generic additive-EM terminal BEL identity | Derived, proof pending; empirically tested | Formal discrete Gaussian integration-by-parts proof, including integrability and arrival-index convention |
| Generic additive-EM Markov running-cost identity | Derived, proof pending; empirically tested | Duhamel/sum proof with right-endpoint quadrature and componentwise reference |
| Endpoint-pinned discrete running-cost identity | Derived, proof pending; empirically tested | Formal proof for the declared \(\rho_n,\Gamma_n\) chain and exclusion of the deterministic final transition |
| Endpoint-only terminal contribution is zero at fixed \(y\) | Algebraic fact for the declared pinned chain | Distinguish a fixed-target potential (globally constant) from an endpoint-pair cost (outer-coupling term) and from an unpinned terminal reward |
| MSE population minimizer is the conditional mean label | Standard \(L^2\) projection result; citation pending | Cite the projection theorem and assume a square-integrable label |
| Conditional mean label equals a state costate | Derived for the discrete kernels, proof pending | Combine the discrete BEL identity with differentiation/conditioning assumptions |
| Neural regression consistently learns the costate | Conjectured/open for the configured model and optimizer | Specify function class, approximation error, optimization error, sampling scheme, and consistency regime |
| Smooth-reward equivalence to a pathwise costate | Partially empirically tested in Brownian calibration | Prove equality under smoothness and pass smooth LQG Gate B |
| Pointwise quadratic Hamiltonian minimizer \(u=-\Sigma^\top p\) | Elementary algebra under the declared coordinates | Record drift/control parameterization and positive quadratic energy |
| Discrete pinned target \(-\sqrt{\rho_n}\Sigma^\top\mathbb E[p_{n+1}\mid X_n,y]\) | Derived, proof pending; one-sample API tested | Bellman one-step derivation plus conditional policy regression test |
| MAM update is a valid policy-improvement step | Conjectured/open | Monotonic-improvement or small-step theorem and controlled Brownian/LQG evidence |
| Equivalence to the Adjoint Matching first variation | Conjectured/open | Match objectives, signs, information sets, and first variations explicitly |
| Reciprocal/Markov projection preserves endpoint marginals | Unimplemented/open | Paper-faithful outer loop plus source/target diagnostics and approximation bounds or qualifications |
| MAM improves over critic-autodiff/path-integral baselines | Empirical hypothesis only | Equal query, tuning, seed, and wall-clock budgets with uncertainty |
| State-dependent or rank-deficient diffusion | Out of V1 scope | New BEL weight, domain/range conditions, stable implementation, and analytic tests |
| Control-dependent diffusion | Out of V1 scope | Second-order value/Hessian treatment and a separate theorem |
| Path-dependent hard events | Out of V1 scope | Markov augmentation or a path-functional BEL theorem |

## Theorem target A: generic discrete BEL identity

For a uniform-grid EM chain

\[
X_{j+1}=X_j+b_\pi(X_j,t_j)\Delta t+\Sigma\Delta W_j,
\]

the target proposition is

\[
\nabla_x\mathbb E[\varphi(X_m)\mid X_k=x]
=
\mathbb E[\varphi(X_m)H^{\mathrm{EM}}_{k,m}\mid X_k=x],
\]

with

\[
H^{\mathrm{EM}}_{k,m}
=\frac{1}{(m-k)\Delta t}
\sum_{j=k}^{m-1}
(\Sigma^{-1}J_{j+1,k})^\top\Delta W_j.
\]

The proof must explicitly establish:

1. the base Gaussian increments are parameter-independent;
2. \(J_{j+1,k}\) is adapted with respect to the increment at step \(j\);
3. each summand gives the same state derivative after Gaussian integration by
   parts;
4. averaging over \(j\) preserves the expectation;
5. the arrival-flow index and transpose solve are correct; and
6. differentiation and expectation may be interchanged for the stated class
   of possibly nonsmooth \(\varphi\).

The reference tests must include a nonlinear or linear-drift case that exposes
the arrival-index error and a nonsymmetric diffusion matrix that exposes the
transpose error. Brownian scalar calibration alone is insufficient.

## Theorem target B: running costs

For the discrete cost-to-go

\[
V_k(x)=\mathbb E\!\left[
g(X_N)+\sum_{m=k+1}^{N}\ell_m(X_m)\Delta t
\;\middle|\;X_k=x
\right],
\]

the target identity is

\[
\nabla_xV_k(x)
=\mathbb E\!\left[
g(X_N)H_{k,N}
+\sum_{m=k+1}^{N}\ell_m(X_m)H_{k,m}\Delta t
\;\middle|\;X_k=x
\right].
\]

The proof must distinguish future right-endpoint values from a cost evaluated
at the anchor. A differentiable anchor cost requires its direct derivative; a
nonsmooth value-only anchor cost is not represented by future-noise weighting.
The current pinned implementation uses this direct term only for known
quadratic control energy.

## Theorem target C: endpoint-pinned chain

Prove the corresponding discrete identity for

\[
X_{n+1}=\rho_nX_n+(1-\rho_n)y
+\Gamma_n(\sqrt{\Delta t}\,u_n+\xi_n),
\qquad n<N-1,
\]

with \(X_N=y\), full-rank \(\Gamma_n\), and the implemented arrival tangents.
The theorem must make clear that it concerns this endpoint-conditioned Markov
chain with observed context \(y\). It must not imply that the mixture over
\(y\) is already an unconditional Markov bridge.

The proof must also separate:

- the zero interior-state derivative of an endpoint-only terminal value;
- the globally constant expectation of a pure target potential when the target
  marginal is fixed, versus an endpoint-pair cost that can alter the coupling;
- running values on stochastic states \(X_m\), \(m\leq N-1\);
- the direct derivative of current quadratic control energy; and
- the blow-up/degeneracy as the remaining stochastic horizon vanishes.

## Theorem target D: regression and Hamiltonian update

The squared-loss statement is only a population projection:

\[
f^*(t,x,y)
=\mathbb E[\widehat p_k\mid t_k=t,X_k=x,Y=y].
\]

A paper claim about the trained network must separately bound or measure label
variance, approximation error, finite-sample error, and optimization error.
No clipping or robust loss may replace theorem-facing MSE without explicitly
changing the target.

For the continuous noise-coordinate model with instantaneous cost
\(\tfrac12\lVert u\rVert^2\), verify the pointwise calculation

\[
\arg\min_u\left\{p^\top\Sigma u+\tfrac12\lVert u\rVert^2\right\}
=-\Sigma^\top p.
\]

This algebra does not prove that replacing the current policy by the learned
target improves the finite-horizon objective. That requires a policy-improvement
argument, a conservative-update error condition, and an analysis of costate
estimation error.

## Outer-loop theorem target

The future generalized-bridge result must specify:

1. the endpoint coupling supplied to each conditional inner problem;
2. the reciprocal projection and the Markov projection;
3. whether the learned controller observes the endpoint context;
4. the objective decreased by each exact or approximate step;
5. approximation error introduced by neural regression;
6. how source and target marginal violations are measured; and
7. conditions under which a finite learned projection preserves or approaches
   the desired marginals.

Endpoint MMD, Sinkhorn divergence, and sample moments are diagnostics, not a
proof of equality. A pass must report uncertainty and sensitivity to sample
size and kernel/regularization choices.

## Gate-A evidence requirements

A scientific `PASS_MAM_ANALYTIC_FOUNDATION` requires all of the following:

- smooth Brownian BEL, pathwise, and common-random finite-difference checks;
- hard-threshold BEL mean against the analytic density derivative;
- sigmoid-smoothing bias/variance results at predeclared temperatures;
- separate terminal, running, and sum checks for the running-cost identity;
- a tuning protocol disjoint from the final analytic evaluation grid;
- held-out relative \(L^2\), cosine, sign, and finite-value gates;
- no theorem-facing clipping or discarded nonfinite samples;
- declared precision, seed derivations, source/config/artifact hashes, device,
  and exact command; and
- a scientific rather than reduced smoke profile.

A reduced profile may emit
`PASS_MAM_ANALYTIC_FOUNDATION_SMOKE_NOT_EVIDENCE`. That status checks plumbing
only. Gate A still says nothing about policy improvement or bridge feasibility.

## Required falsification tests

- use \(J_{j,k}\) instead of \(J_{j+1,k}\) and show the OU reference fails;
- use `solve(Sigma, dW)` instead of the transpose solve and show a nonsymmetric
  matrix reference fails;
- differentiate or omit stopping on a hard indicator and record the invalid
  zero/undefined pathwise label;
- remove the direct anchor control-energy derivative and show the discrete
  reference becomes biased;
- include anchors arbitrarily close to the pinned terminal step and quantify
  tail/conditioning failure;
- compare learned predictions with the analytic conditional mean, not merely
  with noisy training-label MSE;
- compare against critic-autodiff and path-integral baselines at equal budgets;
  and
- demonstrate that a conditional solver can pass while an attempted outer
  bridge still fails endpoint diagnostics.

## Paper-claim stop line

Until the open items above are closed, the strongest defensible statement is:

> The implementation constructs and regresses discrete value-only BEL costate
> labels for restricted fixed-policy and endpoint-pinned Brownian problems.

It must not yet be upgraded to claims of consistent neural learning in general,
Hamiltonian policy improvement, equivalence to Adjoint Matching, superiority to
existing solvers, path-constrained capability, or a completed generalized
Schrödinger bridge.
