# Malliavin Adjoint Matching: implementation contract

This document fixes the mathematical and evidentiary scope of the first JAX
implementation. Three objects must remain distinct:

1. a generic Euler--Maruyama (EM) reference kernel for discrete BEL labels;
2. an endpoint-pinned conditional-path inner solver and costate regressor; and
3. a future reciprocal/Markov-projection outer loop that produces an
   unconditional, endpoint-constrained generalized bridge.

Only the first two exist. In particular, a pinned conditional path for a
caller-supplied endpoint is not by itself a global Schrödinger bridge solver.

## Convention and learned object

Version 1 uses **cost minimization**. For the declared endpoint-pinned Markov
chain, write

\[
C^\pi_k(x;y)
=
\Delta t\,e^\pi_k(x,y)
+
\mathbb E^{\pi,y}\!\left[
  \sum_{m=k+1}^{N-1}
  \bigl(\bar\ell_m(X_m,y)+e^\pi_m(X_m,y)\bigr)\,\Delta t
  \;\middle|\;X_k=x
\right],
\qquad
p^\pi_k(x;y)=\nabla_x C^\pi_k(x;y).
\]

Here \(y\) is fixed context and \(\mathbb E^{\pi,y}\) refers to the discrete
pinned dynamics specified below; \(\bar\ell\) is the value-only potential and
\(e_m^\pi=\tfrac12\lVert u_m\rVert^2\) on steps having a stochastic control.
The first term records the departure-time control energy whose derivative is
handled directly. This notation does not assert that the controlled pinned
chain is the regular conditional law of some already-solved global bridge.

The running cost may be nonsmooth and bounded or suitably square-integrable.
The cost oracle is evaluated only by value. The network takes
`[time, state, endpoint]` and is trained by ordinary squared loss against
stopped BEL labels. If the label is square-integrable, the unrestricted
population MSE minimizer is its conditional mean. Identifying that conditional
mean with \(p^\pi_k\) additionally requires the discrete integration-by-parts
identity below. A finite trained network is an approximation to this population
object, not an exact costate merely because its training loss is small.

For noise-coordinate control with quadratic energy, the continuous-time
Hamiltonian target under this sign convention is

\[
u_{\mathrm{target}}(t,x,y)=-\Sigma^\top p^\pi(t,x;y).
\]

This is a proposed inner update. Neither policy improvement nor equivalence to
the precise Adjoint Matching first variation has yet been established. Applying
the proposal directly to an unconditional process is not guaranteed to preserve
either endpoint marginal.

## Generic discrete EM reference identity

For

\[
X_{j+1}=X_j+b_\pi(X_j,t_j)\Delta t+\Sigma\Delta W_j,
\qquad
A_j=I+\Delta t\,\partial_x b_\pi(X_j,t_j),
\]

let \(J_{j+1,k}=A_jJ_{j,k}\), with \(J_{k,k}=I\). Version 1 assumes
square, constant, invertible \(\Sigma\). The implemented weight is

\[
H_{k,m}^{\mathrm{EM}}
=
\frac{1}{(m-k)\Delta t}
\sum_{j=k}^{m-1}
(\Sigma^{-1}J_{j+1,k})^\top\Delta W_j.
\]

Subject to the differentiability, integrability, and Gaussian
integration-by-parts conditions listed below, this is an exact identity for the
**discrete EM semigroup**:

\[
\nabla_x\mathbb E[\varphi(X_m)\mid X_k=x]
=
\mathbb E[\varphi(X_m)H_{k,m}^{\mathrm{EM}}\mid X_k=x].
\]

It is not an exact finite-step identity for the original continuous SDE; that
requires a separate discretization-limit argument. The arrival index \(j+1\)
is essential. Replacing it with \(J_{j,k}\) is an off-by-one error that Brownian
motion cannot expose. For nonsymmetric \(\Sigma\), the equivalent numerical
operation is `solve(Sigma.T, dW)`, not `solve(Sigma, dW)`.

The generic kernel uses right-endpoint quadrature:

\[
\widehat p_k
=g(X_N)H_{k,N}
+\sum_{m=k+1}^{N}\ell_m(X_m)H_{k,m}\Delta t.
\]

Terminal and running values are stopped before label assembly; no derivative
of either value-only oracle is taken. A cost evaluated directly at the anchor
has no future innovation through which this BEL formula can represent its
state derivative. The pinned solver handles only the known differentiable
anchor control-energy term by adding its explicit, already
quadrature-weighted derivative. It does not differentiate the value-only
potential oracle.

## Endpoint-pinned discrete chain

For a Brownian reference with fixed endpoints \(x_0,y\), and \(n<N-1\),
define

\[
\rho_n=\frac{T-t_{n+1}}{T-t_n},
\qquad
\Gamma_n=\sqrt{\Delta t\,\rho_n}\,\Sigma,
\]

\[
X_{n+1}
=\rho_nX_n+(1-\rho_n)y
+\Gamma_n\bigl(\sqrt{\Delta t}\,u_n+\xi_n\bigr),
\qquad \xi_n\sim\mathcal N(0,I),
\]

and set \(X_N=y\) exactly. The last transition is deterministic and is never
inverted. With \(J_{j+1,k}\) denoting the arrival-flow tangent of this declared
chain, the weight for \(m\leq N-1\) is

\[
H_{k,m}^{\mathrm{pin}}
=\frac{1}{m-k}
\sum_{j=k}^{m-1}
(\Gamma_j^{-1}J_{j+1,k})^\top\xi_j.
\]

Under the same discrete Gaussian integration-by-parts conditions, the
conditional mean of the resulting right-endpoint running-cost label equals the
state derivative of the cost-to-go for this pinned chain. The theorem-facing
label is unregularized and unclipped.

This construction pins every caller-supplied endpoint pair. It does not turn
an endpoint-conditioned controller into an unconditional Markov bridge. A
future outer loop must specify the current endpoint coupling, fit the
reciprocal/Markov projection, and test both endpoint marginals. MMD or Sinkhorn
diagnostics provide empirical discrepancy measurements; they do not prove
exact marginal equality.

### Terminal-cost scope

An endpoint-only terminal cost \(g(X_N)\) is constant with respect to the
interior state because \(X_N=y\) is fixed. Its pinned conditional costate
contribution is therefore zero. If the target marginal is itself fixed, then
\(\mathbb E[g(X_N)]\) is also constant across every feasible global bridge and
does not affect the optimizer. By contrast, a pair cost \(c(X_0,X_N)\) can
change the endpoint coupling even though it remains constant within each fixed
endpoint-pair inner problem; such a pair cost belongs in a future outer loop.
The V1 pinned solver rejects every configured terminal cost rather than
silently choosing among these interpretations.

This restriction does **not** say that BEL methods cannot represent terminal
rewards in general. The standalone generic EM kernel retains terminal-value
support for unpinned fixed-policy problems. A terminal quantity not determined
by the fixed endpoint, or a path-dependent event, is outside the pinned V1 API
and requires either an explicit Markov state augmentation or a separate
theorem.

### Discrete control proposal

For the pinned chain, differentiating one transition and its quadratic
noise-control energy gives

\[
u_{\mathrm{target},n}
=-\sqrt{\rho_n}\,\Sigma^\top
\mathbb E[p_{n+1}(X_{n+1};y)\mid X_n,y].
\]

The public API either returns the continuous target
\(-\Sigma^\top p_n\), or a one-next-state Monte Carlo sample of the discrete
target. A future policy regression must average the latter conditional on the
current state and endpoint. One sample is not the conditional expectation, a
policy-improvement guarantee, or an endpoint projection.

## Version-1 assumptions

The exactness statements above are restricted to:

- finite-dimensional discrete Markov chains on a uniform time grid;
- constant, square, full-rank, sufficiently well-conditioned diffusion;
- equal state and noise dimensions;
- differentiable drift/current feedback, with the policy-state derivative
  included in the transition tangent;
- adapted arrival-flow tangents and the declared Gaussian innovations;
- right-endpoint additive Markov running values and, for the generic EM kernel,
  an integrable terminal value;
- interchange of state differentiation, expectation, and Gaussian integration
  by parts, with boundary terms vanishing;
- finite second moments for theorem-facing labels when MSE regression is used;
- anchors separated from the deterministic/singular terminal transition; and
- no clipping, ridge regularization, or sample filtering in theorem-facing
  labels.

Nonsmooth reward values are allowed when the semigroup derivative and the
required integrability conditions exist; the oracle itself is never
differentiated. State-dependent or rank-deficient diffusion, nonuniform grids,
arbitrary path functionals, and control-dependent diffusion are rejected rather
than silently approximated. An event such as "ever collided" requires a
justified Markov augmentation or a separate path-functional BEL theorem.

## Evidence and status vocabulary

The implementation and experiment statuses are deliberately noninterchangeable:

| Status | What it supports | What it does not support |
|---|---|---|
| `CONDITIONAL_MAM_FOUNDATION` | Capability/scope tag emitted by the inner solver or checkpoint | A passed experiment |
| `PASS_CONDITIONAL_MAM_FOUNDATION` | Discrete-kernel algebra, shape, finite-value, endpoint-pinning, and analytic unit gates passed | Learned Gate-A regression, policy improvement, or a global bridge |
| `PASS_MAM_ANALYTIC_FOUNDATION_SMOKE_NOT_EVIDENCE` | A reduced smoke profile completed its configured diagnostics | Scientific evidence or a paper claim |
| `PASS_MAM_ANALYTIC_FOUNDATION` | A scientific Gate-A run met predeclared smooth, hard-threshold, running-cost, and held-out regression gates | Nonlinear control, Hamiltonian improvement, Adjoint Matching equivalence, or a global bridge |

Gate A uses unpinned analytic Brownian problems to validate the generic BEL
identity and conditional regression. It does not experimentally validate the
future reciprocal/Markov-projection loop. A scientific Gate-A pass requires a
scientific configuration, a disjoint tuning/evaluation protocol, all declared
finite/reference checks, and a complete reproducibility manifest. Selecting a
hyperparameter on the same analytic grid used for the reported error is a
calibration result, not held-out evidence.

No current status supports `PASS_MAM_POLICY_IMPROVEMENT` or a completed
generalized Schrödinger bridge. Those require separate theory and experiment
gates.

## Known failure modes

- BEL variance can diverge or become practically unusable near terminal time;
- nonsmooth values do not guarantee square-integrable labels;
- a low MSE against noisy labels can coexist with biased conditional means;
- finite networks, optimization error, and finite samples introduce regression
  error even when the population identity is exact;
- EM and pinned-chain identities do not by themselves establish a
  continuous-time limit;
- clipping or ridge stabilization changes the theorem-facing target;
- omitting the immediate derivative of a departure-time differentiable cost
  biases a discrete label;
- supplying an endpoint-dependent controller as a global Markov bridge can
  violate the intended information structure; and
- endpoint discrepancy tests can fail even when conditional label tests pass.
