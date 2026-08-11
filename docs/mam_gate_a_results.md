# MAM Gate-A result snapshot

Last updated: 2026-08-11

## Result

The frozen CPU run at `outputs/mam/foundation/seed_0` finished with
`FAIL_MAM_ANALYTIC_FOUNDATION`. This result is preserved as a negative Gate-A
result; the acceptance thresholds were not relaxed after observing it.

A post-run audit also found that the configuration declared uniformly sampled
discrete anchor indices while that implementation sampled anchor times from a
continuous uniform distribution. The run is therefore diagnostic rather than
a valid scientific Gate-A attempt, independently of its failed regression
metrics. The implementation contract was corrected afterward; this artifact
remains bound by its recorded source hashes and is not being rewritten.

The value-only BEL identities themselves passed:

- smooth terminal BEL relative error: 0.10%, with paired BEL-versus-pathwise
  difference at 0.31 standard errors;
- hard-threshold BEL relative error: 1.02%, at 1.55 standard errors;
- right-endpoint running-cost relative error: 0.20%, at 0.18 standard errors;
- every reported label and reference value was finite; and
- no theorem-facing clipping was used.

The learned conditional costate missed all three predeclared regression gates:

| Held-out metric | Result | Required |
|---|---:|---:|
| Relative L2 error | 0.1739 | at most 0.10 |
| Cosine similarity | 0.9857 | at least 0.99 |
| Sign agreement | 0.9219 | at least 0.99 |

The disjoint validation set selected learning rate `0.001`; the final analytic
grid was not used for that selection. Candidate networks shared the same
131,072 training labels, initialization, and minibatch schedule. Validation
used 32,768 separately keyed labels. Training and evaluation took 225.5 seconds
on CPU with JAX/JAXLIB 0.4.38 in float64/highest precision.

## Tail diagnostics

For the hard-threshold training labels, the absolute-value p99/p99.9/max were
5.37/8.21/13.90. The top 1% and top 0.1% of samples contributed 23.2% and 4.7%
of centered label energy. For the running-cost labels those shares were 40.6%
and 12.2%, respectively. These tails were reported without censoring.

## Post-audit contract smoke

After correcting the anchor, running-cost, implementation-equivalence, and
provenance contracts, the reduced run at
`outputs/mam/foundation_smoke_contract_v2/seed_0` completed in 42.6 seconds.
All analytic and eager/JIT/reference-loop checks passed. As expected for the
small diagnostic budget, neural recovery still failed its relaxed gates:
relative L2 was 0.4427, cosine similarity was 0.8977, and sign agreement was
0.875. The artifact is explicitly `smoke_not_evidence` and is not used to
support a scientific claim.

## Interpretation and stop line

This run supports the implemented BEL mean identities for the declared
Brownian calibrations. It does **not** yet support the stronger claim that the
configured ordinary-MSE network recovers the analytic hard-threshold costate
to the frozen tolerance. The failure is therefore in finite-sample/function-
approximation/optimization performance, not evidence that the analytic BEL
identity is biased.

Per the experiment contract, nonlinear policy improvement and the global
reciprocal/Markov bridge loop remain blocked. A follow-up must be declared as a
new experiment rather than tuning against this now-observed held-out grid. Good
next diagnostics are learning curves using validation only, repeated-context
or antithetic BEL targets at a fixed reward-query budget, and the mandatory
value-critic-plus-autodiff baseline.

Source artifacts:

- `outputs/mam/foundation/seed_0/results.json`
- `outputs/mam/foundation/seed_0/run_manifest.json`
- `outputs/mam/foundation/seed_0/raw_samples.npz`
- `outputs/mam/foundation/seed_0/checkpoint.npz`
- `outputs/mam/foundation/seed_0/source_snapshot/`
