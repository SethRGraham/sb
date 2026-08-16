# MAM Gate 1: conditional improvement

This experiment tests the endpoint-conditioned part of MAM before any global
Markov projection. It runs two pinned, two-dimensional problems:

1. a smooth quadratic running cost with an exact discrete-time LQG reference;
2. a value-only binary disk-occupancy cost with an independent replicated
   high-sample path-integral reference.

The implemented harness compares four explicitly different estimators: the
arrival-correct MAM action target, a tangent-free full-return score, a scalar
critic differentiated on untouched queries, and path-integral control. It
also evaluates the final accepted MAM policy against the zero policy using a
fresh paired-common-noise stream. MAM/direct/critic estimate a current-policy
action target and are scored against an independent replicated high-sample
direct score under exactly that frozen policy.

Path integral has a different estimand: the KL-relaxed
path-measure/desirability control. At finite time step this need not equal the
optimum over the frozen Gaussian mean-shift policy class. The harness therefore
compares the low-sample path-integral arm only with an independent high-sample
path-integral reference. It does not label that quantity as hard-task discrete
optimal truth. Smooth LQG alone also receives an exact finite-grid Riccati
optimal-policy diagnostic.

Two baselines in the broader Gate-1 plan are not exposed by the current solver
APIs: joint value-gradient regression and a separate smooth pathwise
Hamiltonian-AM implementation. The harness records both as unavailable with a
reason and blocks a full Gate-1 pass. It does not rename the scalar critic or
differentiate the value-only oracle to manufacture those arms. Adding faithful
implementations is required before this harness can certify the complete
planned comparison set.

The hard task has no discrete optimal-control truth in this harness. The
current-policy direct-score reference is a finite unbiased sample mean under
the declared score identity; the separate self-normalized path-integral
reference is generally finite-sample biased. The harness gates boundary-aware
relative standard error and independent split-half stability for both Monte
Carlo references. It also records ESS where relevant, physical oracle calls,
and usable rows. It never clips or smooths the binary cost.

Run the one-seed plumbing check from the repository root:

```bash
PYTHONPATH=. uv run --no-sync python scripts/run_mam_gate1.py \
  --config experiments/mam_gate1/configs/conditional_smoke.yaml \
  --overwrite
```

The smoke status is always
`COMPLETE_MAM_GATE1_SMOKE_NOT_SCIENTIFIC_EVIDENCE`, even when its numerical
metrics happen to meet the thresholds.

For a more informative development run on a CPU-only laptop, use the checked-in
intermediate profile:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_mam_gate1.py \
  --config experiments/mam_gate1/configs/conditional_local_cpu.yaml
```

This profile uses a nonlinear actor, seven stochastic transitions, two policy
iterations, 56 evaluation queries distributed evenly over the seven anchors,
and independently replicated direct-score and path-integral references.  It is
intended to answer whether the estimator has a usable direction before paying
for the locked experiment.  Its seed is disjoint from the five locked Gate-1
seeds, and it deliberately retains the locked numerical thresholds rather than
relaxing them to manufacture a pass.

The local profile is still a one-seed `smoke` run: it always remains
non-scientific, cannot satisfy the five-seed gate, and cannot repair the two
missing comparison arms.  Treat it as inconclusive unless all estimator rows
are finite, the hard-boundary effective query count is at least 20, both
reference families have weighted relative standard error at most 0.10 and
split-half relative L2 at most 0.30, and the path-integral ESS floor is met.
These are informal local-readiness diagnostics, not the configured locked
Gate-1 thresholds, which remain unchanged and substantially stricter.
The coarse eight-step result does not establish behavior at the locked
64-step discretization.  On a four-core Intel laptop the run is expected to be
compilation-dominated and may take roughly 8--25 minutes; close memory-heavy
applications first.  Use a fresh `--output-dir` for repeated runs instead of
`--overwrite`, which intentionally replaces an existing artifact directory.
The first audited local run is summarized in
[`../../docs/mam_gate1_local_cpu_results.md`](../../docs/mam_gate1_local_cpu_results.md).

The locked full command is:

```bash
PYTHONPATH=. uv run --no-sync python scripts/run_mam_gate1.py \
  --config experiments/mam_gate1/configs/conditional_full.yaml
```

The full configuration fixes five seeds, the user-requested accuracy gates,
all sample counts, and every optimization setting. Only an exact contract
match with complete comparison coverage, a stable clean Git revision, and all
numerical gates can emit `PASS_MAM_GATE1_CONDITIONAL`. The full contract uses a
strictly positive objective-improvement margin of `1e-4`; a merely negative
sample mean is insufficient. Modified or dirty full runs are explicitly
ineligible, and failed/ineligible full runs exit nonzero.
This repository does not claim the full profile passes until that exact
command has completed and its artifacts have been audited.

Each run writes:

- `resolved_config.json` — the complete closed configuration;
- `results.json` — deterministic scientific metrics and sufficient statistics;
- `run_manifest.json` — device, dependency, git, RNG-ledger, artifact hash, and
  compile-inclusive warm-up/steady-state timing metadata.

This is a conditional gate only. It does not establish global endpoint
feasibility, a wall-clock win, a one-GPU memory bound, or superiority to any
baseline.
