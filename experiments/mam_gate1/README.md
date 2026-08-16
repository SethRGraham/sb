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
