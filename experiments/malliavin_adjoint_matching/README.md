# Malliavin Adjoint Matching experiments

This directory contains the reproducible analytic foundation for the
value-only MAM inner solver. It does not run nonlinear policy improvement or
the reciprocal/Markov projection required for a global generalized
Schrödinger bridge.

Run the reduced diagnostic profile from the repository root. It exercises the
complete artifact path but is not guaranteed to satisfy even its relaxed
regression gates; the initial seed-0 run did not.

```bash
PYTHONPATH=. JAX_ENABLE_X64=True uv run --no-sync python \
  scripts/run_mam_foundation.py \
  --config experiments/malliavin_adjoint_matching/configs/brownian_threshold_smoke.yaml \
  --output-dir outputs/mam/foundation_smoke/seed_0 \
  --seed 0 \
  --overwrite
```

Run the frozen scientific Gate-A profile:

```bash
PYTHONPATH=. JAX_ENABLE_X64=True uv run --no-sync python \
  scripts/run_mam_foundation.py \
  --config experiments/malliavin_adjoint_matching/configs/brownian_threshold.yaml \
  --output-dir outputs/mam/foundation/seed_0 \
  --seed 0 \
  --overwrite
```

The run writes `resolved_config.json`, `results.json`, `run_manifest.json`,
`training_metrics.csv`, `raw_samples.npz`, and `checkpoint.npz`. A production
pass is `PASS_MAM_ANALYTIC_FOUNDATION`; the reduced profile is always labeled
`PASS_MAM_ANALYTIC_FOUNDATION_SMOKE_NOT_EVIDENCE` when its looser gates pass.
Both statuses concern analytic fixed-policy costate recovery only.

The first seed-0 artifact and its post-run protocol audit are summarized in
[`../../docs/mam_gate_a_results.md`](../../docs/mam_gate_a_results.md). Do not
reuse that observed evaluation grid as a fresh held-out set.
