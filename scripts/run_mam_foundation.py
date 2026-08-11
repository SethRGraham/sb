#!/usr/bin/env python3
"""Run the reproducible Gate-A Malliavin Adjoint Matching calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.malliavin_adjoint_matching.foundation import load_config, run_foundation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed-policy analytic MAM foundation. This does not train "
            "or certify a global Schrödinger bridge."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Frozen YAML configuration")
    parser.add_argument("--output-dir", type=Path, help="Override experiment.output_dir")
    parser.add_argument("--seed", type=int, help="Override experiment.seed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace artifacts in an existing output directory",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_foundation(
        load_config(args.config),
        output_dir=args.output_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    selected = result["hard_threshold_costate_regression"]["selected"]
    summary = {
        "status": result["status"],
        "hard_threshold_costate_relative_l2": selected["relative_l2"],
        "hard_threshold_costate_cosine": selected["cosine"],
        "smooth_bel_vs_pathwise_status": result["smooth_terminal"]["status"],
        "output_dir": result["output_dir"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if str(result["status"]).startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
