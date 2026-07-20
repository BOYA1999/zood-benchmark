#!/usr/bin/env python
"""Run pipeline across multiple seeds for repeated-run statistics."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oddd.metrics.statistics import bootstrap_cohens_d_ci, compare_repeated_runs, wilcoxon_signed_rank
from oddd.utils.config import load_config
from scripts.run_pipeline import run_single_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    runs_proposed = []
    runs_ucb = []

    for seed in args.seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["experiment"]["seed"] = seed
        run_cfg["experiment"]["output_dir"] = str(
            Path(cfg["experiment"]["output_dir"]) / f"seed_{seed}"
        )
        print(f"\n>>> seed={seed}")
        summary = run_single_split(run_cfg, split_protocol=args.split)
        runs_proposed.append(summary["strategies"]["proposed"])
        runs_ucb.append(summary["strategies"]["ensemble_ucb"])

    stats = {}
    for metric in ["ef_1pct", "top_1pct_hit_rate", "high_confidence_error_rate"]:
        stats[metric] = compare_repeated_runs(
            runs_proposed, runs_ucb, metric, seed=args.seeds[0]
        )

    out = Path(cfg["experiment"]["output_dir"]) / "repeated_run_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seeds": args.seeds, "comparisons": stats}, f, indent=2, default=str)
    print(f"\nRepeated-run stats: {out}")


if __name__ == "__main__":
    main()
