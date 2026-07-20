#!/usr/bin/env python
"""Run pipeline across all OOD split protocols and aggregate results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_pipeline import run_single_split
from oddd.utils.config import load_config


SPLITS = ["random", "scaffold", "time", "cluster", "low_tanimoto", "scaffold_time"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yml")
    parser.add_argument("--splits", nargs="*", default=SPLITS)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    all_results = {}
    for split in args.splits:
        print(f"\n>>> Running split: {split}")
        summary = run_single_split(cfg, split_protocol=split)
        all_results[split] = summary

    out = Path(cfg["experiment"]["output_dir"]) / "all_splits_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAggregated results: {out}")


if __name__ == "__main__":
    main()
