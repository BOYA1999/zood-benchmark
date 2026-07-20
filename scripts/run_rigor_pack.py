#!/usr/bin/env python
"""Minimal rigor pack: random/scaffold/cluster × seeds 42-46 on Tox21 NR-AR."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oddd.utils.config import load_config
from scripts.run_pipeline import run_single_split

DEFAULT_SPLITS = ["random", "scaffold", "cluster"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def main():
    parser = argparse.ArgumentParser(description="Run ODDD minimal rigor pack")
    parser.add_argument("--config", default="configs/tox21_paper.yml")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    args = parser.parse_args()

    base_cfg = load_config(ROOT / args.config)
    pack_root = Path(base_cfg["experiment"]["output_dir"]) / "rigor_pack"
    all_results: dict[str, dict] = {}

    total = len(args.seeds) * len(args.splits)
    done = 0
    for seed in args.seeds:
        for split in args.splits:
            done += 1
            run_cfg = copy.deepcopy(base_cfg)
            run_cfg["experiment"]["seed"] = seed
            run_cfg["experiment"]["output_dir"] = str(pack_root / f"seed_{seed}")
            run_cfg.setdefault("figures", {})["enabled"] = False

            key = f"seed_{seed}/{split}"
            print(f"\n>>> [{done}/{total}] seed={seed} split={split}")
            summary = run_single_split(run_cfg, split_protocol=split)
            all_results[key] = {
                "split_hash": summary["split_hash"],
                "prediction": summary.get("prediction", {}),
                "proposed_coverage": summary.get("proposed_coverage", {}),
                "strategies": summary.get("strategies", {}),
                "bootstrap": summary.get("bootstrap", {}),
            }

    out = pack_root / "rigor_pack_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": args.seeds,
                "splits": args.splits,
                "config": str(args.config),
                "runs": all_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nRigor pack complete: {out}")


if __name__ == "__main__":
    main()
