#!/usr/bin/env python
"""Run Chemprop baseline on representative endpoints (updates benchmark matrix)."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oddd.benchmark.matrix import BenchmarkMatrix
from oddd.data.download import ensure_full_tox21_endpoint
from oddd.utils.config import load_config
from scripts.run_benchmark import DEFAULT_SEEDS, DEFAULT_SPLITS, _resolve_endpoints
from scripts.run_pipeline import run_single_split


def main():
    parser = argparse.ArgumentParser(description="Run Chemprop on representative Tox21 endpoints")
    parser.add_argument("--config", default="configs/tox21_benchmark.yml")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--endpoints", nargs="+", default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    endpoints = args.endpoints or cfg.get("benchmark", {}).get(
        "chemprop_endpoints", ["NR-AR", "NR-PPAR-gamma", "SR-ARE", "NR-AhR"]
    )

    for ep in endpoints:
        ensure_full_tox21_endpoint(ep)

    matrix_path = Path(cfg.get("benchmark", {}).get("matrix_path", "runs/tox21_benchmark/benchmark_matrix.parquet"))
    matrix = BenchmarkMatrix(matrix_path)
    if matrix_path.exists():
        matrix.merge_existing(matrix_path)
        # Drop prior chemprop rows for targeted reruns
        frame = matrix.to_frame()
        matrix._rows = frame[
            ~((frame["model"] == "chemprop_v2") & (frame["endpoint"].isin(endpoints)))
        ].to_dict(orient="records")

    pack_root = Path(cfg["experiment"]["output_dir"])
    total = len(args.seeds) * len(args.splits) * len(endpoints)
    done = 0

    for endpoint in endpoints:
        for seed in args.seeds:
            for split in args.splits:
                done += 1
                run_cfg = copy.deepcopy(cfg)
                run_cfg["data"]["tox21_endpoint"] = endpoint
                run_cfg["benchmark"]["chemprop_only"] = True
                run_cfg["benchmark"]["chemprop_endpoints"] = endpoints
                run_cfg["experiment"]["seed"] = seed
                run_cfg["experiment"]["output_dir"] = str(pack_root / endpoint / f"seed_{seed}")
                run_cfg.setdefault("figures", {})["enabled"] = False
                print(f"\n>>> Chemprop [{done}/{total}] endpoint={endpoint} seed={seed} split={split}")
                run_single_split(run_cfg, split_protocol=split, matrix=matrix)
                matrix.dedupe()
                matrix.save(matrix_path)

    matrix.dedupe()
    matrix.save(matrix_path)
    print(f"\nChemprop matrix updated: {matrix_path} ({len(matrix.to_frame())} rows)")


if __name__ == "__main__":
    main()
