#!/usr/bin/env python
"""Prepare full Tox21 endpoints and run multi-endpoint benchmark matrix."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oddd.benchmark.matrix import BenchmarkMatrix
from oddd.data.download import TOX21_ENDPOINTS, ensure_full_tox21_endpoint
from oddd.utils.config import load_config
from scripts.run_pipeline import run_single_split

DEFAULT_SPLITS = ["random", "scaffold", "cluster"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _resolve_endpoints(cfg: dict) -> list[str]:
    data_cfg = cfg.get("data", {})
    endpoints = data_cfg.get("tox21_endpoints")
    if endpoints:
        return list(endpoints)
    return [data_cfg.get("tox21_endpoint", "NR-AR")]


def main():
    parser = argparse.ArgumentParser(description="Run ODDD Tox21 multi-endpoint benchmark")
    parser.add_argument("--config", default="configs/tox21_benchmark.yml")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--endpoints", nargs="+", default=None, help="Override config endpoints")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-all-endpoints", action="store_true", help="Materialize all 12 Tox21 endpoints")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    endpoints = args.endpoints or _resolve_endpoints(cfg)

    if args.prepare_all_endpoints:
        for ep in TOX21_ENDPOINTS:
            path = ensure_full_tox21_endpoint(ep)
            print(f"Prepared {ep}: {path}")
        if args.prepare_only:
            return

    for ep in endpoints:
        path = ensure_full_tox21_endpoint(ep)
        print(f"Full endpoint cache: {ep} -> {path}")

    if args.prepare_only:
        return

    matrix_path = Path(cfg.get("benchmark", {}).get("matrix_path", "runs/tox21_benchmark/benchmark_matrix.parquet"))
    matrix = BenchmarkMatrix(matrix_path)
    if matrix_path.exists():
        matrix.merge_existing(matrix_path)

    pack_root = Path(cfg["experiment"]["output_dir"])
    total = len(args.seeds) * len(args.splits) * len(endpoints)
    done = 0

    for endpoint in endpoints:
        ep_root = pack_root / endpoint
        for seed in args.seeds:
            for split in args.splits:
                done += 1
                run_cfg = copy.deepcopy(cfg)
                run_cfg["data"]["tox21_endpoint"] = endpoint
                run_cfg["experiment"]["seed"] = seed
                run_cfg["experiment"]["output_dir"] = str(ep_root / f"seed_{seed}")
                run_cfg.setdefault("figures", {})["enabled"] = False
                print(f"\n>>> [{done}/{total}] endpoint={endpoint} seed={seed} split={split}")
                run_single_split(run_cfg, split_protocol=split, matrix=matrix)
                matrix.save(matrix_path)
    matrix.dedupe()
    matrix.save(matrix_path)

    matrix.save(matrix_path)
    matrix.dedupe()
    matrix.save(matrix_path)
    summary_path = pack_root / "benchmark_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": str(args.config),
                "endpoints": endpoints,
                "seeds": args.seeds,
                "splits": args.splits,
                "matrix_path": str(matrix_path),
                "n_rows": len(matrix.to_frame()),
            },
            f,
            indent=2,
        )
    print(f"\nBenchmark matrix: {matrix_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
