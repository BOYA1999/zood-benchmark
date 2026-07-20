#!/usr/bin/env python
"""Rebuild benchmark matrix from per-run metrics_summary.json files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oddd.benchmark.matrix import BenchmarkMatrix
from oddd.data.download import TOX21_ENDPOINTS


def _collect_run_files(benchmark_root: Path) -> list[Path]:
    """Collect benchmark summaries, including Chemprop-only sidecar summaries."""
    nested: list[Path] = []
    legacy: list[Path] = []
    for path in benchmark_root.rglob("metrics*.json"):
        if path.name not in {"metrics_summary.json", "metrics_chemprop_summary.json"}:
            continue
        run_dir = path.parent
        if run_dir.parent.parent.name in TOX21_ENDPOINTS:
            nested.append(path)
        elif run_dir.parent.name.startswith("seed_"):
            legacy.append(path)
    return sorted(set(nested if nested else legacy))


def rebuild_matrix(benchmark_root: Path, matrix_path: Path) -> BenchmarkMatrix:
    matrix = BenchmarkMatrix(matrix_path)
    matrix._rows = []

    for summary_path in _collect_run_files(benchmark_root):
        run_dir = summary_path.parent
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        parts = run_dir.parts
        endpoint = summary.get("data_provenance", {}).get("endpoint")
        seed = None
        for i, part in enumerate(parts):
            if part.startswith("seed_"):
                seed = int(part.replace("seed_", ""))
                break
            if part in {
                "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
            }:
                endpoint = endpoint or part

        if seed is None:
            continue

        run_id = f"seed{seed}_{summary['split_protocol']}_{summary['split_hash']}"
        if endpoint:
            run_id = f"{endpoint}_{run_id}"

        boot = summary.get("bootstrap", {})
        for entry in summary.get("benchmark_baselines", []):
            if entry.get("model") == "chemprop_v2" and entry.get("strategy") == "none":
                continue
            matrix.append(
                run_id=run_id,
                data_provenance=summary.get("data_provenance", {}),
                split_protocol=summary["split_protocol"],
                split_hash=summary["split_hash"],
                seed=seed,
                model=entry["model"],
                strategy=entry["strategy"],
                status=entry["status"],
                failure_reason=entry.get("failure_reason"),
                prediction=entry.get("metrics", {}),
                coverage=entry.get("metrics", {}),
                strategy_metrics=entry.get("batch", {}) or entry.get("metrics", {}),
                bootstrap=boot.get(entry["strategy"], {}),
                split_diagnostics=summary.get("split_diagnostics"),
                output_dir=run_dir,
            )

    matrix.save(matrix_path)
    matrix.dedupe()
    return matrix


def main():
    parser = argparse.ArgumentParser(description="Rebuild benchmark matrix from run artifacts")
    parser.add_argument("--root", default="runs/tox21_benchmark")
    parser.add_argument("--matrix", default="runs/tox21_benchmark/benchmark_matrix.parquet")
    args = parser.parse_args()

    matrix = rebuild_matrix(ROOT / args.root, ROOT / args.matrix)
    print(f"Rebuilt matrix: {ROOT / args.matrix} ({len(matrix.to_frame())} rows)")


if __name__ == "__main__":
    main()
