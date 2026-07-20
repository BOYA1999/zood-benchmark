#!/usr/bin/env python
"""Summarize core evidence for JoC-style rebuttal diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _pareto_dominates(a: pd.Series, b: pd.Series) -> bool:
    # maximize: hit/diversity/feasibility
    ge = (
        a["batch_hit_rate"] >= b["batch_hit_rate"]
        and a["scaffold_diversity"] >= b["scaffold_diversity"]
        and a["feasibility_pass_rate"] >= b["feasibility_pass_rate"]
    )
    gt = (
        a["batch_hit_rate"] > b["batch_hit_rate"]
        or a["scaffold_diversity"] > b["scaffold_diversity"]
        or a["feasibility_pass_rate"] > b["feasibility_pass_rate"]
    )
    return bool(ge and gt)


def _load_matrix(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ODCA core-evidence comparisons")
    parser.add_argument(
        "--matrix",
        default="runs/tox21_benchmark/benchmark_matrix.parquet",
        help="Benchmark matrix parquet/csv path",
    )
    parser.add_argument("--split", default="scaffold")
    parser.add_argument("--out", default="runs/tox21_benchmark/core_evidence_summary.json")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    out_path = Path(args.out)
    df = _load_matrix(matrix_path)
    df = df[(df["status"] == "ok") & (df["model"] == "mlp") & (df["split_protocol"] == args.split)].copy()

    needed = {"odca", "ensemble_ucb", "diversity_filtered_ucb", "maxmin_ucb"}
    df = df[df["strategy"].isin(needed)]
    key_cols = ["endpoint", "seed", "split_protocol"]
    wide = {}
    for strategy in needed:
        s = df[df["strategy"] == strategy][
            key_cols + ["batch_hit_rate", "scaffold_diversity", "feasibility_pass_rate"]
        ].copy()
        wide[strategy] = s.set_index(key_cols)

    common_index = None
    for strategy in needed:
        idx = wide[strategy].index
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or len(common_index) == 0:
        raise RuntimeError("No common endpoint-seed rows across strategies; rerun benchmark first.")

    comp = {}
    for strategy in ["ensemble_ucb", "diversity_filtered_ucb", "maxmin_ucb"]:
        od = wide["odca"].loc[common_index]
        ref = wide[strategy].loc[common_index]
        delta = od - ref
        comp[strategy] = {
            "n_pairs": int(len(delta)),
            "mean_delta_hit_rate": float(delta["batch_hit_rate"].mean()),
            "mean_delta_scaffold_diversity": float(delta["scaffold_diversity"].mean()),
            "mean_delta_feasibility_pass_rate": float(delta["feasibility_pass_rate"].mean()),
            "odca_hit_rate_better_count": int((delta["batch_hit_rate"] > 0).sum()),
            "odca_diversity_better_count": int((delta["scaffold_diversity"] > 0).sum()),
            "odca_feasibility_better_count": int((delta["feasibility_pass_rate"] > 0).sum()),
        }

    odca_dominated = 0
    odca_dominates = 0
    for idx in common_index:
        odca_row = wide["odca"].loc[idx]
        competitors = [wide["ensemble_ucb"].loc[idx], wide["diversity_filtered_ucb"].loc[idx], wide["maxmin_ucb"].loc[idx]]
        if any(_pareto_dominates(c, odca_row) for c in competitors):
            odca_dominated += 1
        if any(_pareto_dominates(odca_row, c) for c in competitors):
            odca_dominates += 1

    summary = {
        "matrix_path": str(matrix_path),
        "split": args.split,
        "n_common_rows": int(len(common_index)),
        "strategy_comparisons": comp,
        "pareto_status": {
            "odca_dominated_rows": int(odca_dominated),
            "odca_dominates_any_rows": int(odca_dominates),
            "odca_dominated_fraction": float(odca_dominated / len(common_index)),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"core evidence summary written: {out_path}")


if __name__ == "__main__":
    main()
