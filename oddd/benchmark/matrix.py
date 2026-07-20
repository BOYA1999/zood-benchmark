from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class BenchmarkMatrix:
    """Unified long-form table for all benchmark runs."""

    COLUMNS = [
        "run_id",
        "timestamp_utc",
        "data_version",
        "data_sha256",
        "n_records",
        "endpoint",
        "split_protocol",
        "split_hash",
        "seed",
        "model",
        "strategy",
        "status",
        "failure_reason",
        "auroc",
        "auprc",
        "ece_test",
        "marginal_coverage",
        "worst_group_coverage",
        "ef_1pct",
        "top_1pct_hit_rate",
        "batch_hit_rate",
        "scaffold_diversity",
        "syn_pass_rate",
        "feasibility_pass_rate",
        "n_selected",
        "budget_fill_rate",
        "bootstrap_json",
        "split_diagnostics_json",
        "output_dir",
    ]

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._rows: list[dict[str, Any]] = []

    def append(
        self,
        *,
        run_id: str,
        data_provenance: dict,
        split_protocol: str,
        split_hash: str,
        seed: int,
        model: str,
        strategy: str,
        status: str,
        failure_reason: str | None = None,
        prediction: dict | None = None,
        coverage: dict | None = None,
        strategy_metrics: dict | None = None,
        bootstrap: dict | None = None,
        split_diagnostics: dict | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        pred = prediction or {}
        cov = coverage or {}
        strat = strategy_metrics or {}
        row = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "data_version": data_provenance.get("provenance_tier") or data_provenance.get("data_source"),
            "data_sha256": data_provenance.get("cache_sha256"),
            "n_records": data_provenance.get("n_records"),
            "endpoint": data_provenance.get("endpoint"),
            "split_protocol": split_protocol,
            "split_hash": split_hash,
            "seed": int(seed),
            "model": model,
            "strategy": strategy,
            "status": status,
            "failure_reason": failure_reason,
            "auroc": pred.get("auroc"),
            "auprc": pred.get("auprc"),
            "ece_test": pred.get("ece_test"),
            "marginal_coverage": cov.get("marginal_coverage") or strat.get("marginal_coverage"),
            "worst_group_coverage": cov.get("worst_group_coverage") or strat.get("worst_group_coverage"),
            "ef_1pct": strat.get("ef_1pct"),
            "top_1pct_hit_rate": strat.get("top_1pct_hit_rate"),
            "batch_hit_rate": strat.get("hit_rate"),
            "scaffold_diversity": strat.get("scaffold_diversity"),
            "syn_pass_rate": strat.get("syn_pass_rate"),
            "feasibility_pass_rate": strat.get("feasibility_pass_rate"),
            "n_selected": strat.get("n_selected"),
            "budget_fill_rate": strat.get("budget_fill_rate"),
            "bootstrap_json": json.dumps(bootstrap, default=str) if bootstrap else None,
            "split_diagnostics_json": json.dumps(split_diagnostics, default=str) if split_diagnostics else None,
            "output_dir": str(output_dir) if output_dir else None,
        }
        self._rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=self.COLUMNS)

    def save(self, path: str | Path | None = None) -> Path:
        out = Path(path or self.path or "benchmark_matrix.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_frame()
        df.to_parquet(out, index=False)
        csv_path = out.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return out

    def merge_existing(self, path: str | Path) -> None:
        p = Path(path)
        if p.exists():
            existing = pd.read_parquet(p)
            self._rows = existing.to_dict(orient="records")

    def dedupe(self) -> None:
        """Keep latest row per (endpoint, seed, split_protocol, model, strategy)."""
        if not self._rows:
            return
        df = pd.DataFrame(self._rows)
        key = ["endpoint", "seed", "split_protocol", "model", "strategy"]
        for col in key:
            if col not in df.columns:
                return
        df = df.sort_values("timestamp_utc").drop_duplicates(subset=key, keep="last")
        self._rows = df.to_dict(orient="records")
