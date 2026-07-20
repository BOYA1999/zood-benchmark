from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ExperimentManifest:
    """Track provenance: config, split hash, raw predictions, metrics."""

    def __init__(self, run_dir: str | Path, config: dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.meta: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "0.1.0",
        }
        self._save_json(self.run_dir / "config.json", config)

    @staticmethod
    def hash_array(arr: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

    @staticmethod
    def hash_file(path: str | Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def record_split(
        self,
        split_name: str,
        train_idx: np.ndarray,
        cal_idx: np.ndarray,
        test_idx: np.ndarray,
        extra: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "split_name": split_name,
            "train_hash": self.hash_array(train_idx),
            "cal_hash": self.hash_array(cal_idx),
            "test_hash": self.hash_array(test_idx),
            "train_size": int(len(train_idx)),
            "cal_size": int(len(cal_idx)),
            "test_size": int(len(test_idx)),
        }
        if extra:
            payload.update(extra)
        split_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        payload["split_hash"] = split_hash
        self._save_json(self.run_dir / f"split_{split_name}.json", payload)
        np.savez(
            self.run_dir / f"split_{split_name}.npz",
            train_idx=train_idx,
            cal_idx=cal_idx,
            test_idx=test_idx,
        )
        return split_hash

    def save_predictions(
        self,
        name: str,
        df: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = self.run_dir / f"predictions_{name}.parquet"
        df.to_parquet(path, index=False)
        meta = {"rows": len(df), "columns": list(df.columns), "path": str(path)}
        if metadata:
            meta.update(metadata)
        self._save_json(self.run_dir / f"predictions_{name}.meta.json", meta)
        return path

    def save_metrics(self, name: str, metrics: dict[str, Any]) -> None:
        self._save_json(self.run_dir / f"metrics_{name}.json", metrics)

    def finalize(self, summary: dict[str, Any] | None = None) -> None:
        payload = {"meta": self.meta, "config": self.config}
        if summary:
            payload["summary"] = summary
        self._save_json(self.run_dir / "manifest.json", payload)

    @staticmethod
    def _save_json(path: Path, obj: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
