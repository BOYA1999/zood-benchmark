from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from oddd.data.datasets import MolecularDataset
from oddd.utils.chemistry import mol_from_smiles

CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache"
TOX21_ENDPOINTS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_manifest(manifest_path: Path, entry: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"datasets": [], "updated_at": None}
    manifest["datasets"] = [d for d in manifest["datasets"] if d.get("id") != entry["id"]]
    manifest["datasets"].append(entry)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _canonicalize_smiles(smiles: str) -> str | None:
    try:
        from rdkit import Chem

        mol = mol_from_smiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _download_tox21_from_url(endpoint: str, cache_dir: Path) -> pd.DataFrame:
    """Fallback: fetch Tox21 label file from public mirror (no PyTDC)."""
    # Canonical Tox21 label archive (12 assays, tab-separated)
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = cache_dir / "tox21_full.csv.gz"
    if not raw.exists():
        req = Request(url, headers={"User-Agent": "ODDD/0.1"})
        with urlopen(req, timeout=120) as resp:
            raw.write_bytes(resp.read())
    df = pd.read_csv(raw, compression="gzip")
    col_map = {c: c for c in df.columns}
    # tox21.csv: smiles column + NR-AR etc.
    if endpoint not in df.columns:
        raise ValueError(f"Endpoint {endpoint} not in tox21.csv columns")
    sub = df[["smiles", endpoint]].dropna()
    sub = sub.rename(columns={"smiles": "smiles", endpoint: "y"})
    sub["y"] = sub["y"].astype(int)
    sub["endpoint"] = endpoint
    sub["canonical_smiles"] = sub["smiles"].map(_canonicalize_smiles)
    sub = sub.dropna(subset=["canonical_smiles"]).drop_duplicates(subset=["canonical_smiles"])
    return sub.reset_index(drop=True)


def _load_tox21_from_local_full_csv(endpoint: str, cache_dir: Path) -> pd.DataFrame:
    """Build endpoint table from locally cached tox21_full.csv.gz (no subsample)."""
    raw = cache_dir / "tox21_full.csv.gz"
    if not raw.exists():
        raise FileNotFoundError(f"Missing local Tox21 archive: {raw}")
    df = pd.read_csv(raw, compression="gzip")
    if endpoint not in df.columns:
        raise ValueError(f"Endpoint {endpoint} not in tox21_full.csv.gz columns")
    smiles_col = "smiles" if "smiles" in df.columns else "Drug"
    sub = df[[smiles_col, endpoint]].dropna(subset=[endpoint])
    sub = sub.rename(columns={smiles_col: "smiles", endpoint: "y"})
    sub["y"] = sub["y"].astype(int)
    sub["endpoint"] = endpoint
    sub["canonical_smiles"] = sub["smiles"].astype(str).map(_canonicalize_smiles)
    sub = sub.dropna(subset=["canonical_smiles"]).drop_duplicates(subset=["canonical_smiles"])
    return sub.reset_index(drop=True)


def ensure_full_tox21_endpoint(
    endpoint: str = "NR-AR",
    cache_dir: Path | None = None,
) -> Path:
    """Materialize full endpoint parquet from local tox21_full.csv.gz."""
    cache_dir = cache_dir or (CACHE_ROOT / "tox21")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{endpoint}_full.parquet"
    df = _load_tox21_from_local_full_csv(endpoint, cache_dir)
    df.to_parquet(out, index=False)
    entry = {
        "id": f"tox21_{endpoint}_full",
        "source": "deepchem/tox21.csv.gz",
        "provenance_tier": "public_mirror_full",
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "endpoint": endpoint,
        "path": str(out),
        "n_records": int(len(df)),
        "positive_rate": float(df["y"].mean()),
        "n_positives": int(df["y"].sum()),
        "sha256": _sha256_file(out),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "filter_rules": f"canonical_smiles dedup; full {endpoint} column from local tox21_full.csv.gz; no subsample",
        "subsample_seed": None,
        "subsample_max_rows": None,
        "local_csv": str(cache_dir / "tox21_full.csv.gz"),
    }
    _save_manifest(cache_dir / "manifest.json", entry)
    return out


def create_offline_tox21_sample(
    endpoint: str = "NR-AR",
    cache_dir: Path | None = None,
    n_rows: int = 500,
    seed: int = 42,
) -> Path:
    """Create a local Tox21-format cache when network / PyTDC unavailable."""
    from oddd.data.datasets import _make_synthetic_classification

    cache_dir = cache_dir or (CACHE_ROOT / "tox21")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{endpoint}.parquet"

    ds = _make_synthetic_classification(n_rows, seed)
    from oddd.utils.chemistry import murcko_scaffold

    murcko = [murcko_scaffold(s) for s in ds.smiles]
    df = pd.DataFrame(
        {
            "smiles": ds.smiles,
            "y": ds.y.astype(int),
            "endpoint": endpoint,
            "canonical_smiles": ds.smiles,
            "murcko_scaffold": murcko,
        }
    )
    df.to_parquet(out, index=False)
    entry = {
        "id": f"tox21_{endpoint}",
        "source": "offline_sample",
        "provenance_tier": "demonstration_not_public_benchmark",
        "endpoint": endpoint,
        "path": str(out),
        "n_records": int(len(df)),
        "n_unique_murcko_scaffolds": int(df["murcko_scaffold"].nunique()),
        "positive_rate": float(df["y"].mean()),
        "sha256": _sha256_file(out),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "filter_rules": "deduplicate by canonical_smiles; no external assay harmonization",
        "replacement_target": "PyTDC Tox21 NR-AR or deepchem tox21.csv.gz",
        "note": "Tox21-style offline demonstration pool only; not official Tox21 release",
    }
    _save_manifest(cache_dir / "manifest.json", entry)
    return out


def download_tox21_endpoint(
    endpoint: str,
    cache_dir: Path | None = None,
    max_rows: int | None = None,
    seed: int = 42,
) -> Path:
    """Download one Tox21 endpoint via PyTDC or public CSV fallback."""
    cache_dir = cache_dir or (CACHE_ROOT / "tox21")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{endpoint}.parquet"

    df = None
    source = "PyTDC/Tox21"
    try:
        from tdc.single_pred import Tox

        split = Tox(name=endpoint)
        raw = split.get_data()
        df = raw.rename(columns={"Drug": "smiles", "Y": "y"})
        df["endpoint"] = endpoint
        df["canonical_smiles"] = df["smiles"].astype(str).map(_canonicalize_smiles)
    except Exception:
        try:
            source = "deepchem/tox21.csv.gz"
            df = _download_tox21_from_url(endpoint, cache_dir)
        except Exception:
            return create_offline_tox21_sample(endpoint, cache_dir, n_rows=max_rows or 500, seed=seed)

    df = df.dropna(subset=["canonical_smiles"]).drop_duplicates(subset=["canonical_smiles"])
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    df.to_parquet(out, index=False)
    entry = {
        "id": f"tox21_{endpoint}",
        "source": source,
        "provenance_tier": "public_mirror_subsample" if source != "offline_sample" else "demonstration_not_public_benchmark",
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz" if "deepchem" in source else None,
        "endpoint": endpoint,
        "path": str(out),
        "n_records": int(len(df)),
        "positive_rate": float(df["y"].mean()),
        "sha256": _sha256_file(out),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "filter_rules": "canonical_smiles dedup; optional max_rows subsample (seed-controlled)",
        "subsample_seed": seed,
        "subsample_max_rows": max_rows,
    }
    _save_manifest(cache_dir / "manifest.json", entry)
    return out


def download_tox21_all(
    cache_dir: Path | None = None,
    max_rows_per_endpoint: int | None = None,
    seed: int = 42,
) -> list[Path]:
    paths = []
    for ep in TOX21_ENDPOINTS:
        paths.append(download_tox21_endpoint(ep, cache_dir, max_rows_per_endpoint, seed))
    return paths


def _fetch_chembl_page(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "ODDD/0.1"})
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def download_chembl_target_activity(
    target_chembl_id: str,
    standard_type: str = "IC50",
    cache_dir: Path | None = None,
    max_records: int = 20000,
    page_size: int = 1000,
) -> Path:
    """
    Download ChEMBL activity records for a target via public REST API.
    Saves standardized pActivity (classification label optional) to parquet.
    """
    cache_dir = cache_dir or (CACHE_ROOT / "chembl")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{target_chembl_id}_{standard_type}.parquet"

    base = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type": standard_type,
        "standard_relation": "=",
        "limit": page_size,
    }
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < max_records:
        params["offset"] = offset
        url = f"{base}?{urlencode(params)}"
        payload = _fetch_chembl_page(url)
        page = payload.get("activities", [])
        if not page:
            break
        for rec in page:
            smiles = rec.get("canonical_smiles")
            val = rec.get("standard_value")
            units = rec.get("standard_units")
            if not smiles or val is None:
                continue
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                continue
            if units and str(units).lower() == "nm" and val_f > 0:
                pactivity = 9.0 - np.log10(val_f * 1e-9)
            else:
                pactivity = np.nan
            rows.append(
                {
                    "smiles": smiles,
                    "standard_value": val_f,
                    "standard_units": units,
                    "pactivity": pactivity,
                    "assay_chembl_id": rec.get("assay_chembl_id"),
                    "document_year": rec.get("document_year"),
                    "target_chembl_id": target_chembl_id,
                    "standard_type": standard_type,
                }
            )
        offset += page_size
        if len(page) < page_size:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No ChEMBL records for {target_chembl_id} ({standard_type})")

    df["canonical_smiles"] = df["smiles"].map(_canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles", "pactivity"])
    df = df.drop_duplicates(subset=["canonical_smiles"], keep="first")
    df = df.reset_index(drop=True)

    df.to_parquet(out, index=False)
    entry = {
        "id": f"chembl_{target_chembl_id}_{standard_type}",
        "source": "ChEMBL REST API",
        "target_chembl_id": target_chembl_id,
        "standard_type": standard_type,
        "path": str(out),
        "n_records": int(len(df)),
        "pactivity_median": float(df["pactivity"].median()),
        "sha256": _sha256_file(out),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_manifest(cache_dir / "manifest.json", entry)
    return out


def load_cached_tox21(
    endpoint: str = "NR-AR",
    cache_dir: Path | None = None,
    max_rows: int | None = None,
    seed: int = 42,
    use_full: bool = False,
) -> MolecularDataset:
    cache_dir = cache_dir or (CACHE_ROOT / "tox21")
    full_path = cache_dir / f"{endpoint}_full.parquet"
    subsample_path = cache_dir / f"{endpoint}.parquet"

    if use_full or max_rows is None:
        if not full_path.exists():
            ensure_full_tox21_endpoint(endpoint, cache_dir)
        path = full_path
    else:
        path = subsample_path
        if not path.exists():
            download_tox21_endpoint(endpoint, cache_dir, max_rows=max_rows, seed=seed)

    df = pd.read_parquet(path)
    if not use_full and max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return MolecularDataset(
        smiles=df["canonical_smiles"].tolist(),
        y=df["y"].astype(int).to_numpy(),
        task_type="classification",
        timestamps=None,
        target_family=np.zeros(len(df), dtype=np.int64),
        assay_id=np.zeros(len(df), dtype=np.int64),
    )


def load_cached_chembl(
    target_chembl_id: str = "CHEMBL203",
    standard_type: str = "IC50",
    cache_dir: Path | None = None,
    max_rows: int | None = None,
    seed: int = 42,
    active_threshold: float = 6.0,
) -> MolecularDataset:
    cache_dir = cache_dir or (CACHE_ROOT / "chembl")
    path = cache_dir / f"{target_chembl_id}_{standard_type}.parquet"
    if not path.exists():
        download_chembl_target_activity(target_chembl_id, standard_type, cache_dir)
    df = pd.read_parquet(path)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    years = df["document_year"].fillna(2018).astype(int).to_numpy()
    return MolecularDataset(
        smiles=df["canonical_smiles"].tolist(),
        y=df["pactivity"].astype(np.float32).to_numpy(),
        task_type="regression",
        timestamps=years,
        target_family=np.zeros(len(df), dtype=np.int64),
        assay_id=pd.factorize(df["assay_chembl_id"].fillna("unknown"))[0].astype(np.int64),
    )
