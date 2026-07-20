#!/usr/bin/env python
"""Download and cache Tox21 / ChEMBL datasets with provenance manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oddd.data.download import (
    TOX21_ENDPOINTS,
    create_offline_tox21_sample,
    download_chembl_target_activity,
    download_tox21_all,
    download_tox21_endpoint,
)


def main():
    parser = argparse.ArgumentParser(description="Download ODDD benchmark datasets")
    parser.add_argument("--dataset", choices=["tox21", "chembl", "all"], required=True)
    parser.add_argument("--endpoint", default="NR-AR", help="Tox21 endpoint name")
    parser.add_argument("--target", default="CHEMBL203", help="ChEMBL target ID (e.g. EGFR)")
    parser.add_argument("--activity-type", default="IC50")
    parser.add_argument("--max-records", type=int, default=20000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline-sample", action="store_true", help="Force offline sample cache")
    args = parser.parse_args()

    if args.dataset == "tox21":
        if args.offline_sample:
            path = create_offline_tox21_sample(args.endpoint, n_rows=args.max_rows or 500, seed=args.seed)
        else:
            path = download_tox21_endpoint(args.endpoint, max_rows=args.max_rows, seed=args.seed)
        print(f"Tox21 {args.endpoint}: {path}")
    elif args.dataset == "chembl":
        path = download_chembl_target_activity(
            args.target, args.activity_type, max_records=args.max_records
        )
        print(f"ChEMBL {args.target}: {path}")
    else:
        paths = download_tox21_all(max_rows_per_endpoint=args.max_rows, seed=args.seed)
        chembl = download_chembl_target_activity(
            args.target, args.activity_type, max_records=args.max_records
        )
        print(f"Downloaded {len(paths)} Tox21 endpoints + ChEMBL -> {chembl}")
        print("Tox21 endpoints:", ", ".join(TOX21_ENDPOINTS))


if __name__ == "__main__":
    main()
