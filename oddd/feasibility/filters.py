from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from oddd.utils.chemistry import mol_from_smiles

# SA score: 1 = easy, 10 = hard. Threshold ≤ 6 is a common med-chem cutoff.
SA_PASS_THRESHOLD = 6.0
QED_PASS_THRESHOLD = 0.4
MW_MAX = 500.0
LOGP_MAX = 5.0
HBD_MAX = 5
HBA_MAX = 10


@lru_cache(maxsize=1)
def _pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


@lru_cache(maxsize=1)
def _brenk_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)


def _sa_score(mol) -> float:
    try:
        from rdkit.Chem import RDConfig
        import os
        import sys

        sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_dir not in sys.path:
            sys.path.append(sa_dir)
        import sascorer  # type: ignore

        return float(sascorer.calculateScore(mol))
    except Exception:
        from oddd.utils.chemistry import sa_score_proxy

        return float(sa_score_proxy(mol) * 10.0)


def compute_feasibility_profile(smiles: str) -> dict[str, float | bool]:
    """Independent synthesis/developability filters — not used in acquisition score."""
    mol = mol_from_smiles(smiles)
    sa = _sa_score(mol)
    qed = float(Descriptors.qed(mol))
    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    pains = bool(_pains_catalog().HasMatch(mol))
    brenk = bool(_brenk_catalog().HasMatch(mol))
    lipinski_violations = int(mw > MW_MAX) + int(logp > LOGP_MAX) + int(hbd > HBD_MAX) + int(hba > HBA_MAX)

    sa_pass = sa <= SA_PASS_THRESHOLD
    qed_pass = qed >= QED_PASS_THRESHOLD
    pains_pass = not pains
    brenk_pass = not brenk
    physchem_pass = lipinski_violations == 0
    composite_pass = bool(sa_pass and qed_pass and pains_pass and brenk_pass and physchem_pass)

    return {
        "sa_score": sa,
        "qed": qed,
        "mw": mw,
        "logp": logp,
        "hbd": float(hbd),
        "hba": float(hba),
        "pains_hit": pains,
        "brenk_hit": brenk,
        "lipinski_violations": float(lipinski_violations),
        "sa_pass": sa_pass,
        "qed_pass": qed_pass,
        "pains_pass": pains_pass,
        "brenk_pass": brenk_pass,
        "physchem_pass": physchem_pass,
        "feasibility_pass": composite_pass,
    }


def compute_feasibility_batch(smiles: list[str]) -> pd.DataFrame:
    rows = [compute_feasibility_profile(s) for s in smiles]
    return pd.DataFrame(rows)


def feasibility_pass_vector(smiles: list[str]) -> np.ndarray:
    return np.array([compute_feasibility_profile(s)["feasibility_pass"] for s in smiles], dtype=bool)
