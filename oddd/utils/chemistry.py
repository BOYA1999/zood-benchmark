from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


@lru_cache(maxsize=100_000)
def murcko_scaffold(smiles: str) -> str:
    try:
        mol = mol_from_smiles(smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return smiles
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return smiles


def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def pairwise_max_tanimoto(fps: np.ndarray) -> np.ndarray:
    """Return per-sample max Tanimoto similarity to reference fingerprints."""
    n = fps.shape[0]
    max_sim = np.zeros(n, dtype=np.float32)
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            inter = np.logical_and(fps[i], fps[j]).sum()
            union = np.logical_or(fps[i], fps[j]).sum()
            sims.append(inter / max(union, 1))
        max_sim[i] = max(sims) if sims else 0.0
    return max_sim


def sa_score_proxy(smiles: str) -> float:
    """Lightweight synthesis-risk proxy in [0, 1]; higher = harder to synthesize."""
    mol = mol_from_smiles(smiles)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    complexity = 0.25 * (mw / 500.0) + 0.25 * (logp / 5.0) + 0.25 * rings / 5.0
    complexity += 0.25 * (1.0 - sp3)
    return float(np.clip(complexity, 0.0, 1.0))


def admet_risk_proxy(smiles: str) -> float:
    """Heuristic ADMET risk proxy in [0, 1]; higher = riskier."""
    mol = mol_from_smiles(smiles)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    risk = 0.35 * violations / 4.0 + 0.35 * max(0.0, (logp - 3.0) / 4.0)
    risk += 0.30 * max(0.0, (90.0 - tpsa) / 90.0)
    return float(np.clip(risk, 0.0, 1.0))
