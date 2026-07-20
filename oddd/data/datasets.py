from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from oddd.utils.chemistry import admet_risk_proxy, murcko_scaffold, sa_score_proxy

# Lazy import in load_dataset to avoid circular imports when download uses MolecularDataset


@dataclass
class MolecularDataset:
    smiles: list[str]
    y: np.ndarray
    task_type: Literal["classification", "regression"]
    timestamps: np.ndarray | None = None
    target_family: np.ndarray | None = None
    assay_id: np.ndarray | None = None
    scaffold: list[str] | None = None
    admet_risk: np.ndarray | None = None
    syn_risk: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.scaffold is None:
            self.scaffold = [murcko_scaffold(s) for s in self.smiles]
        if self.admet_risk is None:
            self.admet_risk = np.array([admet_risk_proxy(s) for s in self.smiles], dtype=np.float32)
        if self.syn_risk is None:
            self.syn_risk = np.array([sa_score_proxy(s) for s in self.smiles], dtype=np.float32)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "smiles": self.smiles,
                "y": self.y,
                "scaffold": self.scaffold,
                "admet_risk": self.admet_risk,
                "syn_risk": self.syn_risk,
                "timestamp": self.timestamps,
                "target_family": self.target_family,
                "assay_id": self.assay_id,
            }
        )


# Distinct Murcko scaffolds for Tox21-style demonstration pools
_SCAFFOLD_CORE_SMILES = [
    "c1ccccc1", "c1ccncc1", "c1cncnc1", "c1ccoc1", "c1ccsc1",
    "C1CCNCC1", "C1CCOCC1", "C1CCSCC1", "C1CNCCN1", "C1COCCO1",
    "c1ccc2[nH]ccc2c1", "c1ccc2ccccc2c1", "c1ccc2ncccc2c1", "c1ccc2sccc2c1",
    "c1ccc2occc2c1", "c1ccc2nccc2c1", "c1ncc2ccccc2n1", "c1ccc2c(c1)OCO2",
    "C1CC2CCC1C2", "C1CCC2CCCCC2C1", "C1CC2CCCCC2C1", "C1NC2CCCCC2C1",
    "c1ccc(N)cc1", "c1ccc(Cl)cc1", "c1ccc(F)cc1", "c1ccc(OC)cc1",
    "c1ccc(C#N)cc1", "c1ccc(C(F)(F)F)cc1", "c1ccc(S(=O)(=O)N)cc1",
    "c1cc(Cl)ccc1Cl", "c1cc(F)ccc1F", "c1cnccc1", "c1cscn1", "c1cocn1",
    "C1=CCCCC1", "C1=CCCC1", "C1CC=CCC1", "C1CC=CC1", "C1COC1",
    "c1ccc2[nH]c(=O)c(=O)n2c1", "c1ccc2c(c1)NC(=O)CO2", "c1ccc2c(c1)NC(=O)CS2",
    "c1ccc2c(c1)oc(=O)c2", "c1ccc2c(c1)sc(=O)n2", "c1ccc2c(c1)ncn2",
    "N1CCNCC1", "N1CCCC1", "O1CCCC1", "S1CCCC1", "C1CCNC(=O)C1",
    "c1ccc2[nH]cnc2c1", "c1ccc2ncnc2c1", "c1ccc2occc2c1", "c1ccc2ccnc2c1",
    "c1ccc2c(c1)OCO2", "c1ccc2c(c1)C(=O)O", "c1ccc2c(c1)C(=O)N",
    "c1ccc2c(c1)S(=O)(=O)", "c1ccc2c(c1)CN", "c1ccc2c(c1)CO",
    "C1C2CC3CC1CC3C2", "C1CC2OC(C1)O2", "c1cc2ccccc2[nH]1",
    "c1ccc2[nH]c3ccccc3c2c1", "c1ccc2c(c1)cccc2", "c1ccc2c(c1)cccn2",
    "c1ccc2c(c1)ccco2", "c1ccc2c(c1)cccs2", "c1ccc2c(c1)ccnn2",
    "C1CC(C)CC(C)C1", "C1CC(CC(C)C)C1", "C1CC2(C1)CNC2",
    "c1nc2ccccc2n1", "c1nc2cccnc2n1", "c1oc2ccccc2o1",
    "c1sc2ccccc2s1", "c1nc2ccncc2n1", "c1cc2c(s1)CCCC2",
    "c1ccc2c(c1)CNC2", "c1ccc2c(c1)COC2", "c1ccc2c(c1)CSC2",
    "c1ccc2c(c1)NC(=O)C2", "c1ccc2c(c1)C(=O)NC2", "c1ccc2c(c1)OC(=O)C2",
    "c1ccc2c(c1)C(=O)OC2", "c1ccc2c(c1)NC(=O)N2", "c1ccc2c(c1)C(=O)C2",
    "C1CCc2ccccc2C1", "C1CCc2cccnc2C1", "C1CCc2ccncc2C1",
    "c1ccc2c(c1)C(=O)c2", "c1ccc2c(c1)N=CN2", "c1ccc2c(c1)NCN2",
    "c1ccc2c(c1)OCN2", "c1ccc2c(c1)SCN2", "c1ccc2c(c1)NC(=O)O2",
    "c1ccc2c(c1)C(=O)NC(=O)N2", "c1ccc2c(c1)C(=O)NC(=O)C2",
    "c1ccc2c(c1)C(=O)NC(=O)O2", "c1ccc2c(c1)C(=O)NC(=O)S2",
]


def _make_synthetic_classification(n: int, seed: int) -> MolecularDataset:
    """Tox21-style demonstration pool; one SMILES per distinct Murcko core."""
    rng = np.random.default_rng(seed)
    smiles = []
    y = []
    timestamps = []
    target_family = []
    assay_id = []
    n_cores = len(_SCAFFOLD_CORE_SMILES)
    for i in range(n):
        core = _SCAFFOLD_CORE_SMILES[i % n_cores]
        smiles.append(core)
        core_idx = i % n_cores
        base = 0.12 + 0.5 * (core_idx % 11) / 11.0
        label = int(base + rng.normal(0, 0.15) > 0.5)
        y.append(label)
        timestamps.append(rng.integers(2015, 2025))
        target_family.append(rng.integers(0, 5))
        assay_id.append(rng.integers(0, 8))
    return MolecularDataset(
        smiles=smiles,
        y=np.asarray(y, dtype=np.int64),
        task_type="classification",
        timestamps=np.asarray(timestamps, dtype=np.int64),
        target_family=np.asarray(target_family, dtype=np.int64),
        assay_id=np.asarray(assay_id, dtype=np.int64),
    )


def _make_synthetic_regression(n: int, seed: int) -> MolecularDataset:
    data = _make_synthetic_classification(n, seed)
    y = 4.0 + 2.5 * data.y.astype(np.float32) + np.random.default_rng(seed + 1).normal(0, 0.4, n)
    return MolecularDataset(
        smiles=data.smiles,
        y=y.astype(np.float32),
        task_type="regression",
        timestamps=data.timestamps,
        target_family=data.target_family,
        assay_id=data.assay_id,
    )


def load_tox21_subset(seed: int, max_rows: int = 5000) -> MolecularDataset | None:
    try:
        from tdc.single_pred import Tox
    except ImportError:
        return None
    # Tox21 NR-AR as representative binary endpoint
    split = Tox(name="NR-AR")
    df = split.get_data()
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    y = df["Y"].astype(int).to_numpy()
    return MolecularDataset(
        smiles=df["Drug"].tolist(),
        y=y,
        task_type="classification",
        timestamps=None,
        target_family=np.zeros(len(df), dtype=np.int64),
        assay_id=np.zeros(len(df), dtype=np.int64),
    )


def load_dataset(
    source: str,
    task_type: str,
    n_samples: int | None,
    seed: int,
    endpoint: str = "NR-AR",
    chembl_target: str = "CHEMBL203",
    chembl_activity: str = "IC50",
    use_full: bool = False,
) -> MolecularDataset:
    if source == "tox21":
        from oddd.data.download import load_cached_tox21

        try:
            return load_cached_tox21(
                endpoint=endpoint,
                max_rows=n_samples,
                seed=seed,
                use_full=use_full or n_samples is None,
            )
        except Exception:
            ds = load_tox21_subset(seed=seed, max_rows=n_samples or 5000)
            if ds is not None:
                return ds
    if source == "chembl":
        from oddd.data.download import load_cached_chembl

        return load_cached_chembl(
            target_chembl_id=chembl_target,
            standard_type=chembl_activity,
            max_rows=n_samples,
            seed=seed,
        )
    if task_type == "regression":
        return _make_synthetic_regression(n_samples or 400, seed)
    return _make_synthetic_classification(n_samples or 400, seed)
