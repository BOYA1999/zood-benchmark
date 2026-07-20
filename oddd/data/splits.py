from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import GroupShuffleSplit

from oddd.data.datasets import MolecularDataset
from oddd.utils.chemistry import mol_from_smiles, murcko_scaffold


SplitProtocol = Literal[
    "random",
    "scaffold",
    "time",
    "cluster",
    "low_tanimoto",
    "scaffold_time",
]


def _butina_clusters(smiles: list[str], cutoff: float = 0.4) -> np.ndarray:
    mols = [mol_from_smiles(s) for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    dists = []
    n = len(fps)
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    labels = np.zeros(n, dtype=np.int64)
    for cid, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cid
    return labels


def _group_split(indices: np.ndarray, groups: np.ndarray, test_ratio: float, seed: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_idx, test_idx = next(splitter.split(indices, groups=groups))
    return indices[train_idx], indices[test_idx]


def _low_tanimoto_holdout(smiles: list[str], test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(smiles)
    order = np.arange(n)
    rng.shuffle(order)
    mols = [mol_from_smiles(smiles[i]) for i in order]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    test_idx_local = []
    train_idx_local = []
    for i in range(n):
        if len(test_idx_local) == 0:
            test_idx_local.append(i)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in test_idx_local])
        if max(sims) < 0.35 and len(test_idx_local) < max(1, int(n * test_ratio)):
            test_idx_local.append(i)
        else:
            train_idx_local.append(i)
    if len(test_idx_local) < max(1, int(n * test_ratio)):
        remaining = [i for i in range(n) if i not in test_idx_local]
        extra = rng.choice(remaining, size=max(1, int(n * test_ratio)) - len(test_idx_local), replace=False)
        test_idx_local.extend(extra.tolist())
        train_idx_local = [i for i in range(n) if i not in test_idx_local]
    test_idx = order[np.asarray(test_idx_local, dtype=np.int64)]
    train_idx = order[np.asarray(train_idx_local, dtype=np.int64)]
    return train_idx, test_idx


def make_split(
    dataset: MolecularDataset,
    protocol: SplitProtocol,
    test_ratio: float,
    cal_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n = len(dataset.smiles)
    all_idx = np.arange(n)

    if protocol == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_test = int(n * test_ratio)
        n_cal = int(n * cal_ratio)
        test_idx = perm[:n_test]
        cal_idx = perm[n_test : n_test + n_cal]
        train_idx = perm[n_test + n_cal :]

    elif protocol == "scaffold":
        groups = np.array([murcko_scaffold(s) for s in dataset.smiles])
        train_pool, test_idx = _group_split(all_idx, groups, test_ratio, seed)
        train_groups = groups[train_pool]
        train_idx, cal_idx = _group_split(train_pool, train_groups, cal_ratio / max(1 - test_ratio, 1e-6), seed + 1)

    elif protocol == "time":
        if dataset.timestamps is None:
            raise ValueError("time split requires timestamps")
        order = np.argsort(dataset.timestamps)
        n_test = int(n * test_ratio)
        n_cal = int(n * cal_ratio)
        test_idx = order[-n_test:]
        cal_idx = order[-(n_test + n_cal) : -n_test]
        train_idx = order[: -(n_test + n_cal)]

    elif protocol == "cluster":
        clusters = _butina_clusters(dataset.smiles)
        train_pool, test_idx = _group_split(all_idx, clusters, test_ratio, seed)
        train_clusters = clusters[train_pool]
        train_idx, cal_idx = _group_split(train_pool, train_clusters, cal_ratio / max(1 - test_ratio, 1e-6), seed + 1)

    elif protocol == "low_tanimoto":
        train_pool, test_idx = _low_tanimoto_holdout(dataset.smiles, test_ratio, seed)
        rng = np.random.default_rng(seed + 2)
        perm = rng.permutation(train_pool)
        n_cal = int(len(train_pool) * cal_ratio / max(1 - test_ratio, 1e-6))
        cal_idx = perm[:n_cal]
        train_idx = perm[n_cal:]

    elif protocol == "scaffold_time":
        groups = np.array([murcko_scaffold(s) for s in dataset.smiles])
        if dataset.timestamps is None:
            raise ValueError("scaffold_time split requires timestamps")
        # First temporal holdout, then scaffold grouping inside train pool
        order = np.argsort(dataset.timestamps)
        n_test = int(n * test_ratio)
        test_idx = order[-n_test:]
        pool = order[:-n_test]
        pool_groups = groups[pool]
        train_idx, cal_idx = _group_split(pool, pool_groups, cal_ratio / max(1 - test_ratio, 1e-6), seed + 3)

    else:
        raise ValueError(f"Unknown split protocol: {protocol}")

    meta = {
        "protocol": protocol,
        "n_train": int(len(train_idx)),
        "n_cal": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
    }
    if protocol in {"scaffold", "scaffold_time", "cluster"}:
        meta["n_train_scaffolds"] = len({dataset.scaffold[i] for i in train_idx})
        meta["n_test_scaffolds"] = len({dataset.scaffold[i] for i in test_idx})
        meta["n_cal_scaffolds"] = len({dataset.scaffold[i] for i in cal_idx})
    return train_idx, cal_idx, test_idx, meta


def compute_split_diagnostics(
    scaffolds: list[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    """Strict split diagnostics: counts, scaffold stats, overlap, positive coverage."""
    from collections import Counter

    y = np.asarray(y).astype(int)
    pool_positives = int(y.sum())
    pool_scaffolds = len(set(scaffolds))

    def _part(idx: np.ndarray, name: str) -> dict:
        scaf = [scaffolds[i] for i in idx]
        counts = Counter(scaf)
        sizes = list(counts.values())
        labels = y[idx]
        n_singletons = sum(1 for s in sizes if s == 1)
        n_scaffolds = len(counts)
        return {
            f"{name}_n": int(len(idx)),
            f"{name}_n_positives": int(labels.sum()),
            f"{name}_positive_rate": float(labels.mean()) if len(labels) else 0.0,
            f"{name}_n_scaffolds": int(n_scaffolds),
            f"{name}_scaffold_size_min": int(min(sizes)) if sizes else 0,
            f"{name}_scaffold_size_median": float(np.median(sizes)) if sizes else 0.0,
            f"{name}_scaffold_size_max": int(max(sizes)) if sizes else 0,
            f"{name}_singleton_scaffold_ratio": float(n_singletons / n_scaffolds) if n_scaffolds else 0.0,
            f"{name}_positive_coverage": float(labels.sum() / pool_positives) if pool_positives else 0.0,
        }

    out: dict = {
        "pool_n": int(len(y)),
        "pool_n_positives": pool_positives,
        "pool_positive_rate": float(y.mean()) if len(y) else 0.0,
        "pool_n_scaffolds": int(pool_scaffolds),
    }
    out.update(_part(train_idx, "train"))
    out.update(_part(cal_idx, "cal"))
    out.update(_part(test_idx, "test"))

    train_scaf = {scaffolds[i] for i in train_idx}
    cal_scaf = {scaffolds[i] for i in cal_idx}
    test_scaf = {scaffolds[i] for i in test_idx}
    out["scaffold_overlap_train_test"] = int(len(train_scaf & test_scaf))
    out["scaffold_overlap_train_cal"] = int(len(train_scaf & cal_scaf))
    out["scaffold_overlap_cal_test"] = int(len(cal_scaf & test_scaf))
    out["scaffold_disjoint_train_test"] = bool(train_scaf.isdisjoint(test_scaf))
    out["scaffold_disjoint"] = out["scaffold_disjoint_train_test"]
    return out


def assign_clusters(
    smiles: list[str],
    sources: list[str],
    min_group_size: int = 15,
    seed: int = 0,
) -> np.ndarray:
    """Fit and transform on the same SMILES list (legacy helper)."""
    assigner = GroupAssigner(min_group_size=min_group_size, seed=seed)
    assigner.fit(smiles, sources)
    groups, _ = assigner.transform(smiles)
    return groups


@dataclass
class GroupAssigner:
    """
    Calibration-fitted Mondrian group assigner.

    Group IDs are defined on the calibration set; test molecules are mapped to
    the same semantic groups via scaffold keys and nearest calibration fingerprint
    cluster assignment. Unmapped keys fall into a dedicated residual bucket.
    """

    min_group_size: int = 15
    seed: int = 0
    residual_id: int = 10_000
    sources: list[str] = field(default_factory=lambda: ["scaffold", "fingerprint"])
    scaf_map_: dict[str, int] = field(default_factory=dict, init=False)
    key_to_group_: dict[str, int] = field(default_factory=dict, init=False)
    cal_fps_: list | None = field(default=None, init=False)
    cal_fp_labels_: np.ndarray | None = field(default=None, init=False)
    diagnostics_: dict = field(default_factory=dict, init=False)

    def _composite_key(self, scaffold: str, fp_cluster: int, sources: list[str]) -> str:
        parts = []
        if "scaffold" in sources:
            scaf_id = self.scaf_map_.get(scaffold)
            parts.append(f"S{scaf_id if scaf_id is not None else 'UNK'}")
        if "fingerprint" in sources:
            parts.append(f"F{fp_cluster}")
        return "|".join(parts) if parts else "G0"

    def _fp_cluster_for_smiles(self, smiles: str) -> int:
        if self.cal_fps_ is None or self.cal_fp_labels_ is None:
            return -1
        fp = AllChem.GetMorganFingerprintAsBitVect(mol_from_smiles(smiles), 2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, self.cal_fps_)
        if not sims:
            return -1
        best_idx = int(np.argmax(sims))
        return int(self.cal_fp_labels_[best_idx])

    def fit(self, smiles: list[str], sources: list[str] | None = None) -> "GroupAssigner":
        self.sources = list(sources or self.sources)
        scaffolds = [murcko_scaffold(s) for s in smiles]
        self.scaf_map_.clear()
        for scaf in scaffolds:
            if scaf not in self.scaf_map_:
                self.scaf_map_[scaf] = len(self.scaf_map_)

        if "fingerprint" in self.sources:
            self.cal_fps_ = [
                AllChem.GetMorganFingerprintAsBitVect(mol_from_smiles(s), 2, nBits=2048) for s in smiles
            ]
            self.cal_fp_labels_ = _butina_clusters(smiles, cutoff=0.45)
        else:
            self.cal_fps_ = None
            self.cal_fp_labels_ = None

        keys = []
        for i, smi in enumerate(smiles):
            fp_cluster = int(self.cal_fp_labels_[i]) if self.cal_fp_labels_ is not None else 0
            keys.append(self._composite_key(scaffolds[i], fp_cluster, self.sources))

        counts: dict[str, int] = defaultdict(int)
        for k in keys:
            counts[k] += 1

        self.key_to_group_.clear()
        for k, c in counts.items():
            if c >= self.min_group_size:
                if k not in self.key_to_group_:
                    self.key_to_group_[k] = len(self.key_to_group_)

        labels = np.full(len(smiles), self.residual_id, dtype=np.int64)
        for i, k in enumerate(keys):
            labels[i] = self.key_to_group_.get(k, self.residual_id)

        n_residual = int((labels == self.residual_id).sum())
        self.diagnostics_ = {
            "n_calibration_molecules": int(len(smiles)),
            "n_effective_groups": int(len(self.key_to_group_)),
            "residual_fraction": float(n_residual / len(smiles)) if smiles else 0.0,
            "min_group_size": int(self.min_group_size),
            "sources": list(self.sources),
        }
        return self

    def transform(self, smiles: list[str]) -> tuple[np.ndarray, dict]:
        if not self.scaf_map_:
            raise RuntimeError("GroupAssigner.fit must be called before transform")

        labels = np.full(len(smiles), self.residual_id, dtype=np.int64)
        unmapped = 0
        for i, smi in enumerate(smiles):
            scaf = murcko_scaffold(smi)
            fp_cluster = self._fp_cluster_for_smiles(smi) if "fingerprint" in self.sources else 0
            key = self._composite_key(scaf, fp_cluster, self.sources)
            gid = self.key_to_group_.get(key)
            if gid is None:
                unmapped += 1
                labels[i] = self.residual_id
            else:
                labels[i] = gid

        diag = {
            **self.diagnostics_,
            "n_test_molecules": int(len(smiles)),
            "unmapped_fraction": float(unmapped / len(smiles)) if smiles else 0.0,
            "mapped_fraction": float(1.0 - unmapped / len(smiles)) if smiles else 0.0,
        }
        return labels, diag
