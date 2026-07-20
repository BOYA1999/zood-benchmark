from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem

from oddd.utils.chemistry import mol_from_smiles, murcko_scaffold


@dataclass
class RoutingResult:
    routed_groups: np.ndarray
    novelty: np.ndarray
    is_ood: np.ndarray
    routed_by_similarity: np.ndarray
    routed_by_scaffold_novelty: np.ndarray


class OODRouter:
    """Route samples to conservative conformal group when far from calibration distribution."""

    def __init__(
        self,
        tanimoto_threshold: float = 0.35,
        ood_group_id: int = -1,
        routing_mode: str = "similarity_only",
        use_scaffold_novelty: bool = False,
    ):
        self.tanimoto_threshold = tanimoto_threshold
        self.ood_group_id = ood_group_id
        self.routing_mode = routing_mode
        self.use_scaffold_novelty = use_scaffold_novelty
        self._cal_fps = None
        self._cal_scaffolds: set[str] = set()

    def fit(self, cal_smiles: list[str], base_groups: np.ndarray) -> "OODRouter":
        self._cal_fps = [
            AllChem.GetMorganFingerprintAsBitVect(mol_from_smiles(s), 2, nBits=2048)
            for s in cal_smiles
        ]
        self._cal_scaffolds = {murcko_scaffold(s) for s in cal_smiles}
        return self

    def route(self, smiles: list[str], base_groups: np.ndarray) -> RoutingResult:
        if self._cal_fps is None:
            raise RuntimeError("OODRouter.fit must be called before route")

        routed = base_groups.copy()
        novelty = np.zeros(len(smiles), dtype=np.float32)
        is_ood = np.zeros(len(smiles), dtype=bool)
        routed_by_similarity = np.zeros(len(smiles), dtype=bool)
        routed_by_scaffold_novelty = np.zeros(len(smiles), dtype=bool)

        for i, smi in enumerate(smiles):
            fp = AllChem.GetMorganFingerprintAsBitVect(mol_from_smiles(smi), 2, nBits=2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, self._cal_fps)
            max_sim = max(sims) if sims else 0.0
            novelty[i] = 1.0 - max_sim
            scaf_novel = murcko_scaffold(smi) not in self._cal_scaffolds

            sim_route = max_sim < self.tanimoto_threshold
            scaf_route = self.use_scaffold_novelty and scaf_novel

            if self.routing_mode == "similarity_only":
                route = sim_route
            elif self.routing_mode == "scaffold_only":
                route = scaf_route
            elif self.routing_mode == "similarity_or_scaffold":
                route = sim_route or scaf_route
            else:
                raise ValueError(f"Unknown routing_mode: {self.routing_mode}")

            routed_by_similarity[i] = sim_route
            routed_by_scaffold_novelty[i] = scaf_route
            if route:
                is_ood[i] = True
                routed[i] = self.ood_group_id

        return RoutingResult(
            routed_groups=routed,
            novelty=novelty,
            is_ood=is_ood,
            routed_by_similarity=routed_by_similarity,
            routed_by_scaffold_novelty=routed_by_scaffold_novelty,
        )
