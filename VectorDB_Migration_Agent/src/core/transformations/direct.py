"""Direct copy (V2 §12) — the safest, cheapest strategy. No transformation is applied;
eligible only when dimension/datatype/metric/vector-type are all EXACT or COMPATIBLE
(never TRANSFORMABLE — that means a real transform is required)."""

from __future__ import annotations

import numpy as np

from core.compatibility.engine import CompatibilityClass, CompatibilityReport
from core.transformations.base import (
    ApplicabilityInputs,
    ApplicabilityResult,
    RepresentationTransformer,
)

_ALLOWED = {CompatibilityClass.EXACT, CompatibilityClass.COMPATIBLE}


class DirectCopyTransformer(RepresentationTransformer):
    strategy = "direct_copy"

    def __init__(self, compatibility: CompatibilityReport) -> None:
        self._compatibility = compatibility

    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        c = self._compatibility
        checks = {
            "dimension": c.dimension,
            "datatype": c.datatype,
            "metric": c.metric,
            "vector_type": c.vector_type,
        }
        blockers = [
            f"{name}={p.classification.value}"
            for name, p in checks.items()
            if p.classification not in _ALLOWED
        ]
        if blockers:
            return ApplicabilityResult(False, "not directly copyable: " + ", ".join(blockers))
        return ApplicabilityResult(
            True, "all properties EXACT/COMPATIBLE — no transformation required"
        )

    async def transform(self, vectors: np.ndarray) -> np.ndarray:
        return np.array(vectors, dtype=np.float32, copy=True)

    def provenance(self) -> dict:
        return {"strategy": self.strategy, "parameters": {}}
