"""Knowledge Distillation (V2 §17) — treats the source representation/retrieval system as
teacher, trains a student network producing target-dimension embeddings. Same honesty
standard as trained_projection.py: a real distillation run needs a teacher-student
training loop with an actual loss curve, not a stub that fabricates one. Not built here;
this is the correct interface a real implementation would fill in.
"""

from __future__ import annotations

import numpy as np

from core.transformations.base import (
    ApplicabilityInputs,
    ApplicabilityResult,
    RepresentationTransformer,
)

MIN_TRAINING_PAIRS = 10_000
LOSS_OBJECTIVE = "embedding distillation + contrastive + ranking + score-distillation loss (V2 §17)"


class DistillationTransformer(RepresentationTransformer):
    strategy = "knowledge_distillation"
    is_stub = True

    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        if inputs.source_dim is None or inputs.target_dim is None:
            return ApplicabilityResult(False, "source or target dimension not discovered")
        return ApplicabilityResult(
            True,
            f"possible in principle (needs >= {MIN_TRAINING_PAIRS} training pairs, a disjoint val "
            f"split, teacher inference access, and a {LOSS_OBJECTIVE}); NOT IMPLEMENTED in this build",
        )

    def prepare(self, sample_vectors: np.ndarray, context) -> None:
        raise NotImplementedError(
            f"knowledge_distillation requires a real teacher-student training pipeline "
            f"(>= {MIN_TRAINING_PAIRS} pairs, disjoint val split, {LOSS_OBJECTIVE}). Honest "
            f"extension point — implement a real trainer before selecting this strategy."
        )

    async def transform(self, vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError("knowledge_distillation.prepare() was never able to succeed")
