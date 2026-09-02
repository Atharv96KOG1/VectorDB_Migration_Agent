"""Matryoshka Representation Learning (V2 §13). MRL is a property of the embedding
*model*, not a generic technique — an unknown model must yield MRL=UNKNOWN, never TRUE
(V2 §13's own explicit rule). `KNOWN_MRL_MODELS` is a small, best-effort, non-exhaustive
registry; anything not listed is UNKNOWN, not FALSE — absence from this dict is not
evidence the model lacks MRL support.
"""

from __future__ import annotations

import numpy as np

from core.transformations.base import (
    ApplicabilityInputs,
    ApplicabilityResult,
    RepresentationTransformer,
)

# model name -> officially documented/supported truncation dimensions
KNOWN_MRL_MODELS: dict[str, list[int]] = {
    "text-embedding-3-small": [512, 1536],
    "text-embedding-3-large": [256, 1024, 3072],
    "nomic-embed-text-v1.5": [64, 128, 256, 512, 768],
    "jina-embeddings-v3": [32, 64, 128, 256, 512, 768, 1024],
}


def detect_mrl_support(model_name: str) -> list[int] | None:
    """Returns the known supported truncation dimensions, or None if the model is not in
    the registry (caller must record this as Capability.UNKNOWN, not FALSE)."""
    return KNOWN_MRL_MODELS.get(model_name)


class MRLTransformer(RepresentationTransformer):
    strategy = "mrl"

    def __init__(self, target_dimension: int) -> None:
        self._target_dimension = target_dimension

    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        mrl = inputs.embedding.mrl
        if not mrl.supported.is_true:
            return ApplicabilityResult(
                False,
                f"embedding model MRL support is {mrl.supported.value}, not confirmed TRUE",
            )
        if self._target_dimension not in mrl.supported_dimensions:
            return ApplicabilityResult(
                False,
                f"{self._target_dimension}D is not in the model's documented supported_dimensions "
                f"{mrl.supported_dimensions}",
            )
        if inputs.source_dim is not None and self._target_dimension >= inputs.source_dim:
            return ApplicabilityResult(False, "MRL truncates; target must be < source dimension")
        return ApplicabilityResult(
            True, "confirmed MRL-capable model, target is a documented prefix length"
        )

    async def transform(self, vectors: np.ndarray) -> np.ndarray:
        truncated = vectors[:, : self._target_dimension].astype(np.float32)
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (truncated / norms).astype(np.float32)

    def provenance(self) -> dict:
        return {
            "strategy": self.strategy,
            "output_dimension": self._target_dimension,
            "parameters": {"renormalized": True},
        }
