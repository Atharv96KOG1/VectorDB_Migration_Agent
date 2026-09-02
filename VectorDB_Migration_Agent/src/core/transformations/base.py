"""RepresentationTransformer interface (V2 §11).

All transformers operate on numpy arrays and are only ever invoked from inside `@tool()`
activities (core/transformations is never imported by src/agent/agent.py — see
tests/unit/test_agent_import_boundary.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from core.models.canonical_ir import EmbeddingProvenance
from core.models.capability import CapabilityReport


@dataclass
class TransformContext:
    seed: int = 42
    documents: list[str] | None = None
    mrl_supported_dimensions: list[int] | None = None
    sparse_scores: dict | None = None
    """Query-id -> {doc_id: bm25_score}, used by rrf.py's hybrid recovery step."""
    extra: dict = field(default_factory=dict)


@dataclass
class ApplicabilityInputs:
    source_dim: int | None
    target_dim: int | None
    embedding: EmbeddingProvenance
    capabilities: CapabilityReport
    context: TransformContext


@dataclass
class ApplicabilityResult:
    possible: bool
    reason: str


class RepresentationTransformer(ABC):
    strategy: str
    is_stub: bool = False

    @abstractmethod
    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        """Deterministic eligibility check — never "unknown maps to possible". If a fact
        needed to decide is UNKNOWN, the candidate must come back not-possible with a
        reason saying what's undiscovered, not silently assumed available."""

    def prepare(self, sample_vectors: np.ndarray, context: TransformContext) -> None:
        """Fit any parameters needed (PCA components, RP matrix, ...). Default: no-op,
        for transformers that need no fitting (DirectCopy, MRL truncation)."""
        return None

    @abstractmethod
    async def transform(self, vectors: np.ndarray) -> np.ndarray:
        """vectors: (N, source_dim) float32 -> (N, target_dim) float32. Declared async
        uniformly because every transformer runs inside an async `@tool()` activity and
        one family member (ReEmbeddingTransformer) needs real network IO here; CPU-only
        transformers simply don't await anything inside their implementation."""

    def provenance(self) -> dict:
        """Reproducibility record fragment (V2 §53): strategy params, seed, checksums."""
        return {"strategy": self.strategy}
