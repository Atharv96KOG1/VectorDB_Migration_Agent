"""Migration provenance (V2 §53) — reproducibility record, distinct from the resumability
checkpoint (core/checkpointing/store.py). Written once, at COMPLETE.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class TransformationProvenance(BaseModel):
    strategy: str
    input_dimension: int | None = None
    output_dimension: int | None = None
    parameters: dict = Field(default_factory=dict)
    """e.g. {"sample_size": 100000, "seed": 42} for PCA/RandomProjection."""
    artifact_id: str | None = None
    artifact_checksum: str | None = None
    """sha256 of the fitted transform's parameters (e.g. PCA components, RP matrix) —
    lets a re-run be verified as bit-identical, not merely "the same strategy"."""
    source_semantic_space_id: str = "unknown/unknown"
    """core.models.canonical_ir.EmbeddingProvenance.semantic_space_id at migration time —
    lets a LATER migration into this same target (compatibility_tools.run_compatibility_check)
    recognize what it wrote, instead of treating a resumed/re-targeted migration's own
    prior data as an unverifiable foreign semantic space."""


class Vec2VecProvenance(BaseModel):
    """V3 §3 mandatory disclosure fields — only populated if vec2vec is ever selected
    (it is not implemented in this build; present for forward-compatibility and so the
    audit report schema already carries the disclosure obligation)."""

    teacher_space: str = ""
    student_space: str = ""
    unpaired_corpus_size: int = 0
    adversarial_training_config: dict = Field(default_factory=dict)
    generalization_test_set: str = ""
    inversion_risk_disclosed: bool = True
    disclosure_text: str = (
        "This migration used vec2vec-style unsupervised embedding translation. "
        "Published research (Jha, Zhang et al., NeurIPS 2025, arXiv:2505.12540) shows "
        "translated embeddings can leak enough signal to reconstruct sensitive "
        "attributes of the source documents. The trained translator must be handled "
        "with credential-tier custody."
    )


class ProvenanceRecord(BaseModel):
    migration_id: str
    source_provider: str
    source_resource: str
    target_provider: str
    target_resource: str

    transformation: TransformationProvenance
    vec2vec: Vec2VecProvenance | None = None

    selected_benchmark_id: str
    quality_gate_passed: bool
    confidence_score: float | None = None
    """Rolled-up single number derived ONLY from already-measured real values (recall,
    ndcg, topk_overlap, integrity check) — never invented. See core/optimizer/scorer.py:
    compute_confidence_score. None if verify never ran or nothing was measurable."""

    code_version: str = "0.1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
