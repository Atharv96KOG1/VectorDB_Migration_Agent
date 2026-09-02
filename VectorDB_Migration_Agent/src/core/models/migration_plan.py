"""Candidate transformation strategies and the migration plan (V2 §11, §21-22, §51-52)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class TransformStrategy(str, Enum):
    DIRECT_COPY = "direct_copy"
    MRL = "mrl"
    MATRYOSHKA_ADAPTOR = "matryoshka_adaptor"
    LEARNED_PROJECTION = "learned_projection"
    RETRIEVAL_AWARE_PROJECTION = "retrieval_aware_projection"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    RE_EMBEDDING = "re_embedding"
    PCA = "pca"
    RANDOM_PROJECTION = "random_projection"
    VEC2VEC = "vec2vec"
    SMEC = "smec"
    RIDGE_MAPPING = "ridge_mapping"
    PROCRUSTES_MAPPING = "procrustes_mapping"
    PROCRUSTES_DIAG_MAPPING = "procrustes_diag_mapping"
    LOW_RANK_AFFINE_MAPPING = "low_rank_affine_mapping"
    RESIDUAL_MLP_MAPPING = "residual_mlp_mapping"


class CandidateStatus(str, Enum):
    UNKNOWN = "unknown"
    POSSIBLE = "possible"
    IMPOSSIBLE = "impossible"
    SELECTED = "selected"
    REJECTED = "rejected"


class TransformCandidate(BaseModel):
    strategy: TransformStrategy
    status: CandidateStatus = CandidateStatus.UNKNOWN
    reason: str = ""
    """Why this candidate is possible/impossible/unknown — e.g. "MRL: embedding model
    unknown, cannot confirm supported_dimensions" or "re_embedding: documents unavailable"."""
    is_stub: bool = False
    """True for strategies whose transformer raises NotImplementedError by design
    (matryoshka_adaptor, learned_projection, retrieval_aware_projection,
    knowledge_distillation, vec2vec in this build) — surfaced so the plan never silently
    implies a capability that doesn't exist yet."""
    benchmark_id: str | None = None


class OptimizationWeights(BaseModel):
    quality_weight: float = 0.50
    cost_weight: float = 0.15
    latency_weight: float = 0.15
    time_weight: float = 0.10
    storage_weight: float = 0.05
    risk_weight: float = 0.05


class MigrationPlan(BaseModel):
    plan_id: str
    migration_id: str
    candidates: list[TransformCandidate] = Field(default_factory=list)
    selected_strategy: TransformStrategy | None = None
    selected_benchmark_id: str | None = None
    """FK into the executed benchmark-results table. A plan is only valid for execution
    if this id resolves — enforced mechanically in tools/planning_tools.py, not merely
    documented (V3 §8 anti-hallucination guardrail)."""
    optimizer_score: float | None = None
    weights: OptimizationWeights = OptimizationWeights()
    rrf_recovery_applied: bool = False
    """V3 §5: set when a lossy dense candidate failed the quality gate and was rescued by
    fusing with a sparse/BM25 signal via Reciprocal Rank Fusion instead of escalating
    straight to re_embedding."""
    adaptive_retrieval_enabled: bool = False
    """V3 §4: set when source is confirmed MRL-capable and the target supports
    prefetch+rerank — stores both the truncated shortlist vector and the full vector."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    approved: bool = False
    approved_by: str | None = None
