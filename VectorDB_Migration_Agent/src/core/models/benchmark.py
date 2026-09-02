"""Benchmark results (V2 §22-23) — the single source of truth the planning guardrail and
the optimizer both read from. Every row here corresponds to one candidate actually
transformed and evaluated on a representative sample; nothing is estimated or assumed.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class BenchmarkResult(BaseModel):
    benchmark_id: str
    migration_id: str
    strategy: str
    sample_size: int

    recall_at_10: float
    ndcg_at_10: float
    topk_overlap: float
    mrr: float | None = None
    rank_correlation: float | None = None
    score_drift: float | None = None

    estimated_cost: float | None = None
    latency_ms_p95: float | None = None

    reconstruction_l2_error: float | None = None
    reconstruction_cosine_similarity: float | None = None
    """Secondary diagnostic (V2 §19/§40), only populated for transformers that support
    inverse_transform (currently PCA). Never gates pass/fail — reconstruction fidelity is
    explicitly NOT retrieval fidelity; this is informational context for a reviewer, not
    a substitute for recall_at_10/ndcg_at_10."""

    passed_quality_gate: bool = False
    gate_reasons: list[str] = Field(default_factory=list)

    is_synthetic_query_set: bool = False
    """True when no golden queries were available and the query set was either
    LLM-generated or fell back to a representative-document sample — must be disclosed,
    never presented with the same confidence as a golden-query result (V2 §25)."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: str = ""
