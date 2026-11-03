"""Storage-based cost estimator for the optimizer's `cost` axis (no-regret addition,
2026-10-13).

Before this, `CandidateMetrics.cost` (core/optimizer/scorer.py) was always 0.0:
`BenchmarkResult.estimated_cost` (core/models/benchmark.py) was a real field that nothing
ever populated, so every candidate tied on cost and `cost_weight` was silently dead code
regardless of what an operator configured.

This estimates STORAGE cost only — vector_count x dimension x dtype_size x provider
$/GB-month. It deliberately does NOT model query/write-op pricing: that's priced
per-request in ways this system has no pre-migration visibility into, and a storage-only
number presented as "the real cost" would be exactly the kind of confident-but-wrong
estimate this project's capability model otherwise refuses to produce. Treat this as a
documented lower bound, not total cost of ownership.

`CostEstimator` is a `Protocol`, not a concrete dependency, specifically so a future
query-cost or live-pricing-API estimator is a new class behind the same call shape, not a
rewrite of every caller (see planning_tools.py / benchmark_tools.py wiring).

Pricing last verified: 2026-10-13, read off each provider's own public pricing page.
Like the Pinecone API-version pin this project already learned to treat as perishable
(see docs/ARCHITECTURE.md's infra-hardening section), `PROVIDER_GB_MONTH_USD` WILL go
stale — it is not a constant to trust indefinitely, it is a cache with a visible
timestamp.
"""

from __future__ import annotations

from typing import Protocol

from core.models.canonical_ir import DataType

DATATYPE_BYTES: dict[DataType, float] = {
    DataType.FLOAT32: 4.0,
    DataType.FLOAT16: 2.0,
    DataType.BFLOAT16: 2.0,
    DataType.INT8: 1.0,
    DataType.BINARY: 0.125,
}

# $/GB-month, storage only, approximate, provider-published. Last verified 2026-10-13.
PROVIDER_GB_MONTH_USD: dict[str, float] = {
    "pinecone": 0.33,
    "qdrant": 0.30,
    "milvus": 0.25,
    "weaviate": 0.33,
    "chroma": 0.25,
    "vespa": 0.30,
    "faiss": 0.0,  # self-hosted library — no managed storage fee to estimate
}


class CostEstimator(Protocol):
    def estimate(
        self,
        *,
        provider: str,
        dimension: int | None,
        datatype: DataType,
        vector_count: int | None,
    ) -> float | None: ...


class StorageCostEstimator:
    """Returns None — never 0.0 — when an input needed to compute a real number is
    missing (unknown dimension, unknown vector count, or a provider with no pricing
    entry). An unestimable cost must never silently look identical to a confirmed-free
    one; that's the same tri-state discipline `core/models/capability.py` already
    applies everywhere else in this project."""

    def __init__(self, pricing: dict[str, float] | None = None) -> None:
        self._pricing = pricing if pricing is not None else PROVIDER_GB_MONTH_USD

    def estimate(
        self,
        *,
        provider: str,
        dimension: int | None,
        datatype: DataType,
        vector_count: int | None,
    ) -> float | None:
        if dimension is None or vector_count is None:
            return None
        rate = self._pricing.get(provider.strip().lower())
        if rate is None:
            return None
        bytes_per_vector = DATATYPE_BYTES.get(datatype, DATATYPE_BYTES[DataType.FLOAT32])
        total_gb = (dimension * bytes_per_vector * vector_count) / (1024**3)
        return round(total_gb * rate, 6)
