"""Multi-objective candidate scorer (V2 §52) with the V3 fix #5 applied directly: every
raw metric is min-max normalized to [0,1] BEFORE weights are applied. Without this, the
raw metric with the largest magnitude (e.g. cost in dollars vs. quality in 0-1) dominates
the weighted sum regardless of its configured weight.

The mandatory-quality-gate rule (V2 §52: "never sacrifice a mandatory quality threshold
merely to reduce cost") is enforced structurally: score_candidates only ever scores
candidates that already passed the quality gate — a candidate that failed simply isn't in
the pool the optimizer can pick from, not a low-scoring option it might still win by cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.models.migration_plan import OptimizationWeights


@dataclass
class CandidateMetrics:
    benchmark_id: str
    strategy: str
    passed_quality_gate: bool
    quality: float  # maximize, e.g. recall_at_10
    cost: float = 0.0  # minimize
    latency_ms: float = 0.0  # minimize
    migration_time_s: float = 0.0  # minimize
    storage_bytes: float = 0.0  # minimize
    risk: float = 0.0  # minimize, 0-1


_NORMALIZE_TIE_EPSILON = 1e-4


def normalize_metric(values: list[float], direction: str) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo <= _NORMALIZE_TIE_EPSILON:
        # A raw-value gap this small (2026-09-01: exposed by random_projection becoming
        # an exact isometry on equal/expanding dimensions, tying direct_copy on quality
        # except for float32 matrix-multiply rounding noise) is not a real distinction —
        # min-max normalization would otherwise stretch it to the FULL [0, 1] range,
        # turning measurement noise into an artificial full-strength signal that then
        # decides the optimizer's pick. Exact equality (the old check) was too strict to
        # catch this; treat a near-tie as tied-best instead.
        return [1.0] * len(values)
    if direction == "maximize":
        return [(v - lo) / (hi - lo) for v in values]
    if direction == "minimize":
        return [(hi - v) / (hi - lo) for v in values]
    raise ValueError(f"direction must be 'maximize' or 'minimize', got {direction!r}")


def score_candidates(
    candidates: list[CandidateMetrics], weights: OptimizationWeights
) -> list[dict]:
    gated = [c for c in candidates if c.passed_quality_gate]
    if not gated:
        return []

    quality_n = normalize_metric([c.quality for c in gated], "maximize")
    cost_n = normalize_metric([c.cost for c in gated], "minimize")
    latency_n = normalize_metric([c.latency_ms for c in gated], "minimize")
    time_n = normalize_metric([c.migration_time_s for c in gated], "minimize")
    storage_n = normalize_metric([c.storage_bytes for c in gated], "minimize")
    risk_n = normalize_metric([c.risk for c in gated], "minimize")

    results = []
    for i, c in enumerate(gated):
        score = (
            weights.quality_weight * quality_n[i]
            + weights.cost_weight * cost_n[i]
            + weights.latency_weight * latency_n[i]
            + weights.time_weight * time_n[i]
            + weights.storage_weight * storage_n[i]
            + weights.risk_weight * risk_n[i]
        )
        results.append({"benchmark_id": c.benchmark_id, "strategy": c.strategy, "score": score})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def build_adaptive_retrieval_config(
    mrl_capable: bool,
    target_supports_prefetch_rerank: bool,
    shortlist_dimension: int,
    full_dimension: int,
) -> dict | None:
    """V3 §4: post-migration optimization, not a compatibility patch. Only offered when
    the source model is confirmed MRL-capable AND the target natively supports
    prefetch+rerank (e.g. Qdrant's query API) — never fabricated for a target that can't
    actually execute the two-stage query."""
    if not (mrl_capable and target_supports_prefetch_rerank):
        return None
    return {
        "enabled": True,
        "strategy": "mrl_prefetch_rerank",
        "shortlist_dimension": shortlist_dimension,
        "full_dimension": full_dimension,
        "note": "store both the truncated shortlist vector and the full vector; query "
        "with prefetch(shortlist) + rerank(full)",
    }


def compute_confidence_score(
    benchmark_recall_at_10: float | None,
    benchmark_ndcg_at_10: float | None,
    benchmark_topk_overlap: float | None,
    verify_recall_at_10: float | None = None,
    integrity_within_tolerance: bool | None = None,
) -> float | None:
    """A single rolled-up number for the audit report, deliberately built from ONLY
    already-measured real values — never invented, never estimated. Plain mean of
    whatever was actually measured this run; a component that was never computed (e.g.
    integrity_within_tolerance is None because the selected strategy wasn't direct_copy)
    is skipped, not defaulted to 0 or 1. Returns None only if literally nothing was
    measured — a migration that never benchmarked or verified anything has no confidence
    score to report, not a fabricated one."""
    components: list[float] = []
    for value in (benchmark_recall_at_10, benchmark_ndcg_at_10, benchmark_topk_overlap, verify_recall_at_10):
        if value is not None:
            components.append(value)
    if integrity_within_tolerance is not None:
        components.append(1.0 if integrity_within_tolerance else 0.0)
    if not components:
        return None
    return sum(components) / len(components)
