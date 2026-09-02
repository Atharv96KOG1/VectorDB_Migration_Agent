"""Retrieval-equivalence metrics (V2 §22-23) and the quality gate (V2 §26). Pure
functions over already-fetched Top-K id/score lists — the IO (running queries against
source/target adapters) happens in tools/benchmark_tools.py; this module is the part
that's cheaply unit-testable with synthetic id lists, no adapters or network needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import spearmanr


@dataclass
class QueryComparison:
    query_id: str
    source_topk_ids: list[str]
    target_topk_ids: list[str]
    source_scores: dict[str, float] | None = None
    target_scores: dict[str, float] | None = None


def recall_at_k(source_topk: list[str], target_topk: list[str], k: int) -> float:
    source_set = set(source_topk[:k])
    if not source_set:
        return 0.0
    target_set = set(target_topk[:k])
    return len(source_set & target_set) / len(source_set)


def topk_overlap(source_topk: list[str], target_topk: list[str], k: int) -> float:
    """Jaccard similarity of the two Top-K sets — distinct from recall_at_k (which is
    intersection-over-source-size): overlap penalizes the target for surfacing
    *extra* items the source didn't rank highly too, recall doesn't."""
    a, b = set(source_topk[:k]), set(target_topk[:k])
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def ndcg_at_k(source_topk: list[str], target_topk: list[str], k: int) -> float:
    """Source's own Top-K ranking defines graded relevance (higher source rank = higher
    relevance); NDCG measures how well the target ranking reproduces that order."""
    source_k = source_topk[:k]
    relevance = {doc_id: (k - rank) for rank, doc_id in enumerate(source_k)}
    if not relevance:
        return 0.0

    def dcg(ids: list[str]) -> float:
        return sum(relevance.get(doc_id, 0) / math.log2(i + 2) for i, doc_id in enumerate(ids[:k]))

    ideal = dcg(source_k)
    if ideal == 0:
        return 0.0
    return dcg(target_topk) / ideal


def reciprocal_rank(source_topk: list[str], target_topk: list[str], k: int) -> float:
    """MRR contribution for one query: 1/rank of the target ranking's first hit among the
    source's Top-K (the source ranking defines relevance, same convention ndcg_at_k uses).
    0.0 when none of the source's Top-K items appear in the target's Top-K at all — not
    None, since "no hit within the cutoff" is itself a valid, meaningful MRR outcome."""
    relevant = set(source_topk[:k])
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(target_topk[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def rank_correlation(source_topk: list[str], target_topk: list[str]) -> float | None:
    """Spearman correlation over ids common to both rankings. None (not 0.0) when fewer
    than 2 ids overlap — there isn't enough signal to compute a correlation, and 0.0 would
    misleadingly read as 'no correlation' rather than 'unmeasurable'.

    Delegates to scipy, which re-ranks whatever's handed to it internally — a hand-rolled
    `1 - 6*sum(d^2)/(n*(n^2-1))` needs d to be differences of a dense 1..n permutation,
    which the common-items' *original list positions* are not (they're a sparse subset of
    0..9, e.g. positions {1, 7} for 2 common items out of a 10-item ranking); plugging
    sparse positions into that formula produced values outside the valid [-1, 1] range on
    real data (confirmed live, 2026-08: -1.13 for a genuinely weak random_projection
    candidate) — caught only because the raw values were captured and inspected, not just
    a threshold pass/fail."""
    source_rank = {doc_id: i for i, doc_id in enumerate(source_topk)}
    target_rank = {doc_id: i for i, doc_id in enumerate(target_topk)}
    common = [doc_id for doc_id in source_topk if doc_id in target_rank]
    n = len(common)
    if n < 2:
        return None
    source_positions = [source_rank[doc_id] for doc_id in common]
    target_positions = [target_rank[doc_id] for doc_id in common]
    correlation, _ = spearmanr(source_positions, target_positions)
    if correlation is None or math.isnan(correlation):
        return None
    return float(correlation)


def score_drift(source_scores: dict[str, float], target_scores: dict[str, float]) -> float | None:
    common = set(source_scores) & set(target_scores)
    if not common:
        return None
    return sum(abs(source_scores[i] - target_scores[i]) for i in common) / len(common)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval_equivalence(comparisons: list[QueryComparison], k: int = 10) -> dict:
    """Aggregates every metric across the full query set. Each field is a query-count
    average; rank_correlation and score_drift average only over queries where they were
    computable (None-valued queries are excluded, not treated as 0)."""
    recalls = [recall_at_k(c.source_topk_ids, c.target_topk_ids, k) for c in comparisons]
    ndcgs = [ndcg_at_k(c.source_topk_ids, c.target_topk_ids, k) for c in comparisons]
    overlaps = [topk_overlap(c.source_topk_ids, c.target_topk_ids, k) for c in comparisons]
    mrrs = [reciprocal_rank(c.source_topk_ids, c.target_topk_ids, k) for c in comparisons]

    correlations = [
        rc
        for c in comparisons
        if (rc := rank_correlation(c.source_topk_ids, c.target_topk_ids)) is not None
    ]

    drifts = [
        sd
        for c in comparisons
        if c.source_scores is not None
        and c.target_scores is not None
        and (sd := score_drift(c.source_scores, c.target_scores)) is not None
    ]

    return {
        "sample_size": len(comparisons),
        "recall_at_10": _mean(recalls),
        "ndcg_at_10": _mean(ndcgs),
        "topk_overlap": _mean(overlaps),
        "mrr": _mean(mrrs),
        "rank_correlation": _mean(correlations) if correlations else None,
        "score_drift": _mean(drifts) if drifts else None,
    }


def passes_quality_gate(
    metrics: dict,
    minimum_recall_at_10: float,
    minimum_ndcg_at_10: float,
    maximum_topk_overlap_drop: float,
    latency_increase_percent: float | None = None,
    maximum_latency_increase_percent: float | None = None,
) -> tuple[bool, list[str]]:
    """V2 §26: machine-evaluable PASS/FAIL against the configured semantic drift budget.
    Never sacrifice a mandatory threshold to reduce cost (V2 §52) — this function has no
    cost awareness at all, by design; cost only enters the optimizer's ranking of
    candidates that already passed here."""
    reasons: list[str] = []

    if metrics["recall_at_10"] < minimum_recall_at_10:
        reasons.append(
            f"recall_at_10 {metrics['recall_at_10']:.4f} < required {minimum_recall_at_10}"
        )
    if metrics["ndcg_at_10"] < minimum_ndcg_at_10:
        reasons.append(f"ndcg_at_10 {metrics['ndcg_at_10']:.4f} < required {minimum_ndcg_at_10}")

    overlap_drop = 1.0 - metrics["topk_overlap"]
    if overlap_drop > maximum_topk_overlap_drop:
        reasons.append(
            f"topk_overlap_drop {overlap_drop:.4f} > allowed {maximum_topk_overlap_drop}"
        )

    if (
        latency_increase_percent is not None
        and maximum_latency_increase_percent is not None
        and latency_increase_percent > maximum_latency_increase_percent
    ):
        reasons.append(
            f"latency increase {latency_increase_percent:.1f}% > allowed {maximum_latency_increase_percent}%"
        )

    return (len(reasons) == 0, reasons)
