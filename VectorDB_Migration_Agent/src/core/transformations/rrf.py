"""Reciprocal Rank Fusion hybrid compensation (V3 §5). When a lossy dense candidate (PCA /
random projection / an eventual vec2vec) fails the quality gate, this is the cheap
recovery step tried before escalating to expensive re-embedding: fuse the lossy dense
ranking with a cheap sparse/BM25 ranking. Pure rank-fusion arithmetic, no ML, no training
— real and unit-testable directly.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """rankings: one ranked id-list per source (e.g. [dense_topk_ids, sparse_topk_ids]).
    Returns (id, fused_score) sorted by fused_score descending. score(d) = sum_i w_i / (k + rank_i(d))
    for every ranking that contains d; a doc missing from one ranking simply doesn't get
    that term (V2 §5's "RRF" recovery is defined this way — no penalty for absence, only
    reward for being ranked)."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have the same length as rankings")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def apply_rrf_recovery(
    dense_ranking: list[str], sparse_ranking: list[str], top_k: int, k: int = 60
) -> list[str]:
    """The V3 §5 recovery step: fuse a failing dense candidate's ranking with a sparse/BM25
    ranking for the same query, return the fused top_k id list. Caller re-evaluates this
    fused ranking against the quality gate exactly like any other candidate — the fusion
    itself is not assumed to pass, it's just cheap to try before re_embedding."""
    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=k)
    return [doc_id for doc_id, _ in fused[:top_k]]
