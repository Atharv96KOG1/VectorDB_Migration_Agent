from __future__ import annotations

from core.benchmarking.engine import (
    QueryComparison,
    evaluate_retrieval_equivalence,
    ndcg_at_k,
    passes_quality_gate,
    rank_correlation,
    recall_at_k,
    reciprocal_rank,
    score_drift,
    topk_overlap,
)


def test_recall_at_k_perfect_match():
    assert recall_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0


def test_recall_at_k_partial_match():
    assert recall_at_k(["a", "b", "c", "d"], ["a", "x", "y", "z"], k=4) == 0.25


def test_recall_at_k_empty_source_is_zero_not_error():
    assert recall_at_k([], ["a"], k=5) == 0.0


def test_topk_overlap_is_jaccard_not_recall():
    # source has 2 items, target has 4 — recall could be 1.0 (both source items present)
    # but overlap must be penalized for the target's extra items.
    source = ["a", "b"]
    target = ["a", "b", "c", "d"]
    assert recall_at_k(source, target, k=4) == 1.0
    assert topk_overlap(source, target, k=4) == 0.5  # |{a,b}| / |{a,b,c,d}|


def test_ndcg_perfect_order_is_one():
    ranking = ["a", "b", "c"]
    assert abs(ndcg_at_k(ranking, ranking, k=3) - 1.0) < 1e-9


def test_ndcg_reversed_order_is_less_than_one():
    source = ["a", "b", "c"]
    reversed_target = ["c", "b", "a"]
    assert ndcg_at_k(source, reversed_target, k=3) < 1.0


def test_reciprocal_rank_is_one_over_position_of_first_hit():
    # target's first hit among the source's top-3 relevant set is "b", at position 2.
    assert reciprocal_rank(["a", "b", "c"], ["x", "b", "a"], k=3) == 0.5


def test_reciprocal_rank_is_zero_when_no_hit_within_k():
    assert reciprocal_rank(["a", "b"], ["x", "y"], k=2) == 0.0


def test_reciprocal_rank_is_zero_not_error_for_empty_source():
    assert reciprocal_rank([], ["a"], k=5) == 0.0


def test_rank_correlation_none_when_insufficient_overlap():
    assert rank_correlation(["a"], ["b"]) is None
    assert rank_correlation([], []) is None


def test_rank_correlation_perfect_when_identical_order():
    ranking = ["a", "b", "c", "d"]
    assert abs(rank_correlation(ranking, ranking) - 1.0) < 1e-9


def test_rank_correlation_always_within_valid_spearman_range():
    # Regression test: a real live benchmark run (2026-08) produced rank_correlation =
    # -1.13 for exactly this shape — few common items scattered at sparse, far-apart
    # positions within two longer Top-K lists. The old formula assumed common-item
    # positions were already a dense 1..n permutation; they're not (they're a sparse
    # subset of 0..9), which pushed the hand-rolled Spearman formula outside [-1, 1].
    source_topk = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    target_topk = ["x", "y", "j", "z", "w", "v", "u", "t", "a", "s"]
    result = rank_correlation(source_topk, target_topk)
    assert result is not None
    assert -1.0 <= result <= 1.0


def test_score_drift_averages_absolute_difference_over_common_ids():
    source_scores = {"a": 0.9, "b": 0.8, "c": 0.1}
    target_scores = {"a": 0.9, "b": 0.7}
    drift = score_drift(source_scores, target_scores)
    assert abs(drift - 0.05) < 1e-9  # only "a" and "b" are common; |0|+|0.1| / 2


def test_score_drift_none_when_no_common_ids():
    assert score_drift({"a": 1.0}, {"b": 1.0}) is None


def test_evaluate_retrieval_equivalence_aggregates_across_queries():
    comparisons = [
        QueryComparison(query_id="q1", source_topk_ids=["a", "b"], target_topk_ids=["a", "b"]),
        QueryComparison(query_id="q2", source_topk_ids=["c", "d"], target_topk_ids=["x", "y"]),
    ]
    metrics = evaluate_retrieval_equivalence(comparisons, k=2)
    assert metrics["sample_size"] == 2
    assert 0.0 <= metrics["recall_at_10"] <= 1.0
    assert metrics["recall_at_10"] == 0.5  # q1 perfect (1.0), q2 zero (0.0) -> mean 0.5


def test_passes_quality_gate_rejects_below_threshold():
    metrics = {"recall_at_10": 0.90, "ndcg_at_10": 0.99, "topk_overlap": 1.0}
    passed, reasons = passes_quality_gate(
        metrics, minimum_recall_at_10=0.99, minimum_ndcg_at_10=0.98, maximum_topk_overlap_drop=0.05
    )
    assert passed is False
    assert any("recall_at_10" in r for r in reasons)


def test_passes_quality_gate_accepts_when_all_thresholds_met():
    metrics = {"recall_at_10": 0.995, "ndcg_at_10": 0.99, "topk_overlap": 0.98}
    passed, reasons = passes_quality_gate(
        metrics, minimum_recall_at_10=0.99, minimum_ndcg_at_10=0.98, maximum_topk_overlap_drop=0.05
    )
    assert passed is True
    assert reasons == []


def test_passes_quality_gate_never_lets_cost_override_a_failed_threshold():
    # This function has no cost parameter at all, by design (V2 §52: never sacrifice a
    # mandatory quality threshold to reduce cost) — this test documents that invariant.
    metrics = {"recall_at_10": 0.5, "ndcg_at_10": 0.99, "topk_overlap": 1.0}
    passed, _ = passes_quality_gate(
        metrics, minimum_recall_at_10=0.99, minimum_ndcg_at_10=0.98, maximum_topk_overlap_drop=0.05
    )
    assert passed is False
