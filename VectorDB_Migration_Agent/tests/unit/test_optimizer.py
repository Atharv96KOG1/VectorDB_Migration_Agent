from __future__ import annotations

from core.models.migration_plan import OptimizationWeights
from core.optimizer.scorer import (
    CandidateMetrics,
    build_adaptive_retrieval_config,
    compute_confidence_score,
    normalize_metric,
    score_candidates,
)


def test_normalize_metric_maximize():
    assert normalize_metric([0.0, 5.0, 10.0], "maximize") == [0.0, 0.5, 1.0]


def test_normalize_metric_minimize():
    assert normalize_metric([0.0, 5.0, 10.0], "minimize") == [1.0, 0.5, 0.0]


def test_normalize_metric_constant_values_treated_as_tied_best():
    assert normalize_metric([3.0, 3.0, 3.0], "maximize") == [1.0, 1.0, 1.0]


def test_normalize_metric_empty_list():
    assert normalize_metric([], "maximize") == []


def test_score_candidates_excludes_failed_gate_entirely():
    # V3 fix: never let a failing candidate win by cost. It must not even appear in the
    # ranking, no matter how cheap it is.
    candidates = [
        CandidateMetrics(
            benchmark_id="b1", strategy="pca", passed_quality_gate=False, quality=0.5, cost=0.0
        ),
        CandidateMetrics(
            benchmark_id="b2",
            strategy="re_embedding",
            passed_quality_gate=True,
            quality=0.99,
            cost=100.0,
        ),
    ]
    ranked = score_candidates(candidates, OptimizationWeights())
    assert len(ranked) == 1
    assert ranked[0]["benchmark_id"] == "b2"


def test_score_candidates_no_passing_candidates_returns_empty():
    candidates = [
        CandidateMetrics(benchmark_id="b1", strategy="pca", passed_quality_gate=False, quality=0.5),
    ]
    assert score_candidates(candidates, OptimizationWeights()) == []


def test_score_candidates_raw_magnitude_does_not_dominate_weighted_sum():
    # V3 fix #5: without min-max normalization, "cost" in dollars (e.g. 500) would swamp
    # "quality" in 0-1 even at a tiny cost_weight. With normalization, the higher-quality,
    # higher-cost candidate should still win under a quality-heavy weighting.
    candidates = [
        CandidateMetrics(
            benchmark_id="cheap", strategy="pca", passed_quality_gate=True, quality=0.990, cost=1.0
        ),
        CandidateMetrics(
            benchmark_id="better",
            strategy="re_embedding",
            passed_quality_gate=True,
            quality=0.999,
            cost=500.0,
        ),
    ]
    weights = OptimizationWeights(
        quality_weight=0.9,
        cost_weight=0.1,
        latency_weight=0.0,
        time_weight=0.0,
        storage_weight=0.0,
        risk_weight=0.0,
    )
    ranked = score_candidates(candidates, weights)
    assert ranked[0]["benchmark_id"] == "better"


def test_adaptive_retrieval_config_requires_both_conditions():
    assert (
        build_adaptive_retrieval_config(
            mrl_capable=False,
            target_supports_prefetch_rerank=True,
            shortlist_dimension=256,
            full_dimension=1536,
        )
        is None
    )
    assert (
        build_adaptive_retrieval_config(
            mrl_capable=True,
            target_supports_prefetch_rerank=False,
            shortlist_dimension=256,
            full_dimension=1536,
        )
        is None
    )

    config = build_adaptive_retrieval_config(
        mrl_capable=True,
        target_supports_prefetch_rerank=True,
        shortlist_dimension=256,
        full_dimension=1536,
    )
    assert config["enabled"] is True
    assert config["shortlist_dimension"] == 256
    assert config["full_dimension"] == 1536


def test_confidence_score_averages_only_measured_components():
    score = compute_confidence_score(
        benchmark_recall_at_10=0.9,
        benchmark_ndcg_at_10=1.0,
        benchmark_topk_overlap=0.8,
        verify_recall_at_10=None,  # not measured this run — must not count as 0
        integrity_within_tolerance=None,
    )
    assert abs(score - (0.9 + 1.0 + 0.8) / 3) < 1e-9


def test_confidence_score_includes_integrity_as_one_or_zero():
    passing = compute_confidence_score(0.9, 0.9, 0.9, integrity_within_tolerance=True)
    failing = compute_confidence_score(0.9, 0.9, 0.9, integrity_within_tolerance=False)
    assert passing > failing


def test_confidence_score_none_when_nothing_measured():
    assert compute_confidence_score(None, None, None) is None
