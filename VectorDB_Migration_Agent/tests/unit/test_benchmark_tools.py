"""Since every integration test now targets Qdrant (real ANN path,
_qdrant_temp_collection_topk), the brute-force fallback (Pinecone-as-target, or any
non-Qdrant target) is otherwise untested end to end — these tests cover it directly.
"""

from __future__ import annotations

import numpy as np

from core.compatibility.engine import analyze_compatibility
from core.models.canonical_ir import (
    DataType,
    Metric,
    NormalizationSpec,
    VectorFieldSpec,
    VectorKind,
    VectorSourceSpec,
    VectorTargetSpec,
)
from core.models.capability import Capability
from core.models.migration_plan import CandidateStatus, TransformCandidate, TransformStrategy
from tools.benchmark_tools import _brute_force_topk, _evaluate_candidate


def test_brute_force_topk_cosine_finds_self_as_nearest():
    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((20, 8)).astype(np.float32)
    corpus_ids = [f"id{i}" for i in range(20)]
    queries = corpus[:5]  # query vectors identical to some corpus points
    result = _brute_force_topk(queries, corpus, corpus_ids, top_k=3, metric="cosine")
    for i in range(5):
        assert result[i][0] == corpus_ids[i]


def test_brute_force_topk_euclidean_smaller_distance_ranks_higher():
    corpus = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]], dtype=np.float32)
    ids = ["a", "b", "c"]
    queries = np.array([[0.1, 0.1]], dtype=np.float32)
    result = _brute_force_topk(queries, corpus, ids, top_k=3, metric="euclidean")
    assert result[0][0] == "a"
    assert result[0][-1] == "c"


async def test_evaluate_candidate_falls_back_to_brute_force_without_a_qdrant_adapter():
    rng = np.random.default_rng(1)
    corpus = rng.standard_normal((15, 8)).astype(np.float32)
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus_ids = [f"id{i}" for i in range(15)]
    id_to_index = {cid: i for i, cid in enumerate(corpus_ids)}
    query_ids = corpus_ids[:4]

    compat = analyze_compatibility(
        source_dim=8,
        target_dim=8,
        source_dt=DataType.FLOAT32,
        target_dt=DataType.FLOAT32,
        source_metric=Metric.COSINE,
        target_metric=Metric.COSINE,
        source_kind=VectorKind.DENSE,
        target_kind=VectorKind.DENSE,
        source_quantized=Capability.FALSE,
        normalization=NormalizationSpec(status="detected", method="l2", confidence=0.99),
    )
    field = VectorFieldSpec(
        name="default",
        source=VectorSourceSpec(dimension=8, datatype=DataType.FLOAT32, metric=Metric.COSINE),
        target=VectorTargetSpec(dimension=8, datatype=DataType.FLOAT32, metric=Metric.COSINE),
    )
    candidate = TransformCandidate(
        strategy=TransformStrategy.DIRECT_COPY, status=CandidateStatus.POSSIBLE
    )
    # Source "returns itself" as the top match for every query — the honest ground truth
    # for an identity-transform candidate queried with its own corpus vectors.
    source_topk_by_query = {qid: [qid] for qid in query_ids}

    result = await _evaluate_candidate(
        candidate,
        field=field,
        compatibility=compat,
        corpus_ids=corpus_ids,
        corpus_vectors=corpus,
        documents=None,
        id_to_index=id_to_index,
        query_ids=query_ids,
        source_topk_by_query=source_topk_by_query,
        migration_id="test-mig",
        minimum_recall_at_10=0.5,
        minimum_ndcg_at_10=0.5,
        maximum_topk_overlap_drop=0.9,
        reembed_api_key=None,
        reembed_model="text-embedding-3-small",
        reembed_dimensions=None,
        qdrant_target_adapter=None,
        qdrant_lock=None,
    )

    assert result.notes == "brute-force exact search"
    assert result.recall_at_10 == 1.0  # direct copy of a query vector must rank itself #1
    assert result.strategy == "direct_copy"


async def test_evaluate_candidate_isolates_a_real_runtime_error_instead_of_raising():
    """A candidate that fails for a genuine reason (here: PCA mathematically requires
    n_samples >= n_components, and a small DRY_RUN sample can't support a large target
    dimension) must come back as a failed row, not propagate and take the whole
    concurrent asyncio.gather() of candidates down with it — see run_benchmarks."""
    rng = np.random.default_rng(2)
    small_corpus = rng.standard_normal((10, 32)).astype(np.float32)  # only 10 samples
    corpus_ids = [f"id{i}" for i in range(10)]
    id_to_index = {cid: i for i, cid in enumerate(corpus_ids)}
    query_ids = corpus_ids[:3]

    compat = analyze_compatibility(
        source_dim=32,
        target_dim=24,  # target > n_samples: PCA.fit must raise
        source_dt=DataType.FLOAT32,
        target_dt=DataType.FLOAT32,
        source_metric=Metric.COSINE,
        target_metric=Metric.COSINE,
        source_kind=VectorKind.DENSE,
        target_kind=VectorKind.DENSE,
        source_quantized=Capability.FALSE,
        normalization=NormalizationSpec(status="detected", method="l2", confidence=0.99),
    )
    field = VectorFieldSpec(
        name="default",
        source=VectorSourceSpec(dimension=32, datatype=DataType.FLOAT32, metric=Metric.COSINE),
        target=VectorTargetSpec(dimension=24, datatype=DataType.FLOAT32, metric=Metric.COSINE),
    )
    candidate = TransformCandidate(strategy=TransformStrategy.PCA, status=CandidateStatus.POSSIBLE)

    result = await _evaluate_candidate(
        candidate,
        field=field,
        compatibility=compat,
        corpus_ids=corpus_ids,
        corpus_vectors=small_corpus,
        documents=None,
        id_to_index=id_to_index,
        query_ids=query_ids,
        source_topk_by_query={qid: [qid] for qid in query_ids},
        migration_id="test-mig",
        minimum_recall_at_10=0.5,
        minimum_ndcg_at_10=0.5,
        maximum_topk_overlap_drop=0.9,
        reembed_api_key=None,
        reembed_model="text-embedding-3-small",
        reembed_dimensions=None,
        qdrant_target_adapter=None,
        qdrant_lock=None,
    )

    assert result.passed_quality_gate is False
    assert result.notes == "error during evaluation"
    assert result.gate_reasons and "ValueError" in result.gate_reasons[0]


async def test_evaluate_candidate_rejects_nan_output_before_any_retrieval_query():
    """The fidelity gate must catch a NaN/Inf transform output BEFORE it's used for a
    retrieval query — a corrupted vector still "works" as an argument to a similarity
    search, just meaninglessly, so this has to be checked explicitly, not inferred from
    a suspiciously bad recall score three steps later."""
    corpus = np.zeros((5, 4), dtype=np.float32)
    corpus[2, 1] = np.nan  # DirectCopyTransformer will pass this straight through
    corpus_ids = [f"id{i}" for i in range(5)]
    id_to_index = {cid: i for i, cid in enumerate(corpus_ids)}
    query_ids = corpus_ids[:2]

    compat = analyze_compatibility(
        source_dim=4,
        target_dim=4,
        source_dt=DataType.FLOAT32,
        target_dt=DataType.FLOAT32,
        source_metric=Metric.COSINE,
        target_metric=Metric.COSINE,
        source_kind=VectorKind.DENSE,
        target_kind=VectorKind.DENSE,
        source_quantized=Capability.FALSE,
        normalization=NormalizationSpec(status="detected", method="l2", confidence=0.99),
    )
    field = VectorFieldSpec(
        name="default",
        source=VectorSourceSpec(dimension=4, datatype=DataType.FLOAT32, metric=Metric.COSINE),
        target=VectorTargetSpec(dimension=4, datatype=DataType.FLOAT32, metric=Metric.COSINE),
    )
    candidate = TransformCandidate(strategy=TransformStrategy.DIRECT_COPY, status=CandidateStatus.POSSIBLE)

    result = await _evaluate_candidate(
        candidate,
        field=field,
        compatibility=compat,
        corpus_ids=corpus_ids,
        corpus_vectors=corpus,
        documents=None,
        id_to_index=id_to_index,
        query_ids=query_ids,
        source_topk_by_query={qid: [qid] for qid in query_ids},
        migration_id="test-mig",
        minimum_recall_at_10=0.5,
        minimum_ndcg_at_10=0.5,
        maximum_topk_overlap_drop=0.9,
        reembed_api_key=None,
        reembed_model="text-embedding-3-small",
        reembed_dimensions=None,
        qdrant_target_adapter=None,
        qdrant_lock=None,
    )

    assert result.passed_quality_gate is False
    assert result.notes == "fidelity check failed: NaN/Inf in transformed output"
    assert result.gate_reasons and "has_nan=True" in result.gate_reasons[0]


async def test_pca_result_carries_reconstruction_diagnostics_but_random_projection_does_not():
    rng = np.random.default_rng(3)
    corpus = rng.standard_normal((40, 16)).astype(np.float32)
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus_ids = [f"id{i}" for i in range(40)]
    id_to_index = {cid: i for i, cid in enumerate(corpus_ids)}
    query_ids = corpus_ids[:5]

    compat = analyze_compatibility(
        source_dim=16,
        target_dim=8,
        source_dt=DataType.FLOAT32,
        target_dt=DataType.FLOAT32,
        source_metric=Metric.COSINE,
        target_metric=Metric.COSINE,
        source_kind=VectorKind.DENSE,
        target_kind=VectorKind.DENSE,
        source_quantized=Capability.FALSE,
        normalization=NormalizationSpec(status="detected", method="l2", confidence=0.99),
    )
    field = VectorFieldSpec(
        name="default",
        source=VectorSourceSpec(dimension=16, datatype=DataType.FLOAT32, metric=Metric.COSINE),
        target=VectorTargetSpec(dimension=8, datatype=DataType.FLOAT32, metric=Metric.COSINE),
    )
    source_topk = {qid: [qid] for qid in query_ids}
    kwargs = dict(
        field=field,
        compatibility=compat,
        corpus_ids=corpus_ids,
        corpus_vectors=corpus,
        documents=None,
        id_to_index=id_to_index,
        query_ids=query_ids,
        source_topk_by_query=source_topk,
        migration_id="test-mig",
        minimum_recall_at_10=0.0,
        minimum_ndcg_at_10=0.0,
        maximum_topk_overlap_drop=1.0,
        reembed_api_key=None,
        reembed_model="text-embedding-3-small",
        reembed_dimensions=None,
        qdrant_target_adapter=None,
        qdrant_lock=None,
    )

    pca_result = await _evaluate_candidate(
        TransformCandidate(strategy=TransformStrategy.PCA, status=CandidateStatus.POSSIBLE), **kwargs
    )
    assert pca_result.reconstruction_l2_error is not None
    assert pca_result.reconstruction_l2_error >= 0.0
    assert pca_result.reconstruction_cosine_similarity is not None
    assert -1.0 <= pca_result.reconstruction_cosine_similarity <= 1.0

    rp_result = await _evaluate_candidate(
        TransformCandidate(strategy=TransformStrategy.RANDOM_PROJECTION, status=CandidateStatus.POSSIBLE), **kwargs
    )
    assert rp_result.reconstruction_l2_error is None
    assert rp_result.reconstruction_cosine_similarity is None
