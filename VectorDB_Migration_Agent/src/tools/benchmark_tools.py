"""BENCHMARK state (V2 §22): evaluate every POSSIBLE, non-eliminated candidate on the
DRY_RUN sample and record a BenchmarkResult row for each — including stub strategies,
which get an honest "not implemented" row instead of silently disappearing from the table.

Ground truth (source Top-K) comes from a real query against the live source index — not
the sample alone — so recall is measured against what the source actually returns.

Candidate Top-K comes from one of two paths:
  - **Qdrant target**: a real, short-lived ephemeral Qdrant collection is created, the
    transformed sample is upserted into it, and Top-K comes from a genuine ANN query
    against it — this is V2 §22's "temporary target collection" literally, not a stand-in.
    Always cleaned up in a `finally` (core/adapters/qdrant_adapter.py:delete_collection).
  - **Pinecone target**: brute-force exact search over the transformed sample in-process.
    A real ephemeral Pinecone index is comparatively slow to create/tear down (serverless
    index creation is not instant) and not free, so this build doesn't spin one up per
    benchmark candidate; brute-force is the honest fallback here, not a universal default.

Both ground-truth query fetching and candidate evaluation run concurrently (bounded by a
semaphore) — independent work that was previously serialized for no reason.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import numpy as np
from aetherion_sdk import tool

from core.adapters.base import VectorRecord
from core.adapters.qdrant_adapter import QdrantAdapter
from core.benchmarking.engine import (
    QueryComparison,
    evaluate_retrieval_equivalence,
    passes_quality_gate,
)
from core.checkpointing.store import load_checkpoint, save_checkpoint
from core.compatibility.engine import CompatibilityReport
from core.models.benchmark import BenchmarkResult
from core.models.canonical_ir import CanonicalVectorIR
from core.models.migration_plan import CandidateStatus, MigrationPlan, TransformCandidate
from core.transformations.base import TransformContext
from core.transformations.dequantize import dequantize
from core.transformations.pca import PCATransformer
from core.validation.engine import distribution_stats, reconstruction_error
from tools._shared import CredentialResolutionError, build_adapter, require_env
from tools._transform_factory import build_transformer

TOP_K = 10
_SOURCE_QUERY_CONCURRENCY = 8
_CANDIDATE_CONCURRENCY = 4
_UPSERT_CHUNK = 500
# Explicit search-time ef for the ephemeral BENCHMARK-only Qdrant collection (see
# QdrantAdapter.query's hnsw_ef docstring) — Qdrant's default is tuned for large,
# well-populated collections and under-serves ANN accuracy on a few-hundred-point scratch
# collection, producing recall/ndcg misses that are pure search approximation, not real
# data loss (confirmed live, 2026-09-01: direct_copy, a byte-identical copy, still scored
# recall@10=0.895 with the default).
_BENCHMARK_HNSW_EF = 256


def _brute_force_topk(
    query_vecs: np.ndarray, corpus_vecs: np.ndarray, corpus_ids: list[str], top_k: int, metric: str
) -> list[list[str]]:
    if metric == "euclidean":
        dists = np.linalg.norm(corpus_vecs[None, :, :] - query_vecs[:, None, :], axis=2)
        order = np.argsort(dists, axis=1)
    else:
        if metric == "cosine":
            qn = query_vecs / np.clip(np.linalg.norm(query_vecs, axis=1, keepdims=True), 1e-9, None)
            cn = corpus_vecs / np.clip(
                np.linalg.norm(corpus_vecs, axis=1, keepdims=True), 1e-9, None
            )
            sims = qn @ cn.T
        else:  # dot / unknown falls back to raw dot product
            sims = query_vecs @ corpus_vecs.T
        order = np.argsort(-sims, axis=1)
    topk_idx = order[:, :top_k]
    return [[corpus_ids[i] for i in row] for row in topk_idx]


async def _fetch_source_topk(
    source_adapter,
    resource: str,
    namespace: str | None,
    query_ids: list[str],
    query_vectors: np.ndarray,
) -> dict[str, list[str]]:
    semaphore = asyncio.Semaphore(_SOURCE_QUERY_CONCURRENCY)

    async def _one(qid: str, qvec: np.ndarray) -> tuple[str, list[str]]:
        async with semaphore:
            matches = await source_adapter.query(
                resource, qvec.tolist(), TOP_K, namespace=namespace
            )
            return qid, [m.id for m in matches]

    pairs = await asyncio.gather(*[_one(qid, qvec) for qid, qvec in zip(query_ids, query_vectors)])
    return dict(pairs)


async def _qdrant_temp_collection_topk(
    adapter: QdrantAdapter,
    lock: asyncio.Lock,
    corpus_ids: list[str],
    transformed_corpus: np.ndarray,
    transformed_queries: np.ndarray,
    target_dim: int,
    target_metric,
    migration_id: str,
    strategy_name: str,
) -> list[list[str]]:
    """Real ANN evaluation (V2 §22, literally): stand up an ephemeral Qdrant collection
    on the same connection as the real target, upsert the transformed sample, query it
    for real, tear it down. Always cleaned up even if evaluation fails partway through.

    `adapter` is a single connection SHARED across every candidate's temp-collection
    lifecycle (built once in run_benchmarks, not per-candidate) — Qdrant's local/on-disk
    mode locks its storage path per open client, so opening a fresh client per candidate
    would deadlock the moment two candidates ran concurrently against a local target.
    `lock` serializes the create/upsert/query/delete sequence on that shared connection —
    qdrant_client's local-mode backend isn't documented as safe for concurrent structural
    operations (create/delete collection) from multiple interleaved coroutines, so this
    is a deliberate, cheap safety margin rather than an assumption. The transform step and
    the source ground-truth fetch still run fully concurrently across candidates; only
    this shared-connection segment is serialized.
    """
    temp_name = f"__migbench_{migration_id}_{strategy_name}_{uuid.uuid4().hex[:8]}"
    async with lock:
        try:
            await adapter.create_collection(temp_name, dimension=target_dim, metric=target_metric)
            records = [
                VectorRecord(id=corpus_ids[i], vector=transformed_corpus[i].tolist())
                for i in range(len(corpus_ids))
            ]
            for start in range(0, len(records), _UPSERT_CHUNK):
                await adapter.upsert_vectors(temp_name, records[start : start + _UPSERT_CHUNK])

            results: list[list[str]] = []
            for i in range(len(transformed_queries)):
                matches = await adapter.query(
                    temp_name,
                    transformed_queries[i].tolist(),
                    TOP_K,
                    hnsw_ef=_BENCHMARK_HNSW_EF,
                )
                results.append([m.id for m in matches])
            return results
        finally:
            try:
                await adapter.delete_collection(temp_name)
            except Exception:
                pass


async def _evaluate_candidate(
    candidate: TransformCandidate,
    *,
    field,
    compatibility: CompatibilityReport,
    corpus_ids: list[str],
    corpus_vectors: np.ndarray,
    documents: list[str] | None,
    id_to_index: dict[str, int],
    query_ids: list[str],
    source_topk_by_query: dict[str, list[str]],
    migration_id: str,
    minimum_recall_at_10: float,
    minimum_ndcg_at_10: float,
    maximum_topk_overlap_drop: float,
    reembed_api_key: str | None,
    reembed_model: str,
    reembed_dimensions: int | None,
    qdrant_target_adapter: QdrantAdapter | None,
    qdrant_lock: asyncio.Lock | None,
) -> BenchmarkResult:
    strategy = candidate.strategy
    benchmark_id = f"{migration_id}-{strategy.value}-{uuid.uuid4().hex[:8]}"
    transformer = build_transformer(
        strategy,
        field.target.dimension,
        compatibility=compatibility,
        reembed_api_key=reembed_api_key,
        reembed_model=reembed_model,
        reembed_dimensions=reembed_dimensions,
    )

    start = time.perf_counter()
    try:
        transformer.prepare(corpus_vectors, TransformContext(documents=documents))
        transformed_corpus = await transformer.transform(corpus_vectors)

        # Vector fidelity check (V2 §21/§40) — nothing upstream of this validated that
        # the transform actually produced usable numbers. A NaN/Inf output is caught
        # here, before it ever reaches a retrieval query, not discovered as a mysterious
        # zero-recall result three steps later.
        fidelity = distribution_stats(transformed_corpus)
        if fidelity["has_nan"] or fidelity["has_inf"]:
            return BenchmarkResult(
                benchmark_id=benchmark_id,
                migration_id=migration_id,
                strategy=strategy.value,
                sample_size=len(corpus_ids),
                recall_at_10=0.0,
                ndcg_at_10=0.0,
                topk_overlap=0.0,
                passed_quality_gate=False,
                gate_reasons=[
                    f"transformed output failed the vector fidelity check: "
                    f"has_nan={fidelity['has_nan']} has_inf={fidelity['has_inf']} — "
                    f"no retrieval evaluation was attempted against unusable numbers"
                ],
                notes="fidelity check failed: NaN/Inf in transformed output",
            )

        transformed_queries = transformed_corpus[[id_to_index[q] for q in query_ids]]

        if qdrant_target_adapter is not None:
            target_topk_lists = await _qdrant_temp_collection_topk(
                qdrant_target_adapter,
                qdrant_lock,
                corpus_ids,
                transformed_corpus,
                transformed_queries,
                field.target.dimension,
                field.target.metric,
                migration_id,
                strategy.value,
            )
        else:
            target_topk_lists = _brute_force_topk(
                transformed_queries,
                transformed_corpus,
                corpus_ids,
                TOP_K,
                field.target.metric.value,
            )
    except NotImplementedError as exc:
        return BenchmarkResult(
            benchmark_id=benchmark_id,
            migration_id=migration_id,
            strategy=strategy.value,
            sample_size=len(corpus_ids),
            recall_at_10=0.0,
            ndcg_at_10=0.0,
            topk_overlap=0.0,
            passed_quality_gate=False,
            gate_reasons=[f"not implemented: {exc}"],
            notes="stub transformer, not executed",
        )
    except Exception as exc:
        # A candidate that fails for a real reason (e.g. PCA needs n_samples >=
        # n_components, and a small DRY_RUN sample can't support a large target
        # dimension) must not take the whole concurrent asyncio.gather down with it —
        # every other candidate still deserves its own verdict. Recorded as an honest
        # failed row, never silently dropped from the benchmark table.
        return BenchmarkResult(
            benchmark_id=benchmark_id,
            migration_id=migration_id,
            strategy=strategy.value,
            sample_size=len(corpus_ids),
            recall_at_10=0.0,
            ndcg_at_10=0.0,
            topk_overlap=0.0,
            passed_quality_gate=False,
            gate_reasons=[f"{type(exc).__name__}: {exc}"],
            notes="error during evaluation",
        )
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Secondary diagnostic only — never gates pass/fail (V2 §19/§40: reconstruction
    # fidelity is explicitly NOT retrieval fidelity). Only PCA currently exposes an
    # inverse_transform; other real transformers (random_projection, direct_copy) have
    # no meaningful reconstruction to compute and are left None, not faked.
    reconstruction_l2_error = None
    reconstruction_cosine_similarity = None
    if isinstance(transformer, PCATransformer):
        try:
            reconstructed = transformer.inverse_transform(transformed_corpus)
            recon = reconstruction_error(corpus_vectors, reconstructed)
            reconstruction_l2_error = recon["mean_l2_error"]
            reconstruction_cosine_similarity = recon["mean_cosine_similarity"]
        except Exception:
            pass

    comparisons = [
        QueryComparison(
            query_id=qid,
            source_topk_ids=source_topk_by_query.get(qid, []),
            target_topk_ids=target_topk_lists[i],
        )
        for i, qid in enumerate(query_ids)
    ]
    metrics = evaluate_retrieval_equivalence(comparisons, k=TOP_K)
    passed, reasons = passes_quality_gate(
        metrics,
        minimum_recall_at_10=minimum_recall_at_10,
        minimum_ndcg_at_10=minimum_ndcg_at_10,
        maximum_topk_overlap_drop=maximum_topk_overlap_drop,
    )

    return BenchmarkResult(
        benchmark_id=benchmark_id,
        migration_id=migration_id,
        strategy=strategy.value,
        sample_size=len(corpus_ids),
        recall_at_10=metrics["recall_at_10"],
        ndcg_at_10=metrics["ndcg_at_10"],
        topk_overlap=metrics["topk_overlap"],
        mrr=metrics["mrr"],
        rank_correlation=metrics["rank_correlation"],
        latency_ms_p95=elapsed_ms,
        reconstruction_l2_error=reconstruction_l2_error,
        reconstruction_cosine_similarity=reconstruction_cosine_similarity,
        passed_quality_gate=passed,
        gate_reasons=reasons,
        notes=(
            "real ANN eval (ephemeral Qdrant collection)"
            if qdrant_target_adapter is not None
            else "brute-force exact search"
        ),
    )


@tool()
async def run_benchmarks(migration_id: str) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")
    req = checkpoint["request"]
    ir = CanonicalVectorIR.model_validate(checkpoint["canonical_ir"])
    field = ir.vectors[0]
    compatibility = CompatibilityReport.model_validate(checkpoint["compatibility"])
    plan = MigrationPlan.model_validate(checkpoint["plan"])
    sample = checkpoint.get("benchmark_sample")
    if not sample:
        raise RuntimeError(
            "no benchmark sample found; run prepare_benchmark_sample (DRY_RUN) first"
        )

    corpus_ids: list[str] = sample["ids"]
    corpus_vectors = np.array(sample["vectors"], dtype=np.float32)
    if field.source.quantized.is_true:
        corpus_vectors = dequantize(corpus_vectors, field.source.datatype)
    query_ids: list[str] = sample["query_ids"]
    documents: list[str] | None = sample.get("documents")
    id_to_index = {doc_id: i for i, doc_id in enumerate(corpus_ids)}
    query_vectors = corpus_vectors[[id_to_index[q] for q in query_ids]]

    source_adapter = build_adapter(
        req["source_provider"], req["source_credential_ref"], req.get("source_endpoint_ref")
    )
    try:
        source_topk_by_query = await _fetch_source_topk(
            source_adapter, req["source_resource"], req.get("namespace"), query_ids, query_vectors
        )
    finally:
        await source_adapter.close()

    try:
        reembed_api_key = require_env("OPENAI_API_KEY")
    except CredentialResolutionError:
        reembed_api_key = None
    reembed_model = req.get("reembed_model", "text-embedding-3-small")
    reembed_dimensions = req.get("reembed_dimensions")

    use_real_ann = req["target_provider"] == "qdrant"
    candidates_to_run = [c for c in plan.candidates if c.status == CandidateStatus.POSSIBLE]

    # One shared connection + lock for every candidate's temp-collection lifecycle — see
    # _qdrant_temp_collection_topk's docstring for why this must not be one-adapter-per-candidate.
    qdrant_target_adapter: QdrantAdapter | None = None
    qdrant_lock: asyncio.Lock | None = None
    if use_real_ann:
        adapter = build_adapter(
            req["target_provider"], req["target_credential_ref"], req.get("target_endpoint_ref")
        )
        if not isinstance(adapter, QdrantAdapter):
            raise TypeError(
                "target_provider is 'qdrant' but build_adapter did not return a QdrantAdapter"
            )
        qdrant_target_adapter = adapter
        qdrant_lock = asyncio.Lock()

    try:
        semaphore = asyncio.Semaphore(_CANDIDATE_CONCURRENCY)

        async def _bounded(candidate: TransformCandidate) -> BenchmarkResult:
            async with semaphore:
                return await _evaluate_candidate(
                    candidate,
                    field=field,
                    compatibility=compatibility,
                    corpus_ids=corpus_ids,
                    corpus_vectors=corpus_vectors,
                    documents=documents,
                    id_to_index=id_to_index,
                    query_ids=query_ids,
                    source_topk_by_query=source_topk_by_query,
                    migration_id=migration_id,
                    minimum_recall_at_10=ir.validation.minimum_recall_at_10,
                    minimum_ndcg_at_10=ir.validation.minimum_ndcg_at_10,
                    maximum_topk_overlap_drop=ir.validation.maximum_topk_overlap_drop,
                    reembed_api_key=reembed_api_key,
                    reembed_model=reembed_model,
                    reembed_dimensions=reembed_dimensions,
                    qdrant_target_adapter=qdrant_target_adapter,
                    qdrant_lock=qdrant_lock,
                )

        results: list[BenchmarkResult] = await asyncio.gather(
            *[_bounded(c) for c in candidates_to_run]
        )
    finally:
        if qdrant_target_adapter is not None:
            await qdrant_target_adapter.close()

    checkpoint["benchmark_results"] = [r.model_dump(mode="json") for r in results]
    checkpoint["history"].append(
        {
            "state": "BENCHMARK",
            "detail": {
                "evaluated": len(results),
                "passed": sum(1 for r in results if r.passed_quality_gate),
                "used_real_ann": use_real_ann,
            },
        }
    )
    save_checkpoint(migration_id, checkpoint)

    return {
        "migration_id": migration_id,
        "used_real_ann": use_real_ann,
        "results": [
            {
                "strategy": r.strategy,
                "benchmark_id": r.benchmark_id,
                "sample_size": r.sample_size,
                "recall_at_10": r.recall_at_10,
                "ndcg_at_10": r.ndcg_at_10,
                "passed": r.passed_quality_gate,
            }
            for r in results
        ],
    }
