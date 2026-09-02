"""VERIFY state (V2 §23-25, §40): integrity diff for direct copies, plus retrieval
equivalence against golden queries when supplied. Synthetic query generation (V2 §25)
needs an LLM call this build doesn't implement (no Anthropic key in this sandbox) — it's
an honest extension point via `resolve_validation_mode`, not a fabricated result: without
golden queries, retrieval equivalence is recorded UNMEASURED, never assumed passing.

Golden queries must include a precomputed `vector` (V2 §24's example query text alone
isn't actionable without access to the original embedding model, which may be unknown —
V2 §55). `expected_ids` recall against the TARGET is the strongest signal (ground truth,
not source-vs-target agreement); source-vs-target Top-K comparison is computed too when
useful for context.

Query vectors are in SOURCE space — querying the target directly with them only works
when source and target share a dimension (direct_copy). For any dimension-changing
strategy (pca, random_projection, mrl), the query vector must go through the SAME fitted
transform the migration itself used before it's valid to query the target with — a real
migration against live Pinecone -> Qdrant Cloud (2026-08) hit this directly: a 384D golden
query against a 32D PCA-reduced target 400'd with "Vector dimension error: expected dim:
32, got 384". Fixed by reusing the already-fitted transformer
(`tools.execution_tools._load_execution_transformer`) to transform the query vector
before the target-side query, same as MIGRATE does for the corpus itself.
"""

from __future__ import annotations

import asyncio

import numpy as np
from aetherion_sdk import tool

from core.benchmarking.engine import (
    QueryComparison,
    evaluate_retrieval_equivalence,
    passes_quality_gate,
)
from core.checkpointing.store import load_checkpoint, save_checkpoint
from core.compatibility.engine import CompatibilityReport
from core.models.canonical_ir import CanonicalVectorIR, DataType
from core.models.migration_plan import MigrationPlan, TransformStrategy
from core.validation.engine import integrity_diff, resolve_validation_mode
from tools._shared import CredentialResolutionError, build_adapter, require_env
from tools.execution_tools import _load_execution_transformer

DEFAULT_SAMPLE_SIZE = 100
TOP_K = 10
_GOLDEN_QUERY_CONCURRENCY = 8


def _expected_id_recall(expected_ids: list[str], topk_ids: list[str]) -> float | None:
    if not expected_ids:
        return None
    return len(set(expected_ids) & set(topk_ids)) / len(set(expected_ids))


async def _evaluate_golden_query(
    gq: dict,
    source_adapter,
    target_adapter,
    source_resource: str,
    target_resource: str,
    namespace: str | None,
    query_transformer,
    target_query_supported: bool,
) -> tuple[QueryComparison, float | None] | None:
    vector = gq.get("vector")
    if not vector or not target_query_supported:
        return None

    target_vector = vector
    if query_transformer is not None:
        transformed = await query_transformer.transform(np.array([vector], dtype=np.float32))
        target_vector = transformed[0].tolist()

    source_matches, target_matches = await asyncio.gather(
        source_adapter.query(source_resource, vector, TOP_K, namespace=namespace),
        target_adapter.query(target_resource, target_vector, TOP_K, namespace=namespace),
    )
    comparison = QueryComparison(
        query_id=gq.get("query", "unnamed"),
        source_topk_ids=[m.id for m in source_matches],
        target_topk_ids=[m.id for m in target_matches],
    )
    recall = _expected_id_recall(gq.get("expected_ids", []), [m.id for m in target_matches])
    return comparison, recall


@tool()
async def verify_migration(
    migration_id: str,
    golden_queries: list[dict] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")
    req = checkpoint["request"]
    ir = CanonicalVectorIR.model_validate(checkpoint["canonical_ir"])
    field = ir.vectors[0]
    plan = MigrationPlan.model_validate(checkpoint["plan"])
    compatibility = CompatibilityReport.model_validate(checkpoint["compatibility"])

    source_adapter = build_adapter(
        req["source_provider"], req["source_credential_ref"], req.get("source_endpoint_ref")
    )
    target_adapter = build_adapter(
        req["target_provider"], req["target_credential_ref"], req.get("target_endpoint_ref")
    )

    integrity = None
    try:
        if plan.selected_strategy == TransformStrategy.DIRECT_COPY:
            sample = await source_adapter.sample_vectors(
                req["source_resource"], sample_size, namespace=req.get("namespace")
            )
            sample = [r for r in sample if r.vector]
            ids = [r.id for r in sample]
            source_by_id = {r.id: r.vector for r in sample}
            target_records = await target_adapter.fetch_by_ids(
                req["target_resource"], ids, namespace=req.get("namespace")
            )
            target_by_id = {r.id: r.vector for r in target_records if r.vector}
            common_ids = [i for i in ids if i in target_by_id]

            if common_ids:
                source_arr = np.array([source_by_id[i] for i in common_ids], dtype=np.float32)
                target_arr = np.array([target_by_id[i] for i in common_ids], dtype=np.float32)
                target_dtype_value = checkpoint["discovery"]["target"]["resource_info"].get(
                    "datatype", "unknown"
                )
                integrity = integrity_diff(source_arr, target_arr, DataType(target_dtype_value))
                integrity["ids_checked"] = len(common_ids)
            integrity = integrity or {}
            integrity["ids_missing_in_target"] = len(ids) - len(common_ids)

        try:
            require_env("ANTHROPIC_API_KEY")
            llm_available = True
        except CredentialResolutionError:
            llm_available = False

        mode = resolve_validation_mode(
            golden_queries, llm_available, representative_documents_available=False
        )

        retrieval_metrics = None
        expected_recalls: list[float] = []
        target_query_supported = True
        if mode == "golden":
            query_transformer = None
            if plan.selected_strategy not in (TransformStrategy.DIRECT_COPY, None):
                try:
                    query_transformer = await _load_execution_transformer(
                        checkpoint, plan, field, compatibility
                    )
                except RuntimeError:
                    # e.g. re_embedding: a query VECTOR can't be pushed through it (it
                    # transforms document TEXT, not vectors) — querying the target with
                    # the raw source-space vector would just repeat the exact dimension
                    # mismatch this fix exists for. Skip target-side evaluation entirely
                    # for every golden query rather than send a doomed request.
                    target_query_supported = False

            semaphore = asyncio.Semaphore(_GOLDEN_QUERY_CONCURRENCY)

            async def _bounded(gq: dict):
                async with semaphore:
                    return await _evaluate_golden_query(
                        gq,
                        source_adapter,
                        target_adapter,
                        req["source_resource"],
                        req["target_resource"],
                        req.get("namespace"),
                        query_transformer,
                        target_query_supported,
                    )

            evaluated = await asyncio.gather(*[_bounded(gq) for gq in golden_queries])
            comparisons = [pair[0] for pair in evaluated if pair is not None]
            expected_recalls = [
                pair[1] for pair in evaluated if pair is not None and pair[1] is not None
            ]

            if comparisons:
                retrieval_metrics = evaluate_retrieval_equivalence(comparisons, k=TOP_K)
    finally:
        await source_adapter.close()
        await target_adapter.close()

    gate_passed = None
    gate_reasons: list[str] = []
    if retrieval_metrics is not None:
        gate_passed, gate_reasons = passes_quality_gate(
            retrieval_metrics,
            minimum_recall_at_10=ir.validation.minimum_recall_at_10,
            minimum_ndcg_at_10=ir.validation.minimum_ndcg_at_10,
            maximum_topk_overlap_drop=ir.validation.maximum_topk_overlap_drop,
        )

    note = None
    if mode == "unavailable":
        note = (
            "retrieval equivalence UNMEASURED: no golden queries supplied and synthetic query "
            "generation is not implemented in this build"
        )
    elif mode == "golden" and not target_query_supported:
        strategy_label = plan.selected_strategy.value if plan.selected_strategy else "the selected strategy"
        note = (
            f"retrieval equivalence UNMEASURED against the target: golden query vectors are in "
            f"source space and {strategy_label} has no vector-level transform this build can apply "
            f"to them (e.g. re_embedding transforms document text, not vectors)"
        )

    validation_result = {
        "mode": mode,
        "integrity": integrity,
        "retrieval_metrics": retrieval_metrics,
        "expected_id_recall_mean": (
            (sum(expected_recalls) / len(expected_recalls)) if expected_recalls else None
        ),
        "gate_passed": gate_passed,
        "gate_reasons": gate_reasons,
        "note": note,
    }

    checkpoint["validation"] = validation_result
    checkpoint["history"].append(
        {"state": "VERIFY", "detail": {"mode": mode, "gate_passed": gate_passed}}
    )
    save_checkpoint(migration_id, checkpoint)

    return {
        "migration_id": migration_id,
        "mode": mode,
        "integrity_within_tolerance": integrity.get("within_tolerance") if integrity else None,
        "gate_passed": gate_passed,
        "gate_reasons": gate_reasons,
    }
