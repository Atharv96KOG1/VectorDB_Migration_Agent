"""DRY_RUN state: pull a representative sample from the source ONLY (V2 §42: "no
production data should be modified during planning") and cache it in the checkpoint for
the BENCHMARK state to consume. No target writes happen here.
"""

from __future__ import annotations

from aetherion_sdk import tool

from core.checkpointing.store import load_checkpoint, save_checkpoint
from tools._shared import build_adapter

_DEFAULT_BENCHMARK_SAMPLE_SIZE = 400  # fallback when the source's real count is unknown
_MAX_BENCHMARK_SAMPLE_SIZE = 5000  # ceiling so a huge production source isn't pulled whole


@tool()
async def prepare_benchmark_sample(
    migration_id: str, sample_size: int | None = None, query_count: int = 40
) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")
    req = checkpoint["request"]

    if sample_size is None:
        # Full-corpus coverage whenever the source's real count is known and modest —
        # a real bug found live, 2026-09-11: a fixed 400-vector cap on a real 900-point
        # Qdrant source made every candidate (including a byte-identical direct_copy)
        # measure recall@10=0.4725, purely because some queries' true nearest neighbors
        # fell outside the sampled 400 — not a real transform or migration defect
        # (confirmed: the identical source/target pair scored a perfect 1.0 once the
        # sample covered all 900 points). Falls back to the previous fixed default only
        # when the count is undiscovered; caps at _MAX_BENCHMARK_SAMPLE_SIZE so a huge
        # production source still gets a bounded sample, not pulled in whole.
        approximate_count = (
            checkpoint.get("discovery", {})
            .get("source", {})
            .get("resource_info", {})
            .get("approximate_count")
        )
        sample_size = (
            min(approximate_count, _MAX_BENCHMARK_SAMPLE_SIZE)
            if approximate_count
            else _DEFAULT_BENCHMARK_SAMPLE_SIZE
        )

    adapter = build_adapter(
        req["source_provider"], req["source_credential_ref"], req.get("source_endpoint_ref")
    )
    try:
        sample = await adapter.sample_vectors(
            req["source_resource"], sample_size, namespace=req.get("namespace")
        )
    finally:
        await adapter.close()

    records = [r for r in sample if r.vector]
    query_count = min(query_count, len(records))

    # document_field (V2 §18/§25): naming a payload key that holds the source text lets
    # re_embedding and the calibration-pair mappings (ridge_mapping/procrustes_mapping,
    # core/transformations/linear_mapping.py) be benchmarked for real instead of always
    # failing with "no documents" — documents[i] stays aligned with vectors[i] since both
    # come from the same `records` list in the same order.
    document_field = req.get("document_field")
    documents = (
        [r.payload.get(document_field, "") for r in records] if document_field else None
    )

    checkpoint["benchmark_sample"] = {
        "ids": [r.id for r in records],
        "vectors": [r.vector for r in records],
        "query_ids": [r.id for r in records[:query_count]],
        "documents": documents,
    }
    checkpoint["history"].append(
        {"state": "DRY_RUN", "detail": {"sample_size": len(records), "query_count": query_count}}
    )
    save_checkpoint(migration_id, checkpoint)

    return {"migration_id": migration_id, "sample_size": len(records), "query_count": query_count}
