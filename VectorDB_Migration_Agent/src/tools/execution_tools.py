"""MIGRATE state (V2 §37-39): batched scan -> transform -> idempotent upsert. The workflow
calls `migrate_batch` in a loop until `done` is true (src/agent/agent.py); each call
processes up to `max_batches` pages internally, double-buffered — while one batch is
being transformed and written, the *next* batch's scan is already in flight on the source
connection (asyncio.create_task, awaited just before it's needed). This overlaps source
IO with transform compute and target IO instead of running everything fully sequentially,
and it also means a large migration needs far fewer Temporal activity round-trips than
one-batch-per-call would. Each inner batch still checkpoints before moving to the next,
so a crash mid-call loses at most one in-flight batch's progress — and since upsert is
idempotent, replaying that batch on resume is free, not just safe.

Idempotency (V2 §39) is structural, not incidental: both shipped adapters' `upsert`
overwrites by id, so a Temporal-level retry of this exact activity re-writes the same
records rather than duplicating them — independent of the JSON checkpoint below, which
exists only for cross-run resume and audit history.
"""

from __future__ import annotations

import asyncio

import numpy as np
from aetherion_sdk import tool

from core.adapters.base import VectorRecord
from core.checkpointing.store import load_checkpoint, save_checkpoint
from core.compatibility.engine import CompatibilityReport
from core.models.canonical_ir import CanonicalVectorIR, Metric, NormalizationSpec
from core.models.migration_plan import MigrationPlan, TransformStrategy
from core.transformations.base import TransformContext
from core.transformations.dequantize import dequantize
from core.transformations.direct import DirectCopyTransformer
from core.transformations.linear_mapping import (
    LowRankAffineMappingTransformer,
    OrthogonalProcrustesTransformer,
    ProcrustesDiagMappingTransformer,
    RidgeMappingTransformer,
)
from core.transformations.mrl import MRLTransformer
from core.transformations.pca import PCATransformer
from core.transformations.random_projection import RandomProjectionTransformer
from core.transformations.reembedding import ReEmbeddingTransformer
from tools._shared import CredentialResolutionError, build_adapter, require_env
from tools._transform_factory import _resolve_reembed_dimensions

DEFAULT_BATCH_SIZE = 200


def maybe_renormalize(
    vectors: np.ndarray, normalization: NormalizationSpec, target_metric: Metric
) -> np.ndarray:
    """V3 fix #8: normalization UNKNOWN + target metric cosine -> re-normalize (L2) at
    write time. A no-op on vectors that are already unit-norm; prevents silent ranking
    corruption for vectors that weren't, which UNKNOWN could otherwise mask."""
    if normalization.status == "unknown" and target_metric == Metric.COSINE:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)
    return vectors


_DIMENSION_PROJECTION_STRATEGIES = (TransformStrategy.PCA, TransformStrategy.RANDOM_PROJECTION)
# residual_mlp_mapping deliberately excluded: honest stub (needs PyTorch, not added as a
# dependency), never reaches MIGRATE since it can never pass the quality gate.
_CALIBRATION_MAPPING_CLASSES = {
    TransformStrategy.RIDGE_MAPPING: RidgeMappingTransformer,
    TransformStrategy.PROCRUSTES_MAPPING: OrthogonalProcrustesTransformer,
    TransformStrategy.PROCRUSTES_DIAG_MAPPING: ProcrustesDiagMappingTransformer,
    TransformStrategy.LOW_RANK_AFFINE_MAPPING: LowRankAffineMappingTransformer,
}


def _fit_projection(
    checkpoint: dict, strategy: TransformStrategy, target_dim: int, source_quantized_dtype
) -> dict:
    sample = checkpoint.get("benchmark_sample")
    if not sample:
        raise RuntimeError(
            f"cannot execute {strategy.value}: no benchmark_sample cached — run DRY_RUN/BENCHMARK first"
        )
    sample_vectors = np.array(sample["vectors"], dtype=np.float32)
    cls = PCATransformer if strategy == TransformStrategy.PCA else RandomProjectionTransformer
    transformer = cls(target_dim)
    transformer.prepare(sample_vectors, TransformContext())
    return {"strategy": strategy.value, "params": transformer.fitted_params}


async def _fit_calibration_mapping(
    checkpoint: dict, strategy: TransformStrategy, target_dim: int
) -> dict:
    """Ridge/Procrustes fitting needs an async embeddings-API call for the calibration
    targets (core/transformations/linear_mapping.py), unlike PCA/RandomProjection's
    synchronous sklearn/numpy fit — so this is fit once here, exactly like
    `_fit_projection`, and its JSON-safe `fitted_params` persisted the same way."""
    sample = checkpoint.get("benchmark_sample")
    if not sample:
        raise RuntimeError(
            f"cannot execute {strategy.value}: no benchmark_sample cached — run DRY_RUN/BENCHMARK first"
        )
    documents = sample.get("documents")
    if not documents:
        raise RuntimeError(
            f"cannot execute {strategy.value}: no calibration documents cached — "
            f"request.document_field must be set before DRY_RUN"
        )
    try:
        api_key = require_env("OPENAI_API_KEY")
    except CredentialResolutionError as exc:
        raise RuntimeError(f"{strategy.value} requires OPENAI_API_KEY to be set") from exc

    req = checkpoint["request"]
    sample_vectors = np.array(sample["vectors"], dtype=np.float32)
    cls = _CALIBRATION_MAPPING_CLASSES[strategy]
    reembed_model = req.get("reembed_model", "text-embedding-3-small")
    transformer = cls(
        api_key=api_key,
        model=reembed_model,
        dimensions=_resolve_reembed_dimensions(
            reembed_model, req.get("reembed_dimensions"), target_dim
        ),
    )
    transformer.prepare(sample_vectors, TransformContext(documents=documents))
    await transformer.ensure_fitted()
    return {"strategy": strategy.value, "params": transformer.fitted_params}


async def _load_execution_transformer(
    checkpoint: dict, plan: MigrationPlan, field, compatibility: CompatibilityReport
):
    """Returns a ready-to-use transformer for strategies with a stable global fit
    (direct_copy/mrl need none; pca/random_projection/ridge_mapping/procrustes_mapping are
    fit ONCE against the benchmark sample and persisted — re-fitting per batch would land
    each batch in a different projected space and make the target collection numerically
    incoherent). Mutates `checkpoint["fitted_transform"]` in place the first time a
    projection is fit.
    """
    strategy = plan.selected_strategy
    target_dim = field.target.dimension

    if strategy == TransformStrategy.DIRECT_COPY:
        return DirectCopyTransformer(compatibility)
    if strategy == TransformStrategy.MRL:
        return MRLTransformer(target_dim)

    if strategy in _DIMENSION_PROJECTION_STRATEGIES:
        fitted = checkpoint.get("fitted_transform")
        cls = PCATransformer if strategy == TransformStrategy.PCA else RandomProjectionTransformer
        if fitted and fitted.get("strategy") == strategy.value:
            return cls.from_fitted(fitted["params"], target_dim)
        fitted = _fit_projection(checkpoint, strategy, target_dim, field.source.datatype)
        checkpoint["fitted_transform"] = fitted
        return cls.from_fitted(fitted["params"], target_dim)

    if strategy in _CALIBRATION_MAPPING_CLASSES:
        fitted = checkpoint.get("fitted_transform")
        cls = _CALIBRATION_MAPPING_CLASSES[strategy]
        if fitted and fitted.get("strategy") == strategy.value:
            return cls.from_fitted(fitted["params"])
        fitted = await _fit_calibration_mapping(checkpoint, strategy, target_dim)
        checkpoint["fitted_transform"] = fitted
        return cls.from_fitted(fitted["params"])

    raise RuntimeError(
        f"strategy {strategy.value} cannot be executed at MIGRATE time in this build "
        f"(stub, or requires per-batch document input handled separately)"
    )


async def _apply_id_collision_policy(
    target_adapter, resource: str, namespace: str | None, records: list[VectorRecord], policy: str
) -> list[VectorRecord]:
    if policy == "overwrite":
        return records
    ids = [r.id for r in records]
    existing = await target_adapter.fetch_by_ids(resource, ids, namespace=namespace)
    existing_ids = {r.id for r in existing}
    if not existing_ids:
        return records
    if policy == "fail":
        raise RuntimeError(
            f"id_collision_policy=fail: {len(existing_ids)} id(s) already exist in target "
            f"(e.g. {sorted(existing_ids)[:5]})"
        )
    if policy == "skip":
        return [r for r in records if r.id not in existing_ids]
    if policy in ("rename", "prefix"):
        return [
            r.model_copy(update={"id": f"{r.id}__migrated"}) if r.id in existing_ids else r
            for r in records
        ]
    raise ValueError(f"unknown id_collision_policy: {policy!r}")


DEFAULT_MAX_BATCHES_PER_CALL = 5


@tool()
async def migrate_batch(
    migration_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: int = DEFAULT_MAX_BATCHES_PER_CALL,
) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")
    req = checkpoint["request"]
    ir = CanonicalVectorIR.model_validate(checkpoint["canonical_ir"])
    field = ir.vectors[0]
    plan = MigrationPlan.model_validate(checkpoint["plan"])
    if plan.selected_strategy is None:
        raise RuntimeError("no strategy selected; run select_strategy first")
    compatibility = CompatibilityReport.model_validate(checkpoint["compatibility"])

    migrate_state = checkpoint.get("migrate") or {
        "cursor": None,
        "vectors_read": 0,
        "vectors_written": 0,
        "done": False,
        "written_ids": [],
        # Every id this migration is known to have written to the target — the exact
        # set rollback_migration deletes on abort. Grows with migration size; fine at
        # this system's demonstrated scale, a real caveat at very large (100M+) scale
        # where a smarter approach (e.g. a separate id-log file) would be needed instead.
    }
    if migrate_state["done"]:
        return {"migration_id": migration_id, "batches_this_call": 0, **migrate_state}

    resource = req["source_resource"]
    target_resource = req["target_resource"]
    namespace = req.get("namespace")
    is_reembedding = plan.selected_strategy == TransformStrategy.RE_EMBEDDING

    source_adapter = build_adapter(
        req["source_provider"], req["source_credential_ref"], req.get("source_endpoint_ref")
    )
    target_adapter = build_adapter(
        req["target_provider"], req["target_credential_ref"], req.get("target_endpoint_ref")
    )

    shared_transformer = None
    if not is_reembedding:
        # Loaded/fit ONCE for the whole call (not per inner batch) — see
        # _load_execution_transformer's docstring on why PCA/RandomProjection must never
        # be re-fit per batch. Persist immediately so a newly-fit projection survives a
        # crash later in this same call.
        shared_transformer = await _load_execution_transformer(checkpoint, plan, field, compatibility)
        checkpoint["migrate"] = migrate_state
        save_checkpoint(migration_id, checkpoint)

    async def _scan(cursor: str | None):
        return await source_adapter.scan_vectors(
            resource, batch_size, cursor=cursor, namespace=namespace
        )

    batches_processed = 0
    try:
        next_page_task = asyncio.create_task(_scan(migrate_state["cursor"]))
        try:
            while batches_processed < max_batches and not migrate_state["done"]:
                page = await next_page_task
                records = [r for r in page.records if r.vector]

                if not records:
                    migrate_state["done"] = True
                    migrate_state["cursor"] = page.next_cursor
                    next_page_task = None
                    break

                # Kick off the NEXT scan now — before this batch's transform+write — so
                # source IO for batch N+1 overlaps with transform/target IO for batch N.
                next_page_task = (
                    asyncio.create_task(_scan(page.next_cursor))
                    if page.next_cursor is not None
                    else None
                )

                vectors = np.array([r.vector for r in records], dtype=np.float32)
                if field.source.quantized.is_true:
                    vectors = dequantize(vectors, field.source.datatype)

                if is_reembedding:
                    document_field = req.get("document_field")
                    if not document_field:
                        raise RuntimeError(
                            "re_embedding requires request.document_field naming the payload text field"
                        )
                    documents = [r.payload.get(document_field, "") for r in records]
                    try:
                        api_key = require_env("OPENAI_API_KEY")
                    except CredentialResolutionError as exc:
                        raise RuntimeError(
                            "re_embedding requires OPENAI_API_KEY to be set"
                        ) from exc
                    reembed_model = req.get("reembed_model", "text-embedding-3-small")
                    transformer = ReEmbeddingTransformer(
                        api_key=api_key,
                        model=reembed_model,
                        dimensions=_resolve_reembed_dimensions(
                            reembed_model, req.get("reembed_dimensions"), field.target.dimension
                        ),
                    )
                    transformer.prepare(vectors, TransformContext(documents=documents))
                else:
                    transformer = shared_transformer

                transformed = await transformer.transform(vectors)
                transformed = maybe_renormalize(
                    transformed, field.embedding.normalization, field.target.metric
                )

                out_records = [
                    VectorRecord(
                        id=r.id,
                        vector=transformed[i].tolist(),
                        payload=r.payload,
                        namespace=r.namespace,
                    )
                    for i, r in enumerate(records)
                ]
                out_records = await _apply_id_collision_policy(
                    target_adapter,
                    target_resource,
                    namespace,
                    out_records,
                    ir.id.collision_policy.value,
                )
                written = await target_adapter.upsert_vectors(
                    target_resource, out_records, namespace=namespace
                )

                migrate_state["vectors_read"] += len(records)
                migrate_state["vectors_written"] += written
                migrate_state["written_ids"].extend(r.id for r in out_records)
                migrate_state["cursor"] = page.next_cursor
                migrate_state["done"] = page.next_cursor is None
                batches_processed += 1

                checkpoint["migrate"] = migrate_state
                save_checkpoint(migration_id, checkpoint)
        finally:
            if next_page_task is not None and not next_page_task.done():
                next_page_task.cancel()
                try:
                    await next_page_task
                except (asyncio.CancelledError, Exception):
                    pass
    finally:
        await source_adapter.close()
        await target_adapter.close()

    checkpoint["migrate"] = migrate_state
    checkpoint["history"].append(
        {
            "state": "MIGRATE",
            "detail": {
                "vectors_written": migrate_state["vectors_written"],
                "done": migrate_state["done"],
                "batches_this_call": batches_processed,
            },
        }
    )
    save_checkpoint(migration_id, checkpoint)

    return {"migration_id": migration_id, "batches_this_call": batches_processed, **migrate_state}


@tool()
async def rollback_migration(migration_id: str) -> dict:
    """Deletes exactly the ids THIS migration wrote to the target (tracked in
    migrate_state["written_ids"]) — never a bulk/whole-resource clear, so a target that
    already held unrelated data before this migration started is untouched. Source is
    never written to by any part of this system, so there is nothing to roll back there.

    This is an explicit, separate action — not auto-invoked on FAILED. An operator calls
    this directly when they decide a migration should be rolled back, matching the same
    "confirm before anything destructive" posture as every other risky action here.
    """
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")
    req = checkpoint["request"]
    migrate_state = checkpoint.get("migrate") or {}
    written_ids: list[str] = migrate_state.get("written_ids", [])

    if not written_ids:
        return {
            "migration_id": migration_id,
            "deleted": 0,
            "note": "nothing recorded as written by this migration; nothing to roll back",
        }

    target_adapter = build_adapter(
        req["target_provider"], req["target_credential_ref"], req.get("target_endpoint_ref")
    )
    try:
        deleted = await target_adapter.delete_vectors(
            req["target_resource"], written_ids, namespace=req.get("namespace")
        )
    finally:
        await target_adapter.close()

    checkpoint["migrate"] = {
        "cursor": None,
        "vectors_read": 0,
        "vectors_written": 0,
        "done": False,
        "written_ids": [],
    }
    checkpoint["history"].append({"state": "ROLLBACK", "detail": {"deleted": deleted}})
    save_checkpoint(migration_id, checkpoint)

    return {"migration_id": migration_id, "deleted": deleted}
