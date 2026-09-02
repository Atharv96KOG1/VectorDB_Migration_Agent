"""NORMALIZE and COMPARE states: assemble the Canonical Vector IR from raw discovery
facts (V2 §5), then run the compatibility engine over it (V2 §8-9).
"""

from __future__ import annotations

from aetherion_sdk import tool

from core.checkpointing.store import load_checkpoint, save_checkpoint
from core.compatibility.engine import analyze_compatibility
from core.models.canonical_ir import (
    CanonicalVectorIR,
    DataType,
    EmbeddingProvenance,
    IdSpec,
    Metric,
    MrlSpec,
    NamespaceSpec,
    NormalizationSpec,
    ResourceRef,
    ValidationPolicy,
    VectorFieldSpec,
    VectorKind,
    VectorSourceSpec,
    VectorTargetSpec,
)
from core.models.capability import Capability


def _metric(value: str | None) -> Metric:
    try:
        return Metric(value) if value else Metric.UNKNOWN
    except ValueError:
        return Metric.UNKNOWN


def _datatype(value: str | None) -> DataType:
    try:
        return DataType(value) if value else DataType.UNKNOWN
    except ValueError:
        return DataType.UNKNOWN


@tool()
async def build_canonical_ir(migration_id: str) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")
    req = checkpoint["request"]
    discovery = checkpoint["discovery"]
    source_info = discovery["source"]["resource_info"]
    target_info = discovery["target"]["resource_info"]

    embedding_dict = discovery["embedding"]
    embedding = EmbeddingProvenance(
        provider=embedding_dict["provider"],
        model=embedding_dict["model"],
        native_dimension=source_info["dimension"],
        mrl=MrlSpec(**embedding_dict["mrl"]),
        normalization=NormalizationSpec(**discovery["normalization"]),
        discovered_from=embedding_dict["discovered_from"],
        confidence=embedding_dict["confidence"],
        languages_detected=embedding_dict.get("languages_detected", []),
        revision=embedding_dict.get("revision"),
        tokenizer=embedding_dict.get("tokenizer"),
        pooling=embedding_dict.get("pooling"),
        query_prefix=embedding_dict.get("query_prefix"),
        document_prefix=embedding_dict.get("document_prefix"),
    )

    source_quantized = Capability(source_info.get("quantized", Capability.UNKNOWN.value))

    field = VectorFieldSpec(
        name="default",
        source=VectorSourceSpec(
            kind=VectorKind.DENSE,
            dimension=source_info["dimension"],
            datatype=_datatype(source_info.get("datatype")),
            metric=_metric(source_info.get("metric")),
            quantized=source_quantized,
        ),
        embedding=embedding,
        target=VectorTargetSpec(
            kind=VectorKind.DENSE,
            dimension=target_info["dimension"],
            datatype=_datatype(target_info.get("datatype")),
            metric=_metric(target_info.get("metric")),
        ),
    )

    ir = CanonicalVectorIR(
        migration_id=migration_id,
        source=ResourceRef(
            provider=req["source_provider"],
            name=req["source_resource"],
            namespace=req.get("namespace"),
        ),
        target=ResourceRef(provider=req["target_provider"], name=req["target_resource"]),
        vectors=[field],
        # .strip() for the same reason as build_adapter's provider parsing — a free-text
        # textarea trigger field can carry trailing whitespace/newlines a dropdown never would.
        id=IdSpec(collision_policy=(req.get("id_collision_policy") or "fail").strip()),
        namespace=NamespaceSpec(
            source_supported=bool(req.get("namespace")),
            target_mapping="payload.__namespace" if req.get("namespace") else None,
        ),
        validation=ValidationPolicy(
            minimum_recall_at_10=req.get("minimum_recall_at_10", 0.80),
            minimum_ndcg_at_10=req.get("minimum_ndcg_at_10", 0.90),
            maximum_topk_overlap_drop=req.get("maximum_topk_overlap_drop", 0.15),
            maximum_latency_increase_percent=req.get("maximum_latency_increase_percent", 20.0),
        ),
    )

    checkpoint["canonical_ir"] = ir.model_dump(mode="json")
    checkpoint["history"].append({"state": "NORMALIZE", "detail": "canonical IR built"})
    save_checkpoint(migration_id, checkpoint)

    return {
        "migration_id": migration_id,
        "source_dimension": source_info["dimension"],
        "target_dimension": target_info["dimension"],
    }


@tool()
async def run_compatibility_check(migration_id: str) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")
    ir = CanonicalVectorIR.model_validate(checkpoint["canonical_ir"])
    field = ir.vectors[0]

    target_count = checkpoint["discovery"]["target"]["resource_info"].get("approximate_count") or 0
    # Undiscoverable via any vector DB API in general (nothing exposes "what model
    # embedded these existing vectors") — the one exception this build takes advantage
    # of is a resumed migration's own prior provenance, which records what WE wrote.
    target_semantic_space_id = (checkpoint.get("provenance") or {}).get("transformation", {}).get(
        "source_semantic_space_id"
    )

    report = analyze_compatibility(
        source_dim=field.source.dimension,
        target_dim=field.target.dimension,
        source_dt=field.source.datatype,
        target_dt=field.target.datatype,
        source_metric=field.source.metric,
        target_metric=field.target.metric,
        source_kind=field.source.kind,
        target_kind=field.target.kind,
        source_quantized=field.source.quantized,
        normalization=field.embedding.normalization,
        target_already_populated=target_count > 0,
        source_semantic_space_id=field.embedding.semantic_space_id,
        target_semantic_space_id=target_semantic_space_id,
    )

    checkpoint["compatibility"] = report.model_dump(mode="json")
    checkpoint["history"].append(
        {"state": "COMPARE", "detail": {"direct_copy_possible": report.direct_copy_possible}}
    )
    save_checkpoint(migration_id, checkpoint)

    return {
        "migration_id": migration_id,
        "direct_copy_possible": report.direct_copy_possible,
        "dimension": report.dimension.classification.value,
        "metric": report.metric.classification.value,
        "datatype": report.datatype.classification.value,
        "vector_type": report.vector_type.classification.value,
        "semantic_space": report.semantic_space.classification.value,
    }
