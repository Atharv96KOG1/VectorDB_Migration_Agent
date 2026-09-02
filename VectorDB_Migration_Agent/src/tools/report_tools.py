"""APPROVAL summary (kept small and reference-based — the full plan/benchmark table is
never embedded, only pointed at, regardless of who or what reads it) and the COMPLETE
state's provenance record + audit report (V2 §53, §57), with redaction (V2 §43) applied
to everything written to disk. `build_approval_summary` still runs at APPROVAL (see
src/agent/agent.py) purely to capture this data in the audit trail — there is no human
sign-off gate reading it, that was removed 2026-09-01.
"""

from __future__ import annotations

from aetherion_sdk import tool

from core.checkpointing.store import (
    DEFAULT_CHECKPOINT_DIR,
    load_checkpoint,
    save_checkpoint,
    write_artifact,
)
from core.models.canonical_ir import CanonicalVectorIR
from core.models.migration_plan import MigrationPlan, TransformStrategy
from core.models.provenance import ProvenanceRecord, TransformationProvenance, Vec2VecProvenance
from core.optimizer.scorer import compute_confidence_score
from core.security.redaction import redact


def _find_benchmark(checkpoint: dict, benchmark_id: str | None) -> dict | None:
    if benchmark_id is None:
        return None
    return next(
        (b for b in checkpoint.get("benchmark_results", []) if b["benchmark_id"] == benchmark_id),
        None,
    )


@tool()
async def build_approval_summary(migration_id: str) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")
    plan = MigrationPlan.model_validate(checkpoint["plan"])
    benchmark = _find_benchmark(checkpoint, plan.selected_benchmark_id)
    if benchmark is None:
        return {
            "migration_id": migration_id,
            "ready": False,
            "reason": "no strategy has been selected yet",
        }

    is_stub = any(c.is_stub for c in plan.candidates if c.strategy == plan.selected_strategy)

    return {
        "migration_id": migration_id,
        "ready": True,
        "strategy": plan.selected_strategy.value,
        "benchmark_id": plan.selected_benchmark_id,
        "recall_at_10": round(benchmark["recall_at_10"], 4),
        "ndcg_at_10": round(benchmark["ndcg_at_10"], 4),
        "topk_overlap": round(benchmark["topk_overlap"], 4),
        "sample_size": benchmark["sample_size"],
        "is_stub_strategy": is_stub,
        "detail_ref": f"migration_id={migration_id} (full plan/benchmark table in the checkpoint, not embedded here — 256KB humanInput ceiling)",
    }


@tool()
async def write_provenance_and_report(migration_id: str) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")
    req = checkpoint["request"]
    ir = CanonicalVectorIR.model_validate(checkpoint["canonical_ir"])
    field = ir.vectors[0]
    plan = MigrationPlan.model_validate(checkpoint["plan"])
    benchmark = _find_benchmark(checkpoint, plan.selected_benchmark_id)
    if benchmark is None:
        raise RuntimeError("cannot write provenance: plan has no selected, executed benchmark")

    fitted = checkpoint.get("fitted_transform") or {}
    transformation = TransformationProvenance(
        strategy=plan.selected_strategy.value,
        input_dimension=field.source.dimension,
        output_dimension=field.target.dimension,
        parameters=(
            fitted.get("params", {})
            if plan.selected_strategy != TransformStrategy.DIRECT_COPY
            else {}
        ),
        artifact_id=benchmark["benchmark_id"],
        source_semantic_space_id=field.embedding.semantic_space_id,
    )
    vec2vec_provenance = (
        Vec2VecProvenance() if plan.selected_strategy == TransformStrategy.VEC2VEC else None
    )

    validation = checkpoint.get("validation") or {}
    verify_retrieval = validation.get("retrieval_metrics") or {}
    integrity = validation.get("integrity") or {}
    confidence_score = compute_confidence_score(
        benchmark_recall_at_10=benchmark["recall_at_10"],
        benchmark_ndcg_at_10=benchmark["ndcg_at_10"],
        benchmark_topk_overlap=benchmark["topk_overlap"],
        verify_recall_at_10=verify_retrieval.get("recall_at_10"),
        integrity_within_tolerance=integrity.get("within_tolerance"),
    )

    record = ProvenanceRecord(
        migration_id=migration_id,
        source_provider=req["source_provider"],
        source_resource=req["source_resource"],
        target_provider=req["target_provider"],
        target_resource=req["target_resource"],
        transformation=transformation,
        vec2vec=vec2vec_provenance,
        selected_benchmark_id=plan.selected_benchmark_id,
        quality_gate_passed=benchmark["passed_quality_gate"],
        confidence_score=confidence_score,
    )

    redacted = redact(checkpoint)
    report = {
        "migration_id": migration_id,
        "history": redacted.get("history"),
        "compatibility": redacted.get("compatibility"),
        "plan": redacted.get("plan"),
        "benchmark_results": redacted.get("benchmark_results"),
        "migrate": redacted.get("migrate"),
        "validation": redacted.get("validation"),
        "provenance": record.model_dump(mode="json"),
    }

    provenance_path = write_artifact(
        DEFAULT_CHECKPOINT_DIR / "provenance",
        f"{migration_id}.json",
        record.model_dump(mode="json"),
    )
    report_path = write_artifact(DEFAULT_CHECKPOINT_DIR / "reports", f"{migration_id}.json", report)

    checkpoint["provenance"] = record.model_dump(mode="json")
    checkpoint["report_ref"] = report_path
    checkpoint["provenance_ref"] = provenance_path
    checkpoint["history"].append({"state": "COMPLETE", "detail": {"report_ref": report_path}})
    save_checkpoint(migration_id, checkpoint)

    return {
        "migration_id": migration_id,
        "report_ref": report_path,
        "provenance_ref": provenance_path,
        "quality_gate_passed": record.quality_gate_passed,
        "confidence_score": record.confidence_score,
        # *_ref are local filesystem paths on whatever worker ran this tool — invisible to
        # the operator on a managed/hosted worker (confirmed: aetherion_sdk has no
        # artifact/blob/download API at all, checked every .pyi stub, 2026-09-01). The
        # provenance record itself is small and fully JSON-safe, so it's returned inline
        # here too — real content the operator can actually see, not just an inaccessible
        # path.
        "provenance": record.model_dump(mode="json"),
    }
