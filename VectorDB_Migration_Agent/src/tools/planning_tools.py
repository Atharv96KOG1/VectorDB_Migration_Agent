"""PLAN state: the representation analyzer decision tree (V2 §9, §21) — never default to
PCA, enumerate every candidate and eliminate impossible ones using real transformer
can_apply() logic (tools/_transform_factory.py) so this doesn't duplicate eligibility
rules the transformers themselves already own. Also the anti-hallucination guardrail
(V3 §8): selecting a strategy is only ever allowed by resolving an id that actually
exists in the checkpoint's benchmark-results table.
"""

from __future__ import annotations

from aetherion_sdk import tool

from core.checkpointing.store import load_checkpoint, save_checkpoint
from core.compatibility.engine import CompatibilityReport
from core.models.canonical_ir import CanonicalVectorIR
from core.models.capability import CapabilityReport
from core.models.migration_plan import (
    CandidateStatus,
    MigrationPlan,
    OptimizationWeights,
    TransformCandidate,
    TransformStrategy,
)
from core.optimizer.scorer import CandidateMetrics, score_candidates
from core.transformations.base import ApplicabilityInputs, TransformContext
from tools._transform_factory import build_transformer

_ALL_STRATEGIES = list(TransformStrategy)


@tool()
async def generate_candidates(migration_id: str) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")

    ir = CanonicalVectorIR.model_validate(checkpoint["canonical_ir"])
    field = ir.vectors[0]
    compatibility = CompatibilityReport.model_validate(checkpoint["compatibility"])
    source_capabilities = CapabilityReport.model_validate(
        checkpoint["discovery"]["source"]["capabilities"]
    )

    target_dim = field.target.dimension
    # re_embedding/ridge_mapping/procrustes_mapping's can_apply() only checks whether
    # documents WILL be available (a truthy context.documents or a TRUE retrieve_documents
    # capability) — at PLAN time there's no benchmark sample yet to hand it real text, so
    # a single placeholder signals "yes, document_field is configured" honestly without
    # fabricating content nothing here will use for computation. Confirmed live,
    # 2026-08-31: without this, all three were silently marked IMPOSSIBLE and never even
    # reached BENCHMARK whenever document_field was set, no matter how good they'd score.
    document_field = checkpoint["request"].get("document_field")
    inputs = ApplicabilityInputs(
        source_dim=field.source.dimension,
        target_dim=target_dim,
        embedding=field.embedding,
        capabilities=source_capabilities,
        context=TransformContext(
            documents=["placeholder"] if document_field else None,
            extra={"historical_queries_available": False},
        ),
    )

    candidates: list[TransformCandidate] = []
    for strategy in _ALL_STRATEGIES:
        transformer = build_transformer(strategy, target_dim, compatibility=compatibility)
        result = transformer.can_apply(inputs)
        candidates.append(
            TransformCandidate(
                strategy=strategy,
                status=CandidateStatus.POSSIBLE if result.possible else CandidateStatus.IMPOSSIBLE,
                reason=result.reason,
                is_stub=transformer.is_stub,
            )
        )

    plan = MigrationPlan(
        plan_id=f"{migration_id}-plan",
        migration_id=migration_id,
        candidates=candidates,
    )

    checkpoint["plan"] = plan.model_dump(mode="json")
    checkpoint.setdefault("benchmark_results", [])
    checkpoint["history"].append(
        {
            "state": "PLAN",
            "detail": {
                "candidate_count": len(candidates),
                "possible": [
                    c.strategy.value for c in candidates if c.status == CandidateStatus.POSSIBLE
                ],
            },
        }
    )
    save_checkpoint(migration_id, checkpoint)

    return {
        "migration_id": migration_id,
        "candidates": [
            {
                "strategy": c.strategy.value,
                "status": c.status.value,
                "is_stub": c.is_stub,
                "reason": c.reason,
            }
            for c in candidates
        ],
    }


def _require_existing_benchmark(checkpoint: dict, benchmark_id: str) -> dict:
    """The anti-hallucination guardrail (V3 §8), enforced mechanically: raises unless
    `benchmark_id` is a row that was actually executed and recorded, not merely proposed."""
    for row in checkpoint.get("benchmark_results", []):
        if row["benchmark_id"] == benchmark_id:
            return row
    raise ValueError(
        f"benchmark_id {benchmark_id!r} does not exist in the executed benchmark-results "
        f"table for migration_id={checkpoint['migration_id']!r} — refusing to select an "
        f"untested strategy"
    )


@tool()
async def select_strategy(migration_id: str, benchmark_id: str | None = None) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")

    benchmark_rows = checkpoint.get("benchmark_results", [])
    passing = [row for row in benchmark_rows if row["passed_quality_gate"]]

    if benchmark_id is not None:
        row = _require_existing_benchmark(checkpoint, benchmark_id)
        if not row["passed_quality_gate"]:
            raise ValueError(
                f"benchmark_id {benchmark_id!r} did not pass the quality gate; cannot select it"
            )
        chosen = row
    else:
        if not passing:
            checkpoint["history"].append(
                {"state": "PLAN", "detail": "no candidate passed the quality gate"}
            )
            save_checkpoint(migration_id, checkpoint)
            return {
                "migration_id": migration_id,
                "selected": False,
                "reason": "no candidate passed the quality gate",
            }

        direct_copy_row = next(
            (row for row in passing if row["strategy"] == TransformStrategy.DIRECT_COPY.value),
            None,
        )
        if direct_copy_row is not None:
            # direct_copy passing the quality gate at all means dimension/metric/dtype
            # were already fully compatible — zero semantic transformation happened, so
            # it carries zero transform risk by construction. It must never lose to
            # another candidate on cost/latency/etc: those secondary metrics for a fast
            # ephemeral-collection benchmark are dominated by shared RPC/setup overhead
            # common to every candidate, not real per-transform cost, and can differ by
            # a few noisy milliseconds run to run — real bug found live, 2026-09-01, once
            # random_projection became an exact isometry (see random_projection.py) and
            # could tie direct_copy's quality bit-for-bit: the generic multi-objective
            # optimizer then let that noise flip the winner between identical runs.
            chosen = direct_copy_row
        else:
            req = checkpoint["request"]
            weights = (
                OptimizationWeights(**req.get("optimization_weights", {}))
                if req.get("optimization_weights")
                else OptimizationWeights()
            )
            metrics = [
                CandidateMetrics(
                    benchmark_id=row["benchmark_id"],
                    strategy=row["strategy"],
                    passed_quality_gate=row["passed_quality_gate"],
                    quality=row["recall_at_10"],
                    cost=row.get("estimated_cost") or 0.0,
                    latency_ms=row.get("latency_ms_p95") or 0.0,
                )
                for row in benchmark_rows
            ]
            ranked = score_candidates(metrics, weights)
            if not ranked:
                return {
                    "migration_id": migration_id,
                    "selected": False,
                    "reason": "optimizer produced no ranking",
                }
            best = ranked[0]
            chosen = _require_existing_benchmark(checkpoint, best["benchmark_id"])

    plan = MigrationPlan.model_validate(checkpoint["plan"])
    plan.selected_strategy = TransformStrategy(chosen["strategy"])
    plan.selected_benchmark_id = chosen["benchmark_id"]
    for candidate in plan.candidates:
        candidate.status = (
            CandidateStatus.SELECTED
            if candidate.strategy == plan.selected_strategy
            else candidate.status
        )
    checkpoint["plan"] = plan.model_dump(mode="json")
    checkpoint["history"].append(
        {
            "state": "PLAN",
            "detail": {
                "selected_strategy": plan.selected_strategy.value,
                "benchmark_id": plan.selected_benchmark_id,
            },
        }
    )
    save_checkpoint(migration_id, checkpoint)

    return {
        "migration_id": migration_id,
        "selected": True,
        "strategy": plan.selected_strategy.value,
        "benchmark_id": plan.selected_benchmark_id,
        "recall_at_10": chosen["recall_at_10"],
        "ndcg_at_10": chosen["ndcg_at_10"],
    }
