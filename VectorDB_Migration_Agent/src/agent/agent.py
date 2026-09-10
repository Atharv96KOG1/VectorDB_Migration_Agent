"""Orchestrator workflow: the migration state machine (V2 §45, +CDC_SYNC per V3 fix #6).

This module is a Temporal workflow (via `@agent()`) and must stay deterministic — it only
ever calls out through `toolExecutor.execute(...)` (Temporal activities). It imports
NOTHING from core/ except the scalar-only DTOs in core.models.workflow_state — never
core/adapters, core/transformations, core/benchmarking, or anything that pulls
numpy/scipy/sklearn/qdrant_client/httpx. tests/unit/test_agent_import_boundary.py asserts
this import graph stays clean.

No `humanInput` usage (removed 2026-09-01, explicit operator request): the aetherion_sdk
human-input mechanism (register_human_input_request) was not reachable from this tool
worker's activity registry in this SDK version, and blocked every migration indefinitely
at APPROVAL and every failure indefinitely at FAILED. A migration that clears the quality
gate now proceeds straight to MIGRATE with no human sign-off gate; a failure now just
records the reason and stops, with no recovery-choice prompt.

All the real work (IO, randomness, ML) lives in tools/*.py and is invoked by name here.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from aetherion_sdk import agent, toolExecutor

from core.models.workflow_state import MigrationRequest, MigrationState, WorkflowResult

_SHORT_TIMEOUT = timedelta(minutes=10)
_LONG_TIMEOUT = timedelta(hours=2)


async def _run(
    tool_name: str, *args: Any, timeout: timedelta = _SHORT_TIMEOUT, **kwargs: Any
) -> Any:
    return await toolExecutor.execute(tool_name, *args, start_to_close_timeout=timeout, **kwargs)


def _result(
    migration_id: str, final_state: MigrationState, status: str, summary: str, **refs: Any
) -> dict:
    return WorkflowResult(
        migration_id=migration_id, final_state=final_state, status=status, summary=summary, **refs
    ).model_dump(mode="json")


async def _enter_failed(migration_id: str, reason: str) -> dict:
    """FAILED is minimal-viable: records the failure and stops. Recovery is an external
    action — a new workflow execution started with `resume=True`, which INIT's
    check_resume picks up from the last checkpointed state; Temporal has no built-in
    "resume a terminated execution in place" primitive (see core/checkpointing/store.py's
    docstring), so this workflow does not attempt to loop back into itself automatically.

    No human-input prompt here (removed 2026-09-01, explicit operator request): the
    aetherion_sdk human-input mechanism (register_human_input_request) was not reachable
    from this tool worker's activity registry in this SDK version, and blocked every
    failed migration indefinitely waiting for a recovery choice that could never arrive.
    """
    return _result(migration_id, MigrationState.FAILED, "failed", reason)


@agent()
async def VectorDB_Migration_Agent(payload: dict) -> dict:
    request = MigrationRequest.model_validate(payload)
    migration_id = request.migration_id

    try:
        resume_info = await _run("check_resume", migration_id)
        if request.resume and resume_info["resumable"] and resume_info["last_state"]:
            state = MigrationState(resume_info["last_state"])
        else:
            await _run("init_checkpoint", request.model_dump(mode="json"))
            state = MigrationState.CONNECT

        if state == MigrationState.INIT:
            state = MigrationState.CONNECT

        if state == MigrationState.CONNECT:
            connection = await _run("connect_and_validate", migration_id)
            if not connection["connected"]:
                return await _enter_failed(
                    migration_id, f"connection validation failed: {connection['detail']}"
                )
            state = MigrationState.DISCOVER

        if state == MigrationState.DISCOVER:
            await _run("discover_resources", migration_id, timeout=_LONG_TIMEOUT)
            state = MigrationState.NORMALIZE

        if state == MigrationState.NORMALIZE:
            await _run("build_canonical_ir", migration_id)
            state = MigrationState.COMPARE

        if state == MigrationState.COMPARE:
            await _run("run_compatibility_check", migration_id)
            state = MigrationState.PLAN

        if state == MigrationState.PLAN:
            await _run("generate_candidates", migration_id)
            state = MigrationState.DRY_RUN

        if state == MigrationState.DRY_RUN:
            await _run("prepare_benchmark_sample", migration_id, timeout=_LONG_TIMEOUT)
            state = MigrationState.BENCHMARK

        if state == MigrationState.BENCHMARK:
            await _run("run_benchmarks", migration_id, timeout=_LONG_TIMEOUT)
            selection = await _run("select_strategy", migration_id)
            if not selection.get("selected"):
                return await _enter_failed(
                    migration_id, f"no candidate passed the quality gate: {selection.get('reason')}"
                )
            state = MigrationState.APPROVAL

        if state == MigrationState.APPROVAL:
            # Human approval removed (2026-09-01, explicit operator request): the
            # aetherion_sdk human-input mechanism (register_human_input_request) was not
            # reachable from this tool worker's activity registry in this SDK version and
            # blocked every migration indefinitely. build_approval_summary still runs so
            # the same recall/ndcg/strategy data lands in the audit trail — a migration
            # that clears the quality gate now proceeds straight to MIGRATE, with no
            # human sign-off in between.
            await _run("build_approval_summary", migration_id)
            state = MigrationState.MIGRATE

        if state == MigrationState.MIGRATE:
            done = False
            while not done:
                batch = await _run("migrate_batch", migration_id, timeout=_LONG_TIMEOUT)
                done = batch["done"]
            state = MigrationState.CDC_SYNC

        if state == MigrationState.CDC_SYNC:
            # No source adapter in this build ever reports migration.cdc = TRUE, so this
            # state is always a capability-gated no-op — its shape exists in the machine
            # (V3 fix #6) without a fake CDC listener behind it.
            state = MigrationState.VERIFY

        if state == MigrationState.VERIFY:
            await _run("verify_migration", migration_id, None, timeout=_LONG_TIMEOUT)
            state = MigrationState.CUTOVER

        if state == MigrationState.CUTOVER:
            state = MigrationState.COMPLETE

        report = await _run("write_provenance_and_report", migration_id)
        return _result(
            migration_id,
            MigrationState.COMPLETE,
            "success",
            "migration completed",
            provenance_ref=report["provenance_ref"],
            report_ref=report["report_ref"],
            quality_gate_passed=report["quality_gate_passed"],
            confidence_score=report["confidence_score"],
            report=report["report"],
        )

    except Exception as exc:  # noqa: BLE001 — top-level workflow guard, routes to FAILED
        return await _enter_failed(migration_id, str(exc))
