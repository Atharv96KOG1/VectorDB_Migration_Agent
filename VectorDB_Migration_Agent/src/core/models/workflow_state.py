"""Scalar-only DTOs safe to import from the orchestrator workflow (src/agent/agent.py).

Temporal's workflow sandbox constrains the *import graph* of a workflow module, not just
its function bodies. This module must never import numpy/scipy/sklearn/qdrant_client/httpx,
directly or transitively — those belong to core/adapters, core/transformations, and
core/benchmarking, which are only ever imported from tools/ (activities). See
tests/unit/test_agent_import_boundary.py, which asserts this module graph stays clean.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MigrationState(str, Enum):
    """The 14-state migration state machine (V2 §45, +CDC_SYNC per V3 §2)."""

    INIT = "INIT"
    CONNECT = "CONNECT"
    DISCOVER = "DISCOVER"
    NORMALIZE = "NORMALIZE"
    COMPARE = "COMPARE"
    PLAN = "PLAN"
    DRY_RUN = "DRY_RUN"
    BENCHMARK = "BENCHMARK"
    APPROVAL = "APPROVAL"
    MIGRATE = "MIGRATE"
    CDC_SYNC = "CDC_SYNC"
    VERIFY = "VERIFY"
    CUTOVER = "CUTOVER"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# Linear happy-path order. FAILED is reachable from any state and is not part of this list.
STATE_ORDER: list[MigrationState] = [
    MigrationState.INIT,
    MigrationState.CONNECT,
    MigrationState.DISCOVER,
    MigrationState.NORMALIZE,
    MigrationState.COMPARE,
    MigrationState.PLAN,
    MigrationState.DRY_RUN,
    MigrationState.BENCHMARK,
    MigrationState.APPROVAL,
    MigrationState.MIGRATE,
    MigrationState.CDC_SYNC,
    MigrationState.VERIFY,
    MigrationState.CUTOVER,
    MigrationState.COMPLETE,
]


class RecoveryAction(str, Enum):
    RETRY = "retry"
    RESUME = "resume"
    ABORT = "abort"


class MigrationRequest(BaseModel):
    """The workflow trigger payload. Credential/endpoint fields carry either an
    environment variable name (looked up inside tools, e.g. "PINECONE_API_KEY") or,
    since 2026-09-01, the raw secret value itself typed directly into the field — an
    explicit operator tradeoff (test live creds without a worker restart) over the
    original "reference only, never a raw secret" design (V2 §7/§43).
    """

    migration_id: str
    source_provider: str
    source_resource: str
    source_credential_ref: str
    source_endpoint_ref: str | None = None
    target_provider: str
    target_resource: str
    target_credential_ref: str
    target_endpoint_ref: str | None = None
    namespace: str | None = None
    minimum_recall_at_10: float = 0.80
    minimum_ndcg_at_10: float = 0.90
    maximum_topk_overlap_drop: float = 0.15
    maximum_latency_increase_percent: float = 20.0
    id_collision_policy: str = "fail"
    resume: bool = False

    document_field: str | None = None
    """Payload key holding source text, required for re_embedding/ridge_mapping/
    procrustes_mapping (tools/discovery_tools.py, tools/benchmark_tools.py,
    tools/execution_tools.py) — without it those strategies are correctly marked
    IMPOSSIBLE at PLAN time, never silently attempted."""

    reembed_model: str = "text-embedding-3-small"
    reembed_dimensions: int | None = None
    """Passed to tools/_transform_factory.py:_resolve_reembed_dimensions — an explicit
    value always wins; otherwise defaults to the target dimension for a model known to
    support OpenAI's native truncation (text-embedding-3-*)."""

    source_embedding_config: dict | None = None
    """Operator-supplied embedding contract (provider/model/revision/tokenizer/pooling/
    query_prefix/document_prefix) — tools/discovery_tools.py:_build_embedding_discovery.
    None of this is discoverable from raw vectors; supplying provider+model turns
    semantic_space_id from a vacuous "unknown/unknown" into a real, checkable identity."""

    optimization_weights: dict | None = None
    """quality/cost/latency/time/storage/risk weights (core/models/migration_plan.py:
    OptimizationWeights) for PLAN's candidate scorer (tools/planning_tools.py). Missing
    here meant this field was silently dropped by MigrationRequest.model_validate on any
    real platform-triggered payload — an operator-supplied weighting was unreachable in
    production even though tools/planning_tools.py always read it correctly (found
    2026-09-01, same class of bug as document_field/reembed_model before it)."""


class WorkflowStatus(BaseModel):
    """Live workflow status, exposed via an aetherion_sdk `query()`."""

    migration_id: str
    state: MigrationState
    detail: str = ""
    error_message: str | None = None
    plan_id: str | None = None
    selected_benchmark_id: str | None = None


class WorkflowResult(BaseModel):
    migration_id: str
    final_state: MigrationState
    status: str = Field(description="'success' or 'failed'")
    summary: str = ""
    provenance_ref: str | None = None
    report_ref: str | None = None
    quality_gate_passed: bool | None = None
    confidence_score: float | None = None
    provenance: dict | None = None
    """The actual provenance record content, inline — not just provenance_ref's local
    path. A managed/hosted worker's filesystem is invisible to the operator (no
    artifact/download API in aetherion_sdk), so a path alone is useless there; this small,
    fully JSON-safe record is returned directly instead."""
