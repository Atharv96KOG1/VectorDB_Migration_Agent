"""Regression test for a real gap found live, 2026-08-31: agent.py's workflow validates
every trigger payload against MigrationRequest before anything else runs
(`MigrationRequest.model_validate(payload)`) — pydantic silently DROPS unknown fields by
default rather than erroring, so document_field/reembed_model/reembed_dimensions/
source_embedding_config (added to the underlying tools across this session's live testing,
always exercised by calling tool functions directly with a raw dict, never through the
real workflow) were invisible to any migration actually triggered through Aetherion.
"""

from __future__ import annotations

from core.models.workflow_state import MigrationRequest

_BASE = dict(
    migration_id="mig-1",
    source_provider="pinecone",
    source_resource="src",
    source_credential_ref="env://X",
    target_provider="qdrant",
    target_resource="tgt",
    target_credential_ref="env://Y",
)


def test_document_field_and_reembed_options_survive_validation():
    request = MigrationRequest.model_validate(
        {
            **_BASE,
            "document_field": "text",
            "reembed_model": "text-embedding-3-large",
            "reembed_dimensions": 1536,
            "source_embedding_config": {"provider": "openai", "model": "text-embedding-3-large"},
        }
    )
    dumped = request.model_dump(mode="json")
    assert dumped["document_field"] == "text"
    assert dumped["reembed_model"] == "text-embedding-3-large"
    assert dumped["reembed_dimensions"] == 1536
    assert dumped["source_embedding_config"] == {
        "provider": "openai",
        "model": "text-embedding-3-large",
    }


def test_optimization_weights_survives_validation():
    # Same class of bug found again 2026-09-01: tools/planning_tools.py always read
    # request.get("optimization_weights"), but the field was never declared here — an
    # operator-supplied weighting was silently dropped on every real platform trigger.
    request = MigrationRequest.model_validate(
        {**_BASE, "optimization_weights": {"quality_weight": 0.9, "cost_weight": 0.1}}
    )
    dumped = request.model_dump(mode="json")
    assert dumped["optimization_weights"] == {"quality_weight": 0.9, "cost_weight": 0.1}


def test_defaults_when_omitted():
    request = MigrationRequest.model_validate(_BASE)
    dumped = request.model_dump(mode="json")
    assert dumped["document_field"] is None
    assert dumped["reembed_model"] == "text-embedding-3-small"
    assert dumped["reembed_dimensions"] is None
    assert dumped["source_embedding_config"] is None
    assert dumped["optimization_weights"] is None
