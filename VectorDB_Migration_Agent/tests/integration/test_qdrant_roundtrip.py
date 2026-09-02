"""End-to-end proof of the pipeline that needs no external credentials: two local,
on-disk Qdrant stores (source and target — see core/adapters/qdrant_adapter.py and
tools/_shared.py's "path://" scheme) taken through the real tool functions exactly as
the orchestrator workflow (src/agent/agent.py) would call them, just without a live
Temporal worker (the `@tool()` decorator leaves the underlying coroutine directly
callable — see aetherion_sdk's decorators.pyi — so this exercises the actual production
code path, not a reimplementation of it).

Two scenarios, both against a real quality gate:
  (a) direct-copy: 128D -> 128D, same metric/dtype -> Recall@10 must be ~1.0
  (b) PCA: 128D -> 64D -> Recall@10 is measured for real and must clear the configured
      quality gate for select_strategy to pick it at all
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from core.adapters.base import VectorRecord
from core.adapters.qdrant_adapter import QdrantAdapter
from core.checkpointing.store import load_checkpoint
from tools.benchmark_tools import run_benchmarks
from tools.checkpoint_tools import init_checkpoint
from tools.compatibility_tools import build_canonical_ir, run_compatibility_check
from tools.connection_tools import connect_and_validate
from tools.discovery_tools import discover_resources
from tools.execution_tools import migrate_batch, rollback_migration
from tools.planning_tools import generate_candidates, select_strategy
from tools.report_tools import build_approval_summary, write_provenance_and_report
from tools.transform_tools import prepare_benchmark_sample
from tools.validation_tools import verify_migration

N_VECTORS = 300
SOURCE_DIM = 128


def _clustered_unit_vectors(n: int, dim: int, n_clusters: int = 8, seed: int = 0) -> np.ndarray:
    """Random-noise-only vectors make PCA meaningless (no structure to preserve), so this
    generates cluster-structured data — closer to real embeddings, and it's what makes a
    128D->64D PCA reduction actually able to preserve nearest-neighbor structure well
    enough to clear a real quality gate."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * 3.0
    assignments = rng.integers(0, n_clusters, size=n)
    noise = rng.standard_normal((n, dim)) * 0.3
    vectors = centers[assignments] + noise
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


async def _seed_source_collection(path: str, dim: int) -> list[str]:
    adapter = QdrantAdapter(path=path)
    await adapter.create_collection("source_col", dimension=dim)
    vectors = _clustered_unit_vectors(N_VECTORS, dim)
    ids = [str(uuid.uuid4()) for _ in range(N_VECTORS)]
    records = [
        {"id": ids[i], "vector": vectors[i].tolist(), "payload": {"seq": i}}
        for i in range(N_VECTORS)
    ]
    from core.adapters.base import VectorRecord

    await adapter.upsert_vectors("source_col", [VectorRecord(**r) for r in records])
    await adapter.close()
    return ids


async def _run_pipeline_through_benchmark(
    request: dict, sample_size: int, query_count: int
) -> dict:
    migration_id = request["migration_id"]
    await init_checkpoint(request)
    connection = await connect_and_validate(migration_id)
    assert connection["connected"] is True

    await discover_resources(migration_id)
    await build_canonical_ir(migration_id)
    compat = await run_compatibility_check(migration_id)
    await generate_candidates(migration_id)
    await prepare_benchmark_sample(migration_id, sample_size=sample_size, query_count=query_count)
    benchmarks = await run_benchmarks(migration_id)
    return {"compat": compat, "benchmarks": benchmarks}


@pytest.fixture
def qdrant_paths(tmp_path):
    return {
        "source": f"path://{tmp_path / 'source_store'}",
        "target_direct": f"path://{tmp_path / 'target_store_direct'}",
        "target_pca": f"path://{tmp_path / 'target_store_pca'}",
    }


async def test_direct_copy_end_to_end_recall_is_near_perfect(tmp_path, qdrant_paths):
    source_path = str(tmp_path / "source_store")
    await _seed_source_collection(source_path, SOURCE_DIM)

    target_adapter = QdrantAdapter(path=str(tmp_path / "target_store_direct"))
    await target_adapter.create_collection("target_col", dimension=SOURCE_DIM)
    await target_adapter.close()

    migration_id = f"mig-direct-{uuid.uuid4().hex[:8]}"
    request = {
        "migration_id": migration_id,
        "source_provider": "qdrant",
        "source_resource": "source_col",
        "source_credential_ref": "env://UNUSED",
        "source_endpoint_ref": qdrant_paths["source"],
        "target_provider": "qdrant",
        "target_resource": "target_col",
        "target_credential_ref": "env://UNUSED",
        "target_endpoint_ref": qdrant_paths["target_direct"],
        "minimum_recall_at_10": 0.95,
        "minimum_ndcg_at_10": 0.90,
        "maximum_topk_overlap_drop": 0.10,
        "id_collision_policy": "fail",
    }

    # Benchmark sample = the full source corpus: the real-ANN "temporary target
    # collection" search (V2 §22, literally — target is Qdrant, so this exercises the
    # actual ephemeral-collection path, not the brute-force fallback) then covers exactly
    # what the live source Top-K query can also return, isolating the transformer's own
    # fidelity from sampling-induced recall loss (a smaller sample would legitimately
    # score lower even for a perfect identity transform, simply because true neighbors
    # outside the sample can't be found — see docs/ARCHITECTURE.md's benchmarking scope note).
    outcome = await _run_pipeline_through_benchmark(request, sample_size=N_VECTORS, query_count=20)
    assert outcome["compat"]["direct_copy_possible"] is True
    assert outcome["benchmarks"]["used_real_ann"] is True

    direct_result = next(
        r for r in outcome["benchmarks"]["results"] if r["strategy"] == "direct_copy"
    )
    assert direct_result["passed"] is True
    assert direct_result["recall_at_10"] > 0.99

    selection = await select_strategy(migration_id)
    assert selection["strategy"] == "direct_copy"

    summary = await build_approval_summary(migration_id)
    assert summary["ready"] is True
    assert summary["is_stub_strategy"] is False

    done = False
    batch = None
    while not done:
        # vectors_written is a running CUMULATIVE total across all batches (mirrors the
        # checkpoint's "migrate" state), not a per-call delta — take the last value, don't sum.
        batch = await migrate_batch(migration_id, batch_size=64)
        done = batch["done"]
    assert batch["vectors_written"] == N_VECTORS

    # Real golden-query verification (V2 §24), not the golden_queries=None degraded path —
    # exercises the concurrent per-query source+target evaluation in validation_tools.py.
    probe_adapter = QdrantAdapter(path=source_path)
    probe_sample = await probe_adapter.sample_vectors("source_col", n=5)
    await probe_adapter.close()
    golden_queries = [
        {"query": f"golden-{i}", "vector": r.vector, "expected_ids": [r.id]}
        for i, r in enumerate(probe_sample)
    ]

    verify_result = await verify_migration(migration_id, golden_queries=golden_queries)
    assert verify_result["integrity_within_tolerance"] is True
    assert verify_result["mode"] == "golden"
    assert verify_result["gate_passed"] is True

    checkpoint_after_verify = load_checkpoint(migration_id)
    assert checkpoint_after_verify["validation"]["expected_id_recall_mean"] == 1.0

    report = await write_provenance_and_report(migration_id)
    assert report["quality_gate_passed"] is True
    # 2026-09-01: report_ref/provenance_ref are local filesystem paths, invisible to an
    # operator on a managed/hosted worker (no artifact/download API in aetherion_sdk) —
    # the actual provenance content must also come back inline, not just as a path.
    assert report["provenance"]["migration_id"] == migration_id
    assert report["provenance"]["quality_gate_passed"] is True

    checkpoint = load_checkpoint(migration_id)
    assert checkpoint["migrate"]["vectors_written"] == N_VECTORS


async def test_pca_reduction_end_to_end_clears_quality_gate(tmp_path, qdrant_paths):
    source_path = str(tmp_path / "source_store")
    await _seed_source_collection(source_path, SOURCE_DIM)

    target_dim = 64
    target_adapter = QdrantAdapter(path=str(tmp_path / "target_store_pca"))
    await target_adapter.create_collection("target_col", dimension=target_dim)
    await target_adapter.close()

    migration_id = f"mig-pca-{uuid.uuid4().hex[:8]}"
    request = {
        "migration_id": migration_id,
        "source_provider": "qdrant",
        "source_resource": "source_col",
        "source_credential_ref": "env://UNUSED",
        "source_endpoint_ref": qdrant_paths["source"],
        "target_provider": "qdrant",
        "target_resource": "target_col",
        "target_credential_ref": "env://UNUSED",
        "target_endpoint_ref": qdrant_paths["target_pca"],
        # Cluster-structured synthetic data lets PCA preserve neighborhoods reasonably
        # well; the threshold is set below what perfect direct-copy would score but high
        # enough that the gate is doing real work, not rubber-stamping everything.
        "minimum_recall_at_10": 0.7,
        "minimum_ndcg_at_10": 0.6,
        "maximum_topk_overlap_drop": 0.4,
        "id_collision_policy": "fail",
    }

    outcome = await _run_pipeline_through_benchmark(request, sample_size=N_VECTORS, query_count=20)
    assert outcome["compat"]["direct_copy_possible"] is False
    assert outcome["compat"]["dimension"] == "incompatible"
    # Target is Qdrant: this run exercises the real ephemeral-collection ANN path (V2 §22)
    # for BOTH pca and random_projection concurrently — the actual regression test for the
    # shared-connection lock in tools/benchmark_tools.py:_qdrant_temp_collection_topk.
    assert outcome["benchmarks"]["used_real_ann"] is True

    pca_result = next(r for r in outcome["benchmarks"]["results"] if r["strategy"] == "pca")
    assert pca_result["sample_size"] == N_VECTORS
    assert pca_result["recall_at_10"] > 0.0  # a real, non-trivial measurement, not a stub row

    random_projection_result = next(
        r for r in outcome["benchmarks"]["results"] if r["strategy"] == "random_projection"
    )
    assert pca_result["recall_at_10"] > random_projection_result["recall_at_10"], (
        "PCA is fit to this data's actual variance; unfit random projection should score lower — "
        "if this ever flips, the benchmark itself, not just the threshold, needs a second look"
    )

    selection = await select_strategy(migration_id)
    assert selection["selected"] is True
    assert selection["strategy"] == "pca"

    done = False
    while not done:
        batch = await migrate_batch(migration_id, batch_size=64)
        done = batch["done"]

    checkpoint = load_checkpoint(migration_id)
    assert checkpoint["migrate"]["vectors_written"] == N_VECTORS
    assert checkpoint["fitted_transform"]["strategy"] == "pca"

    # Spot-check: a migrated vector must actually be 64-dimensional in the target store.
    target_adapter = QdrantAdapter(path=str(tmp_path / "target_store_pca"))
    sample = await target_adapter.sample_vectors("target_col", n=1)
    assert len(sample[0].vector) == target_dim
    await target_adapter.close()

    # Regression test for a real bug found live (2026-08, Pinecone -> Qdrant Cloud): a
    # golden query's vector is in SOURCE-space (128D here); querying a PCA-reduced 64D
    # target directly with it 400s ("Vector dimension error"). verify_migration must run
    # the query vector through the SAME fitted PCA transform before querying the target.
    probe_adapter = QdrantAdapter(path=source_path)
    probe_sample = await probe_adapter.sample_vectors("source_col", n=5)
    await probe_adapter.close()
    golden_queries = [
        {"query": f"golden-{i}", "vector": r.vector, "expected_ids": [r.id]}
        for i, r in enumerate(probe_sample)
    ]
    verify_result = await verify_migration(migration_id, golden_queries=golden_queries)
    assert verify_result["mode"] == "golden"
    assert verify_result["gate_reasons"] == []  # no crash, no dimension-mismatch error

    report = await write_provenance_and_report(migration_id)
    assert report["quality_gate_passed"] is True


async def test_generate_candidates_marks_document_dependent_strategies_possible_when_document_field_set(
    tmp_path, qdrant_paths
):
    """Regression test for a real bug found live (2026-08-31, Pinecone -> a real named-
    vector Qdrant Cloud collection): generate_candidates (PLAN) always passed
    context.documents=None to every candidate's can_apply() check, regardless of whether
    request.document_field was set — so re_embedding/ridge_mapping/procrustes_mapping
    were silently marked IMPOSSIBLE at PLAN time and never even reached BENCHMARK, no
    matter how well they'd have scored. Fixed in tools/planning_tools.py by passing a
    truthy documents placeholder whenever document_field is configured."""
    source_path = str(tmp_path / "source_store")
    await _seed_source_collection(source_path, SOURCE_DIM)

    target_adapter = QdrantAdapter(path=str(tmp_path / "target_store_docfield"))
    await target_adapter.create_collection("target_col", dimension=SOURCE_DIM)
    await target_adapter.close()

    async def _candidate_statuses(migration_id: str, document_field: str | None) -> dict[str, str]:
        request = {
            "migration_id": migration_id,
            "source_provider": "qdrant",
            "source_resource": "source_col",
            "source_credential_ref": "env://UNUSED",
            "source_endpoint_ref": qdrant_paths["source"],
            "target_provider": "qdrant",
            "target_resource": "target_col",
            "target_credential_ref": "env://UNUSED",
            "target_endpoint_ref": str(f"path://{tmp_path / 'target_store_docfield'}"),
            "id_collision_policy": "fail",
        }
        if document_field is not None:
            request["document_field"] = document_field
        await init_checkpoint(request)
        await connect_and_validate(migration_id)
        await discover_resources(migration_id)
        await build_canonical_ir(migration_id)
        await run_compatibility_check(migration_id)
        await generate_candidates(migration_id)
        checkpoint = load_checkpoint(migration_id)
        return {c["strategy"]: c["status"] for c in checkpoint["plan"]["candidates"]}

    without_field = await _candidate_statuses(f"mig-nodocfield-{uuid.uuid4().hex[:8]}", None)
    assert without_field["re_embedding"] == "impossible"
    assert without_field["ridge_mapping"] == "impossible"
    assert without_field["procrustes_mapping"] == "impossible"

    with_field = await _candidate_statuses(f"mig-docfield-{uuid.uuid4().hex[:8]}", "text")
    assert with_field["re_embedding"] == "possible"
    assert with_field["ridge_mapping"] == "possible"
    assert with_field["procrustes_mapping"] == "possible"  # source_dim == target_dim here


async def test_migrate_batch_pipelines_multiple_pages_per_call_without_loss_or_duplication(
    tmp_path,
):
    """Regression test for the double-buffered prefetch in tools/execution_tools.py: with
    batch_size=30 and 250 vectors, a single migrate_batch call (max_batches=5) should
    process multiple 30-vector pages internally — proven by batches_this_call > 1 — and
    the full run must still land every vector exactly once, no duplicates, no gaps.
    """
    dim = 16
    n = 250
    source_path = str(tmp_path / "pipeline_source")
    target_path = str(tmp_path / "pipeline_target")

    source_adapter = QdrantAdapter(path=source_path)
    await source_adapter.create_collection("col", dimension=dim)
    ids = [str(uuid.uuid4()) for _ in range(n)]
    rng = np.random.default_rng(5)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    await source_adapter.upsert_vectors(
        "col", [VectorRecord(id=ids[i], vector=vectors[i].tolist(), payload={}) for i in range(n)]
    )
    await source_adapter.close()

    target_adapter = QdrantAdapter(path=target_path)
    await target_adapter.create_collection("col", dimension=dim)
    await target_adapter.close()

    migration_id = f"mig-pipeline-{uuid.uuid4().hex[:8]}"
    request = {
        "migration_id": migration_id,
        "source_provider": "qdrant",
        "source_resource": "col",
        "source_credential_ref": "env://UNUSED",
        "source_endpoint_ref": f"path://{source_path}",
        "target_provider": "qdrant",
        "target_resource": "col",
        "target_credential_ref": "env://UNUSED",
        "target_endpoint_ref": f"path://{target_path}",
        "minimum_recall_at_10": 0.0,
        "minimum_ndcg_at_10": 0.0,
        "maximum_topk_overlap_drop": 1.0,
        "id_collision_policy": "fail",
    }
    await init_checkpoint(request)
    await connect_and_validate(migration_id)
    await discover_resources(migration_id)
    await build_canonical_ir(migration_id)
    await run_compatibility_check(migration_id)
    await generate_candidates(migration_id)
    await prepare_benchmark_sample(migration_id, sample_size=n, query_count=5)
    await run_benchmarks(migration_id)
    selection = await select_strategy(migration_id)
    assert selection["strategy"] == "direct_copy"

    call_count = 0
    max_batches_seen = 0
    done = False
    batch = None
    while not done:
        batch = await migrate_batch(migration_id, batch_size=30, max_batches=5)
        call_count += 1
        max_batches_seen = max(max_batches_seen, batch["batches_this_call"])
        done = batch["done"]

    assert batch["vectors_written"] == n
    assert max_batches_seen > 1, "expected multiple pages pipelined within a single call"
    assert call_count < (
        n // 30 + 1
    ), "pipelining should need fewer Temporal round-trips than one-batch-per-call"

    verify_adapter = QdrantAdapter(path=target_path)
    info = await verify_adapter.get_resource_info("col")
    assert info.approximate_count == n
    fetched = await verify_adapter.fetch_by_ids("col", ids)
    assert len(fetched) == n
    assert len({r.id for r in fetched}) == n  # no duplicates
    await verify_adapter.close()


async def test_rollback_deletes_only_what_this_migration_wrote(tmp_path):
    """Rollback must delete exactly the ids this migration wrote — not the whole target
    resource — so pre-existing target data (simulated here by seeding one unrelated point
    before the migration starts) survives a rollback untouched."""
    dim = 8
    n = 20
    source_path = str(tmp_path / "rollback_source")
    target_path = str(tmp_path / "rollback_target")

    source_adapter = QdrantAdapter(path=source_path)
    await source_adapter.create_collection("col", dimension=dim)
    ids = [str(uuid.uuid4()) for _ in range(n)]
    rng = np.random.default_rng(9)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    await source_adapter.upsert_vectors(
        "col", [VectorRecord(id=ids[i], vector=vectors[i].tolist(), payload={}) for i in range(n)]
    )
    await source_adapter.close()

    target_adapter = QdrantAdapter(path=target_path)
    await target_adapter.create_collection("col", dimension=dim)
    pre_existing_id = str(uuid.uuid4())
    await target_adapter.upsert_vectors(
        "col", [VectorRecord(id=pre_existing_id, vector=[0.1] * dim, payload={"pre_existing": True})]
    )
    await target_adapter.close()

    migration_id = f"mig-rollback-{uuid.uuid4().hex[:8]}"
    request = {
        "migration_id": migration_id,
        "source_provider": "qdrant",
        "source_resource": "col",
        "source_credential_ref": "env://UNUSED",
        "source_endpoint_ref": f"path://{source_path}",
        "target_provider": "qdrant",
        "target_resource": "col",
        "target_credential_ref": "env://UNUSED",
        "target_endpoint_ref": f"path://{target_path}",
        "minimum_recall_at_10": 0.0,
        "minimum_ndcg_at_10": 0.0,
        "maximum_topk_overlap_drop": 1.0,
        "id_collision_policy": "overwrite",
    }
    await init_checkpoint(request)
    await connect_and_validate(migration_id)
    await discover_resources(migration_id)
    await build_canonical_ir(migration_id)
    await run_compatibility_check(migration_id)
    await generate_candidates(migration_id)
    await prepare_benchmark_sample(migration_id, sample_size=n, query_count=5)
    await run_benchmarks(migration_id)
    await select_strategy(migration_id)

    done = False
    while not done:
        batch = await migrate_batch(migration_id, batch_size=10)
        done = batch["done"]

    verify_adapter = QdrantAdapter(path=target_path)
    info_before = await verify_adapter.get_resource_info("col")
    assert info_before.approximate_count == n + 1  # migrated data + the pre-existing point
    await verify_adapter.close()

    result = await rollback_migration(migration_id)
    assert result["deleted"] == n

    verify_adapter = QdrantAdapter(path=target_path)
    info_after = await verify_adapter.get_resource_info("col")
    assert info_after.approximate_count == 1  # only the pre-existing point survives
    remaining = await verify_adapter.fetch_by_ids("col", [pre_existing_id])
    assert len(remaining) == 1
    migrated_gone = await verify_adapter.fetch_by_ids("col", ids)
    assert migrated_gone == []
    await verify_adapter.close()

    checkpoint = load_checkpoint(migration_id)
    assert checkpoint["migrate"]["vectors_written"] == 0
    assert checkpoint["migrate"]["written_ids"] == []
