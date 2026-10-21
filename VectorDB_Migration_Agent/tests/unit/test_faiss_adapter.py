"""FAISS is always locally testable — no server, no credentials, pure local file. This
exercises a real `faiss.IndexIDMap2` + sidecar JSON file round trip, including the
namespace fold and the deterministic id mapping for non-numeric source ids (FAISS's id
space is int64-only, unlike Milvus's optional varchar primary key).
"""

from __future__ import annotations

import asyncio

import pytest

from core.adapters.base import VectorRecord
from core.adapters.faiss_adapter import FaissAdapter, _is_native_faiss_id, _to_faiss_id
from core.models.canonical_ir import Metric


def test_native_int_ids_pass_through_unmapped():
    assert _is_native_faiss_id("12345") is True
    assert _to_faiss_id("12345") == 12345


def test_non_numeric_ids_are_hashed_deterministically():
    a = _to_faiss_id("cooking-6")
    b = _to_faiss_id("cooking-6")
    c = _to_faiss_id("cooking-7")
    assert a == b
    assert a != c
    assert isinstance(a, int) and a >= 0


async def test_round_trip_with_natural_string_ids(tmp_path):
    path = str(tmp_path / "index.faiss")
    FaissAdapter.create_index(path, dimension=4, metric=Metric.EUCLIDEAN)
    adapter = FaissAdapter(path)

    info = await adapter.get_resource_info("index")
    assert info.dimension == 4
    assert info.metric.value == "euclidean"
    assert info.approximate_count == 0

    records = [
        VectorRecord(id="cooking-6", vector=[1.0, 0.0, 0.0, 0.0], payload={"topic": "cooking"}),
        VectorRecord(id="finance-7", vector=[0.0, 1.0, 0.0, 0.0], payload={"topic": "finance"}),
    ]
    written = await adapter.upsert_vectors("index", records)
    assert written == 2

    fetched = await adapter.fetch_by_ids("index", ["cooking-6", "finance-7"])
    by_id = {r.id: r for r in fetched}
    assert set(by_id) == {"cooking-6", "finance-7"}
    assert by_id["cooking-6"].payload == {"topic": "cooking"}
    assert by_id["cooking-6"].vector == [1.0, 0.0, 0.0, 0.0]

    matches = await adapter.query("index", [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert matches[0].id == "cooking-6"

    page = await adapter.scan_vectors("index", batch_size=10)
    assert {r.id for r in page.records} == {"cooking-6", "finance-7"}

    deleted = await adapter.delete_vectors("index", ["finance-7"])
    assert deleted == 1

    info_after = await adapter.get_resource_info("index")
    assert info_after.approximate_count == 1


async def test_upsert_is_idempotent_and_overwrites(tmp_path):
    path = str(tmp_path / "index2.faiss")
    FaissAdapter.create_index(path, dimension=2, metric=Metric.EUCLIDEAN)
    adapter = FaissAdapter(path)

    await adapter.upsert_vectors("index2", [VectorRecord(id="a", vector=[1.0, 0.0], payload={"v": 1})])
    await adapter.upsert_vectors("index2", [VectorRecord(id="a", vector=[0.0, 1.0], payload={"v": 2})])

    info = await adapter.get_resource_info("index2")
    assert info.approximate_count == 1  # overwritten, not duplicated

    fetched = await adapter.fetch_by_ids("index2", ["a"])
    assert fetched[0].vector == [0.0, 1.0]
    assert fetched[0].payload == {"v": 2}


async def test_namespace_fold_scopes_scan_and_query(tmp_path):
    path = str(tmp_path / "index3.faiss")
    FaissAdapter.create_index(path, dimension=4, metric=Metric.EUCLIDEAN)
    adapter = FaissAdapter(path)

    await adapter.upsert_vectors(
        "index3", [VectorRecord(id="a", vector=[1.0, 0.0, 0.0, 0.0], payload={})], namespace="tenant-1"
    )
    await adapter.upsert_vectors(
        "index3", [VectorRecord(id="b", vector=[0.0, 1.0, 0.0, 0.0], payload={})], namespace="tenant-2"
    )
    page = await adapter.scan_vectors("index3", batch_size=10, namespace="tenant-1")
    assert {r.id for r in page.records} == {"a"}


def test_load_raises_clearly_when_file_missing(tmp_path):
    adapter = FaissAdapter(str(tmp_path / "does_not_exist.faiss"))
    with pytest.raises(RuntimeError, match="does not exist"):
        asyncio.run(adapter.get_resource_info("x"))
