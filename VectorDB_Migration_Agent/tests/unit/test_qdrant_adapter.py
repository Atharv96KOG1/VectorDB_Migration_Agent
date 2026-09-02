"""No dedicated adapter test existed before — every integration test happened to use
`uuid.uuid4()` source ids, which are already Qdrant-native and never exercised the id
mapping path. That's exactly why a real migration with natural string ids ("cooking-6")
broke live (Qdrant rejects any point id that isn't an unsigned int or a UUID) before this
fix existed. These tests target the id-mapping round-trip directly, using Qdrant's
in-memory mode (`:memory:`) — fast, isolated, no disk/locking concerns.
"""

from __future__ import annotations

from qdrant_client import models

from core.adapters.base import VectorRecord
from core.adapters.qdrant_adapter import QdrantAdapter, _is_qdrant_native_id, _to_qdrant_point_id


def test_native_ids_pass_through_unmapped():
    assert _is_qdrant_native_id("12345") is True
    assert _is_qdrant_native_id("550e8400-e29b-41d4-a716-446655440000") is True
    assert _to_qdrant_point_id("12345") == "12345"
    assert _to_qdrant_point_id("550e8400-e29b-41d4-a716-446655440000") == "550e8400-e29b-41d4-a716-446655440000"


def test_natural_string_ids_are_not_native():
    assert _is_qdrant_native_id("cooking-6") is False
    assert _is_qdrant_native_id("sku-9981") is False
    assert _is_qdrant_native_id("") is False


def test_mapping_is_deterministic_and_collision_free_for_similar_ids():
    a = _to_qdrant_point_id("cooking-6")
    b = _to_qdrant_point_id("cooking-6")
    c = _to_qdrant_point_id("cooking-7")
    assert a == b, "same original id must always map to the same Qdrant id (idempotent upsert depends on this)"
    assert a != c


async def test_upsert_fetch_query_round_trips_natural_string_ids():
    adapter = QdrantAdapter(location=":memory:")
    await adapter.create_collection("col", dimension=4)

    records = [
        VectorRecord(id="cooking-6", vector=[1.0, 0.0, 0.0, 0.0], payload={"topic": "cooking"}),
        VectorRecord(id="finance-7", vector=[0.0, 1.0, 0.0, 0.0], payload={"topic": "finance"}),
    ]
    written = await adapter.upsert_vectors("col", records)
    assert written == 2

    fetched = await adapter.fetch_by_ids("col", ["cooking-6", "finance-7"])
    fetched_by_id = {r.id: r for r in fetched}
    assert set(fetched_by_id) == {"cooking-6", "finance-7"}
    assert fetched_by_id["cooking-6"].payload == {"topic": "cooking"}  # internal id field never leaks

    matches = await adapter.query("col", [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert matches[0].id == "cooking-6"  # query results also recover the original id

    await adapter.delete_collection("col")
    await adapter.close()


async def test_upsert_is_idempotent_for_natural_string_ids_on_retry():
    """A Temporal-style retry of the exact same upsert must overwrite, not duplicate —
    the whole idempotency guarantee (V2 §39) depends on the id mapping being stable."""
    adapter = QdrantAdapter(location=":memory:")
    await adapter.create_collection("col", dimension=2)

    record = VectorRecord(id="doc-42", vector=[0.5, 0.5], payload={"v": 1})
    await adapter.upsert_vectors("col", [record])
    await adapter.upsert_vectors("col", [record])  # simulated retry

    info = await adapter.get_resource_info("col")
    assert info.approximate_count == 1

    await adapter.delete_collection("col")
    await adapter.close()


async def test_scan_vectors_recovers_original_ids_for_natural_string_source_ids():
    adapter = QdrantAdapter(location=":memory:")
    await adapter.create_collection("col", dimension=3)
    records = [
        VectorRecord(id=f"item-{i}", vector=[float(i), 0.0, 0.0], payload={})
        for i in range(5)
    ]
    await adapter.upsert_vectors("col", records)

    page = await adapter.scan_vectors("col", batch_size=10)
    assert {r.id for r in page.records} == {f"item-{i}" for i in range(5)}

    await adapter.delete_collection("col")
    await adapter.close()


async def _create_named_vector_collection(adapter: QdrantAdapter, name: str, dimension: int) -> None:
    # A hybrid dense+sparse, named-vector schema — confirmed live, 2026-08-31, against a
    # real Qdrant Cloud collection created via a quick-start template ("dense-vector" +
    # "sparse-vector"), which the plain QdrantAdapter.create_collection() helper never
    # produces itself but must still be able to write into.
    await adapter._client.create_collection(
        collection_name=name,
        vectors_config={
            "dense-vector": models.VectorParams(size=dimension, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"sparse-vector": models.SparseVectorParams()},
    )


async def test_upsert_and_query_target_the_named_vector_in_a_hybrid_collection():
    adapter = QdrantAdapter(location=":memory:")
    await _create_named_vector_collection(adapter, "hybrid_col", dimension=3)

    records = [
        VectorRecord(id="a", vector=[1.0, 0.0, 0.0], payload={"k": "a"}),
        VectorRecord(id="b", vector=[0.0, 1.0, 0.0], payload={"k": "b"}),
    ]
    await adapter.upsert_vectors("hybrid_col", records)

    info = await adapter.get_resource_info("hybrid_col")
    assert info.dimension == 3

    matches = await adapter.query("hybrid_col", [1.0, 0.0, 0.0], top_k=1)
    assert matches[0].id == "a"

    fetched = await adapter.fetch_by_ids("hybrid_col", ["a", "b"])
    assert {r.id for r in fetched} == {"a", "b"}

    await adapter._client.delete_collection("hybrid_col")
    await adapter.close()


async def test_resolve_vector_name_self_heals_on_a_fresh_adapter_instance(tmp_path):
    # Same lesson as PineconeAdapter.query()'s 2026-08-31 fix: a fresh adapter instance
    # for this call must not require get_resource_info to have been called on it first.
    path = str(tmp_path / "qdrant_store")
    setup = QdrantAdapter(path=path)
    await _create_named_vector_collection(setup, "hybrid_col", dimension=2)
    await setup.close()

    fresh = QdrantAdapter(path=path)
    await fresh.upsert_vectors(
        "hybrid_col", [VectorRecord(id="x", vector=[1.0, 0.0], payload={})]
    )
    matches = await fresh.query("hybrid_col", [1.0, 0.0], top_k=1)
    assert matches[0].id == "x"
    await fresh.close()
