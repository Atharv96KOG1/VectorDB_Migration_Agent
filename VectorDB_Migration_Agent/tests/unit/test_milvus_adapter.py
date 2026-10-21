"""Real round-trip tests against Milvus Lite (`milvus-lite`, a dev-only dependency) — a
local, creds-free embedded Milvus, playing the same role `:memory:` plays for
test_qdrant_adapter.py. Covers both primary-key shapes a real target collection can have:
VARCHAR (arbitrary source ids pass straight through) and INT64 (a non-numeric source id
must be hashed and recovered via the __source_id field).
"""

from __future__ import annotations

import numpy as np
import pytest
from pymilvus import DataType, MilvusClient

from core.adapters.base import VectorRecord
from core.adapters.milvus_adapter import (
    MilvusAdapter,
    _is_int64_native,
    _normalize_uri,
    _pack_vector_for_milvus,
    _to_milvus_int_id,
    _unpack_vector_from_milvus,
)


def test_normalize_uri_adds_https_to_a_bare_hostname():
    assert (
        _normalize_uri("in03-xxxx.serverless.aws-eu-central-1.cloud.zilliz.com")
        == "https://in03-xxxx.serverless.aws-eu-central-1.cloud.zilliz.com"
    )


def test_normalize_uri_leaves_an_explicit_scheme_alone():
    assert _normalize_uri("http://localhost:19530") == "http://localhost:19530"
    assert _normalize_uri("https://example.com") == "https://example.com"
    assert _normalize_uri("tcp://example.com:19530") == "tcp://example.com:19530"


def test_normalize_uri_leaves_a_local_db_file_alone():
    assert _normalize_uri("./milvus_demo.db") == "./milvus_demo.db"


def test_pack_int8_vector_produces_a_real_int8_numpy_array():
    """Regression test for a real bug found live, 2026-09-17: a plain Python list of
    floats was rejected outright by pymilvus's row-based upsert for an INT8_VECTOR field
    (`ParamError: invalid input for int8 vector. Expected an np.ndarray with
    dtype=int8`) — confirmed against a real 768-dim Zilliz Cloud collection
    ("Test768"). Milvus Lite doesn't support INT8_VECTOR at all (confirmed live: "Phase
    10.2 supports: ... FLOAT_VECTOR, SPARSE_FLOAT_VECTOR" only), so this can't be a real
    local server round-trip test the way the other Milvus tests are — it verifies the
    packing function directly instead.
    """
    packed = _pack_vector_for_milvus([1.0, -2.0, 127.0, -128.0], DataType.INT8_VECTOR.value)
    assert isinstance(packed, np.ndarray)
    assert packed.dtype == np.int8
    assert packed.tolist() == [1, -2, 127, -128]


def test_pack_float_vector_leaves_a_plain_list_untouched():
    # FLOAT_VECTOR is the one dense type pymilvus already accepts as a plain Python
    # list natively — confirmed by every other test in this file, which all use
    # FLOAT_VECTOR collections and never needed this conversion.
    vector = [0.1, 0.2, 0.3]
    assert _pack_vector_for_milvus(vector, DataType.FLOAT_VECTOR.value) is vector


def test_unpack_int8_vector_round_trips_through_real_bytes():
    """The read-side counterpart bug: row-based query() doesn't return an INT8_VECTOR
    field as a decoded list — it returns the raw bytes wrapped in a single-element list
    (`[b'...']`), confirmed live against the same real collection. Feeding that straight
    into VectorRecord (a list[float]) fails pydantic validation immediately."""
    original = [1, -2, 127, -128]
    raw_bytes = np.array(original, dtype=np.int8).tobytes()
    wrapped = [raw_bytes]  # the real, undocumented shape pymilvus actually returns
    assert _unpack_vector_from_milvus(wrapped, DataType.INT8_VECTOR.value) == original


def test_unpack_float_vector_leaves_an_already_decoded_list_untouched():
    vector = [0.1, 0.2, 0.3]
    assert _unpack_vector_from_milvus(vector, DataType.FLOAT_VECTOR.value) == vector


def test_unpack_unrecognized_bytes_type_returns_none_rather_than_guess():
    # BINARY_VECTOR/BFLOAT16_VECTOR have no verified decode (see _unpack_vector_from_milvus's
    # own docstring) — must return None, not silently pass raw bytes through to a
    # list[float] field or guess at a wrong decode.
    assert _unpack_vector_from_milvus([b"\x00\x01"], DataType.BINARY_VECTOR.value) is None


async def test_get_resource_info_raises_clearly_when_no_dense_vector_field(tmp_path):
    """Regression test for a real bug found live, 2026-09-17: a bare next() with no
    default let a collection with no matching dense-vector field raise a StopIteration
    straight out of this async method — reported by asyncio only as the opaque
    "coroutine raised StopIteration", no indication of which lookup failed or why. A
    collection whose only vector field is sparse (real, valid Milvus schema since 2.4)
    is exactly this case."""
    db_path = str(tmp_path / "milvus_sparse_only.db")
    sync = MilvusClient(uri=db_path)
    schema = sync.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    index_params = sync.prepare_index_params()
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    sync.create_collection(collection_name="sparse_only", schema=schema, index_params=index_params)
    sync.close()

    adapter = MilvusAdapter(uri=db_path)
    try:
        with pytest.raises(RuntimeError, match="no dense"):
            await adapter.get_resource_info("sparse_only")
    finally:
        await adapter.close()


def _make_collection(db_path: str, name: str, pk_type: DataType, metric: str = "COSINE") -> None:
    sync = MilvusClient(uri=db_path)
    if sync.has_collection(name):
        sync.drop_collection(name)
    schema = sync.create_schema(auto_id=False, enable_dynamic_field=True)
    if pk_type == DataType.VARCHAR:
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
    else:
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=4)
    index_params = sync.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type=metric)
    sync.create_collection(collection_name=name, schema=schema, index_params=index_params)
    sync.close()


def test_native_int_ids_pass_through_unmapped():
    assert _is_int64_native("12345") is True
    assert _to_milvus_int_id("12345") == 12345


def test_non_numeric_ids_are_hashed_deterministically():
    a = _to_milvus_int_id("cooking-6")
    b = _to_milvus_int_id("cooking-6")
    c = _to_milvus_int_id("cooking-7")
    assert a == b
    assert a != c
    assert isinstance(a, int) and a >= 0


async def test_round_trip_with_varchar_primary_key(tmp_path):
    db_path = str(tmp_path / "milvus_varchar.db")
    _make_collection(db_path, "col", DataType.VARCHAR)
    adapter = MilvusAdapter(uri=db_path)
    try:
        info = await adapter.get_resource_info("col")
        assert info.dimension == 4
        assert info.metric.value == "cosine"

        records = [
            VectorRecord(id="cooking-6", vector=[1.0, 0.0, 0.0, 0.0], payload={"topic": "cooking"}),
            VectorRecord(id="finance-7", vector=[0.0, 1.0, 0.0, 0.0], payload={"topic": "finance"}),
        ]
        written = await adapter.upsert_vectors("col", records)
        assert written == 2

        fetched = await adapter.fetch_by_ids("col", ["cooking-6", "finance-7"])
        by_id = {r.id: r for r in fetched}
        assert set(by_id) == {"cooking-6", "finance-7"}
        assert by_id["cooking-6"].payload == {"topic": "cooking"}

        matches = await adapter.query("col", [1.0, 0.0, 0.0, 0.0], top_k=1)
        assert matches[0].id == "cooking-6"

        page = await adapter.scan_vectors("col", batch_size=10)
        assert {r.id for r in page.records} == {"cooking-6", "finance-7"}

        deleted = await adapter.delete_vectors("col", ["finance-7"])
        assert deleted == 1
    finally:
        await adapter.close()


async def test_round_trip_with_int64_primary_key_maps_natural_string_ids(tmp_path):
    db_path = str(tmp_path / "milvus_int64.db")
    _make_collection(db_path, "col", DataType.INT64)
    adapter = MilvusAdapter(uri=db_path)
    try:
        records = [
            VectorRecord(id="cooking-6", vector=[1.0, 0.0, 0.0, 0.0], payload={"topic": "cooking"}),
        ]
        written = await adapter.upsert_vectors("col", records)
        assert written == 1

        fetched = await adapter.fetch_by_ids("col", ["cooking-6"])
        assert len(fetched) == 1
        assert fetched[0].id == "cooking-6"  # recovered via __source_id, never leaks the hash
        assert fetched[0].payload == {"topic": "cooking"}

        matches = await adapter.query("col", [1.0, 0.0, 0.0, 0.0], top_k=1)
        assert matches[0].id == "cooking-6"
    finally:
        await adapter.close()


async def test_namespace_fold_scopes_scan_and_fetch(tmp_path):
    db_path = str(tmp_path / "milvus_ns.db")
    _make_collection(db_path, "col", DataType.VARCHAR)
    adapter = MilvusAdapter(uri=db_path)
    try:
        await adapter.upsert_vectors(
            "col",
            [VectorRecord(id="a", vector=[1.0, 0.0, 0.0, 0.0], payload={})],
            namespace="tenant-1",
        )
        await adapter.upsert_vectors(
            "col",
            [VectorRecord(id="b", vector=[0.0, 1.0, 0.0, 0.0], payload={})],
            namespace="tenant-2",
        )
        page = await adapter.scan_vectors("col", batch_size=10, namespace="tenant-1")
        assert {r.id for r in page.records} == {"a"}
    finally:
        await adapter.close()
