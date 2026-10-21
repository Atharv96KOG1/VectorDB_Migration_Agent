"""Proves the hand-rolled Vespa REST adapter's request/response handling directly against
Vespa's documented Document API v1 shape, the same approach test_pinecone_adapter.py uses
for the same reason: no live instance was available while this adapter was first written
(see core/adapters/vespa_adapter.py's module docstring — this one is later live-verified
separately, unlike Pinecone's).
"""

from __future__ import annotations

import json

import httpx

from core.adapters.base import VectorRecord
from core.adapters.vespa_adapter import VespaAdapter, _split_resource

NAMESPACE = "mynamespace"
DOC_TYPE = "product"
RESOURCE = f"{NAMESPACE}/{DOC_TYPE}"

_STORE: dict[str, dict] = {}


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    method = request.method
    prefix = f"/document/v1/{NAMESPACE}/{DOC_TYPE}/docid"

    if method == "GET" and path == "/ApplicationStatus":
        return httpx.Response(200, json={"status": "up"})

    if method == "POST" and path.startswith(prefix + "/"):
        doc_id = path[len(prefix + "/") :]
        body = json.loads(request.content)
        _STORE[doc_id] = body["fields"]
        return httpx.Response(200, json={"pathId": path, "id": f"id:{NAMESPACE}:{DOC_TYPE}::{doc_id}"})

    if method == "GET" and path.startswith(prefix + "/"):
        doc_id = path[len(prefix + "/") :]
        if doc_id not in _STORE:
            return httpx.Response(404, json={"pathId": path})
        return httpx.Response(
            200,
            json={
                "pathId": path,
                "id": f"id:{NAMESPACE}:{DOC_TYPE}::{doc_id}",
                "fields": _STORE[doc_id],
            },
        )

    if method == "DELETE" and path.startswith(prefix + "/"):
        doc_id = path[len(prefix + "/") :]
        if doc_id not in _STORE:
            return httpx.Response(404, json={"pathId": path})
        del _STORE[doc_id]
        return httpx.Response(200, json={"pathId": path, "id": f"id:{NAMESPACE}:{DOC_TYPE}::{doc_id}"})

    if method == "GET" and path == prefix:
        documents = [
            {"id": f"id:{NAMESPACE}:{DOC_TYPE}::{doc_id}", "fields": fields}
            for doc_id, fields in _STORE.items()
        ]
        return httpx.Response(200, json={"pathId": path, "documents": documents})

    if method == "POST" and path == "/search/":
        body = json.loads(request.content)
        assert "nearestNeighbor" in body["yql"]
        children = [
            {"id": f"id:{NAMESPACE}:{DOC_TYPE}::{doc_id}", "relevance": 0.9}
            for doc_id in list(_STORE)[:2]
        ]
        return httpx.Response(200, json={"root": {"children": children}})

    return httpx.Response(404, json={"error": "not found", "path": path})


def _adapter() -> VespaAdapter:
    _STORE.clear()
    return VespaAdapter(
        endpoint="https://vespa.example.com", transport=httpx.MockTransport(_handler)
    )


def test_split_resource_requires_namespace_and_doc_type():
    assert _split_resource("mynamespace/product") == ("mynamespace", "product")


def test_split_resource_rejects_a_bare_name():
    import pytest

    with pytest.raises(ValueError, match="document-type"):
        _split_resource("just-a-name")


async def test_validate_credentials_true_on_200():
    adapter = _adapter()
    assert await adapter.validate_credentials() is True
    await adapter.close()


async def test_upsert_fetch_delete_round_trip():
    adapter = _adapter()
    records = [
        VectorRecord(id="sku-1", vector=[1.0, 0.0, 0.0, 0.0], payload={"topic": "cooking"}),
        VectorRecord(id="sku-2", vector=[0.0, 1.0, 0.0, 0.0], payload={"topic": "finance"}),
    ]
    written = await adapter.upsert_vectors(RESOURCE, records)
    assert written == 2

    fetched = await adapter.fetch_by_ids(RESOURCE, ["sku-1", "sku-2"])
    by_id = {r.id: r for r in fetched}
    assert set(by_id) == {"sku-1", "sku-2"}
    assert by_id["sku-1"].payload == {"topic": "cooking"}
    assert by_id["sku-1"].vector == [1.0, 0.0, 0.0, 0.0]

    deleted = await adapter.delete_vectors(RESOURCE, ["sku-2"])
    assert deleted == 1
    assert await adapter.fetch_by_ids(RESOURCE, ["sku-2"]) == []
    await adapter.close()


async def test_scan_vectors_returns_continuation_cursor():
    adapter = _adapter()
    await adapter.upsert_vectors(
        RESOURCE, [VectorRecord(id="sku-1", vector=[1.0, 0.0, 0.0, 0.0], payload={})]
    )
    page = await adapter.scan_vectors(RESOURCE, batch_size=10)
    assert {r.id for r in page.records} == {"sku-1"}
    await adapter.close()


async def test_query_builds_near_neighbor_yql_and_parses_hits():
    adapter = _adapter()
    await adapter.upsert_vectors(
        RESOURCE, [VectorRecord(id="sku-1", vector=[1.0, 0.0, 0.0, 0.0], payload={})]
    )
    matches = await adapter.query(RESOURCE, [1.0, 0.0, 0.0, 0.0], top_k=2)
    assert matches[0].id == "sku-1"
    await adapter.close()


async def test_namespace_fold_is_written_and_recovered():
    adapter = _adapter()
    await adapter.upsert_vectors(
        RESOURCE,
        [VectorRecord(id="sku-1", vector=[1.0, 0.0, 0.0, 0.0], payload={})],
        namespace="tenant-1",
    )
    fetched = await adapter.fetch_by_ids(RESOURCE, ["sku-1"])
    assert fetched[0].namespace == "tenant-1"
    assert "__namespace" not in fetched[0].payload
    await adapter.close()
