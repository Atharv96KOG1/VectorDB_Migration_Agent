"""Proves the hand-rolled Pinecone REST adapter's request/response handling directly,
since nothing here is reachable live in this sandbox (no credentials, no network to a
real Pinecone project) — see core/adapters/pinecone_adapter.py's module docstring.
"""

from __future__ import annotations

import json

import httpx
import pytest

from core.adapters.base import VectorRecord
from core.adapters.pinecone_adapter import PineconeAdapter
from core.models.canonical_ir import Metric
from core.models.capability import Capability

INDEX_HOST = "test-index-abcd.svc.pinecone.io"


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    method = request.method

    if method == "GET" and path == "/indexes":
        return httpx.Response(200, json={"indexes": []})
    if method == "GET" and path == "/indexes/documents":
        return httpx.Response(
            200,
            json={"name": "documents", "dimension": 1536, "metric": "cosine", "host": INDEX_HOST},
        )
    if method == "POST" and path == "/describe_index_stats":
        return httpx.Response(200, json={"totalVectorCount": 42, "namespaces": {}})
    if method == "GET" and path == "/vectors/list":
        return httpx.Response(
            200, json={"vectors": [{"id": "id-1"}, {"id": "id-2"}], "pagination": {"next": "tok-2"}}
        )
    if method == "GET" and path == "/vectors/fetch":
        ids = request.url.params.get_list("ids")
        vectors = {i: {"values": [0.1, 0.2, 0.3], "metadata": {"k": "v"}} for i in ids}
        return httpx.Response(200, json={"vectors": vectors})
    if method == "POST" and path == "/vectors/upsert":
        body = json.loads(request.content)
        return httpx.Response(200, json={"upsertedCount": len(body["vectors"])})
    if method == "POST" and path == "/vectors/delete":
        return httpx.Response(200, json={})
    if method == "POST" and path == "/query":
        return httpx.Response(
            200, json={"matches": [{"id": "id-1", "score": 0.9}, {"id": "id-2", "score": 0.5}]}
        )
    return httpx.Response(404, json={"error": "not found", "path": path})


def _adapter(index_host: str | None = None) -> PineconeAdapter:
    return PineconeAdapter(
        api_key="test-key", index_host=index_host, transport=httpx.MockTransport(_handler)
    )


async def test_validate_credentials_true_on_200():
    adapter = _adapter()
    assert await adapter.validate_credentials() is True
    await adapter.close()


async def test_discover_capabilities_reports_unknown_not_false_for_undiscoverable_fields():
    adapter = _adapter()
    caps = await adapter.discover_capabilities()
    assert caps.representation.embedding_model is Capability.UNKNOWN
    assert caps.representation.normalization is Capability.UNKNOWN
    assert caps.connectivity.connect is Capability.TRUE
    await adapter.close()


async def test_get_resource_info_populates_dimension_metric_and_host():
    adapter = _adapter()
    info = await adapter.get_resource_info("documents")
    assert info.dimension == 1536
    assert info.metric == Metric.COSINE
    assert info.approximate_count == 42
    assert adapter._index_host == INDEX_HOST
    await adapter.close()


async def test_scan_vectors_lists_then_fetches_values():
    adapter = _adapter(index_host=INDEX_HOST)
    page = await adapter.scan_vectors("documents", batch_size=10)
    assert [r.id for r in page.records] == ["id-1", "id-2"]
    assert all(r.vector == [0.1, 0.2, 0.3] for r in page.records)
    assert page.next_cursor == "tok-2"
    await adapter.close()


async def test_upsert_vectors_returns_count():
    adapter = _adapter(index_host=INDEX_HOST)
    records = [VectorRecord(id="a", vector=[0.1, 0.2, 0.3], payload={})]
    written = await adapter.upsert_vectors("documents", records)
    assert written == 1
    await adapter.close()


async def test_upsert_vectors_empty_list_short_circuits_without_a_request():
    adapter = _adapter(index_host=INDEX_HOST)
    assert await adapter.upsert_vectors("documents", []) == 0
    await adapter.close()


async def test_query_returns_scored_matches():
    adapter = _adapter(index_host=INDEX_HOST)
    matches = await adapter.query("documents", [0.1, 0.2, 0.3], top_k=2)
    assert [m.id for m in matches] == ["id-1", "id-2"]
    assert matches[0].score == 0.9
    await adapter.close()


async def test_query_self_heals_missing_index_host_via_control_plane():
    # Confirmed live 2026-08-31: tools/benchmark_tools.py builds a FRESH PineconeAdapter
    # per call for ground-truth queries — that instance never had get_resource_info called
    # on it, even though a different adapter instance resolved the host during DISCOVER.
    # query() must self-heal the same way scan_vectors already does, not require the
    # caller to have called get_resource_info first.
    adapter = _adapter(index_host=None)
    matches = await adapter.query("documents", [0.1, 0.2, 0.3], top_k=2)
    assert [m.id for m in matches] == ["id-1", "id-2"]
    await adapter.close()


async def test_fetch_by_ids_self_heals_missing_index_host():
    # This exact bug hit live, 2026-08-31, right after query()'s self-heal fix: a
    # centralizing refactor of _data_client() closed it for every data-plane method at
    # once, since the same "fresh adapter, never called get_resource_info" gap applies to
    # all of them, not just whichever one happened to be exercised first.
    adapter = _adapter(index_host=None)
    records = await adapter.fetch_by_ids("documents", ["id-1"])
    assert records[0].id == "id-1"
    await adapter.close()


async def test_upsert_vectors_self_heals_missing_index_host():
    adapter = _adapter(index_host=None)
    written = await adapter.upsert_vectors(
        "documents", [VectorRecord(id="id-1", vector=[0.1, 0.2, 0.3], payload={})]
    )
    assert written == 1
    await adapter.close()


async def test_delete_vectors_self_heals_missing_index_host():
    adapter = _adapter(index_host=None)
    deleted = await adapter.delete_vectors("documents", ["id-1"])
    assert deleted == 1
    await adapter.close()


async def test_query_raises_clear_error_when_control_plane_has_no_host_either():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/indexes/hostless":
            return httpx.Response(200, json={"name": "hostless", "dimension": 8, "metric": "cosine"})
        return httpx.Response(404, json={"error": "not found"})

    adapter = PineconeAdapter(
        api_key="test-key", index_host=None, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(RuntimeError, match="index_host"):
        await adapter.query("hostless", [0.1], top_k=1)
    await adapter.close()


def _handler_with_spec(spec_type: str):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if method == "GET" and path == "/indexes":
            return httpx.Response(200, json={"indexes": []})
        if method == "GET" and path == "/indexes/documents":
            spec = (
                {"serverless": {"cloud": "aws", "region": "us-east-1"}}
                if spec_type == "serverless"
                else {"pod": {"environment": "us-east1-gcp"}}
            )
            return httpx.Response(
                200,
                json={
                    "name": "documents",
                    "dimension": 1536,
                    "metric": "cosine",
                    "host": INDEX_HOST,
                    "spec": spec,
                },
            )
        if method == "POST" and path == "/describe_index_stats":
            return httpx.Response(200, json={"totalVectorCount": 10, "namespaces": {}})
        if method == "GET" and path == "/vectors/list":
            return httpx.Response(200, json={"vectors": [{"id": "id-1"}], "pagination": {}})
        if method == "GET" and path == "/vectors/fetch":
            ids = request.url.params.get_list("ids")
            vectors = {i: {"values": [0.1, 0.2, 0.3], "metadata": {}} for i in ids}
            return httpx.Response(200, json={"vectors": vectors})
        return httpx.Response(404, json={"error": "not found", "path": path})

    return handler


async def test_scan_vectors_raises_clear_error_for_pod_based_index():
    # GET /vectors/list is serverless-only per Pinecone's live docs (checked 2026-08) —
    # a pod-based source must fail with a clear, actionable message, not a raw HTTP error.
    adapter = PineconeAdapter(
        api_key="test-key", transport=httpx.MockTransport(_handler_with_spec("pod"))
    )
    with pytest.raises(RuntimeError, match="pod-based"):
        await adapter.scan_vectors("documents", batch_size=10)
    await adapter.close()


async def test_scan_vectors_works_for_serverless_index():
    adapter = PineconeAdapter(
        api_key="test-key", transport=httpx.MockTransport(_handler_with_spec("serverless"))
    )
    page = await adapter.scan_vectors("documents", batch_size=10)
    assert len(page.records) == 1
    await adapter.close()


async def test_custom_api_version_is_sent_in_request_headers():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["version"] = request.headers.get("x-pinecone-api-version")
        return httpx.Response(200, json={"indexes": []})

    adapter = PineconeAdapter(
        api_key="test-key", transport=httpx.MockTransport(handler), api_version="2099-01"
    )
    await adapter.validate_credentials()
    assert seen["version"] == "2099-01"
    await adapter.close()


def test_default_api_version_is_not_a_stale_multi_year_old_pin():
    from core.adapters.pinecone_adapter import _DEFAULT_API_VERSION

    assert _DEFAULT_API_VERSION >= "2025-01"


def _paginated_list_handler(total_ids: int):
    """Mimics Pinecone's real, live-confirmed behavior: GET /vectors/list rejects any
    limit > 100 with a 400, and paginates via an opaque `paginationToken`."""
    all_ids = [f"id-{i}" for i in range(total_ids)]
    call_log: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if method == "GET" and path == "/indexes/documents":
            return httpx.Response(
                200,
                json={
                    "name": "documents",
                    "dimension": 3,
                    "metric": "cosine",
                    "host": INDEX_HOST,
                    "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
                },
            )
        if method == "GET" and path == "/vectors/list":
            limit = int(request.url.params.get("limit", "0"))
            call_log.append(limit)
            if limit > 100:
                return httpx.Response(
                    400,
                    json={"code": 3, "message": f"Limit must be greater than 0 and less than 100. Request limit was {limit}"},
                )
            token = request.url.params.get("paginationToken")
            offset = int(token) if token else 0
            page = all_ids[offset : offset + limit]
            next_offset = offset + len(page)
            body = {"vectors": [{"id": i} for i in page]}
            if next_offset < total_ids:
                body["pagination"] = {"next": str(next_offset)}
            return httpx.Response(200, json=body)
        if method == "GET" and path == "/vectors/fetch":
            ids = request.url.params.get_list("ids")
            vectors = {i: {"values": [0.1, 0.2, 0.3], "metadata": {}} for i in ids}
            return httpx.Response(200, json={"vectors": vectors})
        return httpx.Response(404, json={"error": "not found", "path": path})

    return handler, call_log


async def test_scan_vectors_never_sends_a_limit_over_100():
    handler, call_log = _paginated_list_handler(total_ids=30)
    adapter = PineconeAdapter(
        api_key="test-key", index_host=INDEX_HOST, transport=httpx.MockTransport(handler)
    )
    page = await adapter.scan_vectors("documents", batch_size=250)
    assert max(call_log) <= 100
    assert len(page.records) == 30  # exhausted the (small) mock corpus
    assert page.next_cursor is None
    await adapter.close()


async def test_scan_vectors_with_large_batch_size_chains_multiple_100_capped_pages():
    handler, call_log = _paginated_list_handler(total_ids=250)
    adapter = PineconeAdapter(
        api_key="test-key", index_host=INDEX_HOST, transport=httpx.MockTransport(handler)
    )
    page = await adapter.scan_vectors("documents", batch_size=250)
    assert len(call_log) == 3  # 100 + 100 + 50, all <= 100
    assert all(limit <= 100 for limit in call_log)
    assert len(page.records) == 250
    assert page.next_cursor is None

    # Confirm no id was skipped or duplicated across the chained pages.
    assert {r.id for r in page.records} == {f"id-{i}" for i in range(250)}
    await adapter.close()


async def test_scan_vectors_resumes_correctly_across_separate_calls():
    handler, call_log = _paginated_list_handler(total_ids=150)
    adapter = PineconeAdapter(
        api_key="test-key", index_host=INDEX_HOST, transport=httpx.MockTransport(handler)
    )
    first = await adapter.scan_vectors("documents", batch_size=100)
    assert len(first.records) == 100
    assert first.next_cursor is not None

    second = await adapter.scan_vectors("documents", batch_size=100, cursor=first.next_cursor)
    assert len(second.records) == 50
    assert second.next_cursor is None

    all_ids = {r.id for r in first.records} | {r.id for r in second.records}
    assert all_ids == {f"id-{i}" for i in range(150)}
    await adapter.close()
