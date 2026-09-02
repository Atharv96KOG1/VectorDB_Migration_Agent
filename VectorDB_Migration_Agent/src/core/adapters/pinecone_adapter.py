"""Pinecone adapter — hand-rolled REST client over httpx rather than the official SDK.

Two API surfaces: the *control plane* (`https://api.pinecone.io`, index metadata: name,
dimension, metric, host) and the *data plane* (`https://{index-host}`, vectors: list,
fetch, upsert, query, describe_index_stats). Neither is reachable from this sandbox (no
credentials, no network to a live Pinecone project), so correctness here is proven by
tests/unit/test_pinecone_adapter.py using httpx.MockTransport — every request's method,
path, headers, and body shape is asserted directly, not merely "it returns something."

Pinecone has no bulk-vector scan endpoint: `scan_vectors` lists ids (`GET /vectors/list`,
opaque `paginationToken` cursor) then fetches values+metadata for that page
(`GET /vectors/fetch`), two calls per page. `GET /vectors/list` is **serverless-only** —
pod-based indexes reject it — so `get_resource_info` records the index's `spec` type from
the control-plane response and `scan_vectors`/`sample_vectors` fail fast with a clear
message on a pod index instead of surfacing a confusing raw HTTP error.

Every request/response shape here (headers, field names, endpoint paths) was checked
against Pinecone's live API reference docs (docs.pinecone.io) as of 2026-08, not just
recalled from training data — see docs/ARCHITECTURE.md. Pinecone versions its API via
the `X-Pinecone-Api-Version` header with a documented ~12-month support window per
version (quarterly releases); `api_version` is a constructor parameter, not just a
hardcoded constant, specifically so a long-lived deployment can pin/upgrade it without a
code change as versions age out.
"""

from __future__ import annotations

import httpx

from core.adapters.base import (
    ResourceInfo,
    ScanPage,
    ScoredMatch,
    VectorDBAdapter,
    VectorRecord,
)
from core.models.canonical_ir import DataType, Metric
from core.models.capability import (
    Capability,
    CapabilityReport,
    ConnectivityCapabilities,
    DataCapabilities,
    IndexCapabilities,
    MigrationCapabilities,
    RepresentationCapabilities,
)

_DEFAULT_API_VERSION = "2026-04"  # last verified against docs.pinecone.io on 2026-08-26
_LIST_MAX_LIMIT = 100  # GET /vectors/list per-request cap, confirmed live 2026-08-26

_METRIC_MAP = {
    "cosine": Metric.COSINE,
    "dotproduct": Metric.DOT,
    "euclidean": Metric.EUCLIDEAN,
}


class PineconeAdapter(VectorDBAdapter):
    provider = "pinecone"

    def __init__(
        self,
        api_key: str,
        index_host: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        api_version: str = _DEFAULT_API_VERSION,
    ) -> None:
        headers = {"Api-Key": api_key, "X-Pinecone-Api-Version": api_version}
        self._control = httpx.AsyncClient(
            base_url="https://api.pinecone.io", headers=headers, transport=transport
        )
        self._data: httpx.AsyncClient | None = None
        self._index_host = index_host
        self._transport = transport
        self._headers = headers
        self._index_spec_type: str | None = None  # "serverless" | "pod" | None (undiscovered)

    async def _data_client(self, name: str | None = None) -> httpx.AsyncClient:
        if self._data is None:
            if not self._index_host and name is not None:
                # Self-heal: a freshly constructed adapter for THIS call (e.g. a new
                # PineconeAdapter built per-tool-call) never had get_resource_info called
                # on it, even if a different adapter instance already resolved the host
                # earlier in the same migration. Same fix applied to query() on
                # 2026-08-31 after a live ReadTimeout-adjacent bug; found AGAIN live in
                # fetch_by_ids() when this centralizing refactor was made — every
                # data-plane method needs this, not just the one that happened to be
                # exercised first.
                await self.get_resource_info(name)
            if not self._index_host:
                raise RuntimeError(
                    "PineconeAdapter needs index_host (from control-plane describe or "
                    "target_endpoint_ref) before any data-plane call."
                )
            self._data = httpx.AsyncClient(
                base_url=f"https://{self._index_host}",
                headers=self._headers,
                transport=self._transport,
            )
        return self._data

    async def validate_credentials(self) -> bool:
        try:
            resp = await self._control.get("/indexes")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def discover_capabilities(self) -> CapabilityReport:
        connected = await self.validate_credentials()
        connect_cap = Capability.TRUE if connected else Capability.FALSE
        return CapabilityReport(
            provider=self.provider,
            connectivity=ConnectivityCapabilities(connect=connect_cap, tls=Capability.TRUE),
            data=DataCapabilities(
                # list/scan is serverless-only (GET /vectors/list rejects pod indexes) —
                # this is a per-INDEX fact, not a provider-wide one, and discover_capabilities
                # has no resource context yet, so it stays UNKNOWN here. get_resource_info
                # resolves it definitively per resource; scan_vectors fails fast and clearly
                # rather than surfacing Pinecone's raw HTTP error if it turns out to be a pod index.
                list_vectors=Capability.UNKNOWN,
                scan_vectors=Capability.UNKNOWN,
                retrieve_vectors=Capability.TRUE,
                retrieve_payload=Capability.TRUE,
                retrieve_documents=Capability.UNKNOWN,
                cdc=Capability.FALSE,
            ),
            representation=RepresentationCapabilities(
                dimension=Capability.TRUE,
                metric=Capability.TRUE,
                datatype=Capability.TRUE,
                dense_vectors=Capability.TRUE,
                sparse_vectors=Capability.TRUE,
                named_vectors=Capability.FALSE,
                normalization=Capability.UNKNOWN,
                embedding_model=Capability.UNKNOWN,
                quantized_storage=Capability.UNKNOWN,
            ),
            index=IndexCapabilities(
                retrieve_config=Capability.TRUE, configure_index=Capability.TRUE
            ),
            migration=MigrationCapabilities(
                resumable_scan=Capability.UNKNOWN, export=Capability.FALSE, cdc=Capability.FALSE
            ),
        )

    async def get_resource_info(self, name: str, namespace: str | None = None) -> ResourceInfo:
        resp = await self._control.get(f"/indexes/{name}")
        resp.raise_for_status()
        data = resp.json()
        if data.get("host"):
            self._index_host = data["host"]

        spec = data.get("spec") or {}
        if "serverless" in spec:
            self._index_spec_type = "serverless"
        elif "pod" in spec:
            self._index_spec_type = "pod"

        count = None
        try:
            client = await self._data_client()
            stats_resp = await client.post("/describe_index_stats", json={})
            stats_resp.raise_for_status()
            stats = stats_resp.json()
            if namespace:
                count = stats.get("namespaces", {}).get(namespace, {}).get("vectorCount")
            else:
                count = stats.get("totalVectorCount")
        except (httpx.HTTPError, RuntimeError):
            pass

        return ResourceInfo(
            name=name,
            dimension=data.get("dimension"),
            metric=_METRIC_MAP.get(data.get("metric", ""), Metric.UNKNOWN),
            datatype=DataType.FLOAT32,
            quantized=Capability.UNKNOWN,
            approximate_count=count,
        )

    async def sample_vectors(
        self, name: str, n: int, namespace: str | None = None
    ) -> list[VectorRecord]:
        page = await self.scan_vectors(name, batch_size=n, cursor=None, namespace=namespace)
        return page.records

    async def scan_vectors(
        self,
        name: str,
        batch_size: int,
        cursor: str | None = None,
        namespace: str | None = None,
    ) -> ScanPage:
        if self._index_spec_type is None:
            # get_resource_info hasn't run yet for this index — discover it now so the
            # error below (if any) is the clear one, not Pinecone's raw 400.
            await self.get_resource_info(name, namespace=namespace)
        if self._index_spec_type == "pod":
            raise RuntimeError(
                f"index {name!r} is pod-based; Pinecone's GET /vectors/list (used for "
                f"scan_vectors/sample_vectors) only supports serverless indexes. A "
                f"pod-based source cannot be scanned by this adapter — see "
                f"core/adapters/pinecone_adapter.py's module docstring."
            )

        client = await self._data_client()
        ids: list[str] = []
        next_cursor = cursor
        # GET /vectors/list caps `limit` at 100 regardless of batch_size (confirmed live
        # against a real Pinecone index, 2026-08: limit=101 returns 400 "Limit must be
        # greater than 0 and less than 100"). A caller asking for a larger batch_size is
        # served transparently by chaining internal 100-capped pages — batch_size is this
        # method's contract, not Pinecone's per-request limit.
        while len(ids) < batch_size:
            page_limit = min(_LIST_MAX_LIMIT, batch_size - len(ids))
            params: dict = {"limit": page_limit}
            if namespace:
                params["namespace"] = namespace
            if next_cursor:
                params["paginationToken"] = next_cursor

            list_resp = await client.get("/vectors/list", params=params)
            list_resp.raise_for_status()
            list_data = list_resp.json()
            page_ids = [v["id"] for v in list_data.get("vectors", [])]
            ids.extend(page_ids)
            next_cursor = list_data.get("pagination", {}).get("next")
            if not next_cursor or not page_ids:
                break

        records = await self.fetch_by_ids(name, ids, namespace=namespace)
        return ScanPage(records=records, next_cursor=next_cursor)

    async def upsert_vectors(
        self, name: str, records: list[VectorRecord], namespace: str | None = None
    ) -> int:
        # Idempotent on VectorRecord.id: Pinecone upsert overwrites the existing vector
        # for that id (V2 §39) — safe under Temporal's own activity RetryPolicy.
        if not records:
            return 0
        client = await self._data_client(name)
        vectors = [{"id": r.id, "values": r.vector, "metadata": r.payload} for r in records]
        body: dict = {"vectors": vectors}
        if namespace:
            body["namespace"] = namespace
        resp = await client.post("/vectors/upsert", json=body)
        resp.raise_for_status()
        data = resp.json()
        return data.get("upsertedCount", len(vectors))

    async def fetch_by_ids(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> list[VectorRecord]:
        if not ids:
            return []
        client = await self._data_client(name)
        params: dict = {"ids": ids}
        if namespace:
            params["namespace"] = namespace
        resp = await client.get("/vectors/fetch", params=params)
        resp.raise_for_status()
        data = resp.json()
        records = []
        for vid, v in data.get("vectors", {}).items():
            records.append(
                VectorRecord(
                    id=vid,
                    vector=v.get("values"),
                    payload=v.get("metadata") or {},
                    namespace=namespace,
                )
            )
        return records

    async def delete_vectors(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> int:
        if not ids:
            return 0
        client = await self._data_client(name)
        body: dict = {"ids": ids}
        if namespace:
            body["namespace"] = namespace
        resp = await client.post("/vectors/delete", json=body)
        resp.raise_for_status()
        return len(ids)

    async def query(
        self,
        name: str,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
    ) -> list[ScoredMatch]:
        client = await self._data_client(name)
        body: dict = {"vector": vector, "topK": top_k, "includeValues": False}
        if namespace:
            body["namespace"] = namespace
        resp = await client.post("/query", json=body)
        resp.raise_for_status()
        data = resp.json()
        return [ScoredMatch(id=m["id"], score=m["score"]) for m in data.get("matches", [])]

    async def close(self) -> None:
        await self._control.aclose()
        if self._data is not None:
            await self._data.aclose()
