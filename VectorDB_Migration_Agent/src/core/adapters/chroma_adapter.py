"""Chroma adapter — backed by `chromadb`'s Python client. Every connection mode this
adapter supports (`chromadb.EphemeralClient`/`PersistentClient`/`HttpClient`/
`CloudClient`) is sync-only — Chroma's `AsyncHttpClient` exists but only for the plain
HTTP-server mode, not local/Cloud, so every call here is wrapped in
`asyncio.to_thread(...)` uniformly across modes rather than mixing async and sync code
paths depending on how the adapter was constructed. `faiss_adapter.py` makes the same
choice for the same reason (its underlying library has no async API at all).

Every response shape here (get()/query()'s dict shapes, collection.metadata's
`hnsw:space` key, delete()'s `{"deleted": n}` return) was confirmed live against a real
local Chroma instance, not just recalled from chromadb's docs.

Chroma is the only one of the 7 providers this system supports where `retrieve_documents`
is a real, first-class TRUE: Chroma stores a `documents` text field natively alongside
each vector (not a payload convention). It's folded into `VectorRecord.payload["__document"]`
on read/write, the same way `__namespace` is folded in for every provider that has no
native tenant concept (Chroma has none — namespace becomes a `__namespace` metadata field
plus a `where` filter on it, matching `qdrant_adapter.py`'s convention).

IDs are arbitrary strings, like Pinecone — no id mapping needed.

Chroma exposes no vector-dimension config directly (unlike Pinecone/Qdrant's index
config) — `get_resource_info` infers it from one real stored vector via `get(limit=1,
include=["embeddings"])`, so it's `None` for a genuinely empty collection rather than a
fabricated value.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

import chromadb

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

_NAMESPACE_FIELD = "__namespace"
_DOCUMENT_FIELD = "__document"

_METRIC_MAP = {
    "cosine": Metric.COSINE,
    "l2": Metric.EUCLIDEAN,
    "ip": Metric.DOT,
}


def _build_client(
    endpoint: str | None, credential_ref: str | None
) -> chromadb.api.ClientAPI:
    if endpoint in (None, "", "memory", ":memory:"):
        return chromadb.EphemeralClient()
    if endpoint.startswith("path://"):
        return chromadb.PersistentClient(path=endpoint[len("path://") :])
    if endpoint.startswith("cloud://"):
        # cloud://<tenant>/<database> — Chroma Cloud has no index/collection-name-only
        # addressing scheme; tenant+database are required alongside the api key.
        tenant, _, database = endpoint[len("cloud://") :].partition("/")
        return chromadb.CloudClient(tenant=tenant, database=database, api_key=credential_ref)
    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    headers = {"Authorization": f"Bearer {credential_ref}"} if credential_ref else None
    return chromadb.HttpClient(
        host=parsed.hostname or "localhost",
        port=parsed.port or 8000,
        ssl=parsed.scheme == "https",
        headers=headers,
    )


class ChromaAdapter(VectorDBAdapter):
    provider = "chroma"

    def __init__(self, endpoint: str | None = None, credential_ref: str | None = None) -> None:
        self._client = _build_client(endpoint, credential_ref)
        self._collections: dict[str, object] = {}

    def _collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = self._client.get_collection(name)
        return self._collections[name]

    async def validate_credentials(self) -> bool:
        try:
            await asyncio.to_thread(self._client.list_collections)
            self.last_connect_error = None
            return True
        except Exception as exc:
            self.last_connect_error = f"{type(exc).__name__}: {exc}"
            return False

    async def discover_capabilities(self) -> CapabilityReport:
        connected = await self.validate_credentials()
        connect_cap = Capability.TRUE if connected else Capability.FALSE
        return CapabilityReport(
            provider=self.provider,
            connectivity=ConnectivityCapabilities(connect=connect_cap, tls=Capability.UNKNOWN),
            data=DataCapabilities(
                list_vectors=Capability.TRUE,
                scan_vectors=Capability.TRUE,
                retrieve_vectors=Capability.TRUE,
                retrieve_payload=Capability.TRUE,
                retrieve_documents=Capability.TRUE,
                cdc=Capability.FALSE,
            ),
            representation=RepresentationCapabilities(
                dimension=Capability.TRUE,
                metric=Capability.TRUE,
                datatype=Capability.TRUE,
                dense_vectors=Capability.TRUE,
                sparse_vectors=Capability.FALSE,
                named_vectors=Capability.FALSE,
                normalization=Capability.UNKNOWN,
                embedding_model=Capability.UNKNOWN,
                quantized_storage=Capability.FALSE,
            ),
            index=IndexCapabilities(
                retrieve_config=Capability.TRUE, configure_index=Capability.TRUE
            ),
            migration=MigrationCapabilities(
                resumable_scan=Capability.TRUE, export=Capability.FALSE, cdc=Capability.FALSE
            ),
        )

    async def get_resource_info(self, name: str, namespace: str | None = None) -> ResourceInfo:
        collection = await asyncio.to_thread(self._collection, name)
        metric = _METRIC_MAP.get((collection.metadata or {}).get("hnsw:space"), Metric.UNKNOWN)
        count = await asyncio.to_thread(collection.count)

        dimension = None
        if count > 0:
            sample = await asyncio.to_thread(collection.get, limit=1, include=["embeddings"])
            embeddings = sample.get("embeddings")
            if embeddings is not None and len(embeddings) > 0:
                dimension = len(embeddings[0])

        return ResourceInfo(
            name=name,
            dimension=dimension,
            metric=metric,
            datatype=DataType.FLOAT32,
            quantized=Capability.FALSE,
            approximate_count=count,
        )

    def _to_record(self, id_: str, embedding, metadata: dict | None, document: str | None) -> VectorRecord:
        payload = dict(metadata or {})
        namespace = payload.pop(_NAMESPACE_FIELD, None)
        if document is not None:
            payload[_DOCUMENT_FIELD] = document
        vector = list(embedding) if embedding is not None else None
        return VectorRecord(id=id_, vector=vector, payload=payload, namespace=namespace)

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
        collection = await asyncio.to_thread(self._collection, name)
        offset = int(cursor) if cursor else 0
        where = {_NAMESPACE_FIELD: namespace} if namespace is not None else None
        # Chroma Cloud enforces a per-request `limit` quota well below what a real
        # DRY_RUN/BENCHMARK sample_size can ask for (confirmed live, 2026-09-11: a plan
        # capped at 300 rejected a single get(limit=400) with a 422 quota error) — chained
        # sub-300 pages instead of one big request, same fix pinecone_adapter.py already
        # applies for GET /vectors/list's hard 100-per-page cap. Harmless extra round
        # trips on self-hosted/local Chroma, which has no such limit.
        page_limit = 300
        ids: list[str] = []
        embeddings: list = []
        metadatas: list = []
        documents: list = []
        current_offset = offset
        while len(ids) < batch_size:
            request_limit = min(page_limit, batch_size - len(ids))
            page = await asyncio.to_thread(
                collection.get,
                where=where,
                limit=request_limit,
                offset=current_offset,
                include=["embeddings", "metadatas", "documents"],
            )
            page_ids = page.get("ids") or []
            if not page_ids:
                break
            ids.extend(page_ids)
            page_embeddings = page.get("embeddings")
            embeddings.extend(page_embeddings if page_embeddings is not None else [None] * len(page_ids))
            metadatas.extend(page.get("metadatas") or [None] * len(page_ids))
            documents.extend(page.get("documents") or [None] * len(page_ids))
            current_offset += len(page_ids)
            if len(page_ids) < request_limit:
                break  # short page: no more data left to fetch

        records = [
            self._to_record(id_, embeddings[i], metadatas[i], documents[i])
            for i, id_ in enumerate(ids)
        ]
        next_cursor = str(offset + len(ids)) if len(ids) == batch_size else None
        return ScanPage(records=records, next_cursor=next_cursor)

    async def upsert_vectors(
        self, name: str, records: list[VectorRecord], namespace: str | None = None
    ) -> int:
        if not records:
            return 0
        collection = await asyncio.to_thread(self._collection, name)
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        for r in records:
            ids.append(r.id)
            embeddings.append(r.vector)
            payload = dict(r.payload)
            document = payload.pop(_DOCUMENT_FIELD, None)
            ns = namespace if namespace is not None else r.namespace
            if ns is not None:
                payload[_NAMESPACE_FIELD] = ns
            metadatas.append(payload or None)
            documents.append(document)
        await asyncio.to_thread(
            collection.upsert,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas if any(metadatas) else None,
            documents=documents if any(documents) else None,
        )
        return len(ids)

    async def fetch_by_ids(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> list[VectorRecord]:
        if not ids:
            return []
        collection = await asyncio.to_thread(self._collection, name)
        where = {_NAMESPACE_FIELD: namespace} if namespace is not None else None
        # Same per-request quota chunking as scan_vectors — a get(ids=[...]) with more
        # than the plan's item-count quota fails the same way a large `limit` does.
        page_limit = 300
        records: list[VectorRecord] = []
        for i in range(0, len(ids), page_limit):
            chunk = ids[i : i + page_limit]
            result = await asyncio.to_thread(
                collection.get,
                ids=chunk,
                where=where,
                include=["embeddings", "metadatas", "documents"],
            )
            result_ids = result.get("ids") or []
            embeddings = result.get("embeddings")
            metadatas = result.get("metadatas") or [None] * len(result_ids)
            documents = result.get("documents") or [None] * len(result_ids)
            records.extend(
                self._to_record(
                    id_, embeddings[j] if embeddings is not None else None, metadatas[j], documents[j]
                )
                for j, id_ in enumerate(result_ids)
            )
        return records

    async def delete_vectors(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> int:
        if not ids:
            return 0
        collection = await asyncio.to_thread(self._collection, name)
        result = await asyncio.to_thread(collection.delete, ids=ids)
        return result.get("deleted", len(ids)) if isinstance(result, dict) else len(ids)

    async def query(
        self,
        name: str,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
    ) -> list[ScoredMatch]:
        collection = await asyncio.to_thread(self._collection, name)
        where = {_NAMESPACE_FIELD: namespace} if namespace is not None else None
        result = await asyncio.to_thread(
            collection.query,
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [ScoredMatch(id=id_, score=float(dist)) for id_, dist in zip(ids, distances)]

    async def close(self) -> None:
        return None
