"""Weaviate adapter — backed by `weaviate-client` v4's collections API
(`WeaviateAsyncClient`, via `weaviate.use_async_with_weaviate_cloud`/
`use_async_with_custom`/`use_async_with_local`).

Unlike Milvus/Chroma/FAISS, no local/embedded verification was possible for this adapter
in this sandbox: `weaviate.connect_to_embedded()` downloads a real server binary at
connect time, and that download failed here on a certificate-verification error blocking
outbound HTTPS to the download host. This adapter was built directly against
weaviate-client's own source (`.pyi` stubs under `collections/`) rather than docs/memory,
but — unlike every other adapter in this codebase — its live-instance smoke test is the
FIRST real exercise of this code end-to-end, not a confirmation of behavior already seen
working locally. Treat any discrepancy the live test surfaces as expected, not a red flag.

IDs: Weaviate v4 requires a valid UUID for every object id — like Qdrant, a non-UUID
source id is mapped through a deterministic uuid5 (same input -> same output, so
idempotent upsert holds) and the original is stashed in a reserved property
(`__source_id`) for recovery on every read path.

Upsert: there is no single "insert or overwrite" primitive in the v4 data API (`insert`
errors if the uuid already exists; `replace` errors if it doesn't) — this adapter checks
`data.exists(uuid)` first and calls the right one, at the cost of one extra round trip per
record. Simpler and unambiguously correct, matching the ABC's idempotent-on-retry
requirement, over guessing at batch-import's real overwrite semantics without a live
instance to confirm them against.

Namespace: Weaviate's real multi-tenancy feature (a collection explicitly configured with
tenants) isn't used here — detecting whether a given collection has it enabled and
routing every call through the right tenant adds real complexity for no immediate
requirement. Namespace is folded into a reserved `__namespace` property + a `Filter`, the
same convention `qdrant_adapter.py`/`chroma_adapter.py` use.

Metric: read from the collection's (possibly named) vector index config
(`distance_metric`) — HAMMING has no `Metric` equivalent and reports UNKNOWN, same
tri-state rule as everywhere else. `sparse_vectors` is FALSE: Weaviate's hybrid/BM25
search isn't the same shape as this system's sparse-vector concept.
"""

from __future__ import annotations

import uuid as uuid_lib
from urllib.parse import urlparse

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery

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
_ORIGINAL_ID_FIELD = "__source_id"
_ID_NAMESPACE = uuid_lib.UUID("9d3e6b2a-7c14-4a1f-9e0d-2b6a5f8c1e3d")  # fixed, arbitrary

_METRIC_MAP = {
    "cosine": Metric.COSINE,
    "dot": Metric.DOT,
    "l2-squared": Metric.EUCLIDEAN,
    "manhattan": Metric.MANHATTAN,
    # "hamming" (binary-vector metric) has no Metric equivalent — UNKNOWN, never guessed.
}


def _is_weaviate_native_id(value: str) -> bool:
    try:
        uuid_lib.UUID(value)
        return True
    except ValueError:
        return False


def _to_weaviate_id(original_id: str) -> str:
    if _is_weaviate_native_id(original_id):
        return original_id
    return str(uuid_lib.uuid5(_ID_NAMESPACE, original_id))


def _build_client(
    endpoint: str | None, credential_ref: str | None
) -> weaviate.WeaviateAsyncClient:
    auth = Auth.api_key(credential_ref) if credential_ref else None
    if endpoint in (None, "", "memory", ":memory:"):
        return weaviate.use_async_with_local(auth_credentials=auth)
    if ".weaviate.cloud" in endpoint or ".weaviate.network" in endpoint:
        return weaviate.use_async_with_weaviate_cloud(cluster_url=endpoint, auth_credentials=auth)
    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    secure = parsed.scheme == "https"
    return weaviate.use_async_with_custom(
        http_host=parsed.hostname or "localhost",
        http_port=parsed.port or (443 if secure else 80),
        http_secure=secure,
        grpc_host=parsed.hostname or "localhost",
        grpc_port=50051,
        grpc_secure=secure,
        auth_credentials=auth,
    )


class WeaviateAdapter(VectorDBAdapter):
    provider = "weaviate"

    def __init__(self, endpoint: str | None = None, credential_ref: str | None = None) -> None:
        self._client = _build_client(endpoint, credential_ref)
        self._connected = False
        self._vector_names: dict[str, str | None] = {}

    async def _ensure_connected(self) -> None:
        if not self._connected:
            await self._client.connect()
            self._connected = True

    def _namespace_filter(self, namespace: str | None):
        if namespace is None:
            return None
        return Filter.by_property(_NAMESPACE_FIELD).equal(namespace)

    async def validate_credentials(self) -> bool:
        try:
            await self._ensure_connected()
            ready = await self._client.is_ready()
            if ready:
                self.last_connect_error = None
            else:
                self.last_connect_error = "Weaviate reported not ready (is_ready() returned False)"
            return ready
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
                retrieve_documents=Capability.UNKNOWN,
                cdc=Capability.FALSE,
            ),
            representation=RepresentationCapabilities(
                dimension=Capability.TRUE,
                metric=Capability.TRUE,
                datatype=Capability.TRUE,
                dense_vectors=Capability.TRUE,
                sparse_vectors=Capability.FALSE,
                named_vectors=Capability.TRUE,
                normalization=Capability.UNKNOWN,
                embedding_model=Capability.UNKNOWN,
                quantized_storage=Capability.UNKNOWN,
            ),
            index=IndexCapabilities(
                retrieve_config=Capability.TRUE, configure_index=Capability.TRUE
            ),
            migration=MigrationCapabilities(
                resumable_scan=Capability.TRUE, export=Capability.FALSE, cdc=Capability.FALSE
            ),
        )

    async def _resolve_vector_name(self, name: str) -> str | None:
        if name not in self._vector_names:
            await self.get_resource_info(name)
        return self._vector_names[name]

    async def get_resource_info(self, name: str, namespace: str | None = None) -> ResourceInfo:
        await self._ensure_connected()
        collection = self._client.collections.get(name)
        config = await collection.config.get()

        dimension = None
        metric = Metric.UNKNOWN
        vector_name = None
        vector_config = getattr(config, "vector_config", None) or {}
        if vector_config:
            first_name, vc = next(iter(vector_config.items()))
            distance = getattr(vc.vector_index_config, "distance_metric", None)
            if distance is not None:
                metric = _METRIC_MAP.get(distance.value, Metric.UNKNOWN)
            # A single-vector collection's one entry (confirmed live: even one literally
            # named "default") is NOT a valid target_vector for near_vector/insert — it
            # must be addressed as the collection's implicit vector (omit target_vector
            # entirely / pass a plain vector, not {name: vector}). Only a genuinely
            # multi-named-vector collection needs the real name for targeting.
            if len(vector_config) > 1:
                vector_name = first_name
        self._vector_names[name] = vector_name

        count = None
        try:
            agg = await collection.aggregate.over_all(total_count=True)
            count = agg.total_count
        except Exception:
            pass

        # Weaviate's config never exposes vector dimension directly (unlike Pinecone's/
        # Qdrant's index config) — infer it from one real stored vector, same approach
        # chroma_adapter.py uses for the same gap. None for a genuinely empty collection.
        if count:
            sample = await collection.query.fetch_objects(limit=1, include_vector=True)
            if sample.objects:
                sample_vector = sample.objects[0].vector
                if isinstance(sample_vector, dict):
                    sample_vector = next(iter(sample_vector.values()), None)
                if sample_vector is not None:
                    dimension = len(sample_vector)

        return ResourceInfo(
            name=name,
            dimension=dimension,
            metric=metric,
            datatype=DataType.FLOAT32,
            quantized=Capability.UNKNOWN,
            approximate_count=count,
        )

    def _to_record(self, obj) -> VectorRecord:
        properties = dict(obj.properties or {})
        namespace = properties.pop(_NAMESPACE_FIELD, None)
        original_id = properties.pop(_ORIGINAL_ID_FIELD, None)
        vector = obj.vector.get("default") if isinstance(obj.vector, dict) else obj.vector
        return VectorRecord(
            id=original_id or str(obj.uuid), vector=vector, payload=properties, namespace=namespace
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
        await self._ensure_connected()
        collection = self._client.collections.get(name)
        result = await collection.query.fetch_objects(
            limit=batch_size,
            after=cursor,
            filters=self._namespace_filter(namespace),
            include_vector=True,
        )
        records = [self._to_record(o) for o in result.objects]
        next_cursor = str(result.objects[-1].uuid) if len(result.objects) == batch_size else None
        return ScanPage(records=records, next_cursor=next_cursor)

    async def upsert_vectors(
        self, name: str, records: list[VectorRecord], namespace: str | None = None
    ) -> int:
        if not records:
            return 0
        await self._ensure_connected()
        vector_name = await self._resolve_vector_name(name)
        collection = self._client.collections.get(name)
        written = 0
        for r in records:
            properties = dict(r.payload)
            ns = namespace if namespace is not None else r.namespace
            if ns is not None:
                properties[_NAMESPACE_FIELD] = ns
            weaviate_id = _to_weaviate_id(r.id)
            if weaviate_id != r.id:
                properties[_ORIGINAL_ID_FIELD] = r.id
            vector = {vector_name: r.vector} if vector_name else r.vector
            if await collection.data.exists(weaviate_id):
                await collection.data.replace(uuid=weaviate_id, properties=properties, vector=vector)
            else:
                await collection.data.insert(
                    uuid=weaviate_id, properties=properties, vector=vector
                )
            written += 1
        return written

    async def fetch_by_ids(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> list[VectorRecord]:
        if not ids:
            return []
        await self._ensure_connected()
        collection = self._client.collections.get(name)
        weaviate_ids = [_to_weaviate_id(i) for i in ids]
        base_filter = Filter.by_id().contains_any(weaviate_ids)
        namespace_filter = self._namespace_filter(namespace)
        filters = base_filter & namespace_filter if namespace_filter is not None else base_filter
        result = await collection.query.fetch_objects(filters=filters, include_vector=True)
        return [self._to_record(o) for o in result.objects]

    async def delete_vectors(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> int:
        if not ids:
            return 0
        await self._ensure_connected()
        collection = self._client.collections.get(name)
        deleted = 0
        for i in ids:
            if await collection.data.delete_by_id(_to_weaviate_id(i)):
                deleted += 1
        return deleted

    async def query(
        self,
        name: str,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
    ) -> list[ScoredMatch]:
        await self._ensure_connected()
        vector_name = await self._resolve_vector_name(name)
        collection = self._client.collections.get(name)
        result = await collection.query.near_vector(
            near_vector=vector,
            target_vector=vector_name,
            limit=top_k,
            filters=self._namespace_filter(namespace),
            return_metadata=MetadataQuery(distance=True),
            return_properties=[_ORIGINAL_ID_FIELD],
        )
        matches = []
        for obj in result.objects:
            original_id = (obj.properties or {}).get(_ORIGINAL_ID_FIELD)
            distance = obj.metadata.distance if obj.metadata else None
            matches.append(
                ScoredMatch(id=original_id or str(obj.uuid), score=float(distance or 0.0))
            )
        return matches

    async def close(self) -> None:
        if self._connected:
            await self._client.close()
