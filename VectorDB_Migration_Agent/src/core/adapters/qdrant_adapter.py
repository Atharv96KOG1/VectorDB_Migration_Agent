"""Qdrant adapter — backed by qdrant_client (AsyncQdrantClient). Supports a live URL/API
key deployment or a local/in-memory instance (`location=":memory:"`), which is what makes
the integration tests creds-free and fully real.

Namespace mapping (V2 §33): Qdrant has no native namespace/tenant concept, so a source
namespace is folded into a reserved payload field (`__namespace`), matching the exact
mapping V2 §33 calls out ("Pinecone namespace -> Qdrant payload"). The mapping is recorded
explicitly in the Canonical IR's NamespaceSpec, never silently dropped.

ID mapping: unlike Pinecone (arbitrary strings up to 512 chars), Qdrant point ids must be
an unsigned 64-bit integer or a UUID — confirmed live (2026-08) against a real migration
whose source used natural string ids like "cooking-6", which Qdrant rejected outright.
Any id that isn't already int/UUID-shaped is mapped through a deterministic uuid5 (same
input -> same output every time, so idempotent upsert still holds) and the original is
stashed in a reserved payload field so every read path (`_to_record`, `query`) recovers
it — callers of this adapter only ever see the original id, never the internal mapping.
"""

from __future__ import annotations

import uuid as uuid_lib

from qdrant_client import AsyncQdrantClient, models

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
_ID_NAMESPACE = uuid_lib.UUID("6f0c2f1e-2e35-5f1a-8b1b-7a1f6e8c9d2a")  # fixed, arbitrary

_DISTANCE_TO_METRIC = {
    models.Distance.COSINE: Metric.COSINE,
    models.Distance.EUCLID: Metric.EUCLIDEAN,
    models.Distance.DOT: Metric.DOT,
    models.Distance.MANHATTAN: Metric.MANHATTAN,
}


def _namespace_filter(namespace: str | None) -> models.Filter | None:
    if namespace is None:
        return None
    return models.Filter(
        must=[models.FieldCondition(key=_NAMESPACE_FIELD, match=models.MatchValue(value=namespace))]
    )


def _encode_cursor(offset: object | None) -> str | None:
    if offset is None:
        return None
    return f"{type(offset).__name__}:{offset}"


def _decode_cursor(cursor: str | None):
    if cursor is None:
        return None
    kind, _, raw = cursor.partition(":")
    if kind == "int":
        return int(raw)
    return raw


def _is_qdrant_native_id(value: str) -> bool:
    if value.isdigit():
        return True
    try:
        uuid_lib.UUID(value)
        return True
    except ValueError:
        return False


def _to_qdrant_point_id(original_id: str) -> str:
    if _is_qdrant_native_id(original_id):
        return original_id
    return str(uuid_lib.uuid5(_ID_NAMESPACE, original_id))


class QdrantAdapter(VectorDBAdapter):
    provider = "qdrant"

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        location: str | None = None,
        path: str | None = None,
    ) -> None:
        if location:
            self._client = AsyncQdrantClient(location=location)
        elif path:
            self._client = AsyncQdrantClient(path=path)
        else:
            # qdrant_client's default timeout is too short for a real batch upsert over a
            # real network connection — confirmed live, 2026-08-31: upserting 200 points
            # of 1536D vectors + payload to Qdrant Cloud raised a bare WriteTimeout
            # (wrapped in ResponseHandlingException) uploading the request body, the same
            # class of problem the embeddings-client REQUEST_TIMEOUT fix addressed for
            # OpenAI. 60s matches that fix's budget for a comparably-sized real transfer.
            self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=60)
        # Per-collection-name cache: None once resolved for a plain (unnamed-vector)
        # collection, a str for a named-vector one (e.g. a hybrid dense+sparse collection
        # created outside this system, like a "dense-vector"+"sparse-vector" schema) —
        # confirmed live, 2026-08-31, against a real Qdrant Cloud collection created via
        # a quick-start template. Keyed by name (not a single scalar) since one adapter
        # instance could in principle be pointed at more than one collection.
        self._vector_names: dict[str, str | None] = {}

    async def validate_credentials(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
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
                sparse_vectors=Capability.TRUE,
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

    @staticmethod
    def _inspect_vectors_config(vectors_config) -> tuple[int | None, Metric, str | None]:
        """Returns (dimension, metric, vector_name). vector_name is None for a plain
        (unnamed) vector collection; otherwise the FIRST named dense vector's name — this
        adapter migrates one dense vector field, so a collection with several named dense
        vectors picks the first deterministically (config-object iteration order) rather
        than guessing which one the operator means. A sibling named *sparse* vector (e.g.
        a hybrid "dense-vector"+"sparse-vector" schema) is never targeted — see V2 §30,
        never auto-densify/sparsify — and is simply left unpopulated by this migration."""
        if isinstance(vectors_config, models.VectorParams):
            metric = _DISTANCE_TO_METRIC.get(vectors_config.distance, Metric.UNKNOWN)
            return vectors_config.size, metric, None
        if isinstance(vectors_config, dict) and vectors_config:
            name, params = next(iter(vectors_config.items()))
            metric = _DISTANCE_TO_METRIC.get(params.distance, Metric.UNKNOWN)
            return params.size, metric, name
        return None, Metric.UNKNOWN, None

    async def _resolve_vector_name(self, name: str) -> str | None:
        """Self-heals the same way PineconeAdapter.query() now does (confirmed live,
        2026-08-31): a freshly constructed adapter for THIS call may never have had
        get_resource_info called on it, even if a different adapter instance already
        resolved it earlier in the same migration."""
        if name not in self._vector_names:
            await self.get_resource_info(name)
        return self._vector_names[name]

    async def get_resource_info(self, name: str, namespace: str | None = None) -> ResourceInfo:
        info = await self._client.get_collection(name)
        dimension, metric, vector_name = self._inspect_vectors_config(info.config.params.vectors)
        self._vector_names[name] = vector_name

        quantized = Capability.UNKNOWN
        quant_config = getattr(info.config.params, "quantization_config", None)
        if quant_config is not None:
            quantized = Capability.TRUE

        return ResourceInfo(
            name=name,
            dimension=dimension,
            metric=metric,
            datatype=DataType.FLOAT32,
            quantized=quantized,
            approximate_count=info.points_count,
        )

    async def sample_vectors(
        self, name: str, n: int, namespace: str | None = None
    ) -> list[VectorRecord]:
        points, _ = await self._client.scroll(
            collection_name=name,
            limit=n,
            with_vectors=True,
            with_payload=True,
            scroll_filter=_namespace_filter(namespace),
        )
        return [self._to_record(p) for p in points]

    async def scan_vectors(
        self,
        name: str,
        batch_size: int,
        cursor: str | None = None,
        namespace: str | None = None,
    ) -> ScanPage:
        points, next_offset = await self._client.scroll(
            collection_name=name,
            limit=batch_size,
            offset=_decode_cursor(cursor),
            with_vectors=True,
            with_payload=True,
            scroll_filter=_namespace_filter(namespace),
        )
        return ScanPage(
            records=[self._to_record(p) for p in points],
            next_cursor=_encode_cursor(next_offset),
        )

    async def upsert_vectors(
        self, name: str, records: list[VectorRecord], namespace: str | None = None
    ) -> int:
        # Upsert-by-id is inherently idempotent in Qdrant (V2 §39) — a retried activity
        # invocation overwrites the same point rather than duplicating it.
        if not records:
            return 0
        vector_name = await self._resolve_vector_name(name)
        points = []
        for r in records:
            payload = dict(r.payload)
            if namespace is not None:
                payload[_NAMESPACE_FIELD] = namespace
            elif r.namespace is not None:
                payload[_NAMESPACE_FIELD] = r.namespace
            point_id = _to_qdrant_point_id(r.id)
            if point_id != r.id:
                payload[_ORIGINAL_ID_FIELD] = r.id
            vector_value = {vector_name: r.vector} if vector_name is not None else r.vector
            points.append(models.PointStruct(id=point_id, vector=vector_value, payload=payload))
        await self._client.upsert(collection_name=name, points=points, wait=True)
        return len(points)

    async def fetch_by_ids(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> list[VectorRecord]:
        if not ids:
            return []
        qdrant_ids = [_to_qdrant_point_id(i) for i in ids]
        points = await self._client.retrieve(
            collection_name=name, ids=qdrant_ids, with_vectors=True, with_payload=True
        )
        return [self._to_record(p) for p in points]

    async def delete_vectors(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> int:
        if not ids:
            return 0
        qdrant_ids = [_to_qdrant_point_id(i) for i in ids]
        await self._client.delete(
            collection_name=name, points_selector=models.PointIdsList(points=qdrant_ids)
        )
        return len(ids)

    async def query(
        self,
        name: str,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
        hnsw_ef: int | None = None,
    ) -> list[ScoredMatch]:
        # hnsw_ef (search-time candidate-list size) trades latency for ANN accuracy.
        # Qdrant's default (unset -> collection's hnsw_config.ef_construct, ~100) is tuned
        # for large, well-populated collections; a small ephemeral BENCHMARK-only
        # collection (a few hundred points) benefits from a higher explicit value —
        # confirmed live, 2026-09-01: direct_copy (a byte-identical vector copy, zero data
        # loss) still scored recall@10=0.895 on such a collection with the default,
        # purely from approximate-search misses, while a direct exact comparison against
        # the real full-size target hit 1.0000. Never affects a real production query
        # unless a caller opts in — default stays None (server default).
        vector_name = await self._resolve_vector_name(name)
        result = await self._client.query_points(
            collection_name=name,
            query=vector,
            using=vector_name,  # None is a no-op for a plain (unnamed-vector) collection
            limit=top_k,
            query_filter=_namespace_filter(namespace),
            with_payload=True,  # needed to recover the original id — see _ORIGINAL_ID_FIELD
            search_params=models.SearchParams(hnsw_ef=hnsw_ef) if hnsw_ef else None,
        )
        matches = []
        for p in result.points:
            original_id = (p.payload or {}).get(_ORIGINAL_ID_FIELD)
            matches.append(ScoredMatch(id=original_id or str(p.id), score=p.score))
        return matches

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _to_record(point) -> VectorRecord:
        payload = dict(point.payload or {})
        namespace = payload.pop(_NAMESPACE_FIELD, None)
        original_id = payload.pop(_ORIGINAL_ID_FIELD, None)
        vector = point.vector
        if isinstance(vector, dict):
            vector = next(iter(vector.values())) if vector else None
        return VectorRecord(
            id=original_id or str(point.id), vector=vector, payload=payload, namespace=namespace
        )

    async def create_collection(
        self, name: str, dimension: int, metric: Metric = Metric.COSINE
    ) -> None:
        """Not part of the adapter interface (no `create_resource` in VectorDBAdapter —
        the spec treats target index configuration as a distinct concern, V2 §35, not
        something the migration engine does implicitly for the REAL target). Used by
        tests/setup for the real target, and by tools/benchmark_tools.py to stand up a
        short-lived ephemeral collection for real ANN-backed candidate evaluation (V2
        §22's "temporary target collection") — see delete_collection below."""
        distance = {
            Metric.COSINE: models.Distance.COSINE,
            Metric.EUCLIDEAN: models.Distance.EUCLID,
            Metric.DOT: models.Distance.DOT,
            Metric.MANHATTAN: models.Distance.MANHATTAN,
        }.get(metric, models.Distance.COSINE)
        await self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dimension, distance=distance),
        )

    async def delete_collection(self, name: str) -> None:
        """Pairs with create_collection for the ephemeral-benchmark-collection lifecycle
        (tools/benchmark_tools.py) — always called from a try/finally so a benchmark
        candidate that errors mid-evaluation doesn't leak a collection."""
        await self._client.delete_collection(collection_name=name)
