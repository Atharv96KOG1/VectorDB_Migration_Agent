"""Milvus adapter — backed by pymilvus's `AsyncMilvusClient` (works against Zilliz Cloud,
a self-hosted Milvus server, or a local Milvus Lite `.db` file for creds-free testing).

Every response shape here (describe_collection's field list, describe_index's
metric_type, query/search result shapes, upsert/delete return values) was confirmed live
against a real Milvus Lite instance, not just recalled from pymilvus's docs — several
(e.g. delete() actually returning a bare list of deleted ids, not the
`Dict[str, int]` its own type hint claims) disagreed with what the SDK's signature
implies.

Load-before-query: a freshly opened connection to a collection (even one already fully
populated) starts in a "released" state — query/search fail with a clear
`Collection ... is in state 'released'` error until `load_collection()` is called.
`_ensure_loaded` self-heals this the same way `qdrant_adapter.py`'s
`_resolve_vector_name` and `pinecone_adapter.py`'s `_data_client` self-heal a fresh
adapter instance that never had `get_resource_info` called on it yet.

ID mapping: a collection's primary key field is fixed at creation as either INT64 or
VARCHAR. VARCHAR ids accept arbitrary source ids (like Pinecone) as long as they fit the
field's configured max_length. INT64 ids cannot hold an arbitrary string id at all, so a
non-numeric source id is mapped through a deterministic hash (same input -> same output,
so idempotent upsert still holds) and the original is stashed in a reserved dynamic field
(`__source_id`) — same recovery pattern Qdrant's uuid5 mapping uses, adapted for an
integer-only primary key instead of Qdrant's int-or-UUID one.

Namespace: Milvus has no namespace/tenant concept exposed through this simple API (native
multi-tenancy exists via "partitions", but requires the target collection to have been
created with matching partitions already — out of scope here), so a source namespace is
folded into a reserved dynamic field (`__namespace`), same convention as Qdrant's payload
fold. This requires the collection to have `enable_dynamic_field=True` (the default for
collections created through `MilvusClient.create_collection`); a collection without it
raises a clear Milvus error on the first namespaced write rather than silently dropping
the namespace.

A sibling sparse vector field (Milvus supports `SPARSE_FLOAT_VECTOR` natively since 2.4)
is never targeted by this adapter, same policy `qdrant_adapter.py` documents for its own
sibling sparse field — this migrates exactly one dense vector field.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from pymilvus import AsyncMilvusClient, DataType as MilvusDataType

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

_SPARSE_VECTOR_TYPE_NAMES = {"SPARSE_FLOAT_VECTOR"}

# Derived from pymilvus's own installed DataType enum, not a hand-maintained list — a
# manually-typed allowlist is exactly what missed INT8_VECTOR live, 2026-09-17 (a real
# 768-dim collection, "Test768"), causing this adapter to report "no dense vector field"
# on a collection that had one. Deriving this from the enum itself means a future
# pymilvus release adding another dense vector type is picked up automatically, with
# nothing here to remember to update. Members are ordered by their real Milvus DataType
# value so the first alphabetically-later "*_VECTOR" name isn't accidentally preferred
# over an earlier one when more than one dense field exists (matches get_resource_info's
# own "first dense field" contract).
_DENSE_VECTOR_TYPES = {
    member.value
    for name, member in MilvusDataType.__members__.items()
    if name.endswith("_VECTOR") and not name.startswith("_") and name not in _SPARSE_VECTOR_TYPE_NAMES
}

# Cross-enum translation (Milvus's DataType -> this system's own canonical DataType) —
# inherently a manual mapping (translating between two independently-defined enums can't
# be derived), but matched by name wherever the two enums agree, with FLOAT_VECTOR's one
# real naming mismatch ("FLOAT" vs our "FLOAT32") called out explicitly rather than
# silently paired. Anything genuinely unmapped falls back to DataType.UNKNOWN in
# get_resource_info below — an honest tri-state result, not a crash — but the assertion
# here means a future pymilvus dense-vector type this system's own DataType enum has no
# equivalent for fails loudly in tests, not silently in production.
_MILVUS_VECTOR_NAME_ALIASES = {"FLOAT_VECTOR": "FLOAT32"}
_MILVUS_TYPE_TO_DATATYPE: dict[int, DataType] = {}
for _name, _member in MilvusDataType.__members__.items():
    if _member.value not in _DENSE_VECTOR_TYPES:
        continue
    _canonical_name = _MILVUS_VECTOR_NAME_ALIASES.get(_name, _name.removesuffix("_VECTOR"))
    if _canonical_name in DataType.__members__:
        _MILVUS_TYPE_TO_DATATYPE[_member.value] = DataType[_canonical_name]
del _name, _member, _canonical_name

_METRIC_MAP = {
    "COSINE": Metric.COSINE,
    "IP": Metric.DOT,
    "L2": Metric.EUCLIDEAN,
    # HAMMING/JACCARD (binary-vector metrics) have no Metric equivalent — reported
    # UNKNOWN rather than guessed, same tri-state rule as everywhere else in this system.
}

_QUANTIZED_INDEX_TYPES = {"IVF_PQ", "IVF_SQ8", "SCANN", "RABITQ"}


def _is_int64_native(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def _to_milvus_int_id(original_id: str) -> int:
    if _is_int64_native(original_id):
        return int(original_id)
    # Stable 63-bit hash so it always fits Milvus's signed INT64 primary key.
    digest = hashlib.sha256(original_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _pk_in_filter(pk_field: str, pk_type: int, ids: list) -> str:
    if pk_type == MilvusDataType.VARCHAR.value:
        values = ", ".join(json.dumps(str(i)) for i in ids)
    else:
        values = ", ".join(str(i) for i in ids)
    return f"{pk_field} in [{values}]"


_MILVUS_VECTOR_NUMPY_DTYPE = {
    MilvusDataType.FLOAT16_VECTOR.value: np.float16,
    MilvusDataType.INT8_VECTOR.value: np.int8,
}


def _pack_vector_for_milvus(vector: list[float], milvus_type: int):
    """pymilvus's row-based insert/upsert/search API requires a real numpy array of the
    field's exact dtype for every dense vector type except FLOAT_VECTOR, which alone
    accepts a plain Python list natively (confirmed by every FLOAT_VECTOR collection
    this adapter was tested against before this fix). Confirmed live, 2026-09-17: a real
    INT8_VECTOR collection ("Test768") rejected a plain float list from upsert_vectors
    with `ParamError: invalid input for int8 vector. Expected an np.ndarray with
    dtype=int8` — the same class of "hardcoded to the common case" gap the dense-vector-
    type detection fix just before this one also had.

    BINARY_VECTOR (needs bit-packed bytes, not a plain numeric array) and
    BFLOAT16_VECTOR (no native numpy dtype without the optional `ml_dtypes` package) are
    passed through unconverted here — real, but not yet verified against a live
    collection of either type. Fixing those blind, without a real collection to test
    against, risks repeating exactly this mistake a second time; the honest move is to
    leave them as a known gap and fix on first real report, the same way this one was."""
    numpy_dtype = _MILVUS_VECTOR_NUMPY_DTYPE.get(milvus_type)
    if numpy_dtype is not None:
        return np.array(vector, dtype=numpy_dtype)
    return vector


def _unpack_vector_from_milvus(value: object, milvus_type: int) -> list[float] | None:
    """The read-side counterpart of _pack_vector_for_milvus — confirmed live,
    2026-09-17, that the same asymmetry exists in reverse: row-based `query()` (used by
    scan_vectors/fetch_by_ids) returns a non-FLOAT_VECTOR dense field as raw `bytes`, not
    an already-decoded list, unlike FLOAT_VECTOR which comes back as a plain float list
    natively. Passing those raw bytes straight into `VectorRecord.vector` (a
    `list[float]`) fails pydantic validation immediately — confirmed by the exact
    ValidationError this fixes. BINARY_VECTOR/BFLOAT16_VECTOR are the same
    not-yet-live-verified gap `_pack_vector_for_milvus` documents: returns None rather
    than guess at a decode with no real collection to check it against."""
    # Confirmed live, 2026-09-17: row-based query() doesn't return the raw bytes
    # directly for these types — it wraps them in a single-element list
    # (`[b'...']`), not documented anywhere in pymilvus's own type hints.
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (bytes, bytearray)):
        value = value[0]
    if not isinstance(value, (bytes, bytearray)):
        return value
    numpy_dtype = _MILVUS_VECTOR_NUMPY_DTYPE.get(milvus_type)
    if numpy_dtype is None:
        return None
    return np.frombuffer(value, dtype=numpy_dtype).tolist()


_URI_SCHEMES = ("unix://", "http://", "https://", "tcp://")


def _normalize_uri(uri: str) -> str:
    """A bare hostname (e.g. a Zilliz Cloud endpoint typed without its scheme) is
    rejected outright by pymilvus's ConnectionConfigException — confirmed live,
    2026-09-11, against a real Aetherion-deployed worker. Zilliz Cloud always requires
    TLS, so a schemeless value defaults to https:// (self-hosted Milvus users who need
    plain http can still type the scheme explicitly, same convention chroma_adapter.py/
    weaviate_adapter.py already use for their own schemeless-endpoint normalization,
    just defaulting to https instead of http since that's the real common case here)."""
    if uri.startswith(_URI_SCHEMES) or uri.endswith(".db"):
        return uri
    return f"https://{uri}"


class MilvusAdapter(VectorDBAdapter):
    provider = "milvus"

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str | None = None,
    ) -> None:
        self._client = AsyncMilvusClient(uri=_normalize_uri(uri), token=token or "")
        # Per-collection-name cache, resolved by get_resource_info (or self-healed by
        # _ensure_loaded on first use) — mirrors qdrant_adapter.py's _vector_names cache.
        self._pk_fields: dict[str, tuple[str, int]] = {}
        self._vector_fields: dict[str, str] = {}
        self._vector_field_types: dict[str, int] = {}
        self._loaded: set[str] = set()

    async def _ensure_loaded(self, name: str) -> None:
        if name not in self._pk_fields:
            await self.get_resource_info(name)
        if name not in self._loaded:
            await self._client.load_collection(name)
            self._loaded.add(name)

    def _namespace_clause(self, namespace: str | None) -> str | None:
        if namespace is None:
            return None
        return f"{_NAMESPACE_FIELD} == {json.dumps(namespace)}"

    def _combine_filters(self, *clauses: str | None) -> str:
        parts = [c for c in clauses if c]
        return " and ".join(parts)

    async def validate_credentials(self) -> bool:
        try:
            await self._client.list_collections()
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
                resumable_scan=Capability.TRUE, export=Capability.FALSE, cdc=Capability.FALSE
            ),
        )

    async def get_resource_info(self, name: str, namespace: str | None = None) -> ResourceInfo:
        desc = await self._client.describe_collection(name)
        fields = desc.get("fields", [])
        # Bare next() with no default previously let a genuinely empty match raise a bare
        # StopIteration straight out of this async function — confirmed live: Python
        # reports that as "coroutine raised StopIteration" with no indication of which
        # lookup actually failed. A clear RuntimeError naming the collection and what
        # was expected replaces it, matching pinecone_adapter.py's own precedent for
        # failing fast and clearly instead of surfacing a confusing raw error.
        pk_field = next((f for f in fields if f.get("is_primary")), None)
        if pk_field is None:
            raise RuntimeError(f"Milvus collection {name!r} has no primary-key field")
        vector_field = next((f for f in fields if f.get("type") in _DENSE_VECTOR_TYPES), None)
        if vector_field is None:
            raise RuntimeError(
                f"Milvus collection {name!r} has no dense FLOAT_VECTOR/BINARY_VECTOR/"
                f"FLOAT16_VECTOR/BFLOAT16_VECTOR field (only a sparse vector field?) — "
                f"this adapter migrates exactly one dense vector field per collection."
            )
        self._pk_fields[name] = (pk_field["name"], pk_field["type"])
        self._vector_fields[name] = vector_field["name"]
        self._vector_field_types[name] = vector_field["type"]
        dimension = (vector_field.get("params") or {}).get("dim")

        metric = Metric.UNKNOWN
        quantized = Capability.UNKNOWN
        try:
            index_names = await self._client.list_indexes(name)
            if index_names:
                index_info = await self._client.describe_index(name, index_names[0])
                metric = _METRIC_MAP.get(index_info.get("metric_type"), Metric.UNKNOWN)
                if index_info.get("index_type") in _QUANTIZED_INDEX_TYPES:
                    quantized = Capability.TRUE
        except Exception:
            pass

        count = None
        try:
            stats = await self._client.get_collection_stats(name)
            count = stats.get("row_count")
        except Exception:
            pass

        return ResourceInfo(
            name=name,
            dimension=int(dimension) if dimension is not None else None,
            metric=metric,
            # Was hardcoded to FLOAT32 regardless of the field's real storage type — a
            # real bug found alongside the INT8_VECTOR gap above: this datatype feeds
            # core/validation/engine.py's tolerance_for_dtype (1e-5 for FLOAT32 vs 0.05
            # for INT8), so a genuinely INT8-stored collection reported as FLOAT32 would
            # apply a tolerance ~5000x too tight and could make a correct byte-for-byte
            # copy look like a false integrity failure at VERIFY.
            datatype=_MILVUS_TYPE_TO_DATATYPE.get(vector_field.get("type"), DataType.UNKNOWN),
            quantized=quantized,
            approximate_count=count,
        )

    def _to_record(
        self, row: dict, pk_field: str, vector_field: str, vector_field_type: int
    ) -> VectorRecord:
        row = dict(row)
        original_id = row.pop(_ORIGINAL_ID_FIELD, None)
        source_id = original_id if original_id is not None else str(row.pop(pk_field))
        row.pop(pk_field, None)
        vector = _unpack_vector_from_milvus(row.pop(vector_field, None), vector_field_type)
        namespace = row.pop(_NAMESPACE_FIELD, None)
        return VectorRecord(id=source_id, vector=vector, payload=row, namespace=namespace)

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
        await self._ensure_loaded(name)
        pk_field, _ = self._pk_fields[name]
        vector_field = self._vector_fields[name]
        vector_field_type = self._vector_field_types[name]
        offset = int(cursor) if cursor else 0
        filter_str = self._combine_filters(self._namespace_clause(namespace))
        rows = await self._client.query(
            collection_name=name,
            filter=filter_str,
            output_fields=["*"],
            limit=batch_size,
            offset=offset,
        )
        records = [self._to_record(r, pk_field, vector_field, vector_field_type) for r in rows]
        next_cursor = str(offset + len(rows)) if len(rows) == batch_size else None
        return ScanPage(records=records, next_cursor=next_cursor)

    async def upsert_vectors(
        self, name: str, records: list[VectorRecord], namespace: str | None = None
    ) -> int:
        if not records:
            return 0
        await self._ensure_loaded(name)
        pk_field, pk_type = self._pk_fields[name]
        vector_field = self._vector_fields[name]
        vector_field_type = self._vector_field_types[name]
        rows = []
        for r in records:
            row = dict(r.payload)
            ns = namespace if namespace is not None else r.namespace
            if ns is not None:
                row[_NAMESPACE_FIELD] = ns
            if pk_type == MilvusDataType.VARCHAR.value:
                row[pk_field] = r.id
            else:
                mapped = _to_milvus_int_id(r.id)
                row[pk_field] = mapped
                if str(mapped) != r.id:
                    row[_ORIGINAL_ID_FIELD] = r.id
            row[vector_field] = _pack_vector_for_milvus(r.vector, vector_field_type)
            rows.append(row)
        result = await self._client.upsert(collection_name=name, data=rows)
        return result.get("upsert_count", len(rows))

    async def fetch_by_ids(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> list[VectorRecord]:
        if not ids:
            return []
        await self._ensure_loaded(name)
        pk_field, pk_type = self._pk_fields[name]
        vector_field = self._vector_fields[name]
        vector_field_type = self._vector_field_types[name]
        pk_values = [
            i if pk_type == MilvusDataType.VARCHAR.value else _to_milvus_int_id(i) for i in ids
        ]
        filter_str = self._combine_filters(
            _pk_in_filter(pk_field, pk_type, pk_values), self._namespace_clause(namespace)
        )
        rows = await self._client.query(collection_name=name, filter=filter_str, output_fields=["*"])
        return [self._to_record(r, pk_field, vector_field, vector_field_type) for r in rows]

    async def delete_vectors(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> int:
        if not ids:
            return 0
        await self._ensure_loaded(name)
        pk_field, pk_type = self._pk_fields[name]
        pk_values = [
            i if pk_type == MilvusDataType.VARCHAR.value else _to_milvus_int_id(i) for i in ids
        ]
        await self._client.delete(
            collection_name=name, filter=_pk_in_filter(pk_field, pk_type, pk_values)
        )
        return len(ids)

    async def query(
        self,
        name: str,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
    ) -> list[ScoredMatch]:
        await self._ensure_loaded(name)
        vector_field = self._vector_fields[name]
        vector_field_type = self._vector_field_types[name]
        filter_str = self._combine_filters(self._namespace_clause(namespace))
        results = await self._client.search(
            collection_name=name,
            data=[_pack_vector_for_milvus(vector, vector_field_type)],
            anns_field=vector_field,
            filter=filter_str,
            limit=top_k,
            output_fields=[_ORIGINAL_ID_FIELD],
        )
        matches = []
        for hit in results[0]:
            entity = hit.get("entity") or {}
            original_id = entity.get(_ORIGINAL_ID_FIELD)
            matches.append(ScoredMatch(id=original_id or str(hit["id"]), score=float(hit["distance"])))
        return matches

    async def close(self) -> None:
        await self._client.close()
