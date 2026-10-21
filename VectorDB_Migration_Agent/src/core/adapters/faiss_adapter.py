"""FAISS adapter — fundamentally different from every other provider here: FAISS is a
local, in-process ANN index *library*, not a server. There is no connection, no
credentials, no namespace/tenant concept, and — critically — no native id-to-payload
store at all: an `IndexIDMap2` gives int64 ids and raw vectors only.

`source_endpoint_ref`/`target_endpoint_ref` is a local filesystem path to the `.faiss`
index file; `source_credential_ref`/`target_credential_ref` is accepted (required by
`MigrationRequest`) and ignored. A JSON sidecar file at `<path>.meta.json` is the only
place payload/namespace/original-id can live at all — `{faiss_id: {original_id, payload,
namespace}}` — and is also the authoritative list of "which ids exist" for scanning,
since the base FAISS `Index` type has no "list all ids" primitive.

ID mapping: FAISS's global id space is int64-only (unlike Milvus, there's no
varchar-primary-key option). A source id that's already a small enough non-negative
integer passes through unmapped for readability; anything else is mapped through a
deterministic 63-bit hash (same input -> same output, so idempotent upsert holds) and the
original is stashed in the sidecar for recovery — same pattern as every other id-mapping
adapter in this codebase, just producing an int64 instead of a UUID.

Reconstruction (recovering a stored vector by id, needed for scan/sample/fetch) only
works natively for Flat and graph-based (HNSW) indexes. An IVF-family index needs
`make_direct_map()` called once after loading — this adapter does that proactively for
any IVF index — which then makes reconstruction work but LOSSY for a product/scalar-
quantized variant (IVF_PQ etc.), since the stored codes are already a compressed
approximation. That's expected, not a bug: `quantized=Capability.TRUE` is reported
for exactly these index types, which is what already triggers this system's existing
mandatory dequantize-before-benchmark handling elsewhere (V3 fix #4).

**Confirmed live, and important**: calling `IndexIDMap2.reconstruct_n(0, n)` on a
mapped index does NOT correctly translate sequential positions through the id map in
this faiss build — it crashed the whole Python process with an uncaught C++
`FaissException` (not a catchable Python exception), rather than returning wrong data or
raising cleanly. This adapter therefore only ever calls `reconstruct(id)` one real,
externally-mapped id at a time, never `reconstruct_n`.

No true upsert primitive exists either: implemented as `remove_ids` (if the id was
already present) followed by `add_with_ids`, then a synchronous `faiss.write_index()` so
every write is immediately durable on disk — matching the ABC's idempotent-on-retry
requirement at the cost of a full index rewrite per batch. Accepted tradeoff for
correctness over throughput, same spirit as `qdrant_adapter.py`'s own documented
tradeoffs. `remove_ids` is not implemented for every index type (e.g. HNSW) — FAISS
itself raises a clear `RuntimeError` in that case, left to propagate rather than caught
and reworded, since the message is already clear on its own.

All faiss/file IO is wrapped in `asyncio.to_thread(...)` — FAISS's Python bindings are
pure sync C++ bindings with no async API at all, same reasoning `chroma_adapter.py`
documents for its own local/embedded modes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

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

_MAX_NATIVE_ID = 2**53  # generous "looks like a real small integer" bound, not FAISS's own limit
_METRIC_MAP = {
    faiss.METRIC_L2: Metric.EUCLIDEAN,
    faiss.METRIC_INNER_PRODUCT: Metric.DOT,
    # Every other faiss.METRIC_* (Lp, Canberra, BrayCurtis, JensenShannon, ...) has no
    # Metric equivalent — reported UNKNOWN rather than guessed.
}
_QUANTIZED_MARKERS = ("PQ", "SQ", "LSQ", "RQ")


def _meta_path(index_path: str) -> Path:
    return Path(f"{index_path}.meta.json")


def _is_native_faiss_id(value: str) -> bool:
    try:
        n = int(value)
        return 0 <= n < _MAX_NATIVE_ID
    except ValueError:
        return False


def _to_faiss_id(original_id: str) -> int:
    if _is_native_faiss_id(original_id):
        return int(original_id)
    digest = hashlib.sha256(original_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


class FaissAdapter(VectorDBAdapter):
    provider = "faiss"

    def __init__(self, path: str, credential_ref: str | None = None) -> None:
        self._path = path
        self._index: faiss.IndexIDMap2 | None = None
        self._meta: dict[str, dict] = {}

    def _load(self) -> None:
        if self._index is not None:
            return
        if not Path(self._path).exists():
            raise RuntimeError(
                f"FAISS index file {self._path!r} does not exist. This adapter never "
                f"creates a target implicitly (same policy as the Pinecone/Qdrant "
                f"adapters) — build the index file first (see FaissAdapter.create_index)."
            )
        index = faiss.read_index(self._path)
        if not isinstance(index, faiss.IndexIDMap2):
            raise RuntimeError(
                f"FAISS index at {self._path!r} is not an IndexIDMap2 — this adapter "
                f"requires id-mapped indexes (arbitrary source ids need real, stable "
                f"FAISS ids to round-trip through)."
            )
        if hasattr(index.index, "make_direct_map"):
            index.index.make_direct_map()
        self._index = index
        meta_file = _meta_path(self._path)
        self._meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    def _persist(self) -> None:
        faiss.write_index(self._index, self._path)
        _meta_path(self._path).write_text(json.dumps(self._meta))

    @staticmethod
    def create_index(
        path: str, dimension: int, metric: Metric = Metric.EUCLIDEAN
    ) -> None:
        """Not part of the adapter interface (no `create_resource` in VectorDBAdapter,
        same reasoning `qdrant_adapter.py`'s own create_collection documents) — used by
        tests/setup to stand up a real target index file."""
        faiss_metric = {
            Metric.EUCLIDEAN: faiss.METRIC_L2,
            Metric.DOT: faiss.METRIC_INNER_PRODUCT,
            Metric.COSINE: faiss.METRIC_INNER_PRODUCT,  # caller normalizes vectors for cosine
        }.get(metric, faiss.METRIC_L2)
        base = faiss.IndexFlat(dimension, faiss_metric)
        index = faiss.IndexIDMap2(base)
        faiss.write_index(index, path)
        _meta_path(path).write_text(json.dumps({}))

    async def validate_credentials(self) -> bool:
        try:
            await asyncio.to_thread(self._load)
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
            connectivity=ConnectivityCapabilities(connect=connect_cap, tls=Capability.FALSE),
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
                named_vectors=Capability.FALSE,
                normalization=Capability.UNKNOWN,
                embedding_model=Capability.UNKNOWN,
                quantized_storage=Capability.UNKNOWN,
            ),
            index=IndexCapabilities(
                retrieve_config=Capability.TRUE, configure_index=Capability.FALSE
            ),
            migration=MigrationCapabilities(
                resumable_scan=Capability.TRUE, export=Capability.TRUE, cdc=Capability.FALSE
            ),
        )

    async def get_resource_info(self, name: str, namespace: str | None = None) -> ResourceInfo:
        await asyncio.to_thread(self._load)
        index = self._index
        metric = _METRIC_MAP.get(index.index.metric_type, Metric.UNKNOWN)
        class_name = type(index.index).__name__
        quantized = (
            Capability.TRUE if any(m in class_name for m in _QUANTIZED_MARKERS) else Capability.FALSE
        )
        return ResourceInfo(
            name=self._path,
            dimension=index.d,
            metric=metric,
            datatype=DataType.FLOAT32,
            quantized=quantized,
            approximate_count=index.ntotal,
        )

    def _record_from_meta(self, faiss_id: int, vector: list[float]) -> VectorRecord:
        entry = self._meta.get(str(faiss_id), {})
        original_id = entry.get("original_id", str(faiss_id))
        payload = entry.get("payload", {})
        namespace = entry.get("namespace")
        return VectorRecord(id=original_id, vector=vector, payload=payload, namespace=namespace)

    def _scoped_ids(self, namespace: str | None) -> list[int]:
        ids = []
        for key, entry in self._meta.items():
            if namespace is not None and entry.get("namespace") != namespace:
                continue
            ids.append(int(key))
        return sorted(ids)

    def _sync_scan(self, batch_size: int, cursor: str | None, namespace: str | None) -> ScanPage:
        self._load()
        ids = self._scoped_ids(namespace)
        offset = int(cursor) if cursor else 0
        page_ids = ids[offset : offset + batch_size]
        records = [
            self._record_from_meta(fid, self._index.reconstruct(fid).tolist()) for fid in page_ids
        ]
        next_cursor = str(offset + len(page_ids)) if offset + len(page_ids) < len(ids) else None
        return ScanPage(records=records, next_cursor=next_cursor)

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
        return await asyncio.to_thread(self._sync_scan, batch_size, cursor, namespace)

    def _sync_upsert(self, records: list[VectorRecord], namespace: str | None) -> int:
        self._load()
        for r in records:
            faiss_id = _to_faiss_id(r.id)
            if str(faiss_id) in self._meta:
                self._index.remove_ids(np.array([faiss_id], dtype="int64"))
            vector = np.array([r.vector], dtype="float32")
            self._index.add_with_ids(vector, np.array([faiss_id], dtype="int64"))
            ns = namespace if namespace is not None else r.namespace
            self._meta[str(faiss_id)] = {
                "original_id": r.id,
                "payload": r.payload,
                "namespace": ns,
            }
        self._persist()
        return len(records)

    async def upsert_vectors(
        self, name: str, records: list[VectorRecord], namespace: str | None = None
    ) -> int:
        if not records:
            return 0
        return await asyncio.to_thread(self._sync_upsert, records, namespace)

    def _sync_fetch(self, ids: list[str], namespace: str | None) -> list[VectorRecord]:
        self._load()
        records = []
        for original_id in ids:
            faiss_id = _to_faiss_id(original_id)
            entry = self._meta.get(str(faiss_id))
            if entry is None:
                continue
            if namespace is not None and entry.get("namespace") != namespace:
                continue
            vector = self._index.reconstruct(faiss_id).tolist()
            records.append(self._record_from_meta(faiss_id, vector))
        return records

    async def fetch_by_ids(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> list[VectorRecord]:
        if not ids:
            return []
        return await asyncio.to_thread(self._sync_fetch, ids, namespace)

    def _sync_delete(self, ids: list[str]) -> int:
        self._load()
        faiss_ids = [_to_faiss_id(i) for i in ids if str(_to_faiss_id(i)) in self._meta]
        if not faiss_ids:
            return 0
        self._index.remove_ids(np.array(faiss_ids, dtype="int64"))
        for fid in faiss_ids:
            self._meta.pop(str(fid), None)
        self._persist()
        return len(faiss_ids)

    async def delete_vectors(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> int:
        if not ids:
            return 0
        return await asyncio.to_thread(self._sync_delete, ids)

    def _sync_query(
        self, vector: list[float], top_k: int, namespace: str | None
    ) -> list[ScoredMatch]:
        self._load()
        query = np.array([vector], dtype="float32")
        # Namespace scoping has no native filter here — over-fetch, then filter and
        # truncate. Fine for BENCHMARK-scale samples; not intended for huge namespaced
        # collections, same class of tradeoff the brute-force benchmark path accepts
        # elsewhere in this system.
        k = top_k if namespace is None else min(self._index.ntotal, max(top_k * 10, top_k))
        if k == 0:
            return []
        distances, ids = self._index.search(query, k)
        matches = []
        for dist, fid in zip(distances[0], ids[0]):
            if fid == -1:
                continue
            entry = self._meta.get(str(fid))
            if entry is None:
                continue
            if namespace is not None and entry.get("namespace") != namespace:
                continue
            matches.append(ScoredMatch(id=entry.get("original_id", str(fid)), score=float(dist)))
            if len(matches) == top_k:
                break
        return matches

    async def query(
        self,
        name: str,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
    ) -> list[ScoredMatch]:
        return await asyncio.to_thread(self._sync_query, vector, top_k, namespace)

    async def close(self) -> None:
        return None
