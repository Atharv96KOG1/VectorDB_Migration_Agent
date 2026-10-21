"""Real round-trip tests against `chromadb.EphemeralClient` — a local, creds-free, purely
in-process Chroma instance, playing the same role Qdrant's `:memory:` plays for
test_qdrant_adapter.py. ChromaAdapter wraps every call in asyncio.to_thread since Chroma's
local client modes have no native async API — these tests exercise that wrapping for
real, not just the underlying sync client directly.
"""

from __future__ import annotations

import chromadb

from core.adapters.base import VectorRecord
from core.adapters.chroma_adapter import ChromaAdapter


def _make_collection(client, name: str, space: str = "cosine") -> None:
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    client.create_collection(name, metadata={"hnsw:space": space})


class _PatchedChromaAdapter(ChromaAdapter):
    """Reuses one already-open EphemeralClient — a fresh EphemeralClient() per adapter
    instance (what the real constructor does for endpoint=None) would each get its own
    isolated in-memory store, making the test's setup client and the adapter's client
    point at two different collections."""

    def __init__(self, client) -> None:
        self._client = client
        self._collections: dict[str, object] = {}


async def test_round_trip_ids_documents_and_namespace_fold():
    client = chromadb.EphemeralClient()
    _make_collection(client, "col")
    adapter = _PatchedChromaAdapter(client)

    info = await adapter.get_resource_info("col")
    assert info.metric.value == "cosine"
    assert info.approximate_count == 0

    records = [
        VectorRecord(
            id="cooking-6",
            vector=[1.0, 0.0, 0.0, 0.0],
            payload={"topic": "cooking", "__document": "a recipe about soup"},
        ),
        VectorRecord(
            id="finance-7",
            vector=[0.0, 1.0, 0.0, 0.0],
            payload={"topic": "finance"},
            namespace="tenant-1",
        ),
    ]
    written = await adapter.upsert_vectors("col", records)
    assert written == 2

    info_after = await adapter.get_resource_info("col")
    assert info_after.dimension == 4
    assert info_after.approximate_count == 2

    fetched = await adapter.fetch_by_ids("col", ["cooking-6", "finance-7"])
    by_id = {r.id: r for r in fetched}
    assert by_id["cooking-6"].payload == {"topic": "cooking", "__document": "a recipe about soup"}
    assert by_id["finance-7"].namespace == "tenant-1"
    assert "__namespace" not in by_id["finance-7"].payload  # internal field never leaks

    matches = await adapter.query("col", [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert matches[0].id == "cooking-6"

    scoped = await adapter.scan_vectors("col", batch_size=10, namespace="tenant-1")
    assert {r.id for r in scoped.records} == {"finance-7"}

    deleted = await adapter.delete_vectors("col", ["finance-7"])
    assert deleted == 1
    assert await asyncio_count(adapter, "col") == 1


async def asyncio_count(adapter: ChromaAdapter, name: str) -> int:
    info = await adapter.get_resource_info(name)
    return info.approximate_count
