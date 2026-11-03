"""Vespa adapter — hand-rolled REST client over httpx against Vespa's Document API v1,
the same architectural choice `pinecone_adapter.py` already makes and for the same
reasons: finer capability introspection than a heavier SDK gives, and every request
directly testable via `httpx.MockTransport` (see tests/unit/test_vespa_adapter.py).
`pyvespa` (the official Python package) is deliberately not a dependency here.

Resource addressing is genuinely different from every other provider this system
supports: Vespa has no simple "index name". `name` here is `"<namespace>/<document-type>"`
— Vespa's own namespace (part of every document id: `id:<namespace>:<doc-type>:...`), not
this system's migration `namespace` parameter, which is a *different* concept (tenant
scoping within one resource) folded into a reserved `__namespace` field, same convention
`qdrant_adapter.py`/`chroma_adapter.py`/`weaviate_adapter.py` use for the same gap.

The tensor field holding the embedding defaults to `"embedding"`; a real deployment can
override it by setting `vector_field` in `source_embedding_config`/passing it through
`endpoint_ref` query-string-style (`https://host:port?vector_field=my_field`) — whichever
convention the operator's actual schema needs, resolved at construction time.

**Two permanent, honest capability gaps, not implementation bugs**: (1) Vespa's data-plane
(feed/query) port exposes no "describe schema" call — dimension, metric, and vector
count are not discoverable at all through this API, so `get_resource_info` reports them
as `None`/`Metric.UNKNOWN` unless the operator supplies them, following the same
tri-state honesty rule as everywhere else. (2) Vector search requires a rank profile
already configured in the deployed schema to support `nearestNeighbor()` ranking — this
adapter assumes one named `"default"` unless overridden; a schema without one will return
a clear Vespa error from `/search/`, not a silent wrong answer.

IDs are arbitrary strings, like Pinecone/Chroma — no id mapping needed.

Scanning uses Vespa's visit API (`GET .../docid?continuation=...`), which has a real,
native continuation-token cursor — maps directly to `ScanPage.next_cursor`.
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

_NAMESPACE_FIELD = "__namespace"
_DEFAULT_VECTOR_FIELD = "embedding"
_DEFAULT_RANK_PROFILE = "default"


def _split_resource(name: str) -> tuple[str, str]:
    namespace, _, doc_type = name.partition("/")
    if not doc_type:
        raise ValueError(
            f"vespa resource {name!r} must be '<namespace>/<document-type>' "
            f"(Vespa has no simple index name)."
        )
    return namespace, doc_type


class VespaAdapter(VectorDBAdapter):
    provider = "vespa"

    def __init__(
        self,
        endpoint: str,
        credential_ref: str | None = None,
        vector_field: str = _DEFAULT_VECTOR_FIELD,
        rank_profile: str = _DEFAULT_RANK_PROFILE,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {credential_ref}"} if credential_ref else {}
        self._client = httpx.AsyncClient(base_url=endpoint, headers=headers, transport=transport)
        self._vector_field = vector_field
        self._rank_profile = rank_profile

    def _doc_path(self, namespace: str, doc_type: str, doc_id: str) -> str:
        return f"/document/v1/{namespace}/{doc_type}/docid/{doc_id}"

    async def validate_credentials(self) -> bool:
        try:
            resp = await self._client.get("/ApplicationStatus")
            if resp.status_code == 200:
                self.last_connect_error = None
                return True
            self.last_connect_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return False
        except httpx.HTTPError as exc:
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
                dimension=Capability.UNKNOWN,  # no schema-introspection call on the data plane
                metric=Capability.UNKNOWN,  # same permanent gap — see module docstring
                datatype=Capability.UNKNOWN,
                dense_vectors=Capability.TRUE,
                sparse_vectors=Capability.UNKNOWN,
                named_vectors=Capability.UNKNOWN,
                normalization=Capability.UNKNOWN,
                embedding_model=Capability.UNKNOWN,
                quantized_storage=Capability.UNKNOWN,
            ),
            index=IndexCapabilities(
                retrieve_config=Capability.FALSE, configure_index=Capability.FALSE
            ),
            migration=MigrationCapabilities(
                resumable_scan=Capability.TRUE, export=Capability.FALSE, cdc=Capability.FALSE
            ),
        )

    async def get_resource_info(self, name: str, namespace: str | None = None) -> ResourceInfo:
        # Confirmed permanent gap (see module docstring): Vespa's feed/query port has no
        # "describe schema" call — validate_credentials is the only real signal available
        # here without an operator-supplied embedding config.
        connected = await self.validate_credentials()
        if not connected:
            raise RuntimeError(f"cannot reach Vespa endpoint to inspect resource {name!r}")
        return ResourceInfo(
            name=name,
            dimension=None,
            metric=Metric.UNKNOWN,
            datatype=DataType.UNKNOWN,
            quantized=Capability.UNKNOWN,
            approximate_count=None,
        )

    def _to_record(self, doc: dict) -> VectorRecord:
        fields = dict(doc.get("fields", {}))
        vector = fields.pop(self._vector_field, None)
        if isinstance(vector, dict):  # Vespa tensor JSON form: {"values": [...]}
            vector = vector.get("values")
        namespace = fields.pop(_NAMESPACE_FIELD, None)
        doc_id = doc["id"].rsplit(":", 1)[-1]
        return VectorRecord(id=doc_id, vector=vector, payload=fields, namespace=namespace)

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
        vespa_namespace, doc_type = _split_resource(name)
        params: dict = {"wantedDocumentCount": batch_size}
        if cursor:
            params["continuation"] = cursor
        if namespace is not None:
            params["selection"] = f"{doc_type}.{_NAMESPACE_FIELD}=='{namespace}'"
        resp = await self._client.get(f"/document/v1/{vespa_namespace}/{doc_type}/docid", params=params)
        resp.raise_for_status()
        data = resp.json()
        records = [self._to_record(d) for d in data.get("documents", [])]
        return ScanPage(records=records, next_cursor=data.get("continuation"))

    async def upsert_vectors(
        self, name: str, records: list[VectorRecord], namespace: str | None = None
    ) -> int:
        if not records:
            return 0
        vespa_namespace, doc_type = _split_resource(name)
        written = 0
        for r in records:
            fields = dict(r.payload)
            fields[self._vector_field] = {"values": r.vector}
            ns = namespace if namespace is not None else r.namespace
            if ns is not None:
                fields[_NAMESPACE_FIELD] = ns
            resp = await self._client.post(
                self._doc_path(vespa_namespace, doc_type, r.id), json={"fields": fields}
            )
            resp.raise_for_status()
            written += 1
        return written

    async def fetch_by_ids(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> list[VectorRecord]:
        if not ids:
            return []
        vespa_namespace, doc_type = _split_resource(name)
        records = []
        for doc_id in ids:
            resp = await self._client.get(self._doc_path(vespa_namespace, doc_type, doc_id))
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            record = self._to_record(resp.json())
            if namespace is not None and record.namespace != namespace:
                continue
            records.append(record)
        return records

    async def delete_vectors(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> int:
        if not ids:
            return 0
        vespa_namespace, doc_type = _split_resource(name)
        deleted = 0
        for doc_id in ids:
            resp = await self._client.delete(self._doc_path(vespa_namespace, doc_type, doc_id))
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            deleted += 1
        return deleted

    async def query(
        self,
        name: str,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
    ) -> list[ScoredMatch]:
        vespa_namespace, doc_type = _split_resource(name)
        yql = (
            f"select * from sources {doc_type} where "
            f"{{targetHits: {top_k}}}nearestNeighbor({self._vector_field}, q)"
        )
        if namespace is not None:
            yql += f" and {_NAMESPACE_FIELD} contains '{namespace}'"
        body = {
            "yql": yql,
            "ranking": {"profile": self._rank_profile},
            "input": {"query(q)": vector},
            "hits": top_k,
        }
        resp = await self._client.post("/search/", json=body)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("root", {}).get("children", [])
        matches = []
        for hit in hits:
            doc_id = hit["id"].rsplit(":", 1)[-1]
            matches.append(ScoredMatch(id=doc_id, score=float(hit.get("relevance", 0.0))))
        return matches

    async def close(self) -> None:
        await self._client.aclose()
