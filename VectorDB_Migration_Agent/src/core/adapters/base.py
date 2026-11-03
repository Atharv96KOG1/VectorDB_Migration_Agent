"""Provider adapter interface (V2 §3.1, §46). Provider-specific logic must stay inside
adapter implementations — the compatibility engine, transformers, and orchestrator never
branch on provider name; they only see this interface plus the Canonical Vector IR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from core.models.canonical_ir import DataType, Metric
from core.models.capability import Capability, CapabilityReport


class VectorRecord(BaseModel):
    id: str
    vector: list[float] | None = None
    sparse_indices: list[int] | None = None
    sparse_values: list[float] | None = None
    payload: dict = Field(default_factory=dict)
    namespace: str | None = None


class ResourceInfo(BaseModel):
    name: str
    dimension: int | None = None
    metric: Metric = Metric.UNKNOWN
    datatype: DataType = DataType.UNKNOWN
    quantized: Capability = Capability.UNKNOWN
    approximate_count: int | None = None


class ScanPage(BaseModel):
    records: list[VectorRecord]
    next_cursor: str | None = None


class ScoredMatch(BaseModel):
    id: str
    score: float


class VectorDBAdapter(ABC):
    """One instance per (provider, resolved credentials) pair. All methods are async and
    are only ever called from inside a `@tool()` activity — never from the orchestrator
    workflow body — so they're free to do real IO, retries, and non-deterministic work.
    """

    provider: str

    last_connect_error: str | None = None
    """Set by validate_credentials() on a failed connection attempt, cleared on a
    successful one. Every adapter's validate_credentials previously swallowed the real
    exception into a bare `False` — confirmed live, 2026-10-05: a Qdrant source whose
    cluster was actively resetting the TLS handshake (a paused/suspended cluster) showed
    the operator nothing beyond `connected: false`, no way to tell network failure from
    bad credentials from a wrong URL without a developer manually reproducing it outside
    the platform. tools/connection_tools.py surfaces this in connect_and_validate's
    per-side result."""

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Cheap auth check (V2 §7: credential lookup -> authentication -> authorization).
        On failure, must set self.last_connect_error to a real, human-readable reason
        (e.g. f"{type(exc).__name__}: {exc}") before returning False — never swallow the
        real exception into a bare boolean with no diagnostic trail."""

    @abstractmethod
    async def discover_capabilities(self) -> CapabilityReport:
        """Must report Capability.UNKNOWN for anything it cannot positively confirm.
        Never coerce UNKNOWN to FALSE (V2 §6)."""

    @abstractmethod
    async def get_resource_info(self, name: str, namespace: str | None = None) -> ResourceInfo: ...

    @abstractmethod
    async def sample_vectors(
        self, name: str, n: int, namespace: str | None = None
    ) -> list[VectorRecord]:
        """Representative sample for normalization analysis, benchmarking, and transform
        fitting (PCA/RandomProjection)."""

    @abstractmethod
    async def scan_vectors(
        self,
        name: str,
        batch_size: int,
        cursor: str | None = None,
        namespace: str | None = None,
    ) -> ScanPage:
        """Resumable cursor-based scan for full migration (V2 §38 checkpointing)."""

    @abstractmethod
    async def upsert_vectors(
        self, name: str, records: list[VectorRecord], namespace: str | None = None
    ) -> int:
        """Must be idempotent on VectorRecord.id — Temporal's own RetryPolicy can re-invoke
        this activity independent of any checkpoint (V2 §39). Returns count written."""

    @abstractmethod
    async def fetch_by_ids(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> list[VectorRecord]: ...

    @abstractmethod
    async def delete_vectors(
        self, name: str, ids: list[str], namespace: str | None = None
    ) -> int:
        """Deletes by id. Used only for rollback (tools/execution_tools.py:rollback_migration) —
        deletes exactly the ids this migration is known to have written, never a bulk/
        whole-resource clear, so it can't touch data this migration didn't create."""

    @abstractmethod
    async def query(
        self,
        name: str,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
    ) -> list[ScoredMatch]:
        """Used by the retrieval-equivalence engine to compute Top-K on both sides."""

    async def close(self) -> None:
        """Override if the adapter holds a live connection/session to release."""
        return None
