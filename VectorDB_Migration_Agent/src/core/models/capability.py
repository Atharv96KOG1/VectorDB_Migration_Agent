"""Tri-state capability discovery model (V2 §6).

The whole point of this model is that TRUE/FALSE/UNKNOWN are three genuinely distinct
states. An adapter that fails to determine a capability MUST report UNKNOWN, never FALSE
— inability to discover a capability is not proof the capability doesn't exist. Every
consumer of a Capability value must handle UNKNOWN explicitly rather than treating it as
falsy.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Capability(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

    @property
    def is_true(self) -> bool:
        return self is Capability.TRUE

    @property
    def is_unknown(self) -> bool:
        return self is Capability.UNKNOWN


class ConnectivityCapabilities(BaseModel):
    connect: Capability = Capability.UNKNOWN
    tls: Capability = Capability.UNKNOWN


class DataCapabilities(BaseModel):
    list_vectors: Capability = Capability.UNKNOWN
    scan_vectors: Capability = Capability.UNKNOWN
    retrieve_vectors: Capability = Capability.UNKNOWN
    retrieve_payload: Capability = Capability.UNKNOWN
    retrieve_documents: Capability = Capability.UNKNOWN
    cdc: Capability = Capability.UNKNOWN


class RepresentationCapabilities(BaseModel):
    dimension: Capability = Capability.UNKNOWN
    metric: Capability = Capability.UNKNOWN
    datatype: Capability = Capability.UNKNOWN
    dense_vectors: Capability = Capability.UNKNOWN
    sparse_vectors: Capability = Capability.UNKNOWN
    named_vectors: Capability = Capability.UNKNOWN
    normalization: Capability = Capability.UNKNOWN
    embedding_model: Capability = Capability.UNKNOWN
    quantized_storage: Capability = Capability.UNKNOWN
    """Source stores vectors in a quantized (int8/binary) representation — V3 fix #4.
    TRUE requires a mandatory dequantize-before-benchmark step upstream of any
    Recall@K computation."""


class IndexCapabilities(BaseModel):
    retrieve_config: Capability = Capability.UNKNOWN
    configure_index: Capability = Capability.UNKNOWN


class MigrationCapabilities(BaseModel):
    resumable_scan: Capability = Capability.UNKNOWN
    export: Capability = Capability.UNKNOWN
    cdc: Capability = Capability.UNKNOWN


class CapabilityReport(BaseModel):
    provider: str
    connectivity: ConnectivityCapabilities = ConnectivityCapabilities()
    data: DataCapabilities = DataCapabilities()
    representation: RepresentationCapabilities = RepresentationCapabilities()
    index: IndexCapabilities = IndexCapabilities()
    migration: MigrationCapabilities = MigrationCapabilities()
