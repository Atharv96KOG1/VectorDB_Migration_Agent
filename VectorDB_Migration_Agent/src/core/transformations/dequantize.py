"""Mandatory dequantize-before-benchmark/transform step (V3 fix #4): if the source stores
quantized vectors, every downstream Recall@K number computed against the raw quantized
integers would be measuring the wrong numeric representation. Best-effort: the true
quantization scale is provider-specific metadata this build's adapters don't currently
discover (both PineconeAdapter and QdrantAdapter report `quantized=UNKNOWN`, never TRUE
today), so this path exists and is unit-tested but is dormant until an adapter actually
surfaces Capability.TRUE for source quantized storage.
"""

from __future__ import annotations

import numpy as np

from core.models.canonical_ir import DataType


def dequantize(vectors: np.ndarray, source_dtype: DataType) -> np.ndarray:
    if source_dtype == DataType.INT8:
        # symmetric int8 rescale to roughly [-1, 1]; real scale/zero-point would come from
        # provider metadata when an adapter discovers it.
        return (vectors.astype(np.float32)) / 127.0
    if source_dtype == DataType.BINARY:
        return np.where(vectors > 0, 1.0, -1.0).astype(np.float32)
    return vectors.astype(np.float32)
