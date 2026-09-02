from __future__ import annotations

import numpy as np

from core.models.canonical_ir import DataType
from core.transformations.dequantize import dequantize


def test_dequantize_int8_rescales_toward_unit_range():
    v = np.array([[127, -127, 0]], dtype=np.float32)
    out = dequantize(v, DataType.INT8)
    assert np.allclose(out, [[1.0, -1.0, 0.0]], atol=1e-3)


def test_dequantize_binary_maps_to_plus_minus_one():
    v = np.array([[1, 0, 5, -3]], dtype=np.float32)
    out = dequantize(v, DataType.BINARY)
    assert np.allclose(out, [[1.0, -1.0, 1.0, -1.0]])


def test_dequantize_float32_passthrough():
    v = np.array([[0.1, 0.2]], dtype=np.float32)
    out = dequantize(v, DataType.FLOAT32)
    assert np.allclose(out, v)
