from __future__ import annotations

import numpy as np
import pytest

from core.models.canonical_ir import DataType
from core.validation.engine import (
    analyze_normalization,
    distribution_stats,
    integrity_diff,
    reconstruction_error,
    resolve_validation_mode,
    tolerance_for_dtype,
)


def test_tolerance_keyed_to_discovered_target_dtype_not_hardcoded_fp32():
    # V3 fix #3: FP16-stored targets need a looser tolerance than FP32, or every correct
    # copy would be flagged as corrupted.
    fp32_atol, _ = tolerance_for_dtype(DataType.FLOAT32)
    fp16_atol, _ = tolerance_for_dtype(DataType.FLOAT16)
    assert fp16_atol > fp32_atol


def test_integrity_diff_passes_for_identical_vectors():
    v = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    result = integrity_diff(v, v.copy(), DataType.FLOAT32)
    assert result["within_tolerance"] is True
    assert result["max_abs_diff"] == 0.0


def test_integrity_diff_fails_for_meaningfully_different_vectors():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((10, 8)).astype(np.float32)
    b = a + 1.0
    result = integrity_diff(a, b, DataType.FLOAT32)
    assert result["within_tolerance"] is False


def test_integrity_diff_rejects_mismatched_shapes():
    a = np.zeros((5, 8), dtype=np.float32)
    b = np.zeros((5, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        integrity_diff(a, b, DataType.FLOAT32)


def test_distribution_stats_detects_nan_and_inf():
    v = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    stats = distribution_stats(v)
    assert stats["has_nan"] is True

    v2 = np.array([[1.0, np.inf]], dtype=np.float32)
    assert distribution_stats(v2)["has_inf"] is True


def test_reconstruction_error_zero_for_identical_vectors():
    v = np.random.default_rng(1).standard_normal((5, 6)).astype(np.float32)
    result = reconstruction_error(v, v.copy())
    assert result["mean_l2_error"] == 0.0
    assert abs(result["mean_cosine_similarity"] - 1.0) < 1e-6


def test_analyze_normalization_detects_unit_norm():
    rng = np.random.default_rng(2)
    v = rng.standard_normal((100, 16)).astype(np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    spec = analyze_normalization(v)
    assert spec.status == "detected"
    assert spec.method == "l2"
    assert spec.confidence >= 0.95


def test_analyze_normalization_flags_not_normalized_rather_than_assuming():
    rng = np.random.default_rng(3)
    v = rng.uniform(2.0, 10.0, size=(50, 16)).astype(np.float32)  # far from unit norm
    spec = analyze_normalization(v)
    assert spec.status == "not_normalized"


def test_analyze_normalization_empty_sample_stays_unknown():
    spec = analyze_normalization(np.array([]))
    assert spec.status == "unknown"


def test_resolve_validation_mode_prefers_golden_queries():
    assert (
        resolve_validation_mode(
            [{"query": "q"}], llm_synthetic_available=True, representative_documents_available=True
        )
        == "golden"
    )


def test_resolve_validation_mode_degrades_to_unavailable_without_llm_or_golden():
    assert (
        resolve_validation_mode(
            None, llm_synthetic_available=False, representative_documents_available=True
        )
        == "unavailable"
    )


def test_resolve_validation_mode_synthetic_only_when_llm_and_documents_both_available():
    assert (
        resolve_validation_mode(
            None, llm_synthetic_available=True, representative_documents_available=False
        )
        == "unavailable"
    )
    assert (
        resolve_validation_mode(
            None, llm_synthetic_available=True, representative_documents_available=True
        )
        == "synthetic"
    )
