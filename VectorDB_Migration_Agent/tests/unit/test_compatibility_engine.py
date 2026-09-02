from __future__ import annotations

from core.compatibility.engine import (
    CompatibilityClass,
    analyze_compatibility,
    classify_datatype,
    classify_dimension,
    classify_metric,
    classify_quantization,
    classify_semantic_space,
    classify_vector_type,
)
from core.models.canonical_ir import DataType, Metric, NormalizationSpec, VectorKind
from core.models.capability import Capability

UNIT_NORM = NormalizationSpec(status="detected", method="l2", confidence=0.99)
NOT_NORMALIZED = NormalizationSpec(status="not_normalized", confidence=0.1)
UNKNOWN_NORM = NormalizationSpec(status="unknown")


def test_dimension_exact():
    assert classify_dimension(1536, 1536).classification == CompatibilityClass.EXACT


def test_dimension_mismatch_is_incompatible_not_transformable():
    # Dimension mismatch always blocks DIRECT copy; the representation analyzer (not this
    # engine) is what decides whether a transform can bridge it.
    assert classify_dimension(1536, 768).classification == CompatibilityClass.INCOMPATIBLE


def test_dimension_unknown_when_undiscovered():
    assert classify_dimension(None, 768).classification == CompatibilityClass.UNKNOWN
    assert classify_dimension(1536, None).classification == CompatibilityClass.UNKNOWN


def test_datatype_exact_and_float_compatible():
    assert (
        classify_datatype(DataType.FLOAT32, DataType.FLOAT32, Capability.FALSE).classification
        == CompatibilityClass.EXACT
    )
    assert (
        classify_datatype(DataType.FLOAT32, DataType.FLOAT16, Capability.FALSE).classification
        == CompatibilityClass.COMPATIBLE
    )


def test_datatype_quantized_is_transformable():
    assert (
        classify_datatype(DataType.INT8, DataType.FLOAT32, Capability.TRUE).classification
        == CompatibilityClass.TRANSFORMABLE
    )


def test_metric_exact():
    assert (
        classify_metric(Metric.COSINE, Metric.COSINE, UNKNOWN_NORM).classification
        == CompatibilityClass.EXACT
    )


def test_metric_cosine_dot_compatible_only_when_normalization_confirmed():
    assert (
        classify_metric(Metric.COSINE, Metric.DOT, UNIT_NORM).classification
        == CompatibilityClass.COMPATIBLE
    )
    assert (
        classify_metric(Metric.COSINE, Metric.DOT, UNKNOWN_NORM).classification
        == CompatibilityClass.UNKNOWN
    )


def test_metric_euclidean_cosine_transformable_under_confirmed_unit_norm():
    # V3 fix #2: ||a-b||^2 = 2 - 2(a.b) under confirmed unit-norm vectors.
    result = classify_metric(Metric.EUCLIDEAN, Metric.COSINE, UNIT_NORM)
    assert result.classification == CompatibilityClass.TRANSFORMABLE
    assert "2 - 2" in result.reason or "2 -" in result.reason


def test_metric_euclidean_cosine_incompatible_when_confirmed_not_normalized():
    result = classify_metric(Metric.EUCLIDEAN, Metric.COSINE, NOT_NORMALIZED)
    assert result.classification == CompatibilityClass.INCOMPATIBLE


def test_metric_euclidean_cosine_unknown_when_normalization_unknown():
    result = classify_metric(Metric.EUCLIDEAN, Metric.COSINE, UNKNOWN_NORM)
    assert result.classification == CompatibilityClass.UNKNOWN


def test_metric_manhattan_vs_cosine_always_incompatible():
    assert (
        classify_metric(Metric.MANHATTAN, Metric.COSINE, UNIT_NORM).classification
        == CompatibilityClass.INCOMPATIBLE
    )


def test_vector_type_dense_sparse_never_auto_transforms():
    assert (
        classify_vector_type(VectorKind.DENSE, VectorKind.DENSE).classification
        == CompatibilityClass.EXACT
    )
    assert (
        classify_vector_type(VectorKind.DENSE, VectorKind.SPARSE).classification
        == CompatibilityClass.INCOMPATIBLE
    )


def test_quantization_classification_never_coerces_unknown_to_false():
    assert classify_quantization(Capability.UNKNOWN).classification == CompatibilityClass.UNKNOWN
    assert classify_quantization(Capability.TRUE).classification == CompatibilityClass.TRANSFORMABLE
    assert classify_quantization(Capability.FALSE).classification == CompatibilityClass.EXACT


def test_direct_copy_possible_true_when_everything_exact():
    report = analyze_compatibility(
        source_dim=1536,
        target_dim=1536,
        source_dt=DataType.FLOAT32,
        target_dt=DataType.FLOAT32,
        source_metric=Metric.COSINE,
        target_metric=Metric.COSINE,
        source_kind=VectorKind.DENSE,
        target_kind=VectorKind.DENSE,
        source_quantized=Capability.FALSE,
        normalization=UNIT_NORM,
    )
    assert report.direct_copy_possible is True


def test_direct_copy_possible_false_on_dimension_mismatch():
    report = analyze_compatibility(
        source_dim=1536,
        target_dim=768,
        source_dt=DataType.FLOAT32,
        target_dt=DataType.FLOAT32,
        source_metric=Metric.COSINE,
        target_metric=Metric.COSINE,
        source_kind=VectorKind.DENSE,
        target_kind=VectorKind.DENSE,
        source_quantized=Capability.FALSE,
        normalization=UNIT_NORM,
    )
    assert report.direct_copy_possible is False


def test_semantic_space_exact_when_target_empty():
    result = classify_semantic_space(target_already_populated=False, source_semantic_space_id="openai/text-embedding-3-large")
    assert result.classification == CompatibilityClass.EXACT


def test_semantic_space_unknown_when_target_populated_and_undiscoverable():
    result = classify_semantic_space(
        target_already_populated=True, source_semantic_space_id="openai/text-embedding-3-large", target_semantic_space_id=None
    )
    assert result.classification == CompatibilityClass.UNKNOWN


def test_semantic_space_exact_when_target_matches_known_prior_space():
    result = classify_semantic_space(
        target_already_populated=True,
        source_semantic_space_id="openai/text-embedding-3-large",
        target_semantic_space_id="openai/text-embedding-3-large",
    )
    assert result.classification == CompatibilityClass.EXACT


def test_semantic_space_incompatible_when_target_holds_a_different_space():
    # This is the real danger the whole check exists for: silently mixing two unrelated
    # embedding spaces in one collection corrupts retrieval for everything already there.
    result = classify_semantic_space(
        target_already_populated=True,
        source_semantic_space_id="cohere/embed-v3",
        target_semantic_space_id="openai/text-embedding-3-large",
    )
    assert result.classification == CompatibilityClass.INCOMPATIBLE


def test_direct_copy_blocked_by_unverifiable_populated_target_even_with_matching_dims():
    # Dimension/dtype/metric all match, but the target already has foreign data whose
    # embedding space this system cannot verify — direct_copy must NOT be offered as safe.
    report = analyze_compatibility(
        source_dim=1536,
        target_dim=1536,
        source_dt=DataType.FLOAT32,
        target_dt=DataType.FLOAT32,
        source_metric=Metric.COSINE,
        target_metric=Metric.COSINE,
        source_kind=VectorKind.DENSE,
        target_kind=VectorKind.DENSE,
        source_quantized=Capability.FALSE,
        normalization=UNIT_NORM,
        target_already_populated=True,
        source_semantic_space_id="openai/text-embedding-3-large",
        target_semantic_space_id=None,
    )
    assert report.semantic_space.classification == CompatibilityClass.UNKNOWN
    assert report.direct_copy_possible is False
