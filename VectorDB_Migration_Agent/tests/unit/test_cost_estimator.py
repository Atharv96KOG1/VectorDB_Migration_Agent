"""StorageCostEstimator (core/optimizer/cost.py) — the optimizer's `cost` axis was
previously always 0.0 (CandidateMetrics.cost default, never populated by a real source),
making cost_weight silently inert. These tests pin the arithmetic and, more importantly,
the honesty contract: an unestimable cost must come back as None, never as a fabricated
0.0 that would look identical to a confirmed-free result.
"""

from __future__ import annotations

from core.models.canonical_ir import DataType
from core.optimizer.cost import DATATYPE_BYTES, PROVIDER_GB_MONTH_USD, StorageCostEstimator


def test_known_inputs_produce_the_documented_formula():
    estimator = StorageCostEstimator()
    dimension, datatype, vector_count, provider = 128, DataType.FLOAT32, 1_000_000, "qdrant"

    result = estimator.estimate(
        provider=provider, dimension=dimension, datatype=datatype, vector_count=vector_count
    )

    expected_gb = (dimension * DATATYPE_BYTES[datatype] * vector_count) / (1024**3)
    assert result == round(expected_gb * PROVIDER_GB_MONTH_USD[provider], 6)
    assert result > 0


def test_unknown_provider_returns_none_not_zero():
    estimator = StorageCostEstimator()
    result = estimator.estimate(
        provider="some-future-db", dimension=128, datatype=DataType.FLOAT32, vector_count=1000
    )
    assert result is None


def test_unknown_dimension_returns_none_not_zero():
    estimator = StorageCostEstimator()
    result = estimator.estimate(
        provider="qdrant", dimension=None, datatype=DataType.FLOAT32, vector_count=1000
    )
    assert result is None


def test_unknown_vector_count_returns_none_not_zero():
    estimator = StorageCostEstimator()
    result = estimator.estimate(
        provider="qdrant", dimension=128, datatype=DataType.FLOAT32, vector_count=None
    )
    assert result is None


def test_provider_with_a_real_zero_rate_is_distinct_from_an_unknown_provider():
    # faiss is a self-hosted library with a genuinely confirmed $0 storage fee — this is
    # a known zero, not a missing value, so it must come back as 0.0, not None.
    estimator = StorageCostEstimator()
    result = estimator.estimate(
        provider="faiss", dimension=128, datatype=DataType.FLOAT32, vector_count=1000
    )
    assert result == 0.0


def test_binary_datatype_costs_less_than_float32_at_the_same_dimension_and_count():
    estimator = StorageCostEstimator()
    kwargs = dict(provider="qdrant", dimension=128, vector_count=1000)
    float32_cost = estimator.estimate(datatype=DataType.FLOAT32, **kwargs)
    binary_cost = estimator.estimate(datatype=DataType.BINARY, **kwargs)
    assert binary_cost < float32_cost


def test_unrecognized_datatype_falls_back_to_float32_sizing():
    estimator = StorageCostEstimator()
    kwargs = dict(provider="qdrant", dimension=128, vector_count=1000)
    assert estimator.estimate(datatype=DataType.UNKNOWN, **kwargs) == estimator.estimate(
        datatype=DataType.FLOAT32, **kwargs
    )


def test_provider_name_matching_is_case_and_whitespace_insensitive():
    estimator = StorageCostEstimator()
    kwargs = dict(dimension=128, datatype=DataType.FLOAT32, vector_count=1000)
    assert estimator.estimate(provider="qdrant", **kwargs) == estimator.estimate(
        provider=" Qdrant ", **kwargs
    )


def test_custom_pricing_table_overrides_the_module_default():
    estimator = StorageCostEstimator(pricing={"acme-db": 10.0})
    result = estimator.estimate(
        provider="acme-db", dimension=128, datatype=DataType.FLOAT32, vector_count=1000
    )
    assert result is not None and result > 0
    # not in the custom table -> unestimable, even though it's in the module default
    assert (
        estimator.estimate(
            provider="qdrant", dimension=128, datatype=DataType.FLOAT32, vector_count=1000
        )
        is None
    )
