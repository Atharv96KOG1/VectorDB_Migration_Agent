"""Regression test for a real bug found live, 2026-08-31: re_embedding/ridge_mapping
requested embeddings at the MODEL's native dimension (1536D for text-embedding-3-small)
regardless of what the target actually required (768D), causing a hard write failure
("Vector dimension error: expected dim: 768, got 1536") rather than a lower score.
"""

from __future__ import annotations

import json

import numpy as np

from core.models.migration_plan import TransformStrategy
from tools._transform_factory import _resolve_reembed_dimensions, provenance_parameters_from_fitted


def test_defaults_to_target_dimension_for_a_truncatable_model():
    assert _resolve_reembed_dimensions("text-embedding-3-small", None, 768) == 768
    assert _resolve_reembed_dimensions("text-embedding-3-large", None, 1536) == 1536


def test_operator_supplied_dimensions_always_wins():
    assert _resolve_reembed_dimensions("text-embedding-3-small", 512, 768) == 512


def test_stays_none_for_a_model_that_does_not_support_truncation():
    # text-embedding-ada-002 and self-hosted/other OpenAI-compatible endpoints reject an
    # unrecognized `dimensions` parameter outright — must never be sent blindly.
    assert _resolve_reembed_dimensions("text-embedding-ada-002", None, 768) is None


def test_provenance_parameters_strips_the_raw_matrix_for_pca():
    # Real bug found live, 2026-09-02: embedding fitted["params"] directly (the raw
    # components matrix) in the workflow result caused visible lag rendering the
    # platform's Result panel — a 1024x1536 matrix alone is 1.5M+ floats as JSON text.
    source_dim, target_dim = 1024, 128
    rng = np.random.default_rng(0)
    fitted_params = {
        "components": rng.standard_normal((target_dim, source_dim)).tolist(),
        "mean": rng.standard_normal(source_dim).tolist(),
    }
    result = provenance_parameters_from_fitted(TransformStrategy.PCA, fitted_params, target_dim)
    assert "components" not in result
    assert "mean" not in result
    assert len(json.dumps(result)) < 500


def test_provenance_parameters_strips_the_raw_matrix_for_random_projection():
    source_dim, target_dim = 768, 1536
    rng = np.random.default_rng(0)
    fitted_params = {"matrix": rng.standard_normal((source_dim, target_dim)).tolist()}
    result = provenance_parameters_from_fitted(
        TransformStrategy.RANDOM_PROJECTION, fitted_params, target_dim
    )
    assert "matrix" not in result
    assert len(json.dumps(result)) < 500


def test_provenance_parameters_strips_the_raw_coef_for_ridge_mapping():
    source_dim, target_dim = 1024, 1536
    rng = np.random.default_rng(0)
    fitted_params = {
        "coef": rng.standard_normal((target_dim, source_dim)).tolist(),
        "intercept": rng.standard_normal(target_dim).tolist(),
    }
    result = provenance_parameters_from_fitted(
        TransformStrategy.RIDGE_MAPPING, fitted_params, target_dim
    )
    assert "coef" not in result
    assert "intercept" not in result
    assert len(json.dumps(result)) < 500


def test_provenance_parameters_empty_for_direct_copy():
    assert provenance_parameters_from_fitted(TransformStrategy.DIRECT_COPY, {}, 1024) == {}
