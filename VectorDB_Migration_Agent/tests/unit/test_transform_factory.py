"""Regression test for a real bug found live, 2026-08-31: re_embedding/ridge_mapping
requested embeddings at the MODEL's native dimension (1536D for text-embedding-3-small)
regardless of what the target actually required (768D), causing a hard write failure
("Vector dimension error: expected dim: 768, got 1536") rather than a lower score.
"""

from __future__ import annotations

from tools._transform_factory import _resolve_reembed_dimensions


def test_defaults_to_target_dimension_for_a_truncatable_model():
    assert _resolve_reembed_dimensions("text-embedding-3-small", None, 768) == 768
    assert _resolve_reembed_dimensions("text-embedding-3-large", None, 1536) == 1536


def test_operator_supplied_dimensions_always_wins():
    assert _resolve_reembed_dimensions("text-embedding-3-small", 512, 768) == 512


def test_stays_none_for_a_model_that_does_not_support_truncation():
    # text-embedding-ada-002 and self-hosted/other OpenAI-compatible endpoints reject an
    # unrecognized `dimensions` parameter outright — must never be sent blindly.
    assert _resolve_reembed_dimensions("text-embedding-ada-002", None, 768) is None
