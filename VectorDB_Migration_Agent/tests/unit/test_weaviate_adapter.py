"""No local/embedded Weaviate instance was reachable in this sandbox
(`weaviate.connect_to_embedded()` downloads a real server binary at connect time, and
that download failed here on a certificate-verification error) — unlike every other new
adapter this build adds, WeaviateAdapter's real end-to-end behavior is verified only via
a live smoke test against a real instance, not a local round-trip test. These tests cover
what's verifiable without one: the id-mapping helpers, which is exactly the same
seam test_qdrant_adapter.py targets for the same reason (a real migration with natural
string ids needs a UUID a real Weaviate server will accept).
"""

from __future__ import annotations

from core.adapters.weaviate_adapter import _is_weaviate_native_id, _to_weaviate_id


def test_native_uuid_ids_pass_through_unmapped():
    assert _is_weaviate_native_id("550e8400-e29b-41d4-a716-446655440000") is True
    assert (
        _to_weaviate_id("550e8400-e29b-41d4-a716-446655440000")
        == "550e8400-e29b-41d4-a716-446655440000"
    )


def test_non_uuid_ids_are_not_native():
    assert _is_weaviate_native_id("cooking-6") is False
    assert _is_weaviate_native_id("12345") is False  # a bare int is not a valid UUID
    assert _is_weaviate_native_id("") is False


def test_mapping_is_deterministic_and_collision_free_for_similar_ids():
    a = _to_weaviate_id("cooking-6")
    b = _to_weaviate_id("cooking-6")
    c = _to_weaviate_id("cooking-7")
    assert a == b, "same original id must always map to the same Weaviate id (idempotent upsert depends on this)"
    assert a != c


def test_mapped_ids_are_valid_uuids():
    import uuid

    mapped = _to_weaviate_id("natural-string-id")
    uuid.UUID(mapped)  # raises ValueError if not a valid UUID string
