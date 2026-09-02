from core.models.canonical_ir import MrlSpec
from tools.discovery_tools import _build_embedding_discovery, _detect_languages


def test_detect_languages_collects_distinct_values_from_common_keys():
    payloads = [
        {"language": "en"},
        {"lang": "fr"},
        {"language_code": "de"},
        {"locale": "en"},
        {"text": "no language key here"},
    ]
    assert _detect_languages(payloads) == ["de", "en", "fr"]


def test_detect_languages_empty_when_no_payload_carries_a_language_key():
    payloads = [{"text": "hello"}, {"category": "cooking"}]
    assert _detect_languages(payloads) == []


def test_detect_languages_ignores_non_string_values():
    payloads = [{"language": 123}, {"lang": None}, {"language": "es"}]
    assert _detect_languages(payloads) == ["es"]


def test_embedding_discovery_defaults_to_unknown_with_low_confidence():
    result = _build_embedding_discovery({}, MrlSpec(), [], normalization_status="unknown")
    assert result["provider"] == "unknown"
    assert result["model"] == "unknown"
    assert result["discovered_from"] == "vector_sample_statistics"
    assert result["confidence"] == 0.0
    assert result["revision"] is None


def test_embedding_discovery_trusts_a_fully_operator_supplied_identity():
    config = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "revision": "2024-01",
        "tokenizer": "cl100k_base",
        "pooling": "mean",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    }
    result = _build_embedding_discovery(config, MrlSpec(), [], normalization_status="detected")
    assert result["provider"] == "openai"
    assert result["model"] == "text-embedding-3-small"
    assert result["discovered_from"] == "operator_supplied"
    assert result["confidence"] == 1.0
    assert result["tokenizer"] == "cl100k_base"
    assert result["query_prefix"] == "query: "


def test_embedding_discovery_partial_config_is_not_treated_as_a_confident_identity():
    # provider without model (or vice versa) isn't a full identity claim — stays honest
    # about the low-confidence vector-sample-statistics path rather than half-trusting it.
    result = _build_embedding_discovery(
        {"provider": "openai"}, MrlSpec(), [], normalization_status="unknown"
    )
    assert result["discovered_from"] == "vector_sample_statistics"
    assert result["confidence"] == 0.0
