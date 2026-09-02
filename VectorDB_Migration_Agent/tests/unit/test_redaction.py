from __future__ import annotations

from core.security.redaction import redact, redact_text


def test_redact_hides_sensitive_keys_recursively():
    data = {
        "provider": "pinecone",
        "connection": {"api_key": "sk-abcdef1234567890", "region": "us-east-1"},
        "nested": [{"secret_ref": "env://X", "value": 1}],
    }
    result = redact(data)
    assert result["connection"]["api_key"] == "***REDACTED***"
    assert result["connection"]["region"] == "us-east-1"
    assert result["provider"] == "pinecone"
    assert result["nested"][0]["secret_ref"] == "***REDACTED***"
    assert result["nested"][0]["value"] == 1


def test_redact_fully_hides_a_container_whose_own_key_looks_sensitive():
    # A key literally named "credential"/"credentials" is redacted wholesale rather than
    # recursed into — conservative by design, since such a container is assumed to be
    # secret material end to end.
    data = {"credential": {"api_key": "x", "region": "us-east-1"}}
    assert redact(data)["credential"] == "***REDACTED***"


def test_redact_leaves_non_sensitive_scalars_alone():
    assert redact({"count": 42, "name": "documents"}) == {"count": 42, "name": "documents"}


def test_redact_text_hides_openai_style_key():
    text = "using key sk-abcdefghijklmnopqrst for the call"
    assert "sk-abcdefghijklmnopqrst" not in redact_text(text)
    assert "***REDACTED***" in redact_text(text)


def test_redact_text_hides_bearer_token_but_keeps_prefix():
    text = "Authorization: Bearer abcdef0123456789"
    redacted = redact_text(text)
    assert "abcdef0123456789" not in redacted
    assert "Bearer" in redacted


def test_redact_text_leaves_ordinary_text_unchanged():
    text = "vectors_written=1000, dimension=1536"
    assert redact_text(text) == text
