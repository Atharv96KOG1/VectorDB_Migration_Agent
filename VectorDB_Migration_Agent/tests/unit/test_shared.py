"""Regression test for a real bug found live, 2026-08-31: source_provider/target_provider
became free-text textarea trigger fields on the Aetherion platform (replacing a buggy
dropdown widget), so a trailing newline/whitespace from the UI or a copy-paste can now
reach build_adapter() where a dropdown's constrained options never would have allowed it.
"""

from __future__ import annotations

import pytest

from core.adapters.pinecone_adapter import PineconeAdapter
from core.adapters.qdrant_adapter import QdrantAdapter
from tools._shared import (
    CredentialResolutionError,
    build_adapter,
    require_env,
    resolve_endpoint_ref,
    resolve_secret_ref,
)


def test_build_adapter_tolerates_surrounding_whitespace_and_case(monkeypatch):
    monkeypatch.setenv("TEST_PINECONE_KEY", "test-key")
    assert isinstance(build_adapter("  Pinecone\n", "TEST_PINECONE_KEY", None), PineconeAdapter)
    # Qdrant with no endpoint_ref resolves to in-memory mode, no credential lookup needed.
    assert isinstance(build_adapter("QDRANT ", "unused", None), QdrantAdapter)


def test_build_adapter_still_rejects_a_genuinely_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        build_adapter("not-a-real-provider", "some-value", None)


def test_resolve_secret_ref_resolves_a_bare_variable_name(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "the-real-value")
    assert resolve_secret_ref("TEST_VAR") == "the-real-value"


def test_resolve_secret_ref_accepts_a_raw_value_typed_in_directly():
    # 2026-09-01: operator wants to paste a live key/connection string straight into the
    # trigger field and test immediately, no env var + worker restart round trip, no
    # prefix/scheme to remember. A ref that doesn't match any set env var name is used
    # as the literal secret value itself.
    assert resolve_secret_ref("pcsk_some_raw_key_value") == "pcsk_some_raw_key_value"


def test_resolve_secret_ref_rejects_empty_ref():
    with pytest.raises(CredentialResolutionError, match="empty"):
        resolve_secret_ref("")


def test_require_env_returns_the_value_when_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
    assert require_env("OPENAI_API_KEY") == "sk-real-key"


def test_require_env_raises_when_unset_never_returns_the_var_name_as_a_bogus_key(monkeypatch):
    # Real bug, 2026-09-01: after resolve_secret_ref dropped its strict env-only lookup,
    # internal call sites that used resolve_secret_ref("OPENAI_API_KEY") purely to probe
    # "is this configured?" silently got back the literal string "OPENAI_API_KEY" as a
    # fake API key instead of an error, since resolve_secret_ref now falls back to
    # treating an unmatched ref as the literal secret value. require_env is the fix: no
    # such fallback, ever.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(CredentialResolutionError, match="is not set"):
        require_env("OPENAI_API_KEY")


def test_resolve_endpoint_ref_resolves_a_bare_name_that_matches_a_set_env_var(monkeypatch):
    monkeypatch.setenv("TEST_ENDPOINT_VAR", "https://real-host.example.com")
    assert resolve_endpoint_ref("TEST_ENDPOINT_VAR") == "https://real-host.example.com"


def test_resolve_endpoint_ref_still_treats_an_unmatched_bare_value_as_a_literal():
    # Backward compatible: ":memory:"/"memory"/"path://..." and a real literal URL are
    # never actual environment variable names, so they still pass through unchanged.
    assert resolve_endpoint_ref(":memory:") == ":memory:"
    assert resolve_endpoint_ref("memory") == "memory"
    assert resolve_endpoint_ref("path:///tmp/qdrant_store") == "path:///tmp/qdrant_store"
    assert (
        resolve_endpoint_ref("https://real-host.example.com") == "https://real-host.example.com"
    )


def test_resolve_endpoint_ref_none_stays_none():
    assert resolve_endpoint_ref(None) is None
