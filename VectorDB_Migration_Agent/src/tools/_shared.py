"""Shared, tool-only glue: credential resolution and adapter construction. Lives in
tools/ (not core/) because os.environ access is exactly the kind of non-deterministic IO
that must never be reachable from the orchestrator workflow body — every function here is
only ever called from inside a `@tool()` activity.
"""

from __future__ import annotations

import os

from core.adapters.base import VectorDBAdapter
from core.adapters.pinecone_adapter import PineconeAdapter
from core.adapters.qdrant_adapter import QdrantAdapter


class CredentialResolutionError(RuntimeError):
    pass


def resolve_secret_ref(ref: str) -> str:
    """Resolves a credential ref to its value — no prefix/scheme of any kind (2026-09-01,
    dropped "env://" entirely: an operator swapping live Pinecone/Qdrant creds per test
    run doesn't want to remember a scheme). If the ref matches a set environment variable
    name, use its value; otherwise treat the ref itself as the literal secret value (a
    real API key/connection string pasted straight into the field)."""
    if not ref:
        raise CredentialResolutionError("credential ref is empty")
    return os.getenv(ref, ref)


def require_env(var_name: str) -> str:
    """Strict lookup for a fixed, code-known secret name (OPENAI_API_KEY,
    ANTHROPIC_API_KEY — never an operator-supplied ref from a trigger field). Unlike
    resolve_secret_ref, there's no "treat the name as the literal value" fallback: a
    hardcoded var_name that isn't set must raise, not silently become the var_name string
    itself used as a bogus API key (a real bug found 2026-09-01 — every
    `resolve_secret_ref("OPENAI_API_KEY")` call site was written to rely on the old
    strict-lookup behavior resolve_secret_ref no longer has)."""
    value = os.environ.get(var_name)
    if not value:
        raise CredentialResolutionError(f"environment variable {var_name!r} is not set")
    return value


def resolve_endpoint_ref(ref: str | None) -> str | None:
    """Endpoints/hosts aren't secrets, so a literal value (a real URL, ":memory:",
    "path://...") is accepted directly. No prefix/scheme, same as resolve_secret_ref: a
    bare value that matches a set env var name resolves to it, otherwise it's used
    as-is."""
    if ref is None:
        return None
    return os.getenv(ref, ref)


def build_adapter(provider: str, credential_ref: str, endpoint_ref: str | None) -> VectorDBAdapter:
    # .strip() matters now that source_provider/target_provider are free-text (a textarea
    # trigger field, not a constrained dropdown, on the Aetherion platform) — a trailing
    # newline from pressing Enter, or copy-paste whitespace, would otherwise turn a
    # correctly-typed "pinecone" into an "unknown provider" error.
    provider = provider.strip().lower()
    endpoint = resolve_endpoint_ref(endpoint_ref)

    if provider == "qdrant":
        if endpoint in (None, "", "memory", ":memory:"):
            return QdrantAdapter(location=":memory:")
        if endpoint.startswith("path://"):
            # Local persistent on-disk mode — unlike ":memory:", this survives across
            # separate QdrantAdapter instances (each tool call opens its own adapter and
            # closes it when done), which is what makes it usable for a multi-step
            # pipeline instead of only single-call smoke tests.
            return QdrantAdapter(path=endpoint[len("path://") :])
        try:
            api_key = resolve_secret_ref(credential_ref)
        except CredentialResolutionError:
            api_key = None  # self-hosted Qdrant may run without an API key
        return QdrantAdapter(url=endpoint, api_key=api_key)

    if provider == "pinecone":
        api_key = resolve_secret_ref(credential_ref)
        return PineconeAdapter(api_key=api_key, index_host=endpoint)

    raise ValueError(f"unknown provider: {provider!r} (supported: pinecone, qdrant)")
