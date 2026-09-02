"""Proves the re-embedding transformer's request/response handling via a mocked OpenAI-
compatible endpoint — same untestable-live tier as the Pinecone adapter, same technique.
"""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

import core.transformations.reembedding as reembedding_module
from core.models.canonical_ir import EmbeddingProvenance
from core.models.capability import Capability, CapabilityReport, DataCapabilities
from core.transformations.base import ApplicabilityInputs, TransformContext
from core.transformations.reembedding import ReEmbeddingTransformer, embed_texts


def _handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    embeddings = [{"embedding": [float(len(text)), 0.0, 1.0]} for text in body["input"]]
    return httpx.Response(200, json={"data": embeddings})


def _inputs(capabilities: CapabilityReport, context: TransformContext) -> ApplicabilityInputs:
    return ApplicabilityInputs(
        source_dim=1536,
        target_dim=768,
        embedding=EmbeddingProvenance(),
        capabilities=capabilities,
        context=context,
    )


def test_can_apply_false_without_documents_capability_or_context_documents():
    caps = CapabilityReport(
        provider="test", data=DataCapabilities(retrieve_documents=Capability.UNKNOWN)
    )
    result = ReEmbeddingTransformer().can_apply(_inputs(caps, TransformContext()))
    assert result.possible is False


def test_can_apply_true_when_documents_supplied_in_context():
    caps = CapabilityReport(provider="test")
    result = ReEmbeddingTransformer().can_apply(
        _inputs(caps, TransformContext(documents=["hello world"]))
    )
    assert result.possible is True


async def test_transform_calls_embeddings_endpoint_in_batches():
    transformer = ReEmbeddingTransformer(
        api_key="test-key", model="text-embedding-3-small", transport=httpx.MockTransport(_handler)
    )
    documents = ["a", "bb", "ccc"]
    transformer.prepare(np.zeros((3, 8), dtype=np.float32), TransformContext(documents=documents))
    out = await transformer.transform(np.zeros((3, 8), dtype=np.float32))
    assert out.shape == (3, 3)
    assert list(out[:, 0]) == [1.0, 2.0, 3.0]  # len("a"), len("bb"), len("ccc")
    await transformer.close()


async def test_transform_without_prepare_raises_clear_error():
    transformer = ReEmbeddingTransformer(api_key="test-key")
    with pytest.raises(RuntimeError, match="prepare"):
        await transformer.transform(np.zeros((1, 8), dtype=np.float32))


async def test_dimensions_is_sent_when_set_and_omitted_when_not():
    seen_bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_bodies.append(body)
        return httpx.Response(200, json={"data": [{"embedding": [1.0]} for _ in body["input"]]})

    with_dims = ReEmbeddingTransformer(
        api_key="k",
        model="text-embedding-3-large",
        dimensions=1536,
        transport=httpx.MockTransport(handler),
    )
    with_dims.prepare(np.zeros((1, 8), dtype=np.float32), TransformContext(documents=["a"]))
    await with_dims.transform(np.zeros((1, 8), dtype=np.float32))
    await with_dims.close()

    without_dims = ReEmbeddingTransformer(
        api_key="k", model="text-embedding-3-small", transport=httpx.MockTransport(handler)
    )
    without_dims.prepare(np.zeros((1, 8), dtype=np.float32), TransformContext(documents=["a"]))
    await without_dims.transform(np.zeros((1, 8), dtype=np.float32))
    await without_dims.close()

    assert seen_bodies[0]["dimensions"] == 1536
    assert "dimensions" not in seen_bodies[1]  # older/other models must not see an unknown param


async def test_embed_texts_retries_a_transient_timeout_then_succeeds(monkeypatch):
    monkeypatch.setattr(reembedding_module, "_RETRY_BASE_DELAY_SECONDS", 0)
    # Confirmed live, 2026-08-31: a real ReadTimeout under concurrent candidate evaluation
    # (multiple strategies calling OpenAI at once) killed a candidate that would have
    # succeeded on a second attempt. A transient failure must be retried, not fatal.
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("simulated timeout", request=request)
        body = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"embedding": [1.0]} for _ in body["input"]]})

    client = httpx.AsyncClient(base_url="https://api.openai.com/v1", transport=httpx.MockTransport(handler))
    out = await embed_texts(client, "test-model", ["a", "b"])
    assert out.shape == (2, 1)
    assert calls["n"] == 2
    await client.aclose()


async def test_embed_texts_retries_a_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(reembedding_module, "_RETRY_BASE_DELAY_SECONDS", 0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        body = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"embedding": [1.0]} for _ in body["input"]]})

    client = httpx.AsyncClient(base_url="https://api.openai.com/v1", transport=httpx.MockTransport(handler))
    out = await embed_texts(client, "test-model", ["a"])
    assert out.shape == (1, 1)
    assert calls["n"] == 2
    await client.aclose()


async def test_embed_texts_does_not_retry_a_permanent_client_error():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "invalid api key"})

    client = httpx.AsyncClient(base_url="https://api.openai.com/v1", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await embed_texts(client, "test-model", ["a"])
    assert calls["n"] == 1  # no retry wasted on an error that will never succeed
    await client.aclose()


async def test_embed_texts_raises_after_exhausting_retries_on_persistent_timeout(monkeypatch):
    monkeypatch.setattr(reembedding_module, "_RETRY_BASE_DELAY_SECONDS", 0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("always times out", request=request)

    client = httpx.AsyncClient(base_url="https://api.openai.com/v1", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.ReadTimeout):
        await embed_texts(client, "test-model", ["a"])
    assert calls["n"] == 3  # _MAX_RETRIES
    await client.aclose()
