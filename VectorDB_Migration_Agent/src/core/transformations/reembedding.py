"""Re-embedding (V2 §18) — the strongest option when source documents are available and
the target needs a genuinely different embedding space, at the cost of inference calls.
Real implementation (unlike the trained-projection family) because it needs no training
loop: call an embeddings API on the documents. Untestable live in this sandbox (needs a
real API key), same tier as the Pinecone adapter — proven with an httpx.MockTransport
unit test instead.

Uses any OpenAI-compatible `/embeddings` endpoint (works for OpenAI itself and compatible
self-hosted/proxy servers) so this isn't locked to one embedding provider.
"""

from __future__ import annotations

import asyncio

import httpx
import numpy as np

from core.transformations.base import (
    ApplicabilityInputs,
    ApplicabilityResult,
    RepresentationTransformer,
)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_BATCH_SIZE = 100
# httpx's default (5s across connect/read/write/pool) is too aggressive for a real
# embeddings call — confirmed live, 2026-08-31: a 200-document calibration batch
# (core/transformations/linear_mapping.py) hit a bare "ReadTimeout" under real concurrent
# candidate evaluation (multiple strategies calling OpenAI at once), while a slower
# same-sized call from a different candidate happened to complete. A generous, explicit
# timeout replaces silent, non-deterministic failures with an honest wait.
REQUEST_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_MAX_RETRIES = 3
_RETRY_BASE_DELAY_SECONDS = 2.0
# Retryable = transient: a timeout, or a 429/5xx the server itself says to retry (real
# behavior under contention, not a code bug) — confirmed live, 2026-08-31: multiple
# candidates (re_embedding + ridge_mapping) calling OpenAI concurrently produced a bare
# ReadTimeout even at a 60s timeout, on a request that succeeds fine in isolation. A 4xx
# other than 429 (bad request, bad key, ...) is permanent and must NOT be retried — it
# will fail identically every time and retrying would just mask a real config problem.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


async def embed_texts(
    client: httpx.AsyncClient, model: str, texts: list[str], dimensions: int | None = None
) -> np.ndarray:
    """Shared batched /embeddings caller — used by ReEmbeddingTransformer for a full
    re-embed and by core/transformations/linear_mapping.py to generate calibration-pair
    targets for a small subset of documents. Retries transient failures (timeout, 429,
    5xx) with exponential backoff; a permanent error (401, 400, ...) raises immediately.

    `dimensions` is OpenAI's native truncation for text-embedding-3-* models (server-side,
    Matryoshka-trained — not a client-side slice) — omitted from the request entirely when
    None, since older models (text-embedding-ada-002) and non-OpenAI-compatible endpoints
    reject an unrecognized parameter rather than ignoring it."""
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        body: dict = {"input": batch, "model": model}
        if dimensions is not None:
            body["dimensions"] = dimensions
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await client.post("/embeddings", json=body)
                resp.raise_for_status()
                data = resp.json()
                embeddings.extend(item["embedding"] for item in data["data"])
                break
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                is_retryable_status = (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code in _RETRYABLE_STATUS_CODES
                )
                is_timeout = isinstance(exc, httpx.TimeoutException)
                if not (is_retryable_status or is_timeout) or attempt == _MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * (2**attempt))
    return np.array(embeddings, dtype=np.float32)


class ReEmbeddingTransformer(RepresentationTransformer):
    strategy = "re_embedding"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        base_url: str = _DEFAULT_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        dimensions: int | None = None,
    ) -> None:
        # Client construction is lazy (see _ensure_client) so can_apply() can be checked
        # for candidate generation without requiring real credentials yet.
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._transport = transport
        self._dimensions = dimensions
        self._client: httpx.AsyncClient | None = None
        self._documents: list[str] = []

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("ReEmbeddingTransformer needs an api_key before transform()")
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                transport=self._transport,
                timeout=REQUEST_TIMEOUT,
            )
        return self._client

    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        if not inputs.capabilities.data.retrieve_documents.is_true and not inputs.context.documents:
            return ApplicabilityResult(
                False,
                "documents_available capability is not TRUE and no documents were supplied "
                "in the transform context — re-embedding requires source text",
            )
        return ApplicabilityResult(
            True, "documents available; will call the embedding API and benchmark the result"
        )

    def prepare(self, sample_vectors: np.ndarray, context) -> None:
        self._documents = context.documents or []

    async def transform(self, vectors: np.ndarray) -> np.ndarray:
        if not self._documents:
            raise RuntimeError(
                "ReEmbeddingTransformer.prepare() must be called with documents first"
            )
        client = self._ensure_client()
        return await embed_texts(client, self._model, self._documents, dimensions=self._dimensions)

    def provenance(self) -> dict:
        return {
            "strategy": self.strategy,
            "parameters": {
                "model": self._model,
                "dimensions": self._dimensions,
                "document_count": len(self._documents),
            },
        }

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
