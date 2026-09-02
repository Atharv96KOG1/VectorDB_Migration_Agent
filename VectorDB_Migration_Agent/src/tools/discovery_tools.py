"""DISCOVER state (V2 §6, §27-28, §44 Discovery Agent): capability discovery per side,
resource info, and — on the source only — a vector sample used to detect normalization
and (best-effort) MRL support. Everything unresolvable is recorded Capability.UNKNOWN,
never coerced to FALSE.
"""

from __future__ import annotations

import numpy as np
from aetherion_sdk import tool

from core.checkpointing.store import load_checkpoint, save_checkpoint
from core.models.canonical_ir import MrlSpec, NormalizationSpec
from core.transformations.mrl import detect_mrl_support
from core.validation.engine import analyze_normalization
from tools._shared import build_adapter

SAMPLE_SIZE = 200
_LANGUAGE_PAYLOAD_KEYS = ("language", "lang", "language_code", "locale")


def _detect_languages(payloads: list[dict]) -> list[str]:
    """Best-effort: distinct values found under a common language-ish payload key.
    NOT a language detector over vector content — there is none here. Empty means no
    such key was present in the sample, not "the source is monolingual"."""
    found: set[str] = set()
    for payload in payloads:
        for key in _LANGUAGE_PAYLOAD_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value:
                found.add(value)
                break
    return sorted(found)


def _build_embedding_discovery(
    embedding_config: dict,
    mrl: MrlSpec,
    languages_detected: list[str],
    normalization_status: str,
) -> dict:
    """The rest of V2 §8/§20's embedding contract (provider, model, revision, tokenizer,
    pooling, prefixes) is not recoverable from raw vector math — it's only ever as good as
    what the operator supplies via the trigger payload's `source_embedding_config`. When
    they supply a real provider+model, that's an explicit, confident identity
    (discovered_from="operator_supplied", confidence=1.0) — a materially different claim
    than the "unknown/unknown" this build defaults to otherwise, since
    core/compatibility/engine.py:classify_semantic_space treats two "unknown/unknown"
    sources as trivially matching, which is only honest when nobody actually knows."""
    provider = embedding_config.get("provider") or "unknown"
    model = embedding_config.get("model") or "unknown"
    operator_supplied_identity = provider != "unknown" and model != "unknown"
    return {
        "provider": provider,
        "model": model,
        "revision": embedding_config.get("revision"),
        "tokenizer": embedding_config.get("tokenizer"),
        "pooling": embedding_config.get("pooling"),
        "query_prefix": embedding_config.get("query_prefix"),
        "document_prefix": embedding_config.get("document_prefix"),
        "mrl": mrl.model_dump(),
        "discovered_from": "operator_supplied" if operator_supplied_identity else "vector_sample_statistics",
        "confidence": 1.0 if operator_supplied_identity else (0.5 if normalization_status != "unknown" else 0.0),
        "languages_detected": languages_detected,
    }


@tool()
async def discover_resources(migration_id: str) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(
            f"no checkpoint found for migration_id={migration_id!r}; run CONNECT first"
        )
    req = checkpoint["request"]
    namespace = req.get("namespace")

    embedding_config = req.get("source_embedding_config") or {}

    discovery: dict = {}
    normalization = NormalizationSpec()
    mrl = MrlSpec()
    languages_detected: list[str] = []

    for side in ("source", "target"):
        provider = req[f"{side}_provider"]
        resource = req[f"{side}_resource"]
        adapter = build_adapter(
            provider, req[f"{side}_credential_ref"], req.get(f"{side}_endpoint_ref")
        )
        try:
            capabilities = await adapter.discover_capabilities()
            info = await adapter.get_resource_info(resource, namespace=namespace)
            discovery[side] = {
                "capabilities": capabilities.model_dump(),
                "resource_info": info.model_dump(),
            }
            if side == "source":
                sample = await adapter.sample_vectors(resource, SAMPLE_SIZE, namespace=namespace)
                vectors = [r.vector for r in sample if r.vector]
                if vectors:
                    normalization = analyze_normalization(np.array(vectors, dtype=np.float32))
                discovery["sample_size"] = len(vectors)
                languages_detected = _detect_languages([r.payload or {} for r in sample])
        finally:
            await adapter.close()

    # Embedding model identity is rarely programmatically discoverable from raw vectors
    # alone (V2 §27) — it stays "unknown" unless the operator supplies it via
    # source_embedding_config, or a future adapter surfaces it from index metadata (e.g.
    # Pinecone integrated-inference indexes). MRL support is looked up against the small
    # curated registry using whatever model name is known; absence from it is UNKNOWN,
    # not FALSE.
    supported_dims = detect_mrl_support(embedding_config.get("model") or "unknown")
    if supported_dims is not None:
        mrl = MrlSpec(supported="TRUE", supported_dimensions=supported_dims)

    discovery["normalization"] = normalization.model_dump()
    discovery["embedding"] = _build_embedding_discovery(
        embedding_config, mrl, languages_detected, normalization.status
    )

    checkpoint["discovery"] = discovery
    checkpoint["history"].append(
        {"state": "DISCOVER", "detail": {"sample_size": discovery.get("sample_size", 0)}}
    )
    save_checkpoint(migration_id, checkpoint)

    return {
        "migration_id": migration_id,
        "source_dimension": discovery["source"]["resource_info"]["dimension"],
        "target_dimension": discovery["target"]["resource_info"]["dimension"],
        "normalization_status": normalization.status,
    }
