"""Ridge and Orthogonal Procrustes calibration mappings — proven the same way as
reembedding.py: a mocked OpenAI-compatible /embeddings endpoint, with the mock handler
returning a KNOWN linear relationship to the calibration source vectors so the test can
assert the fitted mapping actually recovers it, not just that "something" was returned.
"""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
from scipy.linalg import qr

from core.models.canonical_ir import EmbeddingProvenance
from core.models.capability import Capability, CapabilityReport, DataCapabilities
from core.transformations.base import ApplicabilityInputs, TransformContext
from core.transformations.linear_mapping import (
    LowRankAffineMappingTransformer,
    OrthogonalProcrustesTransformer,
    ProcrustesDiagMappingTransformer,
    ResidualMLPMappingTransformer,
    RidgeMappingTransformer,
)

RNG = np.random.default_rng(0)


def _inputs(source_dim, target_dim, capabilities=None, context=None) -> ApplicabilityInputs:
    return ApplicabilityInputs(
        source_dim=source_dim,
        target_dim=target_dim,
        embedding=EmbeddingProvenance(),
        capabilities=capabilities or CapabilityReport(provider="test"),
        context=context or TransformContext(),
    )


def _lookup_handler(target_lookup: np.ndarray):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        embeddings = [
            {"embedding": target_lookup[int(text.split("-")[1])].tolist()}
            for text in body["input"]
        ]
        return httpx.Response(200, json={"data": embeddings})

    return handler


def test_can_apply_false_without_documents():
    caps = CapabilityReport(provider="test", data=DataCapabilities(retrieve_documents=Capability.UNKNOWN))
    result = RidgeMappingTransformer().can_apply(_inputs(1536, 768, capabilities=caps))
    assert result.possible is False


def test_can_apply_true_with_documents_regardless_of_dimension_change():
    result = RidgeMappingTransformer().can_apply(
        _inputs(1536, 768, context=TransformContext(documents=["hi"]))
    )
    assert result.possible is True


def test_procrustes_refuses_when_dimensions_differ_even_with_documents():
    result = OrthogonalProcrustesTransformer().can_apply(
        _inputs(1536, 768, context=TransformContext(documents=["hi"]))
    )
    assert result.possible is False
    assert "source_dim == target_dim" in result.reason


def test_procrustes_possible_when_dimensions_match_and_documents_present():
    result = OrthogonalProcrustesTransformer().can_apply(
        _inputs(1024, 1024, context=TransformContext(documents=["hi"]))
    )
    assert result.possible is True


async def test_transform_without_enough_calibration_documents_raises_clear_error():
    n = 210
    source = RNG.standard_normal((n, 6)).astype(np.float32)
    transformer = RidgeMappingTransformer(api_key="k")
    transformer.prepare(source, TransformContext(documents=["only-a-few"] * 5))
    with pytest.raises(ValueError, match="calibration"):
        await transformer.transform(source)


async def test_transform_without_api_key_raises_clear_error():
    n = 210
    source = RNG.standard_normal((n, 6)).astype(np.float32)
    documents = [f"doc-{i}" for i in range(n)]
    transformer = RidgeMappingTransformer()
    transformer.prepare(source, TransformContext(documents=documents))
    with pytest.raises(RuntimeError, match="api_key"):
        await transformer.transform(source)


async def test_ridge_mapping_recovers_a_known_linear_relationship():
    n, source_dim, target_dim = 220, 6, 4
    source = RNG.standard_normal((n, source_dim)).astype(np.float32)
    w = RNG.standard_normal((source_dim, target_dim)).astype(np.float32)
    bias = RNG.standard_normal(target_dim).astype(np.float32)
    target_lookup = (source @ w + bias).astype(np.float32)
    documents = [f"doc-{i}" for i in range(n)]

    transformer = RidgeMappingTransformer(
        api_key="test-key",
        transport=httpx.MockTransport(_lookup_handler(target_lookup)),
        calibration_size=n,
        alpha=1e-6,
    )
    transformer.prepare(source, TransformContext(documents=documents))
    out = await transformer.transform(source)
    assert out.shape == (n, target_dim)
    assert np.allclose(out, target_lookup, atol=1e-2)
    await transformer.close()


async def test_ridge_mapping_fitted_params_round_trip_matches_original():
    # Proves the MIGRATE-time reload path (tools/execution_tools.py:_load_execution_transformer
    # persists fitted_params once and reconstructs via from_fitted for later batches,
    # never re-fitting or re-calling the embeddings API per batch).
    n, source_dim, target_dim = 210, 5, 3
    source = RNG.standard_normal((n, source_dim)).astype(np.float32)
    w = RNG.standard_normal((source_dim, target_dim)).astype(np.float32)
    target_lookup = (source @ w).astype(np.float32)
    documents = [f"doc-{i}" for i in range(n)]

    transformer = RidgeMappingTransformer(
        api_key="test-key",
        transport=httpx.MockTransport(_lookup_handler(target_lookup)),
        calibration_size=n,
        alpha=1e-6,
    )
    transformer.prepare(source, TransformContext(documents=documents))
    out = await transformer.transform(source)
    await transformer.close()

    reloaded = RidgeMappingTransformer.from_fitted(transformer.fitted_params)
    out_reloaded = await reloaded.transform(source)
    assert np.allclose(out, out_reloaded)


async def test_orthogonal_procrustes_recovers_a_known_rotation():
    n, dim = 220, 5
    source = RNG.standard_normal((n, dim)).astype(np.float32)
    rotation, _ = qr(RNG.standard_normal((dim, dim)))
    translation = RNG.standard_normal(dim).astype(np.float32)
    target_lookup = (source @ rotation + translation).astype(np.float32)
    documents = [f"doc-{i}" for i in range(n)]

    transformer = OrthogonalProcrustesTransformer(
        api_key="test-key",
        transport=httpx.MockTransport(_lookup_handler(target_lookup)),
        calibration_size=n,
    )
    transformer.prepare(source, TransformContext(documents=documents))
    out = await transformer.transform(source)
    assert out.shape == (n, dim)
    assert np.allclose(out, target_lookup, atol=1e-2)
    await transformer.close()


async def test_orthogonal_procrustes_fitted_params_round_trip_matches_original():
    n, dim = 220, 5
    source = RNG.standard_normal((n, dim)).astype(np.float32)
    rotation, _ = qr(RNG.standard_normal((dim, dim)))
    translation = RNG.standard_normal(dim).astype(np.float32)
    target_lookup = (source @ rotation + translation).astype(np.float32)
    documents = [f"doc-{i}" for i in range(n)]

    transformer = OrthogonalProcrustesTransformer(
        api_key="test-key",
        transport=httpx.MockTransport(_lookup_handler(target_lookup)),
        calibration_size=n,
    )
    transformer.prepare(source, TransformContext(documents=documents))
    out = await transformer.transform(source)
    await transformer.close()

    reloaded = OrthogonalProcrustesTransformer.from_fitted(transformer.fitted_params)
    out_reloaded = await reloaded.transform(source)
    assert np.allclose(out, out_reloaded)


def test_procrustes_diag_refuses_when_dimensions_differ_even_with_documents():
    result = ProcrustesDiagMappingTransformer().can_apply(
        _inputs(1536, 768, context=TransformContext(documents=["hi"]))
    )
    assert result.possible is False
    assert "source_dim == target_dim" in result.reason


def test_procrustes_diag_possible_when_dimensions_match_and_documents_present():
    result = ProcrustesDiagMappingTransformer().can_apply(
        _inputs(1024, 1024, context=TransformContext(documents=["hi"]))
    )
    assert result.possible is True


async def test_procrustes_diag_recovers_a_known_sign_flip():
    n, dim = 220, 6
    source = RNG.standard_normal((n, dim)).astype(np.float32)
    diag = np.array([1, -1, 1, 1, -1, -1], dtype=np.float32)
    translation = RNG.standard_normal(dim).astype(np.float32)
    target_lookup = (source * diag + translation).astype(np.float32)
    documents = [f"doc-{i}" for i in range(n)]

    transformer = ProcrustesDiagMappingTransformer(
        api_key="test-key",
        transport=httpx.MockTransport(_lookup_handler(target_lookup)),
        calibration_size=n,
    )
    transformer.prepare(source, TransformContext(documents=documents))
    out = await transformer.transform(source)
    assert out.shape == (n, dim)
    assert np.allclose(out, target_lookup, atol=1e-2)
    await transformer.close()


async def test_procrustes_diag_fitted_params_round_trip_matches_original():
    n, dim = 220, 5
    source = RNG.standard_normal((n, dim)).astype(np.float32)
    diag = np.array([-1, 1, 1, -1, 1], dtype=np.float32)
    target_lookup = (source * diag).astype(np.float32)
    documents = [f"doc-{i}" for i in range(n)]

    transformer = ProcrustesDiagMappingTransformer(
        api_key="test-key",
        transport=httpx.MockTransport(_lookup_handler(target_lookup)),
        calibration_size=n,
    )
    transformer.prepare(source, TransformContext(documents=documents))
    out = await transformer.transform(source)
    await transformer.close()

    reloaded = ProcrustesDiagMappingTransformer.from_fitted(transformer.fitted_params)
    out_reloaded = await reloaded.transform(source)
    assert np.allclose(out, out_reloaded)


async def test_low_rank_affine_recovers_a_known_rank2_relationship():
    n, source_dim, target_dim, rank = 220, 8, 6, 2
    source = RNG.standard_normal((n, source_dim)).astype(np.float32)
    u = RNG.standard_normal((source_dim, rank)).astype(np.float32)
    v = RNG.standard_normal((rank, target_dim)).astype(np.float32)
    w = u @ v
    bias = RNG.standard_normal(target_dim).astype(np.float32)
    target_lookup = (source @ w + bias).astype(np.float32)
    documents = [f"doc-{i}" for i in range(n)]

    transformer = LowRankAffineMappingTransformer(
        api_key="test-key",
        transport=httpx.MockTransport(_lookup_handler(target_lookup)),
        calibration_size=n,
        alpha=1e-6,
        rank=rank,
    )
    transformer.prepare(source, TransformContext(documents=documents))
    out = await transformer.transform(source)
    assert out.shape == (n, target_dim)
    assert np.allclose(out, target_lookup, atol=1e-1)
    await transformer.close()


async def test_low_rank_affine_fitted_params_round_trip_matches_original():
    n, source_dim, target_dim = 220, 5, 3
    source = RNG.standard_normal((n, source_dim)).astype(np.float32)
    w = RNG.standard_normal((source_dim, target_dim)).astype(np.float32)
    target_lookup = (source @ w).astype(np.float32)
    documents = [f"doc-{i}" for i in range(n)]

    transformer = LowRankAffineMappingTransformer(
        api_key="test-key",
        transport=httpx.MockTransport(_lookup_handler(target_lookup)),
        calibration_size=n,
        alpha=1e-6,
    )
    transformer.prepare(source, TransformContext(documents=documents))
    out = await transformer.transform(source)
    await transformer.close()

    reloaded = LowRankAffineMappingTransformer.from_fitted(transformer.fitted_params)
    out_reloaded = await reloaded.transform(source)
    assert np.allclose(out, out_reloaded)


def test_residual_mlp_mapping_can_apply_flags_not_implemented():
    result = ResidualMLPMappingTransformer().can_apply(
        _inputs(1536, 768, context=TransformContext(documents=["hi"]))
    )
    assert result.possible is True
    assert "NOT IMPLEMENTED" in result.reason


async def test_residual_mlp_mapping_prepare_raises_honest_not_implemented():
    n = 210
    source = RNG.standard_normal((n, 6)).astype(np.float32)
    documents = [f"doc-{i}" for i in range(n)]
    transformer = ResidualMLPMappingTransformer(api_key="k")
    transformer.prepare(source, TransformContext(documents=documents))
    with pytest.raises(NotImplementedError, match="PyTorch"):
        await transformer.transform(source)
