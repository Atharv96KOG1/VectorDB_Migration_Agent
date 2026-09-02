from __future__ import annotations

import numpy as np
import pytest

from core.compatibility.engine import analyze_compatibility
from core.models.canonical_ir import (
    DataType,
    EmbeddingProvenance,
    Metric,
    MrlSpec,
    NormalizationSpec,
    VectorKind,
)
from core.models.capability import Capability, CapabilityReport
from core.transformations.base import ApplicabilityInputs, TransformContext
from core.transformations.direct import DirectCopyTransformer
from core.transformations.distillation import DistillationTransformer
from core.transformations.mrl import MRLTransformer, detect_mrl_support
from core.transformations.pca import PCATransformer
from core.transformations.random_projection import RandomProjectionTransformer
from core.transformations.rrf import apply_rrf_recovery, reciprocal_rank_fusion
from core.transformations.smec import SMECTransformer
from core.transformations.trained_projection import RetrievalAwareProjectionTransformer
from core.transformations.vec2vec import VecToVecTransformer

RNG = np.random.default_rng(0)


def _random_unit_vectors(n: int, dim: int) -> np.ndarray:
    v = RNG.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _inputs(
    source_dim, target_dim, embedding=None, capabilities=None, context=None
) -> ApplicabilityInputs:
    return ApplicabilityInputs(
        source_dim=source_dim,
        target_dim=target_dim,
        embedding=embedding or EmbeddingProvenance(),
        capabilities=capabilities or CapabilityReport(provider="test"),
        context=context or TransformContext(),
    )


async def test_direct_copy_is_identity():
    compat = analyze_compatibility(
        source_dim=8,
        target_dim=8,
        source_dt=DataType.FLOAT32,
        target_dt=DataType.FLOAT32,
        source_metric=Metric.COSINE,
        target_metric=Metric.COSINE,
        source_kind=VectorKind.DENSE,
        target_kind=VectorKind.DENSE,
        source_quantized=Capability.FALSE,
        normalization=NormalizationSpec(status="detected", method="l2", confidence=0.99),
    )
    transformer = DirectCopyTransformer(compat)
    result = transformer.can_apply(_inputs(8, 8))
    assert result.possible is True

    vectors = _random_unit_vectors(5, 8)
    out = await transformer.transform(vectors)
    assert np.allclose(out, vectors)
    assert out is not vectors  # copy, not the same array


async def test_pca_reduces_dimension_and_round_trips_fitted_params():
    vectors = _random_unit_vectors(200, 32)
    transformer = PCATransformer(target_dimension=8, seed=42)
    assert transformer.can_apply(_inputs(32, 8)).possible is True
    assert transformer.can_apply(_inputs(8, 32)).possible is False  # can't "reduce" upward

    transformer.prepare(vectors, TransformContext())
    out = await transformer.transform(vectors)
    assert out.shape == (200, 8)

    # from_fitted must reproduce byte-identical output from persisted params (V2 §53
    # reproducibility — this is exactly what tools/execution_tools.py relies on to avoid
    # re-fitting PCA per batch).
    reloaded = PCATransformer.from_fitted(transformer.fitted_params, target_dimension=8)
    out2 = await reloaded.transform(vectors)
    assert np.allclose(out, out2)


async def test_random_projection_is_deterministic_given_seed_and_round_trips():
    vectors = _random_unit_vectors(50, 16)
    t1 = RandomProjectionTransformer(target_dimension=4, seed=7)
    t1.prepare(vectors, TransformContext())
    out1 = await t1.transform(vectors)

    t2 = RandomProjectionTransformer(target_dimension=4, seed=7)
    t2.prepare(vectors, TransformContext())
    out2 = await t2.transform(vectors)
    assert np.allclose(out1, out2), "same seed must produce the same projection matrix"

    reloaded = RandomProjectionTransformer.from_fitted(t1.fitted_params, target_dimension=4)
    out3 = await reloaded.transform(vectors)
    assert np.allclose(out1, out3)

    t3 = RandomProjectionTransformer(target_dimension=4, seed=99)
    t3.prepare(vectors, TransformContext())
    out4 = await t3.transform(vectors)
    assert not np.allclose(out1, out4), "different seed must produce a different projection"


async def test_random_projection_preserves_dot_products_exactly_on_expansion():
    # Real operator question, 2026-09-01: does random_projection lose information when
    # EXPANDING dimension (e.g. 1024D source -> 1536D target)? Answer: no, not when the
    # matrix has orthonormal rows (M @ M.T == I) — achievable only because expansion
    # doesn't force any compression, unlike reduction. This is exact linear algebra, not
    # an approximate/probabilistic (Johnson-Lindenstrauss) guarantee.
    source_dim, target_dim, n = 16, 24, 50
    vectors = _random_unit_vectors(n, source_dim)
    transformer = RandomProjectionTransformer(target_dimension=target_dim, seed=7)
    transformer.prepare(vectors, TransformContext())
    out = await transformer.transform(vectors)

    original_gram = vectors @ vectors.T
    projected_gram = out @ out.T
    assert np.allclose(original_gram, projected_gram, atol=1e-4), (
        "expansion must preserve every pairwise dot product exactly"
    )


async def test_random_projection_reduction_still_uses_the_approximate_jl_matrix():
    # The exact-preservation guarantee is mathematically impossible for reduction (can't
    # fit source_dim mutually-orthonormal directions into fewer target dimensions) — this
    # locks in that the reduction path is unchanged, still the plain scaled-Gaussian JL
    # matrix, not silently switched to an impossible exact-orthonormal-rows construction.
    source_dim, target_dim = 16, 4
    transformer = RandomProjectionTransformer(target_dimension=target_dim, seed=7)
    transformer.prepare(_random_unit_vectors(50, source_dim), TransformContext())
    matrix = transformer.fitted_params["matrix"]
    rows = np.array(matrix)
    gram = rows @ rows.T
    assert not np.allclose(gram, np.eye(source_dim), atol=1e-2), (
        "reduction matrix cannot have exactly orthonormal rows — that would be mathematically impossible"
    )


def test_mrl_detect_returns_none_for_unknown_model():
    assert detect_mrl_support("some-totally-unknown-model") is None
    assert detect_mrl_support("text-embedding-3-small") == [512, 1536]


async def test_mrl_transformer_requires_confirmed_support_and_documented_dimension():
    unsupported = _inputs(
        1536, 512, embedding=EmbeddingProvenance(mrl=MrlSpec(supported=Capability.UNKNOWN))
    )
    assert MRLTransformer(512).can_apply(unsupported).possible is False

    wrong_dim = _inputs(
        1536,
        500,
        embedding=EmbeddingProvenance(
            mrl=MrlSpec(supported=Capability.TRUE, supported_dimensions=[512])
        ),
    )
    assert MRLTransformer(500).can_apply(wrong_dim).possible is False

    ok = _inputs(
        1536,
        512,
        embedding=EmbeddingProvenance(
            mrl=MrlSpec(supported=Capability.TRUE, supported_dimensions=[512])
        ),
    )
    transformer = MRLTransformer(512)
    assert transformer.can_apply(ok).possible is True

    vectors = _random_unit_vectors(10, 1536) * 3.0  # not unit-norm on purpose
    out = await transformer.transform(vectors)
    assert out.shape == (10, 512)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "MRL truncation must renormalize"


def test_reciprocal_rank_fusion_rewards_agreement():
    dense = ["a", "b", "c", "d"]
    sparse = ["b", "a", "e", "f"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    fused_ids = [doc_id for doc_id, _ in fused]
    # "a" and "b" are ranked highly in both lists, so they should fuse to the top.
    assert set(fused_ids[:2]) == {"a", "b"}


def test_apply_rrf_recovery_returns_requested_top_k():
    dense = [f"d{i}" for i in range(20)]
    sparse = [f"d{i}" for i in range(19, -1, -1)]
    result = apply_rrf_recovery(dense, sparse, top_k=5)
    assert len(result) == 5


async def test_trained_projection_stub_raises_not_implemented_not_fake_output():
    transformer = RetrievalAwareProjectionTransformer()
    inputs = _inputs(
        1536, 512, context=TransformContext(extra={"historical_queries_available": True})
    )
    result = transformer.can_apply(inputs)
    assert result.possible is True  # possible-in-principle, per the honest-stub contract
    assert transformer.is_stub is True

    with pytest.raises(NotImplementedError):
        transformer.prepare(_random_unit_vectors(10, 1536), TransformContext())


async def test_distillation_stub_raises_not_implemented():
    transformer = DistillationTransformer()
    with pytest.raises(NotImplementedError):
        transformer.prepare(_random_unit_vectors(10, 32), TransformContext())


def test_smec_is_never_applicable_as_a_post_hoc_transform():
    # Unlike the trained-projection family, SMEC is a training-time objective jointly
    # learned with the source encoder — there's no dimension/capability combination that
    # makes retrofitting it onto already-produced vectors possible.
    transformer = SMECTransformer()
    result = transformer.can_apply(_inputs(1536, 512))
    assert result.possible is False
    assert transformer.is_stub is True

    with pytest.raises(NotImplementedError):
        transformer.prepare(_random_unit_vectors(10, 1536), TransformContext())


def test_vec2vec_only_possible_in_the_documented_worst_case():
    unknown_model = EmbeddingProvenance(model="unknown")
    known_model = EmbeddingProvenance(model="text-embedding-3-small")
    no_docs_caps = CapabilityReport(provider="test")

    worst_case = _inputs(1536, 512, embedding=unknown_model, capabilities=no_docs_caps)
    assert VecToVecTransformer().can_apply(worst_case).possible is True

    known_model_case = _inputs(1536, 512, embedding=known_model, capabilities=no_docs_caps)
    assert VecToVecTransformer().can_apply(known_model_case).possible is False

    queries_available = _inputs(
        1536,
        512,
        embedding=unknown_model,
        capabilities=no_docs_caps,
        context=TransformContext(extra={"historical_queries_available": True}),
    )
    assert VecToVecTransformer().can_apply(queries_available).possible is False
