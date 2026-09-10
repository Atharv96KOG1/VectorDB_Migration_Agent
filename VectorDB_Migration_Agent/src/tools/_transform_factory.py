"""Strategy name -> transformer instance, shared by planning (can_apply checks),
transform/benchmark (prepare+transform on a sample), and execution (prepare+transform on
the full scan) so there's exactly one place that knows how to construct each candidate.
"""

from __future__ import annotations

from core.compatibility.engine import CompatibilityReport
from core.models.migration_plan import TransformStrategy
from core.transformations.base import RepresentationTransformer
from core.transformations.direct import DirectCopyTransformer
from core.transformations.distillation import DistillationTransformer
from core.transformations.linear_mapping import (
    LowRankAffineMappingTransformer,
    OrthogonalProcrustesTransformer,
    ProcrustesDiagMappingTransformer,
    ResidualMLPMappingTransformer,
    RidgeMappingTransformer,
)
from core.transformations.mrl import MRLTransformer
from core.transformations.pca import PCATransformer
from core.transformations.random_projection import RandomProjectionTransformer
from core.transformations.reembedding import ReEmbeddingTransformer
from core.transformations.smec import SMECTransformer
from core.transformations.trained_projection import (
    LearnedProjectionTransformer,
    MatryoshkaAdaptorTransformer,
    RetrievalAwareProjectionTransformer,
)
from core.transformations.vec2vec import VecToVecTransformer

# OpenAI's text-embedding-3-* family supports the `dimensions` request parameter (a real,
# server-side Matryoshka truncation, not a client-side slice) — every other model
# (text-embedding-ada-002, self-hosted/other OpenAI-compatible endpoints) rejects an
# unrecognized parameter outright, so this must never be sent blindly.
_DIMENSION_TRUNCATABLE_MODEL_PREFIX = "text-embedding-3-"


def _resolve_reembed_dimensions(
    reembed_model: str, reembed_dimensions: int | None, target_dimension: int
) -> int | None:
    """Confirmed live, 2026-08-31: re_embedding/ridge_mapping produced vectors at the
    MODEL's native dimension (1536D for text-embedding-3-small) regardless of what the
    target actually required (768D), which isn't just a lower score — it's a hard write
    failure ("Vector dimension error: expected dim: 768, got 1536"). An operator-supplied
    `reembed_dimensions` always wins; otherwise, for a model known to support truncation,
    default to exactly what the target needs instead of leaving a mismatch to fail later."""
    if reembed_dimensions is not None:
        return reembed_dimensions
    if reembed_model.startswith(_DIMENSION_TRUNCATABLE_MODEL_PREFIX):
        return target_dimension
    return None


def build_transformer(
    strategy: TransformStrategy,
    target_dimension: int,
    compatibility: CompatibilityReport | None = None,
    seed: int = 42,
    reembed_api_key: str | None = None,
    reembed_model: str = "text-embedding-3-small",
    reembed_dimensions: int | None = None,
) -> RepresentationTransformer:
    reembed_dimensions = _resolve_reembed_dimensions(
        reembed_model, reembed_dimensions, target_dimension
    )
    if strategy == TransformStrategy.DIRECT_COPY:
        if compatibility is None:
            raise ValueError("direct_copy requires a compatibility report")
        return DirectCopyTransformer(compatibility)
    if strategy == TransformStrategy.PCA:
        return PCATransformer(target_dimension, seed=seed)
    if strategy == TransformStrategy.RANDOM_PROJECTION:
        return RandomProjectionTransformer(target_dimension, seed=seed)
    if strategy == TransformStrategy.MRL:
        return MRLTransformer(target_dimension)
    if strategy == TransformStrategy.MATRYOSHKA_ADAPTOR:
        return MatryoshkaAdaptorTransformer()
    if strategy == TransformStrategy.LEARNED_PROJECTION:
        return LearnedProjectionTransformer()
    if strategy == TransformStrategy.RETRIEVAL_AWARE_PROJECTION:
        return RetrievalAwareProjectionTransformer()
    if strategy == TransformStrategy.KNOWLEDGE_DISTILLATION:
        return DistillationTransformer()
    if strategy == TransformStrategy.RE_EMBEDDING:
        return ReEmbeddingTransformer(
            api_key=reembed_api_key, model=reembed_model, dimensions=reembed_dimensions
        )
    if strategy == TransformStrategy.VEC2VEC:
        return VecToVecTransformer()
    if strategy == TransformStrategy.SMEC:
        return SMECTransformer()
    if strategy == TransformStrategy.RIDGE_MAPPING:
        return RidgeMappingTransformer(
            api_key=reembed_api_key, model=reembed_model, dimensions=reembed_dimensions
        )
    if strategy == TransformStrategy.PROCRUSTES_MAPPING:
        return OrthogonalProcrustesTransformer(
            api_key=reembed_api_key, model=reembed_model, dimensions=reembed_dimensions
        )
    if strategy == TransformStrategy.PROCRUSTES_DIAG_MAPPING:
        return ProcrustesDiagMappingTransformer(
            api_key=reembed_api_key, model=reembed_model, dimensions=reembed_dimensions
        )
    if strategy == TransformStrategy.LOW_RANK_AFFINE_MAPPING:
        return LowRankAffineMappingTransformer(
            api_key=reembed_api_key, model=reembed_model, dimensions=reembed_dimensions
        )
    if strategy == TransformStrategy.RESIDUAL_MLP_MAPPING:
        return ResidualMLPMappingTransformer(
            api_key=reembed_api_key, model=reembed_model, dimensions=reembed_dimensions
        )
    raise ValueError(f"unknown strategy: {strategy}")


# Strategies whose fitted_params holds a full weight matrix/array meant only for the JSON
# checkpoint's MIGRATE-time reload path (core/transformations/*.py's from_fitted) — never
# for a human/UI-facing report. Confirmed live, 2026-09-02: embedding one of these
# directly in the workflow result (as tools/report_tools.py once did) caused real,
# visible lag rendering the platform's Result panel — a 1024x1536 matrix alone is over
# 1.5 million floats as JSON text.
_RECONSTRUCTABLE_FOR_PROVENANCE = {
    TransformStrategy.PCA: PCATransformer,
    TransformStrategy.RANDOM_PROJECTION: RandomProjectionTransformer,
    TransformStrategy.RIDGE_MAPPING: RidgeMappingTransformer,
    TransformStrategy.PROCRUSTES_MAPPING: OrthogonalProcrustesTransformer,
    TransformStrategy.PROCRUSTES_DIAG_MAPPING: ProcrustesDiagMappingTransformer,
    TransformStrategy.LOW_RANK_AFFINE_MAPPING: LowRankAffineMappingTransformer,
}


def provenance_parameters_from_fitted(
    strategy: TransformStrategy,
    fitted_params: dict,
    target_dimension: int,
    reembed_model: str = "text-embedding-3-small",
) -> dict:
    """Small, report-safe summary of a fitted transform — reconstructs just enough of the
    transformer to call its own `.provenance()['parameters']`, never returns the raw
    fitted_params dict itself (see _RECONSTRUCTABLE_FOR_PROVENANCE's docstring). Strategies
    with no meaningful fitted parameters (direct_copy, mrl, stubs) get an empty dict."""
    cls = _RECONSTRUCTABLE_FOR_PROVENANCE.get(strategy)
    if cls is None:
        return {}
    if cls in (PCATransformer, RandomProjectionTransformer):
        transformer = cls.from_fitted(fitted_params, target_dimension)
    else:
        transformer = cls.from_fitted(fitted_params, model=reembed_model)
    return transformer.provenance()["parameters"]
