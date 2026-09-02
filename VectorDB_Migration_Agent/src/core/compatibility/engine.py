"""Compatibility engine (V2 §8-9, §29-30) with the V3 review's correctness fixes folded
in directly (not as a patch layer):

- Euclidean<->{cosine,dot} classifies TRANSFORMABLE (not stuck INCOMPATIBLE/UNKNOWN) once
  normalization is *confirmed* unit-norm: ||a-b||^2 = 2 - 2(a.b) (V3 fix #2).
- Cosine<->Dot classifies COMPATIBLE under the same confirmed-normalization condition
  (V2 §29's own point, just made mechanical).
- Quantized source storage gets its own PropertyCompatibility so callers can't miss the
  mandatory dequantize-before-benchmark step (V3 fix #4).
- Nothing here ever coerces an unconfirmed condition to a permissive classification —
  "normalization status unknown" stays UNKNOWN, not COMPATIBLE.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from core.models.canonical_ir import DataType, Metric, NormalizationSpec, VectorKind
from core.models.capability import Capability


class CompatibilityClass(str, Enum):
    EXACT = "exact"
    COMPATIBLE = "compatible"
    TRANSFORMABLE = "transformable"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class PropertyCompatibility(BaseModel):
    property: str
    classification: CompatibilityClass
    reason: str


class CompatibilityReport(BaseModel):
    dimension: PropertyCompatibility
    datatype: PropertyCompatibility
    metric: PropertyCompatibility
    vector_type: PropertyCompatibility
    quantization: PropertyCompatibility
    semantic_space: PropertyCompatibility

    @property
    def direct_copy_possible(self) -> bool:
        allowed = {CompatibilityClass.EXACT, CompatibilityClass.COMPATIBLE}
        return all(
            p.classification in allowed
            for p in (self.dimension, self.datatype, self.metric, self.vector_type, self.semantic_space)
        )


def classify_dimension(source_dim: int | None, target_dim: int | None) -> PropertyCompatibility:
    if source_dim is None or target_dim is None:
        return PropertyCompatibility(
            property="dimension",
            classification=CompatibilityClass.UNKNOWN,
            reason="source or target dimension not discovered",
        )
    if source_dim == target_dim:
        return PropertyCompatibility(
            property="dimension", classification=CompatibilityClass.EXACT, reason="equal"
        )
    return PropertyCompatibility(
        property="dimension",
        classification=CompatibilityClass.INCOMPATIBLE,
        reason=(
            f"{source_dim}D -> {target_dim}D: direct copy impossible; route through the "
            "representation analyzer for transformation candidates"
        ),
    )


_FLOATS = {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}
_QUANTIZED_DTYPES = {DataType.INT8, DataType.BINARY}


def classify_datatype(
    source_dt: DataType, target_dt: DataType, source_quantized: Capability
) -> PropertyCompatibility:
    if source_dt is DataType.UNKNOWN or target_dt is DataType.UNKNOWN:
        return PropertyCompatibility(
            property="datatype",
            classification=CompatibilityClass.UNKNOWN,
            reason="source or target datatype not discovered",
        )
    if source_dt == target_dt:
        return PropertyCompatibility(
            property="datatype", classification=CompatibilityClass.EXACT, reason="equal"
        )
    if source_quantized.is_true or source_dt in _QUANTIZED_DTYPES or target_dt in _QUANTIZED_DTYPES:
        return PropertyCompatibility(
            property="datatype",
            classification=CompatibilityClass.TRANSFORMABLE,
            reason="quantized <-> float requires dequantize/quantize; lossy, must be benchmarked",
        )
    if source_dt in _FLOATS and target_dt in _FLOATS:
        return PropertyCompatibility(
            property="datatype",
            classification=CompatibilityClass.COMPATIBLE,
            reason="float<->float cast (widen/narrow), negligible precision impact",
        )
    return PropertyCompatibility(
        property="datatype",
        classification=CompatibilityClass.UNKNOWN,
        reason=f"no known rule for {source_dt.value} -> {target_dt.value}",
    )


_COSINE_DOT = {Metric.COSINE, Metric.DOT}
_EUCLIDEAN_PAIR = {Metric.EUCLIDEAN, Metric.COSINE, Metric.DOT}


def classify_metric(
    source_metric: Metric, target_metric: Metric, normalization: NormalizationSpec
) -> PropertyCompatibility:
    if source_metric is Metric.UNKNOWN or target_metric is Metric.UNKNOWN:
        return PropertyCompatibility(
            property="metric",
            classification=CompatibilityClass.UNKNOWN,
            reason="source or target metric not discovered",
        )
    if source_metric == target_metric:
        return PropertyCompatibility(
            property="metric", classification=CompatibilityClass.EXACT, reason="equal"
        )

    pair = {source_metric, target_metric}
    confirmed_unit = normalization.is_confirmed_unit_norm

    if pair == _COSINE_DOT:
        if confirmed_unit:
            return PropertyCompatibility(
                property="metric",
                classification=CompatibilityClass.COMPATIBLE,
                reason="cosine and dot-product rankings are equivalent under confirmed unit-norm vectors (V2 §29)",
            )
        return PropertyCompatibility(
            property="metric",
            classification=CompatibilityClass.UNKNOWN,
            reason="cosine<->dot equivalence requires confirmed unit-norm vectors; normalization not confirmed",
        )

    if Metric.EUCLIDEAN in pair and pair & _COSINE_DOT:
        if confirmed_unit:
            return PropertyCompatibility(
                property="metric",
                classification=CompatibilityClass.TRANSFORMABLE,
                reason="||a-b||^2 = 2 - 2(a.b) under confirmed unit-norm vectors, so Euclidean ranking "
                "is rank-equivalent to cosine/dot (V3 fix #2); no vector transform needed, only "
                "target metric configuration",
            )
        if normalization.status == "unknown":
            return PropertyCompatibility(
                property="metric",
                classification=CompatibilityClass.UNKNOWN,
                reason="Euclidean<->cosine/dot equivalence requires confirmed unit-norm vectors; normalization unknown",
            )
        return PropertyCompatibility(
            property="metric",
            classification=CompatibilityClass.INCOMPATIBLE,
            reason="vectors confirmed NOT unit-norm; Euclidean/cosine rankings are not equivalent as stored "
            "(re-normalizing at write time, V3 fix #8, could establish the condition for a future migration)",
        )

    return PropertyCompatibility(
        property="metric",
        classification=CompatibilityClass.INCOMPATIBLE,
        reason=f"no known equivalence between {source_metric.value} and {target_metric.value}",
    )


def classify_vector_type(source_kind: VectorKind, target_kind: VectorKind) -> PropertyCompatibility:
    if source_kind == target_kind:
        return PropertyCompatibility(
            property="vector_type", classification=CompatibilityClass.EXACT, reason="equal"
        )
    return PropertyCompatibility(
        property="vector_type",
        classification=CompatibilityClass.INCOMPATIBLE,
        reason="dense<->sparse requires an explicit, deliberate strategy; never auto-densify/sparsify (V2 §30)",
    )


def classify_quantization(source_quantized: Capability) -> PropertyCompatibility:
    if source_quantized.is_unknown:
        return PropertyCompatibility(
            property="quantization",
            classification=CompatibilityClass.UNKNOWN,
            reason="source quantized-storage capability not discovered",
        )
    if source_quantized.is_true:
        return PropertyCompatibility(
            property="quantization",
            classification=CompatibilityClass.TRANSFORMABLE,
            reason="source stores quantized vectors; mandatory dequantize-before-benchmark step required (V3 fix #4)",
        )
    return PropertyCompatibility(
        property="quantization",
        classification=CompatibilityClass.EXACT,
        reason="source not quantized",
    )


def classify_semantic_space(
    target_already_populated: bool,
    source_semantic_space_id: str,
    target_semantic_space_id: str | None = None,
) -> PropertyCompatibility:
    """Dimension is not the same thing as embedding space (two different models can both
    output 768D and be geometrically unrelated) — PCA/random_projection/MRL are only
    mathematically valid *within* one space; mixing spaces inside one target collection
    silently corrupts its retrieval consistency. No vector DB API exposes "what model
    embedded these existing vectors", so this can only be checked when we have a reason
    to trust what's already there — an empty target (nothing to conflict with) or a
    resumed migration's own prior provenance (we know what WE wrote). Anything else with
    existing data stays honestly UNKNOWN rather than assumed safe."""
    if not target_already_populated:
        return PropertyCompatibility(
            property="semantic_space",
            classification=CompatibilityClass.EXACT,
            reason="target has no existing vectors; this migration defines its semantic space",
        )
    if target_semantic_space_id is None:
        return PropertyCompatibility(
            property="semantic_space",
            classification=CompatibilityClass.UNKNOWN,
            reason=(
                f"target already contains vectors and this system cannot discover what embedded "
                f"them; source semantic space is {source_semantic_space_id!r} — verify manually "
                f"that the target's existing vectors share this space before proceeding, since "
                f"mixing embedding spaces in one collection silently corrupts retrieval"
            ),
        )
    if target_semantic_space_id == source_semantic_space_id:
        return PropertyCompatibility(
            property="semantic_space",
            classification=CompatibilityClass.EXACT,
            reason=f"target's existing vectors share source's semantic space ({source_semantic_space_id!r})",
        )
    return PropertyCompatibility(
        property="semantic_space",
        classification=CompatibilityClass.INCOMPATIBLE,
        reason=(
            f"target already contains vectors from a different semantic space "
            f"({target_semantic_space_id!r}) than source ({source_semantic_space_id!r}); "
            f"writing source-derived vectors here would mix incompatible spaces in one collection"
        ),
    )


def analyze_compatibility(
    *,
    source_dim: int | None,
    target_dim: int | None,
    source_dt: DataType,
    target_dt: DataType,
    source_metric: Metric,
    target_metric: Metric,
    source_kind: VectorKind,
    target_kind: VectorKind,
    source_quantized: Capability,
    normalization: NormalizationSpec,
    target_already_populated: bool = False,
    source_semantic_space_id: str = "unknown/unknown",
    target_semantic_space_id: str | None = None,
) -> CompatibilityReport:
    return CompatibilityReport(
        dimension=classify_dimension(source_dim, target_dim),
        datatype=classify_datatype(source_dt, target_dt, source_quantized),
        metric=classify_metric(source_metric, target_metric, normalization),
        vector_type=classify_vector_type(source_kind, target_kind),
        quantization=classify_quantization(source_quantized),
        semantic_space=classify_semantic_space(
            target_already_populated, source_semantic_space_id, target_semantic_space_id
        ),
    )
