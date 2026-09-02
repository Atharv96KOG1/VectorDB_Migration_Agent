"""Canonical Vector IR (V2 §5): represents what the data *means*, not how a specific
provider stores it. Every adapter translates to/from this shape so the compatibility
engine, transformers, and benchmarking code never branch on provider name.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, computed_field

from core.models.capability import Capability


class Metric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    UNKNOWN = "unknown"


class DataType(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    BINARY = "binary"
    UNKNOWN = "unknown"


class VectorKind(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"


class IdCollisionPolicy(str, Enum):
    FAIL = "fail"
    OVERWRITE = "overwrite"
    SKIP = "skip"
    RENAME = "rename"
    PREFIX = "prefix"


class MrlSpec(BaseModel):
    supported: Capability = Capability.UNKNOWN
    supported_dimensions: list[int] = Field(default_factory=list)


class NormalizationSpec(BaseModel):
    """Never silently assume normalized (V2 §28). `detected` is only True once a sample
    has actually been measured; `confidence` is the fraction of the sample within
    tolerance of unit norm."""

    status: str = "unknown"  # "unknown" | "detected" | "not_normalized"
    method: str | None = None  # e.g. "l2"
    confidence: float | None = None
    mean_norm: float | None = None
    std_norm: float | None = None

    @property
    def is_confirmed_unit_norm(self) -> bool:
        return self.status == "detected" and (self.confidence or 0.0) >= 0.95


class EmbeddingProvenance(BaseModel):
    provider: str = "unknown"
    model: str = "unknown"
    native_dimension: int | None = None
    mrl: MrlSpec = MrlSpec()
    normalization: NormalizationSpec = NormalizationSpec()
    discovered_from: str = "unknown"
    confidence: float = 0.0
    languages_detected: list[str] = Field(default_factory=list)
    """Best-effort: language codes found in a source-payload language field during
    discovery (e.g. a "language"/"lang" key). Empty means none was found in the sampled
    payloads — NOT "the source is monolingual"; there is no vector-level language
    detector here, only payload inspection."""

    revision: str | None = None
    tokenizer: str | None = None
    pooling: str | None = None
    query_prefix: str | None = None
    document_prefix: str | None = None
    """The rest of V2 §8/§20's embedding contract. None of these are discoverable from
    raw vectors — they're populated only when the operator supplies them via the trigger
    payload's `source_embedding_config` (see tools/discovery_tools.py), never guessed.
    None means "not supplied," not "the model has no tokenizer/pooling/prefix."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def semantic_space_id(self) -> str:
        """Identity of the embedding space, distinct from dimension (dimension ≠ space:
        two different models can both output 768D and still be geometrically unrelated —
        see core/compatibility/engine.py's semantic-space check). "unknown/unknown" when
        neither provider nor model could be discovered; still a valid, honest identifier —
        two UNKNOWN sources are never assumed to share a space just because they're both
        UNKNOWN."""
        return f"{self.provider}/{self.model}"


class VectorSourceSpec(BaseModel):
    kind: VectorKind = VectorKind.DENSE
    dimension: int | None = None
    datatype: DataType = DataType.UNKNOWN
    metric: Metric = Metric.UNKNOWN
    quantized: Capability = Capability.UNKNOWN
    """V3 fix #4: source stores vectors pre-quantized. Requires dequantize-before-benchmark."""


class VectorTargetSpec(BaseModel):
    kind: VectorKind = VectorKind.DENSE
    dimension: int | None = None
    datatype: DataType = DataType.UNKNOWN
    metric: Metric = Metric.UNKNOWN


class TransformationSpec(BaseModel):
    required: bool = False
    selected_strategy: str | None = None
    selected_benchmark_id: str | None = None
    """Anti-hallucination guardrail (V3 §8): a strategy may only be selected if this id
    references a row that actually exists in the executed benchmark-results table. The
    planning tool enforces this as a hard check, not a convention."""


class VectorFieldSpec(BaseModel):
    """One named vector field (V2 §31) — e.g. "title_vector", "body_vector"."""

    name: str
    source: VectorSourceSpec
    embedding: EmbeddingProvenance = EmbeddingProvenance()
    target: VectorTargetSpec
    transformation: TransformationSpec = TransformationSpec()


class PayloadFieldTransform(BaseModel):
    source_type: str
    target_type: str
    operation: (
        str  # rename | cast | flatten | nest | delete | default | derive | parse_datetime ...
    )


class PayloadSpec(BaseModel):
    schema_fields: list[str] = Field(default_factory=list)
    transforms: dict[str, PayloadFieldTransform] = Field(default_factory=dict)


class IdSpec(BaseModel):
    type: str = "string"
    collision_policy: IdCollisionPolicy = IdCollisionPolicy.FAIL


class NamespaceSpec(BaseModel):
    """Never silently drop an isolation boundary (V2 §33) — the mapping must always be
    explicit, even when it's "namespace not supported by target, folded into payload."""

    source_supported: bool = False
    target_mapping: str | None = None  # e.g. "payload.__namespace" or "collection"


class ValidationPolicy(BaseModel):
    minimum_recall_at_10: float = 0.80
    minimum_ndcg_at_10: float = 0.90
    maximum_topk_overlap_drop: float = 0.15
    maximum_latency_increase_percent: float = 20.0


class ResourceRef(BaseModel):
    provider: str
    name: str
    namespace: str | None = None


class CanonicalVectorIR(BaseModel):
    version: str = "1.0"
    migration_id: str
    mode: str = "planned"

    source: ResourceRef
    target: ResourceRef

    vectors: list[VectorFieldSpec]
    payload: PayloadSpec = PayloadSpec()
    id: IdSpec = IdSpec()
    namespace: NamespaceSpec = NamespaceSpec()
    validation: ValidationPolicy = ValidationPolicy()

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
