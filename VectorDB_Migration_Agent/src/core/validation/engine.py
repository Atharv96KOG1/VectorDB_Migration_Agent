"""Data integrity + retrieval-equivalence validation (V2 §23-25, §40).

V3 fix #3: integrity tolerance is keyed to the *discovered target storage dtype*, never
assumed FP32 — many providers silently store FP32 input as FP16/BF16/INT8 internally
(quantized indexes), and diffing against an FP32 tolerance against such a target would
either false-positive-fail a correct copy or mask a real corruption depending on which
way the rounding goes.

V2 §40: never use exact hashes/equality for mathematically transformed vectors — use
dimension/norm/cosine-similarity/L2-error/distribution statistics instead. Exact
element-wise diff (integrity_diff) is only meaningful for DIRECT COPIES, where source and
target share dimension and are expected to be numerically identical up to storage
precision.
"""

from __future__ import annotations

import numpy as np

from core.models.canonical_ir import DataType, NormalizationSpec

# (atol, rtol) per discovered target storage dtype. INT8 is a best-effort approximation
# (real tolerance depends on the provider's actual quantization scale, which isn't always
# discoverable) — flagged as approximate rather than silently treated as exact.
DTYPE_TOLERANCE: dict[DataType, tuple[float, float]] = {
    DataType.FLOAT32: (1e-5, 1e-5),
    DataType.FLOAT16: (1e-2, 1e-2),
    DataType.BFLOAT16: (1e-2, 1e-2),
    DataType.INT8: (0.05, 0.0),  # approximate: ~1/127 quantization step
    DataType.BINARY: (0.0, 0.0),
}


def tolerance_for_dtype(dtype: DataType) -> tuple[float, float]:
    return DTYPE_TOLERANCE.get(dtype, (1e-5, 1e-5))


def integrity_diff(source: np.ndarray, target: np.ndarray, target_dtype: DataType) -> dict:
    """Element-wise diff for direct copies only — source and target must share shape."""
    if source.shape != target.shape:
        raise ValueError(
            f"integrity_diff is only valid for equal-shape (direct-copy) comparisons, got "
            f"{source.shape} vs {target.shape}"
        )
    atol, rtol = tolerance_for_dtype(target_dtype)
    abs_diff = np.abs(source.astype(np.float64) - target.astype(np.float64))
    within = np.allclose(source, target, atol=atol, rtol=rtol)
    return {
        "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "atol": atol,
        "rtol": rtol,
        "target_dtype": target_dtype.value,
        "within_tolerance": bool(within),
    }


def distribution_stats(vectors: np.ndarray) -> dict:
    """Sanity-check stats for transformed output (V2 §40) — dimension, norm distribution,
    and NaN/Inf detection catch a broken transform even when there's no directly
    comparable "ground truth" vector (different dimension than source)."""
    norms = np.linalg.norm(vectors, axis=1) if vectors.size else np.array([])
    return {
        "count": int(vectors.shape[0]) if vectors.ndim else 0,
        "dimension": int(vectors.shape[1]) if vectors.ndim == 2 else None,
        "mean_norm": float(norms.mean()) if norms.size else 0.0,
        "std_norm": float(norms.std()) if norms.size else 0.0,
        "has_nan": bool(np.isnan(vectors).any()) if vectors.size else False,
        "has_inf": bool(np.isinf(vectors).any()) if vectors.size else False,
    }


def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """For transformers offering an inverse mapping (e.g. sklearn PCA.inverse_transform):
    L2 error and cosine similarity between original and reconstructed vectors. This is a
    reconstruction-quality signal, not a retrieval-quality signal — V2 §19 is explicit
    that PCA reconstruction quality must never be treated as equivalent to retrieval
    quality; use it as a secondary sanity check alongside the benchmark, not instead of it.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("reconstruction_error requires matching shapes")
    l2 = np.linalg.norm(original - reconstructed, axis=1)
    denom = np.linalg.norm(original, axis=1) * np.linalg.norm(reconstructed, axis=1)
    denom[denom == 0] = 1.0
    cosine = np.sum(original * reconstructed, axis=1) / denom
    return {
        "mean_l2_error": float(l2.mean()) if l2.size else 0.0,
        "mean_cosine_similarity": float(cosine.mean()) if cosine.size else 0.0,
    }


def analyze_normalization(vectors: np.ndarray, unit_tolerance: float = 0.01) -> NormalizationSpec:
    """Measures vector norms on a representative sample (V2 §28). Never silently assume
    normalized: `status` only becomes "detected" once >=95% of the sample is within
    `unit_tolerance` of unit norm; otherwise it's explicitly "not_normalized", and an
    empty sample stays "unknown" rather than defaulting either way."""
    if vectors.size == 0:
        return NormalizationSpec(status="unknown")
    norms = np.linalg.norm(vectors, axis=1)
    mean_norm = float(norms.mean())
    std_norm = float(norms.std())
    confidence = float(np.mean(np.abs(norms - 1.0) < unit_tolerance))
    if confidence >= 0.95:
        return NormalizationSpec(
            status="detected",
            method="l2",
            confidence=confidence,
            mean_norm=mean_norm,
            std_norm=std_norm,
        )
    return NormalizationSpec(
        status="not_normalized", confidence=confidence, mean_norm=mean_norm, std_norm=std_norm
    )


def resolve_validation_mode(
    golden_queries: list[dict] | None,
    llm_synthetic_available: bool,
    representative_documents_available: bool,
) -> str:
    """Golden queries are weighted more heavily than synthetic (V2 §24); when unavailable,
    synthetic generation is attempted only if an LLM is actually configured (this sandbox
    has no Anthropic key, so that path degrades gracefully rather than hard-failing —
    "unavailable" means the VERIFY state records retrieval-equivalence as UNMEASURED, not
    that the migration fails)."""
    if golden_queries:
        return "golden"
    if llm_synthetic_available and representative_documents_available:
        return "synthetic"
    return "unavailable"
