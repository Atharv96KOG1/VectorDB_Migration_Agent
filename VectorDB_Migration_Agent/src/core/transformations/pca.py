"""PCA (V2 §19) — baseline statistical compression. Optimizes reconstruction
variance, NOT retrieval quality; must never be treated as equivalent to a
retrieval-preserving method, and must always go through the benchmark gate before
selection (V2 §19, §22).

Fit once (on the benchmark sample), then reused via `from_fitted` for the actual full
MIGRATE scan — PCA must not be re-fit per batch, or every batch would land in a
different projected space and the target collection would be numerically incoherent.
`fitted_params`/`from_fitted` round-trip the fit as plain lists so it can live in the
JSON checkpoint between the BENCHMARK and MIGRATE states.
"""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.decomposition import PCA

from core.transformations.base import (
    ApplicabilityInputs,
    ApplicabilityResult,
    RepresentationTransformer,
)


class PCATransformer(RepresentationTransformer):
    strategy = "pca"

    def __init__(self, target_dimension: int, seed: int = 42) -> None:
        self._target_dimension = target_dimension
        self._seed = seed
        self._components: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._sample_size: int = 0

    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        if inputs.source_dim is None or inputs.target_dim is None:
            return ApplicabilityResult(False, "source or target dimension not discovered")
        if inputs.target_dim >= inputs.source_dim:
            return ApplicabilityResult(False, "PCA reduces dimensionality; target >= source")
        return ApplicabilityResult(True, "statistical compression, always fittable given a sample")

    def prepare(self, sample_vectors: np.ndarray, context) -> None:
        self._sample_size = sample_vectors.shape[0]
        pca = PCA(n_components=self._target_dimension, random_state=self._seed)
        pca.fit(sample_vectors.astype(np.float32))
        self._components = pca.components_.astype(np.float32)
        self._mean = pca.mean_.astype(np.float32)

    async def transform(self, vectors: np.ndarray) -> np.ndarray:
        if self._components is None or self._mean is None:
            raise RuntimeError("PCATransformer.prepare() must be called before transform()")
        centered = vectors.astype(np.float32) - self._mean
        return (centered @ self._components.T).astype(np.float32)

    def inverse_transform(self, transformed: np.ndarray) -> np.ndarray:
        """Reconstructs an approximation of the original source-space vectors from the
        reduced ones — used only for the reconstruction-error diagnostic
        (core/validation/engine.py:reconstruction_error), never for retrieval quality
        (V2 §19/§40: reconstruction fidelity is not retrieval fidelity)."""
        if self._components is None or self._mean is None:
            raise RuntimeError("PCATransformer.prepare() must be called before inverse_transform()")
        return (transformed.astype(np.float32) @ self._components + self._mean).astype(np.float32)

    @property
    def fitted_params(self) -> dict:
        if self._components is None or self._mean is None:
            raise RuntimeError("PCATransformer is not fitted yet")
        return {"components": self._components.tolist(), "mean": self._mean.tolist()}

    @classmethod
    def from_fitted(cls, params: dict, target_dimension: int, seed: int = 42) -> PCATransformer:
        obj = cls(target_dimension, seed=seed)
        obj._components = np.array(params["components"], dtype=np.float32)
        obj._mean = np.array(params["mean"], dtype=np.float32)
        return obj

    def provenance(self) -> dict:
        checksum = None
        if self._components is not None:
            checksum = hashlib.sha256(self._components.tobytes()).hexdigest()
        return {
            "strategy": self.strategy,
            "output_dimension": self._target_dimension,
            "parameters": {"sample_size": self._sample_size, "seed": self._seed},
            "artifact_checksum": checksum,
        }
