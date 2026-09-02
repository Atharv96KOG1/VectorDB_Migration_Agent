"""Random projection (V2 §20) — fast, deterministic given a fixed seed, no fitting stage
beyond generating the matrix. Semantic/retrieval quality is not guaranteed and must be
benchmarked like every other candidate; this is a baseline, not a default winner.

Two distinct regimes, real math, not the same guarantee (2026-09-01, prompted by a real
operator question about information loss on a 1024D -> 1536D migration):
  - Dimension REDUCTION (target_dimension < source_dimension): a plain i.i.d. Gaussian
    matrix, scaled per Johnson-Lindenstrauss. JL is a PROBABILISTIC, APPROXIMATE
    guarantee — exact pairwise-distance preservation is mathematically impossible when
    compressing into fewer dimensions (you cannot fit N mutually-orthonormal directions
    into fewer than N dimensions), so this is genuinely the best any linear method can do
    here, real information loss included.
  - Dimension EXPANSION OR EQUAL (target_dimension >= source_dimension): the matrix is
    instead constructed via QR decomposition to have exactly orthonormal ROWS
    (M @ M.T == I). An orthonormal-row matrix preserves every dot product EXACTLY, not
    approximately: (a @ M) . (b @ M) == a . b for any a, b — this is achievable precisely
    because expansion doesn't force any compression, so there is no theoretical reason to
    settle for JL's probabilistic approximation here.
"""

from __future__ import annotations

import hashlib

import numpy as np

from core.transformations.base import (
    ApplicabilityInputs,
    ApplicabilityResult,
    RepresentationTransformer,
)


class RandomProjectionTransformer(RepresentationTransformer):
    strategy = "random_projection"

    def __init__(self, target_dimension: int, seed: int = 42) -> None:
        self._target_dimension = target_dimension
        self._seed = seed
        self._source_dimension: int | None = None
        self._matrix: np.ndarray | None = None

    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        if inputs.source_dim is None or inputs.target_dim is None:
            return ApplicabilityResult(False, "source or target dimension not discovered")
        return ApplicabilityResult(
            True, "always applicable given known dimensions; benchmark before selection"
        )

    def prepare(self, sample_vectors: np.ndarray, context) -> None:
        self._source_dimension = sample_vectors.shape[1]
        rng = np.random.default_rng(self._seed)
        if self._target_dimension >= self._source_dimension:
            # Expansion (or equal): construct a matrix with exactly orthonormal rows via
            # QR, so every dot product is preserved EXACTLY, not just in expectation — see
            # module docstring. Economy QR of the transpose gives a (target_dim,
            # source_dim) matrix with orthonormal columns; its transpose has orthonormal
            # rows (M @ M.T == I_source_dim).
            gaussian = rng.standard_normal((self._target_dimension, self._source_dimension))
            q, _ = np.linalg.qr(gaussian)
            matrix = q.T
        else:
            # Reduction: no exact isometry is possible (cannot fit source_dimension
            # mutually-orthonormal directions into fewer target dimensions) — plain
            # Gaussian random projection, scaled per Johnson-Lindenstrauss so expected
            # norm is preserved. Approximate by necessity, not by omission.
            matrix = rng.standard_normal((self._source_dimension, self._target_dimension))
            matrix = matrix / np.sqrt(self._target_dimension)
        self._matrix = matrix.astype(np.float32)

    async def transform(self, vectors: np.ndarray) -> np.ndarray:
        if self._matrix is None:
            raise RuntimeError(
                "RandomProjectionTransformer.prepare() must be called before transform()"
            )
        return (vectors.astype(np.float32) @ self._matrix).astype(np.float32)

    @property
    def fitted_params(self) -> dict:
        if self._matrix is None:
            raise RuntimeError("RandomProjectionTransformer is not fitted yet")
        return {"matrix": self._matrix.tolist()}

    @classmethod
    def from_fitted(
        cls, params: dict, target_dimension: int, seed: int = 42
    ) -> RandomProjectionTransformer:
        obj = cls(target_dimension, seed=seed)
        obj._matrix = np.array(params["matrix"], dtype=np.float32)
        obj._source_dimension = obj._matrix.shape[0]
        return obj

    def provenance(self) -> dict:
        checksum = None
        if self._matrix is not None:
            checksum = hashlib.sha256(self._matrix.tobytes()).hexdigest()
        return {
            "strategy": self.strategy,
            "input_dimension": self._source_dimension,
            "output_dimension": self._target_dimension,
            "parameters": {"seed": self._seed},
            "artifact_checksum": checksum,
        }
