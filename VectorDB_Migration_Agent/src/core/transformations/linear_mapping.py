"""Ridge and Orthogonal Procrustes calibration-pair mappings — the closed-form members of
the "learned mapping" family the Isotrieve project (github.com/krish1925/isotrieve) is
named after in the migration spec. Unlike the neural trained-projection family
(core/transformations/trained_projection.py: Matryoshka-Adaptor/LearnedProjection/
RetrievalAwareProjection), Ridge regression and Orthogonal Procrustes have closed-form
solutions — no training loop, no gradient descent — so these are REAL implementations,
not honest stubs, same tier as pca.py/random_projection.py.

The mapping is learned from a small CALIBRATION set, not the whole corpus (V2 §17): a
sample of source vectors is paired with fresh target-space embeddings of the SAME
documents (re-embedded via the existing OpenAI-compatible endpoint,
core/transformations/reembedding.py:embed_texts), a linear map is fit on that pair, and
the fitted map is then applied to every vector in the migration — including ones that were
never re-embedded. This is the whole point: avoid re-embedding the full corpus while still
landing in the target's embedding space, not merely projecting to a lower dimension the
source model's own geometry defines (contrast with pca.py/random_projection.py, which need
no target-space signal at all).

Calibration pairing assumption: `context.documents[i]` must correspond to the same record
as the i-th row of the `sample_vectors` array passed to `prepare()` — the same assumption
`tools/execution_tools.py` already relies on when building `documents` for
ReEmbeddingTransformer from the same `records` list used for `vectors`.

Ridge (`RidgeMappingTransformer`) fits a general (possibly non-square, possibly
dimension-changing) affine map `target ~= source @ W + b` via `sklearn.linear_model.Ridge`
— applicable for ANY source_dim/target_dim pair, including genuine dimension reduction
combined with a model change.

Orthogonal Procrustes (`OrthogonalProcrustesTransformer`) fits the classical
translation-corrected orthogonal Procrustes solution (`scipy.linalg.orthogonal_procrustes`
on mean-centered calibration pairs) — mathematically requires source_dim == target_dim, so
it only applies to V2 §20's "same dimension, different embedding model" case, never to
dimension-reducing migrations.

Procrustes-Diag (`ProcrustesDiagMappingTransformer`) restricts the orthogonal transform to
a diagonal matrix with entries in {-1, +1} — an axis-aligned sign-flip only, no rotation
mixing across dimensions. Still closed-form (per-axis sign of the calibration dot
product), still requires source_dim == target_dim like full Procrustes, but is far less
expressive — useful when the two embedding spaces are already axis-aligned (e.g. two
checkpoints of the same model family) and full rotation would overfit a small calibration
set.

Low-Rank Affine (`LowRankAffineMappingTransformer`) fits the same Ridge affine map, then
truncates it to a low-rank approximation via SVD — bounding the effective number of free
parameters well below `source_dim * target_dim`, which is the point when calibration data
is limited relative to the dimensionality (V2 §17's "limited calibration data" case).
Still closed-form: no iterative optimization, just a Ridge fit followed by one SVD.

Not implemented here, deliberately: Residual-MLP mapping (`core/transformations/
trained_projection.py`-style honest stub, added below) needs a real non-linear training
loop and PyTorch, which this build does not add as a dependency (2026-09-01) — same
honesty standard as the trained_projection.py family, correct interface + NotImplementedError,
not a fake implementation.
"""

from __future__ import annotations

import httpx
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge

from core.transformations.base import (
    ApplicabilityInputs,
    ApplicabilityResult,
    RepresentationTransformer,
)
from core.transformations.reembedding import REQUEST_TIMEOUT, embed_texts

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_MIN_CALIBRATION_SAMPLES = 200
_DEFAULT_CALIBRATION_SIZE = 500


class _CalibrationMappingBase(RepresentationTransformer):
    MIN_CALIBRATION_SAMPLES = _MIN_CALIBRATION_SAMPLES

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        base_url: str = _DEFAULT_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        calibration_size: int = _DEFAULT_CALIBRATION_SIZE,
        dimensions: int | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._transport = transport
        self._calibration_size = calibration_size
        self._dimensions = dimensions
        self._client: httpx.AsyncClient | None = None
        self._documents: list[str] = []
        self._calibration_source: np.ndarray | None = None
        self._fitted = False

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(f"{self.strategy} needs an api_key before transform()")
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
                f"{self.strategy} needs calibration pairs generated by re-embedding source "
                f"documents — documents_available capability is not TRUE and no documents "
                f"were supplied in the transform context",
            )
        return ApplicabilityResult(
            True,
            f"possible in principle (needs >= {self.MIN_CALIBRATION_SAMPLES} calibration "
            f"documents matched 1:1 with the sampled source vectors); fits a closed-form "
            f"mapping from a small re-embedded calibration subset, not the full corpus",
        )

    def prepare(self, sample_vectors: np.ndarray, context) -> None:
        self._documents = context.documents or []
        self._calibration_source = np.asarray(sample_vectors, dtype=np.float32)
        self._fitted = False

    async def _fit_if_needed(self) -> None:
        if self._fitted:
            return
        if len(self._documents) < self.MIN_CALIBRATION_SAMPLES:
            raise ValueError(
                f"{self.strategy} requires >= {self.MIN_CALIBRATION_SAMPLES} calibration "
                f"documents (paired 1:1 with sampled source vectors), got "
                f"{len(self._documents)} — see core/transformations/linear_mapping.py"
            )
        cal_n = min(self._calibration_size, len(self._documents), len(self._calibration_source))
        cal_docs = self._documents[:cal_n]
        cal_source = self._calibration_source[:cal_n]
        client = self._ensure_client()
        cal_target = await embed_texts(client, self._model, cal_docs, dimensions=self._dimensions)
        self._fit(cal_source, cal_target)
        self._fitted = True

    def _fit(self, cal_source: np.ndarray, cal_target: np.ndarray) -> None:
        raise NotImplementedError

    def _apply(self, vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    async def ensure_fitted(self) -> None:
        """Public trigger for the fit-once-and-persist path `tools/execution_tools.py`
        needs: MIGRATE-time execution fits this exactly once against the benchmark
        calibration sample (like PCA/RandomProjection), serializes the fitted result via
        `fitted_params`, and every subsequent batch reloads it via `from_fitted` instead
        of re-fitting (and re-calling the embeddings API) per batch."""
        await self._fit_if_needed()

    async def transform(self, vectors: np.ndarray) -> np.ndarray:
        await self._fit_if_needed()
        return self._apply(np.asarray(vectors, dtype=np.float32))

    def provenance(self) -> dict:
        return {
            "strategy": self.strategy,
            "parameters": {
                "model": self._model,
                "calibration_pairs_used": min(self._calibration_size, len(self._documents)),
            },
        }

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


class RidgeMappingTransformer(_CalibrationMappingBase):
    """Ridge regression affine map, fit on a calibration subset — applicable for any
    source_dim/target_dim pair, dimension-changing or not."""

    strategy = "ridge_mapping"

    def __init__(self, *args, alpha: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._coef: np.ndarray | None = None  # (target_dim, source_dim), sklearn's coef_ layout
        self._intercept: np.ndarray | None = None

    def _fit(self, cal_source: np.ndarray, cal_target: np.ndarray) -> None:
        model = Ridge(alpha=self._alpha)
        model.fit(cal_source, cal_target)
        # Pulled out into plain arrays (not the sklearn estimator itself) so fitted_params
        # round-trips through JSON the same way PCA/RandomProjection's do.
        self._coef = model.coef_.astype(np.float32)
        self._intercept = model.intercept_.astype(np.float32)

    def _apply(self, vectors: np.ndarray) -> np.ndarray:
        assert self._coef is not None and self._intercept is not None
        return (vectors @ self._coef.T + self._intercept).astype(np.float32)

    @property
    def fitted_params(self) -> dict:
        if self._coef is None:
            raise RuntimeError("RidgeMappingTransformer is not fitted yet")
        return {"coef": self._coef.tolist(), "intercept": self._intercept.tolist()}

    @classmethod
    def from_fitted(cls, params: dict, **kwargs) -> RidgeMappingTransformer:
        obj = cls(**kwargs)
        obj._coef = np.array(params["coef"], dtype=np.float32)
        obj._intercept = np.array(params["intercept"], dtype=np.float32)
        obj._fitted = True
        return obj

    def provenance(self) -> dict:
        record = super().provenance()
        record["parameters"]["alpha"] = self._alpha
        return record


class OrthogonalProcrustesTransformer(_CalibrationMappingBase):
    """Translation-corrected orthogonal Procrustes mapping — mean-center both calibration
    sets, solve for the orthogonal rotation via scipy.linalg.orthogonal_procrustes, then
    apply rotation + translation. Only mathematically defined when source_dim ==
    target_dim (V2 §20's same-dimension/different-embedding-model case)."""

    strategy = "procrustes_mapping"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rotation: np.ndarray | None = None
        self._source_mean: np.ndarray | None = None
        self._target_mean: np.ndarray | None = None

    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        if inputs.source_dim is None or inputs.target_dim is None:
            return ApplicabilityResult(False, "source or target dimension not discovered")
        if inputs.source_dim != inputs.target_dim:
            return ApplicabilityResult(
                False,
                "orthogonal procrustes requires source_dim == target_dim (it solves for a "
                "rotation, not a dimension-changing projection) — use ridge_mapping instead "
                "when dimensions differ",
            )
        return super().can_apply(inputs)

    def _fit(self, cal_source: np.ndarray, cal_target: np.ndarray) -> None:
        self._source_mean = cal_source.mean(axis=0)
        self._target_mean = cal_target.mean(axis=0)
        centered_source = cal_source - self._source_mean
        centered_target = cal_target - self._target_mean
        rotation, _ = orthogonal_procrustes(centered_source, centered_target)
        self._rotation = rotation

    def _apply(self, vectors: np.ndarray) -> np.ndarray:
        assert self._rotation is not None
        return (
            (vectors - self._source_mean) @ self._rotation + self._target_mean
        ).astype(np.float32)

    @property
    def fitted_params(self) -> dict:
        if self._rotation is None:
            raise RuntimeError("OrthogonalProcrustesTransformer is not fitted yet")
        return {
            "rotation": self._rotation.tolist(),
            "source_mean": self._source_mean.tolist(),
            "target_mean": self._target_mean.tolist(),
        }

    @classmethod
    def from_fitted(cls, params: dict, **kwargs) -> OrthogonalProcrustesTransformer:
        obj = cls(**kwargs)
        obj._rotation = np.array(params["rotation"], dtype=np.float32)
        obj._source_mean = np.array(params["source_mean"], dtype=np.float32)
        obj._target_mean = np.array(params["target_mean"], dtype=np.float32)
        obj._fitted = True
        return obj


class ProcrustesDiagMappingTransformer(_CalibrationMappingBase):
    """Diagonal-only orthogonal Procrustes: restrict the rotation to a diagonal matrix
    with entries in {-1, +1} (per-axis sign flip, no cross-axis mixing). Minimizing
    ||centered_source @ D - centered_target||_F^2 over diagonal D decouples per column —
    each entry is independently d_j = sign(dot(centered_source[:, j], centered_target[:, j])),
    the closed-form solution. Requires source_dim == target_dim, same as full Procrustes."""

    strategy = "procrustes_diag_mapping"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._diag: np.ndarray | None = None
        self._source_mean: np.ndarray | None = None
        self._target_mean: np.ndarray | None = None

    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        if inputs.source_dim is None or inputs.target_dim is None:
            return ApplicabilityResult(False, "source or target dimension not discovered")
        if inputs.source_dim != inputs.target_dim:
            return ApplicabilityResult(
                False,
                "procrustes_diag_mapping requires source_dim == target_dim (axis-aligned "
                "sign-flip only, not a dimension-changing projection) — use ridge_mapping "
                "or low_rank_affine_mapping instead when dimensions differ",
            )
        return super().can_apply(inputs)

    def _fit(self, cal_source: np.ndarray, cal_target: np.ndarray) -> None:
        self._source_mean = cal_source.mean(axis=0)
        self._target_mean = cal_target.mean(axis=0)
        centered_source = cal_source - self._source_mean
        centered_target = cal_target - self._target_mean
        dots = (centered_source * centered_target).sum(axis=0)
        self._diag = np.where(dots >= 0, 1.0, -1.0).astype(np.float32)

    def _apply(self, vectors: np.ndarray) -> np.ndarray:
        assert self._diag is not None
        return (
            (vectors - self._source_mean) * self._diag + self._target_mean
        ).astype(np.float32)

    @property
    def fitted_params(self) -> dict:
        if self._diag is None:
            raise RuntimeError("ProcrustesDiagMappingTransformer is not fitted yet")
        return {
            "diag": self._diag.tolist(),
            "source_mean": self._source_mean.tolist(),
            "target_mean": self._target_mean.tolist(),
        }

    @classmethod
    def from_fitted(cls, params: dict, **kwargs) -> ProcrustesDiagMappingTransformer:
        obj = cls(**kwargs)
        obj._diag = np.array(params["diag"], dtype=np.float32)
        obj._source_mean = np.array(params["source_mean"], dtype=np.float32)
        obj._target_mean = np.array(params["target_mean"], dtype=np.float32)
        obj._fitted = True
        return obj


class LowRankAffineMappingTransformer(_CalibrationMappingBase):
    """Ridge affine map, then truncated via SVD to a low-rank approximation — bounds the
    effective free-parameter count well below source_dim * target_dim, which is the point
    when calibration data is limited relative to dimensionality (a full-rank Ridge fit
    with too few calibration pairs relative to the parameter count risks instability even
    with L2 regularization). Still closed-form: one Ridge fit + one SVD, no iterative
    optimization. Applicable for any source_dim/target_dim pair, same as ridge_mapping."""

    strategy = "low_rank_affine_mapping"

    def __init__(self, *args, alpha: float = 1.0, rank: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._rank = rank
        self._coef: np.ndarray | None = None  # (target_dim, source_dim), low-rank
        self._intercept: np.ndarray | None = None

    def _resolve_rank(self, cal_n: int, source_dim: int, target_dim: int) -> int:
        if self._rank is not None:
            return max(1, min(self._rank, source_dim, target_dim))
        # Heuristic default: keep the parameter count comfortably below the number of
        # calibration pairs actually available, capped by the smaller dimension.
        return max(1, min(source_dim, target_dim, cal_n // 5))

    def _fit(self, cal_source: np.ndarray, cal_target: np.ndarray) -> None:
        model = Ridge(alpha=self._alpha)
        model.fit(cal_source, cal_target)
        full_coef = model.coef_.astype(np.float32)  # (target_dim, source_dim)
        rank = self._resolve_rank(len(cal_source), cal_source.shape[1], cal_target.shape[1])
        u, s, vt = np.linalg.svd(full_coef, full_matrices=False)
        low_rank_coef = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        self._coef = low_rank_coef.astype(np.float32)
        self._intercept = model.intercept_.astype(np.float32)

    def _apply(self, vectors: np.ndarray) -> np.ndarray:
        assert self._coef is not None and self._intercept is not None
        return (vectors @ self._coef.T + self._intercept).astype(np.float32)

    @property
    def fitted_params(self) -> dict:
        if self._coef is None:
            raise RuntimeError("LowRankAffineMappingTransformer is not fitted yet")
        return {"coef": self._coef.tolist(), "intercept": self._intercept.tolist()}

    @classmethod
    def from_fitted(cls, params: dict, **kwargs) -> LowRankAffineMappingTransformer:
        obj = cls(**kwargs)
        obj._coef = np.array(params["coef"], dtype=np.float32)
        obj._intercept = np.array(params["intercept"], dtype=np.float32)
        obj._fitted = True
        return obj

    def provenance(self) -> dict:
        record = super().provenance()
        record["parameters"]["alpha"] = self._alpha
        record["parameters"]["rank"] = None if self._coef is None else int(
            np.linalg.matrix_rank(self._coef)
        )
        return record


class ResidualMLPMappingTransformer(_CalibrationMappingBase):
    """Non-linear calibration mapping: a residual MLP on top of a linear baseline,
    fit on the same calibration pairs as RidgeMapping. Honest extension-point stub
    (2026-09-01, per explicit operator decision): a real implementation needs PyTorch (a
    genuine training loop — held-out validation split, fixed seed, early stopping — not
    something to fake with placeholder code), and this build deliberately does not add
    torch as a dependency. Correct interface + NotImplementedError, not a fake
    implementation that would silently produce an untrained (garbage) mapping."""

    strategy = "residual_mlp_mapping"
    is_stub = True

    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        result = super().can_apply(inputs)
        if not result.possible:
            return result
        return ApplicabilityResult(
            True, f"{result.reason}; NOT IMPLEMENTED in this build (requires PyTorch, see docstring)"
        )

    async def _fit_if_needed(self) -> None:
        # Overrides the base class's version (which would call the real embeddings API
        # for calibration targets before ever reaching _fit) — no network call is needed
        # to know this strategy isn't implemented.
        raise NotImplementedError(
            "residual_mlp_mapping requires a real PyTorch training loop (residual MLP "
            "over a linear baseline, held-out validation split, fixed seed, early "
            "stopping) — this build deliberately does not add a torch dependency. Honest "
            "extension point, not a stub that silently produces an untrained mapping."
        )

    def _fit(self, cal_source: np.ndarray, cal_target: np.ndarray) -> None:
        raise NotImplementedError("residual_mlp_mapping._fit_if_needed already raises")

    def _apply(self, vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError("residual_mlp_mapping.prepare()/_fit() was never able to succeed")

    @property
    def fitted_params(self) -> dict:
        raise RuntimeError("residual_mlp_mapping is not fitted yet — see docstring")

    @classmethod
    def from_fitted(cls, params: dict, **kwargs) -> ResidualMLPMappingTransformer:
        raise NotImplementedError("residual_mlp_mapping has no real fitted representation yet")
