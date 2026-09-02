"""Matryoshka-Adaptor (V2 §14), Learned Projection (§15), and Retrieval-Aware Projection
(§16) — three genuinely distinct training objectives that share one contract, so they
share one base class instead of three near-duplicate files.

These are honest extension-point stubs, not fake implementations. A trained projection
that "runs" on whatever sample happens to be available in a migration sandbox and reports
a benchmark number is worse than no implementation: it would produce a plausible-looking
Recall@10 that isn't backed by a methodologically sound training run (see the disjoint
train/val requirement below — training on the same sample you'd then benchmark against
makes the quality gate meaningless), and a migration operator could ship it. `prepare()`
raises NotImplementedError with the exact contract a real implementation must satisfy,
so this stays a correct, buildable extension point rather than a vague TODO.

Contract any real implementation must satisfy:
  - MIN_PAIRED_SAMPLES paired (source_vector, target_vector | target_signal) examples,
    drawn from the actual migration's data, not a generic public dataset.
  - A held-out validation split, disjoint from both the training set AND from whatever
    sample the benchmark engine (core/benchmarking/engine.py) later evaluates Recall@K
    on — training and grading on the same data invalidates the quality gate.
  - A fixed random seed, recorded in provenance() for reproducibility (V2 §53).
  - The LOSS_OBJECTIVE this subclass declares (see each subclass docstring) — using the
    wrong loss silently turns e.g. a "retrieval-aware" projection into a plain
    reconstruction-error projection with no better guarantee than PCA.
"""

from __future__ import annotations

import numpy as np

from core.transformations.base import (
    ApplicabilityInputs,
    ApplicabilityResult,
    RepresentationTransformer,
)


class _TrainedProjectionBase(RepresentationTransformer):
    is_stub = True
    MIN_PAIRED_SAMPLES: int = 5_000
    LOSS_OBJECTIVE: str = "unspecified"

    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        if inputs.source_dim is None or inputs.target_dim is None:
            return ApplicabilityResult(False, "source or target dimension not discovered")
        if inputs.target_dim >= inputs.source_dim:
            return ApplicabilityResult(
                False, f"{self.strategy} reduces dimensionality; target >= source"
            )
        return ApplicabilityResult(
            True,
            f"possible in principle (needs >= {self.MIN_PAIRED_SAMPLES} paired samples + disjoint "
            f"val split + {self.LOSS_OBJECTIVE} loss); NOT IMPLEMENTED in this build, see docstring",
        )

    def prepare(self, sample_vectors: np.ndarray, context) -> None:
        raise NotImplementedError(
            f"{self.strategy} requires a real training pipeline: >= {self.MIN_PAIRED_SAMPLES} paired "
            f"samples, a validation split disjoint from the benchmark sample, a fixed seed, and a "
            f"{self.LOSS_OBJECTIVE} loss objective. This is an honest extension point, not a stub "
            f"that silently produces an untrained (garbage) projection — implement and wire a real "
            f"trainer before selecting this strategy in production."
        )

    async def transform(self, vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self.strategy}.prepare() was never able to succeed — see its message"
        )


class MatryoshkaAdaptorTransformer(_TrainedProjectionBase):
    """V2 §14: an adapter trained on top of a *fixed* embedding model to produce a
    smaller representation, useful when the source model can't be replaced (e.g. no
    budget to re-embed) but also isn't natively MRL-capable at the target dimension."""

    strategy = "matryoshka_adaptor"
    LOSS_OBJECTIVE = "reconstruction + ranking-preservation (Matryoshka-style nested loss)"


class LearnedProjectionTransformer(_TrainedProjectionBase):
    """V2 §15: a neural projection trained with a semantic/retrieval-preservation
    objective instead of PCA's variance-preservation objective."""

    strategy = "learned_projection"
    LOSS_OBJECTIVE = "contrastive or ranking loss over query-document / positive-negative pairs"


class RetrievalAwareProjectionTransformer(_TrainedProjectionBase):
    """V2 §16 — the flagship differentiator: optimizes retrieval AGREEMENT (Recall@K,
    NDCG@K, Top-K overlap between source and target rankings) directly, rather than a
    proxy like reconstruction error."""

    strategy = "retrieval_aware_projection"
    LOSS_OBJECTIVE = (
        "retrieval-agreement loss (maximize Recall@K / NDCG@K / Top-K overlap vs. source rankings)"
    )
    MIN_PAIRED_SAMPLES = 10_000
