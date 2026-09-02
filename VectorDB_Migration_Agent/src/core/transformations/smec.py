"""SMEC — Semantic-preserving Matryoshka Embedding Compression, as described in the third
spec document's additions. Structurally it looks like another dimensionality-reduction
strategy, but it is fundamentally different from every other transformer in this package:
PCA/RandomProjection/MRL/the trained-projection family all operate POST-HOC on vectors that
already exist. SMEC is a training-time technique — a compression objective baked into the
embedding model itself (jointly trained alongside the encoder, Matryoshka-style nested
loss over the compressed representation), not a function you can apply to embeddings that
were already produced by a frozen, un-cooperating source model.

That means SMEC can never legitimately run as a migration-time transform: there is no
sample of (source_vector -> smec_vector) pairs to learn from, because the "smec_vector"
only exists if the source encoder was SMEC-trained from the start. A migration whose source
index holds ordinary (non-SMEC-trained) embeddings cannot retrofit SMEC onto them — the only
honest path is re-embedding the original documents from scratch with a SMEC-trained model,
which is a re-embedding decision (see reembedding.py), not a representation transform.

So `can_apply()` always returns False here, with that reasoning — unlike the trained
projection family (matryoshka_adaptor/learned_projection/retrieval_aware_projection),
which are at least *possible in principle* given enough paired samples. This class exists
so the strategy has a correct, documented place in the codebase rather than being silently
absent, not because a migration can select it today.
"""

from __future__ import annotations

import numpy as np

from core.transformations.base import (
    ApplicabilityInputs,
    ApplicabilityResult,
    RepresentationTransformer,
)


class SMECTransformer(RepresentationTransformer):
    strategy = "smec"
    is_stub = True

    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        return ApplicabilityResult(
            False,
            "smec is a training-time compression objective jointly learned with the "
            "source encoder, not a post-hoc transform on already-produced embeddings — "
            "it cannot be applied to a source index whose vectors were not produced by a "
            "SMEC-trained model. See core/transformations/smec.py module docstring; the "
            "only honest path to a SMEC target representation is re-embedding the "
            "original documents with a SMEC-trained model (strategy=re_embedding).",
        )

    def prepare(self, sample_vectors: np.ndarray, context) -> None:
        raise NotImplementedError(
            "smec cannot be prepared as a migration-time transform: it requires the "
            "source encoder itself to have been trained with the SMEC compression "
            "objective. There is no sample of (existing source vector -> smec vector) "
            "pairs to fit against. See core/transformations/smec.py module docstring."
        )

    async def transform(self, vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError("smec.prepare() can never succeed — see its message")
