"""vec2vec — unsupervised universal embedding translation (V3 §3), added specifically for
the worst-case cell V2 §55 admits defeat on: unknown embedding model + no documents +
no historical queries + dimension mismatch. Built on the Platonic Representation
Hypothesis (independently-trained encoders converge toward a shared latent geometry);
GAN-style adversarial training between relative representations, no paired data required.
Reference: Jha, Zhang et al., "Harnessing the Universal Geometry of Embeddings",
NeurIPS 2025, arXiv:2505.12540.

`can_apply` implements the real gating logic from V3 §3's decision tree (this candidate
should only ever appear when every semantic alternative is genuinely exhausted — otherwise
you'd fall straight to PCA/random projection with no attempt at a semantics-aware option).
The adversarial training itself is NOT implemented: it is a multi-day GAN training run
requiring a large unpaired embedding corpus in both spaces, well beyond an honest stub.

MANDATORY DISCLOSURE (V3 §3): the same paper shows translated embeddings can leak enough
signal to reconstruct sensitive attributes of the source documents (their own examples:
disease classification from medical embeddings, content inference from corporate email
embeddings). If this transformer is ever completed and selected, the audit report MUST
surface DISCLOSURE_TEXT and the trained translator must be handled with credential-tier
custody (V2 §43) — see core/models/provenance.py's Vec2VecProvenance.
"""

from __future__ import annotations

import numpy as np

from core.transformations.base import (
    ApplicabilityInputs,
    ApplicabilityResult,
    RepresentationTransformer,
)

DISCLOSURE_TEXT = (
    "vec2vec-style unsupervised embedding translation carries a published embedding-"
    "inversion / privacy-leakage risk (Jha, Zhang et al., NeurIPS 2025, arXiv:2505.12540). "
    "Translated embeddings can leak sensitive attributes of the source documents. Treat "
    "the trained translator as credential-tier sensitive material."
)


class VecToVecTransformer(RepresentationTransformer):
    strategy = "vec2vec"
    is_stub = True

    def can_apply(self, inputs: ApplicabilityInputs) -> ApplicabilityResult:
        if inputs.source_dim is None or inputs.target_dim is None:
            return ApplicabilityResult(False, "source or target dimension not discovered")

        model_unknown = inputs.embedding.model in ("unknown", "", None)
        documents_unavailable = (
            not inputs.capabilities.data.retrieve_documents.is_true and not inputs.context.documents
        )
        historical_queries_available = bool(
            inputs.context.extra.get("historical_queries_available")
        )

        if not model_unknown:
            return ApplicabilityResult(
                False, "source embedding model is known — prefer MRL/re-embedding"
            )
        if not documents_unavailable:
            return ApplicabilityResult(False, "documents are available — prefer re-embedding")
        if historical_queries_available:
            return ApplicabilityResult(
                False, "historical queries are available — prefer retrieval_aware_projection"
            )
        return ApplicabilityResult(
            True,
            "worst case confirmed (unknown model + no documents + no historical queries): "
            "vec2vec is the only candidate that could produce a semantics-aware target vector; "
            "NOT IMPLEMENTED in this build (see module docstring) — falls through to "
            "PCA/random_projection until a real adversarial trainer exists",
        )

    def prepare(self, sample_vectors: np.ndarray, context) -> None:
        raise NotImplementedError(
            "vec2vec requires unpaired adversarial (GAN-style) training over relative "
            "representations in both embedding spaces — a multi-day training job, not "
            "something to fake with placeholder code. " + DISCLOSURE_TEXT
        )

    async def transform(self, vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError("vec2vec.prepare() was never able to succeed")
