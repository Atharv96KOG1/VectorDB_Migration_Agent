# Architecture — Universal Vector DB Migration Agent

Design of record for this build. Merges `UNIVERSAL_VECTOR_DB_MIGRATION_AGENT_V2.md`
("V2") with `V2_Review_and_V3_Additions_1.md` ("V3") — every V3 correctness fix and
every V3 addition is accounted for below, either implemented or explicitly deferred with
a reason. Nothing was silently dropped.

## Core idea

Treat migration as a **retrieval-preservation problem**, not a data-copy problem: never
default to a transformation strategy (PCA, etc.) without benchmarking it against a real
quality gate, and always disclose honestly when a capability is unknown rather than
assuming the safe or unsafe answer.

## Infra hardening pass (2026-08-26)

After the initial build, a second pass specifically hardened the Pinecone↔Qdrant path
(scope chosen deliberately: perfect one pair before generalizing to others):

1. **Pinecone API correctness, verified against live docs, not recalled from training
   data.** Every header, endpoint path, and JSON field name in
   `core/adapters/pinecone_adapter.py` was checked against docs.pinecone.io directly.
   Two real issues found and fixed: the pinned `X-Pinecone-Api-Version` was 2 years stale
   (now a constructor parameter defaulting to a current version, not a hardcoded
   constant, since Pinecone's own docs say each version has only a ~12-month support
   window); and `GET /vectors/list` (used by `scan_vectors`/`sample_vectors`) is
   **serverless-only** — pod-based indexes reject it, which the adapter didn't
   previously detect. `get_resource_info` now records the index's `spec` type and
   `scan_vectors` fails fast with a clear message on a pod index instead of surfacing
   Pinecone's raw HTTP error. `discover_capabilities`'s `list_vectors`/`scan_vectors`/
   `resumable_scan` also moved from a blanket `TRUE` to `UNKNOWN` at the provider level,
   since the true answer is per-index, not provider-wide — the tri-state was being
   applied inconsistently before this fix.
2. **Real ANN-backed benchmarking, not just brute-force, for Qdrant targets.**
   `tools/benchmark_tools.py` now stands up an actual ephemeral Qdrant collection per
   candidate strategy (`core/adapters/qdrant_adapter.py:create_collection`/
   `delete_collection`), upserts the transformed sample into it, and queries it for real
   — V2 §22's "temporary target collection" literally, not a numpy stand-in. Brute-force
   remains the fallback for Pinecone targets (real ephemeral Pinecone indexes are slow
   and not free to spin up per candidate). All candidates' temp-collection lifecycles
   share ONE connection, serialized by an `asyncio.Lock` — Qdrant's local/on-disk mode
   locks its storage path per open client, so one-client-per-candidate would deadlock the
   moment two candidates ran concurrently against a local target.
3. **Real concurrency**, not just async syntax: source ground-truth Top-K fetching and
   candidate benchmarking both run concurrently (bounded semaphores,
   `tools/benchmark_tools.py`); golden-query verification runs concurrently
   (`tools/validation_tools.py`); `tools/execution_tools.py:migrate_batch` now
   double-buffers — while one batch is being transformed and written, the next batch's
   source scan is already in flight — cutting both wall-clock time and the number of
   Temporal activity round-trips a large migration needs
   (`tests/integration/test_qdrant_roundtrip.py`'s pipelining test proves multiple pages
   land within a single call, with no loss or duplication).

### First live run against a real Pinecone account (2026-08-26)

The user supplied a real Pinecone API key. 48 real sentence embeddings (6 topics, local
`sentence-transformers/all-MiniLM-L6-v2`, 384D) were loaded into a dedicated demo index
(`migration-demo` — the user's existing `rag` index was deliberately left untouched) and
migrated end-to-end, live, through every tool function exactly as `agent.py` calls them:
PCA was correctly selected (0.85 Recall@10, 0.94 NDCG@10, beating random_projection's
0.49) after real benchmark evaluation, and all 48 vectors landed in the target with a
genuine 384D→32D transform applied. This surfaced three real bugs no synthetic test with
UUID ids had ever exercised:

1. **`GET /vectors/list` caps `limit` at 100** — confirmed live (`limit=101` → 400
   `"Limit must be greater than 0 and less than 100"`). `scan_vectors` previously passed
   the caller's `batch_size` straight through unclamped; fixed to internally chain
   100-capped pages so the method's own contract (serve up to `batch_size`) is honored
   regardless of Pinecone's per-request cap. `tests/unit/test_pinecone_adapter.py` covers
   this with a mock that enforces the same cap.
2. **Qdrant point ids must be an unsigned int or a UUID** — Pinecone allows arbitrary
   strings. Every prior test used `uuid.uuid4()` source ids, which happen to already be
   Qdrant-native, so this was invisible until real data with natural ids ("cooking-6")
   hit a real `ValueError: Point id cooking-6 is not a valid UUID`. Fixed properly in
   `core/adapters/qdrant_adapter.py`: any non-native id is mapped through a deterministic
   `uuid5` (same input → same output, so idempotent upsert still holds) and the original
   is stashed in a reserved payload field, recovered on every read path
   (`_to_record`, `query`) — callers of the adapter only ever see the original id.
   `tests/unit/test_qdrant_adapter.py` is a new file covering this round-trip (no
   dedicated Qdrant adapter unit test existed before this).
3. **One candidate's exception used to crash the whole concurrent benchmark run** —
   surfaced when PCA's `sample_size=48 < target_dim` combination (before the target
   dimension was corrected for this small demo) raised inside `asyncio.gather`. Fixed:
   `_evaluate_candidate` now catches any exception (not just `NotImplementedError`) and
   returns an honest failed row (`notes="error during evaluation"`, the real exception
   message in `gate_reasons`) instead of taking every other candidate down with it.

A fourth thing surfaced that is *not* a bug: `id_collision_policy="fail"` correctly
refused an ambiguous double-write when two pagination calls a few hundred milliseconds
apart, against an index still settling from a bulk upsert seconds earlier, returned
briefly overlapping pages (Pinecone's list ordering isn't guaranteed perfectly stable
under eventual consistency). The safety mechanism worked as designed; the fix was
picking `overwrite` for this migration (architecturally correct here — writing our own
idempotent data into a target we control, not guarding against foreign collisions).

### Second live run: real Qdrant Cloud as the target (2026-08-26)

Same Pinecone source, this time migrating into a real hosted Qdrant Cloud cluster
(`url=`/`api_key=`, not local on-disk) — the first time the ephemeral-benchmark-
collection path (`_qdrant_temp_collection_topk`) and the real target write path both ran
against genuine remote cloud infrastructure rather than a local process. Result: PCA
selected again (0.90 Recall@10, 0.976 NDCG@10), 48/48 vectors landed, golden-query
`VERIFY` passed clean.

Getting `VERIFY` to pass clean surfaced one more real gap, fixed properly:

5. **Golden query vectors are in SOURCE space** — querying a dimension-reduced target
   directly with them is exactly the bug this whole exercise was proving out: a 384D
   query against the 32D PCA target 400'd with `"Vector dimension error: expected dim:
   32, got 384"`. Every prior test happened to be either `direct_copy` (same dimension,
   so the bug was invisible) or ran `golden_queries=None`. Fixed in
   `tools/validation_tools.py`: `verify_migration` now reuses the already-fitted
   transformer (`tools.execution_tools._load_execution_transformer` — the same one
   `MIGRATE` used and persisted to the checkpoint) to transform the query vector before
   querying the target, for any strategy that has a vector-level transform. For
   `re_embedding` (transforms document text, not vectors) target-side evaluation is
   skipped with an explicit `note` rather than attempting a doomed request.
   `tests/integration/test_qdrant_roundtrip.py`'s PCA test now covers this directly.

Deliberately not attempted in this pass: a live run against a real Temporal/Aetherion
worker (would require the user's live cloud account and was explicitly declined) and a
live Pinecone account test (none available in this sandbox) — see "Known limitations".

## Runtime: Aetherion SDK (Temporal)

This project is built inside a proprietary internal scaffold, **Aetherion SDK**, which
wraps Temporal.io:

- `@tool()` (`aetherion_sdk`) wraps an async function as a Temporal **activity** — IO,
  randomness, and ML computation are allowed here. All of `core/adapters`,
  `core/transformations`, `core/benchmarking`, etc. are only ever imported from `tools/`.
- `@agent()` wraps an async function as a Temporal **workflow** — must stay
  deterministic. `src/agent/agent.py` only calls `toolExecutor.execute(tool_name, ...)`;
  it imports nothing beyond `core.models.workflow_state` (scalar-only DTOs).
  `tests/unit/test_agent_import_boundary.py` enforces this at both the AST level and via
  a fresh-subprocess `sys.modules` check.
- **No `humanInput` usage** (removed 2026-09-01 — see Phase 14). The `APPROVAL` and
  `FAILED`-recovery gates originally used `humanInput.request_approval()`/`.request()`,
  but runtime-verified against a real Aetherion sandbox worker
  (`register_human_input_request` not present in the tool worker's activity registry in
  SDK version 0.0.55 — a `NotFoundError`, not a code bug on this project's side), which
  blocked every migration indefinitely. `build_approval_summary` still runs at
  `APPROVAL` purely to record recall/ndcg/strategy data in the audit trail; there is no
  human sign-off gate reading it.
- **Runtime-verified against a real Aetherion sandbox worker** (Phase 14, 2026-09-01) —
  see that section for the real infra bugs this surfaced and fixed. `src/__init__.py`
  defensively adds `src/` itself to `sys.path` so this package's bare imports
  (`core.xxx`, `tools.xxx`) resolve under either plausible discovery mechanism the CLI
  might use; `src/tools/__init__.py` also explicitly imports every tool module so
  `@tool()` registration doesn't depend solely on filesystem-scan discovery.

## State machine

```
INIT -> CONNECT -> DISCOVER -> NORMALIZE -> COMPARE -> PLAN -> DRY_RUN -> BENCHMARK
     -> APPROVAL -> MIGRATE -> CDC_SYNC -> VERIFY -> CUTOVER -> COMPLETE
(FAILED reachable from any state)
```

`CDC_SYNC` is V3 fix #6 — added between `MIGRATE` and `VERIFY` so replication lag has a
first-class state instead of being folded silently into `MIGRATE`. Neither shipped
adapter (`PineconeAdapter`, `QdrantAdapter`) ever reports `migration.cdc = TRUE`, so this
state is always a capability-gated no-op in this build — its *shape* exists without a
fake CDC listener behind it.

`FAILED` is minimal-viable by design: it records the failure reason and stops (no
recovery-choice prompt — see Phase 14 for why `humanInput` was removed). Re-running is an
external action (a new workflow execution started with `resume=true`) — Temporal has no
built-in "resume a terminated execution in place" primitive, so the workflow doesn't loop
back into itself automatically. See `core/checkpointing/store.py`.

## Canonical Vector IR & capability discovery

`core/models/canonical_ir.py` implements V2 §5. `core/models/capability.py` implements
the strict tri-state `Capability` (`TRUE`/`FALSE`/`UNKNOWN`, V2 §6) — every adapter
method that can't positively confirm something returns `UNKNOWN`, never `FALSE`.

## Compatibility engine — V3 correctness fixes applied

`core/compatibility/engine.py` classifies dimension/datatype/metric/vector-type/
quantization as `EXACT`/`COMPATIBLE`/`TRANSFORMABLE`/`INCOMPATIBLE`/`UNKNOWN`:

- **V3 fix #2**: Euclidean↔{cosine,dot} classifies `TRANSFORMABLE` (not stuck
  `INCOMPATIBLE`/`UNKNOWN`) once normalization is *confirmed* unit-norm
  (‖a−b‖²=2−2(a·b)); cosine↔dot classifies `COMPATIBLE` under the same condition.
  Unconfirmed normalization always stays `UNKNOWN` — never coerced permissive.
- **V3 fix #4**: a dedicated `quantization` property; `TRUE` mandates
  dequantize-before-benchmark (`core/transformations/dequantize.py`, wired into
  `tools/benchmark_tools.py` and `tools/execution_tools.py`). Dormant in this build:
  neither shipped adapter currently discovers source-side quantization.
- **V3 fix #3**: integrity-diff tolerance (`core/validation/engine.py:tolerance_for_dtype`)
  is keyed to the *discovered target storage dtype*, never hardcoded FP32.
- **V3 fix #8**: `normalization: UNKNOWN` + target metric cosine → re-normalize (L2) at
  write time (`tools/execution_tools.py:maybe_renormalize`) — a no-op on already-unit
  vectors, a correction otherwise.
- **1536D** is used everywhere as the example dimension (V2's "1538" was a typo — no real
  model outputs that).

## Transformation engine — what's real vs. an honest stub

| Strategy | Status | Why |
|---|---|---|
| `direct_copy` | Real | Identity copy, gated on full compatibility |
| `pca` | Real | sklearn-backed; fit once on the benchmark sample, persisted (`fitted_params`/`from_fitted`) and reused for the whole `MIGRATE` scan — never re-fit per batch |
| `random_projection` | Real | Fixed-seed Gaussian RP, same fit-once/persist pattern as PCA |
| `mrl` | Real | Truncate + renormalize; gated on a confirmed MRL-capable model and a documented `supported_dimensions` entry (`core/transformations/mrl.py`'s small curated registry — absence means `UNKNOWN`, not `FALSE`) |
| `re_embedding` | Real | httpx call to an OpenAI-compatible `/embeddings` endpoint; gated on `documents_available` capability or explicit documents; untestable live in this sandbox (no API key) — proven via `httpx.MockTransport` like the Pinecone adapter |
| `matryoshka_adaptor` / `learned_projection` / `retrieval_aware_projection` | **Stub** | Share `_TrainedProjectionBase` (`core/transformations/trained_projection.py`). `prepare()` raises `NotImplementedError` naming the exact contract (≥5–10k paired samples, a val split disjoint from the benchmark sample, a fixed seed, the specific loss objective). A fake "trained" projection that runs on whatever sample happens to be in a sandbox would produce a plausible-looking, methodologically bogus Recall@10 — worse than no implementation |
| `knowledge_distillation` | **Stub** | Same standard, `core/transformations/distillation.py` |
| `vec2vec` | **Stub** | `core/transformations/vec2vec.py`. **Gating logic is real** — `can_apply` implements V3 §3's exact worst-case decision tree (unknown model AND no documents AND no historical queries) so it only ever surfaces as a candidate when every semantic alternative is exhausted. The adversarial (GAN) training itself is not implemented — multi-day job needing a large unpaired corpus. Carries the **mandatory embedding-inversion disclosure** (`DISCLOSURE_TEXT`, `core/models/provenance.py:Vec2VecProvenance`) per V3 §3, wired into the provenance schema even though never populated with a real training run in this build |
| RRF hybrid compensation (V3 §5) | Real | `core/transformations/rrf.py` — pure rank-fusion arithmetic, no ML. Hook exists in the benchmark path but is not yet wired to a live sparse/BM25 signal source (sparse vectors are V2's own MVP-4, out of this build's scope) |
| Adaptive retrieval (V3 §4) | Real (config emission) | `core/optimizer/scorer.py:build_adaptive_retrieval_config` — gated on confirmed MRL support *and* target prefetch+rerank support; emits the two-stage query config, doesn't execute queries itself |

## Benchmark-gated selection & the anti-hallucination guardrail

`core/benchmarking/engine.py` computes Recall@K/NDCG@K/Top-K overlap (Jaccard, distinct
from recall)/Spearman rank correlation/score drift over already-fetched Top-K id lists.
`tools/benchmark_tools.py` gets ground truth from a **live query against the source
adapter** (concurrent across queries) and candidate rankings from a **real ephemeral
Qdrant collection** when the target is Qdrant (V2 §22's "temporary target collection",
literally — see "Infra hardening pass" above), or a brute-force exact search over the
transformed DRY_RUN sample as the Pinecone-target fallback. Either way, recall is bounded
by how much of the true corpus the sample covers; `tests/integration/test_qdrant_roundtrip.py`
uses a full-coverage sample specifically to isolate transformer fidelity from this
sampling effect, and documents why a smaller real-world sample will legitimately score
lower even for a perfect transform.

**V3 §8 anti-hallucination guardrail**, enforced mechanically, not by convention:
`tools/planning_tools.py:_require_existing_benchmark` raises unless a `benchmark_id`
resolves to a row that was actually executed and recorded — a plan can never cite a
strategy that was never tested.

**V3 fix #5**: `core/optimizer/scorer.py` min-max normalizes every raw metric (quality,
cost, latency, time, storage, risk) *before* applying configured weights, and only ever
scores candidates that already passed the quality gate — a failing candidate cannot win
by being cheap, structurally, not by convention.

## Checkpointing & idempotency

`core/checkpointing/store.py` is a JSON file per `migration_id`, scoped to exactly two
things Temporal's own durable workflow history doesn't already give for free:
cross-run resume after a terminated execution, and a human-readable audit artifact
independent of Temporal's bounded history retention. `INIT` (`check_resume`) looks up an
existing checkpoint and resumes directly to the recorded state when `resume: true`.

Idempotency lives in the write-side adapters themselves (`upsert` overwrites by id in
both Pinecone and Qdrant) — independent of the checkpoint, since Temporal's own
`RetryPolicy` can re-invoke a write activity regardless of checkpoint state.

ID collision policy (V2 §34, default `fail`) is enforced per-batch in
`tools/execution_tools.py:_apply_id_collision_policy` — every batch's ids are checked
against the target before writing, not just the first batch.

## Security

`core/security/redaction.py` — one redaction helper applied at every report boundary
(the `APPROVAL` summary, the audit report, the provenance record). Defense-in-depth: the
architecture already never carries raw secrets on the Canonical IR or workflow state (V2
§7 — only `secret_ref: env://...` references travel through the system;
`tools/_shared.py:resolve_secret_ref` is the only place that reads `os.environ`, and it's
only ever called from a tool, never the workflow).

## Phase 11: third-spec gap analysis and implementation (2026-08-31)

A third spec document (covering SMEC, Autoencoders, Shadow Retrieval, Blue-Green
Migration, Dual Representation, Query-Aware Migration, Canonical Vector Store, Semantic
Space Identity, Confidence Score, Rollback, multilingual validation) was checked against
the actual code, not assumed — every gap below was confirmed by grep/read before being
called a gap. Six concrete, bounded items were then implemented for real:

1. **Semantic space identity & safety check.** `EmbeddingProvenance.semantic_space_id`
   (`f"{provider}/{model}"`, `core/models/canonical_ir.py`) is distinct from dimension —
   two models can share a dimension while producing geometrically incompatible spaces.
   `core/compatibility/engine.py:classify_semantic_space` classifies a migration into an
   already-populated target: `EXACT` if the target is empty, `EXACT` if populated and the
   space id matches, `INCOMPATIBLE` if populated and it differs, `UNKNOWN` if populated
   and undiscoverable — never silently assumed compatible. Wired into
   `CompatibilityReport` and `tools/compatibility_tools.py`.
2. **Vector fidelity gate.** `tools/benchmark_tools.py:_evaluate_candidate` now runs
   `core/validation/engine.py:distribution_stats` on every transform's output and refuses
   to run any retrieval query against it if NaN/Inf is present — an unusable-numbers
   candidate now fails fast and honestly rather than producing a meaningless recall
   number. PCA's `inverse_transform` (`core/transformations/pca.py`) adds a non-gating
   secondary diagnostic — reconstruction L2 error and cosine similarity
   (`core/models/benchmark.py:BenchmarkResult.reconstruction_l2_error`/
   `reconstruction_cosine_similarity`) — populated only for transformers that support it.
3. **Real rollback.** `core/adapters/base.py:delete_vectors` (implemented in both
   adapters) plus `migrate_state["written_ids"]` tracking in
   `tools/execution_tools.py:migrate_batch` back a new
   `tools/execution_tools.py:rollback_migration` tool that deletes precisely the ids
   *this* migration wrote to the target — never a bulk/whole-resource clear, and it
   leaves anything that pre-existed in the target untouched (proven directly in
   `tests/integration/test_qdrant_roundtrip.py`).
4. **Confidence score.** `core/optimizer/scorer.py:compute_confidence_score` is an honest
   mean of only the metrics that were actually measured (benchmark recall/NDCG/overlap,
   verify-time recall, integrity-within-tolerance) — never a fabricated number when a
   metric wasn't computed. Stored on `ProvenanceRecord.confidence_score`
   (`tools/report_tools.py`).
5. **Best-effort language detection.** `tools/discovery_tools.py:_detect_languages` scans
   sampled source payloads for a language-ish key (`language`/`lang`/`language_code`/
   `locale`) and records the distinct values found as
   `EmbeddingProvenance.languages_detected`. This is payload inspection, not a
   vector-level language detector — an empty list means no such key was found in the
   sample, not "the source is monolingual."
6. **SMEC: an honestly-impossible stub, not another trained-projection clone.**
   `core/transformations/smec.py` gives the strategy a correct, documented place in
   `TransformStrategy`/`_transform_factory.py`, but unlike the Matryoshka-Adaptor/
   Learned-Projection/Retrieval-Aware-Projection family, `can_apply()` always returns
   `False`: SMEC is a training-time compression objective jointly learned with the source
   encoder, so it cannot be retrofitted onto vectors a frozen source model already
   produced. The only honest path to a SMEC target representation is re-embedding the
   original documents with a SMEC-trained model (`strategy=re_embedding`), not a
   migration-time transform.

All six shipped with dedicated unit tests; the full suite was green (133/133) after each
step, matching this project's standing discipline of never batching untested changes.

## Phase 12: Isotrieve-style calibration mappings (2026-08-31)

A fourth spec document was checked against the code the same way as the third (grep/read
before calling anything a gap). Its central new ask — closed-form learned mappings in the
style of github.com/krish1925/isotrieve (Ridge, Orthogonal Procrustes, ...) as an
alternative to PCA/re-embedding for dimension- or model-change migrations — was genuinely
missing and got built for real:

1. **Fixed a real, pre-existing gap first.** `tools/benchmark_tools.py:_evaluate_candidate`
   was unconditionally calling `TransformContext(documents=None)` — meaning `re_embedding`
   could never be benchmarked with real documents; it always failed with "prepare() must
   be called with documents first," recorded as an honest failure but silently so, never
   surfaced as a limitation. `tools/transform_tools.py:prepare_benchmark_sample` now reads
   the same `document_field` request key `tools/execution_tools.py` already used for
   `MIGRATE`-time re-embedding, threads `documents` through the DRY_RUN sample (index-
   aligned with `vectors`, since both are built from the same `records` list), and
   `run_benchmarks` passes it into `_evaluate_candidate`. `re_embedding` is now actually
   benchmarkable, not just theoretically wired.
2. **`RidgeMappingTransformer` (`ridge_mapping`).** A real, closed-form
   `sklearn.linear_model.Ridge` affine map fit on a calibration subset — source vectors
   paired 1:1 with freshly re-embedded target-space vectors for the *same* documents
   (`core/transformations/reembedding.py:embed_texts`, factored out of
   `ReEmbeddingTransformer` for reuse) — then applied to the *entire* corpus, avoiding a
   full re-embed. Works for any source_dim/target_dim pair. Gated on
   `MIN_CALIBRATION_SAMPLES = 200` calibration documents; fewer raises a clear `ValueError`
   rather than fitting a meaningless mapping on too little data.
3. **`OrthogonalProcrustesTransformer` (`procrustes_mapping`).** The classical
   translation-corrected orthogonal Procrustes solution
   (`scipy.linalg.orthogonal_procrustes` on mean-centered calibration pairs) — the
   textbook fit for V2 §20's "same dimension, different embedding model" case.
   Mathematically only defined when `source_dim == target_dim`; `can_apply()` refuses
   otherwise and points at `ridge_mapping` instead.
4. Both are REAL implementations (`is_stub = False`), not stubs — same tier as PCA/
   RandomProjection, because a closed-form fit needs no training loop, unlike the neural
   trained-projection family in `trained_projection.py`.

Deliberately not added in this pass, from the same Isotrieve family: **ProcrustesDiag**
and **LowRankAffine** (constrained variants of the same closed-form idea — real, but
narrower value than the two implemented, skipped to keep this batch bounded) and
**Contrastive**/**Residual-MLP** mappings (both need an iterative training loop, so they
belong with `trained_projection.py`'s honest-stub tier, not this closed-form one, if added
later).

All new code shipped with dedicated tests, including two that fit the mapping against a
*known* linear relationship / rotation via a mocked embeddings endpoint and assert the fit
actually recovers it (`tests/unit/test_linear_mapping.py`) — not merely that a same-shaped
array came back. Full suite green (141/141) after this batch.

## Phase 13: MRR, operator-supplied embedding contract, and a correctness fix for Phase 12 (2026-08-31)

1. **MRR.** `core/benchmarking/engine.py:reciprocal_rank` — the doc's §14 metric list
   named it explicitly and it was genuinely missing. Aggregated into
   `evaluate_retrieval_equivalence`'s `mrr` field and `BenchmarkResult.mrr`, same
   convention as the other Top-K metrics (source ranking defines relevance; 0.0, not
   `None`, when the target's Top-K contains no hit at all — a real, meaningful outcome).
2. **Operator-supplied embedding contract.** `EmbeddingProvenance` gained
   `revision`/`tokenizer`/`pooling`/`query_prefix`/`document_prefix` (V2 §8/§20) — none of
   these are recoverable from raw vector math, so they only populate from a new optional
   `source_embedding_config` trigger-payload key, read in
   `tools/discovery_tools.py:_build_embedding_discovery`. This also fixed a real
   correctness gap, not just added fields: without an operator-supplied `provider`+`model`,
   `EmbeddingProvenance.semantic_space_id` was always `"unknown/unknown"` for both source
   and target, which `classify_semantic_space` (Phase 11) would trivially call `EXACT` —
   a vacuous safety check. `discovered_from="operator_supplied"`/`confidence=1.0` only
   fires when BOTH provider and model are given (a partial config isn't treated as a
   confident identity); MRL detection now also checks the supplied model against the
   curated registry instead of always looking up `"unknown"`.
3. **Fixed a correctness bug in Phase 12's `ridge_mapping`/`procrustes_mapping`.** They
   could pass `BENCHMARK` and get selected by `select_strategy`, but `MIGRATE` would then
   raise `"strategy ... cannot be executed at MIGRATE time in this build"` —
   `tools/execution_tools.py:_load_execution_transformer` never had a branch for them.
   Fixed for real, not patched around: both transformers gained `fitted_params`/
   `from_fitted` (Ridge stores plain `coef_`/`intercept_` arrays instead of the sklearn
   estimator, Procrustes stores `rotation`/`source_mean`/`target_mean` — both JSON-safe,
   same convention as `RandomProjectionTransformer`) plus a public `ensure_fitted()` so
   `_load_execution_transformer` can trigger their async calibration fit exactly once and
   persist it, mirroring PCA/RandomProjection's fit-once-and-reload-via-`from_fitted`
   pattern instead of re-fitting (and re-calling the embeddings API) every batch.
   `_load_execution_transformer` and its one caller in `tools/validation_tools.py` became
   `async` to support this.

Coverage note, stated honestly rather than left implicit: the Ridge/Procrustes
`fitted_params`/`from_fitted` round trip is unit-tested directly
(`tests/unit/test_linear_mapping.py`), but — like `re_embedding`'s own MIGRATE-time
branch — `_load_execution_transformer`'s dispatch to these two strategies has no
dedicated integration test through a real local Qdrant run; `_fit_calibration_mapping`
constructs the transformer without a transport-injection seam, so exercising it end-to-end
would need a global `httpx.AsyncClient` monkeypatch rather than the constructor-injected
`httpx.MockTransport` every other transformer test uses. Full suite green (149/149).

## Phase 14: live Aetherion platform testing (2026-09-01)

Everything through Phase 13 was proven live against real Pinecone/Qdrant/OpenAI by
calling the `@tool()` coroutines directly — real, but bypassing the actual Temporal
workflow entrypoint (`src/agent/agent.py`) entirely. This phase ran the real thing: a
live Aetherion sandbox worker, triggered through the actual web UI and CLI, executing
the genuine `VectorDB_Migration_Agent` workflow. It surfaced several real gaps direct
tool-calling could never have caught, since they're specifically about the boundary
between the trigger payload, the workflow, and the worker process:

1. **`MigrationRequest` had silently drifted behind the tools it validates for.**
   `src/agent/agent.py` validates every trigger payload with
   `MigrationRequest.model_validate(payload)` before anything else runs — and pydantic
   silently *drops* unknown fields by default rather than erroring. `document_field`,
   `reembed_model`, `reembed_dimensions`, and `source_embedding_config` (all added across
   Phases 11-13) were never added to this model, so a real trigger through the actual
   workflow would have silently lost every one of them — `re_embedding`/`ridge_mapping`
   would have been unreachable in production despite being fully implemented and tested
   in isolation. Fixed by adding the missing fields to `MigrationRequest`
   (`core/models/workflow_state.py`) and `src/agent/metadata.json`'s trigger schema.
2. **Credential/endpoint refs were too rigid for how operators actually type them.**
   `resolve_secret_ref` only accepted `env://VAR_NAME`, not a bare `VAR_NAME` — confusing
   given a webform's free-text field carries no visual hint that a scheme prefix is
   required. Now accepts both. Symmetrically, `resolve_endpoint_ref` treated a bare value
   as a literal host by default (correct for endpoints, since a literal is legitimate and
   not a secret) — but a bare value that happens to exactly match a set environment
   variable now resolves to it first, falling back to literal otherwise, matching
   `resolve_secret_ref`'s convenience without changing behavior for genuine literals
   (`:memory:`, `path://...`, a real URL typed directly). A raw secret value typed
   directly into either field still correctly fails — neither change weakens the "never
   silently accept an already-exposed secret" property, confirmed by dedicated tests.
3. **`build_adapter`'s provider parsing needed to tolerate real form input.**
   `source_provider`/`target_provider` were originally a platform dropdown widget; a
   live UI bug in that widget (unrelated to this project's code) meant they had to become
   free-text fields instead. Free text can carry trailing whitespace/newlines a
   constrained dropdown never would — `build_adapter` now `.strip()`s before comparing.
4. **`register_human_input_request` is not registered on this SDK version's tool
   worker.** `humanInput.request_approval()` (the `APPROVAL` gate) and `humanInput.request()`
   (the `FAILED` recovery prompt) both depend on an SDK-internal activity,
   `register_human_input_request`, that never appeared in the live tool worker's
   registered-activity list — confirmed via a real `NotFoundError` from Temporal, not a
   guess. Tracing the compiled SDK (`aetherion_sdk` ships Nuitka-compiled with `.pyi`
   stubs, no readable source) showed this activity is meant to populate the same
   `TOOL_REGISTRY` this project's own `@tool()` functions populate, but the tool worker's
   `discover()` step only scans this project's own packages — it never imports
   `aetherion_sdk.human_input`, so the activity is never actually registered on a
   self-hosted worker in SDK 0.0.55. **Removed at explicit operator request**: both
   `humanInput` call sites in `src/agent/agent.py` are gone. `APPROVAL` now runs
   `build_approval_summary` (still real, still lands in the audit trail) and proceeds
   straight to `MIGRATE` with no human sign-off; `FAILED` records its reason and stops
   with no recovery-choice prompt. This is a genuine, real reduction in safety posture —
   a migration that clears the quality gate now writes to the target with no pause for a
   human to review the selected strategy/recall/cost first — accepted explicitly, not
   silently, and worth reinstating if a future SDK version fixes the underlying
   registration gap (or if this project's own tool worker discovery can be made to import
   `aetherion_sdk.human_input` some other way).
5. **A real `uv lock` resolution break, unrelated to this project's own dependencies.**
   Adding `pydantic-settings` (Phase 14, for `core/checkpointing/store.py`'s
   `BaseSettings`-backed checkpoint directory) triggered a resolution failure for
   `python_full_version == '3.13.*'` — `aetherion-sdk` only ships a `cp312` wheel, and
   the existing `[tool.uv] environments` constraint pinned platform/arch but not the
   Python version, so `uv` still tried (and failed) to solve for 3.13. Fixed by adding
   the Python version bound to the same constraint.

None of these are simulated or inferred — every one was reproduced live, traced to its
actual cause (including reading the SDK's compiled stub files directly rather than
guessing), and fixed with a regression test where the surface allowed one
(`tests/unit/test_shared.py`, `tests/unit/test_workflow_state.py`). Full suite green
(173/173) after this phase.

## Explicitly deferred (V3 items not built here, and why)

The third spec's remaining concepts, deliberately not built in this pass:

- **Canonical Vector Store / vendor-neutral intermediate storage** — a genuinely new
  storage layer (state to durably persist between DISCOVER and MIGRATE, beyond the JSON
  checkpoint), not a bounded addition to existing tools; needs its own design pass.
- **Shadow Retrieval / dual-write validation** — running production queries against both
  source and target concurrently and diffing results requires a live query-traffic
  interception point this agent (a batch migration tool, not a proxy) doesn't have.
- **Blue-Green Migration** — cutover orchestration/traffic-switching is an
  application-deployment concern layered on top of a completed migration, not part of
  copying and transforming vectors.
- **Dual Representation (writing old + new embeddings side by side during a model
  transition)** — a valid pattern, but it's a target-side indexing/serving decision for
  the operator's own application, not something this migration engine should silently
  decide to do to someone else's collection.
- **Representation Compatibility Graph** (a registry of which embedding spaces are
  known-comparable/known-incomparable beyond a plain equality check on
  `semantic_space_id`) — plain equality is the honest, bounded version implemented now;
  a graph implies curated cross-model comparability data this project doesn't have a
  source for yet.
- **DB/index configuration migration** (replicas, sharding, HNSW params, quantization
  settings) — V2 already scopes this out to MVP-4+ (see "Explicitly deferred" items
  below); the third spec doesn't change that boundary.

The fourth spec's remaining concepts, also deliberately not built:

- **Exact-search vs. ANN diagnostic split** (§15: run the transformed vectors through
  both brute-force exact search and the real ANN index, and diff the two to attribute a
  recall drop specifically to representation transform vs. ANN/index behavior). The
  infrastructure for both halves already exists (`_brute_force_topk` and
  `_qdrant_temp_collection_topk` in `tools/benchmark_tools.py`) but they're currently
  either/or per target provider, not run side-by-side on the same transformed vectors —
  a real, bounded follow-up, not attempted here to keep this batch's scope closed.
- **Calibration/production domain-similarity check** (§17: warn when the DRY_RUN
  calibration sample's distribution looks different from the full production corpus).
  Currently only one sample is ever drawn (`SAMPLE_SIZE = 200` in
  `tools/discovery_tools.py`, reused as the benchmark corpus) — there's no second,
  independently-drawn corpus sample to compare it against yet, so a real version of this
  check needs a modeling decision (how to draw a second comparison sample cheaply)
  before it can be built honestly rather than compared against itself.
- **Full retrieval-strategy-mismatch modeling for multi-vector/hybrid/reranking**
  (§22–23 Type C). `core/compatibility/engine.py:classify_vector_type` already refuses to
  silently auto-densify/sparsify a dense↔sparse mismatch (V2 §30), but neither adapter's
  real data path implements sparse or multi-vector value storage at all today (Pinecone/
  Qdrant adapters' `upsert_vectors`/`query` only ever handle a single dense array) — so
  modeling hybrid/multi-vector/reranker retrieval-pipeline compatibility would describe
  capabilities the adapters can't yet act on. Deferred until an adapter actually carries
  sparse/multi-vector payloads end-to-end.
- **SVD as a separate candidate from PCA** — truncated SVD on centered data is the same
  operation `PCATransformer` (via `sklearn.decomposition.PCA`) already performs; a
  distinct uncentered-SVD transformer would add a nearly-duplicate candidate for
  negligible differentiation, so it wasn't added as its own strategy.

- **MTEB/BEIR CI regression suite** (V3 §7) — needs public benchmark datasets and network
  access; a standing CI concern independent of any one migration, not part of the agent
  itself.
- **Adapter conformance certification kit** (V3 §9) — only useful once third-party
  adapters beyond Pinecone/Qdrant exist.
- **Wrapping `vector-io`/VDF** (V3 §6) — V2's MVP scope is Pinecone↔Qdrant only; the
  hand-rolled adapters were kept specifically for the fine-grained capability
  introspection V3 §6 itself says `vector-io` doesn't expose.
- **Sparse vectors, named vectors, quantization-aware indexing, real CDC listeners** — V2
  §48 places these at MVP-4, after the MVP-1/2 scope this build targets.
- **LLM-based synthetic query generation** (V2 §25) — `core/validation/engine.py:resolve_validation_mode`
  implements the decision function (golden > synthetic > unavailable) for real, but the
  generation call itself isn't wired up (no LLM key in this sandbox); `VERIFY` degrades to
  golden-query-only or explicitly `UNMEASURED`, never a fabricated pass.

## Known limitations

- Benchmark recall is bounded by DRY_RUN sample coverage (see above) — a real deployment
  with billions of vectors and a small representative sample will see materially lower
  raw recall numbers than a full-corpus benchmark would, by design of the sampling
  approach itself, not as a defect.
- `core/transformations/dequantize.py` is a best-effort linear rescale (int8 ÷127,
  binary → ±1) — the true provider-specific quantization scale isn't discoverable by
  either shipped adapter today.
- Pinecone's REST surface (`core/adapters/pinecone_adapter.py`) is verified request/
  response-shape-correct against live API docs (checked 2026-08) and proven via
  `httpx.MockTransport`, but still never exercised against a real, live Pinecone project
  — no credentials were reachable from this sandbox. The `X-Pinecone-Api-Version` pin
  will age out again; it's a constructor parameter specifically so that's a config
  change, not a code change, when it does.
- Re-embedding during `MIGRATE` requires the caller to set `document_field` in the
  trigger payload (naming which payload key holds the source text) — there's no generic
  document-discovery path since neither adapter's `retrieve_documents` capability is ever
  `TRUE` today.
- No live Temporal/Aetherion worker run was attempted (would need the user's live
  sandbox cloud account; explicitly declined for this pass) — `src/agent/agent.py`'s
  actual state-machine execution through `toolExecutor`/Temporal remains verified only
  by import-graph tests and by calling the underlying tool coroutines directly, not by a
  real workflow run.
- The real-ANN benchmark path's concurrent candidates share one Qdrant connection under
  an `asyncio.Lock` (see "Infra hardening pass") — correct, but it means candidate
  evaluations against a Qdrant target are only as parallel as that lock allows, not fully
  concurrent the way Pinecone-target (brute-force) candidates are.
