"""APPROVAL summary (kept small and reference-based — the full plan/benchmark table is
never embedded, only pointed at, regardless of who or what reads it) and the COMPLETE
state's provenance record + audit report (V2 §53, §57), with redaction (V2 §43) applied
to everything written to disk. `build_approval_summary` still runs at APPROVAL (see
src/agent/agent.py) purely to capture this data in the audit trail — there is no human
sign-off gate reading it, that was removed 2026-09-01.
"""

from __future__ import annotations

from aetherion_sdk import tool

from core.checkpointing.store import (
    DEFAULT_CHECKPOINT_DIR,
    load_checkpoint,
    save_checkpoint,
    write_artifact,
    write_text_artifact,
)
from core.models.canonical_ir import CanonicalVectorIR, VectorFieldSpec
from core.models.migration_plan import MigrationPlan, TransformStrategy
from core.models.provenance import ProvenanceRecord, TransformationProvenance, Vec2VecProvenance
from core.optimizer.scorer import compute_confidence_score
from core.security.redaction import redact
from tools._transform_factory import provenance_parameters_from_fitted


def _md_value(value: object, unit: str = "") -> str:
    if value is None:
        return "*not discovered*"
    if value == [] or value == {}:
        return "*none*"
    return f"{value}{unit}"


def _md_bool_capability(caps: dict, path: list[str]) -> str:
    node: object = caps
    for key in path:
        if not isinstance(node, dict):
            return "*unknown*"
        node = node.get(key)
    return str(node) if node is not None else "*unknown*"


def _build_schema_semantic_report(
    migration_id: str,
    req: dict,
    field: VectorFieldSpec,
    discovery: dict,
    compatibility: dict,
    ir,
) -> str:
    """Real, field-by-field source-vs-target schema/semantic comparison, built entirely
    from the Canonical IR + discovery + compatibility data this migration actually
    measured — never a fabricated/templated value. Two things the request asked for are
    honestly out of scope for this build rather than faked: (1) target-side embedding
    identity — this system tracks one EmbeddingProvenance per vector field, describing
    the SOURCE embedding; the target inherits that same semantic space unless the
    selected strategy is re_embedding (called out explicitly below), because nothing
    re-embeds into a new, independently-identified space otherwise. (2) "Query
    Distribution" — there is no query-log/traffic-pattern analysis anywhere in this
    system; the only real query activity is the DRY_RUN benchmark sample, sized and
    reported honestly below, not a production query distribution.
    """
    src_caps = discovery.get("source", {}).get("capabilities", {})
    tgt_caps = discovery.get("target", {}).get("capabilities", {})
    src_info = discovery.get("source", {}).get("resource_info", {})
    tgt_info = discovery.get("target", {}).get("resource_info", {})
    norm = discovery.get("normalization", {})
    emb = field.embedding

    lines: list[str] = []
    lines.append(f"# Migration Schema & Semantic Report — `{migration_id}`")
    lines.append("")
    lines.append(f"- **Source**: `{req['source_provider']}` / `{req['source_resource']}`")
    lines.append(f"- **Target**: `{req['target_provider']}` / `{req['target_resource']}`")
    lines.append("")
    lines.append(
        "All values below come from this migration's own real discovery, canonical-IR "
        "normalization, and compatibility check — nothing here is a template default. "
        "`UNKNOWN`/`*not discovered*` means the capability genuinely could not be "
        "confirmed on that side, never that it's assumed absent (tri-state discovery, "
        "V2 §6)."
    )
    lines.append("")

    lines.append("## Vector, Embedding & Semantic Space")
    lines.append("")
    lines.append("| Property | Source | Target |")
    lines.append("|---|---|---|")
    lines.append(f"| Provider | {req['source_provider']} | {req['target_provider']} |")
    lines.append(f"| Resource | {req['source_resource']} | {req['target_resource']} |")
    lines.append(f"| Vector Kind | {field.source.kind.value} | {field.target.kind.value} |")
    lines.append(
        f"| Dimension | {_md_value(field.source.dimension)} | {_md_value(field.target.dimension)} |"
    )
    lines.append(
        f"| Data Type | {field.source.datatype.value} | {field.target.datatype.value} |"
    )
    lines.append(
        f"| Similarity Metric | {field.source.metric.value} | {field.target.metric.value} |"
    )
    lines.append(f"| Quantized Storage (source) | {field.source.quantized.value} | — *(not tracked on target spec — target storage format is the target provider's own concern, not this migration's)* |")
    lines.append(
        f"| Embedding Provider | {emb.provider} | *(inherited from source — see note below)* |"
    )
    lines.append(f"| Embedding Model | {emb.model} | *(inherited from source)* |")
    lines.append(
        f"| Model Version / Revision | {_md_value(emb.revision)} | *(inherited from source)* |"
    )
    lines.append(
        f"| **Semantic Space ID** | `{emb.semantic_space_id}` | `{emb.semantic_space_id}` "
        f"*(same space — this migration does not re-embed into a new space unless the "
        f"selected strategy is `re_embedding`; see Transformation section in the full "
        f"provenance record for the strategy actually used)* |"
    )
    lines.append(
        f"| Native Dimension (pre-MRL truncation) | {_md_value(emb.native_dimension)} | — |"
    )
    lines.append(
        f"| Tokenizer | {_md_value(emb.tokenizer)} | — |"
    )
    lines.append(f"| Pooling | {_md_value(emb.pooling)} | — |")
    lines.append(
        f"| Query Prefix / Document Prefix | {_md_value(emb.query_prefix)} / {_md_value(emb.document_prefix)} | — |"
    )
    lines.append(
        f"| Languages Detected (from payload sample) | {_md_value(emb.languages_detected)} | — |"
    )
    lines.append(
        f"| Embedding Identity Confidence | {emb.confidence:.2f} ({emb.discovered_from}) | — |"
    )
    lines.append(
        f"| Approximate Vector Count | {_md_value(src_info.get('approximate_count'))} | {_md_value(tgt_info.get('approximate_count'))} |"
    )
    lines.append("")

    lines.append("## Normalization")
    lines.append("")
    lines.append(
        "Measured once, on a real sample drawn from the **source** during DISCOVER "
        "(`core/validation/engine.py:analyze_normalization`) — this build does not "
        "independently re-measure normalization on the target side; a target receiving "
        "an unmodified copy inherits the source's normalization state exactly."
    )
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    lines.append(f"| Status | {norm.get('status', 'unknown')} |")
    lines.append(f"| Method | {_md_value(norm.get('method'))} |")
    lines.append(f"| Confidence (fraction of sample within tolerance of unit norm) | {_md_value(norm.get('confidence'))} |")
    lines.append(f"| Mean Norm | {_md_value(norm.get('mean_norm'))} |")
    lines.append(f"| Std Norm | {_md_value(norm.get('std_norm'))} |")
    lines.append(f"| MRL Supported | {emb.mrl.supported.value} |")
    lines.append(f"| MRL Supported Dimensions | {_md_value(emb.mrl.supported_dimensions)} |")
    lines.append("")

    lines.append("## Capability Discovery (tri-state)")
    lines.append("")
    lines.append(
        "`TRUE` / `FALSE` / `UNKNOWN` — `UNKNOWN` means this adapter could not "
        "positively confirm the capability, never that it's assumed `FALSE`."
    )
    lines.append("")
    cap_rows = [
        ("Connectivity", "connect", ["connectivity", "connect"]),
        ("Connectivity", "tls", ["connectivity", "tls"]),
        ("Data", "list_vectors", ["data", "list_vectors"]),
        ("Data", "scan_vectors", ["data", "scan_vectors"]),
        ("Data", "retrieve_vectors", ["data", "retrieve_vectors"]),
        ("Data", "retrieve_payload", ["data", "retrieve_payload"]),
        ("Data", "retrieve_documents", ["data", "retrieve_documents"]),
        ("Data", "cdc", ["data", "cdc"]),
        ("Representation", "dimension", ["representation", "dimension"]),
        ("Representation", "metric", ["representation", "metric"]),
        ("Representation", "datatype", ["representation", "datatype"]),
        ("Representation", "dense_vectors", ["representation", "dense_vectors"]),
        ("Representation", "sparse_vectors", ["representation", "sparse_vectors"]),
        ("Representation", "named_vectors", ["representation", "named_vectors"]),
        ("Representation", "normalization", ["representation", "normalization"]),
        ("Representation", "embedding_model", ["representation", "embedding_model"]),
        ("Representation", "quantized_storage", ["representation", "quantized_storage"]),
        ("Index", "retrieve_config", ["index", "retrieve_config"]),
        ("Index", "configure_index", ["index", "configure_index"]),
        ("Migration", "resumable_scan", ["migration", "resumable_scan"]),
        ("Migration", "export", ["migration", "export"]),
        ("Migration", "cdc", ["migration", "cdc"]),
    ]
    lines.append("| Group | Capability | Source | Target |")
    lines.append("|---|---|---|---|")
    for group, name, path in cap_rows:
        lines.append(
            f"| {group} | {name} | {_md_bool_capability(src_caps, path)} | {_md_bool_capability(tgt_caps, path)} |"
        )
    lines.append("")

    lines.append("## Compatibility Classification (source → target)")
    lines.append("")
    lines.append("| Property | Classification | Reason |")
    lines.append("|---|---|---|")
    for prop in ("dimension", "datatype", "metric", "vector_type", "quantization", "semantic_space"):
        entry = compatibility.get(prop, {})
        lines.append(
            f"| {prop} | {entry.get('classification', 'unknown')} | {entry.get('reason', '—')} |"
        )
    lines.append("")

    lines.append("## Metadata / Payload Schema")
    lines.append("")
    lines.append(
        f"- Declared payload fields: {_md_value(ir.payload.schema_fields)} "
        f"*(empty means fields are discovered per-record at MIGRATE time, not "
        f"pre-declared in the plan)*"
    )
    lines.append(f"- ID type: `{ir.id.type}` — collision policy: `{ir.id.collision_policy.value}`")
    lines.append(
        f"- Namespace: source supports namespaces = `{ir.namespace.source_supported}`, "
        f"target mapping = `{_md_value(ir.namespace.target_mapping)}`"
    )
    lines.append("")

    lines.append("## Index Configuration")
    lines.append("")
    lines.append(
        "This build's Canonical IR tracks whether an adapter *can* retrieve/configure "
        "index settings (`Index.retrieve_config`/`configure_index` above), not the "
        "granular parameters themselves (e.g. HNSW `m`/`ef_construct`, IVF `nlist`, PQ "
        "code size) — those are provider-specific and not yet normalized into a common "
        "schema. See each provider's own adapter (`core/adapters/*.py`) for what it can "
        "report at `get_resource_info` time."
    )
    lines.append("")

    lines.append("## Query Distribution")
    lines.append("")
    sample_size = discovery.get("sample_size", 0)
    lines.append(
        f"Not tracked as a production query-traffic distribution — this system has no "
        f"query-log or traffic-pattern analysis feature. The only real query activity "
        f"behind this report is the DRY_RUN/BENCHMARK retrieval-equivalence sample: "
        f"**{sample_size}** source vectors sampled, benchmarked with a real live "
        f"Top-K query per sampled query vector against both source and target."
    )
    lines.append("")

    lines.append("## Validation Thresholds")
    lines.append("")
    lines.append("| Threshold | Value |")
    lines.append("|---|---|")
    lines.append(f"| Minimum Recall@10 | {ir.validation.minimum_recall_at_10} |")
    lines.append(f"| Minimum NDCG@10 | {ir.validation.minimum_ndcg_at_10} |")
    lines.append(f"| Maximum Top-K Overlap Drop | {ir.validation.maximum_topk_overlap_drop} |")
    lines.append(
        f"| Maximum Latency Increase % | {ir.validation.maximum_latency_increase_percent} |"
    )
    lines.append("")

    return "\n".join(lines)


def _find_benchmark(checkpoint: dict, benchmark_id: str | None) -> dict | None:
    if benchmark_id is None:
        return None
    return next(
        (b for b in checkpoint.get("benchmark_results", []) if b["benchmark_id"] == benchmark_id),
        None,
    )


@tool()
async def build_approval_summary(migration_id: str) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")
    plan = MigrationPlan.model_validate(checkpoint["plan"])
    benchmark = _find_benchmark(checkpoint, plan.selected_benchmark_id)
    if benchmark is None:
        return {
            "migration_id": migration_id,
            "ready": False,
            "reason": "no strategy has been selected yet",
        }

    is_stub = any(c.is_stub for c in plan.candidates if c.strategy == plan.selected_strategy)

    return {
        "migration_id": migration_id,
        "ready": True,
        "strategy": plan.selected_strategy.value,
        "benchmark_id": plan.selected_benchmark_id,
        "recall_at_10": round(benchmark["recall_at_10"], 4),
        "ndcg_at_10": round(benchmark["ndcg_at_10"], 4),
        "topk_overlap": round(benchmark["topk_overlap"], 4),
        "sample_size": benchmark["sample_size"],
        "is_stub_strategy": is_stub,
        "detail_ref": f"migration_id={migration_id} (full plan/benchmark table in the checkpoint, not embedded here — 256KB humanInput ceiling)",
    }


@tool()
async def write_provenance_and_report(migration_id: str) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}")
    req = checkpoint["request"]
    ir = CanonicalVectorIR.model_validate(checkpoint["canonical_ir"])
    field = ir.vectors[0]
    plan = MigrationPlan.model_validate(checkpoint["plan"])
    benchmark = _find_benchmark(checkpoint, plan.selected_benchmark_id)
    if benchmark is None:
        raise RuntimeError("cannot write provenance: plan has no selected, executed benchmark")

    fitted = checkpoint.get("fitted_transform") or {}
    transformation = TransformationProvenance(
        strategy=plan.selected_strategy.value,
        input_dimension=field.source.dimension,
        output_dimension=field.target.dimension,
        # A small, report-safe summary — never fitted["params"] itself, which for
        # pca/random_projection/ridge_mapping/procrustes*/low_rank_affine_mapping holds
        # the full weight matrix (a 1024x1536 matrix alone is 1.5M+ floats as JSON text).
        # Real bug found live, 2026-09-02: embedding that raw matrix inline in the
        # workflow result caused visible lag rendering the platform's Result panel.
        parameters=provenance_parameters_from_fitted(
            plan.selected_strategy,
            fitted.get("params", {}),
            field.target.dimension,
            reembed_model=req.get("reembed_model", "text-embedding-3-small"),
        ),
        artifact_id=benchmark["benchmark_id"],
        source_semantic_space_id=field.embedding.semantic_space_id,
    )
    vec2vec_provenance = (
        Vec2VecProvenance() if plan.selected_strategy == TransformStrategy.VEC2VEC else None
    )

    validation = checkpoint.get("validation") or {}
    verify_retrieval = validation.get("retrieval_metrics") or {}
    integrity = validation.get("integrity") or {}
    confidence_score = compute_confidence_score(
        benchmark_recall_at_10=benchmark["recall_at_10"],
        benchmark_ndcg_at_10=benchmark["ndcg_at_10"],
        benchmark_topk_overlap=benchmark["topk_overlap"],
        verify_recall_at_10=verify_retrieval.get("recall_at_10"),
        integrity_within_tolerance=integrity.get("within_tolerance"),
    )

    record = ProvenanceRecord(
        migration_id=migration_id,
        source_provider=req["source_provider"],
        source_resource=req["source_resource"],
        target_provider=req["target_provider"],
        target_resource=req["target_resource"],
        transformation=transformation,
        vec2vec=vec2vec_provenance,
        selected_benchmark_id=plan.selected_benchmark_id,
        quality_gate_passed=benchmark["passed_quality_gate"],
        confidence_score=confidence_score,
    )

    redacted = redact(checkpoint)
    migrate_summary = redacted.get("migrate")
    if migrate_summary is not None:
        # written_ids is one string id per vector written — unbounded for a real
        # migration (thousands+ entries). A count is all an inline result needs; the
        # full list already lives in the on-disk checkpoint for anyone who needs it.
        migrate_summary = {
            **migrate_summary,
            "vectors_written_count": len(migrate_summary.get("written_ids", [])),
        }
        migrate_summary.pop("written_ids", None)
    report = {
        "migration_id": migration_id,
        "history": redacted.get("history"),
        "compatibility": redacted.get("compatibility"),
        "plan": redacted.get("plan"),
        "benchmark_results": redacted.get("benchmark_results"),
        "migrate": migrate_summary,
        "validation": redacted.get("validation"),
        "provenance": record.model_dump(mode="json"),
    }

    provenance_path = write_artifact(
        DEFAULT_CHECKPOINT_DIR / "provenance",
        f"{migration_id}.json",
        record.model_dump(mode="json"),
    )
    report_path = write_artifact(DEFAULT_CHECKPOINT_DIR / "reports", f"{migration_id}.json", report)

    markdown_report = _build_schema_semantic_report(
        migration_id, req, field, checkpoint.get("discovery", {}), checkpoint.get("compatibility", {}), ir
    )
    markdown_path = write_text_artifact(
        DEFAULT_CHECKPOINT_DIR / "reports", f"{migration_id}.md", markdown_report
    )

    checkpoint["provenance"] = record.model_dump(mode="json")
    checkpoint["report_ref"] = report_path
    checkpoint["provenance_ref"] = provenance_path
    checkpoint["markdown_report_ref"] = markdown_path
    checkpoint["history"].append({"state": "COMPLETE", "detail": {"report_ref": report_path}})
    save_checkpoint(migration_id, checkpoint)

    return {
        "migration_id": migration_id,
        "report_ref": report_path,
        "provenance_ref": provenance_path,
        "markdown_report_ref": markdown_path,
        "quality_gate_passed": record.quality_gate_passed,
        "confidence_score": record.confidence_score,
        # *_ref are local filesystem paths on whatever worker ran this tool — invisible to
        # the operator on a managed/hosted worker (confirmed: aetherion_sdk has no
        # artifact/blob/download API at all, checked every .pyi stub, 2026-09-01). The full
        # report content is small and fully JSON-safe (written_ids trimmed to a count), so
        # it's returned inline here too — real content the operator can actually see and
        # preview in the platform's own Result panel, not just an inaccessible path.
        "report": report,
        # Same rationale as above, applied to the human-readable schema/semantic
        # Markdown — returned as text so it renders directly in the platform's Result
        # panel even if structured_results' file_key never resolves to anything (this
        # SDK version has no real upload API behind that convention — see agent.py).
        "markdown_report": markdown_report,
    }
