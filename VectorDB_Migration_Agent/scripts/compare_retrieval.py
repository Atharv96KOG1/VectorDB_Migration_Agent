"""Ad-hoc live comparison tool — not part of the migration pipeline itself, just a
convenience for eyeballing real retrieval quality against the actual Pinecone source and
Qdrant target used throughout this project's live testing.

Two modes:
  --doc-id ID    Fair apples-to-apples comparison: use an existing document's OWN vector
                 as the query on BOTH sides (each in its native embedding space), print
                 top-K from each, and the exact-id overlap between them.
  --text "..."   A real natural-language query, embedded with text-embedding-3-small and
                 run against the QDRANT TARGET ONLY. There is no source-side equivalent
                 for free text: Pinecone's original embedding model is unknown/undiscovered,
                 so an arbitrary new query can't be pushed into the source's space at all.

Usage (run from the project root, with real credentials already exported):
    set -a; source .env; set +a
    uv run python scripts/compare_retrieval.py --doc-id cooking-38
    uv run python scripts/compare_retrieval.py --text "how do I cook chicken"
    uv run python scripts/compare_retrieval.py --doc-id cooking-38 --top-k 5
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, "src")

import httpx  # noqa: E402

from core.adapters.pinecone_adapter import PineconeAdapter 
from core.adapters.qdrant_adapter import QdrantAdapter 
from core.benchmarking.engine import (  
    ndcg_at_k,
    reciprocal_rank,
    recall_at_k,
    topk_overlap,
)

SOURCE_RESOURCE = "migration-demo-1024"
DEFAULT_TARGET_RESOURCE = "MigrationDB"
EMBEDDING_MODEL = "text-embedding-3-small"


def _print_matches(label: str, matches, text_by_id: dict[str, str]) -> None:
    print(f"=== {label} ===")
    for i, m in enumerate(matches, 1):
        text = text_by_id.get(m.id, "?")
        print(f"{i:2}. score={m.score:.4f}  id={m.id:15}  text={text!r}")
    print()


async def _fetch_texts(adapter, resource: str, ids: list[str]) -> dict[str, str]:
    records = await adapter.fetch_by_ids(resource, ids)
    return {r.id: r.payload.get("text", "?") for r in records}


async def compare_by_doc_id(doc_id: str, top_k: int, target_resource: str) -> None:
    pinecone = PineconeAdapter(api_key=os.getenv("PINECONE_API_KEY"))
    qdrant = QdrantAdapter(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    try:
        source_records = await pinecone.fetch_by_ids(SOURCE_RESOURCE, [doc_id])
        if not source_records:
            print(f"'{doc_id}' not found in Pinecone source {SOURCE_RESOURCE!r}")
            return
        target_records = await qdrant.fetch_by_ids(target_resource, [doc_id])
        if not target_records:
            print(f"'{doc_id}' not found in Qdrant target {target_resource!r} (migrate it first)")
            return

        print(f"QUERY DOCUMENT ({doc_id}): {source_records[0].payload.get('text')!r}\n")

        source_matches = await pinecone.query(SOURCE_RESOURCE, source_records[0].vector, top_k=top_k)
        source_texts = await _fetch_texts(pinecone, SOURCE_RESOURCE, [m.id for m in source_matches])
        _print_matches(f"PINECONE source, top {top_k}", source_matches, source_texts)

        target_matches = await qdrant.query(target_resource, target_records[0].vector, top_k=top_k)
        target_texts = await _fetch_texts(qdrant, target_resource, [m.id for m in target_matches])
        _print_matches(f"QDRANT target ({target_resource}), top {top_k}", target_matches, target_texts)

        source_ids = [m.id for m in source_matches]
        target_ids = [m.id for m in target_matches]
        overlap = set(source_ids) & set(target_ids)
        print(f"OVERLAP: {len(overlap)}/{top_k} ids match exactly: {sorted(overlap)}\n")

        # Same functions tools/benchmark_tools.py uses for the real BENCHMARK step — this
        # is that exact metric, computed for just this one query, not a re-implementation.
        print(f"Recall@{top_k}:      {recall_at_k(source_ids, target_ids, top_k):.4f}")
        print(f"NDCG@{top_k}:        {ndcg_at_k(source_ids, target_ids, top_k):.4f}")
        print(f"Top-K overlap:  {topk_overlap(source_ids, target_ids, top_k):.4f}  (Jaccard)")
        print(f"Reciprocal rank:{reciprocal_rank(source_ids, target_ids, top_k):.4f}  (MRR contribution)")
        print(
            "\nNote: this is ONE query, not the averaged benchmark score — expect more "
            "swing than the ~20-40 query averages reported by run_benchmarks."
        )
    finally:
        await pinecone.close()
        await qdrant.close()


async def compare_by_text(
    query_text: str, top_k: int, target_resource: str, model: str, dimensions: int | None
) -> None:
    client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
    )
    qdrant = QdrantAdapter(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    try:
        body: dict = {"input": [query_text], "model": model}
        if dimensions is not None:
            body["dimensions"] = dimensions
        resp = await client.post("/embeddings", json=body)
        resp.raise_for_status()
        vector = resp.json()["data"][0]["embedding"]

        try:
            matches = await qdrant.query(target_resource, vector, top_k=top_k)
        except Exception as exc:
            if "dimension error" in str(exc).lower():
                print(
                    f"'{target_resource}' does not hold {model!r} vectors at this dimension "
                    f"(mismatch: {exc}).\n\n"
                    f"--text mode only works against a target actually built via "
                    f"re_embedding/ridge_mapping with the SAME model+dimensions you pass "
                    f"here (--model / --dimensions) — not one built via "
                    f"direct_copy/pca/random_projection, which just reshapes the "
                    f"ORIGINAL source model's vectors and was never touched by OpenAI at all. "
                    f"Even a dimension MATCH with the wrong model silently returns garbage — "
                    f"same size, different embedding space — so pass the exact model the "
                    f"target was actually migrated with."
                )
                return
            raise
        texts = await _fetch_texts(qdrant, target_resource, [m.id for m in matches])
        print(f"QUERY: {query_text!r}\n")
        _print_matches(f"QDRANT target ({target_resource}), top {top_k}", matches, texts)
    finally:
        await client.aclose()
        await qdrant.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doc-id", help="Existing document id, e.g. cooking-38")
    group.add_argument("--text", help="Free-text natural-language query (target-only)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_RESOURCE,
        help=f"Qdrant collection name to compare against (default: {DEFAULT_TARGET_RESOURCE})",
    )
    parser.add_argument(
        "--model",
        default=EMBEDDING_MODEL,
        help=f"--text mode only: embedding model to match the target's actual content "
        f"(default: {EMBEDDING_MODEL}) — must match whatever model the target was "
        f"migrated with, or results are silently wrong even with no error.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="--text mode only: OpenAI dimensions truncation, if the target was migrated "
        "with one (e.g. text-embedding-3-large truncated to 1536).",
    )
    args = parser.parse_args()

    if args.doc_id:
        asyncio.run(compare_by_doc_id(args.doc_id, args.top_k, args.target))
    else:
        asyncio.run(
            compare_by_text(args.text, args.top_k, args.target, args.model, args.dimensions)
        )


if __name__ == "__main__":
    main()
