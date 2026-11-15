"""Lazy adapter for the shared cross-encoder reranker interface."""

from __future__ import annotations

import copy

from .base import FunctionTool, QUERY, ToolContext, ToolResult, evidence_item, evidence_to_found, object_schema


def _rerank(args: dict, ctx: ToolContext) -> dict:
    # This module can be absent while the search engineer works on A3.
    from app.features.search.rerank import rerank

    chunks = rerank(args["q"], copy.deepcopy(args["chunks"]), scorer=ctx.scorer)
    evidence = []
    by_id = None
    for rank, chunk in enumerate(chunks):
        located = chunk
        needed = {"segment_id", "source_type", "start_sec", "end_sec"}
        if not needed <= chunk.keys() and ctx.index_loader is not None:
            if by_id is None:
                by_id = {item["chunk_id"]: item for item in ctx.load_index().chunks}
            located = {**by_id.get(chunk["chunk_id"], {}), **chunk}
        if needed <= located.keys():
            item = evidence_item(located, located.get("video_id") or ctx.video_id)
            item["priority"] = rank
            evidence.append(item)
    # Minimal candidates remain valid reranker inputs. Only located evidence is
    # citeable; do not fabricate slide IDs or timestamps for unknown chunks.
    return {"chunks": chunks, "evidence": evidence, "found": evidence_to_found(evidence)}


RERANK = FunctionTool("rerank", "Rerank chunks with the shared cross-encoder; preserve source fields.",
                      object_schema({"q": QUERY, "chunks": {"type": "array", "maxItems": 100,
                                     "items": {"type": "object", "required": ["chunk_id", "text"],
                                               "properties": {"chunk_id": {"type": "string", "minLength": 1, "maxLength": 512},
                                                              "text": {"type": "string", "maxLength": 50000}}}}},
                                    ["q", "chunks"]), _rerank)


def rerank(q: str, chunks: list[dict], *, ctx: ToolContext | None = None) -> ToolResult:
    return RERANK.run({"q": q, "chunks": chunks}, ctx)
