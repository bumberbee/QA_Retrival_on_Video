"""Thin wrappers around the existing hybrid retrieval boundary."""

from __future__ import annotations

import copy
import math

from .base import FunctionTool, QUERY, ToolContext, ToolResult, evidence_item, evidence_to_found, object_schema, validate_video_id


def _search(args: dict, ctx: ToolContext, sources: frozenset | None) -> dict:
    video_id = validate_video_id(ctx.video_id)
    if ctx.current_time is not None and (type(ctx.current_time) not in (int, float)
                                         or not math.isfinite(ctx.current_time) or ctx.current_time < 0):
        raise ValueError("current_time must be nonnegative and finite")
    searcher = ctx.searcher
    if searcher is None:
        from app.features.search.retrieval import search
        searcher = search
    found = copy.deepcopy(searcher(video_id, args["q"], current_time=ctx.current_time,
                                  encoder=ctx.encoder, sources=sources))
    weights = {"ocr": args.get("ocr_weight", 1.0), "speech": args.get("speech_weight", 1.0), "region": 1.0}
    evidence, seen = [], set()
    for slide in found["slides"]:
        for chunk in slide.get("evidence_candidates", []):
            if sources is not None and chunk["source_type"] not in sources:
                continue
            weight = weights[chunk["source_type"]]
            if weight == 0:
                continue
            item = evidence_item(chunk, video_id)
            item["title"] = item.get("title") or slide.get("title") or ""   # a chunk may carry an empty title
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            item["retrieval_score"] = float(slide.get("score", 0.0))
            item["source_weight"] = weight
            item["weighted_score"] = item["retrieval_score"] * weight
            evidence.append(item)
    # Retrieval owns E5/BM25/RRF. Only explicit source preferences change its order.
    if args.get("ocr_weight", 1.0) != 1.0 or args.get("speech_weight", 1.0) != 1.0:
        evidence.sort(key=lambda item: -item["weighted_score"])
    evidence = evidence[:args.get("top_k", 20)]
    for rank, item in enumerate(evidence):
        item["priority"] = rank
    adapted = evidence_to_found(evidence)
    adapted.update({key: value for key, value in found.items()
                    if key not in {"slides", "context_evidence", "evidence_map", "context_chars", "context_legacy"}})
    return {"chunks": copy.deepcopy(evidence), "evidence": evidence, "found": adapted}


def _validate_weights(args: dict) -> None:
    if args["ocr_weight"] == args["speech_weight"] == 0:
        raise ValueError("At least one OCR/speech weight must be positive")


SEARCH_HYBRID = FunctionTool(
    "search_hybrid", "Search lecture evidence with the existing E5/BM25 hybrid retrieval.",
    object_schema({"q": QUERY, "top_k": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                   "ocr_weight": {"type": "number", "minimum": 0, "maximum": 100, "default": 1.0},
                   "speech_weight": {"type": "number", "minimum": 0, "maximum": 100, "default": 1.0}}, ["q"]),
    lambda args, ctx: _search(args, ctx, None), search=True, validator=_validate_weights)
SEARCH_OCR = FunctionTool("search_ocr", "Search OCR evidence only (formulas, tables and slide text).",
                          object_schema({"q": QUERY}, ["q"]),
                          lambda args, ctx: _search(args, ctx, frozenset({"ocr"})), search=True)
SEARCH_SPEECH = FunctionTool("search_speech", "Search spoken lecture evidence only.",
                             object_schema({"q": QUERY}, ["q"]),
                             lambda args, ctx: _search(args, ctx, frozenset({"speech"})), search=True)


def search_hybrid(q: str, top_k: int = 20, ocr_weight: float = 1.0, speech_weight: float = 1.0,
                  *, ctx: ToolContext | None = None) -> ToolResult:
    return SEARCH_HYBRID.run({"q": q, "top_k": top_k, "ocr_weight": ocr_weight, "speech_weight": speech_weight}, ctx)


def search_ocr(q: str, *, ctx: ToolContext | None = None) -> ToolResult:
    return SEARCH_OCR.run({"q": q}, ctx)


def search_speech(q: str, *, ctx: ToolContext | None = None) -> ToolResult:
    return SEARCH_SPEECH.run({"q": q}, ctx)
