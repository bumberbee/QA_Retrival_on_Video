"""Full slide reads, adjacent slides, and ordered speech windows."""

from __future__ import annotations

from pathlib import Path

from .base import FunctionTool, SLIDE_ID, VIDEO_ID, ToolContext, ToolResult, evidence_item, evidence_to_found, object_schema


def _is_ad(segment: dict) -> bool:
    from app.features.search.retrieval import is_ad
    return is_ad(segment)


def _segments(index) -> list[dict]:
    return sorted((segment for segment in index.segments if not _is_ad(segment)),
                  key=lambda segment: (segment["start_sec"], segment["end_sec"], segment["segment_id"]))


def _resolve_slide(index, slide_id: str, video_id: str) -> dict:
    for segment in _segments(index):
        if segment["segment_id"] == slide_id or segment["segment_id"] == f"{video_id}:{slide_id}":
            return segment
    raise ValueError(f"Slide not found: {slide_id}")


def _slide(index, segment: dict, video_id: str) -> dict:
    slide_id = segment["segment_id"]
    chunks = sorted((chunk for chunk in index.chunks if chunk["segment_id"] == slide_id),
                    key=lambda chunk: (chunk["start_sec"], chunk["end_sec"], chunk["chunk_id"]))
    visual = segment.get("visual", {})
    # The index may store only OCR new since the previous slide; read the full record.
    text = visual.get("ocr_text_clean") or visual.get("ocr_text") or "\n".join(
        chunk["text"] for chunk in chunks if chunk["source_type"] == "ocr")
    image = visual.get("image_path") or next((chunk.get("image_path") for chunk in chunks if chunk.get("image_path")), None)
    image_path = str(Path(index.folder) / "lecture" / image) if image and index.folder else image
    title = visual.get("title_clean") or visual.get("title") or next((chunk.get("title") for chunk in chunks if chunk.get("title")), "")
    evidence = []
    if text:
        evidence.append(evidence_item({"chunk_id": f"{slide_id}:ocr", "segment_id": slide_id,
                                       "text": text, "source_type": "ocr", "title": title,
                                       "start_sec": segment["start_sec"], "end_sec": segment["end_sec"],
                                       "image_path": image_path}, video_id))
    return {"slide_id": slide_id, "video_id": video_id, "ocr_text": text,
            "t_start": segment["start_sec"], "t_end": segment["end_sec"], "image_path": image_path,
            "title": title, "evidence": evidence}


def _get_slide(args: dict, ctx: ToolContext) -> dict:
    index = ctx.load_index()
    slide = _slide(index, _resolve_slide(index, args["slide_id"], ctx.video_id), ctx.video_id)
    return {**slide, "found": evidence_to_found(slide["evidence"])}


def _neighbors(args: dict, ctx: ToolContext) -> dict:
    index = ctx.load_index()
    ordered = _segments(index)
    anchor = _resolve_slide(index, args["slide_id"], ctx.video_id)
    position = ordered.index(anchor)
    slides = [_slide(index, segment, ctx.video_id)
              for i, segment in enumerate(ordered) if 0 < abs(i - position) <= args["k"]]
    evidence = [item for slide in slides for item in slide["evidence"]]
    return {"slides": slides, "evidence": evidence, "found": evidence_to_found(evidence)}


def _speech_evidence(index, video_id: str) -> list[dict]:
    segments = _segments(index)
    allowed = {segment["segment_id"] for segment in segments}
    precise, evidence = set(), []
    for segment in segments:
        speech = segment.get("speech", {})
        spans = speech.get("spans", [])
        if spans:
            precise.add(segment["segment_id"])
            for position, span in enumerate(spans):
                if span.get("evidence_excluded") or not span.get("text"):
                    continue
                evidence.append(evidence_item({**span, "chunk_id": span.get("span_id") or
                                               f"{segment['segment_id']}:speech_{position:04d}",
                                               "segment_id": segment["segment_id"], "source_type": "speech"}, video_id))
    covered = set(precise)
    for chunk in index.chunks:
        sid = chunk["segment_id"]
        if chunk["source_type"] != "speech" or (index.segments and sid not in allowed):
            continue
        # Coarse fallback ASR spans are not duplicated in the word-aligned spans.
        if sid in precise and chunk.get("timing_precision") != "asr_segment":
            continue
        evidence.append(evidence_item(chunk, video_id))
        covered.add(sid)
    for segment in segments:
        text = segment.get("speech", {}).get("text", "")
        if segment["segment_id"] not in covered and text:
            evidence.append(evidence_item({"chunk_id": f"{segment['segment_id']}:speech",
                                           "segment_id": segment["segment_id"], "text": text,
                                           "source_type": "speech", "start_sec": segment["start_sec"],
                                           "end_sec": segment["end_sec"]}, video_id))
    return evidence


def _get_window(args: dict, ctx: ToolContext) -> dict:
    index = ctx.load_index(args["video_id"])
    # Overlap with [t_start, t_end); preserve the source span rather than inventing
    # clipped timings for words whose positions are not available.
    evidence = [item for item in _speech_evidence(index, args["video_id"])
                if item["start_sec"] < args["t_end"] and item["end_sec"] > args["t_start"]]
    evidence.sort(key=lambda item: (item["start_sec"], item["end_sec"], item["chunk_id"]))
    seen, unique = set(), []
    for item in evidence:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)
    evidence = unique
    return {"video_id": args["video_id"], "t_start": args["t_start"], "t_end": args["t_end"],
            "speech_text": "\n".join(item["text"] for item in evidence), "evidence": evidence,
            "found": evidence_to_found(evidence)}


def _validate_window(args: dict) -> None:
    if args["t_start"] >= args["t_end"]:
        raise ValueError("t_start must be less than t_end")


GET_SLIDE = FunctionTool("get_slide", "Read full OCR, slide timestamps and the image path.",
                         object_schema({"slide_id": SLIDE_ID}, ["slide_id"]), _get_slide)
NEIGHBORS = FunctionTool("neighbors", "Read up to k preceding and k following lecture slides in time order.",
                         object_schema({"slide_id": SLIDE_ID, "k": {"type": "integer", "minimum": 0,
                                                                 "maximum": 10, "default": 1}}, ["slide_id"]), _neighbors)
GET_WINDOW = FunctionTool("get_window", "Read speech spans overlapping a half-open time window, in time order.",
                          object_schema({"video_id": VIDEO_ID, "t_start": {"type": "number", "minimum": 0},
                                         "t_end": {"type": "number", "minimum": 0}},
                                        ["video_id", "t_start", "t_end"]), _get_window, validator=_validate_window)


def get_slide(slide_id: str, *, ctx: ToolContext | None = None) -> ToolResult:
    return GET_SLIDE.run({"slide_id": slide_id}, ctx)


def get_window(video_id: str, t_start: float, t_end: float, *, ctx: ToolContext | None = None) -> ToolResult:
    return GET_WINDOW.run({"video_id": video_id, "t_start": t_start, "t_end": t_end}, ctx)


def neighbors(slide_id: str, k: int = 1, *, ctx: ToolContext | None = None) -> ToolResult:
    return NEIGHBORS.run({"slide_id": slide_id, "k": k}, ctx)
