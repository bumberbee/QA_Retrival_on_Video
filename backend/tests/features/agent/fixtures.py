"""Small in-memory lecture and deterministic dependencies; no persisted index."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np

from app.features.agent.tools import Budget, BudgetLimits, ToolContext
from app.features.search.retrieval import LectureIndex


def lecture_index():
    def segment(sid, start, end, text, kind="slide"):
        return {"segment_id": f"vid:{sid}", "start_sec": start, "end_sec": end, "content_kind": kind,
                "visual": {"title_clean": sid, "ocr_text_clean": text, "image_path": f"{sid}.jpg"},
                "speech": {}}

    segments = [segment("s3", 30, 40, "Third"), segment("ad", 20, 30, "Ad", "ad"),
                segment("s1", 0, 10, "Full first OCR\nA = B + C"), segment("s2", 10, 20, "Second OCR")]
    chunks = [
        {"chunk_id": "sp2", "segment_id": "vid:s2", "source_type": "speech", "text": "Later speech",
         "start_sec": 14, "end_sec": 18},
        {"chunk_id": "ocr1", "segment_id": "vid:s1", "source_type": "ocr", "text": "New OCR only",
         "start_sec": 0, "end_sec": 10},
        {"chunk_id": "sp1", "segment_id": "vid:s1", "source_type": "speech", "text": "Early speech",
         "start_sec": 4, "end_sec": 8},
        {"chunk_id": "adsp", "segment_id": "vid:ad", "source_type": "speech", "text": "Advertisement",
         "start_sec": 20, "end_sec": 30},
    ]
    return LectureIndex(Path("synthetic-index"), chunks, np.ones((4, 1)), segments,
                        {segment["segment_id"]: segment for segment in segments})


def context(index=None, **kwargs):
    return ToolContext(video_id="vid", index_loader=Mock(return_value=index or lecture_index()),
                       budget=Budget(BudgetLimits()), **kwargs)


def found(index=None):
    index = index or lecture_index()
    return {"slides": [{"video_id": "vid", "segment_id": "vid:s1", "title": "First",
                        "start_sec": 0, "end_sec": 10, "score": 0.04, "semantic_score": 0.9,
                        "ocr_text": "New OCR only", "speech_text": "Early speech", "figures": [],
                        "evidence_candidates": [dict(index.chunks[1]), dict(index.chunks[2])]},
                       {"video_id": "vid", "segment_id": "vid:s2", "title": "Second",
                        "start_sec": 10, "end_sec": 20, "score": 0.03, "semantic_score": 0.8,
                        "ocr_text": "", "speech_text": "Later speech", "figures": [],
                        "evidence_candidates": [dict(index.chunks[0])]}],
            "best_semantic": 0.9, "temporal": False}
