"""Evidence search over a video's stage-05 index (kaggle/lecture_rag), for the Q&A endpoint.

Same method as notebook_code/retrieval.py, without PyTorch: the multilingual-E5 query encoder
runs as ONNX (onnxruntime + tokenizers, pinned to the revision the index was built with), the
keyword side is the index's SQLite FTS5 table. Meaning counts more than keywords (weighted
reciprocal rank fusion) because ASR spelling errors and common words make BM25 noisy.

Evidence is grouped per slide: its title, cleaned OCR text, lecture speech, time span and
keyframe image, so the answer model can also read numbers from the slide image itself.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import unicodedata
from contextlib import closing
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .. import storage

INDEX_DIR_NAME = "index"
E5_REPO = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SEMANTIC_WEIGHT, LEXICAL_WEIGHT, RRF_K = 1.0, 0.5, 60
CANDIDATES, MAX_SLIDES, MAX_IMAGES = 20, 3, 2
STOPWORDS = set("""a ai anh ay ba bang bao bi boi cac cach cai can cho chi chua chung co con cua cung da dang
de den deu di do doi duoc gi gio ha hay hon khi khong la lai lam len luc ma minh mot moi muon nao nay
neu ngay nhieu nhu nhung noi o phai qua ra rang rat roi sao se sau tai the thi thoi theo thu tren tro
trong tu va van ve vi voi vua xin what how why which when who is are of an to in on for and or do does""".split())
TEMPORAL = ("slide nay", "doan nay", "luc nay", "cho nay", "hien tai", "this slide", "current slide")


def normalize_search(text: str) -> str:
    """Identical to notebook_code/common.py: the FTS5 index was built with it."""
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return "".join(char for char in text if not unicodedata.combining(char))


def index_dir(video_id: str) -> Path:
    return storage.DATA_DIR / INDEX_DIR_NAME / video_id


def has_index(video_id: str) -> bool:
    folder = index_dir(video_id)
    return all((folder / name).exists() for name in ("chunks.jsonl", "embeddings.npy", "lexical.sqlite3"))


class OnnxE5:
    """multilingual-e5-small query encoder (mean pooling + L2 norm), as sentence-transformers."""

    def __init__(self):
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model = hf_hub_download(E5_REPO, "onnx/model.onnx", revision=E5_REVISION)
        tokenizer = hf_hub_download(E5_REPO, "onnx/tokenizer.json", revision=E5_REVISION)
        self.session = onnxruntime.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tokenizer)
        self.tokenizer.enable_truncation(512)
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.lock = threading.Lock()

    def encode(self, text: str):
        import numpy as np

        encoding = self.tokenizer.encode("query: " + text)
        ids = np.array([encoding.ids], dtype=np.int64)
        mask = np.array([encoding.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        with self.lock:
            hidden = self.session.run(None, feed)[0]
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
        vector = pooled[0]
        return vector / (np.linalg.norm(vector) + 1e-12)


@lru_cache(maxsize=1)
def default_encoder() -> OnnxE5:
    return OnnxE5()


@dataclass
class LectureIndex:
    folder: Path
    chunks: list
    vectors: object
    segments: list
    by_segment: dict = field(default_factory=dict)


@lru_cache(maxsize=16)
def load_index(video_id: str) -> LectureIndex:
    import numpy as np

    folder = index_dir(video_id)
    chunks = [json.loads(line) for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    vectors = np.load(folder / "embeddings.npy", allow_pickle=False)
    if len(vectors) != len(chunks):
        raise ValueError("Chỉ mục hỏng: số vector khác số đoạn.")
    segments_file = folder / "lecture" / "segments.json"
    segments = json.loads(segments_file.read_text(encoding="utf-8")) if segments_file.exists() else []
    return LectureIndex(folder, chunks, vectors, segments, {s["segment_id"]: s for s in segments})


def lexical_hits(folder: Path, question: str, count: int) -> list[str]:
    words = list(dict.fromkeys(re.findall(r"[^\W_]+", normalize_search(question))))
    terms = [word for word in words if word not in STOPWORDS] or words
    if not terms:
        return []
    expression = " OR ".join(f'"{term}"' for term in terms[:64])
    with closing(sqlite3.connect((folder / "lexical.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return [row[0] for row in db.execute(
            "SELECT chunk_id FROM evidence WHERE evidence MATCH ? ORDER BY bm25(evidence) LIMIT ?",
            (expression, count))]


def is_ad(segment: dict) -> bool:
    """An ad segment (content_kind "ad"), or one the ad filter only flagged ("flag" mode)."""
    return segment.get("content_kind") == "ad" or bool(segment.get("ad_flagged"))


def segment_at(segments: list, time_sec: float):
    for index, segment in enumerate(segments):
        if segment["start_sec"] <= time_sec < segment["end_sec"]:
            if is_ad(segment):   # paused on an ad: the slide shown before it
                earlier = [s for s in segments[:index] if not is_ad(s)]
                return earlier[-1] if earlier else None
            return segment
    return None


def search(video_id: str, question: str, current_time: float | None = None, encoder=None) -> dict:
    """Ranked chunks fused per slide -> {'slides': [...], 'best_semantic': float, 'temporal': bool}."""
    import numpy as np

    index = load_index(video_id)
    vector = (encoder or default_encoder()).encode(question)
    similarities = index.vectors @ vector
    by_id = {chunk["chunk_id"]: n for n, chunk in enumerate(index.chunks)}
    scores: dict[str, float] = {}
    for rank, n in enumerate(np.argsort(-similarities, kind="stable")[:CANDIDATES], 1):
        cid = index.chunks[n]["chunk_id"]
        scores[cid] = scores.get(cid, 0) + SEMANTIC_WEIGHT / (RRF_K + rank)
    for rank, cid in enumerate(lexical_hits(index.folder, question, CANDIDATES), 1):
        if cid in by_id:
            scores[cid] = scores.get(cid, 0) + LEXICAL_WEIGHT / (RRF_K + rank)
    ranked = sorted(scores, key=lambda cid: -scores[cid])
    temporal = current_time is not None and any(p in normalize_search(question) for p in TEMPORAL)
    order: list[str] = []
    if temporal:
        current = segment_at(index.segments, current_time)
        if current:
            order.append(current["segment_id"])
    for cid in ranked:
        sid = index.chunks[by_id[cid]]["segment_id"]
        if sid not in order:
            order.append(sid)
        if len(order) >= MAX_SLIDES:
            break
    slides = []
    for rank, sid in enumerate(order):
        segment = index.by_segment.get(sid, {})
        chunks = [c for c in index.chunks if c["segment_id"] == sid]
        visual = segment.get("visual", {})
        speech = segment.get("speech", {}).get("text") or " ".join(c["text"] for c in chunks if c["source_type"] == "speech")
        ocr = visual.get("ocr_text_clean") or " ".join(c["text"] for c in chunks if c["source_type"] == "ocr")
        figures = [c["text"] for c in chunks if c["source_type"] == "region"]
        image = visual.get("image_path") or (chunks[0].get("image_path") if chunks else None)
        best = max((float(similarities[by_id[c["chunk_id"]]]) for c in chunks), default=0.0)
        slides.append({"segment_id": sid, "rank": rank,
                       "start_sec": segment.get("start_sec", chunks[0]["start_sec"] if chunks else 0),
                       "end_sec": segment.get("end_sec", chunks[-1]["end_sec"] if chunks else 0),
                       "title": visual.get("title_clean", visual.get("title")) or (chunks[0].get("title") if chunks else ""),
                       "ocr_text": ocr, "speech_text": speech, "figures": figures,
                       "image": str(index.folder / "lecture" / image) if image else None,
                       "semantic_score": round(best, 4)})
    return {"slides": slides, "temporal": temporal,
            "best_semantic": round(float(similarities.max()), 4) if len(similarities) else 0.0}


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def build_context(found: dict, speech_chars: int = 2500, ocr_chars: int = 1800) -> str:
    """Evidence text for the answer model, slide by slide in time order."""
    blocks = []
    for slide in sorted(found["slides"], key=lambda s: s["start_sec"]):
        parts = [f"[Slide {slide['segment_id'].split(':')[-1]} · {clock(slide['start_sec'])}–{clock(slide['end_sec'])}]"]
        if slide["title"]:
            parts.append(f"Tiêu đề: {slide['title']}")
        if slide["ocr_text"]:
            parts.append("Chữ trên slide (OCR, có thể sai số/dấu — đối chiếu ảnh nếu có):\n" + slide["ocr_text"][:ocr_chars])
        for figure in slide["figures"][:3]:
            parts.append("Hình/bảng/công thức: " + figure[:600])
        if slide["speech_text"]:
            parts.append("Lời giảng:\n" + slide["speech_text"][:speech_chars])
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)
