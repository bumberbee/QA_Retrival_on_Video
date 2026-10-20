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
# Slide text gets a small edge over speech, which has ~2x more chunks and otherwise wins rank 1 for
# answers only written on the slide. 66 questions / 6 videos: hit@1 50 -> 55, hit@3 64 -> 65
# (1.1-1.5, ranking speech and slide separately, or summing chunk scores per slide all did worse).
SOURCE_BOOST = {"ocr": 1.05, "region": 1.05}
# A follow-up usually asks about the slides just cited: their chunks get this extra factor.
FOLLOW_UP_BOOST = 1.1
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
    if storage.cloud():
        from . import supa
        return supa.has_index(video_id)
    folder = index_dir(video_id)
    return all((folder / name).exists() for name in ("chunks.jsonl", "embeddings.npy", "lexical.sqlite3"))


# Query encoders by the embedding model named in an index's manifest (stage 05 or reembed_index.py).
ENCODERS = {
    "intfloat/multilingual-e5-small": {"revision": E5_REVISION, "files": ("onnx/model.onnx",),
                                       "tokenizer": "onnx/tokenizer.json", "prefix": "query: ", "pooling": "mean"},
    "BAAI/bge-m3": {"revision": "5617a9f61b028005a4858fdac845db406aefb181",
                    "files": ("onnx/model.onnx", "onnx/model.onnx_data", "onnx/Constant_7_attr__value"),
                    "tokenizer": "onnx/tokenizer.json", "prefix": "", "pooling": "cls"},
}


class OnnxEncoder:
    """Text encoder in ONNX (no PyTorch), normalized like sentence-transformers: E5 = mean pooling
    with the "query: " prefix, bge-m3 = CLS pooling (its dense vector) without prefix."""

    def __init__(self, repo: str = E5_REPO, max_tokens: int = 512):
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        spec = ENCODERS[repo]
        paths = [hf_hub_download(repo, name, revision=spec["revision"]) for name in spec["files"]]
        tokenizer = hf_hub_download(repo, spec["tokenizer"], revision=spec["revision"])
        self.session = onnxruntime.InferenceSession(paths[0], providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tokenizer)
        self.tokenizer.enable_truncation(max_tokens)
        self.prefix, self.pooling = spec["prefix"], spec["pooling"]
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.outputs = [item.name for item in self.session.get_outputs()]
        self.lock = threading.Lock()

    def encode_many(self, texts: list, prefix: str | None = None):
        """L2-normalized vectors of texts (padded batch); prefix defaults to the query prefix."""
        import numpy as np

        prefix = self.prefix if prefix is None else prefix
        encodings = [self.tokenizer.encode(prefix + text) for text in texts]
        width = max(len(e.ids) for e in encodings)
        ids = np.zeros((len(texts), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        pad = self.tokenizer.token_to_id("<pad>") or 0
        ids[:] = pad
        for row, encoding in enumerate(encodings):
            ids[row, :len(encoding.ids)] = encoding.ids
            mask[row, :len(encoding.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        with self.lock:
            outputs = dict(zip(self.outputs, self.session.run(None, feed)))
        hidden = outputs.get("last_hidden_state", outputs.get("token_embeddings", next(iter(outputs.values()))))
        if "sentence_embedding" in outputs:        # bge-m3's export pools (CLS) itself
            pooled = outputs["sentence_embedding"]
        elif self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
        return pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12)

    def encode(self, text: str):
        return self.encode_many([text])[0]


OnnxE5 = OnnxEncoder   # the old name, E5 by default


@lru_cache(maxsize=2)
def default_encoder(repo: str = E5_REPO) -> OnnxEncoder:
    return OnnxEncoder(repo)


@dataclass
class LectureIndex:
    folder: Path
    chunks: list
    vectors: object
    segments: list
    by_segment: dict = field(default_factory=dict)
    model: str = E5_REPO
    video_id: str = ""   # set for Supabase indexes (folder is None)


@lru_cache(maxsize=16)
def load_index(video_id: str) -> LectureIndex:
    import numpy as np

    folder = index_dir(video_id)
    if storage.cloud():
        from . import supa
        chunks, vectors, segments = supa.load_index(video_id)
        return LectureIndex(None, chunks, vectors, segments, {s["segment_id"]: s for s in segments}, E5_REPO,
                            video_id)
    chunks = [json.loads(line) for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    vectors = np.load(folder / "embeddings.npy", allow_pickle=False)
    if len(vectors) != len(chunks):
        raise ValueError("Chỉ mục hỏng: số vector khác số đoạn.")
    segments_file = folder / "lecture" / "segments.json"
    segments = json.loads(segments_file.read_text(encoding="utf-8")) if segments_file.exists() else []
    manifest = folder / "manifest.json"
    model = E5_REPO
    if manifest.exists():
        model = json.loads(manifest.read_text(encoding="utf-8")).get("parameters", {}).get("embedding_model", E5_REPO)
    return LectureIndex(folder, chunks, vectors, segments, {s["segment_id"]: s for s in segments},
                        model if model in ENCODERS else E5_REPO)


def lexical_hits(folder: Path | None, question: str, count: int, video_id: str = "") -> list[str]:
    words = list(dict.fromkeys(re.findall(r"[^\W_]+", normalize_search(question))))
    terms = [word for word in words if word not in STOPWORDS] or words
    if not terms:
        return []
    if folder is None:   # Supabase: the same terms against the chunks' unaccented tsvector
        from . import supa
        return supa.lexical_hits(video_id, terms, count)
    expression = " OR ".join(f'"{term}"' for term in terms[:64])
    with closing(sqlite3.connect((folder / "lexical.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return [row[0] for row in db.execute(
            "SELECT chunk_id FROM evidence WHERE evidence MATCH ? ORDER BY bm25(evidence) LIMIT ?",
            (expression, count))]


# Matched with diacritics: without them "đó" = "độ", "nó" = "nổ", "còn" = "con".
FOLLOW_UP_WORDS = {"nó", "đó", "kia", "ấy", "vậy", "còn", "thêm", "nữa", "it", "that", "those", "them", "they"}
FOLLOW_UP_STARTS = ("còn ", "thế còn ", "vậy ", "thế thì ", "tại sao vậy", "what about", "and ", "how about", "why is that")


def is_follow_up(question: str) -> bool:
    """A short question that leans on the previous one ("còn cái kia thì sao?", "nó là gì?")."""
    text = unicodedata.normalize("NFC", question).lower().strip()
    words = re.findall(r"[^\W_]+", text)
    if not words or len(words) > 10:
        return False
    return text.startswith(FOLLOW_UP_STARTS) or bool(FOLLOW_UP_WORDS & set(words))


def search_question(question: str, previous: str | None) -> str:
    """The text to search with: a follow-up question is searched together with the one before it."""
    if previous and previous.strip() and is_follow_up(question):
        return f"{previous.strip()} {question.strip()}"
    return question


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


def search(video_id: str, question: str, current_time: float | None = None, encoder=None,
           sources: frozenset | None = None, prefer_segments: frozenset = frozenset()) -> dict:
    """Ranked chunks fused per slide -> {'slides': [...], 'best_semantic': float, 'temporal': bool}.
    sources limits search and evidence to some chunk types ("speech", "ocr", "region") for ablations."""
    import numpy as np

    index = load_index(video_id)
    vector = (encoder or default_encoder(index.model)).encode(question)
    similarities = index.vectors @ vector
    if sources is not None:
        allowed = np.array([c["source_type"] in sources for c in index.chunks], dtype=bool)
        similarities = np.where(allowed, similarities, -1.0)
    by_id = {chunk["chunk_id"]: n for n, chunk in enumerate(index.chunks)}
    kinds = [c["source_type"] for c in index.chunks]
    scores: dict[str, float] = {}
    for rank, n in enumerate(np.argsort(-similarities, kind="stable")[:CANDIDATES], 1):
        cid = index.chunks[n]["chunk_id"]
        scores[cid] = scores.get(cid, 0) + SEMANTIC_WEIGHT / (RRF_K + rank)
    lexical = lexical_hits(index.folder, question, CANDIDATES if sources is None else CANDIDATES * 4, video_id)
    lexical = [cid for cid in lexical if cid in by_id and (sources is None or kinds[by_id[cid]] in sources)]
    for rank, cid in enumerate(lexical[:CANDIDATES], 1):
        scores[cid] = scores.get(cid, 0) + LEXICAL_WEIGHT / (RRF_K + rank)
    for cid in scores:
        scores[cid] *= SOURCE_BOOST.get(kinds[by_id[cid]], 1.0)
        if index.chunks[by_id[cid]]["segment_id"] in prefer_segments:
            scores[cid] *= FOLLOW_UP_BOOST
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
        if sources is not None:
            speech = speech if "speech" in sources else ""
            ocr = ocr if "ocr" in sources else ""
            figures = figures if "region" in sources else []
        image = visual.get("image_path") or (chunks[0].get("image_path") if chunks else None)
        best = max((float(similarities[by_id[c["chunk_id"]]]) for c in chunks), default=0.0)   # -1 = source left out
        fused = max((scores.get(c["chunk_id"], 0.0) for c in chunks), default=0.0)
        slides.append({"segment_id": sid, "rank": rank,
                       "start_sec": segment.get("start_sec", chunks[0]["start_sec"] if chunks else 0),
                       "end_sec": segment.get("end_sec", chunks[-1]["end_sec"] if chunks else 0),
                       "title": visual.get("title_clean", visual.get("title")) or (chunks[0].get("title") if chunks else ""),
                       "ocr_text": ocr, "speech_text": speech, "figures": figures,
                       "image": str(index.folder / "lecture" / image) if image and index.folder else None,
                       "semantic_score": round(best, 4), "score": round(fused, 6),
                       "video_id": video_id})
    return {"slides": slides, "temporal": temporal,
            "best_semantic": round(float(similarities.max()), 4) if len(similarities) else 0.0}


def video_title(video_id: str) -> str:
    meta = storage.load_video_meta(video_id) or {}
    return Path(meta.get("original_filename") or video_id).stem


def search_many(video_ids: list, question: str, current_video: str | None = None,
                current_time: float | None = None, max_slides: int = MAX_SLIDES + 1,
                prefer_segments: frozenset = frozenset()) -> dict:
    """search() on several indexed videos. RRF scores come from per-video ranks (every video's
    top slide scores about the same), so slides are merged by E5 cosine, which all indexes share.
    A "this slide" match in the video being watched stays first."""
    results = [search(vid, question, current_time if vid == current_video else None,
                      prefer_segments=prefer_segments)
               for vid in dict.fromkeys(video_ids) if has_index(vid)]
    if not results:
        return {"slides": [], "temporal": False, "best_semantic": 0.0}
    pinned = [r["slides"][0] for r in results if r["temporal"] and r["slides"]]
    rest = sorted((s for r in results for s in r["slides"] if s not in pinned), key=lambda s: (-s["semantic_score"], -s["score"]))
    slides = (pinned + rest)[:max_slides]
    for slide in slides:
        slide["video_title"] = video_title(slide["video_id"])
    return {"slides": slides, "temporal": bool(pinned),
            "best_semantic": max(r["best_semantic"] for r in results), "videos": len(results)}


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def build_context(found: dict, speech_chars: int = 2500, ocr_chars: int = 1800) -> str:
    """Evidence text for the answer model, slide by slide in time order."""
    blocks = []
    several = len({slide.get("video_id") for slide in found["slides"]}) > 1
    for slide in sorted(found["slides"], key=lambda s: (s.get("video_title", ""), s["start_sec"])):
        where = f"Video {slide.get('video_title', '')} · " if several else ""
        parts = [f"[{where}Slide {slide['segment_id'].split(':')[-1]} · {clock(slide['start_sec'])}–{clock(slide['end_sec'])}]"]
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
