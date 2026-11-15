"""Evidence search over a video's stage-05 index (pipeline), for the Q&A endpoint.

Same method as pipeline/stages/s06_index/retrieval.py, without PyTorch: the multilingual-E5 query encoder
runs as ONNX (onnxruntime + tokenizers, pinned to the revision the index was built with), the
keyword side is the index's SQLite FTS5 table. Meaning counts more than keywords (weighted
reciprocal rank fusion) because ASR spelling errors and common words make BM25 noisy.

Evidence is grouped per slide: its title, cleaned OCR text, lecture speech, time span and
keyframe image, so the answer model can also read numbers from the slide image itself.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import unicodedata
from contextlib import closing
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from app.shared import storage
from .rerank import rerank

INDEX_DIR_NAME = "index"
E5_REPO = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SEMANTIC_WEIGHT, LEXICAL_WEIGHT, RRF_K = 1.0, 0.5, 60
CANDIDATES, MAX_SLIDES, MAX_IMAGES = 20, 3, 2
RERANK = os.environ.get("LECTURE_RERANK", "off").strip().lower() in {"1", "true", "on", "yes"}
RERANK_CANDIDATES = 30
# Legacy boost retained for ablation; disabled after query-time OCR grouping (eval/PHASE2.md).
SOURCE_BOOST = {"ocr": 1.05, "region": 1.05}
# Independent query-time ablations; no index files are changed.
RANKED_CONTEXT = True
NEIGHBOR_EXPANSION = True
VIETNAMESE_LEXICAL = True
LEXICAL_PHRASES = True
OCR_DEDUP = True
SLIDE_DIVERSITY = True
APPLY_SOURCE_BOOST = False
MAX_CONTEXT_CHARS = 8000
NEIGHBOR_GAP_SEC = 2.0


@dataclass(frozen=True)
class RetrievalOptions:
    ranked_context: bool = RANKED_CONTEXT
    neighbors: bool = NEIGHBOR_EXPANSION
    vietnamese_lexical: bool = VIETNAMESE_LEXICAL
    phrases: bool = LEXICAL_PHRASES
    dedup: bool = OCR_DEDUP
    diversity: bool = SLIDE_DIVERSITY
    source_boost: bool = APPLY_SOURCE_BOOST


def retrieval_options(features=None):
    if features is None:
        return RetrievalOptions()
    return RetrievalOptions(**{key: (key in features) if key != "source_boost" else "no_boost" not in features
                               for key in asdict(RetrievalOptions())})


# A follow-up usually asks about the slides just cited: their chunks get this extra factor.
FOLLOW_UP_BOOST = 1.1
# "nhận định 14", "câu 2", "bài 3": embeddings cannot tell numbered items apart, so chunks naming
# exactly that item are lifted above every RRF score (RRF scores are < 0.05). LECTURE_ORDINAL_BOOST=off disables.
ORDINAL_BOOST = (os.environ.get("LECTURE_ORDINAL_BOOST", "on").strip().lower() not in {"0", "false", "off", "no"})
ORDINAL_BONUS = 1.0
ORDINAL_WORDS = (r"nhan dinh|cau hoi|cau|bai tap|bai|phan|vi du|dinh ly|dinh nghia|chuong|muc|buoc|"
                 r"slide|trang|hinh|question|exercise|example|part|section|chapter|step|lesson")
ORDINAL_RE = re.compile(rf"\b({ORDINAL_WORDS})\s+(?:so\s+|thu\s+)?0*(\d{{1,3}})\b")


def ordinal_refs(question: str) -> list[tuple[str, str]]:
    """Numbered items named in the question, e.g. [("nhan dinh", "14")]."""
    return ORDINAL_RE.findall(normalize_search(question))


def mentions_ordinal(text: str, refs: list[tuple[str, str]]) -> bool:
    normal = normalize_search(text)
    return any(re.search(rf"\b{re.escape(word)}\s+(?:so\s+|thu\s+)?0*{number}(?![\d.,]*\d)", normal)
               for word, number in refs)
STOPWORDS = set("""a ai anh ay ba bang bao bi boi cac cach cai can cho chi chua chung co con cua cung da dang
de den deu di do doi duoc gi gio ha hay hon khi khong la lai lam len luc ma minh mot moi muon nao nay
neu ngay nhieu nhu nhung noi o phai qua ra rang rat roi sao se sau tai the thi thoi theo thu tren tro
trong tu va van ve vi voi vua xin what how why which when who is are of an to in on for and or do does""".split())
VI_STOPWORDS = set("""ai anh ấy bằng bao bị bởi các cái cần cho chưa chúng có còn của cũng đã đang
để đến đều đi đó được gì giờ hãy hơn khi không là lại làm mà mình một mỗi muốn nào này
nếu nhiều như những nói ở phải quá ra rằng rất rồi sao sẽ sau tại thì thôi theo trên
trong và vẫn về vì với vừa xin what how why which when who is are of an to in on for and or do does""".split())
TEMPORAL = ("slide nay", "doan nay", "luc nay", "cho nay", "hien tai", "this slide", "current slide")


def normalize_search(text: str) -> str:
    """Identical to pipeline/shared/common.py: the FTS5 index was built with it."""
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return "".join(char for char in text if not unicodedata.combining(char))


def index_dir(video_id: str) -> Path:
    return storage.DATA_DIR / INDEX_DIR_NAME / video_id


def has_index(video_id: str) -> bool:
    if storage.cloud():
        from app.shared import supabase as supa
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
        from app.shared import supabase as supa
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


def lexical_query(question, vietnamese=True, phrases=False):
    # Remove only unambiguous accented function words; bare syllables may be technical terms.
    original = re.findall(r"[^\W_]+", unicodedata.normalize("NFC", question.casefold()))
    words = [normalize_search(word) for word in original]
    keep = [word not in VI_STOPWORDS if vietnamese else folded not in STOPWORDS
            for word, folded in zip(original, words)]
    terms = list(dict.fromkeys(word for word, allowed in zip(words, keep) if allowed))[:64]
    if not vietnamese and not terms:
        terms = list(dict.fromkeys(words))[:64]
    pairs = list(dict.fromkeys((words[i], words[i + 1]) for i in range(len(words) - 1)
                              if keep[i] and keep[i + 1] and words[i] in terms and words[i + 1] in terms))[:32] if phrases else []
    return terms, pairs


def lexical_hits(folder: Path | None, question: str, count: int, video_id: str = "",
                 options: RetrievalOptions | None = None) -> list[str]:
    options = options or RetrievalOptions()
    terms, phrases = lexical_query(question, options.vietnamese_lexical, options.phrases)
    if not terms:
        return []
    if folder is None:   # Supabase: the same terms against the chunks' unaccented tsvector
        from app.shared import supabase as supa
        return supa.lexical_hits(video_id, terms, count, phrases=phrases)
    expression = " OR ".join(f'"{term}"' for term in terms)
    if phrases:
        expression += " OR " + " OR ".join(f'"{a} {b}"' for a, b in phrases)
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


def content_groups(index: LectureIndex):
    """Conservative OCR groups; changed numbers are never treated as near duplicates."""
    if hasattr(index, "_content_groups"):
        return index._content_groups
    groups, exact, representatives = {}, {}, []
    for chunk in index.chunks:
        cid = chunk["chunk_id"]
        if chunk["source_type"] != "ocr":
            groups[cid] = cid
            continue
        normalized = normalize_search(chunk["text"])
        words = re.findall(r"[^\W_]+", normalized)
        identity = " ".join(normalized.split())
        numbers = tuple(re.findall(r"\d+(?:[.,]\d+)*|[=+*/−-]", normalized))
        shingles = {tuple(words[i:i + 3]) for i in range(len(words) - 2)}
        group = exact.get(identity)
        if group is None and len(words) >= 16:
            for other_id, other_numbers, other_shingles in representatives:
                if numbers == other_numbers and len(shingles & other_shingles) / max(1, len(shingles | other_shingles)) >= 0.9:
                    group = other_id
                    break
        if group is None:
            group = cid
            if len(words) >= 16:
                representatives.append((cid, numbers, shingles))
        exact[identity] = group
        groups[cid] = group
    index._content_groups = groups
    return groups


def context_candidates(index, sid, ranked, similarities, options, sources=None, question=""):
    chunks = [c for c in index.chunks if c["segment_id"] == sid
              and (sources is None or c["source_type"] in sources)]
    by_id = {c["chunk_id"]: c for c in index.chunks}
    positions = {c["chunk_id"]: i for i, c in enumerate(index.chunks)}
    matched = [by_id[cid] for cid in ranked if by_id[cid]["segment_id"] == sid]
    anchor = matched[0] if matched else (chunks[0] if chunks else None)
    if anchor is None:
        return [], []
    selected, seen = [], set()

    def add(chunk, reason, priority):
        if chunk["chunk_id"] in seen or (sources is not None and chunk["source_type"] not in sources):
            return
        seen.add(chunk["chunk_id"])
        segment = index.by_segment.get(chunk["segment_id"], {})
        selected.append({**chunk, "selection_reason": reason, "priority": priority,
                         "slide_start_sec": segment.get("start_sec", chunk["start_sec"]),
                         "slide_end_sec": segment.get("end_sec", chunk["end_sec"])})

    add(anchor, "matched", 0)
    for kind in ("ocr", "speech", "region"):
        candidates = [c for c in chunks if c["source_type"] == kind]
        if candidates:
            best = max(candidates, key=lambda c: float(similarities[positions[c["chunk_id"]]]))
            add(best, "same_slide", 1)
        elif kind in {"ocr", "speech"}:
            segment = index.by_segment.get(sid, {})
            text = (segment.get("visual", {}).get("ocr_text_clean") if kind == "ocr"
                    else segment.get("speech", {}).get("text"))
            if text:
                add({"chunk_id": sid + ":" + kind, "segment_id": sid, "source_type": kind,
                     "text": text, "start_sec": segment["start_sec"], "end_sec": segment["end_sec"]},
                    "same_slide", 1)
    for chunk in matched[1:]:
        add(chunk, "matched", 2)
    speech = sorted((c for c in chunks if c["source_type"] == "speech"), key=lambda c: c["start_sec"])
    seeds = [c for c in selected if c["source_type"] == "speech"][:2]
    for seed in seeds:
        position = next((i for i, c in enumerate(speech) if c["chunk_id"] == seed["chunk_id"]), None)
        if position is None:
            continue
        for n in (position - 1, position + 1):
            if 0 <= n < len(speech):
                add(speech[n], "speech_window", 3)
    if options.neighbors:
        segment = index.by_segment.get(sid, {})
        ordered = sorted(index.segments, key=lambda s: s["start_sec"])
        position = next((i for i, s in enumerate(ordered) if s["segment_id"] == sid), None)
        explain = any(p in normalize_search(question) for p in
                      ("giai thich", "cac buoc", "quy trinh", "the nao", "tai sao", "explain", "steps", "how", "why"))
        for delta in (-1, 1):
            boundary = segment.get("start_sec" if delta < 0 else "end_sec", anchor["start_sec"])
            edge = anchor["start_sec" if delta < 0 else "end_sec"]
            needed = explain or len(anchor["text"]) < 80 or abs(edge - boundary) <= NEIGHBOR_GAP_SEC
            if position is None or not needed or not 0 <= position + delta < len(ordered):
                continue
            neighbor = ordered[position + delta]
            gap = (segment.get("start_sec", 0) - neighbor["end_sec"] if delta < 0
                   else neighbor["start_sec"] - segment.get("end_sec", 0))
            if is_ad(neighbor) or gap > NEIGHBOR_GAP_SEC:
                continue
            adjacent = [c for c in index.chunks if c["segment_id"] == neighbor["segment_id"]]
            narration = sorted((c for c in adjacent if c["source_type"] == "speech"),
                               key=lambda c: c["start_sec"], reverse=delta < 0)
            for chunk in narration[:1]:
                # A visual boundary alone does not justify jumping over a long silent gap.
                speech_gap = (anchor["start_sec"] - chunk["end_sec"] if delta < 0
                              else chunk["start_sec"] - anchor["end_sec"])
                if explain or anchor["source_type"] != "speech" or speech_gap <= NEIGHBOR_GAP_SEC:
                    add(chunk, "previous" if delta < 0 else "next", 4)
    return matched, selected


def search(video_id: str, question: str, current_time: float | None = None, encoder=None,
           sources: frozenset | None = None, prefer_segments: frozenset = frozenset(),
           options: RetrievalOptions | None = None, *, rerank_scorer=None) -> dict:
    """Ranked chunks fused per slide -> {'slides': [...], 'best_semantic': float, 'temporal': bool}.
    sources limits search and evidence to some chunk types ("speech", "ocr", "region") for ablations.
    LECTURE_RERANK=on reranks the top 30 RRF candidates before selecting slides.
    Scored pools add best_rerank and chunk-level scores; unavailable models raise.
    rerank_scorer injects a logit scorer without loading a model.
    """
    import numpy as np

    index = load_index(video_id)
    options = options or RetrievalOptions()
    temporal = current_time is not None and any(p in normalize_search(question) for p in TEMPORAL)
    current = segment_at(index.segments, current_time) if temporal else None
    vector = (encoder or default_encoder(index.model)).encode(question)
    similarities = index.vectors @ vector
    if sources is not None:
        allowed = np.array([c["source_type"] in sources for c in index.chunks], dtype=bool)
        similarities = np.where(allowed, similarities, -1.0)
    by_id = {chunk["chunk_id"]: n for n, chunk in enumerate(index.chunks)}
    kinds = [c["source_type"] for c in index.chunks]
    scores: dict[str, float] = {}
    groups = content_groups(index) if options.dedup else {cid: cid for cid in by_id}
    representatives = {}
    # Group before taking the twenty candidates, so copies cannot consume the pool.
    for n in np.argsort(-similarities, kind="stable"):
        cid = index.chunks[n]["chunk_id"]
        if sources is None or kinds[n] in sources:
            representatives.setdefault(groups[cid], cid)
    if options.dedup and (prefer_segments or current):
        def preference(candidate):
            sid = index.chunks[by_id[candidate]]["segment_id"]
            return (bool(current and sid == current["segment_id"]), sid in prefer_segments,
                    float(similarities[by_id[candidate]]))

        for cid in by_id:
            group = groups[cid]
            if group not in representatives:
                continue
            previous = representatives[group]
            if preference(cid) > preference(previous):
                representatives[group] = cid
    semantic = list(representatives.values())[:CANDIDATES]
    for rank, cid in enumerate(semantic, 1):
        scores[cid] = scores.get(cid, 0) + SEMANTIC_WEIGHT / (RRF_K + rank)
    lexical = lexical_hits(index.folder, question, len(index.chunks) if options.dedup else
                           (CANDIDATES if sources is None else CANDIDATES * 4), video_id, options)
    lexical = [cid for cid in lexical if cid in by_id and (sources is None or kinds[by_id[cid]] in sources)]
    if options.dedup:
        lexical = list(dict.fromkeys(representatives[groups[cid]] for cid in lexical))
    for rank, cid in enumerate(lexical[:CANDIDATES], 1):
        scores[cid] = scores.get(cid, 0) + LEXICAL_WEIGHT / (RRF_K + rank)
    for cid in scores:
        if options.source_boost:
            scores[cid] *= SOURCE_BOOST.get(kinds[by_id[cid]], 1.0)
        if prefer_segments and any(c["segment_id"] in prefer_segments for c in index.chunks
                                   if groups[c["chunk_id"]] == groups[cid]):
            scores[cid] *= FOLLOW_UP_BOOST
    refs = ordinal_refs(question) if ORDINAL_BOOST else []
    if refs:
        for n, chunk in enumerate(index.chunks):
            if (sources is None or kinds[n] in sources) and mentions_ordinal(chunk["text"], refs):
                cid = representatives.get(groups[chunk["chunk_id"]], chunk["chunk_id"])
                scores[cid] = scores.get(cid, 0.0) + ORDINAL_BONUS
    ranked = sorted(scores, key=lambda cid: -scores[cid])
    reranked = {}
    if RERANK:
        candidates = [index.chunks[by_id[cid]] for cid in ranked[:RERANK_CANDIDATES]]
        rescored = rerank(question, candidates, scorer=rerank_scorer)
        reranked = {c["chunk_id"]: c for c in rescored}
        ranked = list(reranked)
    order: list[str] = []
    if temporal:
        if current:
            order.append(current["segment_id"])
    diversity_groups = content_groups(index) if options.diversity else groups
    used_groups = {diversity_groups[cid] for cid in ranked if index.chunks[by_id[cid]]["segment_id"] in order}
    for cid in ranked:
        sid = index.chunks[by_id[cid]]["segment_id"]
        if options.diversity and diversity_groups[cid] in used_groups and sid not in order:
            continue
        if sid not in order:
            order.append(sid)
            used_groups.add(diversity_groups[cid])
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
        matched, evidence = context_candidates(index, sid, ranked, similarities, options, sources, question)
        for chunk in evidence:
            scored = reranked.get(chunk["chunk_id"])
            if scored is not None:
                chunk.update(rerank_score=scored["rerank_score"], rerank_rank=scored["rerank_rank"])
        if options.dedup:
            for chunk in evidence:
                chunk["occurrences"] = [{"chunk_id": c["chunk_id"], "segment_id": c["segment_id"],
                                         "start_sec": c["start_sec"], "end_sec": c["end_sec"]}
                                        for c in index.chunks if groups[c["chunk_id"]] == groups.get(chunk["chunk_id"])]
        slides.append({"segment_id": sid, "rank": rank,
                       "start_sec": segment.get("start_sec", chunks[0]["start_sec"] if chunks else 0),
                       "end_sec": segment.get("end_sec", chunks[-1]["end_sec"] if chunks else 0),
                       "title": visual.get("title_clean", visual.get("title")) or (chunks[0].get("title") if chunks else ""),
                       "ocr_text": ocr, "speech_text": speech, "figures": figures,
                       "image": str(index.folder / "lecture" / image) if image and index.folder else None,
                       "semantic_score": round(best, 4), "score": round(fused, 6),
                       "video_id": video_id, "matched_chunk_ids": [c["chunk_id"] for c in matched],
                       "matched_spans": [{"chunk_id": c["chunk_id"], "start_sec": c["start_sec"],
                                          "end_sec": c["end_sec"]} for c in matched],
                       "evidence_candidates": evidence})
    return {"slides": slides, "temporal": temporal,
            "question": question, "retrieval_options": asdict(options),
            "best_semantic": round(float(similarities.max()), 4) if len(similarities) else 0.0,
            **({"best_rerank": next(iter(reranked.values()))["rerank_score"]} if reranked else {})}


def video_title(video_id: str) -> str:
    meta = storage.load_video_meta(video_id) or {}
    return Path(meta.get("original_filename") or video_id).stem


def search_many(video_ids: list, question: str, current_video: str | None = None,
                current_time: float | None = None, max_slides: int = MAX_SLIDES + 1,
                prefer_segments: frozenset = frozenset(), options: RetrievalOptions | None = None) -> dict:
    """search() on several indexed videos. RRF scores come from per-video ranks (every video's
    top slide scores about the same), so slides are merged by E5 cosine, which all indexes share.
    A "this slide" match in the video being watched stays first."""
    results = [search(vid, question, current_time if vid == current_video else None,
                      prefer_segments=prefer_segments, options=options)
               for vid in dict.fromkeys(video_ids) if has_index(vid)]
    if not results:
        return {"slides": [], "temporal": False, "best_semantic": 0.0}
    pinned = [r["slides"][0] for r in results if r["temporal"] and r["slides"]]
    rest = sorted((s for r in results for s in r["slides"] if s not in pinned), key=lambda s: (-s["semantic_score"], -s["score"]))
    slides = (pinned + rest)[:max_slides]
    for slide in slides:
        slide["video_title"] = video_title(slide["video_id"])
    found = {"slides": slides, "temporal": bool(pinned),
             "best_semantic": max(r["best_semantic"] for r in results), "videos": len(results),
             "question": question, "retrieval_options": asdict(options or RetrievalOptions())}
    if any("best_rerank" in r for r in results):
        found["best_rerank"] = max(r["best_rerank"] for r in results if "best_rerank" in r)
    return found


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _evidence_source(item: dict, kind: str) -> dict:
    """Snapshot the cited span, never a deduplicated occurrence or the parent slide span."""
    chunk_ids = [item["chunk_id"]] if "chunk_id" in item else list(item.get("matched_chunk_ids", []))
    return {"video_id": item.get("video_id", ""), "t_start": item["start_sec"],
            "t_end": item["end_sec"], "slide_id": item["segment_id"],
            "segment_id": item["segment_id"], "chunk_ids": chunk_ids, "kind": kind,
            "title": item.get("title") or "", "video_title": item.get("video_title", "")}


def _legacy_context(found: dict, speech_chars: int = 2500, ocr_chars: int = 1800,
                    *, evidence_map: dict | None = None, max_chars: int | None = None) -> str:
    """Evidence text for the answer model, slide by slide in time order."""
    blocks = []
    several = len({slide.get("video_id") for slide in found["slides"]}) > 1
    for slide in sorted(found["slides"], key=lambda s: (s.get("video_title", ""), s["start_sec"])):
        where = f"Video {slide.get('video_title', '')} · " if several else ""
        parts = [f"[{where}Slide {slide['segment_id'].split(':')[-1]} · {clock(slide['start_sec'])}–{clock(slide['end_sec'])}]"]
        if slide["title"]:
            parts.append(f"Tiêu đề: {slide['title']}")
        if slide["ocr_text"]:
            note = ("Chữ trên slide (OCR, có thể sai số/dấu):\n" if evidence_map is not None else
                    "Chữ trên slide (OCR, có thể sai số/dấu — đối chiếu ảnh nếu có):\n")
            parts.append(note + slide["ocr_text"][:ocr_chars])
        for figure in slide["figures"][:3]:
            parts.append("Hình/bảng/công thức: " + figure[:600])
        if slide["speech_text"]:
            parts.append("Lời giảng:\n" + slide["speech_text"][:speech_chars])
        block = "\n".join(parts)
        if evidence_map is not None:
            evidence_id = f"E{len(evidence_map) + 1}"
            block = f"[{evidence_id}] " + block
            if max_chars is not None:
                remaining = max_chars - sum(len(b) for b in blocks) - 2 * len(blocks)
                # Never send a partial ID/header or map an entirely omitted block.
                if remaining <= len(block.split("\n", 1)[0]):
                    break
                block = block[:remaining]
            evidence_map[evidence_id] = _evidence_source(slide, "slide")
        blocks.append(block)
    return "\n\n".join(blocks)


def evidence_block(chunk, several=False, evidence_id: str | None = None):
    where = f"Video {chunk.get('video_title', chunk.get('video_id', ''))} · " if several else ""
    label = {"speech": "Lời giảng", "ocr": "Chữ trên slide (OCR)", "region": "Hình/bảng/công thức"}[chunk["source_type"]]
    return ((f"[{evidence_id}] " if evidence_id else "")
            + f"[{where}Slide {chunk['segment_id'].split(':')[-1]} · "
            f"{clock(chunk['start_sec'])}–{clock(chunk['end_sec'])}] {label}\n"
            + (f"Tiêu đề: {chunk['title']}\n" if chunk.get("title") else "")
            + chunk["text"])


def build_context(found: dict, speech_chars: int = 2500, ocr_chars: int = 1800,
                  max_chars: int = MAX_CONTEXT_CHARS, options: RetrievalOptions | None = None) -> str:
    """Backward-compatible context without evidence IDs."""
    return _build_context(found, speech_chars, ocr_chars, max_chars, options)


def build_context_with_ids(found: dict, speech_chars: int = 2500, ocr_chars: int = 1800,
                           max_chars: int = MAX_CONTEXT_CHARS,
                           options: RetrievalOptions | None = None) -> tuple[str, dict]:
    """Assign request-local immutable IDs in final context order, within the text budget."""
    if max_chars < 1:
        raise ValueError("Context budget must be positive.")
    evidence_map = {}
    context = _build_context(found, speech_chars, ocr_chars, max_chars, options, evidence_map=evidence_map)
    found["evidence_map"] = evidence_map
    found["context_chars"] = len(context)
    return context, evidence_map


def _build_context(found: dict, speech_chars: int, ocr_chars: int, max_chars: int,
                   options: RetrievalOptions | None, *, evidence_map: dict | None = None) -> str:
    """Ranked evidence first, then bounded speech windows and conditional neighbors.

    Slide intervals stay in the response for the UI; each text block cites its own span.
    context_evidence records only text actually sent, for evaluation and debugging.
    """
    options = options or RetrievalOptions(**found.get("retrieval_options", {}))
    if not (options.ranked_context or options.neighbors) or not any("evidence_candidates" in s for s in found["slides"]):
        return _legacy_context(found, speech_chars, ocr_chars, evidence_map=evidence_map,
                               max_chars=max_chars if evidence_map is not None else None)
    if max_chars < 1:
        raise ValueError("Context budget must be positive.")
    several = len({s.get("video_id") for s in found["slides"]}) > 1
    candidates = []
    for rank, slide in enumerate(found["slides"]):
        for chunk in slide.get("evidence_candidates", []):
            if options.ranked_context or chunk["selection_reason"] in {"previous", "next"}:
                candidates.append({**chunk, "video_id": slide.get("video_id", ""),
                                   "video_title": slide.get("video_title", ""), "slide_rank": rank})
    candidates.sort(key=lambda c: (c["priority"], c["slide_rank"]))
    selected, seen, usage = [], set(), {}
    legacy = "" if options.ranked_context else _legacy_context(
        found, speech_chars, ocr_chars, evidence_map=evidence_map,
        max_chars=max_chars if evidence_map is not None else None)[:max_chars]
    context = legacy
    for chunk in candidates:
        kind = chunk["source_type"]
        identity = (chunk.get("video_id"), kind, normalize_search(chunk["text"]).strip())
        if identity in seen:
            continue
        key = (chunk.get("video_id"), chunk["segment_id"], kind)
        limit = speech_chars if kind == "speech" else ocr_chars if kind == "ocr" else 1800
        if usage.get(key, 0) + len(chunk["text"]) > limit:
            continue
        provisional_id = f"E{len(evidence_map) + len(selected) + 1}" if evidence_map is not None else None
        block = evidence_block(chunk, several, provisional_id)
        proposed = context + ("\n\n" if context else "") + block
        if len(proposed) > max_chars:
            continue
        context = proposed
        selected.append(chunk)
        seen.add(identity)
        usage[key] = usage.get(key, 0) + len(chunk["text"])
    selected.sort(key=lambda c: (c.get("video_title", ""), c.get("video_id", ""), c["start_sec"], c["end_sec"]))
    blocks = [legacy] if legacy else []
    for chunk in selected:
        evidence_id = None
        if evidence_map is not None:
            evidence_id = f"E{len(evidence_map) + 1}"
            evidence_map[evidence_id] = _evidence_source(chunk, chunk["source_type"])
            chunk["evidence_id"] = evidence_id
        blocks.append(evidence_block(chunk, several, evidence_id))
    context = "\n\n".join(blocks)
    found["context_evidence"] = selected
    found["context_legacy"] = legacy
    found["context_chars"] = len(context)
    return context
