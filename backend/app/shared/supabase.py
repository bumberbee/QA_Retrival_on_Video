"""Read/write of the app data in Supabase (used when SUPABASE_DB_URL is set).

Same shapes as the file storage, so storage.py, retrieval.py and sessions.py only switch the
source: segments come back as the stage-04 segment dicts (kept whole in segments.extra), chunks
with their E5 vectors so search scores exactly as with the files. Files live in R2 (objects.py).
"""

from __future__ import annotations

import json
from app.shared.models import Transcript, TranscriptSegment, VideoInfo
from app.shared import db


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def list_videos() -> list[VideoInfo]:
    with db.connection() as conn:
        rows = conn.execute(
            "select id, filename, duration_sec, uploaded_at, source_url, source_platform, has_transcript, "
            "has_slides from videos where status = 'ready' order by uploaded_at desc").fetchall()
    return [VideoInfo(video_id=row["id"], filename=row["filename"], has_transcript=row["has_transcript"],
                      has_slides=row["has_slides"], duration_seconds=row["duration_sec"],
                      uploaded_at=_iso(row["uploaded_at"]), source_url=row["source_url"],
                      source_platform=row["source_platform"]) for row in rows]


def video_meta(video_id: str) -> dict | None:
    with db.connection() as conn:
        row = conn.execute("select * from videos where id = %s", (video_id,)).fetchone()
    if not row:
        return None
    return {"original_filename": row["filename"], "uploaded_at": _iso(row["uploaded_at"]),
            "duration_seconds": row["duration_sec"], "source_url": row["source_url"],
            "source_platform": row["source_platform"], "video_key": row["video_key"],
            "language": row["language"], "meta": row["meta"] or {}}


def find_id_by_filename(filename: str) -> str | None:
    with db.connection() as conn:
        row = conn.execute("select id from videos where filename = %s limit 1", (filename,)).fetchone()
    return row["id"] if row else None


def transcript(video_id: str) -> Transcript | None:
    with db.connection() as conn:
        rows = conn.execute("select start_sec, end_sec, text from transcript_segments where video_id = %s "
                            "order by idx", (video_id,)).fetchall()
        language = conn.execute("select language from videos where id = %s", (video_id,)).fetchone()
    if not rows:
        return None
    return Transcript(video_id=video_id, language=(language or {}).get("language"),
                      segments=[TranscriptSegment(start=r["start_sec"], end=r["end_sec"], text=r["text"]) for r in rows])


def keyframe_keys(video_id: str) -> dict[str, str]:
    """Keyframe path (relative to the video's keyframe folder, e.g. seg_00003.jpg or fast/s_0042.jpg)
    -> R2 key; sync_supabase.py records the list in videos.meta.keyframes."""
    names = ((video_meta(video_id) or {}).get("meta") or {}).get("keyframes") or []
    return {name: f"keyframes/{video_id}/{name}" for name in names}


def slide_variants(video_id: str) -> dict | None:
    meta = video_meta(video_id)
    return (meta or {}).get("meta", {}).get("slide_variants")


def slide_reviews(video_id: str) -> dict:
    with db.connection() as conn:
        rows = conn.execute("select variant, slide_key, verdict, note from slide_reviews where video_id = %s",
                            (video_id,)).fetchall()
    reviews: dict = {}
    for row in rows:
        reviews.setdefault(row["variant"], {})[row["slide_key"]] = {"verdict": row["verdict"], "note": row["note"]}
    return reviews


def save_slide_reviews(video_id: str, reviews: dict) -> None:
    with db.connection() as conn, conn.transaction():
        conn.execute("delete from slide_reviews where video_id = %s", (video_id,))
        for variant, entries in reviews.items():
            for key, review in entries.items():
                conn.execute("insert into slide_reviews (video_id, variant, slide_key, verdict, note) "
                             "values (%s, %s, %s, %s, %s)",
                             (video_id, variant, key, review.get("verdict"), review.get("note") or ""))


# ---- search index ----

def has_index(video_id: str) -> bool:
    with db.connection() as conn:
        return conn.execute("select 1 from chunks where video_id = %s limit 1", (video_id,)).fetchone() is not None


def load_index(video_id: str):
    """(chunks, vectors, segments) in the shapes retrieval.load_index builds from the files."""
    import numpy as np

    with db.connection() as conn:
        rows = conn.execute(
            "select id, segment_id, source_type, start_sec, end_sec, title, text, embedding::text as embedding "
            "from chunks where video_id = %s order by id", (video_id,)).fetchall()
        segment_rows = conn.execute("select extra from segments where video_id = %s order by seq",
                                    (video_id,)).fetchall()
    chunks = [{"chunk_id": r["id"], "segment_id": r["segment_id"], "source_type": r["source_type"],
               "start_sec": r["start_sec"], "end_sec": r["end_sec"], "title": r["title"], "text": r["text"],
               "video_id": video_id} for r in rows]
    vectors = (np.array([json.loads(r["embedding"]) for r in rows], dtype=np.float32)
               if rows else np.zeros((0, 384), dtype=np.float32))
    return chunks, vectors, [r["extra"] for r in segment_rows]


def lexical_hits(video_id: str, terms: list[str], count: int, phrases=()) -> list[str]:
    """Chunk ids matching any term, best ts_rank first (terms are already lower-case, unaccented)."""
    words = [term for term in terms if term.isalnum()][:64]
    if not words:
        return []
    query = " | ".join(words)
    pairs = [(a, b) for a, b in phrases if a in words and b in words and a.isalnum() and b.isalnum()]
    if pairs:
        query += " | " + " | ".join(f"({a} <-> {b})" for a, b in pairs)
    with db.connection() as conn:
        rows = conn.execute(
            "select id from chunks where video_id = %s and tsv @@ to_tsquery('simple', %s) "
            "order by ts_rank(tsv, to_tsquery('simple', %s)) desc limit %s",
            (video_id, query, query, count)).fetchall()
    return [row["id"] for row in rows]


# ---- conversations ----

def conversation(conversation_id: str) -> dict | None:
    with db.connection() as conn:
        row = conn.execute("select doc from conversations where id = %s", (conversation_id,)).fetchone()
    return row["doc"] if row else None


def save_conversation(doc: dict) -> None:
    with db.connection() as conn:
        conn.execute(
            "insert into conversations (id, client_id, doc, updated_at) values (%s, %s, %s, now()) "
            "on conflict (id) do update set doc = excluded.doc, updated_at = now()",
            (doc["id"], doc["client_id"], json.dumps(doc, ensure_ascii=False)))


def conversations_of(client_id: str) -> list[dict]:
    with db.connection() as conn:
        rows = conn.execute("select doc from conversations where client_id = %s order by updated_at desc",
                            (client_id,)).fetchall()
    return [row["doc"] for row in rows]


def delete_conversation(conversation_id: str) -> None:
    with db.connection() as conn:
        conn.execute("delete from conversations where id = %s", (conversation_id,))
