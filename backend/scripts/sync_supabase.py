"""Push the local app data (after import_fast_pipeline.py) to Supabase + Cloudflare R2.

    cd backend
    .venv/Scripts/python scripts/sync_supabase.py --data D:/thesis-data/app-data          # every video
    .venv/Scripts/python scripts/sync_supabase.py --data <dir> --videos L25_V011 L25_V037  # some videos
    .venv/Scripts/python scripts/sync_supabase.py --data <dir> --skip-files                # rows only

Reads the file layout under --data (videos/, transcripts/, keyframes/, slide_variants/, index/) and
writes: videos, segments (the stage-04 segment kept whole in extra), transcript_segments and chunks
(with their stored E5 vectors, so search scores match the file version) into SUPABASE_DB_URL; the
mp4 and every keyframe to R2. Re-running replaces a video's rows; files already in R2 are skipped.
Run deploy/supabase/schema.sql once first.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from app.services import db, objects  # noqa: E402


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def upload(path: Path, key: str, skip_files: bool) -> None:
    if skip_files or objects.exists(key):
        return
    objects.upload_file(path, key)


def sync_video(data: Path, video_id: str, skip_files: bool) -> str:
    import numpy as np

    meta = read_json(data / "videos" / f"{video_id}.meta.json")
    if meta is None:
        return f"bỏ qua {video_id}: không có meta"
    filename = meta["original_filename"]
    video_file = next((p for p in (data / "videos").glob(f"{video_id}.*") if not p.name.endswith(".meta.json")), None)
    keyframe_root = data / "keyframes" / video_id
    keyframes = sorted(p.relative_to(keyframe_root).as_posix() for p in keyframe_root.rglob("*") if p.is_file()) \
        if keyframe_root.exists() else []
    index = data / "index" / video_id
    manifest = read_json(index / "manifest.json", {})
    language = "en" if "en" in str(manifest.get("parameters", {}).get("language", "")) else None
    transcript = read_json(data / "transcripts" / f"{video_id}.json")
    language = language or (transcript or {}).get("language") or "vi"
    variants = read_json(data / "slide_variants" / f"{video_id}.json")

    video_key = objects.video_key(video_id, filename) if video_file else None
    if video_file:
        upload(video_file, video_key, skip_files)
    for name in keyframes:
        upload(keyframe_root / name, f"keyframes/{video_id}/{name}", skip_files)

    segments = read_json(index / "lecture" / "segments.json", [])
    chunks = [json.loads(line) for line in (index / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
              if line] if (index / "chunks.jsonl").exists() else []
    vectors = np.load(index / "embeddings.npy") if chunks else None
    if chunks and len(vectors) != len(chunks):
        return f"lỗi {video_id}: {len(chunks)} chunk nhưng {len(vectors)} vector"

    with db.connection() as conn, conn.transaction():
        # cascades to every child table; an older import of the same file has the same segment ids
        conn.execute("delete from videos where id = %s or filename = %s", (video_id, filename))
        conn.execute(
            "insert into videos (id, filename, language, duration_sec, status, video_key, source_url, "
            "source_platform, has_transcript, has_slides, meta, uploaded_at) "
            "values (%s, %s, %s, %s, 'ready', %s, %s, %s, %s, %s, %s, coalesce(%s::timestamptz, now()))",
            (video_id, filename, language, meta.get("duration_seconds"), video_key, meta.get("source_url"),
             meta.get("source_platform"), transcript is not None, any("/" not in k for k in keyframes),
             json.dumps({"keyframes": keyframes, "slide_variants": variants}, ensure_ascii=False),
             meta.get("uploaded_at")))
        with conn.cursor() as cursor:
            cursor.executemany(
                "insert into segments (id, video_id, seq, start_sec, end_sec, kind, ad_flagged, title, ocr_text, "
                "speech_text, image_key, extra) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [(s["segment_id"], video_id, s.get("sequence", n), s.get("start_sec", 0), s.get("end_sec", 0),
                  s.get("content_kind") or "slide", bool(s.get("ad_flagged")),
                  (s.get("visual") or {}).get("title_clean") or (s.get("visual") or {}).get("title") or "",
                  (s.get("visual") or {}).get("ocr_text_clean") or "", (s.get("speech") or {}).get("text") or "",
                  None, json.dumps(s, ensure_ascii=False)) for n, s in enumerate(segments)])
            if transcript:
                cursor.executemany(
                    "insert into transcript_segments (video_id, idx, start_sec, end_sec, text) values (%s, %s, %s, %s, %s)",
                    [(video_id, n, seg["start"], seg["end"], seg["text"]) for n, seg in enumerate(transcript["segments"])])
            known = {s["segment_id"] for s in segments}
            cursor.executemany(
                "insert into chunks (id, video_id, segment_id, source_type, start_sec, end_sec, title, text, embedding) "
                "values (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
                [(c["chunk_id"], video_id, c["segment_id"], c["source_type"], c.get("start_sec", 0), c.get("end_sec", 0),
                  c.get("title") or "", c["text"], db.vector_literal(vectors[n]))
                 for n, c in enumerate(chunks) if c["segment_id"] in known])
    return (f"OK {filename} ({video_id}): {len(segments)} slide, {len(chunks)} chunk, "
            f"{len(transcript['segments']) if transcript else 0} đoạn lời, {len(keyframes)} ảnh"
            + ("" if skip_files else ", file đã lên R2"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="LECTURE_RAG_DATA_DIR của bản chạy trên máy")
    parser.add_argument("--videos", nargs="*", help="tên file không đuôi (mặc định: mọi video)")
    parser.add_argument("--skip-files", action="store_true", help="không đẩy mp4/ảnh lên R2 (chỉ ghi Supabase)")
    args = parser.parse_args()
    if not db.enabled():
        sys.exit("Thiếu SUPABASE_DB_URL trong .env")
    if not args.skip_files and not objects.enabled():
        sys.exit("Thiếu R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET (hoặc dùng --skip-files)")
    missing = db.check_schema()
    if missing:
        sys.exit(f"Supabase chưa có bảng {missing}: chạy deploy/supabase/schema.sql trước")
    ids = sorted(p.name.removesuffix(".meta.json") for p in (args.data / "videos").glob("*.meta.json"))
    if args.videos:
        wanted = {f"{name}.mp4" for name in args.videos}
        ids = [i for i in ids if (read_json(args.data / "videos" / f"{i}.meta.json") or {}).get("original_filename") in wanted]
    for video_id in ids:
        print(sync_video(args.data, video_id, args.skip_files), flush=True)


if __name__ == "__main__":
    main()
