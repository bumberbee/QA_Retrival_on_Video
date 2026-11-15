"""Import notebook-compatible exports into the local application data."""

import json
import shutil
from pathlib import Path

from app.shared import storage
from app.shared.models import Transcript

VARIANT = "F_fast_pipeline"
LABEL = "F · Pipeline nhanh (quét một lượt)"


def is_ad(interval: dict) -> bool:
    return interval.get("content_kind") == "ad" or bool(interval.get("ad_flagged"))


def import_video(folder: Path, videos: Path | None = None, *, video_id: str | None = None,
                 require_index: bool = False, replace_transcript: bool = False) -> str | None:
    """An explicit ID attaches a one-video export to an already uploaded video."""
    storage.require_local_writes()
    folder = Path(folder)
    source_id = folder.name
    video_id = (storage.validate_video_id(video_id) if video_id is not None
                else storage.find_video_id_by_original_filename(f"{source_id}.mp4"))
    if require_index:
        for name in ("chunks.jsonl", "embeddings.npy", "lexical.sqlite3"):
            if not (folder / "05_index" / name).is_file():
                raise ValueError("Export is missing its search index")
        if not (folder / "04_bundle" / "transcript_app.json").is_file():
            raise ValueError("Export is missing its transcript")
    if not video_id:
        video = videos / f"{source_id}.mp4" if videos else None
        if video is None or not video.is_file():
            print(f"Bỏ qua {source_id}: chưa có trong app và không thấy file video (--videos).")
            return
        video_id = storage.new_video_id()
        destination = storage.video_path(video_id, ".mp4")
        shutil.copyfile(video, destination)
        storage.save_video_meta(video_id, f"{source_id}.mp4", destination)
        print(f"Đã thêm video {source_id}.mp4 vào app")
    slides_dir = folder / "01_slides"
    intervals = json.loads((slides_dir / "intervals.json").read_text(encoding="utf-8"))
    audit = json.loads((slides_dir / "reveal_audit.json").read_text(encoding="utf-8"))
    split_reasons = {item["current_state"]: item["decision"] for item in audit}

    target = storage.keyframes_dir(video_id) / "fast"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(slides_dir / "keyframes" / "states", target)

    slides = []
    # slide_speech filter: the flags are decided in stage 04 (segments.json), not in stage 01
    segments_file = folder / "04_bundle" / "segments.json"
    if segments_file.exists():
        flagged = {s["segment_id"] for s in json.loads(segments_file.read_text(encoding="utf-8")) if s.get("ad_flagged")}
        for interval in intervals:
            if interval["segment_id"] in flagged:
                interval["ad_flagged"] = True
    ads = [interval for interval in intervals if is_ad(interval)]
    for interval in intervals:
        if is_ad(interval):
            continue
        states = interval["reveal_states"]
        slides.append({
            "key": f"{interval['start_sec']:.3f}",
            "start_sec": interval["start_sec"],
            "end_sec": interval["end_sec"],
            "image": f"fast/{Path(states[-1]['image_path']).name}",
            "states": [
                {"image": f"fast/{Path(state['image_path']).name}",
                 "start_sec": state["start_sec"], "p_slide": None,
                 "decision": state.get("decision")}
                for state in states
            ],
            "split_reason": split_reasons.get(states[0]["source_state"]),
        })

    # A video imported only through this script needs default slides for the library.
    if replace_transcript:
        for frame in storage.keyframes_dir(video_id).glob("seg_*.jpg"):
            storage.contained_path(storage.keyframes_dir(video_id), frame.name).unlink()
    if not any(storage.keyframes_dir(video_id).glob("seg_*.jpg")):
        for index, slide in enumerate(slides):
            shutil.copyfile(storage.keyframes_dir(video_id) / slide["image"],
                            storage.keyframes_dir(video_id) / f"seg_{index:05d}.jpg")

    path = storage.contained_path(storage.SLIDE_VARIANTS_DIR, f"{video_id}.json")
    payload = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
               else {"source_id": source_id, "default": VARIANT, "variants": {}})
    payload["variants"][VARIANT] = {"label": LABEL, "slides": slides}
    storage.SLIDE_VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    storage._write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))

    note = ""
    index = folder / "05_index"
    if (index / "chunks.jsonl").exists():
        target_index = storage.contained_path(storage.DATA_DIR, "index", video_id)
        shutil.rmtree(target_index, ignore_errors=True)
        shutil.copytree(index, target_index)
        note += ", chép chỉ mục tìm kiếm"
    transcript_file = folder / "04_bundle" / "transcript_app.json"
    if transcript_file.exists() and (replace_transcript or storage.load_transcript(video_id) is None):
        raw = json.loads(transcript_file.read_text(encoding="utf-8"))
        raw["video_id"] = video_id
        storage.save_transcript(Transcript(**raw))
        note = f", thêm transcript ({len(raw['segments'])} đoạn)"
    if ads:
        note += f", ẩn {len(ads)} đoạn quảng cáo ({sum(a['end_sec'] - a['start_sec'] for a in ads):.0f} s)"
    print(f"OK {source_id} -> {video_id}: {len(slides)} slide{note}")
    from app.features.search import retrieval
    retrieval.load_index.cache_clear()
    return video_id

