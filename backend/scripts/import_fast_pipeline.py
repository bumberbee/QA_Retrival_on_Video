"""Thêm kết quả của fast_pipeline_l25.ipynb vào app, thành một bản gộp slide để so sánh.

Với mỗi video (thư mục L25_Vxxx trong fast_pipeline_export):
  - chép ảnh trạng thái (1280px) vào data/keyframes/<id>/fast/;
  - thêm bản "F_fast_pipeline" vào data/slide_variants/<id>.json, giữ các bản cũ;
  - nếu video chưa có transcript thì dùng 04_bundle/transcript_app.json;
  - đoạn quảng cáo (content_kind "ad", hoặc ad_flagged khi bộ lọc chỉ đánh dấu) không thành slide: app ẩn chúng;
  - chép chỉ mục bước 05 vào data/index/<id>/: hỏi đáp tìm bằng chứng theo slide thay vì gửi cả transcript.

Cách dùng:
    cd backend
    .venv/Scripts/python scripts/import_fast_pipeline.py \
        --export ../kaggle/lecture_rag/runs/<run>/outputs/export \
        --videos path/to/mp4s   # chỉ cần cho video chưa có trong app
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import storage  # noqa: E402
from app.models import Transcript  # noqa: E402

VARIANT = "F_fast_pipeline"


def is_ad(interval: dict) -> bool:
    return interval.get("content_kind") == "ad" or bool(interval.get("ad_flagged"))
LABEL = "F · Pipeline nhanh (quét một lượt)"


def import_video(folder: Path, videos: Path | None) -> None:
    source_id = folder.name
    video_id = storage.find_video_id_by_original_filename(f"{source_id}.mp4")
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
    if not any(storage.keyframes_dir(video_id).glob("seg_*.jpg")):
        for index, slide in enumerate(slides):
            shutil.copyfile(storage.keyframes_dir(video_id) / slide["image"],
                            storage.keyframes_dir(video_id) / f"seg_{index:05d}.jpg")

    path = storage.SLIDE_VARIANTS_DIR / f"{video_id}.json"
    payload = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
               else {"source_id": source_id, "default": VARIANT, "variants": {}})
    payload["variants"][VARIANT] = {"label": LABEL, "slides": slides}
    storage.SLIDE_VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    note = ""
    index = folder / "05_index"
    if (index / "chunks.jsonl").exists():
        target_index = storage.DATA_DIR / "index" / video_id
        shutil.rmtree(target_index, ignore_errors=True)
        shutil.copytree(index, target_index)
        note += ", chép chỉ mục tìm kiếm"
    transcript_file = folder / "04_bundle" / "transcript_app.json"
    if transcript_file.exists() and storage.load_transcript(video_id) is None:
        raw = json.loads(transcript_file.read_text(encoding="utf-8"))
        raw["video_id"] = video_id
        storage.save_transcript(Transcript(**raw))
        note = f", thêm transcript ({len(raw['segments'])} đoạn)"
    if ads:
        note += f", ẩn {len(ads)} đoạn quảng cáo ({sum(a['end_sec'] - a['start_sec'] for a in ads):.0f} s)"
    print(f"OK {source_id} -> {video_id}: {len(slides)} slide{note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--videos", type=Path, help="Thư mục chứa L25_Vxxx.mp4 cho video chưa có trong app")
    args = parser.parse_args()
    # one folder per video (L25_V008, MLS-15DA1, ...), as written by the pipeline notebooks
    for folder in sorted(p for p in args.export.iterdir() if (p / "01_slides").is_dir()):
        import_video(folder, args.videos)


if __name__ == "__main__":
    main()
