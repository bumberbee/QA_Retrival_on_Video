"""Đưa kết quả so sánh gộp slide (kernel l25-ocr-resnet-merge-compare) vào app để xem và đánh giá.

Mỗi video cần thư mục review gồm states.json, variants.json và states/*.jpg.
Script:
  - dùng video đã upload có cùng tên gốc (vd. L25_V001.mp4), hoặc chép --video vào app;
  - chép ảnh trạng thái vào data/keyframes/<id>/states/;
  - ghi data/slide_variants/<id>.json với 4 bản: A (dự án cũ), B, B+ResNet, C;
  - thay slide mặc định (seg_*.jpg) bằng bản B; ảnh cũ được chuyển vào data/_backup/.

Cách dùng:
    cd backend
    .venv/Scripts/python scripts/import_slide_variants.py \
        --review <review_dir>/L25_V005 \
        --video path/to/L25_V005.mp4
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.shared import storage  # noqa: E402

VARIANTS = {
    "A_old_no_merge": "A · Dự án cũ (không gộp)",
    "B_visual": "B · Gộp theo hình",
    "B_visual_resnet": "B + ResNet",
    "C_visual_ocr_resnet": "C · Hình + OCR + ResNet",
}
DEFAULT_VARIANT = "B_visual"


def find_or_add_video(source_id: str, video: Path | None) -> str:
    filename = f"{source_id}.mp4"
    video_id = storage.find_video_id_by_original_filename(filename)
    if video_id:
        return video_id
    if video is None or not video.is_file():
        raise SystemExit(f"Chưa có {filename} trong app; truyền --video để chép video vào.")
    video_id = storage.new_video_id()
    destination = storage.video_path(video_id, video.suffix.lower())
    shutil.copyfile(video, destination)
    storage.save_video_meta(video_id, filename, destination)
    return video_id


def slides_for(groups: list[list[int]], states: list[dict], audit: list[dict]) -> list[dict]:
    decisions = {item["current_state"]: item for item in audit}
    slides = []
    for group in groups:
        members = [states[index] for index in group]
        slides.append({
            "key": f"{members[0]['start_sec']:.3f}",
            "start_sec": members[0]["start_sec"],
            "end_sec": members[-1]["end_sec"],
            # The final state of a progressive reveal shows the complete slide.
            "image": f"states/{Path(members[-1]['image_path']).name}",
            "states": [
                {
                    "image": f"states/{Path(state['image_path']).name}",
                    "start_sec": state["start_sec"],
                    "p_slide": state.get("p_slide"),
                    "decision": decisions.get(index, {}).get("decision"),
                }
                for index, state in zip(group, members)
            ],
            # Why this slide was split from the previous one.
            "split_reason": decisions.get(group[0], {}).get("decision"),
        })
    return slides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, type=Path, help="Thư mục review của một video")
    parser.add_argument("--video", type=Path, help="File MP4 nếu video chưa có trong app")
    args = parser.parse_args()

    source_id = args.review.name
    states = json.loads((args.review / "states.json").read_text(encoding="utf-8"))
    variants = json.loads((args.review / "variants.json").read_text(encoding="utf-8"))
    video_id = find_or_add_video(source_id, args.video)

    keyframes = storage.keyframes_dir(video_id)
    old = [path for path in keyframes.glob("seg_*.jpg")] if keyframes.exists() else []
    if old:
        backup = storage.DATA_DIR / "_backup" / f"keyframes_{video_id}_{datetime.now():%Y%m%d_%H%M%S}"
        backup.mkdir(parents=True)
        for path in old:
            shutil.move(str(path), backup / path.name)
        print(f"Đã chuyển {len(old)} slide cũ vào {backup}")
    (keyframes / "states").mkdir(parents=True, exist_ok=True)
    for path in (args.review / "states").glob("*.jpg"):
        shutil.copyfile(path, keyframes / "states" / path.name)

    payload = {"source_id": source_id, "default": DEFAULT_VARIANT, "variants": {}}
    for name, label in VARIANTS.items():
        data = variants[name]
        payload["variants"][name] = {
            "label": label, "slides": slides_for(data["groups"], states, data["audit"]),
        }
    for index, slide in enumerate(payload["variants"][DEFAULT_VARIANT]["slides"]):
        shutil.copyfile(keyframes / slide["image"], keyframes / f"seg_{index:05d}.jpg")

    storage.SLIDE_VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    (storage.SLIDE_VARIANTS_DIR / f"{video_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {name: len(value["slides"]) for name, value in payload["variants"].items()}
    print(f"OK {source_id} -> video_id={video_id}: {counts}")


if __name__ == "__main__":
    main()
