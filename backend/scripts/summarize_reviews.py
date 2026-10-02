"""Tổng hợp đánh giá slide (nút Tốt / Có lỗi trong tab Slide của web app).

    cd backend
    .venv/Scripts/python scripts/summarize_reviews.py            # bản gộp mặc định của mỗi video
    .venv/Scripts/python scripts/summarize_reviews.py --json out.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import storage  # noqa: E402

LABELS = {
    "good": "Tốt",
    "wrong_merge": "Gộp nhầm 2 slide",
    "extra_split": "Tách thừa",
    "bad_image": "Ảnh sai / mờ",
    "not_slide": "Không phải slide",
    "other": "Khác",
}


def clock(seconds):
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", help="Tên bản gộp; mặc định là bản mặc định của video")
    parser.add_argument("--json", type=Path, help="Ghi chi tiết ra file JSON")
    args = parser.parse_args()

    total, rows = Counter(), []
    for video in sorted(storage.list_videos(), key=lambda item: item.filename):
        variants = storage.load_slide_variants(video.video_id)
        if not variants:
            continue
        name = args.variant or variants["default"]
        slides = variants["variants"][name]["slides"]
        reviews = storage.load_slide_reviews(video.video_id).get(name, {})
        if not reviews:
            continue
        counts = Counter(item["verdict"] for item in reviews.values() if item.get("verdict"))
        total.update(counts)
        position = {slide["key"]: index for index, slide in enumerate(slides)}
        errors = sorted(
            (position.get(key, -1), key, item) for key, item in reviews.items()
            if item.get("verdict") and item["verdict"] != "good")
        rows.append({"video": video.filename, "variant": name, "slides": len(slides),
                     "rated": sum(counts.values()), "counts": dict(counts),
                     "errors": [{"slide": index + 1, "start_sec": float(key),
                                 "verdict": item["verdict"], "note": item.get("note", "")}
                                for index, key, item in errors]})

    if not rows:
        print("Chưa có đánh giá nào. Bấm Tốt / Có lỗi dưới các slide trong tab Slide.")
        return
    for row in rows:
        good = row["counts"].get("good", 0)
        print(f"{row['video']}: đã chấm {row['rated']}/{row['slides']} slide · "
              f"tốt {good} ({good / row['rated']:.0%})")
        for error in row["errors"]:
            note = f" — {error['note']}" if error["note"] else ""
            print(f"    slide {error['slide']:>2} ({clock(error['start_sec'])}): "
                  f"{LABELS[error['verdict']]}{note}")
    rated = sum(total.values())
    print("-" * 50)
    print(f"Tổng: {rated} slide đã chấm")
    for verdict, label in LABELS.items():
        if total[verdict]:
            print(f"  {label:<18} {total[verdict]:>4}  ({total[verdict] / rated:.0%})")
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nChi tiết: {args.json}")


if __name__ == "__main__":
    main()
