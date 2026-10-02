"""Review page for gold set 2 (text on slides), built from a fast-pipeline export.

    python kaggle/lecture_rag/gold/build_ocr_review.py <export_dir> <out_dir> [--per-video 20] [--blind 30] [--seed 0]

<export_dir> is fast_pipeline_export.zip unpacked (<video>/04_bundle/segments.json + keyframes/).
Writes <out_dir>/ocr_review.html, thumbs/, images/ and ocr_review_items.json.

Every non-ad keyframe gets a "readable content?" label (scores the OCR gate). --per-video
keyframes per video also get their text typed: one per equal slice of the video, picked at
random without looking at the OCR output, so slides the OCR (or the gate) got wrong are sampled
as often as the others. --blind of those start with an empty box instead of the OCR text, to
measure how much pre-filled text sways the labeller (eval_ocr.py reports both groups).
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ocr import ocr_status, prediction_text  # noqa: E402

THUMB_WIDTH = 360


def pick_spread(items, count, rng):
    """One random item from each of `count` equal slices of `items` (all of them if fewer)."""
    if len(items) <= count:
        return list(items)
    edges = [round(i * len(items) / count) for i in range(count + 1)]
    return [items[rng.randrange(start, end)] for start, end in zip(edges, edges[1:])]


def video_rng(seed, video):
    # str hashes change between Python runs; sha256 keeps the sample reproducible.
    return random.Random(int(hashlib.sha256(f"{seed}:{video}".encode()).hexdigest()[:16], 16))


def build(export_dir, out_dir, per_video=20, blind=30, seed=0):
    export_dir, out_dir = Path(export_dir), Path(out_dir)
    template = Path(__file__).with_name("ocr_review_template.html").read_text(encoding="utf-8")
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    videos, sampled = [], []
    for path in sorted(export_dir.glob("*/04_bundle/segments.json")):
        bundle, video = path.parent, path.parent.parent.name
        keyframes = [segment for segment in json.loads(path.read_text(encoding="utf-8"))
                     if segment.get("content_kind") != "ad" and (segment.get("visual") or {}).get("image_path")]
        chosen = {segment["segment_id"] for segment in pick_spread(keyframes, per_video, video_rng(seed, video))}
        items = []
        for segment in keyframes:
            visual = segment["visual"]
            name = segment["segment_id"].replace(":", "_")
            source = bundle / visual["image_path"]
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
                image.save(out_dir / "thumbs" / f"{name}.jpg", quality=80)
            item = {"id": segment["segment_id"], "start_sec": float(segment["start_sec"]),
                    "end_sec": float(segment["end_sec"]), "kind": segment.get("content_kind"),
                    "ocr_status": ocr_status(visual), "gate_score": visual.get("ocr_gate_score"),
                    "thumb": f"thumbs/{name}.jpg", "transcribe": segment["segment_id"] in chosen}
            if item["transcribe"]:
                shutil.copyfile(source, out_dir / "images" / f"{name}{source.suffix}")
                item["image"] = f"images/{name}{source.suffix}"
                item["prefill"] = prediction_text(visual) if item["ocr_status"] == "read" else ""
                sampled.append(item)
            items.append(item)
        videos.append({"video": video, "items": items})
        print(video, len(items), "keyframe ·", sum(item["transcribe"] for item in items), "cần chép chữ")
    if not videos:
        raise SystemExit(f"Không thấy */04_bundle/segments.json trong {export_dir}")
    # a video folder without a bundle failed at stage 02-04 in that run (its log says "error")
    missing = sorted(path.name for path in export_dir.iterdir()
                     if path.is_dir() and not (path / "04_bundle" / "segments.json").is_file())
    for video in missing:
        print(f"CẢNH BÁO: bỏ qua {video}: export không có 04_bundle/segments.json (video lỗi ở bước 02-04?)")

    for item in random.Random(seed).sample(sampled, min(blind, len(sampled))):
        item["blind"], item["prefill"] = True, ""
    meta = {"export": export_dir.name, "per_video": per_video, "blind": blind, "seed": seed,
            "keyframes": sum(len(v["items"]) for v in videos), "transcribe": len(sampled),
            "missing_videos": missing}
    (out_dir / "ocr_review_items.json").write_text(
        json.dumps({"meta": meta, "videos": videos}, ensure_ascii=False, indent=1), encoding="utf-8")
    data = json.dumps({"meta": meta, "videos": videos}, ensure_ascii=False).replace("</", "<\\/")
    (out_dir / "ocr_review.html").write_text(template.replace("/*DATA*/null", data), encoding="utf-8")
    print("wrote", out_dir / "ocr_review.html", "·", meta["keyframes"], "keyframe ·", len(sampled), "slide chép chữ")
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--per-video", type=int, default=20)
    parser.add_argument("--blind", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    build(args.export_dir, args.out_dir, args.per_video, args.blind, args.seed)


if __name__ == "__main__":
    main()
