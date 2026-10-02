"""Review page for gold set 1 (slide boundaries), built from the pipeline's draft.

    python kaggle/lecture_rag/gold/build_boundary_review.py <stage01_root> <review_dir> <videos_dir> <out_dir>

<review_dir> holds build_review_sheets.py output (<vid>/draft.json), claude_labels.json
(Claude's label for every pipeline event, in event order) and missed.json (changes the
pipeline did not report). Writes <out_dir>/review.html, thumbs/ and draft_labels.json.
All labels are drafts until a human confirms them in review.html.
"""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT, REVIEW, VIDEOS, OUT = (Path(p) for p in sys.argv[1:5])
THUMB = (320, 180)
TUNING = {"L25_V001", "L25_V002", "L25_V005", "L25_V007"}  # used to tune stage 01
(OUT / "thumbs").mkdir(parents=True, exist_ok=True)


def thumb_from_state(video, image_path):
    name = f"{video}_{Path(image_path).stem}.jpg"
    target = OUT / "thumbs" / name
    if not target.exists():
        with Image.open(ROOT / video / "01_slides" / image_path) as image:
            image = image.convert("RGB")
            image.thumbnail(THUMB)
            image.save(target, quality=72)
    return f"thumbs/{name}"


def thumb_from_video(video, seconds):
    name = f"{video}_t{seconds:08.2f}.jpg"
    target = OUT / "thumbs" / name
    if not target.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{seconds:.2f}", "-i", str(VIDEOS / f"{video}.mp4"),
                        "-frames:v", "1", "-vf", f"scale={THUMB[0]}:-2", "-q:v", "5", str(target)], check=True)
    return f"thumbs/{name}"


labels = json.loads((REVIEW / "claude_labels.json").read_text(encoding="utf-8"))
missed = json.loads((REVIEW / "missed.json").read_text(encoding="utf-8"))
videos = []
for folder in sorted(REVIEW.glob("L25_V*")):
    video = folder.name
    draft = json.loads((folder / "draft.json").read_text(encoding="utf-8"))
    events = []
    for event, label in zip(draft["events"], labels[video].split(), strict=True):
        events.append({"time_sec": event["time_sec"], "pipeline": event["pipeline"], "label": label,
                       "before": thumb_from_state(video, event["before"]),
                       "after": thumb_from_state(video, event["after"])})
    for item in missed.get(video, []):
        t = item["time_sec"]
        events.append({"time_sec": t, "pipeline": "missed", "label": item["label"], "note": item["note"],
                       "before": thumb_from_video(video, max(0.0, t - 1.5)), "after": thumb_from_video(video, t + 1.5)})
    events.sort(key=lambda e: e["time_sec"])
    videos.append({"video": video, "split": "tuning" if video in TUNING else "held-out", "events": events})
    print(video, len(events), "events")

(OUT / "draft_labels.json").write_text(json.dumps({
    "status": "draft: Claude's labels on the pipeline's events, not yet human-checked",
    "legend": labels["legend"], "videos": videos}, ensure_ascii=False, indent=1), encoding="utf-8")
template = (Path(__file__).parent / "boundary_review_template.html").read_text(encoding="utf-8")
(OUT / "review.html").write_text(template.replace("/*DATA*/null", json.dumps(
    {"legend": labels["legend"], "videos": videos}, ensure_ascii=False)), encoding="utf-8")
print("wrote", OUT / "review.html")
