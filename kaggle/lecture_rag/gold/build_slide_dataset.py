"""The correct slide list of every video (from gold set 1), as a dataset for Kaggle.

    python kaggle/lecture_rag/gold/build_slide_dataset.py <gold_dir> <stage01_root> <review_dir> <videos_dir> <out_dir>

A new slide starts at every S or SC event (a scrolled document page counts as a slide);
C starts an ad/logo segment, which is dropped; R, X and L stay in the current slide.
Each slide's image is its last state (the most complete one after progressive reveals).
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

GOLD, ROOT, REVIEW, VIDEOS, OUT = (Path(p) for p in sys.argv[1:6])
STARTS, AD = {"S", "SC"}, "C"
# C events where the video comes back from an ad to a slide shown before the ad: the
# lesson resumes here, so a slide starts (the same slide again, not a new one).
RESUME = {("L25_V001", 123.3), ("L25_V010", 110.8)}
gold = json.loads((GOLD / "gold_with_ads.json").read_text(encoding="utf-8"))
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "images").mkdir(parents=True)


def duration(video):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(VIDEOS / f"{video}.mp4")], capture_output=True, text=True, check=True).stdout)


def video_frame(video, seconds, target):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{seconds:.2f}", "-i", str(VIDEOS / f"{video}.mp4"),
                    "-frames:v", "1", "-q:v", "3", str(target)], check=True)


rows = []
for video in gold["videos"]:
    v = video["video"]
    # gold "index" is the review-page order (pipeline events + missed ones); match by time.
    before = {round(e["time_sec"], 2): e["before"]
              for e in json.loads((REVIEW / v / "draft.json").read_text(encoding="utf-8"))["events"]}
    events = sorted(video["events"], key=lambda e: e["time_sec"])
    end_of_video = duration(v)
    (OUT / "images" / v).mkdir()
    slides = []
    for n, event in enumerate(events):
        if (v, round(event["time_sec"], 1)) in RESUME:
            event = event | {"label": "S"}
        if event["label"] not in STARTS | {AD}:
            if slides and slides[-1] is not None:
                slides[-1]["states"] += 1
            continue
        if slides and slides[-1] is not None:
            slides[-1]["end_sec"] = event["time_sec"]
            slides[-1]["last_event"] = n
        slides.append(None if event["label"] == AD else
                      {"start_sec": event["time_sec"], "kind": "page" if event["label"] == "SC" else "slide", "states": 1})
    if slides and slides[-1] is not None:
        slides[-1]["end_sec"], slides[-1]["last_event"] = end_of_video, None
    for k, slide in enumerate(s for s in slides if s is not None):
        image = OUT / "images" / v / f"{k:03d}.jpg"
        nxt = slide.pop("last_event")
        source = events[nxt] if nxt is not None else None
        if source is not None and source["pipeline"] != "missed":
            # The pipeline's "before" image of the next boundary is this slide's last state.
            shutil.copy(ROOT / v / "01_slides" / before[round(source["time_sec"], 2)], image)
        else:
            video_frame(v, (source["time_sec"] if source else slide["end_sec"]) - 1.5, image)
        rows.append({"video": v, "split": video["split"], "slide": k, "kind": slide["kind"],
                     "start_sec": round(slide["start_sec"], 2), "end_sec": round(slide["end_sec"], 2),
                     "reveal_states": slide["states"], "image": f"images/{v}/{k:03d}.jpg"})
    print(v, sum(r["video"] == v for r in rows), "slides")

with open(OUT / "slides.csv", "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
shutil.copy(GOLD / "gold.json", OUT / "boundaries.json")
shutil.copy(GOLD / "gold_with_ads.json", OUT / "boundaries_with_ads.json")
print(len(rows), "slides in total")
