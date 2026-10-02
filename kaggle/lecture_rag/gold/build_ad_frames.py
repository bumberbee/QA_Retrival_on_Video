"""1 Hz frames of the gold-set videos with their lesson/ad label, as a Kaggle dataset.

    python kaggle/lecture_rag/gold/build_ad_frames.py kaggle/lecture_rag/gold/slide_boundaries <videos_dir> <out_dir>

Frame k is the frame at k + 0.5 s (same clock as ad_labels.json), saved as a 384x216 JPEG
in images/<video>/<k>.jpg. frames.csv lists video, split, second, label (1 = ad) and
motion: mean absolute difference of the 64x36 gray thumbnail with the previous second,
a cheap signal for the ad filter's secondary cue (ads move, slides mostly stand still).
"""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

GOLD, VIDEOS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
labels = json.loads((GOLD / "ad_labels.json").read_text(encoding="utf-8"))["videos"]
rows = []
for item in labels:
    video, per_second = item["video"], item["per_second"]
    folder = OUT / "images" / video
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEOS / f"{video}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    targets = [int((k + 0.5) * fps) for k in range(len(per_second))]
    k, frame_no, previous = 0, 0, None
    while k < len(targets) and cap.grab():
        if frame_no == targets[k]:
            _, frame = cap.retrieve()
            cv2.imwrite(str(folder / f"{k:05d}.jpg"), cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
            motion = 0.0 if previous is None else float(np.abs(small - previous).mean() / 255)
            previous = small
            rows.append({"video": video, "split": item["split"], "second": k, "label": per_second[k],
                         "motion": round(motion, 4), "image": f"images/{video}/{k:05d}.jpg"})
            k += 1
        frame_no += 1
    cap.release()
    print(video, k, "frames", sum(per_second[:k]), "ad")

with open(OUT / "frames.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT / "ad_labels.json").write_bytes((GOLD / "ad_labels.json").read_bytes())
print(len(rows), "frames")
