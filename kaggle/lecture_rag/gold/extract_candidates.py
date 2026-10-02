"""Propose candidate slide-change moments for gold labelling (independent of the pipeline).

    python kaggle/lecture_rag/gold/extract_candidates.py <video.mp4> <out_dir>

Decodes the video at 2 fps with ffmpeg, keeps 320x180 thumbnails for review, and scores
each step by the share of grey pixels (80x45) that changed, ignoring pixels that change
in most of the video (animated background, the lecturer). The threshold is low on
purpose: candidates are reviewed by eye, so extra candidates are cheap and misses are not.
Also adds check points inside long quiet stretches so slow reveals can be reviewed.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FPS = 2
THUMB = (320, 180)
SMALL = (80, 45)
PIXEL_CHANGE = 20 / 255
CANDIDATE_THRESHOLD = 0.004   # about 14 of 3600 pixels
BUSY_FRACTION = 0.30
QUIET_CHECK_SEC = 45


def decode(video):
    command = ["ffmpeg", "-v", "error", "-i", str(video), "-vf",
               f"fps={FPS},scale={THUMB[0]}:{THUMB[1]}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    size = THUMB[0] * THUMB[1] * 3
    while True:
        chunk = process.stdout.read(size)
        if len(chunk) < size:
            break
        yield np.frombuffer(chunk, dtype=np.uint8).reshape(THUMB[1], THUMB[0], 3)
    process.wait()


def main():
    video, out = Path(sys.argv[1]), Path(sys.argv[2])
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    small = []
    for index, frame in enumerate(decode(video)):
        image = Image.fromarray(frame)
        image.save(frames_dir / f"{index:05d}.jpg", quality=80)
        small.append(np.asarray(image.convert("L").resize(SMALL, Image.BILINEAR), dtype=np.float32) / 255)
    small = np.stack(small)
    changed = np.abs(np.diff(small, axis=0)) > PIXEL_CHANGE          # (n-1, h, w)
    busy = changed.mean(axis=0) > BUSY_FRACTION
    score = np.array([step[~busy].mean() if (~busy).any() else step.mean() for step in changed])
    # Local maxima above the threshold, at most one per second.
    candidates = []
    for i, value in enumerate(score):
        window = score[max(0, i - FPS): i + FPS + 1]
        if value >= CANDIDATE_THRESHOLD and value == window.max():
            if not candidates or i - candidates[-1] > FPS:
                candidates.append(i)
    events = [{"frame_after": i + 1, "time_sec": round((i + 1) / FPS, 1), "score": round(float(score[i]), 4),
               "kind": "candidate"} for i in candidates]
    # Check points in long stretches with no candidate (slow reveals, handwriting).
    points = [0] + [e["frame_after"] for e in events] + [len(small) - 1]
    for a, b in zip(points, points[1:]):
        step = QUIET_CHECK_SEC * FPS
        for f in range(a + step, b - FPS * 5, step):
            events.append({"frame_after": f, "time_sec": round(f / FPS, 1), "score": 0.0, "kind": "check"})
    events.sort(key=lambda e: e["frame_after"])
    (out / "candidates.json").write_text(json.dumps({
        "video": video.name, "fps": FPS, "frames": len(small), "duration_sec": round(len(small) / FPS, 1),
        "busy_fraction": round(float(busy.mean()), 4), "threshold": CANDIDATE_THRESHOLD, "events": events},
        indent=1), encoding="utf-8")
    print(video.name, "frames", len(small), "candidates", sum(e["kind"] == "candidate" for e in events),
          "checks", sum(e["kind"] == "check" for e in events), "busy", round(float(busy.mean()), 3))


if __name__ == "__main__":
    main()
