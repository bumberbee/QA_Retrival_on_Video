"""Per-second "lesson / ad" labels of gold set 1, for training and scoring the ad filter.

    python kaggle/lecture_rag/gold/build_ad_labels.py kaggle/lecture_rag/gold/slide_boundaries <videos_dir>

The C label of gold set 1 means "scene change" and covers ads, logos, school promos but
also the lecturer's intro/outro and title cards. Every C event was checked by eye on the
frame right after it (contact sheet of all 101 C events, in time order per video):
LESSON_C lists the ones where lesson content follows; all other C events start an ad.

S and SC start lesson content; R, X and L keep the current state. When the video comes
back from an ad to a slide already shown there is often no event at all, so each second
after an ad's last C event is compared with lesson frames of the same video (gray 32x18
thumbnails, correlation); the ad ends at the first second that matches.
Writes ad_labels.json with the ad intervals and a per-second label list (1 = ad).
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

GOLD, VIDEOS = Path(sys.argv[1]), Path(sys.argv[2])
# C events followed by lesson content (index in time order over all C events of the gold set)
LESSON_C = {3, 6, 10, 15, 17, 18, 23, 27, 28, 33, 38, 39, 40, 49, 58, 59, 67, 68, 71, 72, 82, 94}
# Ad starts with no event at all in gold set 1, found when scoring the ad filter and checked
# frame by frame (V005: SIU promo after the lecturer intro; V007: SIU promo after the outro).
MISSED_AD_STARTS = {"L25_V005": [31.3], "L25_V007": [2670.3]}
MATCH = 0.85          # correlation with a lesson frame that ends an ad
gold = json.loads((GOLD / "gold_with_ads.json").read_text(encoding="utf-8"))


def duration(video):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(VIDEOS / f"{video}.mp4")], capture_output=True, text=True, check=True).stdout)


def thumbs(path, end):
    """One 32x18 gray thumbnail per second (frame at k + 0.5 s), zero-mean unit-norm."""
    cap, out = cv2.VideoCapture(str(path)), []
    fps = cap.get(cv2.CAP_PROP_FPS)
    targets, k, frame_no = [int((k + 0.5) * fps) for k in range(int(end))], 0, 0
    while k < len(targets):
        ok = cap.grab()
        if not ok:
            break
        if frame_no == targets[k]:
            _, f = cap.retrieve()
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            g -= g.mean()
            out.append(g / (np.linalg.norm(g) + 1e-6))
            k += 1
        frame_no += 1
    cap.release()
    while len(out) < len(targets):
        out.append(out[-1])
    return np.stack(out)


c_index, out, total_ad = 0, [], 0.0
for video in gold["videos"]:
    v, end = video["video"], duration(video["video"])
    events = sorted(video["events"], key=lambda e: e["time_sec"])
    marks = []                                  # (time, "ad" | "lesson" | None)
    for e in events:
        if e["label"] == "C":
            marks.append((e["time_sec"], "lesson" if c_index in LESSON_C else "ad"))
            c_index += 1
        else:
            marks.append((e["time_sec"], "lesson" if e["label"] in {"S", "SC"} else None))
    marks = sorted(marks + [(t, "ad") for t in MISSED_AD_STARTS.get(v, [])])
    segments = [(0.0, marks[0][0] if marks else end, "lesson")]
    segments += [(t, marks[i + 1][0] if i + 1 < len(marks) else end, kind) for i, (t, kind) in enumerate(marks)]
    th = thumbs(VIDEOS / f"{v}.mp4", end)
    n = len(th)

    def walk(search, lesson_ref=None):
        """Label each second. kind None (R, X, L) keeps the current state; with search, an ad
        stops at the first two seconds in a row that look like this video's lesson frames."""
        labels, state = np.zeros(n, dtype=int), "lesson"
        for i, (a, b, kind) in enumerate(segments):
            explicit = kind is not None
            state = kind or state
            lo, hi = min(int(a + 0.5), n), min(int(b + 0.5), n)
            if state != "ad":
                continue
            # an ad with no lesson event after it runs to the end of the video (closing logo)
            to_end = not any(k == "lesson" for _, _, k in segments[i + 1:])
            stop = hi
            if search and not to_end:
                for k in range(lo + (2 if explicit else 0), hi):
                    if all((lesson_ref @ th[min(j, n - 1)]).max() >= MATCH for j in (k, k + 1)):
                        stop, state = k, "lesson"
                        break
            labels[lo:stop] = 1
        return labels

    first = walk(search=False)
    lesson_ref = th[first == 0]
    if len(lesson_ref) > 400:
        lesson_ref = lesson_ref[np.linspace(0, len(lesson_ref) - 1, 400).astype(int)]
    per_second = walk(search=True, lesson_ref=lesson_ref)
    intervals, k = [], 0
    while k < len(per_second):
        if per_second[k]:
            j = k
            while j < len(per_second) and per_second[j]:
                j += 1
            intervals.append([k, j])
            k = j
        else:
            k += 1
    ad_sec = int(per_second.sum())
    total_ad += ad_sec
    out.append({"video": v, "split": video["split"], "duration_sec": round(end, 2),
                "ad_intervals": intervals, "ad_seconds": ad_sec, "per_second": per_second.tolist()})
    print(f"{v} {video['split']:8} {end:6.0f}s  ad {ad_sec:4d}s  " + " ".join(f"{a}-{b}" for a, b in intervals))

assert c_index == 101, c_index
(GOLD / "ad_labels.json").write_text(json.dumps({
    "note": "1 = ad / logo / school promo, 0 = lesson (incl. lecturer intro/outro and title cards). "
            "From gold_with_ads.json with every C event checked by eye; ad ends without an event "
            "found by matching lesson frames of the same video.",
    "videos": out}, ensure_ascii=False), encoding="utf-8")
print(f"Tổng quảng cáo: {total_ad:.0f}s")
