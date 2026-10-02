"""Gold set 1 from the human-checked review export, plus stage 01 scores against it.

    python kaggle/lecture_rag/gold/finalize_boundaries.py kaggle/lecture_rag/gold/slide_boundaries

Reads slide_boundaries_checked.json (export of review.html) and writes gold.json and
metrics.json. FIXES holds decisions made after the export: the human marked these
events wrong without picking a label, or a missed-change time was placed wrong.
"""

import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
# (video, time rounded to 0.1 s) -> new label, None to drop, or (new time, label)
FIXES = {
    ("L25_V001", 2.1): "S",       # logo -> lecturer greeting: the lesson starts
    ("L25_V010", 2.1): "S",
    ("L25_V002", 1127.5): "S",    # lecturer's closing words are lesson content
    ("L25_V007", 776.4): "R",     # same chart, side text replaced
    ("L25_V001", 123.3): "C",     # back from an ad to a slide already shown
    ("L25_V010", 110.8): "C",
    ("L25_V010", 264.2): "SC",    # document scrolled, not a layout change
    ("L25_V001", 757.2): (760.5, "C"),  # ad start was placed ~3 s early
    ("L25_V003", 135.0): (136.75, "C"),
}
PENDING = {k for k, v in FIXES.items() if not isinstance(v, tuple)}  # still to confirm
SPLIT_LABELS = {"strict": {"S"}, "scroll_is_slide": {"S", "SC"}}
# Main policy (decided with the thesis author): a slide is one stable state of the screen
# content, so scrolling a document to new content (SC) is a new slide. "strict" is reported too.
PRIMARY_POLICY = "scroll_is_slide"
# Ads, logos and school promos (C) are dropped from the gold set: they carry no lesson
# content, so splitting or merging them is neither right nor wrong. gold_with_ads.json keeps them.

checked = json.loads((OUT / "slide_boundaries_checked.json").read_text(encoding="utf-8"))
gold, with_ads, metrics = [], [], {}
for video in checked["videos"]:
    events = []
    for index, e in enumerate(video["events"]):
        key = (video["video"], round(e["time_sec"], 1))
        label, time = e["label"], e["time_sec"]
        if key in FIXES:
            fix = FIXES[key]
            time, label = fix if isinstance(fix, tuple) else (time, fix)
        events.append({"index": index, "time_sec": time, "pipeline": e["pipeline"], "label": label,
                       "source": "claude+human" if key not in PENDING else "claude+human (label to confirm)"})
    with_ads.append({"video": video["video"], "split": video["split"], "events": events})
    events = [e for e in events if e["label"] != "C"]
    gold.append({"video": video["video"], "split": video["split"], "events": events})
    for policy, splits in SPLIT_LABELS.items():
        tp = sum(e["pipeline"] == "slide" and e["label"] in splits for e in events)
        fp = sum(e["pipeline"] == "slide" and e["label"] not in splits | {"L"} for e in events)
        fn = sum(e["pipeline"] != "slide" and e["label"] in splits for e in events)
        metrics.setdefault(policy, {})[video["video"]] = {"split": video["split"], "tp": tp, "fp": fp, "fn": fn}

for policy, per_video in metrics.items():
    for group in ("tuning", "held-out"):
        rows = [m for m in list(per_video.values()) if m.get("split") == group]
        tp, fp, fn = (sum(m[k] for m in rows) for k in ("tp", "fp", "fn"))
        p, r = tp / (tp + fp), tp / (tp + fn)
        per_video[f"_{group}"] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3),
                                  "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3)}
        print(f"{policy:16} {group:9} P={p:.3f} R={r:.3f} F1={2 * p * r / (p + r):.3f}  (tp {tp}, fp {fp}, fn {fn})")

(OUT / "gold.json").write_text(json.dumps({
    "legend": checked["legend"],
    "note": "Pipeline events labelled by Claude, every one checked by a human; L (layout change) "
            "is neither a required split nor an error; ads (C) removed. Main policy: S and SC (document scrolled to new content) are slide boundaries.", "primary_policy": PRIMARY_POLICY, "videos": gold}, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "gold_with_ads.json").write_text(json.dumps({"legend": checked["legend"], "videos": with_ads},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "metrics.json").write_text(json.dumps({"primary_policy": PRIMARY_POLICY, **metrics}, indent=1), encoding="utf-8")
