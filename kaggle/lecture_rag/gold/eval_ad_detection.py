"""Score stage 01 with the ad filter against gold set 1.

    python kaggle/lecture_rag/gold/eval_ad_detection.py kaggle/lecture_rag/gold/slide_boundaries <export_dir>

<export_dir>/<video>/intervals.json comes from notebooks/ad_detection_run.ipynb. Two scores:
  - slide boundaries (main policy: S and SC are boundaries). A run boundary within TOL of a
    gold event takes that event's label; boundaries at an ad edge (gold C event or an edge of
    ad_labels.json) are neutral, like C in gold.json. The run without the ad filter (gold
    events marked pipeline == "slide") is scored the same way for comparison.
  - ad segments: per-second P/R/F1 against ad_labels.json and ad edges found within 2 s.
"""

import json
import sys
from pathlib import Path

GOLD, EXPORT = Path(sys.argv[1]), Path(sys.argv[2])
TOL, AD_TOL = 1.0, 2.0
SPLITS = {"S", "SC"}
gold = {v["video"]: v for v in json.loads((GOLD / "gold_with_ads.json").read_text(encoding="utf-8"))["videos"]}
ads = {v["video"]: v for v in json.loads((GOLD / "ad_labels.json").read_text(encoding="utf-8"))["videos"]}


def score_boundaries(boundaries, events, ad_edges):
    tp = fp = 0
    matched = set()
    for t in boundaries:
        near = [e for e in events if abs(e["time_sec"] - t) <= TOL]
        event = min(near, key=lambda e: abs(e["time_sec"] - t)) if near else None
        if event is not None and event["label"] in SPLITS:
            tp += id(event) not in matched
            matched.add(id(event))
        elif (event is not None and event["label"] in {"C", "L"}) or any(abs(t - x) <= AD_TOL for x in ad_edges):
            continue                                  # ad edge or layout change: neutral
        else:
            fp += 1
    fn = sum(e["label"] in SPLITS and id(e) not in matched for e in events)
    return tp, fp, fn


def prf(tp, fp, fn):
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


rows, totals = [], {}
for folder in sorted(p for p in EXPORT.iterdir() if (p / "intervals.json").exists()):
    video = folder.name
    intervals = json.loads((folder / "intervals.json").read_text(encoding="utf-8"))
    events, label = gold[video]["events"], ads[video]
    ad_edges = [x for a, b in label["ad_intervals"] for x in (a, b)]
    new = [item["start_sec"] for item in intervals[1:]]
    old = [e["time_sec"] for e in events if e["pipeline"] == "slide"]
    per_second = label["per_second"]
    pred = [0] * len(per_second)
    found = []
    for item in intervals:
        if item["content_kind"] == "ad":
            found.append((item["start_sec"], item["end_sec"]))
            for k in range(len(pred)):
                if item["start_sec"] <= k + 0.5 < item["end_sec"]:
                    pred[k] = 1
    a_tp = sum(p and t for p, t in zip(pred, per_second))
    a_fp = sum(p and not t for p, t in zip(pred, per_second))
    a_fn = sum(t and not p for p, t in zip(pred, per_second))
    pred_edges = [x for a, b in found for x in (a, b)]
    edge_hit = sum(any(abs(x - y) <= AD_TOL for y in pred_edges) for x in ad_edges)
    row = {"video": video, "split": gold[video]["split"],
           "new": score_boundaries(new, events, ad_edges), "old": score_boundaries(old, events, ad_edges),
           "ad": (a_tp, a_fp, a_fn), "edges": (edge_hit, len(ad_edges), len(pred_edges))}
    rows.append(row)
    for key in ("new", "old", "ad"):
        group = totals.setdefault((row["split"], key), [0, 0, 0])
        for i in range(3):
            group[i] += row[key][i]
    edges = totals.setdefault((row["split"], "edges"), [0, 0, 0])
    for i in range(3):
        edges[i] += row["edges"][i]
    print(f"{video} {row['split']:8} slide F1 không lọc {prf(*row['old'])[2]:.3f} → có lọc {prf(*row['new'])[2]:.3f}"
          f" · quảng cáo F1 {prf(*row['ad'])[2]:.3f} · đầu/cuối {edge_hit}/{len(ad_edges)}")

print()
for split in ("tuning", "held-out"):
    if (split, "new") not in totals:
        continue
    for key, name in (("old", "slide · không lọc"), ("new", "slide · có lọc"), ("ad", "quảng cáo theo giây")):
        p, r, f = prf(*totals[(split, key)])
        print(f"{split:8} {name:20} P {p:.3f} R {r:.3f} F1 {f:.3f}  (tp/fp/fn {totals[(split, key)]})")
    hit, true, pred = totals[(split, "edges")]
    print(f"{split:8} đầu/cuối quảng cáo ±{AD_TOL:.0f}s: tìm được {hit}/{true} · bộ lọc đặt {pred} mốc")
(EXPORT / "eval.json").write_text(json.dumps({"rows": rows, "totals": {f"{a}|{b}": v for (a, b), v in totals.items()}},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
