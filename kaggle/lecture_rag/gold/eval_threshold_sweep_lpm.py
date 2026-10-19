"""Table of the stage-01 threshold sweep on LPM: slide-boundary P/R/F1 per change_threshold x
stable_samples against the dataset authors' slide times (lpm-videos/slides.csv).

    python kaggle/lecture_rag/gold/eval_threshold_sweep_lpm.py <sweep_dir> <slides.csv>

Gold boundaries are the start_s of every slide but the first. Predicted boundaries are the start_sec
of every segment but the first. Only the span the authors annotated counts (first slide start to
last slide end, ± tolerance), so title cards and credits outside it are ignored. One-to-one greedy
matching by distance; scored at ±1 s and ±3 s (the authors' times are aligned by hand, not to frames).
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

TOLERANCES = (1.0, 3.0)


def match(gold, pred, tol):
    pairs = sorted((abs(g - p), i, j) for i, g in enumerate(gold) for j, p in enumerate(pred) if abs(g - p) <= tol)
    used_g, used_p = set(), set()
    for _, i, j in pairs:
        if i not in used_g and j not in used_p:
            used_g.add(i); used_p.add(j)
    tp = len(used_g)
    return tp, len(pred) - tp, len(gold) - tp


def prf(tp, fp, fn):
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    sweep, slides_csv = Path(sys.argv[1]), sys.argv[2]
    slides = defaultdict(list)
    for row in csv.DictReader(open(slides_csv, encoding="utf-8")):
        slides[row["video_id"]].append((float(row["start_s"]), float(row["end_s"])))
    rows = []
    for config in sorted(p for p in sweep.iterdir() if p.is_dir()):
        totals = {tol: [0, 0, 0] for tol in TOLERANCES}
        for video in sorted(p for p in config.iterdir() if p.is_dir()):
            spans = sorted(slides[video.name])
            lo, hi = spans[0][0], spans[-1][1]
            gold = [s for s, _ in spans[1:]]
            intervals = json.loads((video / "intervals.json").read_text(encoding="utf-8"))
            for tol in TOLERANCES:
                pred = [i["start_sec"] for i in intervals[1:] if lo - tol <= i["start_sec"] <= hi + tol]
                for k, v in enumerate(match(gold, pred, tol)):
                    totals[tol][k] += v
        change, stable = config.name[1:].split("_s")
        row = {"change": float(change), "stable": int(stable)}
        for tol in TOLERANCES:
            p, r, f = prf(*totals[tol])
            row.update({f"P@{tol:g}": round(p, 3), f"R@{tol:g}": round(r, 3), f"F1@{tol:g}": round(f, 3)})
        rows.append(row)
    rows.sort(key=lambda r: (r["stable"], r["change"]))
    best = max(rows, key=lambda r: r["F1@3"])
    print(f"{'khác':>6} {'yên':>4} | {'P ±1s':>6} {'R':>6} {'F1':>6} | {'P ±3s':>6} {'R':>6} {'F1':>6}")
    for r in rows:
        mark = "  ← tốt nhất (±3 s)" if r is best else ""
        mark += "  ← hiện tại" if (r["change"], r["stable"]) == (0.055, 2) else ""
        print(f"{r['change']:>6} {r['stable']:>4} | {r['P@1']:>6} {r['R@1']:>6} {r['F1@1']:>6} | "
              f"{r['P@3']:>6} {r['R@3']:>6} {r['F1@3']:>6}{mark}")
    (sweep / "sweep_table.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
