"""Content filter A: compare the ad scorers per second on the signal dump (no GPU).

    python kaggle/lecture_rag/gold/eval_outlier.py <signals_dir> [--grid]

<signals_dir> holds <video>.npz from notebooks/signal_dump.ipynb and index.json (video groups).
For each scorer -- student (learned on L25), outlier (adfilter.outlier_scores, the video against
its own majority) and both (flag only when both agree) -- the scores go through the pipeline's
ad_mask() and ad_guard(), then per second:
  L25 (gold ad_labels.json)       lesson kept, ad caught
  ad_free (Oxford, LPM, SlideSpeech) lesson kept (every second is lesson)
  vn_new (no labels)              share of the video flagged (look at the frames when > 0)
--grid sweeps the outlier options and prints the best settings first.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_adfilter():
    namespace = {}
    exec((HERE.parent / "notebook_code" / "adfilter.py").read_text(encoding="utf-8"), namespace)
    return namespace


def flagged_seconds(code, scores, times, threshold, duration):
    mask = code["ad_mask"](list(scores), threshold, 5, 5, 2)
    runs = [(times[a], times[min(b, len(times) - 1)]) for a, b in code["ad_runs"](mask)]
    if code["ad_guard"](runs, duration):
        return np.zeros(int(duration) + 1, bool), True
    out = np.zeros(int(duration) + 1, bool)
    for a, b in runs:
        out[int(a):int(b) + 1] = True
    return out, False


def evaluate(code, signals, groups, gold, scorer):
    rows = []
    for name, data in signals.items():
        times = data["times"]
        duration = float(times[-1]) + 1
        scores, threshold = scorer(data)
        flags, guarded = flagged_seconds(code, scores, times, threshold, duration)
        group = groups[name]
        row = {"video": name, "group": group, "guard": guarded, "flagged": round(float(flags.mean()), 3)}
        if group == "l25" and name in gold:
            truth = np.array(gold[name]["per_second"][:len(flags)], bool)
            flags = flags[:len(truth)]
            row["lesson_kept"] = round(float((~flags[~truth]).mean()), 4)
            row["ad_caught"] = round(float(flags[truth].mean()), 4)
        elif group == "ad_free":
            row["lesson_kept"] = round(float(1 - flags.mean()), 4)
        rows.append(row)
    return rows


def summary(rows):
    l25 = [r for r in rows if r["group"] == "l25" and "ad_caught" in r]
    free = [r for r in rows if r["group"] == "ad_free"]
    vn = [r for r in rows if r["group"] == "vn_new"]
    return {"l25_lesson_kept": round(np.mean([r["lesson_kept"] for r in l25]), 4) if l25 else None,
            "l25_ad_caught": round(np.mean([r["ad_caught"] for r in l25]), 4) if l25 else None,
            "ad_free_lesson_kept_min": round(min(r["lesson_kept"] for r in free), 4) if free else None,
            "ad_free_lesson_kept_mean": round(np.mean([r["lesson_kept"] for r in free]), 4) if free else None,
            "vn_flagged_mean": round(np.mean([r["flagged"] for r in vn]), 4) if vn else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("signals", type=Path)
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    code = load_adfilter()
    index = json.loads((args.signals / "index.json").read_text(encoding="utf-8"))
    groups = {name: group for group, names in index["videos"].items() for name in names}
    signals = {p.stem: dict(np.load(p)) for p in sorted(args.signals.glob("*.npz"))}
    for data in signals.values():
        data["emb"] = data["emb"].astype("float32")
    gold = {v["video"]: v for v in json.loads((HERE / "slide_boundaries" / "ad_labels.json").read_text(encoding="utf-8"))["videos"]}

    student = lambda d: (d["student"], 0.55)
    def outlier(options):
        return lambda d: (np.array(code["outlier_scores"](d["emb"], d["change"], **options)[0]), 0.5)
    def both(options):
        return lambda d: (np.minimum(d["student"] / 0.55, np.array(code["outlier_scores"](d["emb"], d["change"], **options)[0]) / 0.5) * 0.5, 0.5)

    if args.grid:
        results = []
        for k, offset, weight in itertools.product([4, 6, 8], [2.0, 3.0, 4.0, 5.0], [0.0, 0.5, 1.0]):
            options = {"k": k, "offset": offset, "weight_change": weight}
            s = summary(evaluate(code, signals, groups, gold, outlier(options)))
            results.append((options, s))
        # first keep the lecture (ad-free min, then L25), then catch ads
        results.sort(key=lambda r: ((r[1]["ad_free_lesson_kept_min"] or 0) < 0.99,
                                    (r[1]["l25_lesson_kept"] or 0) < 0.99, -(r[1]["l25_ad_caught"] or 0)))
        for options, s in results[:10]:
            print(options, s)
        return
    for label, scorer in [("student", student), ("outlier", outlier({})), ("both", both({}))]:
        rows = evaluate(code, signals, groups, gold, scorer)
        print(f"\n== {label} ==", summary(rows))
        for r in rows:
            print("  ", r)


if __name__ == "__main__":
    main()
