"""Content filter, phase 0: how much lesson the index keeps and how much ad it leaves out.

    python kaggle/lecture_rag/gold/eval_content_filter.py <export_dir> [<export_dir> ...]
        [--ad-free MLS-15DA1,...] [--simulate] [--json out.json]

<export_dir>/<video>/01_slides/ contains intervals.json and ad_filter.json (fast pipeline,
run_pipeline or walkthrough exports). Truth per second: gold/slide_boundaries/ad_labels.json
for the L25 videos; --ad-free names videos known to have no ad (Oxford M3AV, LPM): every
second is lesson. Other videos are skipped.

A second counts as indexed when its segment reaches stage 05: not content_kind "ad" (unless
04 marked it ad_suspect) and not ad_flagged. --simulate re-applies the phase-1 rule to an old
"cut" run from its ad_filter.json: ad intervals are left out of the index unless ad_guard()
distrusts them (adfilter.py), so a phase-1 run is predicted without the GPU.

Scores: lesson_kept = indexed lesson seconds / lesson seconds (target >= 0.99);
ad_left_out = ad seconds not indexed / ad seconds (L25 target >= 0.95).
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLD_ADS = HERE / "slide_boundaries" / "ad_labels.json"


def load_guard():
    """ad_guard() from notebook_code/adfilter.py without importing the notebook modules."""
    namespace = {}
    source = (HERE.parent / "notebook_code" / "adfilter.py").read_text(encoding="utf-8")
    start = source.index("def ad_guard(")
    exec(source[start:source.index("\ndef ", start + 1)], namespace)
    return namespace["ad_guard"]


def video_dirs(exports):
    """(video name, 01_slides dir) for every video folder of the exports; a walkthrough export
    holds the stages directly, its name comes from the segment ids."""
    for export in exports:
        export = Path(export)
        candidates = [export] if (export / "01_slides").is_dir() else sorted(
            path for path in export.iterdir() if (path / "01_slides").is_dir())
        for folder in candidates:
            intervals = json.loads((folder / "01_slides" / "intervals.json").read_text(encoding="utf-8"))
            yield intervals[0]["segment_id"].split(":")[0], folder


def indexed_ranges(folder, simulate, guard):
    """[(start, end)] of the time that reaches the index, and the ad report summary."""
    intervals = json.loads((folder / "01_slides" / "intervals.json").read_text(encoding="utf-8"))
    report_path = folder / "01_slides" / "ad_filter.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    suspect, flagged04 = set(), set()
    segments_path = folder / "04_bundle" / "segments.json"
    if segments_path.exists():
        segments = json.loads(segments_path.read_text(encoding="utf-8"))
        suspect = {s["segment_id"] for s in segments if s.get("ad_suspect")}
        flagged04 = {s["segment_id"] for s in segments if s.get("ad_flagged")}   # slide_speech (stage 04)
    duration = max(item["end_sec"] for item in intervals)
    ads = report.get("ad_intervals", [])
    info = {"mode": report.get("mode", "cut"), "ad_runs": len(ads),
            "ad_share": round(sum(b - a for a, b in ads) / duration, 3)}
    if simulate:
        reason = guard(ads, duration)
        info.update(mode="flag (simulated)", guard=reason)
        if reason:
            return [(0.0, duration)], info
        kept, cursor = [], 0.0
        for a, b in sorted(ads):
            if a > cursor:
                kept.append((cursor, a))
            cursor = max(cursor, b)
        return kept + ([(cursor, duration)] if cursor < duration else []), info
    if report.get("guard"):
        info["guard"] = report["guard"]["reason"]
    return [(item["start_sec"], item["end_sec"]) for item in intervals
            if not item.get("ad_flagged") and item["segment_id"] not in flagged04
            and (item["content_kind"] != "ad" or item["segment_id"] in suspect)], info


def score(per_second, ranges):
    """per_second: 1 = ad, 0 = lesson (one value per second, from t = 0)."""
    def indexed(second):
        middle = second + 0.5
        return any(a <= middle < b for a, b in ranges)
    lesson = [s for s, label in enumerate(per_second) if label == 0]
    ads = [s for s, label in enumerate(per_second) if label == 1]
    kept = sum(indexed(s) for s in lesson)
    left_out = sum(not indexed(s) for s in ads)
    return {"lesson_sec": len(lesson), "lesson_kept": round(kept / len(lesson), 4) if lesson else None,
            "lesson_lost_sec": len(lesson) - kept, "ad_sec": len(ads),
            "ad_left_out": round(left_out / len(ads), 4) if ads else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exports", nargs="+")
    parser.add_argument("--ad-free", default="", help="comma-separated videos with no ad at all")
    parser.add_argument("--simulate", action="store_true", help="predict phase 1 from an old cut run")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    gold = {v["video"]: v for v in json.loads(GOLD_ADS.read_text(encoding="utf-8"))["videos"]}
    ad_free = {name for name in args.ad_free.split(",") if name}
    guard = load_guard()
    rows = []
    for name, folder in video_dirs(args.exports):
        ranges, info = indexed_ranges(folder, args.simulate, guard)
        if name in gold:
            truth, split = gold[name]["per_second"], gold[name]["split"]
        elif name in ad_free:
            duration = max(b for _, b in ranges) if ranges else 0
            intervals = json.loads((folder / "01_slides" / "intervals.json").read_text(encoding="utf-8"))
            truth, split = [0] * int(max(item["end_sec"] for item in intervals)), "ad-free"
        else:
            print(f"bỏ qua {name}: không có nhãn quảng cáo (thêm --ad-free nếu biết chắc không có quảng cáo)")
            continue
        rows.append({"video": name, "split": split, **info, **score(truth, ranges)})
    header = f"{'video':12} {'split':9} {'mode':17} {'runs':>4} {'share':>6} {'kept':>7} {'lost s':>7} {'ad out':>7}  guard"
    print(header)
    for r in rows:
        print(f"{r['video']:12} {r['split']:9} {r['mode']:17} {r['ad_runs']:>4} {r['ad_share']:>6.1%} "
              f"{r['lesson_kept']:>7.2%} {r['lesson_lost_sec']:>7} "
              f"{'' if r['ad_left_out'] is None else format(r['ad_left_out'], '.2%'):>7}  {r.get('guard') or ''}")
    for split in sorted({r["split"] for r in rows}):
        group = [r for r in rows if r["split"] == split]
        lesson = sum(r["lesson_sec"] for r in group)
        kept = lesson - sum(r["lesson_lost_sec"] for r in group)
        ads = sum(r["ad_sec"] for r in group)
        out = sum(round((r["ad_left_out"] or 0) * r["ad_sec"]) for r in group)
        print(f"TỔNG {split:9} giữ bài giảng {kept / lesson:.2%} ({lesson - kept} s mất)"
              + (f" · loại quảng cáo {out / ads:.2%}" if ads else ""))
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
