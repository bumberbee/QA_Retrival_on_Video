"""Table of the stage-01 threshold sweep (notebooks threshold_sweep): slide-boundary P/R/F1 per
change_threshold x stable_samples on gold set 1, tuning videos (choose) and held-out videos (report).

    python kaggle/lecture_rag/gold/eval_threshold_sweep.py <sweep_dir>

<sweep_dir>/c<change>_s<stable>/<video>/intervals.json, as unzipped from sweep_<n>.zip.
Each config is scored by gold/eval_ad_detection.py (main policy: S and SC are boundaries, ±1 s).
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def prf(tp, fp, fn):
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    sweep = Path(sys.argv[1])
    rows = []
    for config in sorted(p for p in sweep.iterdir() if p.is_dir()):
        subprocess.run([sys.executable, str(HERE / "eval_ad_detection.py"), str(HERE / "slide_boundaries"), str(config)],
                       check=True, capture_output=True)
        totals = json.loads((config / "eval.json").read_text(encoding="utf-8"))["totals"]
        change, stable = config.name[1:].split("_s")
        row = {"change": float(change), "stable": int(stable)}
        for split in ("tuning", "held-out"):
            p, r, f = prf(*totals[f"{split}|new"])
            row.update({f"{split}_P": round(p, 3), f"{split}_R": round(r, 3), f"{split}_F1": round(f, 3)})
        rows.append(row)
    rows.sort(key=lambda r: (r["stable"], r["change"]))
    best = max(rows, key=lambda r: r["tuning_F1"])
    print(f"{'khác':>6} {'yên':>4} | {'chỉnh P':>8} {'R':>6} {'F1':>6} | {'test P':>7} {'R':>6} {'F1':>6}")
    for r in rows:
        mark = "  ← chọn theo video chỉnh" if r is best else ("  ← hiện tại" if (r["change"], r["stable"]) == (0.055, 2) else "")
        print(f"{r['change']:>6} {r['stable']:>4} | {r['tuning_P']:>8} {r['tuning_R']:>6} {r['tuning_F1']:>6} | "
              f"{r['held-out_P']:>7} {r['held-out_R']:>6} {r['held-out_F1']:>6}{mark}")
    (sweep / "sweep_table.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
