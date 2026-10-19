"""Score the web app's evidence search on the questions with known answers (no LLM call).

    cd backend
    .venv/Scripts/python scripts/eval_retrieval.py [--questions ...] [--ablation] [--verbose]

For each video of the question file that is in the app with a stage-05 index: hit@1 = the first
returned slide overlaps the answer range, hit@3 = one of the returned slides does. Questions
with answer null are out of the lecture: their best cosine is printed (lower is better).
--ablation searches three ways (speech only, slide only = OCR + figure regions, both) and splits
the scores by where the answer is ("where": speech / ocr / both, when the question has it).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import storage  # noqa: E402
from app.services import retrieval  # noqa: E402

QUESTIONS = Path(__file__).resolve().parents[2] / "kaggle" / "lecture_rag" / "gold" / "qa" / "questions.json"
CONFIGS = {"lời giảng": frozenset({"speech"}), "slide (OCR + hình)": frozenset({"ocr", "region"}),
           "cả hai": None}


def score(video_id, items, sources, verbose):
    """{where: [hit@1, hit@3, n]} plus 'all', and the best cosines of out-of-lecture questions."""
    table = defaultdict(lambda: [0, 0, 0])
    outside = []
    for item in items:
        found = retrieval.search(video_id, item["q"], item.get("current_time"), sources=sources)
        if item["answer"] is None:
            outside.append(found["best_semantic"])
            continue
        a, b = item["answer"]
        inside = [s["start_sec"] < b and s["end_sec"] > a for s in found["slides"]]
        h1, h3 = bool(inside[:1] and inside[0]), any(inside)
        for key in ("all", item.get("where") or "?"):
            row = table[key]
            row[0] += h1; row[1] += h3; row[2] += 1
        if verbose:
            print(f"    {'✓' if h1 else ('~' if h3 else '✗')} [{item.get('where') or '?':6}] {item['q'][:70]}")
    return table, outside


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.questions.read_text(encoding="utf-8"))["videos"]
    configs = CONFIGS if args.ablation else {"cả hai": None}
    total = {name: defaultdict(lambda: [0, 0, 0]) for name in configs}
    for source_id, questions in data.items():
        video_id = storage.find_video_id_by_original_filename(f"{source_id}.mp4")
        if not video_id or not retrieval.has_index(video_id):
            print(f"Bỏ qua {source_id}: chưa có trong app hoặc chưa có chỉ mục")
            continue
        print(f"\n== {source_id} ==")
        for name, sources in configs.items():
            table, outside = score(video_id, questions, sources, args.verbose)
            h1, h3, n = table["all"]
            cos = f" · ngoài bài cos {max(outside):.3f}" if outside else ""
            print(f"  {name:20} hit@1 {h1}/{n} · hit@3 {h3}/{n}{cos}")
            for key, row in table.items():
                for k in range(3):
                    total[name][key][k] += row[k]
    print("\n== Tổng (hit@1 / hit@3) theo nơi chứa đáp án ==")
    keys = ["all", "speech", "ocr", "both"]
    print(f"  {'cấu hình':20} " + " ".join(f"{k:>16}" for k in keys))
    for name, table in total.items():
        cells = []
        for key in keys:
            h1, h3, n = table.get(key, [0, 0, 0])
            cells.append(f"{h1:>3}/{n:<3} {h3:>3}/{n:<3}" if n else f"{'-':>16}")
        print(f"  {name:20} " + " ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
