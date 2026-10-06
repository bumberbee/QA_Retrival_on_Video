"""Score the web app's evidence search on the questions with known answers (no LLM call).

    cd backend
    .venv/Scripts/python scripts/eval_retrieval.py [--questions ../kaggle/lecture_rag/gold/qa/questions.json]

For each video of the question file that is in the app with a stage-05 index: hit@1 = the first
returned slide overlaps the answer range, hit@3 = one of the returned slides does. Questions
with answer null are out of the lecture: their best cosine is printed (lower is better).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import storage  # noqa: E402
from app.services import retrieval  # noqa: E402

QUESTIONS = Path(__file__).resolve().parents[2] / "kaggle" / "lecture_rag" / "gold" / "qa" / "questions.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    args = parser.parse_args()
    data = json.loads(args.questions.read_text(encoding="utf-8"))["videos"]
    for source_id, questions in data.items():
        video_id = storage.find_video_id_by_original_filename(f"{source_id}.mp4")
        if not video_id or not retrieval.has_index(video_id):
            print(f"Bỏ qua {source_id}: chưa có trong app hoặc chưa có chỉ mục")
            continue
        hit1 = hit3 = total = 0
        print(f"\n== {source_id} ==")
        for item in questions:
            found = retrieval.search(video_id, item["q"], item.get("current_time"))
            spans = [(s["start_sec"], s["end_sec"]) for s in found["slides"]]
            if item["answer"] is None:
                print(f"  ngoài bài  cos {found['best_semantic']:.3f}  {item['q'][:60]}")
                continue
            a, b = item["answer"]
            inside = [start < b and end > a for start, end in spans]
            total += 1
            hit1 += bool(inside[:1] and inside[0])
            hit3 += any(inside)
            mark = "✓" if inside[:1] and inside[0] else ("~" if any(inside) else "✗")
            first = f"{spans[0][0]:.0f}-{spans[0][1]:.0f}s" if spans else "-"
            print(f"  {mark} {first:>12}  {'[cắt cũ] ' if item.get('in_cut_ad') else ''}{item['q'][:60]}")
        print(f"  hit@1 {hit1}/{total} · hit@3 {hit3}/{total}")


if __name__ == "__main__":
    main()
