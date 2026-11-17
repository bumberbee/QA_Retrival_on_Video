"""Coverage/refusal table for the interim Q8 gate (best E5 cosine), from a dev evaluation.

Reads the ``retrieval.json`` written by eval_retrieval.py; loads no model and runs no search.
A question is refused when its best cosine is <= the threshold.

    python pipeline/eval/calibrate_q8.py <run>/retrieval.json [--max-false-refusal 0.10]
"""

import argparse
import json
from pathlib import Path

THRESHOLDS = [round(0.78 + 0.005 * n, 3) for n in range(21)]   # 0.78 .. 0.88


def table(questions: list) -> list[dict]:
    answerable = [q for q in questions if q["type"] != "unanswerable"]
    unanswerable = [q for q in questions if q["type"] == "unanswerable"]
    rows = []
    for threshold in THRESHOLDS:
        refused = [q for q in answerable if q["best_semantic"] <= threshold]
        caught = [q for q in unanswerable if q["best_semantic"] <= threshold]
        rows.append({"threshold": threshold, "answerable": len(answerable), "unanswerable": len(unanswerable),
                     "false_refusals": len(refused),
                     "false_refusals_with_hit3": sum(bool(q["hit3"]) for q in refused),
                     "false_refusal_rate": len(refused) / len(answerable) if answerable else 0.0,
                     "caught": len(caught),
                     "caught_rate": len(caught) / len(unanswerable) if unanswerable else 0.0})
    return rows


def choose(rows: list[dict], max_false_refusal: float) -> dict | None:
    """The highest threshold whose false-refusal rate stays within the limit."""
    allowed = [row for row in rows if row["false_refusal_rate"] <= max_false_refusal]
    return max(allowed, key=lambda row: row["threshold"]) if allowed else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--max-false-refusal", type=float, default=0.10)
    args = parser.parse_args(argv)
    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows = table(data["questions"])
    print(f"questions: {data.get('question_status', 'unknown status')}")
    print("threshold | answerable refused (retrieval was right) | unanswerable caught")
    for row in rows:
        print(f"{row['threshold']:.3f} | {row['false_refusals']}/{row['answerable']} "
              f"({row['false_refusal_rate']:.1%}), {row['false_refusals_with_hit3']} with Hit@3 | "
              f"{row['caught']}/{row['unanswerable']} ({row['caught_rate']:.0%})")
    chosen = choose(rows, args.max_false_refusal)
    print("chosen reject threshold:", chosen["threshold"] if chosen else "none within the limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
