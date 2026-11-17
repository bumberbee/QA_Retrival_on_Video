import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import calibrate_q8  # noqa: E402


def question(score, kind="speech", hit3=True):
    return {"best_semantic": score, "type": kind, "hit3": hit3}


def test_table_counts_refusals_at_or_below_each_threshold():
    questions = ([question(0.90)] * 8 + [question(0.83), question(0.80, hit3=False)]
                 + [question(0.82, "unanswerable"), question(0.86, "unanswerable")])
    rows = {row["threshold"]: row for row in calibrate_q8.table(questions)}
    assert (rows[0.8]["false_refusals"], rows[0.8]["false_refusals_with_hit3"], rows[0.8]["caught"]) == (1, 0, 0)
    assert (rows[0.83]["false_refusals"], rows[0.83]["false_refusals_with_hit3"], rows[0.83]["caught"]) == (2, 1, 1)
    assert rows[0.86]["caught_rate"] == 1.0 and rows[0.86]["false_refusal_rate"] == 0.2


def test_choice_is_the_highest_threshold_within_the_limit():
    questions = [question(0.90)] * 9 + [question(0.83), question(0.82, "unanswerable")]
    rows = calibrate_q8.table(questions)
    assert calibrate_q8.choose(rows, 0.10)["threshold"] == 0.88
    assert calibrate_q8.choose(rows, 0.05)["threshold"] == 0.825
    assert calibrate_q8.choose([{**row, "false_refusal_rate": 1.0} for row in rows], 0.10) is None
