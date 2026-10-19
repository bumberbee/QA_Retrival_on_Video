"""Scores stage-03 speech recognition against gold set 2 (speech), exported from asr_review.html.

    python kaggle/lecture_rag/gold/eval_asr.py <asr_gold.json>

Only clips ticked as checked count. Both sides are lower-cased with punctuation removed;
words marked [?] (not heard clearly) are dropped from the gold side.
  WER  word edit distance / gold word count (micro: summed over clips), per video and total
"""

import json
import re
import sys
import unicodedata
from collections import defaultdict


def words(text):
    text = unicodedata.normalize("NFC", text.lower()).replace("[?]", " ")
    return re.sub(r"[^\w\s]", " ", text).split()


def edits(ref, hyp):
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def main():
    items = [i for i in json.load(open(sys.argv[1], encoding="utf-8")) if i.get("checked")]
    per = defaultdict(lambda: [0, 0])
    for item in items:
        ref, hyp = words(item["gold"]), words(item["asr"])
        per[item["video_id"]][0] += edits(ref, hyp)
        per[item["video_id"]][1] += len(ref)
    for video, (e, n) in sorted(per.items()):
        print(f"{video:14} WER {e / max(n, 1):.3f}  ({e}/{n} từ)")
    e, n = map(sum, zip(*per.values())) if per else (0, 0)
    print(f"{'TỔNG':14} WER {e / max(n, 1):.3f}  ({e}/{n} từ, {len(items)} đoạn)")


if __name__ == "__main__":
    main()
