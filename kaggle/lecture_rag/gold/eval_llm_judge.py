"""Content filter, LLM judge step 3: score the LLM labels of the suspect stretches.

    python kaggle/lecture_rag/gold/eval_llm_judge.py <candidates.json> <llm_judge.json> [--unlabelled=a,b,...]

A second is flagged when it lies in a stretch the LLM labelled with one of the kinds of a rule
(ad only; ad + channel; ad + channel + off_topic). Truth: gold ad_labels.json for L25; videos named
in --unlabelled have no labels (the share flagged is printed); every other video is ad-free.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--unlabelled=")]
    unlabelled = {v for a in sys.argv[1:] if a.startswith("--unlabelled=") for v in a.split("=", 1)[1].split(",")}
    videos = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    judged = json.loads(Path(args[1]).read_text(encoding="utf-8"))["results"]
    gold = {v["video"]: v for v in json.loads((HERE / "slide_boundaries" / "ad_labels.json").read_text(encoding="utf-8"))["videos"]}
    from collections import Counter
    print("nhãn LLM:", Counter(r["label"] for r in judged))
    rules = {"không lọc (chỉ đoạn nghi vấn)": None, "ad": {"ad"}, "ad + channel": {"ad", "channel"},
             "ad + channel + off_topic": {"ad", "channel", "off_topic"}}
    for label, kinds in rules.items():
        caught, kept_l25, kept_free, flagged_new = [], [], [], {}
        for video in videos:
            seconds = int(video["duration_sec"]) + 1
            mask = np.zeros(seconds, bool)
            for r in judged:
                if r["video"] == video["video"] and (kinds is None or r["label"] in kinds):
                    mask[int(r["start_sec"]):int(r["end_sec"]) + 1] = True
            name = video["video"]
            if name in gold:
                truth = np.array(gold[name]["per_second"][:seconds], bool)
                m = mask[:len(truth)]
                caught.append(float(m[truth].mean()) if truth.any() else 1.0)
                kept_l25.append(float((~m[~truth]).mean()))
            elif name in unlabelled:
                flagged_new[name] = float(mask.mean())
            else:
                kept_free.append((name, float(1 - mask.mean())))
        worst = min(kept_free, key=lambda r: r[1]) if kept_free else ("-", 1)
        print(f"{label:30} L25 bắt QC {np.mean(caught):.1%} · L25 giữ bài {np.mean(kept_l25):.1%} · "
              f"không-QC giữ bài TB {np.mean([k for _, k in kept_free]):.1%}, kém nhất {worst[1]:.1%} ({worst[0]}) · "
              f"kênh mới bị đánh dấu TB {np.mean(list(flagged_new.values())):.1%}")


if __name__ == "__main__":
    main()
