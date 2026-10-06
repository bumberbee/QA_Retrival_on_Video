"""Content filter B: does the speech-topic score (alignment.speech_relevance) tell ads from lesson?

    cd backend
    .venv/Scripts/python ../kaggle/lecture_rag/gold/eval_speech_relevance.py <export_dir> [...] [--ad-free MLS-15DA1]

Reads <video>/03_audio/transcript.json (+ 02_visual/visual.json for the slide text) of pipeline
exports, embeds 45 s speech windows with the app's ONNX E5 (CPU), and scores them against
gold/slide_boundaries/ad_labels.json: a window is "ad" when >= half of its seconds are ad.
Prints per video the AUC of -topic_z (higher = better separation), the AUC of ad_terms, and for
ad-free videos the share of windows with topic_z < Z_FLAG (should be ~0).
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "backend"))
Z_FLAG = -2.5


def load_alignment():
    import re
    namespace = {"re": re}
    for name in ("alignment.py",):
        exec((HERE.parent / "notebook_code" / name).read_text(encoding="utf-8"), namespace)
    return namespace


def auc(positive, negative):
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exports", nargs="+")
    parser.add_argument("--ad-free", default="")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    from app.services.retrieval import default_encoder
    import numpy as np

    encoder = default_encoder()
    embed = lambda texts: np.array([encoder.encode(t) for t in texts])
    code = load_alignment()
    gold = {v["video"]: v for v in json.loads((HERE / "slide_boundaries" / "ad_labels.json").read_text(encoding="utf-8"))["videos"]}
    ad_free = set(filter(None, args.ad_free.split(",")))
    folders = []
    for export in map(Path, args.exports):
        folders += [export] if (export / "03_audio").is_dir() else sorted(p for p in export.iterdir() if (p / "03_audio").is_dir())
    rows, all_pos, all_neg, all_tp, all_tn = [], [], [], [], []
    for folder in folders:
        transcript = json.loads((folder / "03_audio" / "transcript.json").read_text(encoding="utf-8"))
        name = transcript["video_id"]
        if name not in gold and name not in ad_free:
            continue
        visual_path = folder / "02_visual" / "visual.json"
        slides = []
        if visual_path.exists():
            slides = [v["ocr_text"] for v in json.loads(visual_path.read_text(encoding="utf-8"))
                      if v.get("content_kind") != "ad" and len(v.get("ocr_text") or "") >= 40][:40]
        windows = code["speech_relevance"](code["speech_windows"](transcript["words"]), embed, slides)
        row = {"video": name, "windows": len(windows)}
        if name in gold:
            per_second = gold[name]["per_second"]
            def is_ad(w):
                seconds = range(int(w["start_sec"]), min(int(w["end_sec"]), len(per_second)))
                return sum(per_second[s] for s in seconds) >= len(seconds) / 2 if seconds else False
            pos = [w for w in windows if is_ad(w)]
            neg = [w for w in windows if not is_ad(w)]
            row.update(ad_windows=len(pos), auc_topic=auc([-w["topic_z"] for w in pos], [-w["topic_z"] for w in neg]),
                       auc_terms=auc([w["ad_terms"] for w in pos], [w["ad_terms"] for w in neg]),
                       lesson_flagged=round(sum(w["topic_z"] < Z_FLAG for w in neg) / max(1, len(neg)), 3),
                       ads_flagged=round(sum(w["topic_z"] < Z_FLAG for w in pos) / max(1, len(pos)), 3))
            all_pos += [-w["topic_z"] for w in pos]; all_neg += [-w["topic_z"] for w in neg]
            all_tp += [w["ad_terms"] for w in pos]; all_tn += [w["ad_terms"] for w in neg]
        else:
            row.update(lesson_flagged=round(sum(w["topic_z"] < Z_FLAG for w in windows) / max(1, len(windows)), 3),
                       lowest=[(round(w["start_sec"]), w["topic_z"], w["text"][:70]) for w in sorted(windows, key=lambda w: w["topic_z"])[:3]])
        rows.append(row)
        print(row, flush=True)
    print(f"\nL25 gộp: AUC chủ đề {auc(all_pos, all_neg):.3f} · AUC từ quảng cáo {auc(all_tp, all_tn):.3f}")
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
