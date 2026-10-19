"""Content filter without a learned ad model: flag a second as an ad only when SigLIP says the frame
is not lecture content (zero-shot slide score, signal_dump) AND the speech around it is off the
lecture topic (alignment.speech_relevance, E5 via the app's ONNX encoder).

    cd backend
    .venv/Scripts/python ../kaggle/lecture_rag/gold/eval_slide_speech.py <signals_dir> <export_dir> [...]

Videos need both a signal dump and a transcript (03_audio/transcript.json in an export). Truth:
gold ad_labels.json for L25, every second is lesson for the others (ad-free). Prints, per rule,
lesson kept and ads caught, as gold/eval_content_filter.py does.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "backend"))


def main():
    from app.services.retrieval import default_encoder

    args = [a for a in sys.argv[1:] if not a.startswith("--unlabelled=")]
    unlabelled = {v for a in sys.argv[1:] if a.startswith("--unlabelled=") for v in a.split("=", 1)[1].split(",")}
    signals, exports = Path(args[0]), [Path(p) for p in args[1:]]
    adfilter, alignment = {}, {"re": re}
    exec((HERE.parent / "notebook_code" / "adfilter.py").read_text(encoding="utf-8"), adfilter)
    exec((HERE.parent / "notebook_code" / "alignment.py").read_text(encoding="utf-8"), alignment)
    gold = {v["video"]: v for v in json.loads((HERE / "slide_boundaries" / "ad_labels.json").read_text(encoding="utf-8"))["videos"]}
    encoder = default_encoder()
    embed = lambda texts: np.array([encoder.encode(t) for t in texts])
    folders = {}
    for export in exports:
        for folder in ([export] if (export / "03_audio").is_dir() else export.iterdir()):
            if (folder / "03_audio" / "transcript.json").exists():
                folders[json.loads((folder / "03_audio" / "transcript.json").read_text(encoding="utf-8"))["video_id"]] = folder
    videos = []
    for name, folder in sorted(folders.items()):
        if not (signals / f"{name}.npz").exists():
            continue
        data = np.load(signals / f"{name}.npz")
        words = json.loads((folder / "03_audio" / "transcript.json").read_text(encoding="utf-8"))["words"]
        windows = alignment["speech_relevance"](alignment["speech_windows"](words), embed)
        seconds = len(data["times"])
        topic = np.full(seconds, 0.0)
        for second in range(seconds):   # lowest z of the windows covering this second
            near = [w["topic_z"] for w in windows if w["start_sec"] <= second < w["end_sec"]]
            topic[second] = min(near) if near else 0.0
        truth = np.array(gold[name]["per_second"][:seconds], bool) if name in gold else np.zeros(seconds, bool)
        videos.append((name, "gold" if name in gold else "unlabelled" if name in unlabelled else "free",
                       1 - data["gate"], topic, truth))
        print("đã chấm", name, flush=True)

    rules = {
        "chỉ hình: không-slide > 0,5": lambda ns, z: ns,
        "chỉ lời: z < -2": lambda ns, z: (z < -2).astype(float),
        "hình VÀ lời (ns > 0,5, z < -1,5)": lambda ns, z: ((ns > 0.5) & (z < -1.5)).astype(float),
        "hình VÀ lời (ns > 0,5, z < -1)": lambda ns, z: ((ns > 0.5) & (z < -1.0)).astype(float),
        "hình VÀ lời (ns > 0,3, z < -1)": lambda ns, z: ((ns > 0.3) & (z < -1.0)).astype(float),
        "gác bỏ OCR (ns > 0,85) VÀ lời z < -1,5": lambda ns, z: ((ns > 0.85) & (z < -1.5)).astype(float),
        "gác bỏ OCR (ns > 0,85) VÀ lời z < -2": lambda ns, z: ((ns > 0.85) & (z < -2.0)).astype(float),
        "hình VÀ lời (ns > 0,5, z < -2)": lambda ns, z: ((ns > 0.5) & (z < -2.0)).astype(float),
        "hình VÀ lời (ns > 0,7, z < -2)": lambda ns, z: ((ns > 0.7) & (z < -2.0)).astype(float),
        "hình VÀ lời (ns > 0,8, z < -2,5)": lambda ns, z: ((ns > 0.8) & (z < -2.5)).astype(float),
        "hình VÀ lời (ns > 0,9, z < -3)": lambda ns, z: ((ns > 0.9) & (z < -3.0)).astype(float),
    }
    for label, rule in rules.items():
        kept_l25, caught, kept_free, flagged_new = [], [], [], {}
        for name, labelled, not_slide, topic, truth in videos:
            mask = np.array(adfilter["ad_mask"](list(rule(not_slide, topic)), 0.5, 5, 5, 2), bool)
            n = min(len(mask), len(truth))
            mask, truth = mask[:n], truth[:n]
            if labelled == "unlabelled":
                flagged_new[name] = float(mask.mean())
            elif labelled == "gold":
                kept_l25.append(float((~mask[~truth]).mean()))
                caught.append(float(mask[truth].mean()))
            else:
                kept_free.append(float((~mask).mean()))
        print(f"{label:34} L25 bắt QC {np.mean(caught):.1%} · L25 giữ bài {np.mean(kept_l25):.1%}"
              + (f" · không-QC ({len(kept_free)} video) giữ bài TB {np.mean(kept_free):.1%}, kém nhất {min(kept_free):.1%}" if kept_free else "")
              + (f" · kênh mới bị đánh dấu TB {np.mean(list(flagged_new.values())):.1%}" if flagged_new else ""))
        if flagged_new and label.startswith("hình VÀ lời (ns > 0,5, z < -1,5)"):
            print("    kênh mới:", {k: f"{v:.1%}" for k, v in flagged_new.items()})
            worst = sorted(((n, float((~np.array(adfilter["ad_mask"](list(rule(ns_, z_)), 0.5, 5, 5, 2), bool)).mean()))
                            for n, l, ns_, z_, t_ in videos if l == "free"), key=lambda r: r[1])[:3]
            print("    không-QC giữ bài thấp nhất:", worst)


if __name__ == "__main__":
    main()
