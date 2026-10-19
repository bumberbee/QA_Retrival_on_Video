"""Content filter, LLM judge step 1: the suspect stretches an LLM should read.

    cd backend
    .venv/Scripts/python ../kaggle/lecture_rag/gold/build_llm_candidates.py <signals_dir> <out.json> <export_dir> [...]

A wide slide+speech rule (frame not lecture content > NOT_SLIDE, speech topic z < TOPIC_Z) gives
high-recall suspect stretches per video. For each one the file keeps its speech text and, as the
lecture context, the speech of the video's most on-topic windows. notebooks/llm_judge then asks
an LLM to label each stretch; gold/eval_llm_judge.py scores the result.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "backend"))
NOT_SLIDE, TOPIC_Z = 0.5, -1.5


def main():
    from app.services.retrieval import default_encoder

    signals, out, exports = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
    alignment = {"re": re}
    exec((HERE.parent / "notebook_code" / "alignment.py").read_text(encoding="utf-8"), alignment)
    encoder = default_encoder()
    embed = lambda texts: np.array([encoder.encode(t) for t in texts])
    folders = {}
    for export in exports:
        for folder in ([export] if (export / "03_audio").is_dir() else export.iterdir()):
            path = folder / "03_audio" / "transcript.json"
            if path.exists():
                folders[json.loads(path.read_text(encoding="utf-8"))["video_id"]] = folder
    videos = []
    for name, folder in sorted(folders.items()):
        if not (signals / f"{name}.npz").exists():
            continue
        data = np.load(signals / f"{name}.npz")
        transcript = json.loads((folder / "03_audio" / "transcript.json").read_text(encoding="utf-8"))
        windows = alignment["speech_relevance"](alignment["speech_windows"](transcript["words"]), embed)
        content = {"times": [float(t) for t in data["times"]], "scores": [float(s) for s in data["gate"]]}
        stretches = alignment["slide_speech_flags"](content, windows, not_slide=NOT_SLIDE, topic_z=TOPIC_Z)
        topic = " … ".join(w["text"] for w in sorted(windows, key=lambda w: -w["topic_z"])[:4])[:1500]
        candidates = []
        for start, end in stretches:
            words = [w["text"] for w in transcript["words"] if start <= w["start_sec"] < end]
            candidates.append({"start_sec": round(start, 2), "end_sec": round(end, 2),
                               "text": " ".join(words)[:1500]})
        videos.append({"video": name, "language": transcript.get("language"), "duration_sec": float(data["times"][-1]) + 1,
                       "topic": topic, "candidates": candidates})
        print(name, len(candidates), "đoạn nghi vấn", flush=True)
    out.write_text(json.dumps(videos, ensure_ascii=False, indent=1), encoding="utf-8")
    print("tổng", sum(len(v["candidates"]) for v in videos), "đoạn ·", out)


if __name__ == "__main__":
    main()
