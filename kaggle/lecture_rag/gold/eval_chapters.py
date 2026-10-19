"""Score proposal (b): chapters + LLM labels/summaries (notebooks chapter_bench) for every embedder.

    cd backend
    .venv/Scripts/python ../kaggle/lecture_rag/gold/eval_chapters.py <chapter_bench.json> --unlabelled=a,b,...
        [--index L25_V008=<05_index dir> --index MLS-15DA1=<05_index dir>]

1. content filter: seconds inside chapters labelled ad (or ad/channel) are flagged; same scores as
   eval_llm_judge.py (L25 gold, ad-free videos, share flagged on unlabelled new channels).
2. cost: embedding + LLM seconds per 30 min of video.
3. two-level retrieval: rank chapters by E5 cosine of the question to "title. summary", keep the
   chunks inside the best TOP_CHAPTERS chapters, rank them by E5 cosine; compared with the same
   ranking over all chunks (one level). hit@1 / hit@3 of the slide on gold/qa/questions.json.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "backend"))
TOP_CHAPTERS = 2


def flag_scores(results, gold, unlabelled, kinds):
    caught, kept, free, new = [], [], [], {}
    for r in results:
        seconds = int(max(c["end_sec"] for c in r["chapters"])) + 1
        mask = np.zeros(seconds, bool)
        for c in r["chapters"]:
            if c["label"] in kinds:
                mask[int(c["start_sec"]):int(c["end_sec"])] = True
        name = r["video"]
        if name in gold:
            truth = np.array(gold[name]["per_second"][:seconds], bool)
            m = mask[:len(truth)]
            truth = truth[:len(m)]
            caught.append(float(m[truth].mean()) if truth.any() else 1.0)
            kept.append(float((~m[~truth]).mean()))
        elif name in unlabelled:
            new[name] = float(mask.mean())
        else:
            free.append((name, float(1 - mask.mean())))
    worst = min(free, key=lambda f: f[1]) if free else ("-", 1.0)
    return (f"L25 bắt QC {np.mean(caught):.1%} · L25 giữ bài {np.mean(kept):.1%} · không-QC giữ bài TB "
            f"{np.mean([f for _, f in free]):.1%}, kém nhất {worst[1]:.1%} ({worst[0]}) · kênh mới bị đánh dấu TB {np.mean(list(new.values())):.1%}")


def retrieval(results, indexes, questions):
    from app.services.retrieval import default_encoder
    encoder = default_encoder()
    out = {}
    for video, folder in indexes.items():
        folder = Path(folder)
        chunks = [json.loads(line) for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        vectors = np.load(folder / "embeddings.npy")
        segments = {s["segment_id"]: s for s in json.loads((folder / "lecture" / "segments.json").read_text(encoding="utf-8"))}
        for embedder in sorted({r["embedder"] for r in results}):
            chapters = next(r["chapters"] for r in results if r["video"] == video and r["embedder"] == embedder)
            chapter_vectors = np.array([encoder.encode(f"{c['title']}. {c['summary']}") for c in chapters])
            for mode in ("1 tầng", "2 tầng"):
                h1 = h3 = n = 0
                for item in questions[video]:
                    if item["answer"] is None:
                        continue
                    q = encoder.encode(item["q"])
                    allowed = np.ones(len(chunks), bool)
                    if mode == "2 tầng":
                        best = np.argsort(-(chapter_vectors @ q))[:TOP_CHAPTERS]
                        spans = [(chapters[i]["start_sec"], chapters[i]["end_sec"]) for i in best]
                        allowed = np.array([any(a <= c["start_sec"] < b for a, b in spans) for c in chunks])
                        if not allowed.any():
                            allowed[:] = True
                    scores = np.where(allowed, vectors @ q, -9)
                    slides = []
                    for i in np.argsort(-scores):
                        sid = chunks[i]["segment_id"]
                        if sid not in slides:
                            slides.append(sid)
                        if len(slides) == 3:
                            break
                    a, b = item["answer"]
                    inside = [segments[s]["start_sec"] < b and segments[s]["end_sec"] > a for s in slides]
                    h1 += inside[0]; h3 += any(inside); n += 1
                out[(video, embedder, mode)] = f"hit@1 {h1}/{n} · hit@3 {h3}/{n}"
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    unlabelled = {v for a in sys.argv[1:] if a.startswith("--unlabelled=") for v in a.split("=", 1)[1].split(",")}
    indexes = dict(a.split("=", 1)[1].split("=", 1) for a in sys.argv[1:] if a.startswith("--index="))
    results = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    gold = {v["video"]: v for v in json.loads((HERE / "slide_boundaries" / "ad_labels.json").read_text(encoding="utf-8"))["videos"]}
    for embedder in sorted({r["embedder"] for r in results}):
        rows = [r for r in results if r["embedder"] == embedder]
        minutes = sum(max(c["end_sec"] for c in r["chapters"]) for r in rows) / 60
        cost = sum(r["embed_sec"] + r["llm_sec"] for r in rows)
        print(f"\n== {embedder} · {np.mean([len(r['chapters']) for r in rows]):.1f} chương/video · "
              f"{cost / minutes * 30:.0f} s cho mỗi 30 phút video · nhãn {dict(Counter(c['label'] for r in rows for c in r['chapters']))}")
        print("  lọc QC = ad          ", flag_scores(rows, gold, unlabelled, {"ad"}))
        print("  lọc QC = ad+channel  ", flag_scores(rows, gold, unlabelled, {"ad", "channel"}))
    if indexes:
        questions = json.loads((HERE / "qa" / "questions.json").read_text(encoding="utf-8"))["videos"]
        print()
        for key, value in retrieval(results, indexes, questions).items():
            print(" ", " · ".join(key), "→", value)


if __name__ == "__main__":
    main()
