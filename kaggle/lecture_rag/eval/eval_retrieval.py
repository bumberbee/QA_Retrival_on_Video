"""Measure backend retrieval ablations and the actual text returned by build_context."""

import os

for _name in set(os.environ) | {"SUPABASE_DB_URL", "SUPABASE_URL", "SUPABASE_ANON_KEY",
                               "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"}:
    if _name.startswith(("SUPABASE_", "R2_")):
        os.environ[_name] = ""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from unittest.mock import patch

from offline_common import (ROOT, VIDEOS, encode_batches, experiment_dir, load_backend,
                            read_json, write_json)

AD_PHRASES = ("tuyển sinh", "học phí", "học bổng", "đăng ký kênh", "ủng hộ kênh",
              "kênh để ủng hộ", "subscribe", "hotline", "khuyến mãi", "siu.edu.vn",
              "trường quốc tế á châu", "saigon international university")
# Conservative catalogue: explicit contextual regressions documented in audit 3.
BAD_CORRECTIONS = {("trong kinh", "chọn kinh"), ("trọng một", "chọn một"),
                   ("chân không", "trạng không"), ("học chương", "học trường")}
TYPES = {"speech", "slide", "both", "numeric", "term", "code", "visual", "unanswerable"}


def overlap(start, end, spans):
    return any(start < b and end > a for a, b in spans)


def folded(text):
    return load_backend().normalize_search(text)


def tokens(text):
    return re.findall(r"[^\W_]+", folded(text))


def term_match(term, text, threshold=0.88):
    needle, haystack = tokens(term), tokens(text)
    if not needle:
        return False
    wanted = " ".join(needle)
    for i in range(len(haystack) - len(needle) + 1):
        candidate = " ".join(haystack[i:i + len(needle)])
        if candidate == wanted:
            return True
        # Numbers and short identifiers must match exactly (1.5 must never match 4.5).
        if not any(c.isdigit() for c in wanted) and len(wanted) >= 5:
            if SequenceMatcher(None, wanted, candidate, autojunk=False).ratio() >= threshold:
                return True
    return False


def repetition_runs(text, minimum=8):
    words = tokens(text)
    runs, i = [], 0
    while i < len(words):
        best = None
        for width in range(1, 9):
            phrase = words[i:i + width]
            if len(phrase) < width or not any(any(c.isalpha() for c in w) for w in phrase):
                continue
            end = i + width
            while words[end:end + width] == phrase:
                end += width
            count = (end - i) // width
            if count >= minimum and (best is None or end > best["end_token"]):
                best = {"phrase": " ".join(phrase), "repeats": count,
                        "start_token": i, "end_token": end}
        if best:
            runs.append(best)
            i = best["end_token"]
        else:
            i += 1
    return runs


def context_trace(found, index, question, context):
    """Track retained words/chunks through production's character slicing, without changing it."""
    if "context_evidence" in found:
        trace = evidence_context_trace(found, index, question, context)
        if found.get("context_legacy"):
            legacy = context_trace({"slides": found["slides"], "bounded_legacy": True}, index, question, found["context_legacy"])
            trace["recalled"] = trace["recalled"] or legacy["recalled"]
            trace["legacy"] = legacy
        return trace
    spans, entries = question["gold_spans"], []
    word_file = index.folder / "lecture" / "words.json"
    words = {w["word_id"]: w for w in read_json(word_file)} if word_file.exists() else {}
    for slide in found["slides"]:
        sid = slide["segment_id"]
        segment = index.by_segment.get(sid, {})
        full_speech, speech = slide["speech_text"], slide["speech_text"][:2500]
        ocr = slide["ocr_text"][:1800]
        if found.get("bounded_legacy"):
            start = context.find(f"Slide {sid.split(':')[-1]} · ")
            end = context.find("\n\n[", start) if start >= 0 else 0
            block = context[start:end if end >= 0 else len(context)] if start >= 0 else ""
            speech_start = block.find("Lời giảng:\n")
            speech = speech[:len(block[speech_start + len("Lời giảng:\n"):])] if speech_start >= 0 else ""
            ocr_start = block.find("Chữ trên slide (OCR, có thể sai số/dấu — đối chiếu ảnh nếu có):\n")
            if ocr_start < 0:
                ocr = ""
            else:
                body = block[ocr_start:].split("\n", 1)[1]
                ocr = ocr[:len(body)]
        selected, offset, missing = [], 0, []
        for wid in segment.get("speech", {}).get("word_ids", []):
            word = words.get(wid)
            if word is None:
                missing.append(wid)
                continue
            text = word["text"].strip()
            pos = full_speech.find(text, offset)
            if pos < 0:
                missing.append(wid)
                continue
            offset = pos + len(text)
            if offset <= len(speech):
                selected.append(word)
        retained_chunks = []
        for c in index.chunks:
            if c["segment_id"] != sid:
                continue
            body = speech if c["source_type"] == "speech" else ocr
            if c["source_type"] == "region":
                body = "\n".join(f[:600] for f in slide["figures"][:3])
            # Require most of the chunk's words to survive; keep partial overlaps explicit.
            ct, bt = tokens(c["text"]), tokens(body)
            share = SequenceMatcher(None, ct, bt, autojunk=False).find_longest_match(
                0, len(ct), 0, len(bt)).size / max(1, len(ct))
            if share > 0:
                retained_chunks.append({"chunk_id": c["chunk_id"], "source_type": c["source_type"],
                                        "start_sec": c["start_sec"], "end_sec": c["end_sec"],
                                        "retained_token_share": round(share, 4)})
        timed_words = [w for w in selected if overlap(w["start_sec"], w["end_sec"], spans)]
        if words and segment.get("speech", {}).get("word_ids"):
            speech_evidence = " ".join(w["text"] for w in timed_words)
            speech_time = bool(timed_words)
        else:
            speech_time = any(c["source_type"] == "speech" and c["retained_token_share"] >= 0.5
                              and overlap(c["start_sec"], c["end_sec"], spans) for c in retained_chunks)
            speech_evidence = speech if speech_time else ""
        visual_text = "\n".join([slide["title"], ocr])
        ocr_time = bool(visual_text.strip()) and overlap(slide["start_sec"], slide["end_sec"], spans)
        figure_text = "\n".join(f[:600] for f in slide["figures"][:3])
        terms = question["key_terms"]
        def checked(text, timing):
            matched = [term_match(t, text) and term_match(t, context) for t in terms]
            return {"time_overlap": timing, "term_matches": matched,
                    "recalled": timing and all(matched)}
        source = question.get("evidence_source", "either")
        checks = {"speech": checked(speech_evidence, speech_time),
                  "ocr": checked(visual_text, ocr_time),
                  "region": checked(figure_text, bool(figure_text) and ocr_time)}
        eligible = [checks[source]] if source in checks else list(checks.values())
        entries.append({"segment_id": sid, "speech_chars_included": len(speech),
                        "speech_chars_omitted": max(0, len(full_speech) - len(speech)),
                        "ocr_chars_included": len(ocr), "included_word_ids": [w["word_id"] for w in selected],
                        "gold_overlap_word_ids": [w["word_id"] for w in timed_words],
                        "unmapped_word_ids": missing, "retained_chunks": retained_chunks,
                        "checks": checks, "recalled": any(c["recalled"] for c in eligible)})
    return {"recalled": any(e["recalled"] for e in entries), "slides": entries,
            "definition": "Gold time overlap AND all annotated key terms in the retained evidence and build_context text."}


def retained_words(chunk, words):
    candidates = [w for w in words if w["start_sec"] < chunk["end_sec"] and w["end_sec"] > chunk["start_sec"]]
    word_tokens, owners, counts = [], [], {}
    for word in candidates:
        pieces = tokens(word["text"])
        word_tokens.extend(pieces)
        owners.extend([word["word_id"]] * len(pieces))
        counts[word["word_id"]] = len(pieces)
    included = defaultdict(int)
    for match in SequenceMatcher(None, word_tokens, tokens(chunk["text"]), autojunk=False).get_matching_blocks():
        for owner in owners[match.a:match.a + match.size]:
            included[owner] += 1
    return [w for w in candidates if counts[w["word_id"]] and included[w["word_id"]] == counts[w["word_id"]]]


def evidence_context_trace(found, index, question, context):
    """Use only retained blocks; neighbors keep their own segment and word-span timing."""
    word_file = index.folder / "lecture" / "words.json"
    words = read_json(word_file) if word_file.exists() else []
    entries = []
    by_segment = defaultdict(list)
    for chunk in found["context_evidence"]:
        by_segment[chunk["segment_id"]].append(chunk)
    for sid, chunks in by_segment.items():
        speech = [c for c in chunks if c["source_type"] == "speech"]
        selected = {w["word_id"]: w for c in speech for w in retained_words(c, words)}
        selected = sorted(selected.values(), key=lambda w: (w["start_sec"], w["end_sec"]))
        timed = [w for w in selected if overlap(w["start_sec"], w["end_sec"], question["gold_spans"])]
        speech_text = " ".join(w["text"] for w in timed) if words else "\n".join(
            c["text"] for c in speech if overlap(c["start_sec"], c["end_sec"], question["gold_spans"]))
        texts = {"speech": speech_text}
        for kind in ("ocr", "region"):
            texts[kind] = "\n".join(c.get("title", "") + "\n" + c["text"] for c in chunks
                                    if c["source_type"] == kind and overlap(
                                        c["start_sec"], c["end_sec"], question["gold_spans"]))
        checks = {}
        for kind, text in texts.items():
            matched = [term_match(t, text) and term_match(t, context) for t in question["key_terms"]]
            checks[kind] = {"time_overlap": bool(text), "term_matches": matched,
                            "recalled": bool(text) and all(matched)}
        source = question.get("evidence_source", "either")
        eligible = [checks[source]] if source in checks else list(checks.values())
        entries.append({"segment_id": sid, "included_word_ids": [w["word_id"] for w in selected],
                        "gold_overlap_word_ids": [w["word_id"] for w in timed], "unmapped_word_ids": [],
                        "retained_chunks": [{k: c[k] for k in ("chunk_id", "source_type", "start_sec", "end_sec",
                                                                 "selection_reason")} for c in chunks],
                        "checks": checks, "recalled": any(c["recalled"] for c in eligible)})
    return {"recalled": any(e["recalled"] for e in entries), "slides": entries,
            "definition": "Gold time overlap AND all key terms in actually retained evidence and context."}


def quality_lints(index):
    chunks = index.chunks
    promotions = [c["chunk_id"] for c in chunks
                  if any(folded(p) in folded(c["text"]) for p in AD_PHRASES)]
    ocr = [c for c in chunks if c["source_type"] == "ocr"]
    groups = defaultdict(list)
    for c in ocr:
        groups[c["text"]].append(c["chunk_id"])
    unique = [next(c for c in ocr if c["chunk_id"] == ids[0]) for ids in groups.values()]
    near = []
    for i, a in enumerate(unique):
        at = tokens(a["text"])
        ash = {tuple(at[j:j + 3]) for j in range(len(at) - 2)}
        if len(at) < 16:
            continue
        for b in unique[i + 1:]:
            bt = tokens(b["text"])
            bsh = {tuple(bt[j:j + 3]) for j in range(len(bt) - 2)}
            ratio = len(ash & bsh) / max(1, len(ash | bsh))
            if len(bt) >= 16 and ratio >= 0.9:
                near.append({"a": a["chunk_id"], "b": b["chunk_id"], "jaccard": round(ratio, 4)})
    loops = []
    for c in chunks:
        runs = repetition_runs(c["text"])
        if runs:
            loops.append({"chunk_id": c["chunk_id"], "source_type": c["source_type"], "runs": runs})
    folder = index.folder / "lecture"
    corrections = read_json(folder / "asr_corrections.json") if (folder / "asr_corrections.json").exists() else []
    words = read_json(folder / "words.json") if (folder / "words.json").exists() else []
    changed = [w for w in words if "text_asr" in w and w["text_asr"] != w["text"]]
    bad = [c for c in corrections if (c["heard"], c["fixed"]) in BAD_CORRECTIONS
           and any(abs(w["start_sec"] - c["at_sec"]) < 3 for w in changed)]
    empty = sum(not s.get("visual", {}).get("title_clean", s.get("visual", {}).get("title"))
                for s in index.segments)
    return {"chunks": len(chunks), "segments": len(index.segments), "ocr_chunks": len(ocr),
            "promotional_chunks": len(promotions), "promotional_chunk_ids": promotions,
            "exact_duplicate_ocr_excess": sum(len(ids) - 1 for ids in groups.values()),
            "exact_duplicate_ocr_groups": [ids for ids in groups.values() if len(ids) > 1],
            "near_duplicate_unique_ocr_pairs": len(near), "near_duplicate_pairs": near,
            "ocr_loop_chunks": sum(c["source_type"] == "ocr" for c in loops),
            "asr_loop_chunks": sum(c["source_type"] == "speech" for c in loops), "loops": loops,
            "chunks_under_32_tokens": sum(c["encoded_tokens"] < 32 for c in chunks),
            "empty_titles": empty, "empty_title_rate": empty / max(1, len(index.segments)),
            "empty_chunk_title_rate": sum(not c.get("title") for c in chunks) / max(1, len(chunks)),
            "asr_correction_events": len(corrections), "changed_asr_words": len(changed),
            "known_bad_asr_corrections": len(bad), "known_bad_events": bad,
            "known_bad_catalogue": sorted([list(x) for x in BAD_CORRECTIONS])}


def validate_questions(data):
    seen = set()
    for q in data["questions"]:
        if q["id"] in seen or q["type"] not in TYPES or q["video"] not in VIDEOS:
            raise ValueError(f"Invalid question: {q['id']}")
        seen.add(q["id"])
        if q["type"] == "unanswerable":
            if q["gold_spans"]:
                raise ValueError("Unanswerable questions must have no gold spans")
        elif not q["gold_spans"] or not q.get("key_terms") or not q["answer"]:
            raise ValueError(f"Missing gold evidence: {q['id']}")
        for a, b in q["gold_spans"]:
            if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b):
                raise ValueError(f"Invalid span: {q['id']}")


def aggregate(rows):
    answered = [r for r in rows if r["type"] != "unanswerable"]
    n = len(answered)
    return {"answerable": n, "unanswerable": len(rows) - n,
            "avg_context_chars": sum(len(r.get("context", "")) for r in answered) / n if n else None,
            "max_context_chars": max((len(r.get("context", "")) for r in answered), default=0),
            **{key: sum(r[key] for r in answered) / n if n else None
               for key in ("hit1", "hit3", "hit5", "mrr", "context_recall")},
            "unanswerable_top_scores": [{"id": r["id"], "semantic": r["best_semantic"],
                                         "rrf": r["top_rrf"]}
                                        for r in rows if r["type"] == "unanswerable"]}


def markdown(report):
    lines = ["# Retrieval evaluation", "", f"Questions: {report['question_status']}", "",
             "Hit@5 uses the unchanged production maximum of three slides, so it equals Hit@3.", "",
             "| Video / type | Answerable | Hit@1 | Hit@3 | Hit@5 | MRR | Context recall | Avg context chars |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, metrics in [*report["per_video"].items(), ("OVERALL", report["overall"]),
                           *[("type: " + k, v) for k, v in report["per_type"].items()]]:
        if not metrics["answerable"]:
            continue
        values = [f"{metrics[k]:.3f}" for k in ("hit1", "hit3", "hit5", "mrr", "context_recall")]
        lines.append(f"| {label} | {metrics['answerable']} | " + " | ".join(values) + f" | {metrics['avg_context_chars']:.0f} |")
    lines += ["", "| Video | Chunks | Ads* | Exact OCR excess | Near OCR pairs* | OCR / ASR loop chunks* | <32 tokens | Empty titles | Bad corrections* |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for video, lint in report["quality"].items():
        lines.append(f"| {video} | {lint['chunks']} | {lint['promotional_chunks']} | {lint['exact_duplicate_ocr_excess']} | "
                     f"{lint['near_duplicate_unique_ocr_pairs']} | {lint['ocr_loop_chunks']} / {lint['asr_loop_chunks']} | "
                     f"{lint['chunks_under_32_tokens']} | {lint['empty_titles']}/{lint['segments']} | {lint['known_bad_asr_corrections']} |")
    lines += ["", "*Diagnostic rules, not verified corpus error rates. Near pairs: unique OCR texts, 3-token Jaccard ≥0.90 and ≥16 tokens; loops: ≥8 consecutive repeats of 1–8 tokens. Ads: fixed phrase catalogue. Bad corrections: four documented phrase pairs, checked against text_asr changes.",
              "", "| Unanswerable ID | Top E5 cosine | Top fused RRF |", "|---|---:|---:|"]
    for row in report["overall"]["unanswerable_top_scores"]:
        lines.append(f"| {row['id']} | {row['semantic']:.4f} | {row['rrf']:.6f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index-root", type=Path, help="Root containing <video>/05_index")
    group.add_argument("--index-dir", type=Path, help="A single stage-05 index; use --videos VIDEO")
    parser.add_argument("--videos", nargs="+", default=VIDEOS, choices=VIDEOS)
    parser.add_argument("--questions", type=Path, default=ROOT / "eval/questions_dev.json")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, help="Root for evaluation reports")
    parser.add_argument("--features", nargs="*", choices=("ranked_context", "neighbors", "vietnamese_lexical",
                                                        "phrases", "dedup", "diversity", "no_boost"),
                        default=None, help="Omit for app defaults; empty list reproduces Phase 1.")
    args = parser.parse_args()
    if args.index_dir and len(args.videos) != 1:
        parser.error("--index-dir requires exactly one --videos value")
    data = read_json(args.questions)
    validate_questions(data)
    questions = [q for q in data["questions"] if q["video"] in args.videos]
    output = experiment_dir(args.name, args.output_root)
    folders = {v: args.index_dir or args.index_root / v / "05_index" for v in args.videos}
    backend = load_backend()
    options = backend.retrieval_options(args.features)
    encoder = backend.OnnxEncoder()
    vectors = encode_batches(encoder, [q["question"] for q in questions], "query: ")
    report = {"question_status": data["status"], "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
              "measured_utc": datetime.now(timezone.utc).isoformat(),
              "harness_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), ROOT / "eval/offline_common.py"]},
              "backend_sha256": hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest(),
              "encoder": {"model": backend.E5_REPO, "revision": backend.E5_REVISION, "device": "CPU ONNX"},
              "retrieval": {"max_slides": backend.MAX_SLIDES, "candidates": backend.CANDIDATES,
                            "rrf_k": backend.RRF_K, "semantic_weight": backend.SEMANTIC_WEIGHT,
                            "lexical_weight": backend.LEXICAL_WEIGHT, "source_boost": backend.SOURCE_BOOST,
                            "speech_chars": 2500, "ocr_chars": 1800,
                            "options": backend.asdict(options), "max_context_chars": backend.MAX_CONTEXT_CHARS},
              "questions": [], "quality": {},
              "index_inputs": {v: {"path": str(p.resolve()),
                                   "chunks_sha256": hashlib.sha256((p / "chunks.jsonl").read_bytes()).hexdigest()}
                               for v, p in folders.items()}}
    backend.load_index.cache_clear()
    with patch.object(backend, "index_dir", side_effect=lambda v: folders[v]):
        for video in args.videos:
            index = backend.load_index(video)
            report["quality"][video] = quality_lints(index)
        for question_index, (q, vector) in enumerate(zip(questions, vectors)):
            fixed = type("EncodedQuery", (), {"encode": lambda self, text, v=vector: v})()
            found = backend.search(q["video"], q["question"], encoder=fixed, options=options)
            context = backend.build_context(found)
            hits = [overlap(s["start_sec"], s["end_sec"], q["gold_spans"]) for s in found["slides"]]
            row = {"id": q["id"], "video": q["video"], "type": q["type"], "question": q["question"],
                   "best_semantic": found["best_semantic"],
                   "top_rrf": found["slides"][0]["score"] if found["slides"] else 0,
                   "slides": [{k: v for k, v in s.items() if k not in {"speech_text", "ocr_text", "figures"}}
                              for s in found["slides"]], "context": context,
                   "context_chars": len(context), "context_evidence": found.get("context_evidence", [])}
            if q["type"] != "unanswerable":
                trace = context_trace(found, backend.load_index(q["video"]), q, context)
                row.update({"hit1": int(any(hits[:1])), "hit3": int(any(hits[:3])),
                            "hit5": int(any(hits[:5])), "mrr": next((1 / (i + 1) for i, h in enumerate(hits) if h), 0),
                            "context_recall": int(trace["recalled"]), "context_trace": trace})
            report["questions"].append(row)
            if question_index + 1 == len(questions) or questions[question_index + 1]["video"] != q["video"]:
                print(q["video"], "evaluated", flush=True)
    backend.load_index.cache_clear()
    report["overall"] = aggregate(report["questions"])
    report["per_video"] = {v: aggregate([r for r in report["questions"] if r["video"] == v]) for v in args.videos}
    report["per_type"] = {t: aggregate([r for r in report["questions"] if r["type"] == t]) for t in sorted(TYPES)}
    write_json(output / "retrieval.json", report)
    (output / "retrieval.md").write_text(markdown(report), encoding="utf-8")
    if "torch" in sys.modules:
        raise RuntimeError("Unexpected torch import")
    print(json.dumps(report["overall"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
