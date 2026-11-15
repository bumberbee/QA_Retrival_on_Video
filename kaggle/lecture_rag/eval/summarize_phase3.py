"""Write Phase-3 tables and question/evidence diagnostics from completed CPU experiments."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from offline_common import ROOT, read_json, write_json
from eval_retrieval import overlap, term_match

METRICS = ("hit1", "hit3", "mrr", "context_recall")
LINTS = ("chunks", "segments", "promotional_chunks", "exact_duplicate_ocr_excess",
         "near_duplicate_unique_ocr_pairs", "ocr_loop_chunks", "asr_loop_chunks",
         "chunks_under_32_tokens", "empty_titles", "asr_correction_events", "changed_asr_words",
         "known_bad_asr_corrections")


def totals(report):
    return {key: sum(q[key] for q in report["quality"].values()) for key in LINTS}


def metric_cells(row):
    return " | ".join(f"{row[key]:.3f}" for key in METRICS)


def drops(before, after):
    return [key for key in METRICS if after[key] < before[key] - 1e-9]


def gold_terms(chunks, question):
    source = question["evidence_source"]
    relevant = [chunk for chunk in chunks if overlap(chunk["start_sec"], chunk["end_sec"], question["gold_spans"])
                and (source == "either" or source == "speech" and chunk["source_type"] == "speech"
                     or source == "ocr" and chunk["source_type"] == "ocr")]
    return [term for term in question["key_terms"]
            if any(term_match(term, chunk["text"]) for chunk in relevant)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--final", required=True, help="Final rebuild experiment name, without -eval")
    parser.add_argument("--experiment", action="append", default=[], help="Override one variant's experiment as variant=name")
    parser.add_argument("--variants", nargs="+", default=["reference", "ocr", "titles", "asr-off", "asr-glossary", "failures", "ads", "combined"])
    args = parser.parse_args()
    exp = ROOT / "runs/exp"
    experiments = {variant: args.prefix + "-" + variant for variant in args.variants}
    for value in args.experiment:
        variant, name = value.split("=", 1)
        if variant not in experiments:
            parser.error(f"Unknown variant override: {variant}")
        experiments[variant] = name
    reports = {variant: read_json(exp / (experiments[variant] + "-eval") / "retrieval.json") for variant in args.variants}
    final = read_json(exp / (args.final + "-eval") / "retrieval.json")
    reports["Final defaults"] = final
    reference = reports["reference"]
    final_build = read_json(exp / args.final / "rebuild_report.json")
    config = final_build["config"]
    questions = read_json(ROOT / "eval/questions_dev.json")["questions"]
    gold = {q["id"]: q for q in questions}
    reviewed_questions = read_json(ROOT / 'eval/phase3_question_review.json')
    base_rows = {q["id"]: q for q in reference["questions"]}
    chunks = {}
    for name, report in reports.items():
        chunks[name] = {video: [json.loads(line) for line in
                               (Path(info["path"]) / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
                        for video, info in report["index_inputs"].items()}
    changes = []
    for name, report in reports.items():
        if name == "reference":
            continue
        for row in report["questions"]:
            before = base_rows[row["id"]]
            metric_change = row["type"] != "unanswerable" and any(row[k] != before[k] for k in METRICS)
            ranking_changed = [s["segment_id"] for s in row["slides"]] != [s["segment_id"] for s in before["slides"]]
            if not metric_change and not ranking_changed and row["context"] == before["context"]:
                continue
            q = gold[row["id"]]
            old_terms = gold_terms(chunks["reference"][row["video"]], q)
            new_terms = gold_terms(chunks[name][row["video"]], q)
            missing = [term for term in old_terms if term not in new_terms]
            gained = [term for term in new_terms if term not in old_terms]
            note = ("Gold-overlap chunk terms removed: " + ", ".join(missing) if missing else
                    "Gold-overlap chunk terms gained: " + ", ".join(gained) if gained else
                    "Gold key-term availability unchanged; ranking/context selection changed.")
            changes.append({"variant": name, "id": row["id"], "type": row["type"],
                            "gold_spans": q["gold_spans"], "key_terms": q["key_terms"],
                            "before": {k: before.get(k) for k in METRICS}, "after": {k: row.get(k) for k in METRICS},
                            "before_slides": [s["segment_id"] for s in before["slides"]],
                            "after_slides": [s["segment_id"] for s in row["slides"]],
                            "metric_change": metric_change, "ranking_changed": ranking_changed,
                            "gold_chunk_terms_before": old_terms, "gold_chunk_terms_after": new_terms,
                            "note": note,
                            **({"review_note": reviewed_questions[row['id']]} if name == 'Final defaults'
                               and row['id'] in reviewed_questions else {})})
    output = exp / args.final
    write_json(output / "question_changes.json", changes)
    write_json(output / "phase3_matrix.json", {name: {"overall": r["overall"], "quality": totals(r),
                                                    "per_video_quality": r["quality"],
                                                    "per_video": r["per_video"], "per_type": r["per_type"]}
                                               for name, r in reports.items()})
    lines = ["# Phase 3: CPU text processing", "",
             "All measurements use the phase-2 default backend retrieval, the unchanged 120-question development set (110 answerable), and raw stage-02 OCR / stage-03 ASR from the read-only export. Questions are auto-written and still need human review. These are development comparisons, not held-out results. No per-video threshold, question, gold span, retrieval weight, or candidate limit was tuned. No GPU, download, package installation, commit, push, branch switch, `docs/` edit, or `backend/app/` edit was performed.", "",
             f"Question SHA-256: `{reference['questions_sha256']}`. Backend retrieval SHA-256: `{reference['backend_sha256']}`. Stage-01 detection and backend router/storage work by other agents are outside this text experiment; raw stage-01 intervals are always taken from the export. Source fingerprints may record their concurrent file changes without executing stage 01.", "",
             "## Overall ablations", "",
             "Each alone row enables only that text change relative to the exported-index + phase-2 retrieval reference. Full frozen configs are in `eval/phase3_configs/`. Hit@5 equals Hit@3 because retrieval still returns at most three slides. DROP means any metric decreased relative to the reference.", "",
             "| Variant | Hit@1 | Hit@3 | MRR | Context recall | Chunks | Flag |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, report in reports.items():
        lines.append(f"| {name} | {metric_cells(report['overall'])} | {totals(report)['chunks']} | {'DROP: ' + ', '.join(drops(reference['overall'], report['overall'])) if drops(reference['overall'], report['overall']) else ''} |")
    lines += ["", "The combination names describe successive global policies, not interchangeable final defaults. The figure interaction uses the same prototype config with and without figures; it predates the final subscript and Asian School signature refinements.", "",
              "| Variant | OCR | Heading heuristic | ASR | Failure guard | Speech ads | Generated / semantic-only |",
              "|---|---|---|---|---|---|---|"]
    for name in args.variants:
        settings = read_json(exp / experiments[name] / 'rebuild_report.json')['config']
        lines.append(f"| {name} | {settings['conservative_ocr']} | {settings['heading_titles']} | "
                     f"{settings['asr_correction']} | {settings['failure_guard']} | {settings['speech_ad_filter']} | "
                     f"{settings['include_generated_descriptions']} / {settings.get('generated_semantic_only', False)} |")
    lines += ["", "### Kept / rejected", "",
              f"The final defaults are the measured `{args.final}` config; the final row repeats that measured configuration. Selection uses global flags, not per-video parameters. All regressions, including the retained config's group/question regressions, remain visible below.", "",
              "| Policy | Decision | Measurement / reason |", "|---|---|---|",
              f"| Conservative OCR | {'Kept' if config['conservative_ocr'] else 'Rejected'} | Initial layout-only ablation preserves H1/recall but H3 falls by 1/110 and MRR by 0.003. The additional subscript-digit ablation preserves chemical formula spellings recognized by lexical retrieval. It must be paired with failure detection so raw loops are not revived as evidence. |",
              f"| Heading heuristic | {'Kept' if config['heading_titles'] else 'Rejected as default; opt-in remains'} | Alone: {metric_cells(reports['titles']['overall']).replace(' | ', ' / ')} versus reference {metric_cells(reference['overall']).replace(' | ', ' / ')}. Its first-block heuristics do not justify a default ranking regression; an explicit visible-heading field needs a GPU extraction change. |",
              f"| OCR-driven ASR correction | `{config['asr_correction']}` kept; unrestricted legacy rejected | Off, local glossary and lecture glossary are measured separately; their retrieval losses are visible. The final reviewed policy adds only a literal local printed-code feedback repair. See its alone row for the no-regression check; the known-bad catalogue must be zero. |",
              f"| Failure detector | {'Kept' if config['failure_guard'] else 'Rejected'} | Corpus-integrity fix: excludes known failed OCR and loop words with retry provenance. Some image-backed gold OCR terms disappear; that tradeoff is documented rather than relabelled. |",
              f"| Promotional speech filter | {'Kept' if config['speech_ad_filter'] else 'Rejected'} | Exact bounded spots and subscription calls, with full removed/partial chunk review and no gold-interval overlap. |",
              f"| Generated figures | {'Kept as semantic-only' if config['include_generated_descriptions'] else 'Rejected as default; opt-in remains'} | Alone produces no change in any overall or visual metric while context grows. Combination interactions are measured separately. Full-frame, unverified descriptions have no demonstrated standalone benefit on this set. |"]
    lines += ["", "## Policies and decisions", "",
              "| Change | Final setting | Why / limit |", "|---|---|---|",
              f"| Conservative OCR | `{config['conservative_ocr']}` | Repetition alone cannot identify boilerplate. Keeps headings, assignments, formulas, bare numbers and symbols. Only explicit attribution, branding and prefixed page-number rules remove lines. canonical_subscripts maps subscript digits to their plain-digit equivalent outside code/quoted lines, preserving powers and raw OCR. All Qwen boxes cover the full frame, so positional deletion is unsupported. |",
              f"| Heading titles | `{config['heading_titles']}` | Legacy title extraction remains the default after the measured regression. The opt-in heuristic examines the first six lines, allows short headings, rejects code/body/UI, and returns empty when none qualifies. |",
              f"| ASR correction | `{config['asr_correction']}` | Reviewed terms: xác suất, áp suất, nhiệt độ sôi, chi phí, cao trào, mặt trận and suất điện động, with local or repeated lecture OCR support. Only literal printed-code feedback adds the reviewed sai rồi alias. Punctuation boundaries and atomic matching prevent partial/overlapping repairs; every word retains raw text_asr. |",
              f"| Failure guard | `{config['failure_guard']}` | Detects long repeats (OCR phrases up to 64 tokens; ASR up to 8), drops only loop words for speech, and excludes dominated/placeholder OCR including its title and regions. Partial healthy OCR stays flagged. Raw words, timings and ownership survive. |",
              f"| Speech ads | `{config['speech_ad_filter']}` | SIU: institution/brand to closing slogan, at most 240 seconds. Asian School: exact name plus its observed opening (tự hào là ngôi trường mang đến nền giáo dục tiên tiến) and closing (tương lai của chính mình). Literal subscription calls are separate. Keeps surrounding teaching words. AD_FILTER remains None: slide_speech needs missing stage-01 content_scores and risks coarse exclusions. |",
              f"| Generated figures | `{config['include_generated_descriptions']}` | Measured alone and with the combination. Separate region chunks retain model_generated_unverified provenance, reject no-figure/empty/loop output, and use semantic-only postings. They can contribute semantic RRF weight 1.0, rather than both branches' 1.0 + 0.5. This reduces fusion opportunity, not a guaranteed rank ceiling. Full-frame generated interpretations still need human review. |", "",
              "The fixed overlapping-match implementation prefers longer phrases and permits compatible overlap; the adopted glossary matches the complete term atomically. Thus tính sắc xuất becomes tính xác suất, rather than tính xác xuất. The legacy greedy path is used only by explicit reference ablations.", "",
              "The local glossary requires canonical OCR in the same segment; lecture support requires canonical OCR in at least two segments of that lecture. The code-feedback extension repairs xai rồi → sai rồi only when the local OCR contains a literal print message starting with sai roi (accents folded). It cannot use distant lecture support. That extension was reviewed after the lone lecture-glossary regression, y1CvB950VlU_q01: changing a distractor's speech changed its rank, while the correct triple-quote gold still lacked a4. The printed feedback and raw ASR support the repair; it is a globally applied rule, not a video exception. These policy choices use development evidence and still require held-out validation.", "",
              "Conservative parsing recovers raw Qwen indentation and the index splits OCR by complete lines to preserve it. The exported Python raw answers contain 24 indented code lines across 14 records. This cannot restore indentation or quotes that Qwen never produced, including seg_00017 and seg_00005.", "",
              "Raw layout recovery also restores subscript glyphs that the old tokenizer round-trip normalized: Cl₂, NaHCO₃ and CO₂. Those caused two chemistry context-recall losses in the initial combinations even though the formulas were still present. The subscript-digit ablation measures an equivalent plain-text spelling in stage 04; the harness/gold identifier matcher is unchanged. This normalization excludes code/quoted lines and does not convert superscript powers such as x² into x2.", "",
              "The initial ad filter and bounded_schools prototype use SIU closing slogans and still miss the Asian School spot. The school_signatures refinement uses that spot's actual repeated opening and closing, verified from raw speech in all three L25 videos; no school mention or general education topic alone triggers exclusion. The separate prototype and final-signature ablations expose this change rather than retroactively crediting the earlier filter.", "",
              "Generated figures alone leave all four visual metrics unchanged. In the prototype combination, L25_V011_q06 (the UN mind map) moves from rank 3 to 2: visual MRR 0.383 → 0.400, overall MRR +0.0015, with no H1/H3/recall gain. It adds 181 chunks and increases average context by 143 characters. This small development interaction is retained in the report; the production flag stays off pending verified visual evidence.", "",
              "## Corpus lints", "",
              "Counts are diagnostics, not measured error rates. Empty titles can be correct when the frame has no heading; duplicate OCR is retained for lesson completeness and deduplicated by phase-2 retrieval. Loop lints count indexed chunks, while retry records refer to raw failures.", "",
              "| Lint | " + " | ".join(reports) + " |", "|---|" + "---:|" * len(reports)]
    for key in LINTS:
        lines.append("| " + key + " | " + " | ".join(str(totals(r)[key]) for r in reports.values()) + " |")
    lines += ["", "Final corpus by video (all variants' detailed per-video lints and IDs are preserved in phase3_matrix.json):", "",
              "| Video | Chunks | Promotional lint | OCR / ASR loop chunks | Bad corrections | Exact OCR excess | Near OCR pairs | Empty titles |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for video, quality in final['quality'].items():
        lines.append(f"| {video} | {quality['chunks']} | {quality['promotional_chunks']} | "
                     f"{quality['ocr_loop_chunks']} / {quality['asr_loop_chunks']} | {quality['known_bad_asr_corrections']} | "
                     f"{quality['exact_duplicate_ocr_excess']} | {quality['near_duplicate_unique_ocr_pairs']} | {quality['empty_titles']} |")
    lines += ["", "## Per-video and per-type results", "",
              "Each table includes all variants; cells are Hit@1 / Hit@3 / MRR / context recall. A ↓ marks any regression from the reference."]
    names = [name for name in reports if name != "reference"]
    for field in ("per_video", "per_type"):
        lines += ["", "### " + field.replace("per_", "").title()]
        for start in range(0, len(names), 4):
            block = ["reference", *names[start:start + 4]]
            lines += ["", "| Group | N | " + " | ".join(block) + " |",
                      "|---|---:|" + "---|" * len(block)]
            for group, base in reference[field].items():
                if not base['answerable']:
                    continue
                cells = []
                for name in block:
                    row = reports[name][field][group]
                    cells.append(" / ".join(f"{row[k]:.3f}" for k in METRICS) + (" ↓" if drops(base, row) else ""))
                lines.append(f"| {group} | {base['answerable']} | " + " | ".join(cells) + " |")
    lines += ["", "## Changed questions and gold evidence", "",
              "`question_changes.json` in the final experiment records every question whose ranking, context or metric changed, with gold spans, before/after slides and the key terms present in source-compatible chunks overlapping gold. This availability diagnostic is coarser than the unchanged context-recall trace; it does not fabricate new gold or give credit for deleted words. All full contexts and retained-word traces remain in each `*-eval/retrieval.json`.", "",
              "Final metric changes are listed below. A loss with unchanged key-term availability reflects competition or bounded context selection; a removed term requires checking the raw evidence/retry record. Gold times and answers were not edited.", "",
              "| Question | Reference H1/H3/MRR/recall | Final H1/H3/MRR/recall | Flag / evidence check |",
              "|---|---|---|---|"]
    for change in changes:
        if change['variant'] != 'Final defaults' or not change['metric_change']:
            continue
        a, b = change['before'], change['after']
        lines.append(f"| {change['id']} | " + " / ".join(f"{a[k]:.3f}" for k in METRICS) + " | " +
                     " / ".join(f"{b[k]:.3f}" for k in METRICS) + f" | {'DROP; ' if drops(a,b) else ''}{change['note']} " +
                     change.get('review_note', '') + " |")
    retries = read_json(output / "gpu_retry_segments.json")
    reviewed = read_json(ROOT / "eval/phase3_gpu_review.json")
    write_json(output / "gpu_retry_review.json", reviewed)
    lines += ["", "## GPU retry list", "",
              f"Automatic retry records: {len(retries)}. Reviewed audit entries: {len(reviewed)} (some split across segment boundaries). Both lists are in the final experiment as `gpu_retry_segments.json` and `gpu_retry_review.json`; the latter is also tracked at `eval/phase3_gpu_review.json`. Automatic flags change evidence eligibility; audit entries are review metadata only. Deduplicate by source/video/segment when scheduling, while retaining all reasons and the narrower ASR time spans.", "",
              "| Video / segment | Source | Seconds | Reason |", "|---|---|---:|---|"]
    for item in [*retries, *reviewed]:
        lines.append(f"| {item['segment_id']} | {item['source']} | {item['start_sec']:.2f}–{item['end_sec']:.2f} | " +
                     ", ".join(item['reasons']) + ("; " + item['note'] if item.get('note') else '') + " |")
    lines += ["", "CPU cleaning cannot recover missing true/false statements, correct wrong handwritten numbers/quotes, replace invented methane evidence with the spacecraft question, resolve UI occlusion, or hear missing Bayes/len/for/end/PPF correctly. Stage-01 merged board states and wrong keyframe selection require video/frame work as well as GPU extraction. The retry lists are not a claim to detect every model error.", "",
              "## Promotional review", ""]
    ad_totals, ad_chunks, ad_gold_overlaps = Counter(), [], []
    for video in final['per_video']:
        filter_path = output / video / '04_bundle/content_filter.json'
        if filter_path.exists():
            for span in read_json(filter_path).get('excluded_spans', []):
                for q in questions:
                    if q['video'] == video and overlap(span['start_sec'], span['end_sec'], q['gold_spans']):
                        ad_gold_overlaps.append({'question_id': q['id'], 'span': span})
        path = output / video / '05_index/ad_chunk_review.json'
        if path.exists():
            data = read_json(path)
            ad_chunks.extend(data['chunks'])
            for key in ('removed_chunks', 'partially_filtered_chunks', 'chunk_count_delta'):
                ad_totals[key] += data[key]
    lines += [f"Relative to the same final text config before ad filtering: {ad_totals['removed_chunks']} whole speech chunks removed, {ad_totals['partially_filtered_chunks']} original chunks partially filtered, net chunk delta {ad_totals['chunk_count_delta']:+d}. IDs/text/times and retained/excluded word IDs are listed in each video’s `05_index/ad_chunk_review.json`; exact matched speech is in `04_bundle/content_filter.json`. Chunk IDs can shift when units regroup, so the review uses original pre-filter chunks and word provenance.", "",
              "Conservative misses remain: introductory/closing promotional fragments outside the explicit slogan boundary, recognition drift before the first chemistry brand anchor, and school mentions without a bounded spot. This speech filter does not remove OCR-only promotional frames unless their lines match the conservative branding rules; the promotional lint lists remaining evidence for review. No labelled Kaggle ad data was downloaded. False exclusions of teaching are treated as worse than missed promotional fragments.", "",
              f"Matched promotional spans overlap {len(ad_gold_overlaps)} annotated gold evidence intervals. This is a useful check on this draft question set, not a labelled ad precision estimate. Any overlaps are recorded in ad_gold_overlap_review.json. The following original chunks were removed or modified; partial rows retain their teaching words.", "",
              "| Original chunk | Seconds | Action | Original text excerpt |", "|---|---:|---|---|"]
    for chunk in ad_chunks:
        excerpt = chunk['text'][:110].replace('\n', ' ').replace('|', '\\|')
        lines.append(f"| {chunk['chunk_id']} | {chunk['start_sec']:.2f}–{chunk['end_sec']:.2f} | {chunk['review_status']} | {excerpt}… |")
    write_json(output / 'ad_gold_overlap_review.json', ad_gold_overlaps)
    lines += ["", "Remaining promotional-lint matches are conservative misses, all retained for review:", "",
              "| Chunk | Source | Seconds | Text excerpt |", "|---|---|---:|---|"]
    for video, quality in final['quality'].items():
        by_id = {c['chunk_id']: c for c in chunks['Final defaults'][video]}
        for chunk_id in quality['promotional_chunk_ids']:
            chunk = by_id[chunk_id]
            excerpt = chunk['text'][:140].replace('\n', ' ').replace('|', '\\|')
            lines.append(f"| {chunk_id} | {chunk['source_type']} | {chunk['start_sec']:.2f}–{chunk['end_sec']:.2f} | {excerpt} |")
    lines += ["",
              "## Reproduction and validation", "",
              "The runner calls rebuild_offline.py and eval_retrieval.py for every fresh variant, with phase-2 options omitted to use defaults. Exact (prefix,text) CPU ONNX vectors are reused in memory across variants only to save execution cost; all vectors originate from the pinned real encoder and remain unit normalized. Warm reads use only completed per-video stage-05 outputs, even when a later video in that experiment was interrupted. Audio is restored by the unchanged normalization path and export/MP4 checksums are verified. See BASELINE.md for the restored-WAV byte-identity limitation. Rebuild refuses existing staged inputs.", "",
              "Repeated asset copies filled C: and interrupted several exploratory runs. The harness now shares only checksum-identical manifest assets with a completed workspace reference via NTFS hard links; both source and destination hashes are verified before replacement, differing text/vectors remain independent, and resolved write paths must stay in runs/exp. Incomplete stages are never repaired or counted as results. Existing archived experiment junctions on D: are read only. No source export asset was changed.", "",
              "```powershell", "& backend/.venv/Scripts/python.exe kaggle/lecture_rag/eval/run_phase3.py --prefix phase3-reproduction --audio-reference kaggle/lecture_rag/runs/exp/phase3-reproduction-reference --variants " + " ".join(args.variants),
              "& backend/.venv/Scripts/python.exe kaggle/lecture_rag/eval/rebuild_offline.py --export-root D:/thesis-data/run-20261020/export --name phase3-production-reproduction --config kaggle/lecture_rag/eval/production_config.json --audio-reference kaggle/lecture_rag/runs/exp/phase3-a-reference --mp4-root D:/thesis-data/mp4-extra --ffmpeg D:/FilmTools/Shotcut/ffmpeg.exe",
              "& backend/.venv/Scripts/python.exe kaggle/lecture_rag/eval/eval_retrieval.py --index-root kaggle/lecture_rag/runs/exp/phase3-production-reproduction --name phase3-production-reproduction-eval",
              "python kaggle/lecture_rag/build_notebooks.py", "python kaggle/lecture_rag/build_fast_pipeline.py",
              "python kaggle/lecture_rag/build_full_walkthrough.py", "python kaggle/lecture_rag/build_run_pipeline.py",
              "$env:PATH = 'D:/FilmTools/Shotcut;' + $env:PATH", "python -m pytest kaggle/lecture_rag/tests -q",
              "& backend/.venv/Scripts/python.exe -m unittest discover -s kaggle/lecture_rag/tests -p test_eval_harness.py -q",
              "& backend/.venv/Scripts/python.exe -m unittest discover -s kaggle/lecture_rag/tests -p test_text_phase3.py -q",
              "& backend/.venv/Scripts/python.exe -m unittest discover -s backend/tests -q", "```", "",
              "Use fresh names when reproducing. Per-run config/code/input hashes, restored audio provenance and rebuild comparisons are in rebuild_report.json. The reference rebuild must retain full text/ID/token parity and cosine >0.999; production defaults and the captured JSON are checked by a focused test. Generated notebooks embed the current sources."]
    lines += ["", "Actual experiment mapping (all interrupted partial runs are excluded):", "",
              "```json", json.dumps(experiments, indent=2), "```", "",
              "To reproduce this report from those existing artifacts:", "", "```powershell",
              f"& backend/.venv/Scripts/python.exe kaggle/lecture_rag/eval/summarize_phase3.py --prefix {args.prefix} --final {args.final} --variants " +
              " ".join(args.variants) + " " + " ".join("--experiment " + v + "=" + n for v, n in experiments.items() if n != args.prefix + "-" + v), "```"]
    validation_path = ROOT / 'eval/phase3_validation.json'
    if validation_path.exists():
        validation = read_json(validation_path)
        lines += ["", "### Recorded verification", "",
                  f"Recorded UTC: {validation['measured_utc']}. Full command output and exit codes are in eval/phase3_validation.json. FFmpeg/FFprobe are provided by D:/FilmTools/Shotcut; no packages were installed.", "",
                  "| Check | Exit code | Result |", "|---|---:|---|"]
        for check in validation['checks']:
            output_text = check['output']
            summary = re.search(r"\d+ passed[^\n]*|Ran \d+ tests[^\n]*", output_text)
            status = re.search(r"OK(?: \(skipped=\d+\))?", output_text)
            lines.append(f"| `{check['command']}` | {check['exit_code']} | " +
                         (summary.group().strip() if summary else 'See full output') +
                         ('; ' + status.group() if status else '') + " |")
    (ROOT / 'eval/PHASE3.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print({name: {key: r['overall'][key] for key in METRICS} for name, r in reports.items()})


if __name__ == '__main__':
    main()
