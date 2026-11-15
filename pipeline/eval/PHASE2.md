# Phase 2: query-time retrieval and bounded context

Measured against the unchanged read-only export `D:/thesis-data/run-20261020/export` on branch `upgrade/extraction-fixes`. No re-indexing, GPU work, package installation, commit, push, branch switch or edits under `docs/` were performed.

**Development set:** the same 120 draft questions (110 answerable, 10 unanswerable), auto-written by Codex and still requiring human review. This is a development comparison, not a held-out or independently reviewed benchmark. No question, gold span or per-video parameter was changed.

## Measurement and reproduction contract

- The harness calls the real backend `search()` and `build_context()` in isolated file mode using the pinned CPU ONNX E5 encoder. Every run still returns at most **three slides**. Hit@3 and MRR use the original slide intervals; neighbor evidence does not become extra returned slides or artificially widen hit intervals. Hit@5 remains Hit@3.
- The all-off ablation reproduces every original question's slide order, scores, best semantic score and context byte-for-byte, including all four baseline metrics. The previous default command now evaluates the improved app; append an empty `--features` to recover Phase 1.
- Context recall keeps Phase 1's gold-time overlap plus annotated-key-term definition. The new trace checks only actually retained blocks, their source and segment, and retained transcript words at gold times. It aligns words against each retained chunk body so split parts sharing a coarse interval cannot receive credit for omitted words. OCR/title terms count only in included visual blocks overlapping gold. Neighbor blocks retain their own timestamps. Legacy-prefix tracing is unchanged except that the neighbors-only experiment accounts for its total character cap.
- Average/max context characters below use the 110 answerable questions and include labels/titles/separators actually sent by `build_context()`. The final cap also passed for all 10 unanswerable questions. Qwen tokens and generated answers were not measured. The character cap bounds retrieved evidence, not the complete prompt with question, system text and conversation history.
- Candidates remain **20 distinct evidence groups per branch**, RRF k=60, semantic/lexical weights 1.0/0.5, MAX_SLIDES=3. Dedup reads enough raw lexical postings to form twenty groups; this prevents duplicate copies from occupying those slots. It is not a tuned increase in the fused candidate pool.
- The final matrix was regenerated with one shared batch of real CPU ONNX query vectors (execution-cost reuse only). The standalone commands below reproduce each configuration. All reports use the same final backend/harness hashes and unchanged input hashes.

Question SHA-256: `e07c70b3a8122eb73c9cf6c1a343b794117da56c15e54c0c657001a38d7005c2`. Final backend retrieval SHA-256: `0928d3749350e7165ed2c1cc64d1bd574395d720f18d9c3bdf16d0bfbd62baaa`. Final matrix UTC: `2026-11-01T15:47:01.815729+00:00`. Per-run JSON/Markdown and full contexts are in `runs/exp/phase2-*/` (git-ignored).

## Overall results

Each “alone” row enables just the named change against Phase 1. Source boost stays on unless the row explicitly removes it. The interaction rows were measured because phrase matching and the boost depend on lexical normalization/OCR grouping; no weight, pool-size or threshold grid was searched.

| Configuration | Hit@1 | Hit@3 | MRR | Context recall | Avg context chars | Max chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (all flags off; legacy boost on) | 0.518 | 0.755 | 0.621 | 0.609 | 5649 | 11425 |
| Ranked span context alone | 0.518 | 0.755 | 0.621 | 0.645 | 5218 | 8000 |
| Neighbors alone (legacy prefixes + bounded addition) | 0.518 | 0.755 | 0.621 | 0.636 | 6241 | 8000 |
| Accented stopword fix alone | 0.527 | 0.755 | 0.624 | 0.600 | 5684 | 11425 |
| Syllable bigrams alone (legacy stopwords) | 0.500 | 0.755 | 0.611 | 0.609 | 5548 | 11425 |
| OCR dedup alone (legacy boost on) | 0.509 | 0.755 | 0.612 | 0.609 | 5678 | 11425 |
| Content-group diversity alone | 0.518 | 0.764 | 0.624 | 0.618 | 5673 | 11425 |
| Remove source boost alone | 0.545 | 0.773 | 0.648 | 0.627 | 5641 | 11425 |
| Dedup + remove source boost | 0.555 | 0.782 | 0.653 | 0.636 | 5652 | 11425 |
| Ranked context + neighbors | 0.518 | 0.755 | 0.621 | 0.682 | 5951 | 8000 |
| Combined without bigrams | 0.573 | 0.782 | 0.664 | 0.709 | 6025 | 8000 |
| Combined without neighbors | 0.600 | 0.791 | 0.682 | 0.700 | 5287 | 7991 |
| All proposed changes combined | 0.600 | 0.791 | 0.682 | 0.745 | 6002 | 7997 |
| Final app defaults | 0.600 | 0.791 | 0.682 | 0.745 | 6002 | 7997 |

Final vs baseline: Hit@1 +0.082; Hit@3 +0.036; MRR +0.061; context recall +0.136. Correct context increases from 67 to 82 of 110 questions. Average context grows by +354 characters (6.3%), while the observed maximum falls from 11,425 to 7,997.

## Final per-video and per-type results

Cells show final value (delta from baseline). A DROP flags any decrease in Hit@1, Hit@3, MRR or context recall; context-size changes are reported separately.

### Video

| Group | N | Hit@1 (delta) | Hit@3 (delta) | MRR (delta) | Context recall (delta) | Avg chars (delta) | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cs_0001 | 11 | 0.636 (+0.273) | 0.818 (+0.000) | 0.712 (+0.136) | 0.818 (+0.000) | 6215 (+2290) |  |
| D_kuulxKuxc | 11 | 0.364 (+0.000) | 0.455 (-0.091) | 0.409 (-0.030) | 0.455 (-0.091) | 1595 (+514) | DROP |
| hVU5Mw19V_c | 11 | 0.455 (+0.091) | 0.818 (+0.000) | 0.591 (+0.061) | 0.818 (+0.364) | 7063 (+388) |  |
| L25_V008 | 11 | 0.818 (+0.091) | 1.000 (+0.000) | 0.909 (+0.061) | 1.000 (+0.091) | 7691 (-538) |  |
| L25_V011 | 11 | 0.818 (+0.000) | 0.909 (+0.000) | 0.864 (+0.015) | 0.818 (+0.000) | 7441 (-399) |  |
| L25_V037 | 11 | 0.818 (-0.091) | 0.909 (+0.000) | 0.864 (-0.045) | 0.909 (+0.091) | 7775 (-606) | DROP |
| tHE1mC_n0-E | 11 | 0.545 (+0.273) | 0.636 (+0.273) | 0.576 (+0.258) | 0.545 (+0.182) | 5571 (+966) |  |
| v2YTz8ToEag | 11 | 0.545 (+0.091) | 0.818 (+0.091) | 0.652 (+0.076) | 0.636 (+0.182) | 6742 (+712) |  |
| y1CvB950VlU | 11 | 0.364 (+0.000) | 0.636 (+0.091) | 0.485 (+0.030) | 0.636 (+0.273) | 2460 (+780) |  |
| _eDGTHhyQ7M | 11 | 0.636 (+0.091) | 0.909 (+0.000) | 0.758 (+0.045) | 0.818 (+0.273) | 7468 (-571) |  |

### Question type

| Group | N | Hit@1 (delta) | Hit@3 (delta) | MRR (delta) | Context recall (delta) | Avg chars (delta) | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| both | 13 | 0.923 (+0.154) | 0.923 (-0.077) | 0.923 (+0.064) | 0.923 (-0.077) | 5906 (+1) | DROP |
| code | 4 | 0.000 (+0.000) | 0.500 (+0.250) | 0.208 (+0.083) | 0.250 (+0.000) | 2520 (+822) |  |
| numeric | 26 | 0.731 (+0.154) | 0.923 (+0.192) | 0.808 (+0.167) | 0.808 (+0.192) | 6034 (+375) |  |
| slide | 22 | 0.455 (-0.045) | 0.591 (-0.045) | 0.508 (-0.053) | 0.591 (+0.045) | 6492 (+161) | DROP |
| speech | 23 | 0.696 (+0.087) | 0.957 (+0.000) | 0.819 (+0.051) | 0.957 (+0.348) | 5737 (+226) |  |
| term | 12 | 0.583 (+0.250) | 0.667 (+0.000) | 0.611 (+0.139) | 0.750 (+0.083) | 6632 (+1376) |  |
| visual | 10 | 0.200 (-0.100) | 0.600 (+0.000) | 0.383 (-0.050) | 0.400 (+0.100) | 6211 (+58) | DROP |

Final video regressions: **D_kuulxKuxc** loses one Hit@3/context hit (`q05`, the elephant's 13−4 example at 694.52–720.64 s); its returned neighboring visual occurrences end before the gold speech. **L25_V037** loses one Hit@1 (`q09`, rights demanded in 1936–1939); the correct slide moves to rank two, so Hit@3 and context recall survive. No per-video exception was added. Question-level context losses also remain for `cs_0001_q02`, `tHE1mC_n0-E_q03`, and `_eDGTHhyQ7M_q01`; aggregate video gains do not imply every question improved.

## Kept and rejected

| Decision | Reason |
|---|---|
| Keep ranked span context | Preserves matched chunk IDs/spans. Selects ranked evidence and the best same-slide complementary modality, then adjacent speech windows. Context alone gains four recalled questions with fewer characters. Selected blocks render chronologically within each video, with their own times. |
| Keep conditional neighbors | Previous/next segment boundary speech is eligible when the anchor touches a boundary, is shorter than 80 characters, or the question requests steps/explanation. Only immediate segments within a 2-second segment gap; flagged ads are skipped. Natural speech continuation also requires a short speech gap; explanatory questions may request broader adjoining material. All uses share the same total budget. In the final retrieval configuration, neighbors add five recalled questions (0.700 → 0.745). |
| Keep accented stopwords and syllable bigrams together | Stopword removal occurs on NFC accented forms before accent folding; ambiguous technical syllables are retained without a per-course term catalogue. Phrases use only originally adjacent retained tokens, never bridging a removed particle. The single lexical-fix row is mixed and bigrams alone reduce ranking, but removing bigrams from the full combination loses three Hit@1 and four context successes. |
| Keep exact/near OCR grouping and content diversity | Group OCR before either candidate limit, preserve all occurrence IDs/times, and select one representative by semantic relevance with current/follow-up occurrence preference. Near grouping uses ≥16 tokens and trigram Jaccard ≥0.90, while preserving numbers/math operators. Across this export 146 OCR copies are collapsed into existing content groups. Content-group selection also prevents duplicate OCR from taking separate slide slots when dedup is ablated. |
| Remove the 1.05 source boost from defaults | With dedup, removing it raises Hit@1 0.509 → 0.555 and Hit@3 0.755 → 0.782. Keep the legacy value behind `source_boost` for reproducibility. |
| Reject larger candidate pools / weight tuning | No evidence justified additional fine tuning on these draft questions. Twenty groups per branch and the original RRF weights remain. |
| Reject per-video/query exceptions | All thresholds and budgets are global; the two video regressions are disclosed for review rather than patched by video or question ID. |

The new defaults are all feature constants enabled and `APPLY_SOURCE_BOOST=False`. `RetrievalOptions` allows independent overrides in `search`, `search_many`, and `build_context`. The response retains `segment_id`, slide `start_sec/end_sec`, title, full OCR/speech, image, video identity and scores. Additional `matched_chunk_ids`, `matched_spans`, `evidence_candidates` and `context_evidence` provide span/provenance tracing. LLM blocks and instructions cite matched/retained span timestamps; the existing UI source cards retain their slide interval contract.

## File/cloud parity and limits

Both modes receive the same bounded, folded term list and adjacent syllable pairs from `lexical_query`. SQLite uses quoted FTS5 OR terms/phrases; Postgres keeps `to_tsquery('simple')` against `tsv`, with `|` and `<->` for corresponding phrases. Backend and Kaggle lexical logic/SQLite results are checked for both flag states, accents, stopwords and operator-like user input. Cloud SQL is tested through a stub connection, so no Supabase/R2 connection occurs. Native BM25 and `ts_rank` remain different ranking functions; live Postgres ranking equivalence has not been demonstrated. The two existing cloud integration tests remain skipped.

An 8,000-character evidence cap with whole-chunk selection and per-segment limits (2,500 speech / 1,800 OCR characters) keeps context finite for CPU Qwen. It does not guarantee that an unusually long conversation plus evidence fits 8k Qwen tokens; exact model-token accounting and prompt-history budgeting remain a follow-up. Speech citations inherit indexed word-span/coarse timing; OCR uses slide intervals. Duplicate occurrences are provenance, not extra credited gold hits.

This phase cannot repair missing/wrong OCR or ASR, omitted figure descriptions, promotional chunks already present without ad flags, or unsampled visual states. Continuity expansion may still include an unflagged promotional neighbor. The benchmark remains draft and small (11 answerable questions per video); test on held-out human-reviewed questions before making thesis quality claims. No LLM answer-quality or abstention test was performed; unanswerable queries still produce high semantic scores.

## Tests

- Backend: **57 run, 55 passed, 2 skipped** (the existing unavailable local Postgres integration tests). The original 45 tests still pass; twelve added tests cover stopwords/phrases, file/notebook parity, stubbed cloud query construction, dedup/numeric safety and pre-limit grouping, occurrence preference, diversity/temporal pinning, matched tail context/timing, speech windows, conditional neighbors/ad/gap handling, total budget and multi-video identity.
- Harness: **15 passed**, including two new regressions for omitted words sharing a span and neighbor timing/source attribution.
- `git diff --check` passes. `_isolate` is imported before app imports and now adds the backend import path, making the user's root-level unittest command work without a PYTHONPATH prerequisite. No real cloud settings were used; the offline loader never imports app startup, dotenv, Supabase or R2.

## Exact commands

Run from the repository root in PowerShell. An empty `--features` means the Phase 1 configuration; omit it entirely for the new app defaults. JSON/Markdown go to the named directory under `pipeline/runs/exp/`.

```powershell
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-baseline --features
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-context --features ranked_context
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-neighbors --features neighbors
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-lexical --features vietnamese_lexical
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-phrases --features phrases
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-dedup --features dedup
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-diversity --features diversity
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-no-boost --features no_boost
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-dedup-no-boost --features dedup no_boost
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-context-neighbors --features ranked_context neighbors
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-no-phrases --features ranked_context neighbors vietnamese_lexical dedup diversity no_boost
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-all-no-neighbors --features ranked_context vietnamese_lexical phrases dedup diversity no_boost
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-all --features ranked_context neighbors vietnamese_lexical phrases dedup diversity no_boost
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name phase2-final

& backend/.venv/Scripts/python.exe -m unittest discover -s backend/tests -q
& backend/.venv/Scripts/python.exe -m unittest discover -s pipeline/tests -p test_eval_harness.py -q
git diff --check -- backend pipeline
```

## Per-change deltas (anti-overfitting audit)

Every group is included, including zero deltas. Deltas are absolute metric changes, not percentage-point notation. DROP flags any retrieval/context decrease from Phase 1. These tables also cover the measured interactions and rejected alternatives.

### Ranked span context alone

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | +0.000 | +0.000 | +0.000 | +0.036 | -431 |  |
| cs_0001 | +0.000 | +0.000 | +0.000 | +0.000 | +271 |  |
| D_kuulxKuxc | +0.000 | +0.000 | +0.000 | +0.000 | -25 |  |
| hVU5Mw19V_c | +0.000 | +0.000 | +0.000 | +0.091 | -306 |  |
| L25_V008 | +0.000 | +0.000 | +0.000 | +0.000 | -752 |  |
| L25_V011 | +0.000 | +0.000 | +0.000 | +0.000 | -411 |  |
| L25_V037 | +0.000 | +0.000 | +0.000 | +0.091 | -738 |  |
| tHE1mC_n0-E | +0.000 | +0.000 | +0.000 | -0.091 | -1262 | DROP |
| v2YTz8ToEag | +0.000 | +0.000 | +0.000 | +0.091 | +50 |  |
| y1CvB950VlU | +0.000 | +0.000 | +0.000 | +0.000 | +58 |  |
| _eDGTHhyQ7M | +0.000 | +0.000 | +0.000 | +0.182 | -1194 |  |
| type: both | +0.000 | +0.000 | +0.000 | +0.000 | -247 |  |
| type: code | +0.000 | +0.000 | +0.000 | +0.000 | +22 |  |
| type: numeric | +0.000 | +0.000 | +0.000 | -0.077 | -605 | DROP |
| type: slide | +0.000 | +0.000 | +0.000 | +0.045 | -748 |  |
| type: speech | +0.000 | +0.000 | +0.000 | +0.217 | -345 |  |
| type: term | +0.000 | +0.000 | +0.000 | -0.083 | +222 | DROP |
| type: visual | +0.000 | +0.000 | +0.000 | +0.100 | -680 |  |

### Neighbors alone (legacy prefixes + bounded addition)

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | +0.000 | +0.000 | +0.000 | +0.027 | +592 |  |
| cs_0001 | +0.000 | +0.000 | +0.000 | +0.000 | +2530 |  |
| D_kuulxKuxc | +0.000 | +0.000 | +0.000 | +0.000 | +565 |  |
| hVU5Mw19V_c | +0.000 | +0.000 | +0.000 | +0.091 | +338 |  |
| L25_V008 | +0.000 | +0.000 | +0.000 | +0.000 | -373 |  |
| L25_V011 | +0.000 | +0.000 | +0.000 | +0.000 | -214 |  |
| L25_V037 | +0.000 | +0.000 | +0.000 | +0.000 | -464 |  |
| tHE1mC_n0-E | +0.000 | +0.000 | +0.000 | +0.000 | +1441 |  |
| v2YTz8ToEag | +0.000 | +0.000 | +0.000 | +0.091 | +1105 |  |
| y1CvB950VlU | +0.000 | +0.000 | +0.000 | +0.182 | +1160 |  |
| _eDGTHhyQ7M | +0.000 | +0.000 | +0.000 | -0.091 | -164 | DROP |
| type: both | +0.000 | +0.000 | +0.000 | +0.000 | +354 |  |
| type: code | +0.000 | +0.000 | +0.000 | +0.000 | +1219 |  |
| type: numeric | +0.000 | +0.000 | +0.000 | +0.038 | +666 |  |
| type: slide | +0.000 | +0.000 | +0.000 | +0.000 | +406 |  |
| type: speech | +0.000 | +0.000 | +0.000 | +0.087 | +455 |  |
| type: term | +0.000 | +0.000 | +0.000 | +0.000 | +1301 |  |
| type: visual | +0.000 | +0.000 | +0.000 | +0.000 | +338 |  |

### Accented stopword fix alone

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | +0.009 | +0.000 | +0.003 | -0.009 | +36 | DROP |
| cs_0001 | +0.000 | -0.091 | -0.045 | -0.091 | +47 | DROP |
| D_kuulxKuxc | +0.000 | -0.091 | -0.030 | -0.091 | -39 | DROP |
| hVU5Mw19V_c | +0.000 | +0.000 | +0.000 | +0.000 | +71 |  |
| L25_V008 | +0.000 | +0.000 | -0.015 | +0.000 | -32 | DROP |
| L25_V011 | +0.091 | +0.000 | +0.061 | +0.000 | -190 |  |
| L25_V037 | -0.091 | +0.000 | -0.045 | +0.000 | +261 | DROP |
| tHE1mC_n0-E | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| v2YTz8ToEag | +0.000 | +0.091 | +0.030 | +0.000 | +156 |  |
| y1CvB950VlU | +0.091 | +0.000 | +0.045 | +0.000 | -86 |  |
| _eDGTHhyQ7M | +0.000 | +0.091 | +0.030 | +0.091 | +166 |  |
| type: both | +0.000 | -0.077 | -0.026 | -0.077 | +258 | DROP |
| type: code | +0.000 | +0.000 | +0.000 | +0.000 | -155 |  |
| type: numeric | +0.000 | +0.038 | +0.013 | +0.000 | -50 |  |
| type: slide | +0.000 | +0.000 | -0.008 | +0.000 | +148 | DROP |
| type: speech | +0.043 | +0.000 | +0.022 | +0.000 | -39 |  |
| type: term | +0.000 | +0.000 | +0.000 | +0.000 | +22 |  |
| type: visual | +0.000 | +0.000 | +0.000 | +0.000 | -13 |  |

### Syllable bigrams alone (legacy stopwords)

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | -0.018 | +0.000 | -0.011 | +0.000 | -101 | DROP |
| cs_0001 | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| D_kuulxKuxc | -0.182 | -0.091 | -0.136 | -0.091 | -23 | DROP |
| hVU5Mw19V_c | +0.091 | +0.000 | +0.061 | +0.000 | -220 |  |
| L25_V008 | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| L25_V011 | +0.091 | +0.000 | +0.061 | +0.000 | +18 |  |
| L25_V037 | -0.091 | +0.000 | -0.045 | +0.000 | -301 | DROP |
| tHE1mC_n0-E | +0.000 | +0.091 | +0.015 | +0.091 | +2 |  |
| v2YTz8ToEag | -0.091 | +0.000 | -0.045 | +0.000 | -227 | DROP |
| y1CvB950VlU | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| _eDGTHhyQ7M | +0.000 | +0.000 | -0.015 | +0.000 | -257 | DROP |
| type: both | +0.000 | +0.000 | +0.013 | +0.000 | -159 |  |
| type: code | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| type: numeric | -0.077 | -0.038 | -0.064 | -0.038 | -9 | DROP |
| type: slide | -0.045 | +0.045 | -0.008 | +0.045 | -102 | DROP |
| type: speech | +0.043 | +0.000 | +0.029 | +0.000 | -86 |  |
| type: term | +0.000 | +0.000 | -0.014 | +0.000 | -236 | DROP |
| type: visual | +0.000 | +0.000 | +0.000 | +0.000 | -171 |  |

### OCR dedup alone (legacy boost on)

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | -0.009 | +0.000 | -0.009 | +0.000 | +29 | DROP |
| cs_0001 | +0.000 | +0.091 | +0.030 | +0.091 | +135 |  |
| D_kuulxKuxc | +0.000 | -0.091 | -0.030 | -0.091 | +22 | DROP |
| hVU5Mw19V_c | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| L25_V008 | +0.000 | +0.000 | +0.000 | +0.000 | +300 |  |
| L25_V011 | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| L25_V037 | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| tHE1mC_n0-E | -0.091 | +0.000 | -0.076 | +0.000 | -264 | DROP |
| v2YTz8ToEag | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| y1CvB950VlU | +0.000 | +0.000 | +0.000 | +0.000 | +96 |  |
| _eDGTHhyQ7M | +0.000 | +0.000 | -0.015 | +0.000 | +0 | DROP |
| type: both | +0.000 | -0.077 | -0.026 | -0.077 | +4 | DROP |
| type: code | +0.000 | +0.000 | +0.000 | +0.000 | +234 |  |
| type: numeric | +0.000 | +0.038 | +0.006 | +0.038 | -38 |  |
| type: slide | -0.045 | +0.000 | -0.030 | +0.000 | -8 | DROP |
| type: speech | +0.000 | +0.000 | +0.000 | +0.000 | -12 |  |
| type: term | +0.000 | +0.000 | -0.014 | +0.000 | +278 | DROP |
| type: visual | +0.000 | +0.000 | +0.000 | +0.000 | +28 |  |

### Content-group diversity alone

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | +0.000 | +0.009 | +0.003 | +0.009 | +24 |  |
| cs_0001 | +0.000 | +0.091 | +0.030 | +0.091 | +135 |  |
| D_kuulxKuxc | +0.000 | -0.091 | -0.030 | -0.091 | +22 | DROP |
| hVU5Mw19V_c | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| L25_V008 | +0.000 | +0.000 | +0.000 | +0.000 | +300 |  |
| L25_V011 | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| L25_V037 | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| tHE1mC_n0-E | +0.000 | +0.091 | +0.030 | +0.091 | -226 |  |
| v2YTz8ToEag | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| y1CvB950VlU | +0.000 | +0.000 | +0.000 | +0.000 | +10 |  |
| _eDGTHhyQ7M | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| type: both | +0.000 | -0.077 | -0.026 | -0.077 | +4 | DROP |
| type: code | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |
| type: numeric | +0.000 | +0.077 | +0.026 | +0.077 | -29 |  |
| type: slide | +0.000 | +0.000 | +0.000 | +0.000 | -12 |  |
| type: speech | +0.000 | +0.000 | +0.000 | +0.000 | -12 |  |
| type: term | +0.000 | +0.000 | +0.000 | +0.000 | +325 |  |
| type: visual | +0.000 | +0.000 | +0.000 | +0.000 | +0 |  |

### Remove source boost alone

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | +0.027 | +0.018 | +0.027 | +0.018 | -7 |  |
| cs_0001 | +0.000 | +0.091 | +0.045 | +0.091 | +135 |  |
| D_kuulxKuxc | +0.000 | -0.091 | -0.030 | -0.091 | +50 | DROP |
| hVU5Mw19V_c | +0.000 | -0.091 | -0.015 | +0.000 | +237 | DROP |
| L25_V008 | +0.091 | +0.000 | +0.061 | +0.000 | +0 |  |
| L25_V011 | -0.091 | +0.000 | -0.045 | +0.000 | -510 | DROP |
| L25_V037 | -0.091 | +0.000 | -0.045 | +0.000 | +214 | DROP |
| tHE1mC_n0-E | +0.182 | +0.182 | +0.182 | +0.182 | -132 |  |
| v2YTz8ToEag | +0.091 | +0.000 | +0.045 | +0.000 | +0 |  |
| y1CvB950VlU | +0.000 | +0.091 | +0.030 | +0.000 | -19 |  |
| _eDGTHhyQ7M | +0.091 | +0.000 | +0.045 | +0.000 | -48 |  |
| type: both | +0.077 | -0.077 | +0.026 | -0.077 | +237 | DROP |
| type: code | +0.000 | +0.250 | +0.083 | +0.000 | +86 |  |
| type: numeric | +0.038 | +0.115 | +0.083 | +0.115 | +150 |  |
| type: slide | +0.045 | -0.045 | +0.008 | +0.000 | -222 | DROP |
| type: speech | -0.043 | +0.000 | -0.014 | +0.000 | -125 | DROP |
| type: term | +0.167 | +0.000 | +0.083 | +0.000 | +22 |  |
| type: visual | -0.100 | +0.000 | -0.067 | +0.000 | -63 | DROP |

### Dedup + remove source boost

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | +0.036 | +0.027 | +0.032 | +0.027 | +4 |  |
| cs_0001 | +0.091 | +0.091 | +0.091 | +0.091 | +135 |  |
| D_kuulxKuxc | +0.000 | -0.091 | -0.030 | -0.091 | +52 | DROP |
| hVU5Mw19V_c | +0.000 | -0.091 | -0.015 | +0.000 | +237 | DROP |
| L25_V008 | +0.091 | +0.000 | +0.045 | +0.000 | +0 |  |
| L25_V011 | -0.091 | +0.000 | -0.045 | +0.000 | -510 | DROP |
| L25_V037 | -0.091 | +0.000 | -0.045 | +0.000 | +214 | DROP |
| tHE1mC_n0-E | +0.182 | +0.273 | +0.197 | +0.273 | -137 |  |
| v2YTz8ToEag | +0.091 | +0.000 | +0.045 | +0.000 | +0 |  |
| y1CvB950VlU | +0.000 | +0.091 | +0.030 | +0.000 | +94 |  |
| _eDGTHhyQ7M | +0.091 | +0.000 | +0.045 | +0.000 | -48 |  |
| type: both | +0.077 | -0.077 | +0.026 | -0.077 | +237 | DROP |
| type: code | +0.000 | +0.250 | +0.083 | +0.000 | +284 |  |
| type: numeric | +0.115 | +0.115 | +0.115 | +0.115 | +144 |  |
| type: slide | +0.000 | -0.045 | -0.023 | +0.000 | -229 | DROP |
| type: speech | -0.043 | +0.000 | -0.014 | +0.000 | -137 | DROP |
| type: term | +0.167 | +0.083 | +0.111 | +0.083 | +82 |  |
| type: visual | -0.100 | +0.000 | -0.067 | +0.000 | -36 | DROP |

### Ranked context + neighbors

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | +0.000 | +0.000 | +0.000 | +0.073 | +303 |  |
| cs_0001 | +0.000 | +0.000 | +0.000 | +0.000 | +2305 |  |
| D_kuulxKuxc | +0.000 | +0.000 | +0.000 | +0.000 | +510 |  |
| hVU5Mw19V_c | +0.000 | +0.000 | +0.000 | +0.182 | +40 |  |
| L25_V008 | +0.000 | +0.000 | +0.000 | +0.000 | -510 |  |
| L25_V011 | +0.000 | +0.000 | +0.000 | +0.000 | -296 |  |
| L25_V037 | +0.000 | +0.000 | +0.000 | +0.091 | -568 |  |
| tHE1mC_n0-E | +0.000 | +0.000 | +0.000 | -0.091 | +30 | DROP |
| v2YTz8ToEag | +0.000 | +0.000 | +0.000 | +0.182 | +941 |  |
| y1CvB950VlU | +0.000 | +0.000 | +0.000 | +0.182 | +991 |  |
| _eDGTHhyQ7M | +0.000 | +0.000 | +0.000 | +0.182 | -418 |  |
| type: both | +0.000 | +0.000 | +0.000 | +0.000 | +119 |  |
| type: code | +0.000 | +0.000 | +0.000 | +0.000 | +1136 |  |
| type: numeric | +0.000 | +0.000 | +0.000 | -0.038 | +161 | DROP |
| type: slide | +0.000 | +0.000 | +0.000 | +0.045 | +59 |  |
| type: speech | +0.000 | +0.000 | +0.000 | +0.348 | +297 |  |
| type: term | +0.000 | +0.000 | +0.000 | -0.083 | +1237 | DROP |
| type: visual | +0.000 | +0.000 | +0.000 | +0.100 | +2 |  |

### Combined without bigrams

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | +0.055 | +0.027 | +0.042 | +0.100 | +376 |  |
| cs_0001 | +0.091 | +0.000 | +0.045 | +0.000 | +2327 |  |
| D_kuulxKuxc | +0.000 | -0.091 | -0.030 | -0.091 | +612 | DROP |
| hVU5Mw19V_c | +0.000 | -0.091 | -0.015 | +0.182 | +433 | DROP |
| L25_V008 | +0.182 | +0.000 | +0.106 | +0.091 | -553 |  |
| L25_V011 | +0.000 | +0.000 | +0.015 | +0.000 | -609 |  |
| L25_V037 | -0.091 | +0.000 | -0.045 | +0.091 | -570 | DROP |
| tHE1mC_n0-E | +0.182 | +0.273 | +0.197 | +0.182 | +857 |  |
| v2YTz8ToEag | +0.091 | +0.091 | +0.076 | +0.182 | +918 |  |
| y1CvB950VlU | +0.000 | +0.091 | +0.030 | +0.182 | +898 |  |
| _eDGTHhyQ7M | +0.091 | +0.000 | +0.045 | +0.182 | -552 |  |
| type: both | +0.077 | -0.077 | +0.026 | -0.077 | +202 | DROP |
| type: code | +0.000 | +0.250 | +0.083 | +0.000 | +822 |  |
| type: numeric | +0.154 | +0.154 | +0.154 | +0.115 | +417 |  |
| type: slide | +0.000 | -0.091 | -0.045 | +0.000 | +159 | DROP |
| type: speech | +0.000 | +0.000 | +0.007 | +0.348 | +146 |  |
| type: term | +0.167 | +0.083 | +0.111 | +0.000 | +1455 |  |
| type: visual | -0.100 | +0.000 | -0.050 | +0.100 | +30 | DROP |

### Combined without neighbors

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | +0.082 | +0.036 | +0.061 | +0.091 | -362 |  |
| cs_0001 | +0.273 | +0.000 | +0.136 | +0.000 | +453 |  |
| D_kuulxKuxc | +0.000 | -0.091 | -0.030 | -0.091 | -87 | DROP |
| hVU5Mw19V_c | +0.091 | +0.000 | +0.061 | +0.273 | -26 |  |
| L25_V008 | +0.091 | +0.000 | +0.061 | +0.091 | -720 |  |
| L25_V011 | +0.000 | +0.000 | +0.015 | +0.000 | -705 |  |
| L25_V037 | -0.091 | +0.000 | -0.045 | +0.091 | -764 | DROP |
| tHE1mC_n0-E | +0.273 | +0.273 | +0.258 | +0.182 | -495 |  |
| v2YTz8ToEag | +0.091 | +0.091 | +0.076 | +0.091 | +3 |  |
| y1CvB950VlU | +0.000 | +0.091 | +0.030 | +0.000 | -37 |  |
| _eDGTHhyQ7M | +0.091 | +0.000 | +0.045 | +0.273 | -1239 |  |
| type: both | +0.154 | -0.077 | +0.064 | -0.077 | -371 | DROP |
| type: code | +0.000 | +0.250 | +0.083 | +0.000 | +98 |  |
| type: numeric | +0.154 | +0.192 | +0.167 | +0.154 | -299 |  |
| type: slide | -0.045 | -0.045 | -0.053 | +0.045 | -600 | DROP |
| type: speech | +0.087 | +0.000 | +0.051 | +0.217 | -464 |  |
| type: term | +0.250 | +0.000 | +0.139 | +0.000 | +269 |  |
| type: visual | -0.100 | +0.000 | -0.050 | +0.100 | -695 | DROP |

### All proposed changes combined

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | +0.082 | +0.036 | +0.061 | +0.136 | +354 |  |
| cs_0001 | +0.273 | +0.000 | +0.136 | +0.000 | +2290 |  |
| D_kuulxKuxc | +0.000 | -0.091 | -0.030 | -0.091 | +514 | DROP |
| hVU5Mw19V_c | +0.091 | +0.000 | +0.061 | +0.364 | +388 |  |
| L25_V008 | +0.091 | +0.000 | +0.061 | +0.091 | -538 |  |
| L25_V011 | +0.000 | +0.000 | +0.015 | +0.000 | -399 |  |
| L25_V037 | -0.091 | +0.000 | -0.045 | +0.091 | -606 | DROP |
| tHE1mC_n0-E | +0.273 | +0.273 | +0.258 | +0.182 | +966 |  |
| v2YTz8ToEag | +0.091 | +0.091 | +0.076 | +0.182 | +712 |  |
| y1CvB950VlU | +0.000 | +0.091 | +0.030 | +0.273 | +780 |  |
| _eDGTHhyQ7M | +0.091 | +0.000 | +0.045 | +0.273 | -571 |  |
| type: both | +0.154 | -0.077 | +0.064 | -0.077 | +1 | DROP |
| type: code | +0.000 | +0.250 | +0.083 | +0.000 | +822 |  |
| type: numeric | +0.154 | +0.192 | +0.167 | +0.192 | +375 |  |
| type: slide | -0.045 | -0.045 | -0.053 | +0.045 | +161 | DROP |
| type: speech | +0.087 | +0.000 | +0.051 | +0.348 | +226 |  |
| type: term | +0.250 | +0.000 | +0.139 | +0.083 | +1376 |  |
| type: visual | -0.100 | +0.000 | -0.050 | +0.100 | +58 | DROP |

### Final app defaults

| Video / type | Delta Hit@1 | Delta Hit@3 | Delta MRR | Delta context recall | Delta avg chars | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVERALL | +0.082 | +0.036 | +0.061 | +0.136 | +354 |  |
| cs_0001 | +0.273 | +0.000 | +0.136 | +0.000 | +2290 |  |
| D_kuulxKuxc | +0.000 | -0.091 | -0.030 | -0.091 | +514 | DROP |
| hVU5Mw19V_c | +0.091 | +0.000 | +0.061 | +0.364 | +388 |  |
| L25_V008 | +0.091 | +0.000 | +0.061 | +0.091 | -538 |  |
| L25_V011 | +0.000 | +0.000 | +0.015 | +0.000 | -399 |  |
| L25_V037 | -0.091 | +0.000 | -0.045 | +0.091 | -606 | DROP |
| tHE1mC_n0-E | +0.273 | +0.273 | +0.258 | +0.182 | +966 |  |
| v2YTz8ToEag | +0.091 | +0.091 | +0.076 | +0.182 | +712 |  |
| y1CvB950VlU | +0.000 | +0.091 | +0.030 | +0.273 | +780 |  |
| _eDGTHhyQ7M | +0.091 | +0.000 | +0.045 | +0.273 | -571 |  |
| type: both | +0.154 | -0.077 | +0.064 | -0.077 | +1 | DROP |
| type: code | +0.000 | +0.250 | +0.083 | +0.000 | +822 |  |
| type: numeric | +0.154 | +0.192 | +0.167 | +0.192 | +375 |  |
| type: slide | -0.045 | -0.045 | -0.053 | +0.045 | +161 | DROP |
| type: speech | +0.087 | +0.000 | +0.051 | +0.348 | +226 |  |
| type: term | +0.250 | +0.000 | +0.139 | +0.083 | +1376 |  |
| type: visual | -0.100 | +0.000 | -0.050 | +0.100 | +58 | DROP |
