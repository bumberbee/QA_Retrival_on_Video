# Phase 3: CPU text processing

All measurements use the phase-2 default backend retrieval, the unchanged 120-question development set (110 answerable), and raw stage-02 OCR / stage-03 ASR from the read-only export. Questions are auto-written and still need human review. These are development comparisons, not held-out results. No per-video threshold, question, gold span, retrieval weight, or candidate limit was tuned. No GPU, download, package installation, commit, push, branch switch, `docs/` edit, or `backend/app/` edit was performed.

Question SHA-256: `e07c70b3a8122eb73c9cf6c1a343b794117da56c15e54c0c657001a38d7005c2`. Backend retrieval SHA-256: `0928d3749350e7165ed2c1cc64d1bd574395d720f18d9c3bdf16d0bfbd62baaa`. Stage-01 detection and backend router/storage work by other agents are outside this text experiment; raw stage-01 intervals are always taken from the export. Source fingerprints may record their concurrent file changes without executing stage 01.

## Overall ablations

Each alone row enables only that text change relative to the exported-index + phase-2 retrieval reference. Full frozen configs are in `eval/phase3_configs/`. Hit@5 equals Hit@3 because retrieval still returns at most three slides. DROP means any metric decreased relative to the reference.

| Variant | Hit@1 | Hit@3 | MRR | Context recall | Chunks | Flag |
|---|---:|---:|---:|---:|---:|---|
| reference | 0.600 | 0.791 | 0.682 | 0.745 | 1233 |  |
| ocr | 0.600 | 0.782 | 0.679 | 0.745 | 1260 | DROP: hit3, mrr |
| ocr-subscript | 0.600 | 0.782 | 0.679 | 0.764 | 1260 | DROP: hit3, mrr |
| titles | 0.545 | 0.773 | 0.645 | 0.736 | 1233 | DROP: hit1, hit3, mrr, context_recall |
| asr-off | 0.591 | 0.755 | 0.664 | 0.727 | 1233 | DROP: hit1, hit3, mrr, context_recall |
| asr-glossary | 0.591 | 0.764 | 0.667 | 0.736 | 1233 | DROP: hit1, hit3, mrr, context_recall |
| asr-glossary-lecture | 0.600 | 0.782 | 0.679 | 0.745 | 1233 | DROP: hit3, mrr |
| asr-glossary-code | 0.600 | 0.791 | 0.682 | 0.745 | 1233 |  |
| failures | 0.609 | 0.773 | 0.680 | 0.736 | 1220 | DROP: hit3, mrr, context_recall |
| ads | 0.582 | 0.791 | 0.671 | 0.745 | 1231 | DROP: hit1, mrr |
| ads-schools | 0.582 | 0.791 | 0.671 | 0.745 | 1231 | DROP: hit1, mrr |
| ads-signature | 0.582 | 0.791 | 0.670 | 0.745 | 1222 | DROP: hit1, mrr |
| figures | 0.600 | 0.791 | 0.682 | 0.745 | 1414 |  |
| combined | 0.536 | 0.755 | 0.635 | 0.727 | 1221 | DROP: hit1, hit3, mrr, context_recall |
| combined-reviewed | 0.591 | 0.773 | 0.671 | 0.736 | 1221 | DROP: hit1, hit3, mrr, context_recall |
| combined-code-schools | 0.591 | 0.782 | 0.674 | 0.745 | 1221 | DROP: hit1, hit3, mrr |
| combined-code-schools-figures | 0.591 | 0.782 | 0.676 | 0.745 | 1402 | DROP: hit1, hit3, mrr |
| combined-subscript | 0.591 | 0.782 | 0.674 | 0.764 | 1212 | DROP: hit1, hit3, mrr |
| Final defaults | 0.591 | 0.782 | 0.674 | 0.764 | 1212 | DROP: hit1, hit3, mrr |

The combination names describe successive global policies, not interchangeable final defaults. The figure interaction uses the same prototype config with and without figures; it predates the final subscript and Asian School signature refinements.

| Variant | OCR | Heading heuristic | ASR | Failure guard | Speech ads | Generated / semantic-only |
|---|---|---|---|---|---|---|
| reference | False | False | legacy | False | False | False / False |
| ocr | True | False | legacy | False | False | False / False |
| ocr-subscript | canonical_subscripts | False | legacy | False | False | False / False |
| titles | False | True | legacy | False | False | False / False |
| asr-off | False | False | off | False | False | False / False |
| asr-glossary | False | False | glossary | False | False | False / False |
| asr-glossary-lecture | False | False | glossary_lecture | False | False | False / False |
| asr-glossary-code | False | False | glossary_code | False | False | False / False |
| failures | False | False | legacy | True | False | False / False |
| ads | False | False | legacy | False | True | False / False |
| ads-schools | False | False | legacy | False | bounded_schools | False / False |
| ads-signature | False | False | legacy | False | school_signatures | False / False |
| figures | False | False | legacy | False | False | True / True |
| combined | True | True | glossary | True | True | False / False |
| combined-reviewed | True | False | glossary_lecture | True | True | False / False |
| combined-code-schools | True | False | glossary_code | True | bounded_schools | False / False |
| combined-code-schools-figures | True | False | glossary_code | True | bounded_schools | True / True |
| combined-subscript | canonical_subscripts | False | glossary_code | True | school_signatures | False / False |

### Kept / rejected

The final defaults are the measured `phase3-m-combined-subscript` config; the final row repeats that measured configuration. Selection uses global flags, not per-video parameters. All regressions, including the retained config's group/question regressions, remain visible below.

| Policy | Decision | Measurement / reason |
|---|---|---|
| Conservative OCR | Kept | Initial layout-only ablation preserves H1/recall but H3 falls by 1/110 and MRR by 0.003. The additional subscript-digit ablation preserves chemical formula spellings recognized by lexical retrieval. It must be paired with failure detection so raw loops are not revived as evidence. |
| Heading heuristic | Rejected as default; opt-in remains | Alone: 0.545 / 0.773 / 0.645 / 0.736 versus reference 0.600 / 0.791 / 0.682 / 0.745. Its first-block heuristics do not justify a default ranking regression; an explicit visible-heading field needs a GPU extraction change. |
| OCR-driven ASR correction | `glossary_code` kept; unrestricted legacy rejected | Off, local glossary and lecture glossary are measured separately; their retrieval losses are visible. The final reviewed policy adds only a literal local printed-code feedback repair. See its alone row for the no-regression check; the known-bad catalogue must be zero. |
| Failure detector | Kept | Corpus-integrity fix: excludes known failed OCR and loop words with retry provenance. Some image-backed gold OCR terms disappear; that tradeoff is documented rather than relabelled. |
| Promotional speech filter | Kept | Exact bounded spots and subscription calls, with full removed/partial chunk review and no gold-interval overlap. |
| Generated figures | Rejected as default; opt-in remains | Alone produces no change in any overall or visual metric while context grows. Combination interactions are measured separately. Full-frame, unverified descriptions have no demonstrated standalone benefit on this set. |

## Policies and decisions

| Change | Final setting | Why / limit |
|---|---|---|
| Conservative OCR | `canonical_subscripts` | Repetition alone cannot identify boilerplate. Keeps headings, assignments, formulas, bare numbers and symbols. Only explicit attribution, branding and prefixed page-number rules remove lines. canonical_subscripts maps subscript digits to their plain-digit equivalent outside code/quoted lines, preserving powers and raw OCR. All Qwen boxes cover the full frame, so positional deletion is unsupported. |
| Heading titles | `False` | Legacy title extraction remains the default after the measured regression. The opt-in heuristic examines the first six lines, allows short headings, rejects code/body/UI, and returns empty when none qualifies. |
| ASR correction | `glossary_code` | Reviewed terms: xác suất, áp suất, nhiệt độ sôi, chi phí, cao trào, mặt trận and suất điện động, with local or repeated lecture OCR support. Only literal printed-code feedback adds the reviewed sai rồi alias. Punctuation boundaries and atomic matching prevent partial/overlapping repairs; every word retains raw text_asr. |
| Failure guard | `True` | Detects long repeats (OCR phrases up to 64 tokens; ASR up to 8), drops only loop words for speech, and excludes dominated/placeholder OCR including its title and regions. Partial healthy OCR stays flagged. Raw words, timings and ownership survive. |
| Speech ads | `school_signatures` | SIU: institution/brand to closing slogan, at most 240 seconds. Asian School: exact name plus its observed opening (tự hào là ngôi trường mang đến nền giáo dục tiên tiến) and closing (tương lai của chính mình). Literal subscription calls are separate. Keeps surrounding teaching words. AD_FILTER remains None: slide_speech needs missing stage-01 content_scores and risks coarse exclusions. |
| Generated figures | `False` | Measured alone and with the combination. Separate region chunks retain model_generated_unverified provenance, reject no-figure/empty/loop output, and use semantic-only postings. They can contribute semantic RRF weight 1.0, rather than both branches' 1.0 + 0.5. This reduces fusion opportunity, not a guaranteed rank ceiling. Full-frame generated interpretations still need human review. |

The fixed overlapping-match implementation prefers longer phrases and permits compatible overlap; the adopted glossary matches the complete term atomically. Thus tính sắc xuất becomes tính xác suất, rather than tính xác xuất. The legacy greedy path is used only by explicit reference ablations.

The local glossary requires canonical OCR in the same segment; lecture support requires canonical OCR in at least two segments of that lecture. The code-feedback extension repairs xai rồi → sai rồi only when the local OCR contains a literal print message starting with sai roi (accents folded). It cannot use distant lecture support. That extension was reviewed after the lone lecture-glossary regression, y1CvB950VlU_q01: changing a distractor's speech changed its rank, while the correct triple-quote gold still lacked a4. The printed feedback and raw ASR support the repair; it is a globally applied rule, not a video exception. These policy choices use development evidence and still require held-out validation.

Conservative parsing recovers raw Qwen indentation and the index splits OCR by complete lines to preserve it. The exported Python raw answers contain 24 indented code lines across 14 records. This cannot restore indentation or quotes that Qwen never produced, including seg_00017 and seg_00005.

Raw layout recovery also restores subscript glyphs that the old tokenizer round-trip normalized: Cl₂, NaHCO₃ and CO₂. Those caused two chemistry context-recall losses in the initial combinations even though the formulas were still present. The subscript-digit ablation measures an equivalent plain-text spelling in stage 04; the harness/gold identifier matcher is unchanged. This normalization excludes code/quoted lines and does not convert superscript powers such as x² into x2.

The initial ad filter and bounded_schools prototype use SIU closing slogans and still miss the Asian School spot. The school_signatures refinement uses that spot's actual repeated opening and closing, verified from raw speech in all three L25 videos; no school mention or general education topic alone triggers exclusion. The separate prototype and final-signature ablations expose this change rather than retroactively crediting the earlier filter.

Generated figures alone leave all four visual metrics unchanged. In the prototype combination, L25_V011_q06 (the UN mind map) moves from rank 3 to 2: visual MRR 0.383 → 0.400, overall MRR +0.0015, with no H1/H3/recall gain. It adds 181 chunks and increases average context by 143 characters. This small development interaction is retained in the report; the production flag stays off pending verified visual evidence.

## Corpus lints

Counts are diagnostics, not measured error rates. Empty titles can be correct when the frame has no heading; duplicate OCR is retained for lesson completeness and deduplicated by phase-2 retrieval. Loop lints count indexed chunks, while retry records refer to raw failures.

| Lint | reference | ocr | ocr-subscript | titles | asr-off | asr-glossary | asr-glossary-lecture | asr-glossary-code | failures | ads | ads-schools | ads-signature | figures | combined | combined-reviewed | combined-code-schools | combined-code-schools-figures | combined-subscript | Final defaults |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chunks | 1233 | 1260 | 1260 | 1233 | 1233 | 1233 | 1233 | 1233 | 1220 | 1231 | 1231 | 1222 | 1414 | 1221 | 1221 | 1221 | 1402 | 1212 | 1212 |
| segments | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 | 502 |
| promotional_chunks | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 22 | 9 | 9 | 5 | 23 | 9 | 9 | 9 | 9 | 5 | 5 |
| exact_duplicate_ocr_excess | 119 | 121 | 121 | 119 | 119 | 119 | 119 | 119 | 115 | 119 | 119 | 119 | 119 | 113 | 113 | 113 | 113 | 113 | 113 |
| near_duplicate_unique_ocr_pairs | 72 | 97 | 97 | 72 | 72 | 72 | 72 | 72 | 72 | 72 | 72 | 72 | 72 | 79 | 79 | 79 | 79 | 79 | 79 |
| ocr_loop_chunks | 0 | 11 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| asr_loop_chunks | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 0 | 5 | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| chunks_under_32_tokens | 188 | 181 | 181 | 249 | 188 | 188 | 188 | 188 | 186 | 203 | 203 | 203 | 239 | 255 | 195 | 195 | 249 | 195 | 195 |
| empty_titles | 168 | 168 | 168 | 388 | 168 | 168 | 168 | 168 | 175 | 168 | 168 | 168 | 168 | 390 | 175 | 175 | 175 | 175 | 175 |
| asr_correction_events | 112 | 112 | 112 | 112 | 0 | 71 | 84 | 85 | 112 | 112 | 112 | 112 | 112 | 68 | 84 | 85 | 85 | 85 | 85 |
| changed_asr_words | 137 | 137 | 137 | 137 | 0 | 109 | 127 | 128 | 137 | 137 | 137 | 137 | 137 | 106 | 127 | 128 | 128 | 128 | 128 |
| known_bad_asr_corrections | 12 | 12 | 12 | 12 | 0 | 0 | 0 | 0 | 12 | 12 | 12 | 12 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |

Final corpus by video (all variants' detailed per-video lints and IDs are preserved in phase3_matrix.json):

| Video | Chunks | Promotional lint | OCR / ASR loop chunks | Bad corrections | Exact OCR excess | Near OCR pairs | Empty titles |
|---|---:|---:|---:|---:|---:|---:|---:|
| cs_0001 | 59 | 0 | 0 / 0 | 0 | 2 | 0 | 3 |
| D_kuulxKuxc | 186 | 0 | 0 / 0 | 0 | 24 | 0 | 91 |
| hVU5Mw19V_c | 73 | 0 | 0 / 0 | 0 | 0 | 4 | 0 |
| L25_V008 | 46 | 1 | 0 / 0 | 0 | 1 | 0 | 4 |
| L25_V011 | 31 | 0 | 0 / 0 | 0 | 0 | 0 | 5 |
| L25_V037 | 33 | 0 | 0 / 0 | 0 | 0 | 0 | 4 |
| tHE1mC_n0-E | 447 | 0 | 0 / 0 | 0 | 79 | 60 | 41 |
| v2YTz8ToEag | 131 | 0 | 0 / 0 | 0 | 0 | 0 | 8 |
| y1CvB950VlU | 110 | 0 | 0 / 0 | 0 | 5 | 15 | 11 |
| _eDGTHhyQ7M | 96 | 4 | 0 / 0 | 0 | 2 | 0 | 8 |

## Per-video and per-type results

Each table includes all variants; cells are Hit@1 / Hit@3 / MRR / context recall. A ↓ marks any regression from the reference.

### Video

| Group | N | reference | ocr | ocr-subscript | titles | asr-off |
|---|---:|---|---|---|---|---|
| cs_0001 | 11 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 | 0.455 / 0.818 / 0.621 / 0.818 ↓ | 0.636 / 0.818 / 0.712 / 0.818 |
| D_kuulxKuxc | 11 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 |
| hVU5Mw19V_c | 11 | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 | 0.364 / 0.636 / 0.470 / 0.727 ↓ |
| L25_V008 | 11 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 | 0.636 / 1.000 / 0.803 / 1.000 ↓ | 0.818 / 1.000 / 0.909 / 1.000 |
| L25_V011 | 11 | 0.818 / 0.909 / 0.864 / 0.818 | 0.818 / 0.909 / 0.848 / 0.909 ↓ | 0.818 / 0.909 / 0.848 / 0.909 ↓ | 0.818 / 0.909 / 0.848 / 0.818 ↓ | 0.818 / 0.909 / 0.864 / 0.818 |
| L25_V037 | 11 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 | 0.727 / 0.909 / 0.818 / 0.909 ↓ | 0.818 / 0.909 / 0.864 / 0.909 |
| tHE1mC_n0-E | 11 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.727 / 0.636 / 0.636 | 0.545 / 0.636 / 0.576 / 0.545 |
| v2YTz8ToEag | 11 | 0.545 / 0.818 / 0.652 / 0.636 | 0.455 / 0.818 / 0.621 / 0.818 ↓ | 0.455 / 0.818 / 0.621 / 0.818 ↓ | 0.455 / 0.727 / 0.576 / 0.636 ↓ | 0.545 / 0.818 / 0.652 / 0.636 |
| y1CvB950VlU | 11 | 0.364 / 0.636 / 0.485 / 0.636 | 0.455 / 0.636 / 0.530 / 0.636 | 0.455 / 0.636 / 0.530 / 0.636 | 0.455 / 0.545 / 0.500 / 0.545 ↓ | 0.364 / 0.545 / 0.455 / 0.636 ↓ |
| _eDGTHhyQ7M | 11 | 0.636 / 0.909 / 0.758 / 0.818 | 0.636 / 0.818 / 0.727 / 0.545 ↓ | 0.636 / 0.818 / 0.727 / 0.727 ↓ | 0.545 / 0.818 / 0.652 / 0.727 ↓ | 0.636 / 0.818 / 0.727 / 0.727 ↓ |

| Group | N | reference | asr-glossary | asr-glossary-lecture | asr-glossary-code | failures |
|---|---:|---|---|---|---|---|
| cs_0001 | 11 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 |
| D_kuulxKuxc | 11 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 |
| hVU5Mw19V_c | 11 | 0.455 / 0.818 / 0.591 / 0.818 | 0.364 / 0.636 / 0.470 / 0.727 ↓ | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 |
| L25_V008 | 11 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 |
| L25_V011 | 11 | 0.818 / 0.909 / 0.864 / 0.818 | 0.818 / 0.909 / 0.864 / 0.818 | 0.818 / 0.909 / 0.864 / 0.818 | 0.818 / 0.909 / 0.864 / 0.818 | 0.818 / 0.909 / 0.864 / 0.818 |
| L25_V037 | 11 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 |
| tHE1mC_n0-E | 11 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 |
| v2YTz8ToEag | 11 | 0.545 / 0.818 / 0.652 / 0.636 | 0.545 / 0.818 / 0.652 / 0.636 | 0.545 / 0.818 / 0.652 / 0.636 | 0.545 / 0.818 / 0.652 / 0.636 | 0.455 / 0.727 / 0.576 / 0.636 ↓ |
| y1CvB950VlU | 11 | 0.364 / 0.636 / 0.485 / 0.636 | 0.364 / 0.545 / 0.455 / 0.636 ↓ | 0.364 / 0.545 / 0.455 / 0.636 ↓ | 0.364 / 0.636 / 0.485 / 0.636 | 0.455 / 0.636 / 0.530 / 0.636 |
| _eDGTHhyQ7M | 11 | 0.636 / 0.909 / 0.758 / 0.818 | 0.636 / 0.909 / 0.758 / 0.818 | 0.636 / 0.909 / 0.758 / 0.818 | 0.636 / 0.909 / 0.758 / 0.818 | 0.727 / 0.818 / 0.773 / 0.727 ↓ |

| Group | N | reference | ads | ads-schools | ads-signature | figures |
|---|---:|---|---|---|---|---|
| cs_0001 | 11 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 |
| D_kuulxKuxc | 11 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 |
| hVU5Mw19V_c | 11 | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 |
| L25_V008 | 11 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 |
| L25_V011 | 11 | 0.818 / 0.909 / 0.864 / 0.818 | 0.636 / 0.909 / 0.773 / 0.818 ↓ | 0.636 / 0.909 / 0.773 / 0.818 ↓ | 0.636 / 0.909 / 0.758 / 0.818 ↓ | 0.818 / 0.909 / 0.864 / 0.818 |
| L25_V037 | 11 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 |
| tHE1mC_n0-E | 11 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 |
| v2YTz8ToEag | 11 | 0.545 / 0.818 / 0.652 / 0.636 | 0.545 / 0.818 / 0.652 / 0.636 | 0.545 / 0.818 / 0.652 / 0.636 | 0.545 / 0.818 / 0.652 / 0.636 | 0.545 / 0.818 / 0.652 / 0.636 |
| y1CvB950VlU | 11 | 0.364 / 0.636 / 0.485 / 0.636 | 0.364 / 0.636 / 0.470 / 0.636 ↓ | 0.364 / 0.636 / 0.470 / 0.636 ↓ | 0.364 / 0.636 / 0.470 / 0.636 ↓ | 0.364 / 0.636 / 0.485 / 0.636 |
| _eDGTHhyQ7M | 11 | 0.636 / 0.909 / 0.758 / 0.818 | 0.636 / 0.909 / 0.758 / 0.818 | 0.636 / 0.909 / 0.758 / 0.818 | 0.636 / 0.909 / 0.758 / 0.818 | 0.636 / 0.909 / 0.758 / 0.818 |

| Group | N | reference | combined | combined-reviewed | combined-code-schools | combined-code-schools-figures |
|---|---:|---|---|---|---|---|
| cs_0001 | 11 | 0.636 / 0.818 / 0.712 / 0.818 | 0.455 / 0.818 / 0.606 / 0.818 ↓ | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 |
| D_kuulxKuxc | 11 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 |
| hVU5Mw19V_c | 11 | 0.455 / 0.818 / 0.591 / 0.818 | 0.364 / 0.636 / 0.470 / 0.727 ↓ | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 |
| L25_V008 | 11 | 0.818 / 1.000 / 0.909 / 1.000 | 0.636 / 1.000 / 0.803 / 1.000 ↓ | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 |
| L25_V011 | 11 | 0.818 / 0.909 / 0.864 / 0.818 | 0.818 / 0.909 / 0.864 / 0.909 | 0.727 / 0.909 / 0.803 / 0.909 ↓ | 0.727 / 0.909 / 0.803 / 0.909 ↓ | 0.727 / 0.909 / 0.818 / 0.909 ↓ |
| L25_V037 | 11 | 0.818 / 0.909 / 0.864 / 0.909 | 0.727 / 0.909 / 0.818 / 0.909 ↓ | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 |
| tHE1mC_n0-E | 11 | 0.545 / 0.636 / 0.576 / 0.545 | 0.455 / 0.636 / 0.545 / 0.545 ↓ | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 |
| v2YTz8ToEag | 11 | 0.545 / 0.818 / 0.652 / 0.636 | 0.455 / 0.818 / 0.621 / 0.818 ↓ | 0.455 / 0.818 / 0.621 / 0.818 ↓ | 0.455 / 0.818 / 0.621 / 0.818 ↓ | 0.455 / 0.818 / 0.621 / 0.818 ↓ |
| y1CvB950VlU | 11 | 0.364 / 0.636 / 0.485 / 0.636 | 0.455 / 0.545 / 0.500 / 0.545 ↓ | 0.364 / 0.545 / 0.455 / 0.545 ↓ | 0.364 / 0.636 / 0.485 / 0.636 | 0.364 / 0.636 / 0.485 / 0.636 |
| _eDGTHhyQ7M | 11 | 0.636 / 0.909 / 0.758 / 0.818 | 0.636 / 0.818 / 0.712 / 0.545 ↓ | 0.727 / 0.818 / 0.773 / 0.545 ↓ | 0.727 / 0.818 / 0.773 / 0.545 ↓ | 0.727 / 0.818 / 0.773 / 0.545 ↓ |

| Group | N | reference | combined-subscript | Final defaults |
|---|---:|---|---|---|
| cs_0001 | 11 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 | 0.636 / 0.818 / 0.712 / 0.818 |
| D_kuulxKuxc | 11 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 | 0.364 / 0.455 / 0.409 / 0.455 |
| hVU5Mw19V_c | 11 | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 | 0.455 / 0.818 / 0.591 / 0.818 |
| L25_V008 | 11 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 | 0.818 / 1.000 / 0.909 / 1.000 |
| L25_V011 | 11 | 0.818 / 0.909 / 0.864 / 0.818 | 0.727 / 0.909 / 0.803 / 0.909 ↓ | 0.727 / 0.909 / 0.803 / 0.909 ↓ |
| L25_V037 | 11 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 | 0.818 / 0.909 / 0.864 / 0.909 |
| tHE1mC_n0-E | 11 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 | 0.545 / 0.636 / 0.576 / 0.545 |
| v2YTz8ToEag | 11 | 0.545 / 0.818 / 0.652 / 0.636 | 0.455 / 0.818 / 0.621 / 0.818 ↓ | 0.455 / 0.818 / 0.621 / 0.818 ↓ |
| y1CvB950VlU | 11 | 0.364 / 0.636 / 0.485 / 0.636 | 0.364 / 0.636 / 0.485 / 0.636 | 0.364 / 0.636 / 0.485 / 0.636 |
| _eDGTHhyQ7M | 11 | 0.636 / 0.909 / 0.758 / 0.818 | 0.727 / 0.818 / 0.773 / 0.727 ↓ | 0.727 / 0.818 / 0.773 / 0.727 ↓ |

### Type

| Group | N | reference | ocr | ocr-subscript | titles | asr-off |
|---|---:|---|---|---|---|---|
| both | 13 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 | 0.846 / 0.923 / 0.872 / 0.923 ↓ | 0.923 / 0.923 / 0.923 / 0.923 |
| code | 4 | 0.000 / 0.500 / 0.208 / 0.250 | 0.000 / 0.500 / 0.208 / 0.500 | 0.000 / 0.500 / 0.208 / 0.500 | 0.250 / 0.250 / 0.250 / 0.250 ↓ | 0.000 / 0.250 / 0.125 / 0.250 ↓ |
| numeric | 26 | 0.731 / 0.923 / 0.808 / 0.808 | 0.731 / 0.923 / 0.808 / 0.846 | 0.731 / 0.923 / 0.808 / 0.846 | 0.692 / 0.885 / 0.769 / 0.808 ↓ | 0.731 / 0.885 / 0.795 / 0.769 ↓ |
| slide | 22 | 0.455 / 0.591 / 0.508 / 0.591 | 0.455 / 0.545 / 0.492 / 0.500 ↓ | 0.455 / 0.545 / 0.492 / 0.545 ↓ | 0.364 / 0.545 / 0.455 / 0.545 ↓ | 0.455 / 0.545 / 0.492 / 0.545 ↓ |
| speech | 23 | 0.696 / 0.957 / 0.819 / 0.957 | 0.696 / 0.957 / 0.819 / 0.957 | 0.696 / 0.957 / 0.819 / 0.957 | 0.652 / 0.957 / 0.790 / 0.957 ↓ | 0.652 / 0.913 / 0.775 / 0.957 ↓ |
| term | 12 | 0.583 / 0.667 / 0.611 / 0.750 | 0.583 / 0.667 / 0.611 / 0.667 ↓ | 0.583 / 0.667 / 0.611 / 0.667 ↓ | 0.500 / 0.667 / 0.569 / 0.667 ↓ | 0.583 / 0.667 / 0.611 / 0.750 |
| visual | 10 | 0.200 / 0.600 / 0.383 / 0.400 | 0.200 / 0.600 / 0.383 / 0.500 | 0.200 / 0.600 / 0.383 / 0.600 | 0.100 / 0.700 / 0.367 / 0.500 ↓ | 0.200 / 0.600 / 0.383 / 0.400 |

| Group | N | reference | asr-glossary | asr-glossary-lecture | asr-glossary-code | failures |
|---|---:|---|---|---|---|---|
| both | 13 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 |
| code | 4 | 0.000 / 0.500 / 0.208 / 0.250 | 0.000 / 0.250 / 0.125 / 0.250 ↓ | 0.000 / 0.250 / 0.125 / 0.250 ↓ | 0.000 / 0.500 / 0.208 / 0.250 | 0.000 / 0.500 / 0.208 / 0.250 |
| numeric | 26 | 0.731 / 0.923 / 0.808 / 0.808 | 0.731 / 0.885 / 0.795 / 0.769 ↓ | 0.731 / 0.923 / 0.808 / 0.808 | 0.731 / 0.923 / 0.808 / 0.808 | 0.731 / 0.885 / 0.795 / 0.808 ↓ |
| slide | 22 | 0.455 / 0.591 / 0.508 / 0.591 | 0.455 / 0.591 / 0.508 / 0.591 | 0.455 / 0.591 / 0.508 / 0.591 | 0.455 / 0.591 / 0.508 / 0.591 | 0.455 / 0.545 / 0.492 / 0.545 ↓ |
| speech | 23 | 0.696 / 0.957 / 0.819 / 0.957 | 0.652 / 0.913 / 0.775 / 0.957 ↓ | 0.696 / 0.957 / 0.819 / 0.957 | 0.696 / 0.957 / 0.819 / 0.957 | 0.739 / 0.957 / 0.841 / 0.957 |
| term | 12 | 0.583 / 0.667 / 0.611 / 0.750 | 0.583 / 0.667 / 0.611 / 0.750 | 0.583 / 0.667 / 0.611 / 0.750 | 0.583 / 0.667 / 0.611 / 0.750 | 0.583 / 0.667 / 0.611 / 0.750 |
| visual | 10 | 0.200 / 0.600 / 0.383 / 0.400 | 0.200 / 0.600 / 0.383 / 0.400 | 0.200 / 0.600 / 0.383 / 0.400 | 0.200 / 0.600 / 0.383 / 0.400 | 0.200 / 0.600 / 0.383 / 0.400 |

| Group | N | reference | ads | ads-schools | ads-signature | figures |
|---|---:|---|---|---|---|---|
| both | 13 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 |
| code | 4 | 0.000 / 0.500 / 0.208 / 0.250 | 0.000 / 0.500 / 0.208 / 0.250 | 0.000 / 0.500 / 0.208 / 0.250 | 0.000 / 0.500 / 0.208 / 0.250 | 0.000 / 0.500 / 0.208 / 0.250 |
| numeric | 26 | 0.731 / 0.923 / 0.808 / 0.808 | 0.731 / 0.923 / 0.808 / 0.808 | 0.731 / 0.923 / 0.808 / 0.808 | 0.731 / 0.923 / 0.808 / 0.808 | 0.731 / 0.923 / 0.808 / 0.808 |
| slide | 22 | 0.455 / 0.591 / 0.508 / 0.591 | 0.455 / 0.591 / 0.508 / 0.591 | 0.455 / 0.591 / 0.508 / 0.591 | 0.455 / 0.591 / 0.508 / 0.591 | 0.455 / 0.591 / 0.508 / 0.591 |
| speech | 23 | 0.696 / 0.957 / 0.819 / 0.957 | 0.609 / 0.957 / 0.768 / 0.957 ↓ | 0.609 / 0.957 / 0.768 / 0.957 ↓ | 0.609 / 0.957 / 0.768 / 0.957 ↓ | 0.696 / 0.957 / 0.819 / 0.957 |
| term | 12 | 0.583 / 0.667 / 0.611 / 0.750 | 0.583 / 0.667 / 0.611 / 0.750 | 0.583 / 0.667 / 0.611 / 0.750 | 0.583 / 0.667 / 0.611 / 0.750 | 0.583 / 0.667 / 0.611 / 0.750 |
| visual | 10 | 0.200 / 0.600 / 0.383 / 0.400 | 0.200 / 0.600 / 0.383 / 0.400 | 0.200 / 0.600 / 0.383 / 0.400 | 0.200 / 0.600 / 0.367 / 0.400 ↓ | 0.200 / 0.600 / 0.383 / 0.400 |

| Group | N | reference | combined | combined-reviewed | combined-code-schools | combined-code-schools-figures |
|---|---:|---|---|---|---|---|
| both | 13 | 0.923 / 0.923 / 0.923 / 0.923 | 0.846 / 0.923 / 0.872 / 0.923 ↓ | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 |
| code | 4 | 0.000 / 0.500 / 0.208 / 0.250 | 0.250 / 0.250 / 0.250 / 0.250 ↓ | 0.000 / 0.250 / 0.125 / 0.250 ↓ | 0.000 / 0.500 / 0.208 / 0.500 | 0.000 / 0.500 / 0.208 / 0.500 |
| numeric | 26 | 0.731 / 0.923 / 0.808 / 0.808 | 0.654 / 0.885 / 0.750 / 0.808 ↓ | 0.731 / 0.923 / 0.808 / 0.846 | 0.731 / 0.923 / 0.808 / 0.846 | 0.731 / 0.923 / 0.808 / 0.846 |
| slide | 22 | 0.455 / 0.591 / 0.508 / 0.591 | 0.364 / 0.545 / 0.455 / 0.500 ↓ | 0.455 / 0.545 / 0.492 / 0.500 ↓ | 0.455 / 0.545 / 0.492 / 0.500 ↓ | 0.455 / 0.545 / 0.492 / 0.500 ↓ |
| speech | 23 | 0.696 / 0.957 / 0.819 / 0.957 | 0.652 / 0.913 / 0.775 / 0.957 ↓ | 0.652 / 0.957 / 0.797 / 0.957 ↓ | 0.652 / 0.957 / 0.797 / 0.957 ↓ | 0.652 / 0.957 / 0.797 / 0.957 ↓ |
| term | 12 | 0.583 / 0.667 / 0.611 / 0.750 | 0.500 / 0.667 / 0.556 / 0.667 ↓ | 0.583 / 0.667 / 0.611 / 0.667 ↓ | 0.583 / 0.667 / 0.611 / 0.667 ↓ | 0.583 / 0.667 / 0.611 / 0.667 ↓ |
| visual | 10 | 0.200 / 0.600 / 0.383 / 0.400 | 0.100 / 0.600 / 0.350 / 0.500 ↓ | 0.200 / 0.600 / 0.383 / 0.500 | 0.200 / 0.600 / 0.383 / 0.500 | 0.200 / 0.600 / 0.400 / 0.500 |

| Group | N | reference | combined-subscript | Final defaults |
|---|---:|---|---|---|
| both | 13 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 | 0.923 / 0.923 / 0.923 / 0.923 |
| code | 4 | 0.000 / 0.500 / 0.208 / 0.250 | 0.000 / 0.500 / 0.208 / 0.500 | 0.000 / 0.500 / 0.208 / 0.500 |
| numeric | 26 | 0.731 / 0.923 / 0.808 / 0.808 | 0.731 / 0.923 / 0.808 / 0.846 | 0.731 / 0.923 / 0.808 / 0.846 |
| slide | 22 | 0.455 / 0.591 / 0.508 / 0.591 | 0.455 / 0.545 / 0.492 / 0.545 ↓ | 0.455 / 0.545 / 0.492 / 0.545 ↓ |
| speech | 23 | 0.696 / 0.957 / 0.819 / 0.957 | 0.652 / 0.957 / 0.797 / 0.957 ↓ | 0.652 / 0.957 / 0.797 / 0.957 ↓ |
| term | 12 | 0.583 / 0.667 / 0.611 / 0.750 | 0.583 / 0.667 / 0.611 / 0.667 ↓ | 0.583 / 0.667 / 0.611 / 0.667 ↓ |
| visual | 10 | 0.200 / 0.600 / 0.383 / 0.400 | 0.200 / 0.600 / 0.383 / 0.600 | 0.200 / 0.600 / 0.383 / 0.600 |

## Changed questions and gold evidence

`question_changes.json` in the final experiment records every question whose ranking, context or metric changed, with gold spans, before/after slides and the key terms present in source-compatible chunks overlapping gold. This availability diagnostic is coarser than the unchanged context-recall trace; it does not fabricate new gold or give credit for deleted words. All full contexts and retained-word traces remain in each `*-eval/retrieval.json`.

Final metric changes are listed below. A loss with unchanged key-term availability reflects competition or bounded context selection; a removed term requires checking the raw evidence/retry record. Gold times and answers were not edited.

| Question | Reference H1/H3/MRR/recall | Final H1/H3/MRR/recall | Flag / evidence check |
|---|---|---|---|
| L25_V011_q06 | 0.000 / 1.000 / 0.500 / 0.000 | 0.000 / 1.000 / 0.333 / 1.000 | DROP; Gold-overlap chunk terms gained: MỤC TIÊU, VAI TRÒ Conservative cleaning restores the repeated MỤC TIÊU and VAI TRÒ branches. The gold mind-map interval remains rank 3 rather than rank 2, but all three heading terms now reach context. |
| L25_V011_q11 | 1.000 / 1.000 / 1.000 / 1.000 | 0.000 / 1.000 / 0.500 / 1.000 | DROP; Gold key-term availability unchanged; ranking/context selection changed. The correct ASEAN interval seg_00006 moves from rank 1 to 2. Its Việt Nam/ASEAN teaching words remain; the Asian School exclusion starts at 327.29 s, after the 295.20–325.00 s gold speech. This is a ranking loss, not deletion of that lesson speech. |
| v2YTz8ToEag_q01 | 1.000 / 1.000 / 1.000 / 1.000 | 0.000 / 1.000 / 0.500 / 1.000 | DROP; Gold key-term availability unchanged; ranking/context selection changed. The correct 100-triệu interval seg_00004 moves behind seg_00024, from rank 1 to 2. The amount is still indexed and recalled; no gold term was removed. |
| v2YTz8ToEag_q08 | 0.000 / 1.000 / 0.333 / 0.000 | 0.000 / 1.000 / 0.500 / 1.000 | Gold-overlap chunk terms gained: G The short G label survives conservative cleaning, and the correct PPF interval seg_00026 moves from rank 3 to 2. This restores the label-based context check; geometric answer accuracy was not graded. |
| v2YTz8ToEag_q09 | 0.000 / 1.000 / 0.333 / 0.000 | 0.000 / 1.000 / 0.333 / 1.000 | Gold-overlap chunk terms gained: 11 The short numeric label 11 survives cleaning alongside 27. The returned slide order is unchanged; keeping the printed number restores context recall without inventing a coordinate. |
| y1CvB950VlU_q01 | 0.000 / 1.000 / 0.333 / 0.000 | 0.000 / 1.000 / 0.333 / 1.000 | Gold-overlap chunk terms gained: a4 The repeated a4 assignment survives conservative cleaning and reaches context at the original gold interval. The key-term proxy checks a3/a4, not triple-quote correctness: raw seg_00005 still has broken quotes and an invented a5, so GPU re-OCR remains necessary. |
| y1CvB950VlU_q03 | 0.000 / 0.000 / 0.000 / 1.000 | 0.000 / 0.000 / 0.000 / 0.000 | DROP; Gold key-term availability unchanged; ranking/context selection changed. The ký tự speech remains indexed in seg_00015 at 246.76–249.90 s. Retrieval replaces seg_00016 with seg_00017; the one-neighbor context policy then includes seg_00016 but misses seg_00015. This is a bounded-context loss, not cleaning of the definition. |
| _eDGTHhyQ7M_q02 | 0.000 / 1.000 / 0.333 / 1.000 | 0.000 / 0.000 / 0.000 / 0.000 | DROP; Gold-overlap chunk terms removed: dầu, hoả Gold oil-storage terms at 612.38–615.75 s belong to seg_00006, whose raw OCR is dominated by a repetition loop. The failure guard excludes its OCR/title/regions while retaining valid speech. The inspected gold image remains valid; re-OCR is needed instead of relabelling gold or indexing the failed output. |
| _eDGTHhyQ7M_q03 | 0.000 / 1.000 / 0.500 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | Gold key-term availability unchanged; ranking/context selection changed. Valid speech in the same failed-OCR interval seg_00006 remains indexed. It moves from rank 2 to 1 and retains the reaction-with-water explanation, showing that the guard does not remove the whole teaching interval. |

## GPU retry list

Automatic retry records: 15. Reviewed audit entries: 16 (some split across segment boundaries). Both lists are in the final experiment as `gpu_retry_segments.json` and `gpu_retry_review.json`; the latter is also tracked at `eval/phase3_gpu_review.json`. Automatic flags change evidence eligibility; audit entries are review metadata only. Deduplicate by source/video/segment when scheduling, while retaining all reasons and the narrower ASR time spans.

| Video / segment | Source | Seconds | Reason |
|---|---|---:|---|
| L25_V011:seg_00007 | asr | 585.14–592.96 | asr_repetition_loop |
| tHE1mC_n0-E:seg_00077 | ocr | 1097.12–1104.48 | empty_ocr_placeholder |
| tHE1mC_n0-E:seg_00097 | ocr | 1278.92–1283.72 | ocr_repetition_loop |
| tHE1mC_n0-E:seg_00164 | ocr | 2103.84–2107.04 | empty_ocr_placeholder |
| tHE1mC_n0-E:seg_00190 | ocr | 2372.80–2377.12 | empty_ocr_placeholder |
| tHE1mC_n0-E:seg_00193 | ocr | 2396.68–2411.44 | ocr_repetition_loop |
| tHE1mC_n0-E:seg_00200 | ocr | 2479.92–2568.68 | ocr_repetition_loop |
| tHE1mC_n0-E:seg_00201 | ocr | 2568.68–2628.96 | ocr_repetition_loop |
| v2YTz8ToEag:seg_00040 | ocr | 2542.59–2545.35 | ocr_repetition_loop |
| y1CvB950VlU:seg_00019 | asr | 286.53–295.41 | asr_repetition_loop |
| y1CvB950VlU:seg_00021 | asr | 385.03–398.64 | asr_repetition_loop |
| y1CvB950VlU:seg_00022 | asr | 385.03–398.64 | asr_repetition_loop |
| _eDGTHhyQ7M:seg_00006 | ocr | 285.29–615.75 | ocr_repetition_loop |
| _eDGTHhyQ7M:seg_00007 | ocr | 615.75–671.47 | ocr_repetition_loop |
| _eDGTHhyQ7M:seg_00021 | asr | 2383.87–2385.55 | asr_repetition_loop |
| hVU5Mw19V_c:seg_00000 | ocr | 0.00–578.96 | audited_numeric_error; Class probability denominator 330 in OCR; keyframe says 370. |
| hVU5Mw19V_c:seg_00012 | ocr | 1549.64–1706.08 | audited_numeric_error; OCR says 4.5 kg / 46.35%; keyframe says 1.5 kg / 46.75%. |
| L25_V011:seg_00006 | ocr | 158.28–409.88 | audited_word_error; UN mind-map quy?n b?nh ??ng becomes quy?n ??ng k?. |
| y1CvB950VlU:seg_00005 | ocr | 64.13–126.97 | audited_code_error; Raw Qwen loses triple quotes and invents a5; requires a crop reread. |
| y1CvB950VlU:seg_00017 | ocr | 256.10–259.27 | raw_code_indentation_missing; Raw response already lacks indentation; cannot recover it on CPU. |
| y1CvB950VlU:seg_00037 | ocr | 620.23–638.43 | audited_ui_occlusion; Tooltip contaminates the selected keyframe; needs another frame/crop. |
| y1CvB950VlU:seg_00038 | ocr | 638.43–642.90 | audited_ui_occlusion; Autocomplete / tooltip covers incomplete code; needs another frame/crop. |
| hVU5Mw19V_c:seg_00000 | asr | 5.00–16.00 | audited_technical_term_error; Bayes is heard as Bias/BS; glossary correction does not invent replacements. |
| hVU5Mw19V_c:seg_00012 | asr | 1557.16–1558.00 | audited_technical_term_error; Bayes is heard as BIS. |
| y1CvB950VlU:seg_00009 | asr | 196.17–207.00 | audited_technical_term_error; len is heard as LAN. |
| y1CvB950VlU:seg_00010 | asr | 196.17–207.00 | audited_technical_term_error; len is heard as LAN. |
| y1CvB950VlU:seg_00011 | asr | 196.17–207.00 | audited_technical_term_error; len is heard as LAN. |
| y1CvB950VlU:seg_00019 | asr | 309.53–310.81 | audited_technical_term_error; for is heard as 4 4. |
| y1CvB950VlU:seg_00023 | asr | 410.42–419.68 | audited_technical_term_error; end identifier is not recovered from ASR. |
| v2YTz8ToEag:seg_00026 | asr | 1659.45–1661.00 | audited_technical_term_error; PPF is heard as PPS. |
| L25_V037:seg_00006 | asr | 262.97–271.12 | audited_technical_term_error; v? trang is heard as ph? trang; outside the adopted probability glossary. |

CPU cleaning cannot recover missing true/false statements, correct wrong handwritten numbers/quotes, replace invented methane evidence with the spacecraft question, resolve UI occlusion, or hear missing Bayes/len/for/end/PPF correctly. Stage-01 merged board states and wrong keyframe selection require video/frame work as well as GPU extraction. The retry lists are not a claim to detect every model error.

## Promotional review

Relative to the same final text config before ad filtering: 18 whole speech chunks removed, 22 original chunks partially filtered, net chunk delta -11. IDs/text/times and retained/excluded word IDs are listed in each video’s `05_index/ad_chunk_review.json`; exact matched speech is in `04_bundle/content_filter.json`. Chunk IDs can shift when units regroup, so the review uses original pre-filter chunks and word provenance.

Conservative misses remain: introductory/closing promotional fragments outside the explicit slogan boundary, recognition drift before the first chemistry brand anchor, and school mentions without a bounded spot. This speech filter does not remove OCR-only promotional frames unless their lines match the conservative branding rules; the promotional lint lists remaining evidence for review. No labelled Kaggle ad data was downloaded. False exclusions of teaching are treated as worse than missed promotional fragments.

Matched promotional spans overlap 0 annotated gold evidence intervals. This is a useful check on this draft question set, not a labelled ad precision estimate. Any overlaps are recorded in ad_gold_overlap_review.json. The following original chunks were removed or modified; partial rows retain their teaching words.

| Original chunk | Seconds | Action | Original text excerpt |
|---|---:|---|---|
| D_kuulxKuxc:seg_00101:u0000:p000 | 1882.37–1889.85 | partially_filtered | Thầy cô, bố mẹ và các em học sinh ơi, hãy like và subscribe kênh youtube vtv7 để đón xem trọn bộ trường trình … |
| L25_V008:seg_00002:u0000:p000 | 9.00–44.05 | partially_filtered | Thầy xin chào tất cả các em Hôm nay chúng ta sẽ đến với chuyên đề cuối cùng Trong chương trình ôn tập môn toán… |
| L25_V008:seg_00002:u0001:p000 | 44.05–89.92 | partially_filtered | tạo đa dạng, đa lĩnh vực SIU mở ra cho chúng tôi nhiều cơ hội lựa chọn nghề nghiệp trong tương lai Tất cả sinh… |
| L25_V008:seg_00008:u0000:p000 | 465.01–503.87 | removed | Trường Quốc tế Á Châu, tự hào là ngôi trường mang đến nền giáo dục tiên tiến, hiện đại Nhưng vẫn giữ dịnh bản … |
| L25_V008:seg_00008:u0001:p000 | 515.91–526.02 | removed | Nơi đây với tôi còn hơn cả một môi trường. Nơi đây là nền tảng giúp chúng tôi đặt những viên gạch đầu tiên vữn… |
| L25_V008:seg_00008:u0002:p000 | 529.78–541.22 | removed | Giáo dục không phải là chuẩn bị cho cuộc sống, nó chính là cuộc sống. Từ môi trường giáo dục này, chúng tôi đã… |
| L25_V008:seg_00009:u0003:p000 | 642.56–682.95 | partially_filtered | đến số 40 Từ app dùng công thức, chúng ta sẽ có số cuối trừ số đầu chia khoảng cách cộng 1 và ta sẽ được 14 số… |
| L25_V008:seg_00010:u0003:p000 | 911.00–930.62 | partially_filtered | Vậy thì công việc chúng ta đã hoàn thành Vậy thì số phần tử của biến cố E Ta sẽ lấy 6 giai thừa nhân cho A37 V… |
| L25_V008:seg_00011:u0000:p000 | 930.62–960.24 | removed | Đại học Quốc tế Sài Gòn SIU Không chỉ cung cấp chương trình đào tạo tiêu dựng quốc tế SIU nỗ lực tạo ra bộ trư… |
| L25_V008:seg_00011:u0001:p000 | 960.72–990.38 | partially_filtered | SIU cam kết hỗ trợ sinh viên với nhiều chính sách học bỏng, 20 số học bỏng chủ tình SIU, 100 số học bỏng tiếp … |
| L25_V011:seg_00003:u0000:p000 | 20.54–56.81 | partially_filtered | Trong chương trình Ông Tạp ngày hôm nay, thầy xin giới thiệu với tất cả chúng ta một chuyên đề về các tổ chức … |
| L25_V011:seg_00003:u0001:p000 | 57.32–86.94 | partially_filtered | SIU cam kết hỗ trợ sinh viên với nhiều chính sách học bỏng, 20 xuất học bỏng chủ tình SIU, 100 xuất học bỏng t… |
| L25_V011:seg_00006:u0005:p000 | 327.29–365.75 | removed | Trường Quốc tế Á Châu, tự hào là nguôi trường mang đến nền giáo dục tiên tiến, hiện đại, nhưng vẫn giữ dịnh bả… |
| L25_V011:seg_00006:u0006:p000 | 377.77–387.91 | removed | Nơi đây với tôi còn hơn cả một môi trường. Nơi đây là nền tảng giúp chúng tôi đặt những phiên gạch đầu tiên vữ… |
| L25_V011:seg_00006:u0007:p000 | 391.69–403.16 | removed | Giáo dục không phải là chuẩn bị cho cuộc sống, nó chính là cuộc sống. Từ môi trường giáo dục này, chúng tôi đã… |
| L25_V011:seg_00008:u0001:p000 | 743.33–790.03 | partially_filtered | Như vậy, chúng ta vừa mới điểm xa qua những mốc thời gian quan trọng dẫn đến sự ra đời của Liên minh châu Âu. … |
| L25_V011:seg_00008:u0002:p000 | 790.03–826.00 | removed | nghề nghiệp trong tương lai Tất cả sinh viên đều có cơ hội để thể hiện và phát triển năng lực Những ý tưởng lu… |
| L25_V037:seg_00003:u0000:p000 | 28.62–66.17 | partially_filtered | biết cách mạng tháng 8 đã diễn ra thành công dựa đến sự ra đời của nước Việt Nam Dân Chủ Cộng Hòa. Và một tron… |
| L25_V037:seg_00003:u0001:p000 | 66.17–101.96 | removed | nghiệp trong tương lai Tất cả sinh viên đều có cơ hội để thể hiện và phát triển năng lực Những ý tưởng luôn đư… |
| L25_V037:seg_00007:u0000:p000 | 363.98–412.49 | partially_filtered | đầu thập niên 30, sự xuất hiện của trường nghĩa fascist ở Đức, Ý, Nhật đã là nguy cơ đe dọa hà bình thế giới. … |
| L25_V037:seg_00007:u0001:p000 | 414.20–423.72 | removed | Ngoài những giờ học trên lớp, học sinh trường quốc tế Á Châu còn được trải nghiệm, thử thách bản thân và phản … |
| L25_V037:seg_00007:u0002:p000 | 435.89–446.05 | removed | Nơi đây với tôi còn hơn cả một ngôi trường. Nơi đây là nền tảng giúp chúng tôi đặt những viên gạch đầu tiên vữ… |
| L25_V037:seg_00007:u0003:p000 | 449.64–461.25 | removed | Giáo dục không phải là chuẩn bị cho cuộc sống, nó chính là cuộc sống. Từ môi trường giáo dục này, chúng tôi đã… |
| L25_V037:seg_00008:u0003:p000 | 756.26–791.58 | removed | Trường Đại học Quốc tế Sài Gòn SIU Không chỉ cung cấp chương trình đạo tạo tiêu dụng quốc tế SIU nỗ lực tạo ra… |
| L25_V037:seg_00008:u0004:p000 | 791.58–825.11 | partially_filtered | SIU 100 xuất học bổng Tiếp bước thành trình Quỹ Tài Năn Trạng Học bổng 50% học phí toàn khoá cho những học sin… |
| y1CvB950VlU:seg_00019:u0000:p000 | 270.15–286.53 | partially_filtered | hãy đăng ký kênh và đăng ký kênh để ủng hộ kênh và đăng ký kênh để ủng hộ kênh và đăng ký kênh để ủng hộ kênh … |
| y1CvB950VlU:seg_00019:u0001:p000 | 295.41–332.10 | partially_filtered | kênh và đăng k Ví dụ như chúng ta cái chuỗi A2 chúng ta để xóa ra cách đi Chuỗi A2 chúng ta nè Thì chúng ta dù… |
| y1CvB950VlU:seg_00021:u0000:p000 | 346.24–385.03 | partially_filtered | thằng đen A2 thôi Rồi cái đoạn này bắt đầu nó print ra này Rồi cái đoạn này bắt đầu nó print ra này Tức là nó … |
| y1CvB950VlU:seg_00022:u0000:p000 | 398.64–410.42 | partially_filtered | kênh để Mình muốn là nó theo một cái hàng ngang Tức là xuất theo hàng ngang, xuất nối lui nhau Thì chúng ta sử… |
| _eDGTHhyQ7M:seg_00003:u0000:p000 | 88.08–134.83 | partially_filtered | là một quốc gia đầu tiên ở Việt Nam. Tôi muốn mấy quốc gia ở Asia to achieve international accreditation of IA… |
| _eDGTHhyQ7M:seg_00003:u0001:p000 | 135.29–169.58 | removed | SIU mang đến những chương trình đào tạo chức lượng cập nhật theo số hướng toàn cầu với 29 chuyên ngành bật đại… |
| _eDGTHhyQ7M:seg_00003:u0002:p000 | 169.58–206.24 | removed | lửa South Korea, India, and Taiwan, China, offering us with many parts of study. SIU tự hào với sứ mệnh giữ bả… |
| _eDGTHhyQ7M:seg_00003:u0003:p000 | 207.05–247.50 | partially_filtered | SIU không ngừng nỗ lực xây dựng một môi trường học tập năng động, đa văn hóa, đạt nguyện quốc tế, giúp sinh vi… |
| _eDGTHhyQ7M:seg_00003:u0004:p000 | 248.16–264.66 | removed | Sinh viên, cựu sinh viên, học viên, cựu học viên SIU đã gặt hái được những thành tựu xuất sắc, thành công tron… |
| _eDGTHhyQ7M:seg_00011:u0000:p000 | 1521.33–1538.02 | partially_filtered | Cảm ơn các em nha Bản đồ giáo dục toàn tô, ngoài trang bị nền tảng, cơ sở vật chất, thiện nghi, hiện đại Chúng… |
| _eDGTHhyQ7M:seg_00021:u0001:p000 | 2391.32–2424.61 | partially_filtered | giáo dục tiên phong, tôn cảm sự khác biệt và đa quan hóa. SRU đã đạt kiểm định các cơ sở giáo dục, cũng như Ch… |
| _eDGTHhyQ7M:seg_00021:u0002:p000 | 2425.26–2460.70 | removed | Năm 2024, chương trình tử nhân khoa học máy tính của SIU đã được công nhận đạt chủng quốc tế bởi ủy ban kiểm đ… |
| _eDGTHhyQ7M:seg_00021:u0003:p000 | 2461.14–2507.66 | removed | công nghệ thông tin, công nghệ giáo dụng, nguồn nữ Anh, đông phương học, tâm lý học, quan hệ công chúng, truyề… |
| _eDGTHhyQ7M:seg_00021:u0004:p000 | 2508.02–2548.52 | partially_filtered | Chương trình thực tập của sinh viên được hỗ trợ bởi hơn 105 công ty trong và ngoài nước. SIU không ngừng nỗ lự… |
| _eDGTHhyQ7M:seg_00021:u0005:p000 | 2548.52–2571.18 | partially_filtered | 100% sinh viên có TOEFL IBTE 79 trở lên, TOEAS 500 trở lên sinh viên, tự sinh viên, học viên, cựu học viên SIU… |

Remaining promotional-lint matches are conservative misses, all retained for review:

| Chunk | Source | Seconds | Text excerpt |
|---|---|---:|---|
| L25_V008:seg_00002:u0002:p000 | ocr | 8.41–89.89 | SIU INTERNATIONAL UNIVERSITY THE SAIGON INTERNATIONAL UNIVERSITY www.siu.edu.vn Lewis Campus |
| _eDGTHhyQ7M:seg_00002:u0000:p000 | ocr | 52.32–54.55 | THE SAIGON INTERNATIONAL UNIVERSITY 2007 The Saigon International University |
| _eDGTHhyQ7M:seg_00004:u0001:p000 | ocr | 271.44–274.27 | THE SAIGON INTERNATIONAL UNIVERSITY 2007 Vietnam's spirit, global education. |
| _eDGTHhyQ7M:seg_00020:u0000:p000 | ocr | 2359.06–2361.86 | THE SAIGON INTERNATIONAL UNIVERSITY 2007 The Saigon International University |
| _eDGTHhyQ7M:seg_00022:u0000:p000 | ocr | 2577.68–2580.78 | THE SAIGON INTERNATIONAL UNIVERSITY 2007 Vietnam's spirit, global education. |

## Reproduction and validation

The runner calls rebuild_offline.py and eval_retrieval.py for every fresh variant, with phase-2 options omitted to use defaults. Exact (prefix,text) CPU ONNX vectors are reused in memory across variants only to save execution cost; all vectors originate from the pinned real encoder and remain unit normalized. Warm reads use only completed per-video stage-05 outputs, even when a later video in that experiment was interrupted. Audio is restored by the unchanged normalization path and export/MP4 checksums are verified. See BASELINE.md for the restored-WAV byte-identity limitation. Rebuild refuses existing staged inputs.

Repeated asset copies filled C: and interrupted several exploratory runs. The harness now shares only checksum-identical manifest assets with a completed workspace reference via NTFS hard links; both source and destination hashes are verified before replacement, differing text/vectors remain independent, and resolved write paths must stay in runs/exp. Incomplete stages are never repaired or counted as results. Existing archived experiment junctions on D: are read only. No source export asset was changed.

```powershell
& backend/.venv/Scripts/python.exe pipeline/eval/run_phase3.py --prefix phase3-reproduction --audio-reference pipeline/runs/exp/phase3-reproduction-reference --variants reference ocr ocr-subscript titles asr-off asr-glossary asr-glossary-lecture asr-glossary-code failures ads ads-schools ads-signature figures combined combined-reviewed combined-code-schools combined-code-schools-figures combined-subscript
& backend/.venv/Scripts/python.exe pipeline/eval/rebuild_offline.py --export-root D:/thesis-data/run-20261020/export --name phase3-production-reproduction --config pipeline/eval/production_config.json --audio-reference pipeline/runs/exp/phase3-a-reference --mp4-root D:/thesis-data/mp4-extra --ffmpeg D:/FilmTools/Shotcut/ffmpeg.exe
& backend/.venv/Scripts/python.exe pipeline/eval/eval_retrieval.py --index-root pipeline/runs/exp/phase3-production-reproduction --name phase3-production-reproduction-eval
python pipeline/builders/build_notebooks.py
python pipeline/builders/build_fast_pipeline.py
python pipeline/builders/build_full_walkthrough.py
python pipeline/builders/build_run_pipeline.py
$env:PATH = 'D:/FilmTools/Shotcut;' + $env:PATH
python -m pytest pipeline/tests -q
& backend/.venv/Scripts/python.exe -m unittest discover -s pipeline/tests -p test_eval_harness.py -q
& backend/.venv/Scripts/python.exe -m unittest discover -s pipeline/tests -p test_text_phase3.py -q
& backend/.venv/Scripts/python.exe -m unittest discover -s backend/tests -q
```

Use fresh names when reproducing. Per-run config/code/input hashes, restored audio provenance and rebuild comparisons are in rebuild_report.json. The reference rebuild must retain full text/ID/token parity and cosine >0.999; production defaults and the captured JSON are checked by a focused test. Generated notebooks embed the current sources.

Actual experiment mapping (all interrupted partial runs are excluded):

```json
{
  "reference": "phase3-a-reference",
  "ocr": "phase3-a-ocr",
  "ocr-subscript": "phase3-m-ocr-subscript",
  "titles": "phase3-b-titles",
  "asr-off": "phase3-g-asr-off",
  "asr-glossary": "phase3-g-asr-glossary",
  "asr-glossary-lecture": "phase3-i-asr-glossary-lecture",
  "asr-glossary-code": "phase3-l-asr-glossary-code",
  "failures": "phase3-g-failures",
  "ads": "phase3-g-ads",
  "ads-schools": "phase3-l-ads-schools",
  "ads-signature": "phase3-n-ads-signature",
  "figures": "phase3-c-figures",
  "combined": "phase3-g-combined",
  "combined-reviewed": "phase3-i-combined-reviewed",
  "combined-code-schools": "phase3-l-combined-code-schools",
  "combined-code-schools-figures": "phase3-l-combined-code-schools-figures",
  "combined-subscript": "phase3-m-combined-subscript"
}
```

To reproduce this report from those existing artifacts:

```powershell
& backend/.venv/Scripts/python.exe pipeline/eval/summarize_phase3.py --prefix phase3-l --final phase3-m-combined-subscript --variants reference ocr ocr-subscript titles asr-off asr-glossary asr-glossary-lecture asr-glossary-code failures ads ads-schools ads-signature figures combined combined-reviewed combined-code-schools combined-code-schools-figures combined-subscript --experiment reference=phase3-a-reference --experiment ocr=phase3-a-ocr --experiment ocr-subscript=phase3-m-ocr-subscript --experiment titles=phase3-b-titles --experiment asr-off=phase3-g-asr-off --experiment asr-glossary=phase3-g-asr-glossary --experiment asr-glossary-lecture=phase3-i-asr-glossary-lecture --experiment failures=phase3-g-failures --experiment ads=phase3-g-ads --experiment ads-signature=phase3-n-ads-signature --experiment figures=phase3-c-figures --experiment combined=phase3-g-combined --experiment combined-reviewed=phase3-i-combined-reviewed --experiment combined-subscript=phase3-m-combined-subscript
```

### Recorded verification

Recorded UTC: 2026-11-01T17:51:47.236282+00:00. Full command output and exit codes are in eval/phase3_validation.json. FFmpeg/FFprobe are provided by D:/FilmTools/Shotcut; no packages were installed.

| Check | Exit code | Result |
|---|---:|---|
| `python -m pytest pipeline/tests -q` | 0 | 122 passed, 14 subtests passed in 21.77s |
| `backend/.venv/Scripts/python.exe -m unittest discover -s pipeline/tests -p test_eval_harness.py -q` | 0 | Ran 15 tests in 0.384s; OK |
| `backend/.venv/Scripts/python.exe -m unittest discover -s pipeline/tests -p test_text_phase3.py -q` | 0 | Ran 24 tests in 0.904s; OK |
| `backend/.venv/Scripts/python.exe -m unittest discover -s backend/tests -q` | 0 | Ran 57 tests in 3.277s; OK (skipped=2) |
