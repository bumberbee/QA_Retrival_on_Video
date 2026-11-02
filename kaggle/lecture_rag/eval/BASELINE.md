# Phase 1 baseline: Multimodal Lecture RAG

Recorded with the unchanged exported production run at `D:/thesis-data/run-20261020/export` on branch `upgrade/extraction-fixes`. Measured UTC: `2026-11-01T15:23:26.667602+00:00`. No pipeline/backend source behavior was changed, no GPU inference was run, and no commit, branch switch or push was performed. Source data on D: was read only.

The question set contains **120 questions: 11 answerable and 1 unanswerable per video**. Its status is **auto-written by Codex, needs human review**. Codex inspected 71 selected stage-01 keyframes, read the corresponding raw transcript word spans, and read additional surrounding spans for spoken explanations. Image references, times, reference answers and checked quotes are recorded per question. This is a draft development set, not an independently human-reviewed benchmark. No acoustic listening was performed. The seed example visibly says type II 1.5 kg but uses 2/5 in the calculation; that lecturer inconsistency is preserved in the annotation.

Question SHA-256: `e07c70b3a8122eb73c9cf6c1a343b794117da56c15e54c0c657001a38d7005c2`. Backend retrieval SHA-256: `6ef35821ec1bb8debfae11a6360d8abb650d7ca86d11b759ea7230a0375c47a7`. JSON results also record harness hashes and input chunk hashes.

## Measurement contract

- Each question calls the real backend `search(video, question)` and `build_context(found)` in single-video file mode, with no previous question or playback time. Query vectors come from its pinned CPU ONNX encoder. Batched query encoding changes only execution cost.
- A returned slide hits when its half-open time interval has any positive overlap with a gold span. Gold uses video seconds, never chunk IDs. Most visual gold is narrowly bounded around an actually inspected keyframe; speech gold uses inspected word ranges. Long production slide intervals can inflate time hits; this is not a timestamp-accuracy metric.
- Production returns at most **three slides**. **Hit@5 therefore equals Hit@3**. MRR uses the first hit among the actual returned slides, and zero if none hits. No production constant is expanded to obtain five candidates.
- Unanswerable questions are excluded from hit/MRR/context denominators. Their top E5 cosine and fused RRF scores are retained below; no abstention threshold is applied.
- **Text context recall** requires gold-time overlap and every annotated key term in both the retained evidence and the actual `build_context` output. Speech words are mapped to character offsets, then checked against the 2,500-character cutoff. OCR/title evidence is checked against the 1,800-character OCR cutoff and the returned slide interval. Figures follow the actual three-figure/600-character limits. Source annotations distinguish speech, OCR, and either. Fuzzy matching folds accents and uses a ratio of 0.88; numbers and short identifiers require exact token matches. This is a key-term proxy, not complete semantic answer grading. It measures text, not the images the multimodal LLM also receives.
- The JSON includes actual contexts, retained word IDs, partially retained chunks, character omissions, time/term checks and per-question results. There are **zero unmapped word IDs** in the evaluated contexts. The economics risk/choice question `v2YTz8ToEag_q06` has Hit@1=1 and context recall=0: 3,635 speech characters from its matching slide are omitted.

## Rebuild parity

The driver calls the existing `run_alignment` and `run_indexing`, loaded in the same shared namespace as the notebooks. It reads generator configuration and uses the original chunking, cleaning, correction, relevance and SQLite functions. A narrow SentenceTransformer-interface adapter uses backend `OnnxEncoder.encode_many`; the embedding text already has exactly one `passage: ` prefix, while speech relevance uses `query: `. The adapter preserves the production Transformers decoder's punctuation cleanup and uses an untruncated tokenizer for chunk splitting. No torch is imported.

All **1,233 chunk IDs, order, texts, embedding texts and encoded token counts match exactly**. The minimum rebuilt/export embedding cosine is **0.99999988**, comfortably above 0.999. Exported and rebuilt retrieval have identical returned segment order, contexts, scores and metrics for all 120 questions.

| Video | Chunks | IDs/text/counts | Minimum cosine | Rebuild seconds |
|---|---:|---|---:|---:|
| cs_0001 | 59 | exact | 0.99999994 | 11.84 |
| D_kuulxKuxc | 186 | exact | 0.99999988 | 22.56 |
| hVU5Mw19V_c | 73 | exact | 0.99999988 | 14.26 |
| L25_V008 | 49 | exact | 0.99999994 | 26.30 |
| L25_V011 | 33 | exact | 0.99999988 | 23.56 |
| L25_V037 | 37 | exact | 1.00000000 | 28.31 |
| tHE1mC_n0-E | 455 | exact | 0.99999988 | 193.06 |
| v2YTz8ToEag | 130 | exact | 0.99999988 | 41.16 |
| y1CvB950VlU | 109 | exact | 0.99999988 | 15.64 |
| _eDGTHhyQ7M | 102 | exact | 0.99999988 | 58.02 |

Stage-04 words, visual records, aligned speech and clip time metadata match the export. Seven of 1,122 relevance windows differ at rounding boundaries: at most 0.0001 in topic score and 0.001 in topic z. Two segment z values differ by 0.001. This is the expected numerical difference between CPU ONNX and the original encoder; speech filtering is disabled in production, so it changes no chunks or rankings.

The portable export omits its manifest-required WAVs. The harness checks all available input checksums, copies stages 01/02/03 under the experiment, verifies the MP4 source hash, and calls the existing CPU `normalize_audio` using local FFmpeg. Only copied stage-03 manifests are re-signed. **All ten restored WAV file hashes differ from the unavailable exported originals**; their byte identity and original PCM identity cannot be verified. Real audio is regenerated, with no dummy samples or relaxed validation. This distinction is recorded in `restored_assets`; the requested text/vector parity still passes.

## Retrieval and quality results

Questions: auto-written by Codex, needs human review

Hit@5 uses the unchanged production maximum of three slides, so it equals Hit@3.

| Video / type | Answerable | Hit@1 | Hit@3 | Hit@5 | MRR | Context recall |
|---|---:|---:|---:|---:|---:|---:|
| cs_0001 | 11 | 0.364 | 0.818 | 0.818 | 0.576 | 0.818 |
| D_kuulxKuxc | 11 | 0.364 | 0.545 | 0.545 | 0.439 | 0.545 |
| hVU5Mw19V_c | 11 | 0.364 | 0.818 | 0.818 | 0.530 | 0.455 |
| L25_V008 | 11 | 0.727 | 1.000 | 1.000 | 0.848 | 0.909 |
| L25_V011 | 11 | 0.818 | 0.909 | 0.909 | 0.848 | 0.818 |
| L25_V037 | 11 | 0.909 | 0.909 | 0.909 | 0.909 | 0.818 |
| tHE1mC_n0-E | 11 | 0.273 | 0.364 | 0.364 | 0.318 | 0.364 |
| v2YTz8ToEag | 11 | 0.455 | 0.727 | 0.727 | 0.576 | 0.455 |
| y1CvB950VlU | 11 | 0.364 | 0.545 | 0.545 | 0.455 | 0.364 |
| _eDGTHhyQ7M | 11 | 0.545 | 0.909 | 0.909 | 0.712 | 0.545 |
| OVERALL | 110 | 0.518 | 0.755 | 0.755 | 0.621 | 0.609 |
| type: both | 13 | 0.769 | 1.000 | 1.000 | 0.859 | 1.000 |
| type: code | 4 | 0.000 | 0.250 | 0.250 | 0.125 | 0.250 |
| type: numeric | 26 | 0.577 | 0.731 | 0.731 | 0.641 | 0.615 |
| type: slide | 22 | 0.500 | 0.636 | 0.636 | 0.561 | 0.545 |
| type: speech | 23 | 0.609 | 0.957 | 0.957 | 0.768 | 0.609 |
| type: term | 12 | 0.333 | 0.667 | 0.667 | 0.472 | 0.667 |
| type: visual | 10 | 0.300 | 0.600 | 0.600 | 0.433 | 0.300 |

| Video | Chunks | Ads* | Exact OCR excess | Near OCR pairs* | OCR / ASR loop chunks* | <32 tokens | Empty titles | Bad corrections* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cs_0001 | 59 | 0 | 2 | 0 | 0 / 0 | 6 | 3/21 | 0 |
| D_kuulxKuxc | 186 | 1 | 25 | 0 | 0 / 0 | 92 | 91/103 | 0 |
| hVU5Mw19V_c | 73 | 0 | 0 | 4 | 0 / 0 | 4 | 0/18 | 0 |
| L25_V008 | 49 | 5 | 1 | 0 | 0 / 0 | 1 | 4/15 | 0 |
| L25_V011 | 33 | 3 | 0 | 0 | 0 / 2 | 0 | 5/10 | 0 |
| L25_V037 | 37 | 4 | 0 | 0 | 0 / 0 | 2 | 4/10 | 0 |
| tHE1mC_n0-E | 455 | 0 | 81 | 53 | 0 / 0 | 41 | 37/203 | 2 |
| v2YTz8ToEag | 130 | 0 | 0 | 0 | 0 / 0 | 6 | 7/43 | 7 |
| y1CvB950VlU | 109 | 3 | 6 | 15 | 0 / 2 | 28 | 11/54 | 0 |
| _eDGTHhyQ7M | 102 | 7 | 4 | 0 | 0 / 1 | 8 | 6/25 | 3 |

*Diagnostic rules, not verified corpus error rates. Near pairs: unique OCR texts, 3-token Jaccard ≥0.90 and ≥16 tokens; loops: ≥8 consecutive repeats of 1–8 tokens. Ads: fixed phrase catalogue. Bad corrections: four documented phrase pairs, checked against text_asr changes.

| Unanswerable ID | Top E5 cosine | Top fused RRF |
|---|---:|---:|
| cs_0001_q12 | 0.8193 | 0.025820 |
| D_kuulxKuxc_q12 | 0.8709 | 0.023243 |
| hVU5Mw19V_c_q12 | 0.8265 | 0.022903 |
| L25_V008_q12 | 0.8335 | 0.023938 |
| L25_V011_q12 | 0.8401 | 0.024544 |
| L25_V037_q12 | 0.8331 | 0.024771 |
| tHE1mC_n0-E_q12 | 0.8593 | 0.021434 |
| v2YTz8ToEag_q12 | 0.8351 | 0.025134 |
| y1CvB950VlU_q12 | 0.8486 | 0.023942 |
| _eDGTHhyQ7M_q12 | 0.8422 | 0.024361 |

Corpus lint totals:

| Lint | Count |
|---|---:|
| chunks | 1233 |
| segments | 502 |
| promotional_chunks | 23 |
| exact_duplicate_ocr_excess | 119 |
| near_duplicate_unique_ocr_pairs | 72 |
| ocr_loop_chunks | 0 |
| asr_loop_chunks | 5 |
| chunks_under_32_tokens | 188 |
| empty_titles | 168 |
| asr_correction_events | 112 |
| changed_asr_words | 137 |
| known_bad_asr_corrections | 12 |

The empty-title rate is 168/502 = 33.47% of all segment records, including skipped OCR; JSON also reports empty chunk-title rate. Exact OCR excess counts copies after the first of each exact text. Near-duplicate pairs count distinct exact-text groups, so they are not an additional count of redundant chunks. Promotional counts use an expanded phrase catalogue and are review candidates, not verified ad durations.

The known-bad correction catalogue contains the four explicitly documented phrase pairs: “trong kinh” → “chọn kinh”, “trọng một” → “chọn một”, “chân không” → “trạng không”, and “học chương” → “học trường”. It counts 12 events supported by actual `text_asr` changes, out of 112 total events / 137 changed words. It is a conservative subset of the audit's at-least-18 contextual regressions; unenumerated contextual errors are not guessed. Loop counts concern the cleaned index using the stated rule; zero OCR loop chunks does not imply healthy raw OCR (the audit found seven raw OCR loop records). The five indexed ASR loop chunks are also rule-based; a different repeat threshold can produce a different count.

## Reproduction

Run from the repository root in PowerShell. Existing canonical artifacts are under `runs/exp/baseline`, `runs/exp/baseline-rebuild-clean`, and `runs/exp/baseline-rebuilt-eval`. The entire runs directory is git-ignored. Choose a fresh rebuild name because the driver refuses to overwrite staged inputs.

```powershell
& backend/.venv/Scripts/python.exe kaggle/lecture_rag/eval/eval_retrieval.py --index-root D:/thesis-data/run-20261020/export --name baseline

& backend/.venv/Scripts/python.exe kaggle/lecture_rag/eval/rebuild_offline.py --export-root D:/thesis-data/run-20261020/export --name reproduce-production --config kaggle/lecture_rag/eval/production_config.json --mp4-root D:/thesis-data/mp4-extra --ffmpeg D:/FilmTools/Shotcut/ffmpeg.exe

& backend/.venv/Scripts/python.exe kaggle/lecture_rag/eval/eval_retrieval.py --index-root kaggle/lecture_rag/runs/exp/reproduce-production --name reproduce-production-eval
```

Both drivers accept `--videos cs_0001 ...`. For a single arbitrary stage-05 directory, use `eval_retrieval.py --index-dir PATH --videos VIDEO --name NAME`. `--questions` selects another time-based question file. `--config` accepts partial JSON overrides of the captured production config; unknown keys/model revisions fail explicitly. The current harness supports the pinned `intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. HF files are read from the local cache with offline mode enabled; no downloads are attempted. FFmpeg's explicit path can be changed for another Windows machine.

`retrieval.json` / `retrieval.md` are emitted under each evaluation name. `rebuild_report.json` records config, source-module hashes, restored-audio provenance, bundle differences and parity. Each rebuilt video contains `_inputs`, `04_bundle`, and `05_index`. The backend is loaded under an isolated package with a file-only storage stub and a temporary `index_dir` monkeypatch; app startup, real storage, `.env`, Supabase and R2 modules are never imported. Every script blanks all present SUPABASE_/R2_ variables plus the six test-isolation settings before imports.

## Tests and limits

The requested venv command was attempted and failed with `No module named pytest`; pytest was not installed there. The venv also lacks Pillow/OpenCV/PyAV for the existing media tests. No package was installed globally or in the venv, and torch was not installed.

The **13 fast harness unittest tests pass in the backend venv**. The **complete suite passes with the already-installed system Python: 77 passed, 6 subtests passed, zero skipped**, after exposing the existing local FFmpeg/FFprobe to that test process. The initial system run skipped 16 media tests because FFmpeg was not in PATH; those tests were subsequently executed successfully.

```powershell
& backend/.venv/Scripts/python.exe -m pytest kaggle/lecture_rag/tests -q  # unavailable: pytest missing
& backend/.venv/Scripts/python.exe -m unittest discover -s kaggle/lecture_rag/tests -p test_eval_harness.py -q

$env:PATH = 'D:/FilmTools/Shotcut;' + $env:PATH
python -m pytest kaggle/lecture_rag/tests -q
```

Questions still need human review, including acoustic confirmation of spoken-only references and unanswerable labels. Offline recomputation cannot recover board states never selected by stage 01, OCR rejected by the GPU gate, or missing/wrong raw OCR/ASR evidence. Phase 1 records those limits and leaves all production behavior unchanged.
