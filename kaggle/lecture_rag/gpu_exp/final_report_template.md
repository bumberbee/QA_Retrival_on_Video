# Multimodal Lecture RAG — private GPU experiments

Completed on 2 November 2026 (Asia/Saigon). Measurements, failed attempts and review artifacts are retained under `../runs/gpu/`. The results support **plain faster-whisper medium for the next run with the current retrieval stack, and Qwen3-VL-4B with content crops, bounded pixels and EOS/loop validation for OCR**. This is a pipeline decision; no acoustic WER winner is established.

## Recommendation for the next full run

**ASR:** keep `Systran/faster-whisper-medium`, pinned below, FP16, beam 5, temperature 0, `condition_on_previous_text=False`, no `initial_prompt` or `hotwords`, and the existing pause-aware approximately 45-second windows with 1.25-second overlap. Keep Vietnamese/English language selection, VAD silence 500ms and padding 250ms consistent with the reproduced baseline. The exported five-video baseline was reproduced exactly, including word text, times and probabilities. It has the strongest measured context support (23/24 questions), joint-best Hit@1 (17/24), and joint-best MRR on the controlled benchmark. Its measured decode cost is **0.07557 GPU execution hours per audio hour** on one T4, excluding loading. With two allocated T4s and one used, that becomes approximately **0.15113 allocated T4-hours per audio hour** before setup. Retaining it has no demonstrated ASR accuracy gain; it avoids the measured downstream regressions of the alternatives.

Do not deploy the tested double-glossary/fallback configuration. Medium copied the supplied glossary, introduced five long repetition loops, and reduced Hit@1 from 17/24 to 13/24 and context recall from 23/24 to 16/24 while increasing decoding cost 3.50×. PhoWhisper is a promising Vietnamese wording candidate: it recovered `xác suất` far more consistently and removed the history loop. Adoption should follow reviewed references, alignment validation and an indexing/number-format experiment. Plain Pho costs 2.22× medium decoding here; its raw context recall is 17/24, or 20/24 under a post-hoc shared numeric-format sensitivity check. These measurements do not justify changing the recognizer in isolation.

Turbo is the throughput alternative: it cuts measured decoding time by 46.1% relative to medium, but loses two Hit@1 questions and recovers fewer canonical `xác suất` occurrences in this sample. ChunkFormer is much faster, but its English regression and emission-time contract prevent choosing it as the sole recognizer for this corpus.

**OCR:** use `Qwen/Qwen3-VL-4B-Instruct` FP16/SDPA, a content crop that preserves all slide text, processor pixel bounds `256*32*32` to `1280*32*32`, the existing OCR prompt plus literal code/quote/indentation instructions, greedy generation, 1536 output tokens and repetition penalty 1.05. Require a genuine EOS and reject eight or more consecutive repeats of phrases up to 32 tokens. Retry once with literal-only instructions, 2048 tokens and penalty 1.10; retain an invalid status if it still fails. All 29 enhanced predictions passed offline EOS/loop review; no retry was needed. The originally executed broad comparison used the shorter loop validator, so the 32-token validation is an additional defensive setting, not a measured retry effect.

On the 20 draft references, enhanced Qwen reduced CER **0.33755→0.31033**, increased `1−WER` **0.62562→0.64484**, and removed four unfinished/loop outputs. Diacritic agreement fell **0.92053→0.91267**. Mean generation cost was **16.787 seconds/image**, about **0.004663 execution GPU-hours/slide** (twice that for the allocated dual-T4 machine), excluding loading. The measured 47.8% mean time saving comes from avoiding four failures: on the 25 originally valid images, mean times are **15.5069 whole / 15.5602 enhanced**. Do not extrapolate a 48% full-run speedup without its loop rate.

Use tight region rereads for suspicious numbers/formulas/code, preserving the original crop and both outputs for review. The five focused value anchors improved from 0/5 to 4/5, but neither 4B nor 8B produced the exact percentage and unit, and neither preserved all code quotes. PP-OCRv5 Latin can supply a cheap independent numeric/box check (0.574 seconds/image here); its Vietnamese text quality is unsuitable as the main OCR. Qwen8B NF4 fits a T4 but showed no advantage across the five focused anchors and cost 1.66× the 4B crop generation time. No corpus-level 8B CER was measured.

For the optional answer stage, retain **qwen2.5:7b** as the tested baseline: 21/30 fully correct draft-reference judgments versus Vintern's 13/30. Qwen3's non-thinking behavior could not be verified on the pinned integration, despite two retries; this part remains an incomplete model comparison.

{{OCR_FINAL_NOTE}}

## Setup, authorization and inputs

Owner: `thnhtrungnguynmif`. Transport: authenticated Kaggle CLI **2.2.4**. Credentials were never opened, printed or copied; Kaggle MCP was not used. Every pushed kernel is private, GPU/internet enabled, and uses `machine_shape=NvidiaTeslaT4`. Returned metadata is retained in `../runs/gpu/metadata/`; [kernel_verification.json](../runs/gpu/analysis/kernel_verification.json) verifies the final versions. No dataset was created and no artifact was made public. At most two experiment kernels ran concurrently; only CUDA device 0 performed inference, while budget accounting counts both allocated T4s.

Pinned image: `gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435`. The image reported Python 3.13.15, torch/torchaudio 2.11.0+cu128, torchvision 0.26.0+cu128, driver 580.178.04 and two Tesla T4s. Per-run environment JSON captures installed packages.

Branch remains `upgrade/extraction-fixes`; no Git commit, push or branch switch. Experiment code is confined to this directory. Existing `backend/`, `build_fast_pipeline.py` and `notebook_code/*.py` were not edited by this work. Concurrent changes elsewhere in the working tree were left in place.

Dataset membership was checked through the prior kernel metadata and `kaggle datasets files`:

| Input | Videos/artifacts used |
|---|---|
| `thnhtrungnguynmif/vn-lecture-test12` | `D_kuulxKuxc`, `hVU5Mw19V_c`, `tHE1mC_n0-E`, `v2YTz8ToEag`, `y1CvB950VlU`, `_eDGTHhyQ7M` |
| `thnhtrungnguynmif/l25a-lecture-videos-media` | `L25_V008`, `L25_V011`, `L25_V037`, under `Videos_L25_a/video/` |
| `thnhtrungnguynmif/slidespeech-train100` | English `cs_0001`, under `slidespeech-train100/video/` |
| Kernel `thnhtrungnguynmif/fast-pipeline-10-mixed-20261020` | Fixed stage01 keyframes, stage02 OCR, stage03 transcripts and historical helper code |
| `thnhtrungnguynmif/ocr-bench-vi-slides` | 20 draft GT images: 19 printed, one handwritten; includes an English suffix slide |

E1 transcribed the **complete** `hVU5Mw19V_c`, `v2YTz8ToEag`, `y1CvB950VlU`, `L25_V011`, `cs_0001` lectures: **7534.27 seconds / 2.093 audio hours**. MP4 hashes were checked against production manifests before normalization. Baseline normalized WAV hashes and all transcript objects match the exports; see [baseline_parity.json](../runs/gpu/analysis/baseline_parity.json).

## Pins

| Model | Exact model ID | Revision |
|---|---|---|
| Medium | `Systran/faster-whisper-medium` | `08e178d48790749d25932bbc082711ddcfdfbc4f` |
| Large-v3 | `Systran/faster-whisper-large-v3` | `edaa852ec7e145841d8ffdb056a99866b5f0a478` |
| Turbo | `dropbox-dash/faster-whisper-large-v3-turbo` | `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` |
| PhoWhisper | `vinai/PhoWhisper-large` | `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb` |
| ChunkFormer | `khanhld/chunkformer-ctc-large-vie` | `311fc03558a895dc2b32957f2fb4236c7fb1455b` |
| Qwen4B | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Qwen8B | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| PaddleOCR-VL 1.6 | `PaddlePaddle/PaddleOCR-VL-1.6` | `c5630abae1d940eafe0697512a0325494b02ab42` |
| PP detector | `PaddlePaddle/PP-OCRv5_server_det` | `ca867c897ecbca8873081573a802ad70d499cb94` |
| Latin recognizer | `PaddlePaddle/latin_PP-OCRv5_mobile_rec` | `ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57` |
| Vintern | `5CD-AI/Vintern-1B-v3_5` | `b98f263eab246eb5269ade64edbdca8a887dc44d` |

ASR pins: faster-whisper 1.2.1, CTranslate2 4.8.3, numpy 2.1.3, hub 0.36.0, av 15.1.0. Pho uses an FP16 CT2 conversion with transformers 4.57.1, accelerate 1.11.0, sentencepiece 0.2.1; conversion takes 39.47 seconds separately from decoding. ChunkFormer package 1.2.2 installs cleanly; the requested `khanhld/chunkformer-large-vie` redirects to the pinned CTC repository.

Qwen pins: transformers 4.57.1, hub 0.36.0, accelerate 1.11.0, Pillow 11.3.0. Qwen8B uses bitsandbytes 0.48.2, NF4 double quantization, FP16 compute. PP pins: paddlepaddle-gpu 3.2.1/cu126, paddleocr 3.7.0, paddlex 3.7.2, nvidia-nccl-cu12 2.28.9 restored after installation. PaddleVL native-loader failures retain incompatible package versions in logs; the final remote-code adapter uses transformers 4.57.1/hub 0.36.0.

Ollama **0.12.6**, Linux archive SHA-256 `de82adce2ab79235115d511ff22fcb099ac53b67127870f12b80198c033ec0a1`. Registry manifest SHA-256 pins: `qwen2.5:7b` = `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`; `qwen3:4b` = `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`. Model metadata is saved beside the responses. Vintern dependencies additionally pin einops 0.8.1 and timm 1.0.22.

## E1 — ASR measurements and review

There is no human transcript. Canonical/alias fractions below count matching recognized forms, **not omission-aware term accuracy or WER**. `0/0` means neither canonical nor primary-alias form was detected. Glossary copying can make these fractions look perfect even when the transcript is wrong.

| Video | Reviewed slide-supported glossary | Primary tracked mishearings |
|---|---|---|
| Bayes | Bayes, xác suất, biến cố, công thức xác suất đầy đủ | BS/Bias/BIS; sát xuất/sắc xuất/xác xuất/sát suất |
| Economics | PPF, đường giới hạn khả năng sản xuất, chi phí cơ hội, khan hiếm, kinh tế học | PPS |
| Python | Python, len, for, print, end, sep, chuỗi, string, a1, a2 | LAN; `4 4` |
| History | Liên Hiệp Quốc, Hội đồng Bảo an, quyền bình đẳng, chủ quyền | Counts retained; no predeclared alternative-name accuracy denominator |
| English | FIFO, MSCs, RSC, synchronous, asynchronous, partial order, message passing, decidable, undecidable | Counts retained; no acoustic reference |

Full supporting slide IDs are in [glossary_sources.json](glossary_sources.json); all metric term lists are in [analyze.py](analyze.py). `quyền bình đẳng` was checked against the original image rather than inherited OCR `quyền đăng ký`.

Long loops require eight consecutive repeats of a phrase of **1–32 word records**; every system was rescored after longer copied phrases appeared. Timing flags count word duration ≤1ms and >2s; they do not establish acoustic alignment error. RTF excludes downloads, audio/model loading and conversion. CT2 VRAM is sampled device usage after windows, a lower bound on peak; ChunkFormer VRAM is torch peak allocated tensors and is not directly equivalent.

{{ASR_TABLE}}

Decode totals for medium/large/turbo/Pho/Chunk are **569.34 / 922.25 / 307.01 / 1266.46 / 47.73 seconds**. Pho hints takes **1729.50s**, medium hints **1990.83s**. The partial medium-hints run was resumed for history/English with identical settings; seed 20261101 restarts in the resume. Curated full-five outputs include per-video source provenance under `../runs/gpu/curated/hints/medium_hints/`.

Twenty fixed textual spans per system were inspected: four per video, covering probability terms, PPF, Python identifiers, the history loop and English communication models. [review_all.md](../runs/gpu/analysis/review_all.md) holds all wording/time spans. Acoustic listening was unavailable in this runtime; 20 source WAV clips and hashes are retained in `../runs/gpu/analysis/span_audio/` for subsequent human review. No listening judgment or human WER is claimed.

| Checked span | Concrete result |
|---|---|
| Bayes opening / ~1557s | Medium writes Bias/BIS; large/turbo write bias. Pho writes `bai ét / bái ét`, Chunk `bay ét`, outside the primary catalogue. Pho's 68/68 `xác suất` contrasts with medium's 16/65; Chunk has five `xác xuất` occurrences. |
| Economics 1638–1675s | Plain large/turbo improve PPS→PPF. Pho also spells spaced `p p s`, omitted from the primary alias catalogue; Chunk writes `pp`. Pho hints writes PFF in the reviewed explanation. |
| Python 188–214s / 300–318s | Large/turbo improve LAN→len and `hàm 4 4`→`for for`; large still says `hàm lăn`, outside the catalogue. Plain native systems do not preserve canonical `print`/`end`: Pho has `pờ rin / en`, Chunk `prin / en`. |
| Python 407–429s | Plain Whisper systems still say `phím N bằng / hàm N bằng` for `end`. Medium hints has no timed words in this span. |
| History 578–600s | Medium repeats `tổ chức` 49 times at 585.14–592.96s; large/turbo/Pho/Chunk recover coherent ASEAN speech. |
| Economics timings | Large gives `kênh` a 452.63–470.77s duration with probability 0.99837, and `dân` 385.59–397.59s at probability 1.0. Fewer collapsed times and high confidence do not prove better alignment. |
| English | Plain Pho preserves reviewed meaning but changes FIFO to `five for`; Chunk's native output is garbled, e.g. `a tu m can of comication weith charles sinker`, with zero tracked canonical English terms. It is unsuitable as the sole bilingual recognizer. |

The tested glossary configuration supplies **both initial_prompt and hotwords**, fallback temperatures `[0,0.2,0.4,0.6]`, compression ratio 2.4, hallucination silence threshold 2.0, previous conditioning off, seed 20261101. This combined experiment does not isolate each option. Pho hints improves Bayes to 10/10, `xác suất` 73/73, len 9/9 and canonical print count 17, but still has no canonical end. English decode rises 185.7→635.0s and the 700–720s span loses much timed wording.

Medium hints is unsafe in this test. Bayes 1548–1572s collapses to `Bayes, xác suất, biến cố, công thức xác suất`. Economics repeatedly lists `PPF, đường giới hạn khả năng sản xuất, chi phí cơ hội, khan hiếm, kinh tế học`. Runs include `kinh tế học`×32, `chi phí cơ hội`×10 in 0.56s, `truy vấn chuỗi`×31, `sep`×73 and `sep chuỗi`×32. Its full-five totals are **656 collapsed durations and 30 durations >2s**. The apparently perfect term fractions are copied text, not evidence of transcription accuracy.

### Controlled downstream

[downstream.py](downstream.py) wraps the existing `rebuild_offline.py` and `eval_retrieval.py`; it rebuilds stage04/05 from downloaded transcripts with fixed stage01/02, frozen CPU source/config/backend, ASR correction **off for all systems**, and phase-2 query-time defaults. The same **24 speech/term/both questions** are used for all seven systems. Source exports are read-only; only copied transcript manifests are re-signed. Pho fallback spans receive deterministic IDs without text/time changes. Backend snapshot SHA-256: `0928d3749350e7165ed2c1cc64d1bd574395d720f18d9c3bdf16d0bfbd62baaa`.

{{DOWNSTREAM_TABLE}}

Large-v3 has the same ranks as medium on aggregate but loses context support on `cs_0001_q04`, `L25_V011_q11`, `v2YTz8ToEag_q01`, `y1CvB950VlU_q05`, `y1CvB950VlU_q07`. Changed selections are in [downstream_changes.json](../runs/gpu/analysis/downstream_changes.json). Context recall combines key-term matching with time support; it is not generated-answer correctness, and long slide intervals can inflate hits.

A shared **post-hoc numeric-format sensitivity** keeps evidence rankings and times fixed and maps five reviewed number spellings to digits (`một trăm`, `bảy mươi`, `ba mươi`, `mười hai`, `không phẩy ba sáu`). Recall changes: Pho **0.70833→0.83333**, Chunk **0.50000→0.62500**, Pho hints **0.54167→0.66667**; medium/large/turbo/medium hints stay **0.95833/0.75000/0.83333/0.66667**. This is diagnostic analysis, not a production normalization improvement. Pho's correctly rendered `một trăm triệu đồng` and `không phẩy ba sáu` can fail draft keys `100 triệu` and `0,36`. Questions derived from old outputs may favor their spelling. [numeric_sensitivity.json](../runs/gpu/analysis/numeric_sensitivity.json) retains per-question differences.

### Pairwise disagreement

Symmetric token edit distance within identical production windows, weighted by the larger token count. Lower disagreement is not proof of correctness. The English regression contributes to Chunk's larger differences. Full per-video/window distances are in [asr.json](../runs/gpu/analysis/asr.json).

{{DISAGREEMENT}}

### CTC timing contract

The adapter retains blank-separated repeats and uses the real **80ms emission clock**. ChunkFormer 1.2.2's native timestamp helper removes blanks before duplicate collapse, and pads native segment starts up to two frames. Adapter/native non-whitespace character counts: English 11354/11307, Bayes 17782/17723, history 10159/10150, economics 34445/34270, Python 8786/8723. These conventions explain small text/start differences; emission times are not forced word boundaries. Native English itself is garbled. [ctc_adapter_parity.json](../runs/gpu/analysis/ctc_adapter_parity.json) and `../runs/gpu/discovery/chunkformer_output_source.py` document the comparison. Current v1 did not save raw frame arrays; the future runner now does, but this run cannot be re-aligned without inference.

## E2 — OCR measurements and literal audit

The 20-image GT says it was **AI-transcribed and not human-checked**. It inconsistently includes branding despite its scope note. Scores below measure draft-reference agreement, not certified Vietnamese OCR accuracy. Scoring lowercases NFC text, normalizes punctuation/whitespace, computes Levenshtein CER and `1−WER` (which can be negative), and scores accented characters including đ via edit alignment, counting deletions wrong. Formula formatting still affects text scores. Qwen figure-description lines are excluded from literal OCR scoring.

Reference quality is constant across systems. The nine audit frames are scored separately for visible numeric/code/semantic anchors. Enhanced Qwen jointly changes content crop, pixels, prompt and validation; the separate contributions were not isolated. Crop coordinates are manually selected for the audit images; an automatic full-corpus crop detector was not evaluated.

{{OCR_TABLE}}

Qwen whole/enhanced torch peaks are **9049.40 / 9069.15 MiB**. Broad Qwen's four whole-frame failures take 119.76–142.89 seconds each; all enhanced predictions terminate normally. PP-OCR predicts all 29 images with boxes, no detected long text loops, but loses vowel marks and mixes column reading order.

| Audit frame | Broad/focused findings |
|---|---|
| `hVU5Mw19V_c seg_00000` | Both broad Qwen setups read denominators 330 instead of 370 for all three fractions. Tight Qwen4 reads `100/370`, `12Ω/370`, `150/370`; tight Qwen8 reads all three exact fractions. PP independently reads all three 370 denominators but one numerator is `12Q`. |
| `hVU5Mw19V_c seg_00012` | Broad Qwen reads 4,5kg and 46,35% instead of 1,5kg and 46,75%. Both tight models fix 1,5kg; 4B reads `46,75` with a degree symbol; 8B writes `√46,35°`. Neither has the correct percentage plus unit. PP recovers the two numeric values, with symbols/order still unreliable. |
| `L25_V011 seg_00006` | Enhanced crop restores `quyền bình đẳng` from inherited `quyền đăng ký`. |
| `tHE1mC_n0-E seg_00097` | Broad enhanced output restores four statements but changes statement c's magnetic-field direction into a force-direction assertion and BIℓ→BIL. Tight crops restore field direction; BIℓ and handwritten truth marks still fail. Crop includes only part of the following Câu4, so complete-slide coverage is not claimed. |
| `tHE1mC_n0-E seg_00201` | Enhanced output restores the oxygen recovery from CO2 topic; chemistry/numbers remain corrupt. A later methane question is genuinely present and is distinguished from the first paragraph. |
| `v2YTz8ToEag seg_00040` | Enhanced crop removes the 324-repeat `Câu hỏi` loop. |
| `_eDGTHhyQ7M seg_00006` | Enhanced crop removes a 192-repeat phrase and recovers a table, retaining semantic errors. |
| `y1CvB950VlU seg_00005` | Broad enhanced output removes invented a5, but loses triple quotes/a4. Tight models restore a3's paired triple double quotes. 4B omits a4; 8B turns visible triple single quotes into doubles. Neither is fully literal code. |
| `y1CvB950VlU seg_00020` | Enhanced crop retains a2/indentation but merges visible IDE breadcrumb `for i in a2` into OCR text. Direct image review confirms the breadcrumb exists; it is not an invented line of source code. |

Focused comparison uses **five identical tight crops**, with references checked visually before predictions; prompts contain no expected literals:

| Model | Five numeric value anchors | Seconds/crop | Torch peak MiB | Limits |
|---|---|---|---|---|
| Qwen4B FP16 | 4/5 | 3.4896 | 8753.69 | `12Ω/370`, percentage unit wrong, a4 missing |
| Qwen8B NF4 | 4/5 | 5.7805 | 6383.85 | Wrong percentage value/unit, a4 quote kind wrong |

The percentage anchor scores only its value; unit correctness is explicitly reviewed separately. Both broad models have 0/5 value anchors; tight rereads have 4/5. This is a targeted audit gain, not a full-corpus numeric accuracy estimate. [focus.json](../runs/gpu/analysis/focus.json) and `../runs/gpu/ocr-focus-v2/results/` retain crops/generations. M3AV formula evaluation was omitted to prioritize E1 and the failing Paddle compatibility investigation within the budget.

{{PADDLE_FINAL_DETAILS}}

## E3 — fixed-context answer models

Thirty fixed questions, three per video across all ten, were selected from `questions_dev.json`, covering technical/speech, numeric/slide and code/visual/both types. Contexts are unchanged from `phase2-final/retrieval.json`; context source hashes are in build provenance. References are saved for offline judging and **never passed in inference prompts**. Ollama uses temperature 0, context 8192 and maximum 384 tokens. Vintern receives the same text plus the first retrieved slide, using four 448×448 tiles and a thumbnail, FP16, greedy decoding and penalty 1.05. This is a multimodal-system comparison, not a pure model-size ablation.

All 90 original responses were manually compared textually with the **draft, auto-written references**. Rubric: 2 = correct target answer without material contradiction; 1 = partial/ambiguous/unfinished; 0 = wrong, abstained or absent target answer. Ratings and notes are in [llm_ratings.json](../runs/gpu/analysis/llm_ratings.json); no certified human accuracy claim is made. Some abstentions follow missing context, so these scores include evidence limitations.

{{LLM_TABLE}}

The original Qwen3 run is **not a valid non-thinking comparison**. `think:false` failed to suppress visible English reasoning; all 30 responses hit `done_reason=length`. Nineteen contain partial target facts in unfinished reasoning, earning at most partial under the rubric. Qwen2.5 has one length stop and 29 normal stops. The explicit `/no_think` retry uses identical questions, contexts, model manifest and token budget:

{{LLM_RETRY_TABLE}}

The `/no_think` retry still exposes reasoning: the pinned Ollama model template unconditionally inserts `<think>` at the assistant prefix. A final short retry bypasses that template with `raw=true` and explicit ChatML ending in `<think>\n\n</think>\n\n`, the empty closed thinking block. It retains the same pinned weights, text contexts and generation options; its prompt format/hash are recorded per answer.

{{LLM_RAW_TABLE}}

{{LLM_FINAL_NOTE}}

Latency is measured generation request time, not download/setup. Ollama means include the first model load: Qwen2.5 first request 63.192s, original Qwen3 26.532s. Vintern's model loading/preprocessing is excluded from its per-response generation timer, so cross-runtime latency is not fully comparable.

Concrete original-run examples: Qwen2.5 answers the corrupted Bayes denominator **370**, while Vintern only describes the image. Vintern correctly reads English MSC labels **p,q,r**, where Qwen2.5 abstains on text evidence. Vintern invents a wrong Bayes formula after naming it correctly, confuses burn proportions with complications, and answers an unrelated nuclear-binding question for the melting/vaporization questions. Qwen2.5 substitutes the PPF slope relation for the requested general opportunity-cost definition, writes `Xô Việt` instead of `Xô viết`, gives a partially Chinese answer for saline concentration, and permits ordinary double quotes in a multiline-string answer. These remain concrete limitations even where target facts are otherwise recovered.

## Failures and recovery

Every failed run is charged to the budget and retains raw tracebacks:

| Attempt | Error/result and recovery |
|---|---|
| Initial OCR native child | Transformers 5.22.1 conflicts with hub 1.4.1; PP installation replaces NCCL and torch fails with undefined `ncclCommShrink`. |
| OCR native v1 | TF5.22.1/hub2.4.0 loader fails: `PaddleOCRVisionConfig ... no attribute rope_parameters`. PP defaults to v6 against supplied v5 config. |
| OCR native v2 | TF5.0.0/hub1.4.1 fails: `PaddleOCRVLConfig ... no attribute text_config`. Explicit PP-v5 model names and restored NCCL succeed. No VL accuracy exists from these failed attempts. |
| OCR remote v1 | Pinned remote AutoModel with TF4.57.1 loads, but all 29 forwards fail: `create_causal_mask() got an unexpected keyword argument inputs_embeds`; API expects `input_embeds`. Downgrading to 4.55.4 would retain the singular API and was not attempted. |
| OCR focus v1 | Missing crops directory aborts 4B; retaining 4B causes 8B OOM. Focus v2 creates the directory/releases 4B, and 8B NF4 fits. |
| Medium glossary v1 | Soft budget stops after three complete videos. A separate kernel resumes the other two; full-five outputs and provenance are curated before evaluation. |
| Original E3 | Qwen3 `think:false` does not suppress visible reasoning; all outputs hit the token limit. Explicit `/no_think` retry is recorded separately. |
| E3 /no_think retry | Model template still opens a thinking block. Raw ChatML with a preclosed empty thinking block is tested separately. |
| Local CPU rebuilds | An inherited phase3 configuration default is incompatible with the frozen sources. Wrapper freezes the successful config and supplies schema IDs for copied Pho fallback records. Final controlled runs share sources/config/backend. Changed-transcript index parity against old exports is expected to fail; ASR baseline parity itself is exact. |

The final Paddle adapter changes only `inputs_embeds`→`input_embeds` for eligible imported mask functions, forwarding the identical tensor to the unchanged mask calculation. In this run only `create_causal_mask` was present and adapted. Its remote module hash and mapping are saved in `paddlevl_api_adapter.json`. This is an experimental compatibility adapter, not an unmodified official pipeline.

## Kernel ledger and GPU budget

Every slug below has owner prefix **`thnhtrungnguynmif/`** and is a private Kaggle code reference, e.g. `https://www.kaggle.com/code/thnhtrungnguynmif/<slug>`. Separate versions/failed attempts are counted individually:

{{LEDGER_TABLE}}

Total: **{{GPU_HOURS}} allocated T4-hours**, including failed attempts, downloads/installation/model loading while scripts ran and the unused second T4. These are measured script elapsed seconds ×2/3600; any missing session is conservatively charged its hard timeout. They are not a Kaggle quota API reading; scheduling/container-start overhead outside the script is unmeasured. No further kernels are queued/running. The launch guard capped simultaneous kernels at two and cumulative measured/reserved device hours below eight.

## Reproduction and remaining uncertainty

[build_kernels.py](build_kernels.py) embeds pinned historical helpers in private kernel directories; each has metadata/provenance. `submitted_versions/vN/` archives the latest Kaggle-returned source, with hashes in [submitted_versions.json](../runs/gpu/analysis/submitted_versions.json). Historical version-suffixed pulls returned HTTP 403 with CLI 2.2.4; the two older v1 sources were not recovered, although their output logs and runtime/model pins remain. [launch_private.py](launch_private.py) checks privacy/concurrency/budget; [watch_runs.py](watch_runs.py) polls/downloads. [analyze.py](analyze.py), [summarize_results.py](summarize_results.py), [sensitivity_numeric.py](sensitivity_numeric.py), [focus_analysis.py](focus_analysis.py), [llm_analysis.py](llm_analysis.py) and [build_final_report.py](build_final_report.py) produce the measured tables. CPU source/config snapshots are under `cpu_snapshot/`. Retries reuse the original frozen E3 payload/hash if the concurrently edited phase2 report moves or changes.

```powershell
python kaggle/lecture_rag/gpu_exp/analyze.py ocr --root kaggle/lecture_rag/runs/gpu --output kaggle/lecture_rag/runs/gpu/analysis/ocr.json
python kaggle/lecture_rag/gpu_exp/summarize_results.py
python kaggle/lecture_rag/gpu_exp/sensitivity_numeric.py --output kaggle/lecture_rag/runs/gpu/analysis/numeric_sensitivity.json
python kaggle/lecture_rag/gpu_exp/build_final_report.py
```

ASR references and acoustic listening, human-reviewed OCR GT/questions, automatic crop selection, independent prompt-vs-hotwords-vs-fallback ablations, production anti-loop retry effectiveness, literal code/formula/units, CTC forced alignment, and full-corpus OCR/downstream effects remain uncertain. The CPU cleaning work being edited concurrently was held fixed in these comparisons; its final interaction with new models was not tested. Expected gains above are observed sample effects and decode cost estimates, not promises for the next full run.
