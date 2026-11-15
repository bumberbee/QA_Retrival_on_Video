# Verified tools and models for Multimodal Lecture RAG — v3

Research snapshot: **1 November 2026**. Target: IUH bachelor thesis; mostly Vietnamese lecture videos; extraction on Kaggle **two separate T4 GPUs, 16 GB each**; Windows backend using CPU ONNX and Ollama. **Direct HTTP access worked.** This report verifies existence, metadata, documentation and advertised inference routes. It does **not** claim that new models were run on our videos: no weights, repositories or packages were downloaded or installed.

GitHub popularity is `stargazers_count`; last update is `pushed_at`, not necessarily a release date. Hugging Face popularity is the API's `downloads` counter (normally a rolling recent-download measure, not lifetime users); update is `lastModified`. **R** means repository, **M** means model. Counts are a snapshot and do not establish quality. All verified repositories are unarchived except **stable-ts**, which the API marks archived. Model totals come from HF `safetensors.total` where available; other sizes are explicitly attributed to project documentation. Unknown sizes remain unknown.

**Fit is an engineering assessment, not a measured peak-memory result.** FP16 weights alone require approximately `2 × parameters` bytes; actual memory also includes the vision encoder, activations, caches, runtime and batches. A nominal Q4 payload starts around `0.5 × parameters` bytes, with additional quantization metadata and runtime memory. T4 does not support native BF16 or FlashAttention 2: use FP16 and a supported eager/SDPA or llama.cpp path, small batches and bounded image/context sizes. Modern vLLM/FlashAttention examples are not proof of T4 compatibility. Two T4s do not automatically provide one 32 GB device; use independent workers or unload models between stages.

**Licensing is separate for code and weights.** NC means non-commercial; these weights can be evaluated for the thesis within their stated terms. Custom, missing or conflicting license declarations are flagged. “Try now” means an isolated, bounded pilot is justified; it does not authorize replacing production based on a model card. “Try later” means an adapter, dependency, license clarification or larger machine is needed. “Skip” concerns the proposed upgrade, not the existence of the project.

## Local baseline and experimental contract

Read locally: `docs/paper/upgrade-v3-research.md`, the five supplied `result-*.md` audits, `kaggle/lecture_rag/eval/{BASELINE.md,PHASE2.md,eval_retrieval.py,questions_dev.json}`. The older research plan's network limitation no longer applies to this new report.

**Use Phase 2 app defaults as the comparator.** The local PHASE2 report records Hit@1 **0.600**, Hit@3 **0.791**, MRR **0.682**, context recall **0.745** on 110 answerable questions. These are internal development results, not external model benchmarks. Context packing, OCR group deduplication, source-boost removal and Vietnamese lexical fixes have already changed; do not attribute those improvements to a new model.

The 120-question set has 110 answerable and 10 unanswerable questions and is explicitly **auto-written, awaiting human review**. Review images and listen to audio before treating gold answers as accuracy evidence. Preserve the lecturer's genuine inconsistencies, including the seed example's displayed type-II mass versus its calculation. The ten videos are:

`cs_0001`, `D_kuulxKuxc`, `hVU5Mw19V_c`, `L25_V008`, `L25_V011`, `L25_V037`, `tHE1mC_n0-E`, `v2YTz8ToEag`, `y1CvB950VlU`, `_eDGTHhyQ7M`.

The existing harness reports **Hit@1/3/5, MRR, context recall and quality lints**. Production returns three slides, so **Hit@5 equals Hit@3**; this is not a genuine five-result benchmark. Local context recall checks gold-time overlap and retained key terms; it is neither full answer correctness nor RAGAS context recall. Long board intervals can inflate time-overlap hits. WER/CER, literal numeric/code accuracy, boundary F1, timestamp MAE/p95, generated-answer accuracy and judge agreement are **additional manual/component measurements**, not metrics already implemented in this directory.

Keep source/question hashes, candidate counts, cleaning rules and context budgets fixed for paired comparisons. Save future pilot outputs in isolated experiment directories. `rebuild_offline.py` supports the pinned E5 configuration, not arbitrary models; the existing `--features` list does not contain PyVi or reranking. New OCR/ASR schemas, lexical tokenizers, encoders and rerankers need deliberate adapters. This report proposes them without changing source code.

## 1. Slide transitions, board states and readable keyframes

| Name / link | Type | License | Last update | Popularity | Vietnamese evidence | Size; T4 / CPU fit | ONNX / faster inference | Specific weakness addressed | Verdict and reason |
|---|---|---|---|---|---|---|---|---|---|
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | Library / CLI | BSD-3-Clause | R 2026-10-31 | 5,311 stars | Language-neutral image differences; no text understanding | No model weights; CPU yes; GPU unnecessary | OpenCV/PyAV; ContentDetector, AdaptiveDetector, ThresholdDetector | Camera cuts/fades; candidate boundaries around visual changes | try now — cheap CPU comparator; still needs board ROI and reveal-state logic |
| [TransNetV2](https://github.com/soCzech/TransNetV2) | Shot-boundary model / code | MIT | R 2023-12-04 | 1,071 stars | Language-neutral; no Vietnamese lecture evaluation documented | Exact parameter count not verified; low-resolution 27×48 inference makes one T4 plausible; CPU torch possible | PyTorch inference documented; no official ONNX artifact verified | Hard cuts and transitions missed by fixed frame-difference thresholds | try later — shot boundaries do not capture writing, erasure or code replacement within a shot |
| [video2slides](https://github.com/binh234/video2slides) | Slide-extraction app | MIT | R 2024-03-14 | 64 stars | Language-neutral extraction; no VN OCR | No neural weights; CPU yes | Frame_Diff / KNN / GMG; image hashes | Reusable slide-export UI and background-subtraction baseline | skip as primary upgrade — README uses the same class of frame differences and reports animation limitations |
| [SliTraNet](https://github.com/asindel/SliTraNet) | Lecture-transition research code | Unclear: API license null | R 2023-12-17 | 11 stars | Language-neutral slide ROI; no VN validation | Size/peak memory not documented in fetched README; T4 fit not established | PyTorch; pretrained-weight Google Drive link documented, its accessibility not checked | Slide-region-aware transition detection | skip direct integration — unclear license and externally hosted checkpoints; source is verified, runnable setup is not |
| [AccessMath ICFHR 2018](https://github.com/bhargavaurala/accessmath-icfhr2018) | Whiteboard extraction / summarization tool | GPL-3.0 | R 2019-08-22 | 23 stars | Handwritten visual components, not Vietnamese recognition | Classical stages CPU; detector requires external model; size/T4 fit unverified | Text detection, temporal stability, component grouping and video segmentation; no ONNX verified | Board writing/erasure chronology and stable visible content | try later — strong algorithm reference; old stack and Box-hosted detector/data prevent a verified turnkey pilot |
| [AccessMath Pose](https://github.com/adaniefei/AccessMath_Pose) | Speaker-action / board extraction code | CC-BY-SA-4.0 repository; dependency terms separate | R 2021-05-29 | 6 stars | Language-neutral speaker pose/action | OpenPose plus compiled C helper; model memory not documented; stock Kaggle setup not verified | OpenPose; Python 3.6-era setup; no ONNX route verified | Select board frames when lecturer is writing or exposing content | skip turnkey reuse — extensive old dependencies and external artifacts; compare its action-aware selection idea |
| [SAM 2 / SAM 2.1 tiny](https://github.com/facebookresearch/sam2) | Prompted image/video segmentation | Apache-2.0 code and project checkpoints | R 2026-05-30 | 20,084 stars | Language-neutral; prompt lecturer using points/box | Tiny model size not independently enumerated here; one T4 plausible for bounded clips; CPU torch slow | Torch >=2.5.1; video propagation; no official ONNX deployment verified | Track presenter occlusion across frames | try later — requires initial prompts, clip memory control and T4-safe precision |
| [rembg + U²-Net human mask](https://github.com/danielgatis/rembg) | ONNX segmentation library / CLI | MIT wrapper; U²-Net source Apache-2.0; no separate weight grant established | R 2026-10-21 | 25,283 stars | Language-neutral; U²-Net README explicitly describes human segmentation | Compact CNN; exact ONNX bytes not fetched; documented CPU support; T4 optional | ONNX Runtime CPU/CUDA; select u2net_human_seg, not a cloud backend | Mask lecturer motion for board-change scoring; select less-occluded keyframes | try now — direct CPU path; mask errors and missed board pixels require human QA |

A presenter mask can exclude motion from change scoring and identify a clearer **real** frame. It cannot recover writing hidden behind a lecturer. Retain complementary frames and preserve each state's onset; do not inpaint missing evidence or combine later text into an earlier timestamp. PySceneDetect/TransNetV2 propose cuts; ROI additions **and deletions**, replacement-safe grouping and caption-strip text detection remain necessary.

## 2. OCR, layout, code, formulas and tables

| Name / link | Type | License | Last update | Popularity | Vietnamese evidence | Size; T4 / CPU fit | ONNX / faster inference | Specific weakness addressed | Verdict and reason |
|---|---|---|---|---|---|---|---|---|---|
| [PaddleOCR + Latin PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) / [weights](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec) | OCR library + detector/recognizer | Apache-2.0 | R 2026-09-16; M 2025-10-16 | 92,065 stars; 33,426 HF downloads | Official PP-OCRv5 language table explicitly lists Vietnamese / vi and Latin recognizer | Lightweight recognizer; exact size not published in fetched metadata; CPU yes; one T4 yes | Paddle CPU/GPU; polygon boxes/confidence; official HF files are Paddle, not ONNX | Literal numbers/diacritics, text boxes, caption fallback and line geometry for indentation | try now — strongest non-generative Vietnamese baseline; reconstruct code whitespace from boxes |
| [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) / [1.5 GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5-GGUF) | Document VLM / structured parser | Apache-2.0 | R 2026-09-16; M 2026-08-08 | 92,065 stars; 85,301 HF downloads | Multilingual family claim; fetched VL cards do not explicitly enumerate Vietnamese; accuracy unproven | 0.959B total (~1.92 GB FP16 weights alone); one T4 likely at batch 1; CPU pipeline documented | Paddle / Transformers; official 1.5 GGUF model + mmproj exist; no official ONNX VLM files | Reading order, typed blocks, tables and formulas absent from whole-slide Qwen OCR | try now — small structured parser pilot; verify Vietnamese first; 1.5 GGUF is not proof of a 1.6 GGUF export |
| [dots.ocr](https://huggingface.co/dots-studio/dots.ocr) / [code](https://github.com/studio-dots-ai/dots.ocr) | Layout/OCR VLM | MIT | R 2026-03-24; M 2025-10-31 | 9,184 stars; 447,614 HF downloads | Multilingual/low-resource claims; no explicit VN row found in fetched card | 3.039B total incl. vision, despite 1.7B LLM description (~6.08 GB FP16 weights); T4 conditional | Transformers and vLLM; default examples use BF16/FlashAttention 2; no ONNX files | Layout boxes, reading order and table/code-region parsing | try later — test FP16/eager compatibility and Vietnamese crops before full slides |
| [olmOCR-2-7B-1025](https://huggingface.co/allenai/olmOCR-2-7B-1025) / [olmocr](https://github.com/allenai/olmocr) | PDF linearization toolkit / VLM | Apache-2.0 | R 2026-03-25; M 2025-10-22 | 19,813 stars; 78,914 HF downloads | Card language en; no Vietnamese validation | 8.292B total (~16.58 GB BF16 weights alone); standard BF16 unsuitable for a comfortable T4 run; CPU client can call remote server | vLLM; FP8 sibling referenced, but T4 lacks native FP8; no ONNX in inspected model | Rotation/retry handling, reading order and degeneration checks | skip VN replacement — English focus and hardware mismatch; borrow retry/validation ideas |
| [MinerU](https://github.com/opendatalab/MinerU) / [MinerU2.5 1.2B](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) | Document app / parser + VLM | Current code: Apache-2.0 with custom attribution/commercial thresholds; inspected 2.5 weights: AGPL-3.0 | R 2026-10-31; M 2026-04-09 | 82,937 stars; 7,979 HF downloads | Inspected VLM card zh/en; Vietnamese accuracy not established | 1.156B VLM (~2.31 GB FP16 weights); T4 conditional; current toolkit documents CPU support | Current 4.x uses ONNX small models and llama.cpp/vLLM VLM routes; inspected model has no ONNX files | Coarse layout then native-resolution text/formula/table crops | try later — attractive crop architecture; version/model-license mismatch and VN evidence require a pinned pilot |
| [Docling](https://github.com/docling-project/docling) | Document conversion library | MIT code; each OCR/layout/VLM model has separate terms | R 2026-11-01 | 69,951 stars | OCR-engine dependent; README does not certify Vietnamese recognition | CPU/local operation; model stack and memory vary; T4 optional | PyPI exposes onnxruntime, rapidocr and tesserocr extras; full torch-free dependency closure not verified | Structured blocks, reading order and table representation | try later — useful container for output schema; configure a verified VN OCR engine rather than trust defaults |
| [Surya 2.5](https://github.com/datalab-to/surya) / [weights](https://huggingface.co/datalab-to/surya-ocr-2.5) | OCR/layout/table library + VLM | Apache-2.0 code; modified AI Pubs OpenRAIL-M weights, commercial restrictions | R 2026-10-29; M 2026-10-29 | 21,566 stars; 9,400 HF downloads | Model card explicitly lists vi; its own internal table reports Surya 2.5 63.6% pass rate, not VN word accuracy | 689M VLM (~1.38 GB FP16 weights) plus detector; T4 plausible with float16; CPU llama.cpp supported | vLLM or llama.cpp VLM; detector/wrapper require torch; settings explicitly warn T4 needs float16 | Bounding boxes, tables and reading order with a small multilingual model | try later — credible VN candidate; new server/dependency stack is less simple than PP-OCR |
| [Marker 2.x](https://github.com/datalab-to/marker) | Document-to-Markdown/JSON app | Apache-2.0 code; Surya modified OpenRAIL-M weights | R 2026-10-29 | 40,740 stars | Uses Surya OCR; inherits its documented vi support; no independent VN lecture benchmark | Current Surya VLM plus layout/repair models; CPU supported, wrapper still torch; T4 conditional | llama.cpp CPU / vLLM GPU; optional Ollama LLM repair; no complete ONNX pipeline verified | Structured output, equations and tables | try later — first assess underlying Surya; disable optional generative repair for literal-number evaluation |
| [VietOCR](https://github.com/pbcquoc/vietocr) | Vietnamese line-recognition library | Apache-2.0 repository; external weight-link terms not separately established | R 2025-01-19 | 827 stars | Vietnamese README and examples; pretrained text-line VGG Transformer / Seq2Seq models documented | Exact model totals not verified; cropped-line T4/CPU inference plausible; torch is needed separately | VGG/Seq2Seq or Transformer; no official ONNX artifact verified; needs a detector | Vietnamese crop-level literal reading and a second-engine disagreement check | try later — recognition only; old pinned dependencies and external checkpoint links need setup verification |
| [pix2tex / LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR) | Formula crop recognizer / tool | MIT | R 2025-01-18 | 16,601 stars | Math symbols are language-neutral; Vietnamese prose not supported as an OCR replacement | Exact parameter total not verified; CPU torch and one T4 plausible for isolated crops | PyTorch; no official ONNX artifact verified | LaTeX formulas, subscripts and fraction structure | try later — targeted printed formulas; do not send an entire board or expect handwriting guarantees |
| [UniMERNet](https://github.com/opendatalab/UniMERNet) | Mathematical-expression model / library | Apache-2.0 | R 2025-09-28 | 506 stars | Formula-specific; no VN prose evidence | README lists a 1.3 GB base-model repository download, not VRAM; one T4 likely for crops; CPU torch slower | PyTorch; pinned Transformers 4.42.4 in PyPI; no ONNX artifact verified | Printed/handwritten expression recognition rather than free-form VLM transcription | try later — formula-only comparator after visible crop labels are ready |
| [GOT-OCR2_0](https://huggingface.co/stepfun-ai/GOT-OCR2_0) | OCR VLM | Apache-2.0 | M 2025-02-04 | 172,815 HF downloads | Multilingual card without explicit VN result; prior local rejection supplied by user | 0.716B total; one T4 likely; no torch-free CPU package verified | Transformers custom code; no ONNX files in inspected repo | Unified OCR output | skip — already rejected locally for Vietnamese words; no new failure hypothesis justifies repeating it |

The PP-OCRv5 Latin model card has an apparent copied sentence saying it recognizes Korean and lists only en/zh metadata. The **official algorithm language table**, fetched separately, explicitly assigns Vietnamese to `latin_PP-OCRv5_mobile_rec`; that is the evidence used here. Its table's **46.8 is an improvement percentage, not a model size**.

PaddleOCR-VL original, 1.5 and 1.6 were all verified: respectively **9,702 / 28,127 / 85,301** HF downloads; original and 1.6 modified **2026-08-08**, 1.5 **2026-08-11**. Each inspected total is 958,588,736 parameters. Prefer a pinned 1.6 feasibility pilot; the official **1.5 GGUF** alternative has **15,080** downloads and update **2026-05-29**. No model-card benchmark is treated as Vietnamese lecture accuracy.

Preserve raw text, boxes and confidence. Layout output alone does not preserve Python indentation: infer lines and indentation from x/y coordinates, retain uncertainty and inspect actual code. Use a formula engine only on a formula crop, and a table parser only on a table region. None of these tools repairs over-aggressive cleanup, missing gate admissions or content that never reached the keyframe.

## 3. ASR recognition, technical vocabulary and repetition

| Name / link | Type | License | Last update | Popularity | Vietnamese evidence | Size; T4 / CPU fit | ONNX / faster inference | Specific weakness addressed | Verdict and reason |
|---|---|---|---|---|---|---|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Inference library (already used) | MIT | R 2026-10-31 | 26,367 stars | Via chosen multilingual Whisper/PhoWhisper checkpoint | Model-dependent; documented CPU INT8 / GPU FP16 or INT8 | CTranslate2; runtime does not need torch; conversion uses Transformers/torch | Technical-term hints and previous-text repetition feedback | try now — ablate hotwords and condition_on_previous_text=False on existing medium before a global rerun |
| [PhoWhisper-large / small](https://huggingface.co/vinai/PhoWhisper-large) / [code](https://github.com/VinAIResearch/PhoWhisper) | Vietnamese-finetuned ASR models | BSD-3-Clause code/cards | R 2024-11-12; M 2024-11-12 | 264 stars; 11,568 HF downloads | Card explicitly Vietnamese; diverse accents; project README gives 1.55B large / 244M small | One T4 likely, preferably CT2 INT8/FP16, short batches; small is a CPU fallback | HF originals have no CT2/ONNX files; planned CTranslate2 conversion via Whisper architecture | Vietnamese recognition and terminology damaged by generic medium | try now — native Vietnamese evidence and compatible architecture; measure code-switched English separately |
| [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | Multilingual ASR model | HF card declares Apache-2.0 | M 2024-08-12 | 3,474,365 HF downloads | HF language list explicitly vi | 1.543B total; one T4 likely with CT2 INT8/FP16; CPU INT8 feasible but throughput unmeasured | faster-whisper/CTranslate2 conversion; no ONNX in original HF repo | General ASR accuracy on Vietnamese plus English lecture terms | try later — higher-cost control after native-VN pilots; no guaranteed gain over medium |
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | Distilled multilingual ASR model | MIT card | M 2024-10-04 | 5,677,585 HF downloads | HF language list explicitly vi | 0.809B total; one T4 likely; CPU CT2 INT8 feasible | faster-whisper/CTranslate2; not an English-only distil-whisper checkpoint | Faster multilingual decoding and an inexpensive medium replacement control | try now — practical control; quality and timestamps still need local tests |
| [ChunkFormer CTC large vie](https://huggingface.co/khanhld/chunkformer-ctc-large-vie) / [code](https://github.com/khanld/chunkformer) | Vietnamese long-form ASR model / toolkit | Code CC-BY-4.0; weights CC-BY-NC-4.0 | R 2026-06-09; M 2025-10-08 | 87 stars; 478 HF downloads | Card explicitly Vietnamese; 110M parameters per card | 110M; one T4 very plausible; ONNX CPU route documented, actual throughput unmeasured | Current source exports encoder + CTC/RNN-T graphs; numpy/onnxruntime feature-input host; no prebuilt ONNX in inspected CTC model | Vietnamese long-form ASR; test a CTC alternative to autoregressive repetition loops | try now — particularly promising; source ONNX features postdate PyPI 1.2.2; CPU fbank adapter needed |
| [VietASR](https://github.com/dangvansam/viet-asr) | Vietnamese ASR library / small model | Apache-2.0 repository | R 2026-10-06 | 180 stars | README explicitly Vietnamese; 13M model per README | 13M; CPU/T4 fit plausible; no current lecture accuracy evidence | Older torch/KenLM setup; main README references a separate v2.0 branch; no ONNX verified | Small Vietnamese recognizer for resource-constrained use | skip first round — old setup and limited evidence for lecture/technical vocabulary |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Offline speech inference toolkit | Apache-2.0 toolkit; checkpoint terms vary | R 2026-10-29 | 15,658 stars | Multilingual Whisper backend can accept vi; supported hotword model must be checked separately | Model-dependent; documented CPU/CUDA ONNX; small Whisper models fit CPU/T4 | Native ONNX Runtime, Whisper/CTC/transducer APIs; hotwords_file documented for modified-beam transducers | Torch-free deployment, contextual biasing and current Whisper word-timing API | try later — useful runtime; do not assume transducer hotwords apply to Whisper or an arbitrary VN CTC checkpoint |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | C/C++ Whisper inference tool | MIT code; checkpoint terms separate | R 2026-10-30 | 54,843 stars | Multilingual Whisper checkpoints; choose vi-capable, not .en weights | Model-dependent; CPU quantized / CUDA / Vulkan supported | GGML quantization; OpenVINO and native backends; no Python torch required | CPU transcription deployment and streaming/offline tools | skip redundant migration now — CTranslate2 already provides a torch-free path; retain as portability comparator |

**Hotwords are hints, not a guaranteed decoding constraint.** The fetched faster-whisper implementation documents that hotwords have no effect when `prefix` is supplied. It also explicitly says disabling previous-text conditioning can reduce repetition loops/timestamp desynchronization. Build a short glossary from reviewed slide text; compare hints on/off and record false insertions. Avoid unconstrained post-ASR replacements.

For PhoWhisper-small the verified HF metadata is **8,535 downloads**, update **2024-11-12**. Its card and large's card declare BSD-3-Clause; conversion must retain the source terms and pin the source revision. ChunkFormer's ONNX documentation says greedy CTC/RNN-T inference is supported, while beam/attention rescoring remains PyTorch-only. `transcribe_file()` still needs torch/torchaudio; **precomputed-feature** `transcribe()`/`push_features()` is the documented numpy/onnxruntime route. Do not infer a torch-free installation from the package name.

## 4. Word timestamps and slide–speech alignment

| Name / link | Type | License | Last update | Popularity | Vietnamese evidence | Size; T4 / CPU fit | ONNX / faster inference | Specific weakness addressed | Verdict and reason |
|---|---|---|---|---|---|---|---|---|---|
| [WhisperX + default vi aligner](https://github.com/m-bain/whisperX) / [vi weights](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vi-vlsp2020) | Forced-alignment tool + CTC model | BSD-2-Clause WhisperX; aligner CC-BY-NC-4.0 | R 2026-10-22; M 2023-02-21 | 24,992 stars; 1,223,022 HF downloads | Fetched alignment.py maps vi to wav2vec2-base-vi-vlsp2020; card explicitly VN | Base wav2vec2 ~95M per project comparison; T4 plausible after unloading ASR; CPU torch allowed for extraction, not app | faster-whisper ASR plus torch CTC alignment; no original-model ONNX files | Bad word times, long single-word durations, boundary ownership errors | try now — align existing text first; missing-dictionary numbers/code words must be reported |
| [wav2vec2-base-vietnamese-250h](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h) | Alternative Vietnamese CTC aligner | Card CC-BY-NC-4.0; repo includes a CC-BY-NC-SA-4.0-named file: conflict flagged | M 2021-11-04 | 48,955 HF downloads | Card explicitly Vietnamese; 16 kHz speech | 95M per fetched ChunkFormer comparison table; CPU torch / one T4 plausible | Transformers CTC; no ONNX file verified | Alternative emissions when default Vietnamese forced alignment fails | try later — older alternative, conflicting license markers; prefer the verified current WhisperX default |
| [stable-ts](https://github.com/jianfch/stable-ts) | Timestamp stabilization / forced-alignment tool | MIT | R 2026-05-30 | 2,281 stars | Inherits multilingual Whisper; no independent VN alignment result documented | Whisper-dependent; CPU/GPU torch; not torch-free backend | VAD-based timestamp adjustment, alignment/regrouping; PyPI still available | Word-duration/timestamp outliers and punctuation regrouping | skip new dependency — API marks archived; WhisperX has an explicit current Vietnamese mapping |

Forced alignment places a supplied transcript against audio; it does not fix a wrong recognized word, reconstruct a missed board state or prove when a revealed bullet first appeared. Retain raw ASR and corrected-view provenance, count unaligned tokens, and keep sentence evidence across slide boundaries without moving visual facts earlier. Optional diarization adds its own model/license/auth requirements and is unnecessary for the first single-lecturer pilot.

## 5. Vietnamese lexical processing

| Name / link | Type | License | Last update | Popularity | Vietnamese evidence | Size; T4 / CPU fit | ONNX / faster inference | Specific weakness addressed | Verdict and reason |
|---|---|---|---|---|---|---|---|---|---|
| [PyVi](https://github.com/trungtv/pyvi) | CRF Vietnamese tokenizer / accent utilities | MIT | R 2024-09-26 | 279 stars | README demonstrates ViTokenizer and ViUtils.add_accents/remove_accents on Vietnamese | Small CRF resources; exact aggregate size not verified; CPU yes; no GPU | scikit-learn / sklearn-crfsuite; no torch / ONNX requirement | BM25 multi-syllable words such as chi phí cơ hội; optional query aliases for unaccented text | try now segmentation — test against Phase 2 syllable bigrams; accent restoration only as a separate query-alias ablation |
| [Underthesea](https://github.com/undertheseanlp/underthesea) | Vietnamese NLP library | Apache-2.0 | R 2026-11-01 | 1,814 stars | README explicitly documents word_tokenize/sent_tokenize and Vietnamese NLP | Base NLP CPU; exact resource size not verified; native underthesea_core wheel availability must be checked | Base PyPI dependencies have no torch; deep extra adds torch/Transformers | Word and sentence segmentation; sentence-aware chunk boundaries | try later — alternate tokenizer if PyVi fails technical terms; restrict installation to NLP base |
| [VnCoreNLP / RDRSegmenter](https://github.com/vncorenlp/VnCoreNLP) | Java Vietnamese NLP toolkit | GPL-3.0-or-later from LICENSE.md | R 2023-02-12 | 679 stars | README explicitly Vietnamese word segmentation | CPU Java; JAR + segmentation resources; exact deployment bytes not checked | Java API/CLI; Python bridge possible; no torch or ONNX needed | Multi-syllable Vietnamese word boundaries and an external segmentation comparator | try later — credible baseline but JVM and copyleft packaging cost exceed a small Python pilot |
| [vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords) | Plain-text stopword resource | MIT text in README; GitHub API license null | R 2022-07-25 | 200 stars | Vietnamese list explicitly documented | Plain text; CPU only; no model | Copy vetted list; no npm install needed | Consistent lexical normalization and reviewed low-information terms | try later curated subset — Phase 2 already fixes accent-before-stopword handling; full lists can delete physics terms/negation |

Preserve accented original evidence and exact numbers/code identifiers. Segment **before** aggressive accent folding, and keep syllable fallback so domain words and code-switched English remain retrievable. Do not automatically overwrite ASR/OCR with restored accents: ambiguity such as `cung/cứng` is exactly why query-side aliases are safer.

**Punctuation lead excluded:** `justtuananh/VietnamesePunctuation` exists (44 downloads, updated 2022-10-01), but the fetched metadata has no license/language declaration and its card supplies insufficient usable documentation. It is not a verified recommendation. Whisper-family punctuation is available as baseline output; a separate Vietnamese restoration model still needs a well-documented checkpoint and an exact word-ID mapping.

## 6. Hybrid retrieval, reranking and CPU embeddings

| Name / link | Type | License | Last update | Popularity | Vietnamese evidence | Size; T4 / CPU fit | ONNX / faster inference | Specific weakness addressed | Verdict and reason |
|---|---|---|---|---|---|---|---|---|---|
| [ViRanker](https://huggingface.co/namdp-ptit/ViRanker) | Vietnamese cross-encoder reranker | Apache-2.0 | M 2025-06-16 | 1,294 HF downloads | vi card; evaluation described on Vietnamese mMARCO reranking, not lecture QA | 567.8M; one T4 yes at bounded pair batches; CPU ONNX memory plausible, latency unknown | No official ONNX files; scalar XLMRobertaForSequenceClassification config verified; export/INT8 is a proposed adapter | RRF near misses and technical phrase relevance after candidate dedup | try now isolated pilot — score the existing candidate pool; prove ONNX parity before adding CPU runtime |
| [Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker) | Vietnamese cross-encoder reranker | Apache-2.0 | M 2025-05-06 | 7,383 HF downloads | vi card; Vietnamese fine-tuning; reported evaluation is legal-domain oriented | 567.8M; one T4 yes with bounded inputs; CPU ONNX conditional | No official ONNX files; one-label XLM-R config verified; export proposed | Vietnamese query–passage relevance beyond embedding similarity | try later — second reranker control; legal-domain results cannot establish lecture gains |
| [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Multilingual cross-encoder reranker | Apache-2.0 | M 2024-06-24 | 14,567,359 HF downloads | Multilingual card; no Vietnamese-specific result used | 567.8M; T4 yes for short pair batches; CPU export possible but not verified | No ONNX file in official inspected repo; Transformers/FlagEmbedding | Multilingual relevance scoring, including English control videos | try later — useful parent-model control; BGE-M3 embedding's local loss does not prove this reranker will lose |
| [Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding) | Vietnamese bi-encoder | Apache-2.0 | M 2025-08-25 | 13,717 HF downloads | vi card; Vietnamese fine-tuning; published evaluation focuses on legal retrieval | 567.8M, 1024 dimensions; one T4 yes; CPU ONNX exists but heavier than E5 | Official onnx/model.onnx plus external model.onnx_data; CLS pooling config verified | Vietnamese semantic matching when reviewed terms remain missed | try later — larger BGE-derived replacement, full re-embedding and correct CLS adapter required |
| [vietnamese-embedding](https://huggingface.co/dangvantuan/vietnamese-embedding) | PhoBERT sentence embedding | Apache-2.0 | M 2025-10-23 | 1,620 HF downloads | vi card; Vietnamese STS/NLI-oriented training | 135.0M, 768 dimensions; max_seq_length 512 in config; CPU/T4 plausible | Official onnx/model.onnx; card shows mean pooling; normalization/tokenizer contract still needs parity QA | Small Vietnamese semantic encoder alternative | try later — closer CPU size, but STS gains do not establish retrieval gains and English may regress |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | Current multilingual bi-encoder | MIT | M 2026-04-02 | 9,293,871 HF downloads | vi explicitly listed; current local baseline verified | 117.7M, 384 dimensions; deployed CPU ONNX; T4 unnecessary | Official ONNX fp32/O4/qint8 AVX512-VNNI variants; current revision already pinned | Stable multilingual retrieval baseline and CPU latency comparator | skip replacement without evidence — retain as control; quantized variants need compatible CPU and rank-parity checks |

The AITeamVN reranker card's “1024 output dimensionality” describes hidden width, **not a 1024-vector retrieval encoder**: fetched config has a one-label classification head. Both inspected Vietnamese rerankers output a scalar score per query/passage pair. Rerank **distinct evidence groups before final three-slide selection**, preserve speech timestamps and only then assemble context.

An ONNX file's presence verifies availability, not compatibility with our `OnnxEncoder`. Inspect input/output names, pooling, normalization, truncation and auxiliary files. In particular, AITeamVN Vietnamese_Embedding uses **CLS pooling**, while dangvantuan's card shows **mean pooling**. Reusing E5's mean-pooling adapter for the former would be wrong. Conversion can use torch on Kaggle; the deployed scorer must use tokenizer + ONNX Runtime without importing torch. No CPU throughput claim is made here.

## 7. Grounded answers and slide-image reading

| Name / link | Type | License | Last update | Popularity | Vietnamese evidence | Size; T4 / CPU fit | ONNX / faster inference | Specific weakness addressed | Verdict and reason |
|---|---|---|---|---|---|---|---|---|---|
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Small text LLM | Apache-2.0 | M 2025-07-26 | 8,424,710 HF downloads | Card claims 100+ languages; official Qwen3 blog explicitly lists Vietnamese | 4.022B (~8.04 GB FP16 weights); T4 likely at bounded context; CPU Q4 practical candidate, RAM/latency unmeasured | Card explicitly lists Ollama/llama.cpp support; no ONNX/GGUF in original inspected repo | Vietnamese instruction following and a smaller answer model for CPU | try now — bounded non-thinking answer pilot against current Qwen2.5; evaluate citations and numbers |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Text LLM | Apache-2.0 | M 2025-07-26 | 7,660,335 HF downloads | Same official Vietnamese language coverage | 8.191B (~16.38 GB FP16 weights before cache); use Q4 on T4; CPU quantization slower than 4B expected | Ollama/llama.cpp documented; no original ONNX artifact | More answer capacity if 4B loses technical reasoning | try later — test only after 4B; FP16 is not a comfortable 16 GB deployment |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Current text LLM | Apache-2.0 | M 2025-01-12 | 5,601,971 HF downloads | Card explicitly names Vietnamese | 7.616B (~15.23 GB FP16 weights); existing Ollama quantized CPU path; use Q4 on T4 | Current app Ollama; original HF has no ONNX | Grounded Vietnamese answers with existing context/timestamps | skip replacement-by-default — preserve this measured app baseline while evidence extraction improves |
| [Vintern-1B-v3.5](https://huggingface.co/5CD-AI/Vintern-1B-v3_5) / [CPU Q8](https://huggingface.co/ngxson/Vintern-1B-v3_5-GGUF) | Vietnamese image-reading VLM | MIT original and inspected GGUF card | M 2025-12-10 | 2,780 HF downloads | Card explicitly Vietnamese-specific visual/OCR training and English retention; explicitly says Colab T4 usable | 0.938B total; T4 use documented; CPU demo recommends >=8 cores / 8 GB RAM (project recommendation, not our measurement) | Transformers on GPU; verified Q8 GGUF + mmproj and llama-server/vicuna demo; no ONNX | Read retrieved slide diagrams, labels and numeric regions missing from text | try now — strongest small VN image reader; pin conversion revision and score literal visual facts |
| [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Current OCR VLM / image-conditioned answering | Apache-2.0 | M 2025-10-15 | 4,062,866 HF downloads | Card says 32 OCR languages; no explicit VN list in fetched card; local audits show VN use with errors | 4.438B (~8.88 GB FP16 weights); current T4 pipeline already runs it; CPU torch-free path not verified here | Transformers; no original ONNX file; bound image token count | Use source images at answer time for diagrams instead of relying only on corrupted OCR | try later selective QA — retain raw image evidence; current hallucinations forbid treating its descriptions as verified facts |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Small image/video VLM | Qwen Research License: non-commercial only, verified LICENSE | M 2025-04-06 | 1,541,027 HF downloads | Multilingual family; no explicit Vietnamese result used | 3.755B total (~7.51 GB FP16 weights); T4 plausible; CPU torch-free deployment not verified | Transformers; bounded pixels; no official ONNX file | Image-conditioned answers and sparse timestamp-labelled frame review | try later research comparator — custom NC license and no advantage over current VLM established |
| [SeaLLMs-v3-7B-Chat](https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat) | SEA-language chat LLM | Custom SeaLLMs terms; model-card link and inspected family LICENSE; restricted output use | M 2024-09-02 | 1,259 HF downloads | Chat card explicitly lists Vietnamese and SEA languages | 7.616B total; FP16 tight on T4, use supported quantization; CPU artifact not verified | Card demonstrates vLLM; no ONNX/GGUF in inspected chat model | SEA-language instruction following and abstention comparator | try later — older custom-license alternative; CPU setup less verified than Qwen3 |

**Excluded:** Vistral-7B-Chat's API metadata exists (AFL-3.0, 7.294B, 1,069 downloads, modified 2024-02-27), but its requested raw model card returned **HTTP 401**. Its documented Vietnamese behavior and deployment recipe could not be verified, so it is dropped from recommendations. The older SeaLLMs-v3-7B base model was also inspected (158 downloads, modified 2024-07-30); use the verified **Chat** variant for any future answering comparison.

The Vintern CPU conversion is **community-produced**, not the original HF safetensors. Its metadata reports **156 downloads**, update **2025-05-15**, and both Q8 language weights and Q8 mmproj files. The demo has **247 stars**, update **2025-05-16**. Those dates differ from the current original model's revision; establish conversion provenance rather than assume byte-equivalent models.

Always pass image IDs with original timestamps plus ranked speech spans. Ask for visible facts, distinguish OCR from interpretation, and validate citations against evidence that actually supports the answer. A VLM's video “timestamp” capability is not proof of correct lecture evidence onset.

## 8. Evaluation libraries and Vietnamese QA resources

| Name / link | Type | License | Last update | Popularity | Vietnamese evidence | Size; T4 / CPU fit | ONNX / faster inference | Specific weakness addressed | Verdict and reason |
|---|---|---|---|---|---|---|---|---|---|
| [RAGAS](https://github.com/vibrantlabsai/ragas) | RAG evaluation library | Apache-2.0 | R 2026-02-24 | 16,203 stars | Language-agnostic framework; Vietnamese judge/prompts not validated by README | CPU orchestration; judge/embedding model determines memory and cost | External/OpenAI-compatible judge adapter can use a local endpoint; optional extras can add torch | Faithfulness, answer relevance and context evaluation beyond retrieval hits | try later — calibrate against human VN labels; local small-model self-judging is not independent accuracy evidence |
| [DeepEval](https://github.com/confident-ai/deepeval) | LLM/RAG evaluation library | Apache-2.0 | R 2026-10-29 | 19,226 stars | Custom prompts/judge possible; no Vietnamese-specific benchmark evidence | CPU orchestration; chosen judge can run separately on CPU/T4 | README explicitly supports local/custom LLMs; deterministic and judge-based metrics | Answer hallucination, faithfulness and multimodal evaluation scaffolding | try later — use one framework with a reviewed rubric; avoid adding both it and RAGAS initially |
| [ARES](https://github.com/stanford-futuredata/ARES) | RAG evaluation / judge-training framework | Apache-2.0 | R 2025-03-28 | 740 stars | No Vietnamese-specific validation documented | README asks >~100 GB disk and GPU for local setup and documents a 16 GB-class OOM; not a turnkey T4 pilot | Local vLLM / API judges; optional fine-tuning and statistical calibration | Calibrated automated relevance/faithfulness evaluation | skip first round — training/resource/data effort is disproportionate to our ten-video draft evaluation |
| [UIT-ViQuAD2.0 (HF mirror)](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0) | Vietnamese QA dataset / benchmark | Unclear: inspected mirror card/API omit license | D 2024-04-13 | 906 HF dataset downloads | Card explicitly Vietnamese Wikipedia QA with unanswerable questions | Text/parquet dataset; CPU metrics; model memory independent | No inference engine; evaluate extractive answer EM/F1 and abstention in a separate adapter | External Vietnamese QA sanity checks and unanswerable behavior | try later — clarify original release terms; no images, lecture timestamps or lecture-domain gold |

Do not relabel our key-term/time-overlap metric as RAGAS context recall. Current answer-generation correctness is not implemented by `eval_retrieval.py`; save model outputs and use reviewed exact numeric/code checks plus human evidence/citation labels. Translate or author Vietnamese judge criteria, examine judge agreement on errors, and avoid using the answer generator as its only judge. Public QA sets are useful supplementary language checks, not substitutes for reviewed lecture questions.

Searches for ViRAG/ViMMRC did not yield a sufficiently verified, licensed timestamped Vietnamese lecture-RAG benchmark. No such benchmark is claimed here. Dataset mirrors with missing licenses are not promoted as openly licensed datasets.

## 9. Complete video / lecture RAG applications

| Name / link | Type | License | Last update | Popularity | Vietnamese evidence | Size; T4 / CPU fit | ONNX / faster inference | Specific weakness addressed | Verdict and reason |
|---|---|---|---|---|---|---|---|---|---|
| [VideoRAG / Vimo](https://github.com/HKUDS/VideoRAG) | Multimodal video-corpus RAG framework / desktop app | Project dual notice: architecture MIT; current integrated stack NC due to ImageBind | R 2026-03-18 | 3,503 stars | No Vietnamese lecture result in fetched README; inherits selected ASR/LLM capabilities | README advertises RTX 3090 24 GB capability; stock one-T4 fit not established; CPU end-to-end route not verified | Graph-based multimodal retrieval; MiniCPM/ImageBind integrations; no full ONNX CPU port verified | Reusable ingestion/retrieval architecture and corpus-level video QA comparator | try later — strongest complete-app reference, but adapt models/endpoint and measure memory; thesis NC use fits notice |
| [Video-RAG](https://github.com/Leon1207/Video-RAG-master) | Visually aligned long-video QA research pipeline | Unclear: repository API license null and root has no license file | R 2026-06-26 | 464 stars | README setup includes English NLP; no explicit Vietnamese result | LLaVA-Video-7B plus separate APE tools/service; peak VRAM not stated; stock T4 fit unknown | Multi-model retrieval/QA scripts, not a complete CPU Ollama app | Use auxiliary visual/audio/OCR retrieval for long-video QA | skip direct reuse — unclear license, multi-service research setup; useful method comparison only |
| [Video Chatbot](https://github.com/Mozeel-V/video-chatbot) | Lecture transcript chatbot app | MIT code; Gemini service/weights are external | R 2025-11-30 | 0 stars | Whisper can be multilingual; no VN validation or visual retrieval shown | Audio chunking documented for 8 GB host RAM; generation/embeddings use cloud Gemini; no fully local fit claim | Streamlit, transcript retrieval, Gemini 2.5 Flash | Basic chat/upload UI and transcript chunking | skip as multimodal upgrade — transcript-only/cloud-dependent; UI can be a reference |
| [Teaching Assistant 2.0 RAG](https://github.com/Vrushank264/Teaching-Assistant-2.0-RAG) | Lecture/video chatbot app | Unclear: API license null | R 2023-09-25 | 7 stars | No Vietnamese evidence; instructor-xl and WizardLM listed | GPU/AWS/Lambda services in README; exact model sizes/T4/CPU fit unverified | Open-model microservices; coarse 5–10 minute source references | Lecture upload/chat/source UX | skip — unclear reuse terms, old deployment and coarse timestamps; not an extraction/alignment upgrade |

These are verified **source repositories**, not verified drop-in replacements for our Windows ONNX/Ollama backend. VideoRAG and Video-RAG are different projects. No inspected complete app demonstrated all of Vietnamese slide/board extraction, reliable word alignment, CPU-only ONNX hybrid retrieval and precise supported answer timestamps together.

## 10. Ten most promising candidates: independent 1–2 day pilots

Each pilot below is **one to two developer days independently**; the full list is not a two-day schedule. Start with reviewed short clips/crops rather than rerunning all 279 minutes. Freeze a comparator run with current Phase 2 defaults. The shared existing command is:

```powershell
& backend/.venv/Scripts/python.exe kaggle/lecture_rag/eval/eval_retrieval.py --index-root PILOT_INDEX_ROOT --videos VIDEO_ID ... --name PILOT_NAME
```

Use `--index-dir PATH --videos VIDEO_ID` for a single exported index. `--questions` can select a reviewed question subset. New extraction must first be adapted into the stage-02/03/04 schema and isolated stage-05 index; new rerankers/tokenizers must be wired into the isolated backend used by the harness. There is **no existing command-line model/reranker switch**. Component and answer metrics below must be recorded separately because the harness does not yet compute them.

1. **PaddleOCR + Latin PP-OCRv5 — literal Vietnamese OCR and caption fallback.**  
   **Day 1:** on Windows CPU or one Kaggle T4, run the documented OCR pipeline with the Latin recognizer on a proposed 60–80 reviewed regions: `hVU5Mw19V_c` board at 464–519 s and 1490–1590 s; `D_kuulxKuxc` captions at 78–103 s; `y1CvB950VlU` code at 332–346 s; `_eDGTHhyQ7M` Solvay table at 2210–2355 s; a few English regions from `cs_0001`. Include rejected caption frames, not just admitted keyframes. Preserve polygons/confidences and derive line order/indentation from geometry.  
   **Day 2:** adapt only reviewed OCR outputs into isolated bundles, keeping cleaning/retrieval settings fixed; compare Qwen OCR on the same crops. **Existing metrics:** numeric/code/slide Hit@1/3, MRR, context recall; OCR-loop and empty-title lints. **Additional:** accented CER, exact numbers/identifiers, indentation/block retention, caption-readable recall, runtime. Continue only if it recovers literal evidence without introducing new numeric/code errors.

2. **PaddleOCR-VL-1.6 — structured blocks, tables and formulas.**  
   **Day 1:** one T4, FP16, batch 1, supported attention backend, bounded image resolution/output length. Use the same reviewed crops as pilot 1, especially `hVU5Mw19V_c` fractions/seed masses, `_eDGTHhyQ7M` table cells and `y1CvB950VlU` code. First prove the installed engine starts on T4; do not use BF16/FlashAttention defaults. Compare typed blocks/reading order against Qwen and PP-OCR. If GPU setup fails, assess documented CPU use; official 1.5 GGUF is a **separately pinned version**, not an equivalent 1.6 fallback.  
   **Day 2:** index literal blocks with provenance and real bounds; flag loops/truncation before caching. **Existing metrics:** numeric/code/visual Hit@3, MRR and context recall, raw-output loop count plus existing index lints. **Additional:** exact values, table-cell accuracy, formula edit distance, reading order, peak VRAM and seconds/region. Reject a whole-slide replacement if Vietnamese accents or numbers regress, even if JSON looks cleaner.

3. **rembg human segmentation — presenter-aware board states and keyframes.**  
   **Day 1:** Windows ONNX CPU, `u2net_human_seg`; sample short board clips from `hVU5Mw19V_c` around 460–530 s and 1490–1590 s, plus presenter-overlay clips from `tHE1mC_n0-E` around 1800–1840 s. Human-label masks on a small stratified frame set. Score changes inside the board ROI **outside the lecturer mask**, including removed content; pick the sharpest least-occluded real frame and retain a second complementary frame when necessary.  
   **Day 2:** reread only changed selections with the fixed OCR baseline and rebuild affected indices. **Existing metrics:** affected numeric/visual Hit@3 and context recall, duplicate-OCR lints. **Additional:** readable-board coverage, mask leakage, boundary precision/recall/F1 at stated tolerance, onset MAE/p95 and CPU frames/s. Do not accept better time-overlap hits obtained by making intervals longer.

4. **PhoWhisper-large via CTranslate2 — Vietnamese recognition with controlled hints.**  
   **Day 1:** convert a pinned original checkpoint on Kaggle, then run one T4 with small FP16 or INT8 batches. Compare medium on reviewed speech clips from `v2YTz8ToEag` 388–433 s and 1466–1516 s, `_eDGTHhyQ7M` 1670–1750 s, and `tHE1mC_n0-E` 625–683 s/2284–2296 s. Test a short slide-derived glossary on/off and previous-text conditioning on/off; retain raw recognition. Use `cs_0001` 137–166 s as an English/code-switch control, without assuming a VN fine-tune improves English.  
   **Day 2:** map segments/words into the current schema and isolate an index. **Existing metrics:** speech/term Hit@3, MRR, context recall, ASR-loop and bad-correction lints. **Additional:** reviewed syllable-WER/CER, technical-term and number accuracy, false glossary insertions, real-time factor and peak VRAM. Keep generic Whisper for English if the paired control regresses.

5. **ChunkFormer CTC large vie + current ONNX exporter — small long-form ASR.**  
   **Day 1:** one T4; run the same VN clips as pilot 4 and an approximately 15–20 minute contiguous section of `v2YTz8ToEag`. Use the current source's documented `tools/export_onnx.py --checkpoint khanhld/chunkformer-ctc-large-vie --out-dir ...` and parity utility. Pin source and NC model revisions; do not expect the old PyPI release to contain the June 2026 exporter.  
   **Day 2:** try the self-contained ONNX runtime on Windows with **precomputed fbank features** and numpy; verify the host does not import torch. Audio-to-fbank needs a CPU adapter; if not completed, report GPU recognition and export feasibility separately rather than claim deployment is done. **Existing metrics:** speech/term Hit@3, MRR/context recall, ASR-loop lints. **Additional:** WER/CER, glossary recall, ONNX/eager transcript agreement, CPU real-time factor and timing availability. A useful small model must retain evidence and provide a workable timing/alignment path, not merely short transcripts.

6. **WhisperX + Vietnamese wav2vec2 forced alignment — honest word timestamps.**  
   **Day 1:** Kaggle CPU/torch or one T4 after unloading ASR; align **existing reviewed transcript text** with the explicit default `nguyenvulebinh/wav2vec2-base-vi-vlsp2020`. Start with `y1CvB950VlU` 250–285 s, including the audited long “hãy” word, and `v2YTz8ToEag` 1466–1516 s. Use built-in English alignment separately for `cs_0001` 1250–1270 s. Skip diarization. Label a small sample of true word onsets by listening and count dictionary-uncovered numbers/English identifiers.  
   **Day 2:** remap aligned words while retaining IDs/text provenance and the original visual-state boundaries. **Existing metrics:** speech/term context recall, Hit@3/MRR as secondary checks; quality lints and unmapped-word integrity diagnostics. **Additional:** word onset/end MAE/p95, unaligned-token rate and crossing-boundary ownership errors. Require improved acoustic timing; retrieval overlap alone cannot validate it.

7. **PyVi segmentation — Vietnamese BM25 words without changing dense vectors.**  
   **Day 1:** CPU-only isolated adapter; segment accented query and passage text consistently, preserving numbers/code and syllable fallback. Compare against current Phase 2 lexical behavior on all ten videos, emphasizing `v2YTz8ToEag_q05/q06` (chi phí cơ hội), `_eDGTHhyQ7M_q09` (độ cứng), `tHE1mC_n0-E` physics terms and all `y1CvB950VlU` code questions. Keep E5 vectors, RRF weights, group dedup and candidate budget unchanged. Accent restoration is an optional separate query-alias test, not evidence rewriting.  
   **Day 2:** replay all 120 draft questions and manually review changed rankings. **Existing metrics:** per-video/type Hit@1/3, MRR and context recall; code/numeric regressions. **Additional:** segmentation errors, lexical branch recall, query latency and English-control ranking changes on `cs_0001`. Adopt only if segmentation helps beyond the already-fixed Phase 2 stopword/bigram baseline.

8. **ViRanker with a CPU ONNX adapter — rerank evidence, preserve timestamps.**  
   **Day 1:** on one T4, score query/passage pairs from the frozen fused candidate pool, after dedup and before final three-slide selection; use all ten videos, with detailed review of `v2YTz8ToEag`, `_eDGTHhyQ7M`, `tHE1mC_n0-E` and English `cs_0001`. Export the verified one-label XLM-R head to ONNX in the GPU environment. Compare scalar logits/rank order to eager output before attempting INT8. Official ONNX weights do not already exist.  
   **Day 2:** deploy only tokenizer + ONNX Runtime in an isolated Windows scorer, limit pair lengths/batches, then evaluate via a temporary harness/backend adapter. **Existing metrics:** Hit@1/3, MRR and context recall by question type; final diversity and code/numeric losses. **Additional:** ONNX/torch rank agreement and CPU p50/p95 added query latency. Keep it only if paired relevance gains survive the CPU version and do not crowd out useful speech.

9. **Vintern-1B-v3.5 — Vietnamese image-conditioned answer verification.**  
   **Day 1:** one T4 using the original model, or Windows llama.cpp using the verified Q8 model **and mmproj** with the documented vicuna template. Record both artifact revisions; check conversion provenance before comparing runtimes. Feed the retrieved real slide/board crop plus retained speech and timestamp labels for `hVU5Mw19V_c_q03/q07`, `_eDGTHhyQ7M_q05/q10`, `D_kuulxKuxc_q04`, `y1CvB950VlU_q11`, and `cs_0001_q07/q08`. Request literal visible facts and uncertainty.  
   **Day 2:** compare image-assisted answers with the current text-only Qwen2.5 answers on the same retrieval trace. **Existing metrics:** visual/slide/numeric context recall and Hit@3 document whether evidence arrived; they will usually remain unchanged for an answer-only branch. **Additional:** human visual-fact accuracy, exact numeric/label matches, supported-citation rate, invented objects/values and CPU/T4 latency. Pilot success means better supported visual answers, not more detailed descriptions.

10. **Qwen3-4B in non-thinking mode — smaller Vietnamese CPU answer model.**  
    **Day 1:** Windows Ollama/llama.cpp, supported quantized artifact (proposed Q4); verify the runtime's non-thinking control, freeze template/context and a bounded answer budget against existing `qwen2.5:7b`. Use reviewed numeric/code questions in `hVU5Mw19V_c`, `y1CvB950VlU`, `_eDGTHhyQ7M`, speech questions in `v2YTz8ToEag`, and English `cs_0001`. Include the ten `q12` unanswerable questions and the two history/international-organization controls `L25_V037`/`L25_V011`.  
    **Day 2:** replay the full draft set if runtime permits, retain answers/citation traces and manually score the reviewed subset. **Existing metrics:** rerun Hit@3/MRR/context recall to prove the retrieval comparator is unchanged; these metrics cannot choose the better LLM. **Additional:** answer correctness, exact numbers/code, abstention precision/recall, citation support, tokens/s and end-to-end p50/p95 latency. Keep 4B only if its CPU speed/memory benefit comes with acceptable paired factual quality.

Prioritize pilots 1, 3, 4 and 6 when time is tight: they recover literal visual/speech evidence and its timing. Reranking and answer-model swaps cannot recover a fact erased during extraction or cleanup. No new public benchmark gain is predicted.

## 11. Verification exclusions and packaging notes

Wrong-owner leads `khanhld/chunkformer` and `vinai/PhoWhisper` returned 404; the verified repositories are **khanld/chunkformer** and **VinAIResearch/PhoWhisper**. The supplied ChunkFormer model name redirects to **khanhld/chunkformer-ctc-large-vie**. dots.ocr redirects to **studio-dots-ai** on GitHub and **dots-studio** on HF. RAGAS redirects to **vibrantlabsai/ragas**. Repository README filename case was verified for PyVi, VnCoreNLP and Video-RAG rather than dropping their valid projects because README.md returned 404.

`jonatasgrosman/wav2vec2-large-xlsr-53-vietnamese` returned HTTP 401 from the model API and was dropped. Vistral's raw card returned 401 and was dropped. These are individual HTTP authorization/not-found responses, **not a network outage**; verified alternatives were available. Unsupported/incomplete search leads are not promoted into the tables.

Current package metadata was also fetched to check installability: **scenedetect 0.7.2**, **rembg 2.0.86**, **paddleocr 3.7.0**, **faster-whisper 1.2.1**, **whisperx 3.8.6**, **chunkformer 1.2.2**, **pyvi 0.1.1**, **underthesea 9.5.0**, **docling 2.145.0**, **mineru 4.1.3**, **marker-pdf 2.1.0**, **surya-ocr 0.23.0**, **vietocr 0.3.13**, **pix2tex 0.1.4**, **unimernet 0.2.3**, **ragas 0.4.3**, **deepeval 4.3.2** and archived stable-ts's **2.19.1**. Latest package and latest source capabilities can differ. WhisperX/Surya/Marker have substantial torch/runtime dependencies; Underthesea's base versus deep extras differ; Docling's ONNX extras do not by themselves prove its complete dependency stack is torch-free. Create isolated future pilot environments and measure imports instead of adding these packages to the app blindly.

## Sources actually fetched

The following lists **every distinct requested network URL** in this research run, including failed probes; repeated requests are listed once. Requests used GET/READ only. API HTTP redirects are retained under the requested URL; canonical repository/model identities are recorded above. A URL appearing inside a README was not fetched unless it also appears below.

### GitHub repository metadata and file inventories

- <https://api.github.com/repos/adaniefei/AccessMath_Pose>
- <https://api.github.com/repos/allenai/olmocr>
- <https://api.github.com/repos/asindel/SliTraNet>
- <https://api.github.com/repos/bhargavaurala/accessmath-icfhr2018>
- <https://api.github.com/repos/binh234/video2slides>
- <https://api.github.com/repos/Breakthrough/PySceneDetect>
- <https://api.github.com/repos/confident-ai/deepeval>
- <https://api.github.com/repos/DAMO-NLP-SG/SeaLLMs>
- <https://api.github.com/repos/dangvansam/viet-asr>
- <https://api.github.com/repos/danielgatis/rembg>
- <https://api.github.com/repos/datalab-to/marker>
- <https://api.github.com/repos/datalab-to/surya>
- <https://api.github.com/repos/docling-project/docling>
- <https://api.github.com/repos/explodinggradients/ragas>
- <https://api.github.com/repos/facebookresearch/sam2>
- <https://api.github.com/repos/ggml-org/whisper.cpp>
- <https://api.github.com/repos/HKUDS/VideoRAG>
- <https://api.github.com/repos/jianfch/stable-ts>
- <https://api.github.com/repos/k2-fsa/sherpa-onnx>
- <https://api.github.com/repos/khanhld/chunkformer> — HTTP 404
- <https://api.github.com/repos/khanld/chunkformer>
- <https://api.github.com/repos/Leon1207/Video-RAG-master>
- <https://api.github.com/repos/Leon1207/Video-RAG-master/contents>
- <https://api.github.com/repos/lukas-blecher/LaTeX-OCR>
- <https://api.github.com/repos/m-bain/whisperX>
- <https://api.github.com/repos/Mozeel-V/video-chatbot>
- <https://api.github.com/repos/ngxson/vintern-realtime-demo>
- <https://api.github.com/repos/opendatalab/MinerU>
- <https://api.github.com/repos/opendatalab/UniMERNet>
- <https://api.github.com/repos/PaddlePaddle/PaddleOCR>
- <https://api.github.com/repos/pbcquoc/vietocr>
- <https://api.github.com/repos/rednote-hilab/dots.ocr>
- <https://api.github.com/repos/SeaLLMs/SeaLLMs>
- <https://api.github.com/repos/soCzech/TransNetV2>
- <https://api.github.com/repos/stanford-futuredata/ARES>
- <https://api.github.com/repos/stopwords/vietnamese-stopwords>
- <https://api.github.com/repos/SYSTRAN/faster-whisper>
- <https://api.github.com/repos/trungtv/pyvi>
- <https://api.github.com/repos/trungtv/pyvi/contents>
- <https://api.github.com/repos/tuanio/vietocr>
- <https://api.github.com/repos/undertheseanlp/underthesea>
- <https://api.github.com/repos/vinai/PhoWhisper> — HTTP 404
- <https://api.github.com/repos/VinAIResearch/PhoWhisper>
- <https://api.github.com/repos/vncorenlp/VnCoreNLP>
- <https://api.github.com/repos/vncorenlp/VnCoreNLP/contents>
- <https://api.github.com/repos/Vrushank264/Teaching-Assistant-2.0-RAG>
- <https://api.github.com/repos/xuebinqin/U-2-Net>

### GitHub discovery queries

- <https://api.github.com/search/repositories?q=AccessMath&per_page=4>
- <https://api.github.com/search/repositories?q=lecture%20video%20chatbot&per_page=4>
- <https://api.github.com/search/repositories?q=SliTraNet&per_page=4>
- <https://api.github.com/search/repositories?q=Video-RAG&per_page=4>
- <https://api.github.com/search/repositories?q=video2slides&per_page=4>
- <https://api.github.com/search/repositories?q=VietASR&per_page=4>
- <https://api.github.com/search/repositories?q=whiteboard%20extraction%20lecture&per_page=4>

### GitHub README, code, configuration and license text

- <https://raw.githubusercontent.com/adaniefei/AccessMath_Pose/master/README.md>
- <https://raw.githubusercontent.com/allenai/olmocr/main/README.md>
- <https://raw.githubusercontent.com/asindel/SliTraNet/main/README.md>
- <https://raw.githubusercontent.com/bhargavaurala/accessmath-icfhr2018/master/README.md>
- <https://raw.githubusercontent.com/binh234/video2slides/main/README.md>
- <https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/README.md>
- <https://raw.githubusercontent.com/confident-ai/deepeval/main/README.md>
- <https://raw.githubusercontent.com/DAMO-NLP-SG/DAMO-SeaLLMs/main/README.md>
- <https://raw.githubusercontent.com/dangvansam/viet-asr/main/README.md>
- <https://raw.githubusercontent.com/danielgatis/rembg/main/README.md>
- <https://raw.githubusercontent.com/datalab-to/marker/master/README.md>
- <https://raw.githubusercontent.com/datalab-to/surya/master/README.md>
- <https://raw.githubusercontent.com/datalab-to/surya/master/surya/settings.py>
- <https://raw.githubusercontent.com/docling-project/docling/main/README.md>
- <https://raw.githubusercontent.com/facebookresearch/sam2/main/README.md>
- <https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/README.md>
- <https://raw.githubusercontent.com/HKUDS/VideoRAG/main/LICENSE>
- <https://raw.githubusercontent.com/HKUDS/VideoRAG/main/README.md>
- <https://raw.githubusercontent.com/jianfch/stable-ts/main/README.md>
- <https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/master/README.md>
- <https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/master/sherpa-onnx/python/sherpa_onnx/offline_recognizer.py>
- <https://raw.githubusercontent.com/khanld/chunkformer/main/examples/onnx/README.md>
- <https://raw.githubusercontent.com/khanld/chunkformer/main/README.md>
- <https://raw.githubusercontent.com/Leon1207/Video-RAG-master/main/readme.md>
- <https://raw.githubusercontent.com/Leon1207/Video-RAG-master/main/README.md> — HTTP 404
- <https://raw.githubusercontent.com/lukas-blecher/LaTeX-OCR/main/README.md>
- <https://raw.githubusercontent.com/m-bain/whisperX/main/README.md>
- <https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/alignment.py>
- <https://raw.githubusercontent.com/Mozeel-V/video-chatbot/main/README.md>
- <https://raw.githubusercontent.com/ngxson/vintern-realtime-demo/main/README.md>
- <https://raw.githubusercontent.com/opendatalab/MinerU/master/LICENSE.md>
- <https://raw.githubusercontent.com/opendatalab/MinerU/master/README.md>
- <https://raw.githubusercontent.com/opendatalab/UniMERNet/main/README.md>
- <https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.md>
- <https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.md>
- <https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/README.md>
- <https://raw.githubusercontent.com/pbcquoc/vietocr/master/README.md>
- <https://raw.githubusercontent.com/SeaLLMs/SeaLLMs/main/README.md>
- <https://raw.githubusercontent.com/soCzech/TransNetV2/master/inference-pytorch/README.md>
- <https://raw.githubusercontent.com/soCzech/TransNetV2/master/README.md>
- <https://raw.githubusercontent.com/stanford-futuredata/ARES/main/README.md>
- <https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/README.md>
- <https://raw.githubusercontent.com/studio-dots-ai/dots.ocr/master/README.md>
- <https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/faster_whisper/transcribe.py>
- <https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/README.md>
- <https://raw.githubusercontent.com/trungtv/pyvi/master/pyvi/ViDiac.py>
- <https://raw.githubusercontent.com/trungtv/pyvi/master/README.md> — HTTP 404
- <https://raw.githubusercontent.com/trungtv/pyvi/master/README.rst>
- <https://raw.githubusercontent.com/tuanio/vietocr/master/README.md>
- <https://raw.githubusercontent.com/undertheseanlp/underthesea/main/README.md>
- <https://raw.githubusercontent.com/vibrantlabsai/ragas/main/README.md>
- <https://raw.githubusercontent.com/VinAIResearch/PhoWhisper/main/README.md>
- <https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/LICENSE> — HTTP 404
- <https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/LICENSE.md>
- <https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/Readme.md>
- <https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/README.md> — HTTP 404
- <https://raw.githubusercontent.com/Vrushank264/Teaching-Assistant-2.0-RAG/main/README.md>
- <https://raw.githubusercontent.com/xuebinqin/U-2-Net/master/README.md>

### Hugging Face model/dataset metadata and discovery

- <https://huggingface.co/api/datasets?search=ViMMRC&limit=3>
- <https://huggingface.co/api/datasets?search=ViQuAD&limit=5>
- <https://huggingface.co/api/datasets?search=ViRAG&limit=5>
- <https://huggingface.co/api/datasets/taidng/UIT-ViQuAD2.0>
- <https://huggingface.co/api/models?search=vietnamese%20punctuation&sort=downloads&limit=5>
- <https://huggingface.co/api/models/5CD-AI/Vintern-1B-v3_5>
- <https://huggingface.co/api/models/AITeamVN/Vietnamese_Embedding>
- <https://huggingface.co/api/models/AITeamVN/Vietnamese_Reranker>
- <https://huggingface.co/api/models/allenai/olmOCR-2-7B-1025>
- <https://huggingface.co/api/models/BAAI/bge-reranker-v2-m3>
- <https://huggingface.co/api/models/dangvantuan/vietnamese-embedding>
- <https://huggingface.co/api/models/datalab-to/surya-ocr-2.5>
- <https://huggingface.co/api/models/intfloat/multilingual-e5-small>
- <https://huggingface.co/api/models/jonatasgrosman/wav2vec2-large-xlsr-53-vietnamese> — HTTP 401
- <https://huggingface.co/api/models/justtuananh/VietnamesePunctuation>
- <https://huggingface.co/api/models/khanhld/chunkformer-large-vie>
- <https://huggingface.co/api/models/namdp-ptit/ViRanker>
- <https://huggingface.co/api/models/nguyenvulebinh/wav2vec2-base-vi-vlsp2020>
- <https://huggingface.co/api/models/nguyenvulebinh/wav2vec2-base-vietnamese-250h>
- <https://huggingface.co/api/models/ngxson/Vintern-1B-v3_5-GGUF>
- <https://huggingface.co/api/models/openai/whisper-large-v3>
- <https://huggingface.co/api/models/openai/whisper-large-v3-turbo>
- <https://huggingface.co/api/models/OpenDataLab/MinerU2.5-2509-1.2B>
- <https://huggingface.co/api/models/PaddlePaddle/latin_PP-OCRv3_mobile_rec>
- <https://huggingface.co/api/models/PaddlePaddle/latin_PP-OCRv5_mobile_rec>
- <https://huggingface.co/api/models/PaddlePaddle/PaddleOCR-VL>
- <https://huggingface.co/api/models/PaddlePaddle/PaddleOCR-VL-1.5>
- <https://huggingface.co/api/models/PaddlePaddle/PaddleOCR-VL-1.5-GGUF>
- <https://huggingface.co/api/models/PaddlePaddle/PaddleOCR-VL-1.6>
- <https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct>
- <https://huggingface.co/api/models/Qwen/Qwen2.5-VL-3B-Instruct>
- <https://huggingface.co/api/models/Qwen/Qwen3-4B>
- <https://huggingface.co/api/models/Qwen/Qwen3-8B>
- <https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct>
- <https://huggingface.co/api/models/rednote-hilab/dots.ocr>
- <https://huggingface.co/api/models/SeaLLMs/SeaLLM3-7B-Chat>
- <https://huggingface.co/api/models/SeaLLMs/SeaLLMs-v3-7B>
- <https://huggingface.co/api/models/stepfun-ai/GOT-OCR2_0>
- <https://huggingface.co/api/models/Viet-Mistral/Vistral-7B-Chat>
- <https://huggingface.co/api/models/vinai/PhoWhisper-large>
- <https://huggingface.co/api/models/vinai/PhoWhisper-small>

### Hugging Face cards, configuration and license text

- <https://huggingface.co/5CD-AI/Vintern-1B-v3_5/raw/main/README.md>
- <https://huggingface.co/AITeamVN/Vietnamese_Embedding/raw/main/1_Pooling/config.json>
- <https://huggingface.co/AITeamVN/Vietnamese_Embedding/raw/main/README.md>
- <https://huggingface.co/AITeamVN/Vietnamese_Reranker/raw/main/config.json>
- <https://huggingface.co/AITeamVN/Vietnamese_Reranker/raw/main/README.md>
- <https://huggingface.co/allenai/olmOCR-2-7B-1025/raw/main/README.md>
- <https://huggingface.co/BAAI/bge-reranker-v2-m3/raw/main/README.md>
- <https://huggingface.co/dangvantuan/vietnamese-embedding/raw/main/README.md>
- <https://huggingface.co/dangvantuan/vietnamese-embedding/raw/main/sentence_bert_config.json>
- <https://huggingface.co/datalab-to/surya-ocr-2.5/raw/main/LICENSE>
- <https://huggingface.co/datalab-to/surya-ocr-2.5/raw/main/README.md>
- <https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0/raw/main/README.md>
- <https://huggingface.co/dots-studio/dots.ocr/raw/main/README.md>
- <https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md>
- <https://huggingface.co/justtuananh/VietnamesePunctuation/raw/main/README.md>
- <https://huggingface.co/khanhld/chunkformer-ctc-large-vie/raw/main/README.md>
- <https://huggingface.co/namdp-ptit/ViRanker/raw/main/config.json>
- <https://huggingface.co/namdp-ptit/ViRanker/raw/main/README.md>
- <https://huggingface.co/nguyenvulebinh/wav2vec2-base-vi-vlsp2020/raw/main/README.md>
- <https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h/raw/main/README.md>
- <https://huggingface.co/ngxson/Vintern-1B-v3_5-GGUF/raw/main/README.md>
- <https://huggingface.co/openai/whisper-large-v3-turbo/raw/main/README.md>
- <https://huggingface.co/openai/whisper-large-v3/raw/main/README.md>
- <https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B/raw/main/README.md>
- <https://huggingface.co/PaddlePaddle/latin_PP-OCRv3_mobile_rec/raw/main/README.md>
- <https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec/raw/main/README.md>
- <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5-GGUF/raw/main/README.md>
- <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/raw/main/README.md>
- <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/raw/main/README.md>
- <https://huggingface.co/PaddlePaddle/PaddleOCR-VL/raw/main/README.md>
- <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/README.md>
- <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE>
- <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/README.md>
- <https://huggingface.co/Qwen/Qwen3-4B/raw/main/README.md>
- <https://huggingface.co/Qwen/Qwen3-8B/raw/main/README.md>
- <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/README.md>
- <https://huggingface.co/SeaLLMs/SeaLLM3-7B-Chat/raw/main/README.md>
- <https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B/raw/main/LICENSE>
- <https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B/raw/main/README.md>
- <https://huggingface.co/stepfun-ai/GOT-OCR2_0/raw/main/README.md>
- <https://huggingface.co/Viet-Mistral/Vistral-7B-Chat/raw/main/README.md> — HTTP 401
- <https://huggingface.co/vinai/PhoWhisper-large/raw/main/README.md>
- <https://huggingface.co/vinai/PhoWhisper-small/raw/main/README.md>

### PyPI release/dependency metadata

- <https://pypi.org/pypi/chunkformer/json>
- <https://pypi.org/pypi/deepeval/json>
- <https://pypi.org/pypi/docling/json>
- <https://pypi.org/pypi/faster-whisper/json>
- <https://pypi.org/pypi/marker-pdf/json>
- <https://pypi.org/pypi/mineru/json>
- <https://pypi.org/pypi/paddleocr/json>
- <https://pypi.org/pypi/pix2tex/json>
- <https://pypi.org/pypi/pyvi/json>
- <https://pypi.org/pypi/ragas/json>
- <https://pypi.org/pypi/rembg/json>
- <https://pypi.org/pypi/scenedetect/json>
- <https://pypi.org/pypi/sherpa-onnx/json>
- <https://pypi.org/pypi/stable-ts/json>
- <https://pypi.org/pypi/surya-ocr/json>
- <https://pypi.org/pypi/underthesea/json>
- <https://pypi.org/pypi/unimernet/json>
- <https://pypi.org/pypi/vietocr/json>
- <https://pypi.org/pypi/whisperx/json>

### Official project website

- <https://qwenlm.github.io/blog/qwen3/>

Local audit/evaluation files named above are additional local evidence; they are not claimed as web fetches. Network verification completed without a transport failure. No weights were downloaded, no packages installed, no application source changed, and no commit made.

