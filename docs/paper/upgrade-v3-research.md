# Upgrade v3 research: Multimodal Lecture RAG

Research date: 1 November 2026. Target: IUH lecture videos, primarily Vietnamese, with answers grounded in slide images and speech at accurate timestamps.

The goal is to repair evidence loss and corruption before spending the two Kaggle T4 GPUs on replacement models. Experiments below are selected for a direct connection to the five audits, an observable outcome, and a practical CPU or inference-only GPU implementation. Effort estimates are implementation and validation estimates for one developer, not reported paper results.

**Verification limitation: this is a provisional, locally corroborated synthesis, not a completed web-verified literature review.** Direct HTTPS access failed with Windows socket error 10013; the browser host was unavailable; native Chrome access was rejected with “Computer Use was not approved to use Google Chrome”; a non-UI fetch also failed. Consequently, no live arXiv/ACL search succeeded and no additional 2023–2026 paper is claimed to have been discovered or verified online. The supplied new leads are excluded from the evidence tables and recommendations as research evidence until their primary records can be checked. Their verification status is recorded below. This limitation matters especially for the 2025–2026 OCR, Vietnamese reranker, and timestamped lecture-QA leads.

The evidence tables use the available primary PDFs in `docs/paper/` and `docs/paper/edu-rag/`, rather than treating the existing summaries as fresh verification. Titles and author names were checked against those PDFs. Years generally denote the first arXiv release; a conference is named only where the inspected PDF explicitly identifies it. This does not verify the latest online version or repository availability.

Reading labels:

- **P — primary PDF:** complete relevant method/result pages were read from the local full paper. This does not mean every appendix was read. Page numbers refer to PDF pages.
- **A — abstract only:** the relevant claim was read only in the local PDF's abstract. No full-paper experimental claim is inferred from it.
- **S — prior summary:** explicitly secondary evidence; not used to supply a new numerical paper result.
- **U — unverified lead:** no accessible primary record/full paper in this session; excluded as evidence. A supplied label is not a verified canonical title.

## Starting point and evaluation contract

The five audits cover 10 videos, 279.4 minutes, 502 intervals, 51,187 ASR words, and 1,233 chunks. These are corpus counts, not ground-truth accuracy estimates. Particularly actionable observations are:

| Failure | Concrete audit evidence | Implication |
|---|---|---|
| Context assembly loses successful retrieval | 57 speech chunks start beyond the 2,500-character cutoff; the opportunity-cost hit begins at character 5,301 | Fix evidence transport before changing embeddings or the answer model |
| Cleaning destroys valid material | 187 line occurrences removed by frequency filtering; all 33 repeated `a2` assignments removed | Preserve semantic blocks and use position to identify boilerplate |
| Unsafe ASR correction | At least 18 contextual regressions among 112 edits, 16.1% | This is a targeted audit finding, not a randomly sampled correction error rate; stop unrestricted phrase replacement |
| Duplicate OCR crowds results | 119 excess exact copies among 499 OCR chunks; one question has 15 identical copies | Group content before candidate limits and final slide selection |
| OCR/ASR degeneration survives | Seven OCR loop-bearing records; five long ASR repetition runs | Detect failed extraction before caching/indexing; collapsing a loop cannot recover omitted content |
| Visual chronology is wrong | A revealed bullet is indexed 83.1 seconds before its onset; two board intervals last 578.96 and 650.4 seconds | Preserve visual states and evidence onset separately from presentation groups |
| Ads and diagrams have opposite integration errors | 13 verified ad-bearing intervals remain searchable; 205 generated figure records produce zero region chunks | Filter promotional spans while retaining reviewed, provenance-labelled visual evidence |

The previous roadmap already implemented Qwen2.5-7B, timestamp display, streaming, E5 prefixes, VAD, follow-up rewriting, source ablations, OCR loop collapsing, and a SigLIP prompt/threshold experiment. It also selected Qwen OCR, rejected GOT as a Vietnamese replacement, and found E5 better than BGE-M3 on its larger local question set. Those choices are baselines here. Do not repeat those comparisons without a new failure hypothesis. PhoWhisper, RAGAS, neighbour expansion, and image-conditioned answering were proposed but not demonstrated as complete; the experiments below make their scope and acceptance criteria more precise.

Use `pipeline/eval/` as the existing evaluation entry point:

- `rebuild_offline.py` reconstructs stages 04/05 from raw exported OCR/ASR with CPU ONNX E5, isolated experiment outputs, and parity/provenance checks. Its configuration is an allowlist, and it currently only supports the pinned production E5. New chunking, query construction, reranking, and model variants require deliberate adapters later; they are not already arbitrary harness switches.
- `eval_retrieval.py` measures the production search and the actual `build_context` output, reporting Hit@1/3/5, MRR, context recall, and quality lints. **The production maximum is three slides, so current Hit@5 equals Hit@3.** Obtain five genuine candidates before interpreting an experimental Hit@5.
- Current “context recall” is a binary evidence-retention diagnostic using gold spans and key terms, not exhaustive gold-sentence recall or the identically named RAGAS metric. Some term matching is fuzzy. Add human checks for exact numbers, code, and visual relationships; these cannot be certified by a fuzzy term hit.
- Slide-overlap Hit@k/MRR can reward an incorrectly long board interval. Supplement them with narrow, human-checked evidence spans and timestamp error; keep retrieval success, evidence retention, factual accuracy, and temporal accuracy separate.
- Promotional-phrase counts, near duplicates, repetition flags, short chunks, missing titles, and the four known bad-correction pairs are **lints**, not corpus error rates. Report both flagged counts and precision on a manually reviewed sample.

Before tuning, review the existing development questions against the video/keyframe/audio, including the previously unreviewed machine-written questions. Reserve entire videos, lecturers, or topics for a held-out set where feasible. Freeze question/source hashes, model revisions, processing settings, and a baseline context trace. Use paired changes on the same questions and report per-video and per-type results; small differences need uncertainty estimates rather than a winner declared from one or two questions.

## 1. Slides & gate

### Papers

| Paper and authors | Identifier / venue and link | Year | Finding relevant to this module | Reading |
|---|---|---:|---|---|
| *Sigmoid Loss for Language Image Pre-Training* — Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer | [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) | 2023 | Pairwise sigmoid training does not require global softmax normalization; this is not a validation of a softmax over 14 hand-written prompts as a readability probability | A, PDF p.1 |
| *Multimodal Lecture Presentations Dataset: Understanding Multimodality in Educational Slides* — Dong Won Lee, Chaitanya Ahuja, Paul Pu Liang, Sanika Natu, Louis-Philippe Morency | [arXiv:2208.08080](https://arxiv.org/abs/2208.08080) | 2022 | Aligned educational slides and speech expose weak crossmodal alignment and technical/visual difficulties; useful as an existing external reference, not a board-segmentation benchmark | A, PDF p.1; older exception |
| *Qwen2.5-VL Technical Report* — Qwen Team | [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) | 2025 | Dynamic-resolution processing, spatial localization, and absolute-time video encoding support an inexpensive candidate-review experiment; the abstract does not establish lecture/board boundary accuracy | A, PDF p.1 |

**Evidence gap:** neither SliTraNet nor the Kenny/Davila/Zanibbi whiteboard leads could be checked online. No recent VLM lecture/blackboard segmentation or educational ad detector is verified here. The following are audit-driven engineering experiments, not claimed reproductions of those papers.

### What to try in our pipeline

| Experiment | Audit weakness addressed and concrete implementation | Effort | GPU needed | Measurement |
|---|---|---|---|---|
| S1. Local board-content changes | Configure a board/editor/slide ROI and presenter masks per layout. Compare local foreground/text-region additions **and deletions**, rather than whole-frame colour changes. Evaluate 1 Hz versus 2 Hz sampling and a periodic snapshot fallback for dynamic intervals longer than 120 s. Start with `hVU5Mw19V_c` | 3–5 days | No, CPU frame analysis; GPU only for optional OCR adjudication | Human-labelled board content-state boundaries, boundary precision/recall/F1 at stated tolerances, missed-example count, onset error, states/minute, runtime; then Hit@k and context recall for affected questions |
| S2. Replacement-safe reveal groups | Compare directional old-content retention, deleted regions, and new content. Code-line replacement must split states even when aggregate pixels mostly survive. Keep a presentation-group ID plus independently timed state IDs; index a new bullet/formula from its actual onset | 3–5 days, including schema propagation | No for geometry; yes only for ambiguous OCR rereads | Reproduce code states 58–60 and the 83.1-second early bullet; count false merges and missing states, visual onset MAE/p95, coverage/order lints |
| S3. Readability fallback separate from lesson relevance | Retain the existing SigLIP baseline. Before `skipped_no_content`, check a full-width caption strip, slide/board ROI, and cheap CPU text/numeral detector. Expand prompt coverage for instructional captions/counting only after labelling misses. Do not infer “ad” from “not a slide” | 1–2 days | No for fallback; existing SigLIP inference can remain on GPU | Readable-content recall and false-admission rate on rejected captions, fruit numbers, title cards, blank frames, and ads; count recovered correct OCR facts rather than admitted frames alone |
| S4. Content-aware keyframes and camera identity | Score sharpness inside the lesson ROI; penalize presenter occlusion and transient tooltips. Retain two complementary frames when one cannot expose the board. Normalize the displayed screen region across camera views; link repeated content without erasing genuine later revisions | 2–4 days | No; optional existing VLM review uses one T4 | Human readability/occlusion labels, OCR numeric accuracy, missing-board-region count, duplicate OCR excess, and downstream Hit@k |
| S5. Promotional-span filtering | Evaluate the existing joint `slide_speech` path with speech windows, explicit recruitment/subscription cues, slide evidence, and temporal smoothing. Use existing thresholds only as starting values. Split mixed ad/lecture intervals and retain raw evidence. Low topic similarity alone must not remove the legitimate radium explanation | 2–3 days | No; existing E5 ONNX suffices | Manually label promotional spans and lecturer-only negatives; span precision/recall, lesson seconds falsely removed, ad chunks in top results, Hit@k/context recall regressions |
| S6. Sparse VLM boundary review, optional | Give the existing Qwen3-VL model two or three timestamp-labelled ROI frames around uncertain changes; ask for visible added/deleted content and an evidence-based split/merge suggestion. Apply it only after the CPU detector proposes candidates. Require a human-reviewed evaluation before automated decisions | 2–3 days after S1/S2 | Yes, one T4; inference only | Compare candidate decisions against humans and the CPU baseline; added recall per extra GPU minute. Do not extrapolate Qwen video benchmarks to board segmentation |

## 2. OCR

### Papers

| Paper and authors | Identifier / venue and link | Year | Finding relevant to this module | Reading |
|---|---|---:|---|---|
| *Nougat: Neural Optical Understanding for Academic Documents* — Lukas Blecher, Guillem Cucurull, Thomas Scialom, Robert Stojnic | [arXiv:2308.13418](https://arxiv.org/abs/2308.13418) | 2023 | Repetition occurred on 1.5% of test pages and increased out of domain; alternating/modified loops can evade exact matching, motivating generation-time logit-variance diagnostics | P, pp.8–9, §5.4 |
| *Qwen2.5-VL Technical Report* — Qwen Team | [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) | 2025 | Spatial localization and structured document/table extraction motivate explicit region/type/title output; this is not evidence that current Qwen3-VL OCR is numerically faithful | A, p.1 |
| *M3AV: A Multimodal, Multigenre, and Multipurpose Audio-Visual Academic Lecture Dataset* — Zhe Chen, Heyang Liu, Wenyi Yu, Guangzhi Sun, Hongcheng Liu, Ji Wu, Chao Zhang, Yu Wang, Yanfeng Wang | [arXiv:2403.14168](https://arxiv.org/abs/2403.14168) | 2024 | Provides 24,956 slides and 772 pages associated with 8,217 blocks in its handwriting/formula subset; blocks on a page with complex content are also counted, so not all 8,217 are formulas | P, p.4, Table 4 |
| *General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model* — Haoran Wei, Chenglong Liu, Jinyue Chen et al. | [arXiv:2409.01704](https://arxiv.org/abs/2409.01704) | 2024 | A small unified OCR model is an existing comparator; the repository already found poor Vietnamese performance, so no repeat whole-slide replacement experiment is justified | S, existing GOT summary/roadmap; no new numerical result used |

**Evidence gap:** PaddleOCR-VL, PaddleOCR-VL-1.5, dots.ocr, olmOCR/olmOCR-Bench, and OmniDocBench remain unverified leads. No Vietnamese OCR benchmark was verified in this session. M3AV is English and cannot establish Vietnamese OCR quality. No published numerical gain is claimed for the proposed crop/engine self-consistency checks.

### What to try in our pipeline

| Experiment | Audit weakness addressed and concrete implementation | Effort | GPU needed | Measurement |
|---|---|---|---|---|
| O1. Preserve semantic content before cleaning | Keep raw OCR and a typed representation. Protect titles, code whitespace, table cells, formulas, and short symbolic lines. Frequency alone must not delete a line; remove only explicit branding or position-confirmed footer/sidebar material. Do not title a slide from the first surviving long line | 1–2 days | No; rebuild from raw OCR | Recover all audited `a2` definitions, headings, and `i=1` bounds; exact protected-block retention, code indentation checks, numeric exact match, title precision, then term/code/numeric Hit@k and context recall |
| O2. Validate before caching | Store token count, EOS/finish status, input dimensions, generation settings, repetition features, and `read/uncertain/failed/no_text` status. Reject “no text” boilerplate as searchable content. Flag exact, alternating, and near-repeat loops; a token-limit hit is a retry candidate, not a complete read | 1–2 days plus targeted reruns | No for text validation; yes for rereads | OCR/ASR loop lints, no-text chunks, truncation flags, detector precision/recall on reviewed outputs, content recovered after retry, failure/reuse counts |
| O3. Targeted crop verification of numbers and structure | Reread only a flagged numeric/formula/table/code region at preserved resolution. Compare whole-frame versus crop output; optionally compare a second existing OCR engine for literal text. Preserve both candidates and mark disagreements. Check arithmetic/table totals only to flag inconsistencies, never to silently “correct” a teacher's slide | 2–3 days | Yes, existing Qwen3-VL on one T4; classical checks on CPU | Manually transcribe audit values `100/370`, `1,5 kg`, `46,75%`; numeric/identifier exact match, normalized formula edit distance, table-cell accuracy, disagreement precision, extra seconds/slide |
| O4. Explicit title and bounded blocks | Request visible `title`, `blocks[{type,bbox,text}]`, code in fences, LaTeX, and Markdown tables. Validate boxes against image coordinates and separate reading order from position. Empty/uncertain title is preferable to “Clear Update.” Keep intentionally incomplete live code instead of enforcing successful compilation | 2–4 days | Yes for structured rereads; parser/validators CPU | Title accuracy, block coverage/reading order, indentation and delimiter preservation, table-cell/formula fidelity, OCR CER by content type; title/short-chunk lints are supporting diagnostics |
| O5. Make diagrams usable with provenance | Replace the full-frame pseudo-region with real diagram crops where possible. Extract literal labels, axes, values, and arrow endpoints separately from interpretation. Index reviewed literal evidence first. Test generated descriptions as a separate, uncertainty-labelled retrieval channel, excluding empty/no-figure text; do not relabel them as human verified | 2–4 days | No for reviewed existing descriptions/index wiring; yes for new region extraction | Human review of figure facts; visual-question Hit@k/context recall, unsupported description rate, coverage of the 188 non-noise figure records, new region chunks and their provenance |
| O6. Calibrate generation-time degeneration, optional | Borrow Nougat's idea of compact statistics over generated logits, not its exact threshold. Record only selected summaries where possible; retaining every vocabulary logit can be costly. Combine with text loops and coverage/EOS checks, then crop-retry. Anti-repetition training from Nougat is out of scope | 2–3 days after O2 | Yes for instrumented inference | Loop-detection recall/false positives against human labels, latency saved, missed content after early stopping. Existing loop collapse is the baseline |

For crop self-consistency, deterministic rereading of the same input is not independent evidence. Vary the crop/resolution or engine and evaluate correlated errors explicitly. Agreement can still be wrong. Create a small human-transcribed Vietnamese slide/board/code set from the actual lecture corpus, stratified by diacritics, handwriting, formulas, tables, captions, and code; call it an internal evaluation set rather than a public benchmark. A suggested starting size is 60–100 regions, not a statistically justified corpus error estimate. Avoid duplicating the same slide state across development and test.

After verification becomes possible, test one new OCR candidate at a time on those regions and the six non-reused loop reads. Check license, code/model availability, Vietnamese coverage, actual T4 peak memory, output fidelity, and throughput before any full rerun. The audit's OCR wall-time share is 98.3%, so global rereading is expensive; prioritize deduplication and selective retries.

## 3. ASR

### Papers

| Paper and authors | Identifier / venue and link | Year | Finding relevant to this module | Reading |
|---|---|---:|---|---|
| *SlideSpeech: A Large Scale Slide-Enriched Audio-Visual Corpus* — Haoxu Wang, Fan Yu, Xian Shi, Yuezhang Wang, Shiliang Zhang, Ming Li | [arXiv:2309.05396](https://arxiv.org/abs/2309.05396) | 2023 | On L95 test, the keyword-conditioned system reduces WER 12.89→12.38 and keyword B-WER 12.70→9.32; it is a trained contextual CTC/AED system, not a Whisper-prompt result | P, pp.2–4, Table 2 |
| *M3AV: A Multimodal, Multigenre, and Multipurpose Audio-Visual Academic Lecture Dataset* — Zhe Chen et al. | [arXiv:2403.14168](https://arxiv.org/abs/2403.14168) | 2024 | RNN-T+TCPGen with OCR-only context reduces test BWER 38.0→25.0, 34.2% relative; adding same-topic rare-word distractors worsens it to 31.7 | P, p.7, Table 5 |
| *PhoWhisper: Automatic Speech Recognition for Vietnamese* — Thanh-Thien Le, Linh The Nguyen, Dat Quoc Nguyen | [arXiv:2406.02555](https://arxiv.org/abs/2406.02555), ICLR 2024 Tiny Paper | 2024 | Medium achieves VIVOS WER 4.97 and VLSP Task-2 WER 26.85; large achieves 4.67 and 26.68. The 844-hour fine-tuning and test domains do not guarantee lecture-term accuracy | P, pp.1–2, Table 2 |

**Evidence gap:** the LCB-net, KG-Whisper, prompting/rare-word Whisper, Whisper TCPGen, hallucination/repetition, VietASR, ChunkFormer, and generative-error-correction leads are not primary-verified here. M3AV verifies an RNN-T TCPGen result, **not** the supplied Whisper TCPGen paper. No checked paper establishes large-v3/turbo superiority on Vietnamese lectures. Keep those checkpoints as unproven engineering comparison candidates, not a supported replacement recommendation.

### What to try in our pipeline

| Experiment | Audit weakness addressed and concrete implementation | Effort | GPU needed | Measurement |
|---|---|---|---|---|
| A1. Remove unrestricted OCR phrase correction | Retain `text_asr` as immutable evidence. Compare no correction with a reviewed terminology/alias allowlist. Require local slide support, whole-term/longest-match replacement, punctuation boundaries, and an edit log. Generic phrases such as “trong kinh” must never enter the term list | 1–2 days | No; existing exports are enough | Review all 112 edits, correct-edit precision and harmful-edit rate, non-term WER/CER, known-bad-correction lint, term retrieval. Zero known regressions is necessary but not proof all remaining edits are safe |
| A2. Local visual context in decoding, not lecture-wide substitutions | Extract approved technical terms from the current slide and, near a boundary, the adjacent state. Start with a small reviewed glossary (`Bayes`, `PPF`, `len`, `for`, `end`, `xác suất`, `vũ trang`). Compare no prompt, course glossary, and local-slide glossary. Preserve the fast first ASR pass; retry only affected windows once OCR is ready | 2–3 days | Yes, one T4 for ASR retries | Human-reviewed term error/recall and BWER-style scores including biased insertions, non-term WER, prompt-induced false insertions, runtime, term/code Hit@k. Use the same audio windows for every variant |
| A3. Repetition and hallucination diagnostics | Retain segment compression ratio, average log probability, no-speech score, VAD span, repetition runs, and suspicious timing. Flag high-confidence loops too: the audit has a loop with mean word probability 0.939. Retry with surrounding audio, different windowing, and tested decoding fallbacks; listen before excluding persistent cases | 1–2 days plus retry evaluation | No for diagnostics; yes for ASR retries | Human hallucination/loop precision and recall, repeated-output duration, recovered words, genuine repetition falsely flagged, ASR-loop lints; do not infer hallucination solely from silence or confidence |
| A4. Controlled Vietnamese model comparison | Build a human transcript set covering board probability, economics, Python identifiers, southern/northern accents, noisy speech, silence, and mixed English. Compare current medium with PhoWhisper small/medium; only then add multilingual large-v3 and turbo if their artifacts are verified and runnable. Keep the English lecture as a separate holdout | 3–5 days including annotation | Yes, one model per T4, sequential tests; CPU possible but slow | Report accent-sensitive CER, syllable WER and a documented segmented-word WER, terminology accuracy, English regression, loop rate, RTF, peak VRAM, timing quality. Do not compare paper WER directly with a differently normalized local score |
| A5. Targeted timing repair | Flag ≤1 ms timings, implausibly long words, and words spanning slide changes. Retain coarse segment timing/uncertainty until audio-grounded alignment succeeds. Evaluate a Vietnamese-capable aligner only after model/language support is verified; otherwise expose segment spans rather than manufactured precision | 2–3 days | No for flagging/coarse export; GPU may be needed for selected alignment model | Manually label word/phrase anchors; start/end MAE, p95 error, boundary slide assignment accuracy, coarse/word timing coverage. The audit's 630 artificially short words are flags, not a measured aligner error rate |

A short-term contextual prompt is a translation of the visual-context hypothesis, not a reproduction of SlideSpeech or TCPGen. Local OCR must pass O1–O4 first, or a wrong OCR term becomes a strong decoding bias. A 45-second external window can span several slides; derive the glossary from actual overlapping states instead of assuming one window equals one slide. Retest loss of sequential context separately from previous-text conditioning so decoder-loop controls do not become confounded with vocabulary guidance.

Do not jump to LLM generative correction. Once HyPoradise/GenSEC/ClozeGER are verified, they would still need local tests of edit precision, acoustically unsupported insertions, number preservation, and meaningful context. No N-best list currently described means an N-best correction method is not automatically plug-and-play. Decoder-level guidance and a restricted glossary are the lower-cost experiments.

## 4. Sync & chunking

### Papers

| Paper and authors | Identifier / venue and link | Year | Finding relevant to this module | Reading |
|---|---|---:|---|---|
| *SlideSpeech: A Large Scale Slide-Enriched Audio-Visual Corpus* — Haoxu Wang et al. | [arXiv:2309.05396](https://arxiv.org/abs/2309.05396) | 2023 | Validates candidate transcripts with subtitle-text alignment while explicitly avoiding unreliable subtitle timestamps; its 473-hour high-confidence subset is not evidence of accurate timestamps in our pipeline | P, pp.2–3 |
| *Learning Engagement Assistant (LEA): Cross-Course Scalability and Classroom Evaluation of an Agentic AI Tutoring System* — Teri Rumble, Javad Zarrin, P. George Lovell, Ruth Falconer | [arXiv:2607.13370](https://arxiv.org/abs/2607.13370), extension of ICAART 2026 work | 2026 | CMP202 context recall is 0.404 versus CMP511 0.609; the authors hypothesize code sensitivity of 500-token sentence chunking, without a causal chunking ablation or independent reference answers | P, pp.21–22, Table 6 |
| *VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents* — Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, Maosong Sun | [arXiv:2410.10594](https://arxiv.org/abs/2410.10594), ICLR 2025 | 2024 | The evaluated SlideVQA subset has 556 queries and an average 1.26 positive pages/query, reinforcing the need to preserve multi-page evidence rather than assuming one slide per answer | P, pp.5–6, Table 1 |

**Evidence gap:** Reconstructing Context and Jina late chunking were not verified. Sentence/window/parent-child experiments below are retrieval designs to test, not published gains. Late chunking would require token-level encoder outputs from a sufficiently long-context model; concatenating many tokens into the existing 512-token E5-small call is not late chunking.

### What to try in our pipeline

| Experiment | Audit weakness addressed and concrete implementation | Effort | GPU needed | Measurement |
|---|---|---|---|---|
| C1. Transport the retrieved child passage | Carry ranked chunk IDs, source text, word/state spans, and provenance through context assembly. Allocate a model-token budget to the actual hits before adding parent material. Replace whole-slide prefix slicing; cite the matched span, not automatically the parent start | 1–2 days | No | Opportunity-cost regression at 1466.12–1501.16 s; retained-token share, context recall, matching-chunk survival, citation onset error. Hit@k should be unchanged if ranking is held fixed |
| C2. Sentence/pause children and bounded parent windows | Preserve midpoint ownership for per-slide accounting, but build speech children from complete sentences or pause-bounded clauses. Allow multiple overlapping slide references. Compare current 18 s/70-word chunks with 2–4-sentence children and retrieval-centred windows; use token bounds and keep OCR tables/code/formulas atomic | 2–3 days | No, CPU E5 reindexing | Sentence-cut/orphan-fragment rate, <32-token lint, Hit@k/MRR/context recall, answer accuracy under a fixed context budget; explicit `go`/`back` boundary regression |
| C3. Conditional previous/next expansion | Compare no expansion, always one neighbour each side, and conditional expansion for cut sentences, unresolved referents, continued derivations, or queries asking about successive steps. Start with one adjacent segment; cap tokens/time and stop at ads, unrelated topics, or missing continuity | 1–2 days after C1 | No; optional LLM continuity check is a separate costly arm | Multi-span question recall, retained central hit, irrelevant-context share, tokens/question, answer faithfulness, latency. Keep primary retrieved hits separate from added neighbours in traces |
| C4. Chronological evidence and provenance | Model `presentation_group`, `content_state`, and independent `speech_span` IDs. A cross-boundary sentence links to all applicable states. Keep raw seconds, actual visual onset, and timing precision. Sort selected evidence chronologically for explanations while retaining original relevance ranks in metadata | 2–3 days; coordinate with S2 | No | Timeline coverage/order lints plus human timestamp MAE; no evidence cited before appearance. A content duplicate may have several occurrences, each with its own speech alignment |
| C5. Portable export validation | Declare which assets are included versus reconstructible; preserve source hashes and regeneration metadata for omitted audio. Validate lineage, chunk/vector agreement, completion state, and availability before import. This protects reproducible timing/OCR/ASR experiments | 1–2 days | No | Remove the 1,014 manifest-asset inconsistencies through a valid contract, not by pretending absent files exist; export/import validation and rebuild parity |

Use the same encoder and source text when comparing chunk boundaries. Semantic sentence embedding breakpoints can be a later CPU experiment against pause/sentence splitting, but noisy ASR punctuation and Vietnamese word boundaries require review. Parent-child retrieval should not mean embedding whole long slides and discarding the child that matched. Neighbours provide continuity, not additional unearned retrieval hits: report both base retrieval and expanded context coverage.

## 5. Retrieval

### Papers

| Paper and authors | Identifier / venue and link | Year | Finding relevant to this module | Reading |
|---|---|---:|---|---|
| *Multilingual E5 Text Embeddings: A Technical Report* — Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei | [arXiv:2402.05672](https://arxiv.org/abs/2402.05672) | 2024 | Offers small/base/large multilingual models; retaining the existing small model makes preprocessing/ranking ablations inexpensive and attributable | A, p.1; existing baseline |
| *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation* — Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu | [arXiv:2402.03216](https://arxiv.org/abs/2402.03216) | 2024 | MKQA Vietnamese Recall@100 is 76.9 for all modes versus 75.4 for mE5-large; this is Vietnamese queries over **English Wikipedia**, not same-language Vietnamese lecture retrieval | P, pp.6–7, Tables 1–2 |
| *ColPali: Efficient Document Retrieval with Vision Language Models* — Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo | [arXiv:2407.01449](https://arxiv.org/abs/2407.01449), ICLR 2025 | 2024 | Direct page-image multi-vector retrieval preserves visual evidence that text extraction loses; Vietnamese lecture effectiveness is not demonstrated by the inspected abstract | A, p.1; optional diagnostic reference |

**Evidence gap:** ViRanker, AITeamVN/Vietnamese_Reranker, Vietnamese legal segmentation papers, AdaGReS, and DF-RAG are unverified. No verified Vietnamese reranker accuracy or T4 memory result is asserted. Exact grouping and an MMR-style selection objective can still be implemented and tested as engineering baselines without claiming a particular paper's result.

### What to try in our pipeline

| Experiment | Audit weakness addressed and concrete implementation | Effort | GPU needed | Measurement |
|---|---|---|---|---|
| R1. Repair Vietnamese lexical normalization | Remove only validated accented stopwords before accent folding; preserve domain phrases. Compare this with no stopword deletion. Add a segmented/phrase-aware field alongside the accent-folded syllable fallback. Protect `lực từ`, `độ cứng`, `chi phí cơ hội`, identifiers, and numbers. Keep query/index normalization compatible | 1–2 days | No | Lexical token retention and BM25-only Hit@k/MRR for audited terms; accented/unaccented query pairs; full hybrid metrics and latency. Word segmentation is a hypothesis, not a guaranteed legal-to-lecture transfer |
| R2. Deduplicate before limits | Canonicalize exact OCR bodies and cluster reviewed near duplicates before branch top-N truncation. Store every occurrence timestamp and state ID. Retrieve content groups, then select the occurrence using relevant speech/local temporal context. Do not merge different numbers, formulas, or program revisions | 2–3 days | No | Exact/near duplicate lints, unique evidence groups in candidates/top three, radium regression, temporal occurrence accuracy, Hit@k/MRR/context recall |
| R3. Recalibrate fusion after grouping | Sweep a small development grid for lexical weight, branch candidate depth, and OCR boost including 1.0. Keep E5, BM25, and weighted RRF as the baseline. Count unique content groups rather than rewarding repeated occurrences; do not sum repeated-slide evidence indiscriminately | 1 day after R1/R2 | No | Per-type Hit@k/MRR, context recall, distinct-source coverage; held-out performance. The audited 1.05 boost can outrank a speech chunk ranked first by both branches |
| R4. Query-aware diversity within a budget | Compare group-only deduplication with relevance-minus-redundancy selection among the top candidate groups. Permit complementary speech/OCR/diagram evidence. Tune the diversity penalty on development data; do not force diversity when one passage fully answers a precise fact | 1–2 days | No, existing embeddings/text similarity | Unique evidence coverage, multi-span context recall, numeric/term regressions, irrelevant-context fraction, tokens/question |
| R5. Rerank a bounded candidate set | Once a Vietnamese or multilingual checkpoint is primary/model-card verified, rerank approximately 30–40 unique query/passage pairs from the hybrid union. Include provenance and validated titles; rank passages before deciding which slide occurrence to show. CPU ONNX is a possible arm only if an export actually exists/works | 2–3 days after verification | No for a feasible small CPU model; yes for GPU comparison | MRR/Hit@1 versus the repaired hybrid baseline, per-question latency, peak memory, numeric/code performance; retrieval ceiling from candidate recall before reranking |
| R6. Small visual retrieval diagnostic, optional | On questions whose correct evidence is absent from OCR, compare text retrieval with a verified ColPali-style image branch on a small subset. This revisits the old “text first” decision only if R1–R5/O5 cannot supply the visual evidence; do not replace the main index pre-emptively | 3–5 days | Yes for image embeddings; small exact index can be searched on CPU | Visual-question retrieval accuracy, index size/time, Vietnamese-query failures, gain beyond image-conditioned answering on the same text-retrieved slides |

The existing local E5-versus-BGE result takes priority over an unrelated public benchmark. Do not propose another dense-only BGE-M3 swap. A distinct sparse/multi-vector hypothesis could be revisited later, but it requires a new harness adapter and a credible candidate-recall failure after lexical repairs. Neither RRF scores nor E5 cosine values are calibrated probabilities of answerability; do not transfer the old E5 threshold to a new encoder or reranker.

## 6. Answer generation

### Papers

| Paper and authors | Identifier / venue and link | Year | Finding relevant to this module | Reading |
|---|---|---:|---|---|
| *VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents* — Shi Yu et al. | [arXiv:2410.10594](https://arxiv.org/abs/2410.10594), ICLR 2025 | 2024 | §3.2.2 compares page concatenation, per-page weighted selection, and multi-image generation; multi-image evidence is relevant to adjacent lecture slides | P, pp.4–6; the headline 20–40% gain was abstract-only and is not a predicted gain here |
| *M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding* — Jaemin Cho, Debanjan Mahata, Ozan İrsoy, Yujie He, Mohit Bansal | [arXiv:2411.04952](https://arxiv.org/abs/2411.04952) | 2024 | Uses retrieved page images for answering; its open-domain benchmark contains 2,441 multi-hop questions over 3,368 PDFs/41,005 pages, not Vietnamese timestamped lecture QA | P, pp.4–5, §§2.3/3 |
| *Qwen2.5-VL Technical Report* — Qwen Team | [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) | 2025 | Open 3B/7B variants and structured visual localization support a small VLM answering comparator; performance/fit of the current Qwen3-VL-4B answering path must be measured | A, p.1 |

**Evidence gap:** “From Slides to Chatbots,” CourseTimeQA, EduVidQA, and Video-MMLU could not be primary-verified. No lecture-QA benchmark score is borrowed from them. The prior roadmap already suggested slide images and an abstention prompt; the upgrade here is a controlled image-routing experiment and measurable evidence/citation validation.

### What to try in our pipeline

| Experiment | Audit weakness addressed and concrete implementation | Effort | GPU needed | Measurement |
|---|---|---|---|---|
| G1. Text-retrieved, image-conditioned answers | Keep retrieval fixed. Compare Qwen2.5-7B text context with the existing Qwen3-VL-4B receiving the same matched speech/OCR plus one or two relevant state images/crops. Begin with diagrams, changed numbers, and equations. Include visible state times and distinguish speech from image evidence | 2–4 days | Yes, one T4, one VLM request at a time | Human exact numeric/diagram/code correctness, claim support, citation accuracy, abstention, latency/peak VRAM. Hold retrieved sources fixed to separate generation gain from retrieval gain |
| G2. Route visual questions selectively | Use explicit mentions of diagrams, axes, arrows, tables, code layout, or conflicts between OCR and speech as an initial route. Compare all-text, routed VLM, and all-VLM on the same labelled questions. A route must not turn uncertain OCR into presumed fact | 1–2 days after G1 | No for rule routing; yes on routed requests | Route recall/precision, visual accuracy gain, false visual routes, mean/p95 response time, GPU minutes/question |
| G3. Evidence sufficiency and calibrated abstention | Extend the existing prompt with structured supported/unsupported status and source IDs for each answer claim. Calibrate on truly unanswerable questions and answerable questions with missing/corrupted evidence. Return partial supported explanations where appropriate; disclose a conflict rather than synthesizing a confident wrong number | 1–2 days plus evaluation | No required; current Ollama model on CPU is usable, GPU speeds it up | Unsupported-claim rate, false-abstention rate, coverage versus correctness, unanswerable precision/recall, human support judgements; no threshold based on fused rank score alone |
| G4. Validate citations against selected evidence | Generate evidence IDs internally, then render `[mm:ss]` from allowed speech/state spans. Reject invented IDs or out-of-span times; coarse timing must remain coarse. Explain using the selected chronological evidence and preserve the primary matched passage under budget limits | 1–2 days | No for validation | Citation-ID validity, timestamp start/end MAE and video-level correctness, claim-to-source support, percentage of answers with supported citations. A valid timestamp string is not proof the claim is supported |

A VLM can check the pixels when OCR says `4,5 kg`, but can hallucinate the same value. Neither agreement nor “multimodal” status certifies truth. Use exact human-transcribed critical facts in the visual set and measure abstention on occluded/unreadable regions. Keep generated figure interpretations visibly distinct from literal transcription in the model context. Avoid treating “Không có hình…” as a diagram description.

Kaggle feasibility is an inference assumption to validate, not a paper result: the present Qwen3-VL-4B OCR path already runs on T4, making reuse a reasonable first candidate. T4 does not offer native BF16/FlashAttention-2 support; use a supported FP16/quantized attention path, small image/token budgets, and measure memory. Two 16 GB cards do not automatically form one 32 GB memory pool. Run independent workers or sequential models; a 7B VLM may need quantization or explicit sharding. Do not attempt 72B inference or contextual-ASR training for this thesis experiment.

## 7. Evaluation

### Papers

| Paper and authors | Identifier / venue and link | Year | Finding relevant to this module | Reading |
|---|---|---:|---|---|
| *Ragas: Automated Evaluation of Retrieval Augmented Generation* — Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert | [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) | 2023 | On 50-page English WikiEval pairwise comparisons, agreement is 0.95 for faithfulness, 0.78 for answer relevance, and 0.70 for context relevance; this is not Vietnamese/education validation | P, pp.4–5, Table 1 |
| *Learning Engagement Assistant (LEA): Cross-Course Scalability and Classroom Evaluation of an Agentic AI Tutoring System* — Teri Rumble et al. | [arXiv:2607.13370](https://arxiv.org/abs/2607.13370), extension of ICAART 2026 work | 2026 | Across 660 questions, faithfulness falls 0.685→0.499 across courses despite high relevancy/precision; context precision/recall use its own generated answers as stand-in references | P, pp.21–22, Table 6 |
| *Aligning LLMs for the Classroom with Knowledge-Based Retrieval: A Comparative RAG Study* — Amay Jain, Liu Cui, Si Chen | [arXiv:2509.07846](https://arxiv.org/abs/2509.07846) | 2025 | EduScopeQA covers 3,176 questions and different query scopes; useful to stratify evaluation, but the inspected abstract does not establish judge reliability | A, p.1; AB–BA protocol is S from the existing summary |

**Evidence gap:** ARES and the four new lecture-QA leads were not verified; no invented ARES arXiv/ACL identifier or score is supplied. The educational papers above are not a substitute for primary verification of those benchmarks.

### What to try in our pipeline

| Experiment | Audit weakness addressed and concrete implementation | Effort | GPU needed | Measurement |
|---|---|---|---|---|
| E1. Review and freeze gold evidence | Human-check existing questions, answers, required facts, source modality, and narrow time spans. Add targeted blackboard replacement, reveal onset, captions, numeric conflicts, code indentation, ads, cross-slide continuity, and unanswerable cases. Separate development from held-out material | 2–4 days of annotation/review | No | Annotation disagreements and resolution, per-type/video counts, question hashes, interval-versus-span agreement; baseline harness run with actual context traces |
| E2. One change at a time, then interactions | Run the repaired context builder, cleaner, correction policy, BM25, deduplication, and filter individually against the same frozen exports/questions. Then test their combined version; run GPU extraction experiments only after CPU gains are known | 1–2 days for initial runs, then incremental | No for CPU arms; yes only for extraction/model arms | Paired Hit@k/MRR/context recall, lints with manually checked precision, fixed budget/latency. Report a modest change as uncertain when the held-out set cannot distinguish it |
| E3. Human-calibrated automated judging | On a suggested 30–50-answer calibration sample, two Vietnamese-capable reviewers mark factual correctness, claim support, explanation quality, and citation timing. Compare a fixed judge/prompt with humans, including wrong numbers and fluent unsupported answers. Shuffle/blind configuration names; if pairwise judging is used, swap A/B positions and report disagreement | 2–3 days | No required; GPU/API can accelerate judging, but API access is not assumed | Judge–human agreement, false acceptance of unsupported facts, position consistency, repeated-judgement stability, per-modality failures. Suggested sample size is a practical pilot, not a powered reliability study |
| E4. RAGAS with independent references | Add pinned RAGAS faithfulness/relevance metrics alongside deterministic harness outputs. Supply human references for metrics that require them. Never reuse the answer under test as gold. Pin library/metric names: the original paper's context relevance differs from later context precision/recall APIs | 1–2 days after E3 | No required; current local model is slow, one T4 optional | Metric agreement with calibrated human scores, missing/failed judge calls, generation correctness conditional on good retrieval, cited evidence support |
| E5. Timestamp and extraction fidelity | Add a reviewed evaluation protocol, not just timeline-validity lints: phrase/visual-onset errors, exact numbers, code tokens/whitespace, formula/table fidelity, term errors, and promotional-span precision. Report failed/uncertain evidence rather than hiding it in aggregate averages | 2–3 days alongside module annotation | No for scoring; yes for model trials | Timestamp MAE/p95 and tolerance success, numeric EM, CER/WER with specified normalization, diagram relation accuracy, ad false exclusions; relate each to QA/context recall |
| E6. Budget and reproducibility | Record checkpoints/revisions, model/processor settings, source/code/question hashes, cache identity, raw versus cleaned evidence, individual inference time versus queue time, reuse/retry rate, peak memory, and answer latency | 0.5–1 day initially | No for bookkeeping; GPU measurements during inference | Rebuild parity, revision-sensitive caches, valid portable manifests, seconds/video, seconds/slide, RTF, tokens/question, p50/p95 response time |

The acceptance principle is **better supported answers at accurate times under a stated cost**, not simply higher slide Hit@k. A very long erroneous board interval can overlap the gold time; a context with the right key term can still contain the wrong number; an answer can be faithful to corrupted OCR. Use separate labels for source correctness and answer faithfulness. Report retrieval ceiling, evidence retained in the actual prompt, answer correctness conditional on retained evidence, and timestamp support.

The original source-only ablations and no-RAG comparison remain useful baselines, but repeating them is lower priority than testing the new failure mechanisms. A small IUH student study may assess usability, but satisfaction is not an accuracy metric and is not a substitute for gold evidence or judge calibration.

## Prioritised roadmap

Prerequisite for every action: review/freeze a baseline development set and keep a held-out set; run the existing harness without changing production behaviour. Rank reflects likely benefit from documented failures divided by implementation/inference cost, not measured improvement or a claimed paper result.

| Rank | Action / experiment IDs | Why this comes first | Effort | Execution | Main decision evidence |
|---:|---|---|---|---|---|
| 1 | Preserve matched passages and add bounded conditional neighbours — C1/C3 | Retrieval already finds evidence that prefix slicing removes; continuity is a supervisor requirement | 2–3 days | **Offline-CPU** | Context recall/matched-chunk survival rises; citations track the hit; added context stays relevant |
| 2 | Protect headings, assignments, code, formulas and titles — O1 | Repairs known deletion using existing raw OCR | 1–2 days | **Offline-CPU** | Audited facts recover without extra noise; term/code/numeric QA improves |
| 3 | Stop unsafe OCR phrase correction; keep reviewed local terms — A1 | At least 18 harmful edits are already documented | 1–2 days | **Offline-CPU** | Known regressions removed; reviewed edit precision and non-term speech preserved |
| 4 | Repair accented stopwords and phrase-aware BM25 — R1 | Technical query terms currently vanish before retrieval | 1–2 days | **Offline-CPU** | Lexical term retention, BM25 and hybrid Hit@k/MRR |
| 5 | Group duplicate OCR before candidate limits; retune boost/diversity — R2/R3/R4 | Removes documented crowding without replacing the encoder | 3–4 days | **Offline-CPU** | Distinct evidence groups, radium regression, MRR/context recall, correct temporal occurrence |
| 6 | Filter reviewed promotional spans, preserving mixed intervals — S5 | Existing filter is disabled; topic score alone has legitimate false positives | 2–3 days | **Offline-CPU** | Ad retrieval falls; lesson false exclusions remain within a predeclared reviewed limit |
| 7 | Validate OCR/ASR before cache/index, retry only failures — O2/O3/A3 | Failed reads and loops are reused as ordinary evidence | 2–4 days | **Offline-CPU diagnostics + Kaggle-GPU retries** | Detector precision, fewer failed chunks, recovered numeric/content coverage, measured retry cost |
| 8 | Board ROI changes, replacement-safe states, actual reveal onset — S1/S2/S4/C4 | Missing examples and early citations cannot be fixed downstream | 4–7 days | **Offline-CPU video analysis + selective Kaggle-GPU OCR** | State/boundary quality, visual-onset error, readable evidence, affected-question QA |
| 9 | Structured title/code/diagram evidence and reviewed figure indexing — O4/O5 | Figure records are unsearchable and block semantics are damaged | 3–5 days | **Offline-CPU integration + Kaggle-GPU extraction** | Region coverage/provenance, visual/code/numeric QA and context recall |
| 10 | Local-glossary ASR retries; controlled PhoWhisper/model and timing trial — A2/A4/A5 | Tests Bayes/PPF/len errors with real local gold and avoids broad replacement | 3–5 days | **Kaggle-GPU**, CPU annotation/scoring | Term error, non-term WER, English regression, RTF, word/phrase timing error |
| 11 | Bounded reranking after candidate repair and checkpoint verification — R5 | A reranker cannot rescue evidence missing from its candidates | 2–3 days after verification | **Offline-CPU if feasible; otherwise Kaggle-GPU** | MRR/Hit@1 gain over repaired hybrid, candidate recall ceiling, added latency |
| 12 | Routed slide-image answering and calibrated support/citation checks — G1–G4/E3/E4 | Addresses visual questions while retaining a cheaper text route | 3–5 days | **Kaggle-GPU**, CPU validation | Human visual accuracy/claim support, false abstention, accurate citations, judge agreement and latency |

Portable manifest/cache fixes (C5/E6) should accompany relevant actions. Do not run a costly full-corpus extraction until the failure-focused GPU pilot improves held-out evidence/QA quality. Defer graph construction, retriever training, contextual-ASR training, blanket OCR ensembles, and a whole-image retrieval migration until these lower-cost repairs are measured.

## Verification status of supplied leads

The entries below are a coverage ledger, **not a recommended-paper table**. U items are dropped from research evidence; no authors, venue, canonical title, claimed improvement, or hardware suitability is asserted for them. Links with supplied arXiv IDs are lookup destinations that could not be visited successfully.

| Lead group | Locally corroborated | Excluded pending primary-source verification |
|---|---|---|
| Slides/boards | Existing SigLIP and lecture-presentation PDFs only | **U:** SliTraNet `2202.03540`; Kenny et al. ICDAR 2017; Davila & Zanibbi whiteboard work. Older exceptions would need a verified title/record and a clear reason to retain them |
| OCR | Existing Nougat/M3AV/Qwen PDFs; GOT remains an already-rejected comparator | **U:** PaddleOCR-VL `2510.14528`; PaddleOCR-VL-1.5 `2601.21957`; dots.ocr `2512.02498`; olmOCR/olmOCR-Bench; OmniDocBench; any new Vietnamese OCR benchmark |
| Contextual ASR | SlideSpeech and M3AV; the latter verifies RNN-T TCPGen only | **U:** LCB-net `2401.06390`; KG-Whisper `2406.02649`; Whisper prompting study `2406.05806`; rare-word Whisper `2502.11572`; Whisper TCPGen `2306.01942` |
| Hallucination/repetition | Audit evidence and transferable Nougat OCR diagnostics | **U:** Whisper hallucination investigation `2501.11378`; detection `2606.23060`; mechanistic repetition `2508.15882` |
| Vietnamese ASR | PhoWhisper `2406.02555`, local primary PDF | **U:** VietASR `2505.21527`; ChunkFormer ICASSP 2025; Vietnamese large-v3/turbo comparative findings |
| Generative correction | No new primary paper available | **U:** HyPoradise `2309.15701`; GenSEC `2409.09785`; ClozeGER `2405.10025` |
| Chunking/redundancy | Existing LEA and VisRAG evidence, with limitations above | **U:** Reconstructing Context `2504.19754`; Jina late chunking; AdaGReS `2512.25052`; DF-RAG `2601.17212` |
| Vietnamese retrieval | BGE-M3's cross-lingual result only; not a Vietnamese lecture reranker study | **U:** ViRanker `2509.09131`; AITeamVN/Vietnamese_Reranker; legal-retrieval/segmentation `2409.13699` and `2507.14619` |
| Lecture QA / evaluation | Existing RAGAS/LEA/Classroom-RAG/VisRAG/M3DocRAG PDFs | **U:** CourseTimeQA `2512.00360`; From Slides to Chatbots `2510.22272`; EduVidQA EMNLP 2025; Video-MMLU `2504.14693`; ARES and new judge-reliability papers |

When web access is available, verify each U item against arXiv/ACL Anthology before including it: canonical title, author list, first/current version dates, accepted venue if any, inspected abstract versus method/result pages, exact task/dataset/metric/table for any number, and official code/model link. A real identifier alone is insufficient: it must match the proposed topic. For model artifacts, verify license, checkpoint revision, Vietnamese support, dependencies, and a T4 inference pilot. Until then, the incomplete literature coverage should remain explicit in any thesis use of this file.

## Sources

### Primary paper lookup URLs corroborated with local PDFs

These URLs were not successfully fetched live in this session. Bibliographic/content checks used the matching repository PDFs. “P” and “A” readings are distinguished in each module table.

1. SigLIP — https://arxiv.org/abs/2303.15343
2. Multimodal Lecture Presentations — https://arxiv.org/abs/2208.08080
3. Qwen2.5-VL — https://arxiv.org/abs/2502.13923
4. Nougat — https://arxiv.org/abs/2308.13418
5. M3AV — https://arxiv.org/abs/2403.14168
6. GOT — https://arxiv.org/abs/2409.01704 (S reading only)
7. SlideSpeech — https://arxiv.org/abs/2309.05396
8. PhoWhisper — https://arxiv.org/abs/2406.02555
9. LEA — https://arxiv.org/abs/2607.13370
10. Multilingual E5 — https://arxiv.org/abs/2402.05672
11. M3-Embedding/BGE-M3 — https://arxiv.org/abs/2402.03216
12. ColPali — https://arxiv.org/abs/2407.01449
13. VisRAG — https://arxiv.org/abs/2410.10594
14. M3DocRAG — https://arxiv.org/abs/2411.04952
15. RAGAS — https://arxiv.org/abs/2309.15217
16. Classroom comparative RAG — https://arxiv.org/abs/2509.07846

### Code/model destinations printed in inspected local PDFs

Availability, latest revisions, and licenses were not checked live. These are follow-up destinations, not downloaded or tested artifacts.

17. SigLIP/big_vision — https://github.com/google-research/big_vision
18. Lecture presentations dataset — https://github.com/dondongwon/MLPDataset
19. Qwen2.5-VL — https://github.com/QwenLM/Qwen2.5-VL
20. Nougat — https://github.com/facebookresearch/nougat
21. SlideSpeech — https://slidespeech.github.io/
22. PhoWhisper — https://github.com/VinAIResearch/PhoWhisper
23. E5 — https://github.com/microsoft/unilm/tree/master/e5
24. ColPali/ViDoRe — https://hf.co/vidore
25. VisRAG — https://github.com/openbmb/visrag
26. RAGAS — https://github.com/explodinggradients/ragas

### Supplied arXiv lookup destinations: unverified and excluded

These are **not supporting sources for claims** in this document. No search-result snippets were accessible, so there are no claims based on live snippets.

27. https://arxiv.org/abs/2202.03540
28. https://arxiv.org/abs/2510.14528
29. https://arxiv.org/abs/2601.21957
30. https://arxiv.org/abs/2512.02498
31. https://arxiv.org/abs/2401.06390
32. https://arxiv.org/abs/2406.02649
33. https://arxiv.org/abs/2406.05806
34. https://arxiv.org/abs/2502.11572
35. https://arxiv.org/abs/2306.01942
36. https://arxiv.org/abs/2501.11378
37. https://arxiv.org/abs/2606.23060
38. https://arxiv.org/abs/2508.15882
39. https://arxiv.org/abs/2505.21527
40. https://arxiv.org/abs/2309.15701
41. https://arxiv.org/abs/2409.09785
42. https://arxiv.org/abs/2405.10025
43. https://arxiv.org/abs/2504.19754
44. https://arxiv.org/abs/2509.09131
45. https://arxiv.org/abs/2409.13699
46. https://arxiv.org/abs/2507.14619
47. https://arxiv.org/abs/2512.25052
48. https://arxiv.org/abs/2601.17212
49. https://arxiv.org/abs/2512.00360
50. https://arxiv.org/abs/2510.22272
51. https://arxiv.org/abs/2504.14693

### Local audit and implementation sources

- The five supplied `result-1-slides.md`, `result-2-ocr.md`, `result-3-asr.md`, `result-4-align-index.md`, and `result-5-e2e.md` under `C:/Users/nguye/AppData/Local/Temp/claude/C--Users-nguye-OneDrive-M-y-t-nh-Thesis/96f4c665-1d84-4329-8c3e-c7e072f42ac3/scratchpad/codex-pipeline/`.
- `docs/paper/nang-cap-pipeline.md` for completed/rejected prior experiments and earlier proposals.
- Matching PDF files under `docs/paper/` and `docs/paper/edu-rag/`; existing summaries are secondary and may describe older pipeline behaviour, including neighbour expansion that the production audit did not find.
- `pipeline/eval/eval_retrieval.py`, `rebuild_offline.py`, and `offline_common.py` for the actual harness scope. No new evaluation results or model runs were produced for this research document.
