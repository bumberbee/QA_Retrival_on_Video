# Production extraction port v3

2026-11-02 · `upgrade/extraction-fixes` · prepared locally for the next ten-video run.

## Changes and source pointers

| Change | Source |
|---|---|
| Global state onset export and content identity merging; audited board/editor options only | [build_fast_pipeline.py:55](../builders/build_fast_pipeline.py#L55), detector call at [line 231](../builders/build_fast_pipeline.py#L231) |
| Pinned Qwen recipe, content crops, pixel bounds, validation and retry controls | [build_fast_pipeline.py:105](../builders/build_fast_pipeline.py#L105), OCR call at [line 244](../builders/build_fast_pipeline.py#L244) |
| Pure EOS/loop validation and exactly one retry | [visual.py:714](../stages/s03_ocr/visual.py#L714), [retry orchestration:740](../stages/s03_ocr/visual.py#L740) |
| Crop selection and pinned FP16/SDPA model adapter | [visual.py:779](../stages/s03_ocr/visual.py#L779), [loader:804](../stages/s03_ocr/visual.py#L804) |
| Attempt provenance/cache fingerprint and failed OCR count | [visual.py:922](../stages/s03_ocr/visual.py#L922), [manifest:1181](../stages/s03_ocr/visual.py#L1181) |
| Keep stage-02 failures through phase-3 cleaning and `gpu_retry_segments.json` | [alignment.py:262](../stages/s05_align/alignment.py#L262), [retry records:321](../stages/s05_align/alignment.py#L321) |
| Carry state evidence windows and copy their primary images into the stage-04 bundle | [alignment.py:96](../stages/s05_align/alignment.py#L96), [asset copying:382](../stages/s05_align/alignment.py#L382) |
| Cite representative visual state onset; preserve group-wide speech ownership | [indexing.py:102](../stages/s06_index/indexing.py#L102), [alignment.py:53](../stages/s05_align/alignment.py#L53) |
| Experiment-pinned turbo and conservative ASR batch setting | [build_fast_pipeline.py:135](../builders/build_fast_pipeline.py#L135), [ASR call:261](../builders/build_fast_pipeline.py#L261) |
| Reuse one E5 instance for speech relevance and indexing on GPU 1 | [build_fast_pipeline.py:224](../builders/build_fast_pipeline.py#L224), [index call:310](../builders/build_fast_pipeline.py#L310) |
| Single-video defaults reuse production config; generic stage notebooks enable onset/content merging and turbo | [build_full_walkthrough.py:85](../builders/build_full_walkthrough.py#L85), [build_notebooks.py:35](../builders/build_notebooks.py#L35), [ASR config:143](../builders/build_notebooks.py#L143) |
| Focused validation, retry, model adapter, config, failure/cache and citation tests | [test_port_v3.py:37](../tests/test_port_v3.py#L37); updated Qwen stub in [test_pipeline.py:996](../tests/test_pipeline.py#L996) |
| Prevent backend tests from opening `.env`, including older dotenv versions | [backend/tests/_isolate.py:10](../../backend/tests/_isolate.py#L10) |

The implementation lives in maintainable Python. All nine primary notebooks under `notebooks/` were regenerated with the existing builders; eight changed, while stage 06 remained byte-identical. `build_run_pipeline.py` also refreshes its existing ignored local run copy. No notebook was edited by hand. `detection.py`, `transcription.py` and the phase-3 text algorithms did not need edits.

## Defaults and behavior

| Config | Production default |
|---|---|
| `OCR_ENGINE` | `qwen` |
| `QWEN_OCR_MODEL` | `Qwen/Qwen3-VL-4B-Instruct` |
| `QWEN_OCR_REVISION` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| `QWEN_CONTENT_CROP` | `True` |
| `QWEN_CONTENT_ROIS` | Board `hVU5Mw19V_c`: `(0,.08,1,.96)`; editor `y1CvB950VlU`: `(.25,.05,.95,.76)` |
| `QWEN_MIN_PIXELS`, `QWEN_MAX_PIXELS` | `256*32*32` / `1280*32*32` = 262,144 / 1,310,720 |
| `QWEN_VALIDATE`, `QWEN_RETRY_INVALID` | Both `True` |
| `EXPORT_STATE_ONSETS`, `MERGE_SAME_CONTENT` | Both `True` |
| `ASR_MODEL` | `dropbox-dash/faster-whisper-large-v3-turbo` |
| `ASR_REVISION` | `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` |
| `ASR_MODE`, `ASR_BATCH_SIZE` | `windowed`, `1` (batch size is unused in windowed mode) |
| `ASR_DEVICE_INDEX`, `EMBEDDING_DEVICE` | `1`, `cuda:1` for Qwen/hybrid; GPU 0 for EasyOCR |
| `ASR_CORRECTION`, `FAILURE_GUARD` | `glossary_code`, `True` |

Qwen first prefers a configured segment/video crop, then the existing configured document border, then the final state's saved display ROI expanded by 2% per side to restore its bezel margin. It preserves the full frame when no conservative crop exists. The saved crop, normalized ROI and `ocr_crop_method` make the choice inspectable. Tight numeric crops from the focused experiment are not used as whole-slide crops. EasyOCR and hybrid retain their existing paths.

The first generation uses the existing language-specific slide prompt plus literal code/quote/indentation instructions, greedy decoding, 1,536 output tokens and repetition penalty 1.05. A valid output must end in a genuine generated EOS token and contain fewer than eight consecutive repetitions of any 1–32-word phrase. Empty output also fails. One invalid generation gets exactly one literal-only retry with 2,048 tokens and penalty 1.10. Vietnamese retry instructions match the experiment; English slides get the equivalent English instructions. Generation exceptions are also retained and receive at most one retry.

Every processed image retains both raw attempts, prompt hashes, EOS/loop results, token counts, generation settings, model revision and image/grid dimensions in `ocr_generation`. If the retry fails, its record remains in `visual.json` with `ocr_status='failed'` and phase-3-compatible `ocr_quality` (`status`, `reasons`, `evidence_excluded`). The manifest/timing row reports the failure. Alignment preserves this status, writes a GPU retry provenance entry, and excludes the failed text/regions from retrieval evidence. Cache identity includes the pin, crops, bounds, validation recipe and retry settings, so old unvalidated Qwen outputs are not reused as validated results.

Stage 01 uses the exact scoped settings from [STAGE01.md](../eval/segmentation/STAGE01.md): board sampling/masking/snapshots and complementary keyframes only for `hVU5Mw19V_c`, with reveal merging disabled there; strict replacement safety and editor ROI scoring only for `y1CvB950VlU`. Other videos retain the ordinary reveal predicate and original detection thresholds. The generic stage-01 notebook keeps board/editor heuristics off. The single-video walkthrough/run builders support the same audited scoping as production.

Alignment retains `representative_content_onset_sec`, `visual_evidence_windows` and the long-dynamic review flag. OCR/visual/region chunks start at the representative state's onset with `timing_precision='visual_state'`; old exports fall back to their group interval. For the documented `cs_0001` case, new visual evidence starts near 117.83 s instead of 34.73 s. Speech still uses its original word spans and group ownership; no per-line OCR timing is invented.

The turbo pin is verified offline in [build_kernels.py:23](build_kernels.py#L23) and [returned turbo metrics](../runs/gpu/asr-v1/results/turbo/hVU5Mw19V_c/metrics.json). No fallback repository or unverified revision is used. ASR retains beam 5, temperature 0, VAD 500 ms / 250 ms padding, word timestamps, disabled previous-text conditioning and an empty initial prompt. Phase-3 loop exclusion, raw-word provenance and `glossary_code` corrections remain enabled. The experiment's glossary/hotword/fallback ASR ablation was not adopted.

## T4 memory and time estimates

The existing loader selects FP16 on CUDA and int8 on CPU ([transcription.py:94](../stages/s04_asr/transcription.py#L94)). The five-video experiment measured turbo at 2,199 MiB sampled device memory versus medium at 2,039 MiB. Enhanced Qwen peaked at 9,069.15 MiB of torch allocations. Together with a single E5-small instance, these leave several GiB on a 16 GB T4 for contexts/activations. One ASR window and one Qwen image run per worker; the optional ASR batch size is explicitly 1. Indexing now uses the same model/revision/device cache key as speech relevance, avoiding an extra E5 instance on another GPU. No compute-type reduction was necessary. Concurrent peak memory, especially a 2,048-token retry, remains unmeasured; the existing per-GPU sampler is retained.

Timing evidence comes from [GPU_EXPERIMENTS.md](GPU_EXPERIMENTS.md) and the local returned `runs/gpu/asr-v1/results/{medium,turbo}/*/metrics.json` files. These estimates are inference execution time, excluding downloads/loading/setup and shared-GPU contention.

| Per 30-minute video | Estimate and basis |
|---|---|
| Medium ASR | **136.02 s**, scaled from 569.34 s decode / 7,534.27 s audio |
| Turbo ASR | **73.35 s**, scaled from 307.01 s decode / the same audio |
| Turbo minus medium | **−62.67 s**, about 46.1% less decoding; −0.01741 execution GPU-hours |
| OCR retry overhead at observed enhanced failure rate | **0 s**: all 29 enhanced outputs validated; no retry was executed |
| OCR retry planning estimate | **~22.4 s per retry**, scaling measured 16.787 s/image by 2,048/1,536 tokens; this is an extrapolation, not a measured retry |
| Example: ~66 representative images, 1% invalid first attempts | **~15 s extra** |
| Same workload, 5% invalid first attempts | **~74 s extra** |
| Stress assumption: old whole-frame rate 4/29 | **~204 s (~3.4 min) extra**; this is not the enhanced recipe's measured failure rate |

The representative-image example uses Stage 01's 613 grouped intervals / 279.4 video minutes × 30 ≈ 66, before OCR gate/hash/near-duplicate reuse. Actual slide density matters more than video length. Reading every state plus board complements in a future extension would raise the planning workload to about 99 observations/30 minutes (924/279.4×30), and retry overhead proportionally. Greedy retries that hit the token cap can be slower than the linear estimate. Whole-frame failed attempts took 119.76–142.89 s in the experiment; a retry can extend an already slow first attempt.

Enhanced OCR's CER was **0.33755 → 0.31033**, with loops/invalid outputs **4/29 → 0/29**. Its 16.787 vs 32.171 s/image aggregate gain largely reflects avoided failures: on the 25 originally valid images, means were **15.5069 vs 15.5602 s**. Do not assume a 48% full-run OCR speedup. Two allocated T4s running for the same elapsed time count double the execution-time GPU-hours; ASR runs concurrently with OCR, so the 63-second decoding saving need not reduce total pipeline wall time by 63 seconds.

## Verification

Builders run locally, in order:

```powershell
python pipeline/builders/build_notebooks.py
python pipeline/builders/build_fast_pipeline.py
python pipeline/builders/build_full_walkthrough.py
python pipeline/builders/build_run_pipeline.py
```

Required checks, from repository root with `D:/FilmTools/Shotcut` added to PATH and HF/Transformers offline settings:

| Command | Final result |
|---|---|
| `python -m pytest pipeline/tests -q` (system Python) | **136 passed, 23 subtests passed in 20.66 s** |
| `backend/.venv/Scripts/python.exe -m unittest discover -s backend/tests` | **100 tests run in 2.027 s: 98 passed, 2 skipped, OK** |

The two backend skips require a local Postgres/pgvector service. `TEST_DB_URL` was explicitly set to localhost for this run. Cloud settings are blanked by the existing test isolation; dotenv is now explicitly patched before app imports because the installed package ignores `PYTHON_DOTENV_DISABLED`. Final tests do not use Supabase, R2 or Kaggle. Existing notebook syntax/schema/embedded-source checks and production text-harness config parity pass. Focused tests exercise the real adapter through stubs, including pins/pixels/FP16/EOS and the exact retry settings, with no model download.

## Remaining risks

- **Turbo retrieval tradeoff:** the experiment itself recommends keeping medium for its stronger controlled context support. This requested switch is intentional: turbo's Hit@1 was 15/24 versus medium's 17/24, and context recall 20/24 versus 23/24. Speed and some code/PPF improvements do not establish an overall accuracy gain. The interaction with final phase-3 corrections needs full-run evaluation; no human ASR WER or acoustic alignment validation is claimed.
- **Crop/retry generalization:** broad OCR references were AI-drafted, and audit crops were manually chosen. Automatic expanded-screen selection and the scoped production crop boxes are not a corpus-level OCR benchmark. A conservative full-frame fallback prevents guessing on unknown layouts, but edge clipping, perspective views and presenter occlusion remain possible. English retry behavior and actual failed-retry timings were not measured by the winning experiment.
- **State coverage:** stage 02 still reads one representative image per grouped interval. Earlier reveal states and board complements are preserved as observations but not separately OCRed. Representative onset fixes early citations; it can delay citations for text already present in earlier reveals and does not recover text erased before the final frame. Per-state OCR and per-line onset attribution remain future work.
- **Board/editor limits:** the scoped green-board mask and editor replacement heuristic remain unlabelled for content-state precision. Some short board states are occluded; persistent IDE popups can still enter OCR. Unmatched lecturer-only/perspective exam views still cause extra short groups. Extra board groups increase OCR time.
- **Shared GPU/runtime:** sampled CT2 memory is a lower bound and torch peak allocations omit some device overhead. No concurrent T4 test, model download or next ten-video execution was performed in this port. Existing dependency version ranges remain; compatibility with a changed Kaggle image is not verified offline.

No commit, push, branch switch or Kaggle action was performed. The unrelated deleted `.mcp.json` and `AGENTS.md`, and pre-existing unrelated untracked files, remain untouched.
