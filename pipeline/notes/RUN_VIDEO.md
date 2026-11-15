# Single-video Linux runner

Run from the repository root with Python 3.11, FFmpeg/FFprobe on PATH, and one
RTX 3090/4090 (24 GB). Use a CUDA 12.8 image and a compatible NVIDIA driver;
CTranslate2 4.8.3 requires CUDA 12 cuBLAS and cuDNN 9. The production PyTorch
environment was 2.11.0+cu128. Installation commands below are for the GPU machine;
no packages or models were downloaded during the offline CLI implementation.

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg python3-venv
python3 -m venv pipeline/.venv
source pipeline/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r pipeline/requirements-gpu.txt
# Make pip-installed CUDA libraries visible to the CTranslate2 dynamic loader.
export LD_LIBRARY_PATH="$(python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pipeline.run_video \
  --video /data/lecture.mp4 --video-id lecture_001 \
  --out /work/lecture_001 --language vi --device cuda \
  > /work/lecture_001-progress.jsonl 2> /work/lecture_001.log
```

`--video`, `--video-id`, `--out`, and `--language vi|en|auto` are required.
`--device cuda|cpu` defaults to `cuda`; every GPU stage uses logical device 0
(use `CUDA_VISIBLE_DEVICES` to select a physical GPU). CPU execution retains the
same recipe, with the existing Whisper int8 CPU setting, and is substantially
slower. Paths come from arguments; there is no Kaggle input search or two-GPU
preflight. Keep at least 2 GiB free, and budget additional space for normalized
audio, keyframes, caches, and the Hugging Face model cache.

`--language auto` passes `None` to Whisper's existing language detection and
records its first detected window language. OCR uses the existing English
fallback prompt because OCR precedes ASR; select `vi` for Vietnamese OCR prompts.
Skipped/no-audio auto ASR has a null transcript language, accepted by the importer.
`--skip-ocr` skips both SigLIP and Qwen but retains slide images and empty visual
records. `--skip-asr` writes an explicit skipped transcript without decoding audio
or loading Whisper. Alignment/indexing still run. An explicit skip may produce
an empty `(0, 384)` index; normal runs retain production's no-evidence failure.

Models load only during their stage. All shared model/closure owners are deleted,
garbage collection runs, and `torch.cuda.empty_cache()` runs between stages,
including failures. SigLIP decisions are computed with `gate_keyframe` using the
same content ROIs as OCR, stored in `gate.json`, and consumed after SigLIP unloads.
E5 is loaded for alignment speech relevance and released before index construction.

The builder's configuration and production queue wrappers supply the defaults:
single-pass segmentation at 1/8 Hz with reveal/state merging and per-video ROI
overrides; SigLIP frame+crops threshold 0.15; pinned Qwen3-VL-4B crop OCR, genuine
EOS and repetition guard, exactly one literal-only 2048-token retry; pinned
faster-whisper medium, FP16 on CUDA, windowed VAD and word timestamps; conservative
canonical-subscript OCR cleaning, glossary/code ASR correction, failure guard,
school-signature speech filtering; pinned multilingual E5-small with visual-state
timing and the production description/deduplication policies.

First use resolves model files in the normal Hugging Face cache. For a preloaded
machine, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to require cached weights.
Requirements preserve the builder's exact pins. Its ranges are fixed to versions
recorded in the 20261102 production manifests where available; previously
unversioned hybrid extras have explicit compatible pins. Those extras are not
used by the Qwen recipe. Model IDs/commit revisions are also listed in the
requirements file; SigLIP retains the upstream loader's unpinned revision.

Stdout contains one JSON object per line, with stage names `s01_segment`,
`s02_gate`, `s03_ocr`, `s04_asr`, `s05_align`, `s06_index`. Stage starts have
`status: running, progress: 0`; gate/image/window counts provide intermediate
progress, and finished/skipped stages have progress 1. Logs and tracebacks go to
stderr. The last object is `{"event":"done","export_dir":"..."}` on exit 0,
or `{"event":"error","stage":"...","message":"..."}` on exit 1 (including
preflight/argument failures). Failed exports have no completed marker.

The validated output is `/work/lecture_001/export/lecture_001/`:

```text
01_slides/intervals.json, reveal_audit.json, keyframes/states/*, manifest.json
02_visual/visual.json, content_crops/*, manifest.json
03_audio/transcript.json, audio/lecture_16k.wav, manifest.json
04_bundle/segments.json, transcript_app.json, words.json, keyframes/*, manifest.json
05_index/chunks.jsonl, chunk_ids.json, embeddings.npy, lexical.sqlite3, lecture/*, manifest.json
gate.json, completed.json
```

Optional assets such as content crops/audio exist when applicable. Raw answer and
ASR caches remain under their stage `_cache` directories. Use a separate output
directory per input/configuration; this CLI does not promise notebook queue resume
semantics. `progress.json` and `fast_pipeline_timings.json` live directly under
`--out`. The importer takes the export **parent**, e.g.
`--export /work/lecture_001/export`, then finds each video's unchanged notebook
directories. No renaming to `s01_*` artifact directories is needed.

| CLI stage | Expected GPU memory | Time planning reference |
| --- | --- | --- |
| s01 segment | CPU only | Previous T4 runs: 75–275 s/video, CPU/media dependent |
| s02 gate | Budget ~1 GiB, estimate | Image count dependent; no isolated measurement |
| s03 OCR | ~9.1 GiB measured Qwen crop budget; ~9069 MiB torch allocations in the crop experiment | About 16–17 s/new image on T4; retries can be longer |
| s04 ASR | ~2 GiB sampled CT2 device use (2039 MiB; lower bound on peak) | Medium decode RTF 0.07557: ~136 s for 30 min, excluding load/audio normalization |
| s05 align | Budget <1 GiB for E5, estimate | Previous runs 1–11 s plus separate model loading |
| s06 index | Budget <1 GiB for E5, estimate | Previous runs 1–11 s plus separate model loading |

For a 30-minute lecture with 30 unique OCR images, budget roughly 12–20 minutes
plus initial weights/setup as a conservative T4-derived estimate. The CLI adds
stage times sequentially, whereas Kaggle overlapped ASR/OCR. RTX 3090/4090 latency,
actual stage peaks (including CUDA context and retries), Linux dependency
installation, and live model accuracy/parity require a real GPU validation run.

Offline verification from the repository root:

```bash
python -m pytest pipeline/tests -q -p no:cacheprovider
```

The CLI tests use a tiny FFmpeg MP4 and fake loaders, exercise the real stage
algorithms/manifest validator, and replicate the backend importer's file reader
checks. They never initialize a real OCR, ASR, gate, or embedding model.
