# A3: offline reranker measurement (pending)

Implementation and fake-scorer tests do not establish a retrieval improvement.
**Neither the evaluator nor a model/GPU run was launched. Ask the user before
running evaluation; GPU approval is required for each proposed GPU run.** A3
mentions approximately 0.6 T4-hour remaining; verify the current budget first.

## Runtime contract and local assets

- Backend flag: `LECTURE_RERANK=on` before process startup; default off.
- Model: `BAAI/bge-reranker-v2-m3`, batches of eight, joint question/text length
  limited to 512 tokens, sigmoid of one logit per pair, top 30 after RRF.
- Installed backend packages checked: ONNX Runtime 1.30.0 and tokenizers 0.23.2;
  no torch, transformers or sentence-transformers in `backend/.venv`.
- Set `LECTURE_RERANK_MODEL_DIR` to a pre-existing local snapshot/export. ONNX
  needs `onnx/model.onnx` (or `model.onnx`), all external tensor data beside it,
  and `tokenizer.json` (or `onnx/tokenizer.json`). A local Transformers snapshot
  is usable only if those libraries are already installed.
- `RERANK_REVISION` has an explicit commit-hash TODO. Fill it with the reviewed
  upstream hash before using a cached Hub snapshot or publishing measurements;
  no moving `main` revision is substituted. Explicit local directories should
  be audited against the same hash and recorded with their file checksums.
- No model, metadata or package downloads. Missing assets/libraries raise
  `RerankerUnavailable`. The E5 encoder in the evaluator also resolves only local
  files/cache using the existing pinned E5 revision.

## Paired dev-120 protocol

`pipeline/eval/rerank_eval.py` rebuilds **retrieval results**, using existing
`<video>/05_index` exports; it never rebuilds or writes indexes. Both arms share
the same chunks, SQLite FTS, vectors, E5 query vectors and retrieval options.
The only treatment is `RERANK=False` versus `True`, before slide selection.
Retained context uses the existing production builder, including deduplication,
bounded windows, slide diversity and temporal rules.

The JSON and Markdown reports contain Hit@1, Hit@3, MRR@3 and context success,
per-question paired deltas, mean deltas, counts improved/regressed, per-video
and per-type summaries. Context success uses `context_trace`: gold time overlap
AND all annotated key terms in retained text from the requested evidence source.
Unanswerable questions are excluded from retrieval metric denominators, but
retain their best scores for later A8 threshold calibration. MRR stops at rank 3.
Input/source checksums, model revision constants, scores, contexts and traces
are recorded. The dev bank is still labeled as needing human review; preserve
that status. Do not use the held-out test bank or tune thresholds on it.

After approval, a local CPU command from the repository root is:

```powershell
$env:PYTHONPATH = (Resolve-Path backend/.venv/Lib/site-packages).Path
python pipeline/eval/rerank_eval.py --index-root <cached-export-root> --encoder-dir <local-e5-snapshot> --reranker-dir <local-bge-snapshot> --name dev120-rerank
```

Reports default to `pipeline/runs/exp/dev120-rerank/{rerank.json,rerank.md}`.
`--index-dir` with one `--videos` value supports a smoke subset of dev-120;
use all ten videos for the final paired comparison. Optionally supply
`--query-vectors <cache.npz>` containing `vectors` (N x D), `question_ids`
(ordered Unicode IDs for the selected questions), scalar Unicode `model`,
`revision` and `questions_sha256` (hash of the entire question-bank file).
Cache metadata must match exactly; arrays are read with `allow_pickle=False`.

## Kaggle notebook builder

Build the notebook locally without executing any cells:

```powershell
python pipeline/eval/build_rerank_notebook.py pipeline/notebooks/rerank_eval.ipynb
```

This builder embeds the evaluator, isolated backend loader, retrieval/reranker
modules, source-path map and dev questions in a three-cell `.ipynb`. It does
not import the HTTP app, agent slice, dotenv or cloud adapters. It never launches,
uploads or publishes a Kaggle job. Do not reuse the older confirmation builder's
online package-install setup for this experiment.

Attach already provisioned datasets containing the cached ten-video stage-05
indexes (including `lecture/segments.json` and available `lecture/words.json`),
the pinned local E5 ONNX snapshot, and the audited BGE model files. Disable
Internet. Edit the notebook input paths; outputs are written to
`/kaggle/working/rerank-results`. Required packages must already be installed;
the notebook stops with a missing-package error and never installs anything.

CPU is the default and uses the same ONNX-preferred scorer as the backend.
The optional GPU cell uses a local Transformers cross-encoder on CUDA with the
same batching, raw logits, sigmoid and 512-token pair limit. **Ask the user and
obtain approval before enabling a T4 session or executing GPU cells.** Only then
set `USE_GPU=True` and `GPU_RUN_APPROVED=True`. Provisioning/exporting weights is
a separate task; no export, package install or download is included here.

Budget estimate, not measured: at most 3,600 pairs for 120 questions. Allow
**5–10 minutes on one T4** for reranking (0.08–0.17 T4-hour), plus up to 5 minutes
for startup/E5 encoding (total 0.08–0.25 T4-hour). Use one GPU; disable it promptly
after saving reports. CPU estimate from A3's 1–3 seconds per 20 pairs is roughly
3–10 minutes for scoring, plus model load/E5 encoding, with potentially much
longer times on constrained CPUs. Measure actual cold/warm latency in the run.

Remaining: audit/pin/provision model files, obtain approval for the chosen run,
execute the paired comparison, review regressions and record measured resource
costs. A8 must calibrate Q8/Q9 thresholds on dev before they support refusal claims.
