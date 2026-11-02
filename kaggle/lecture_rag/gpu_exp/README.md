Private Kaggle experiments for the Multimodal Lecture RAG thesis. No credentials are read by this code or written to outputs. The locally authenticated Kaggle CLI handles transport.

`build_kernels.py` embeds historical helpers from the completed 20261020 kernel. Generated scripts are self-contained; only `run.py` is executed remotely. `runtime.py`, the runners, `analyze.py`, `review_spans.py`, and `downstream.py` are the editable sources. Raw data belongs under the git-ignored `../runs/gpu/` tree.

```powershell
python kaggle/lecture_rag/gpu_exp/build_kernels.py asr
python kaggle/lecture_rag/gpu_exp/launch_private.py kaggle/lecture_rag/gpu_exp/lecture-rag-gpu-asr-20261101 --timeout 5400
kaggle kernels status thnhtrungnguynmif/lecture-rag-gpu-asr-20261101
kaggle kernels output thnhtrungnguynmif/lecture-rag-gpu-asr-20261101 -p kaggle/lecture_rag/runs/gpu/asr-v1
```

Do not exceed two active kernels. `NvidiaTeslaT4` may allocate two T4s; the runtime records the actual inventory. Only CUDA device 0 performs inference. Bound the sum of allocated-device hours, including failed attempts, to approximately eight hours. Each kernel checkpoints complete videos/images and records installation/inference failures.

The downstream wrapper uses the existing `rebuild_offline.py` and `eval_retrieval.py`, freezes notebook sources under `cpu_snapshot/`, writes only under `runs/gpu/downstream/`, and re-signs only copied stage-03 manifests. The controlled configuration switches ASR correction off for every system. Existing stage-01/02 evidence is held fixed.

```powershell
& backend/.venv/Scripts/python.exe kaggle/lecture_rag/gpu_exp/downstream.py --raw-root kaggle/lecture_rag/runs/gpu/asr-v1/results --systems exported medium large_v3 turbo --prefix controlled-v1
```

Use fresh prefixes: the harness refuses to overwrite staged inputs. Term counts and pairwise disagreement are proxies, not WER. OCR ground truth and retrieval questions both explicitly require human review. `review_spans.py` creates twenty fixed short textual inspection spans per complete ASR system; it does not perform acoustic listening.

The completed findings, settings, limitations and per-version GPU budget are in [GPU_EXPERIMENTS.md](GPU_EXPERIMENTS.md). The final report is generated from `final_report_template.md`, raw-derived analysis JSON and explicitly reviewed `final_review_notes.json`. Kernel directories hold the latest built script; `submitted_versions/vN/` archives the latest Kaggle-returned source. Historical version-suffixed pulls returned HTTP 403 in CLI 2.2.4; older logs/pins are retained but their exact source was not recovered. Editing a runner alone does not change a completed experiment.
