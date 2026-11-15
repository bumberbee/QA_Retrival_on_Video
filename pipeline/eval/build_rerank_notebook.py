"""Build a self-contained offline Kaggle notebook. Building never runs evaluation."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

CONFIG = '''from pathlib import Path
# Attach pre-existing indexes, E5 ONNX files and BGE weights as Kaggle inputs.
# Internet OFF. No pip install or model download is part of this notebook.
INDEX_ROOT = Path("/kaggle/input/cached-lecture-indexes")  # <video>/05_index
ENCODER_DIR = Path("/kaggle/input/local-e5-small")  # onnx/model.onnx + onnx/tokenizer.json
RERANKER_DIR = Path("/kaggle/input/local-bge-reranker-v2-m3")
QUERY_CACHE = None  # optional audited NPZ cache, matching the selected question IDs
OUTPUT_ROOT = Path("/kaggle/working/rerank-results")
RUN_NAME = "dev120-bge-rerank"
USE_GPU = False
# Set True only after the user explicitly approves this specific GPU run.
GPU_RUN_APPROVED = False
'''

RUNTIME = '''import importlib.util, os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.dont_write_bytecode = True
if USE_GPU and not GPU_RUN_APPROVED:
    raise RuntimeError("Ask the user and obtain approval before running on a GPU")
required = ["numpy", "onnxruntime", "tokenizers", "huggingface_hub"]
if USE_GPU:
    required += ["torch", "transformers"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError(f"Missing preinstalled packages: {missing}; no installation attempted")
for path in (INDEX_ROOT, ENCODER_DIR, RERANKER_DIR):
    if not path.is_dir():
        raise FileNotFoundError(f"Attach the pre-existing local input: {path}")
base = Path("/kaggle/working/rerank-eval-repo")
for name, source in files.items():
    target = base / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
sys.path.insert(0, str(base / "pipeline/eval"))
scorer = None
if USE_GPU:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if not torch.cuda.is_available():
        raise RuntimeError("Approved GPU run requested but CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(str(RERANKER_DIR), local_files_only=True,
                                             trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(RERANKER_DIR), local_files_only=True, trust_remote_code=False).to("cuda").eval()
    def scorer(pairs):
        inputs = tokenizer([q for q, _ in pairs], [text for _, text in pairs],
                           padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        return logits.reshape(-1).float().cpu().tolist()
    scorer.backend_name = "local-only Transformers CUDA " + torch.cuda.get_device_name(0)
import rerank_eval
sys.argv = ["rerank_eval.py", "--index-root", str(INDEX_ROOT), "--encoder-dir", str(ENCODER_DIR),
            "--reranker-dir", str(RERANKER_DIR), "--name", RUN_NAME, "--output-root", str(OUTPUT_ROOT)]
if QUERY_CACHE is not None:
    sys.argv += ["--query-vectors", str(QUERY_CACHE)]
rerank_eval.main(scorer=scorer)
'''


def build(destination):
    files = [ROOT / "eval" / name for name in
             ("offline_common.py", "eval_retrieval.py", "rerank_eval.py", "questions_dev.json")]
    files += [ROOT / "shared/source_paths.py"]
    files += [REPO / "backend/app/features/search" / name for name in ("retrieval.py", "rerank.py")]
    payload = {p.relative_to(REPO).as_posix(): p.read_text(encoding="utf-8") for p in files}
    sources = [("markdown", "# Offline paired dev-120 reranking\n\n"
                "Attach cached indexes and local model files. Disable Internet. No installation/download.\n\n"
                "CPU is the default. Ask the user before any GPU run; expected T4 time: 5–10 minutes "
                "(unmeasured estimate), plus up to 5 minutes for startup/E5 encoding.\n\n"
                "Edit input paths below, then run. Reports go to /kaggle/working. See RERANK_PLAN.md."),
               ("code", CONFIG), ("code", "files = " + repr(payload) + "\n" + RUNTIME)]
    cells = []
    for number, (kind, source) in enumerate(sources):
        cell = {"id": f"rerank-{number}", "cell_type": kind, "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            compile(source, "rerank-notebook-cell", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(build(parser.parse_args().destination))
