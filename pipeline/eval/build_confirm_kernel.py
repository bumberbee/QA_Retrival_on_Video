"""Build one private CPU kernel with small eval sources embedded, no input re-upload."""
import argparse
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from shared.source_paths import LEGACY_GOLD, module_path
SLUG = "lecture-rag-confirm-cpu-eval-20261104"
OWNER = "thnhtrungnguynmif"


def build(destination):
    files = [ROOT / "eval" / name for name in
             ("offline_common.py", "eval_retrieval.py", "legacy_offline.py", "confirm_kaggle.py", "questions_dev.json")]
    files += [ROOT / "shared/source_paths.py"]
    files += [module_path(name) for name in ("common.py", "indexing.py")]
    files += [REPO / "backend/app/features/search" / name for name in ("retrieval.py", "rerank.py")]
    payload = {str(p.relative_to(REPO)).replace("\\", "/"): p.read_text(encoding="utf-8") for p in files}
    payload["pipeline/eval/questions_legacy.json"] = (
        LEGACY_GOLD / "qa/questions.json").read_text(encoding="utf-8")
    preamble = '''import os, sys, subprocess, runpy
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.dont_write_bytecode = True
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime", "tokenizers", "huggingface-hub"], check=True)
base = Path("/kaggle/temp/confirm-eval/repo")
'''
    code = preamble + "files = " + repr(payload) + "\n"
    code += '''for name, source in files.items():
    target = base / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
sys.path.insert(0, str(base / "pipeline/eval"))
runpy.run_path(str(base / "pipeline/eval/confirm_kaggle.py"), run_name="__main__")
'''
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "run.py").write_text(code, encoding="utf-8")
    metadata = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [],
                "kernel_sources": [f"{OWNER}/lecture-rag-fast-10-confirm-20261102"]}
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    compile(code, "run.py", "exec")
    print("Built", metadata["id"], "bytes", len(code.encode("utf-8")), "private CPU, one kernel source")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    build(parser.parse_args().destination)
