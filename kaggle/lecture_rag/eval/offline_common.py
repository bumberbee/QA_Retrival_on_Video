"""Isolated access to the production notebook and backend functions."""

import os

for _name in set(os.environ) | {
    "SUPABASE_DB_URL", "SUPABASE_URL", "SUPABASE_ANON_KEY",
    "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
}:
    if _name.startswith(("SUPABASE_", "R2_")):
        os.environ[_name] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
EXPERIMENTS = ROOT / "runs" / "exp"
VIDEOS = ["cs_0001", "D_kuulxKuxc", "hVU5Mw19V_c", "L25_V008", "L25_V011",
          "L25_V037", "tHE1mC_n0-E", "v2YTz8ToEag", "y1CvB950VlU", "_eDGTHhyQ7M"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def experiment_dir(name):
    path = (EXPERIMENTS / name).resolve()
    if not path.is_relative_to(EXPERIMENTS.resolve()) or path == EXPERIMENTS.resolve():
        raise ValueError("Experiment name must stay inside runs/exp.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_notebook(name):
    # Production embeds these source files into one namespace; they are not standalone modules.
    module = types.ModuleType("offline_notebook_" + name)
    extras = ["textclean", "adfilter"] if name == "alignment" else []
    for filename in ["common", *extras, name]:
        path = ROOT / "notebook_code" / (filename + ".py")
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def load_backend():
    # Do not import app, storage, models, main, dotenv, supa or objects, even transiently.
    name = "_lecture_eval_backend"
    if name + ".services.retrieval" in sys.modules:
        return sys.modules[name + ".services.retrieval"]
    for package in (name, name + ".services"):
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module
    storage = types.ModuleType(name + ".storage")
    storage.DATA_DIR = EXPERIMENTS / "_unused_data"
    storage.cloud = lambda: False
    storage.load_video_meta = lambda video: None
    sys.modules[storage.__name__] = storage
    path = REPO / "backend" / "app" / "services" / "retrieval.py"
    spec = importlib.util.spec_from_file_location(name + ".services.retrieval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def production_config():
    tree = ast.parse((ROOT / "build_fast_pipeline.py").read_text(encoding="utf-8"))
    config = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in node.targets))
    values = {}
    for node in ast.parse(config).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return {
        "ad_suspect_overlap": values["AD_SUSPECT_OVERLAP"],
        "speech_relevance": values["SPEECH_RELEVANCE"],
        "slide_speech": values["SLIDE_SPEECH"] if values["AD_FILTER"] == "slide_speech" else None,
        "embedding_model": values["EMBEDDING_MODEL"], "revision": values["EMBEDDING_REVISION"],
        "dedupe_previous": values["OCR_DEDUPE_PREVIOUS"],
        "conservative_ocr": values["CONSERVATIVE_OCR"],
        "heading_titles": values["HEADING_TITLES"],
        "asr_correction": values["ASR_CORRECTION"],
        "failure_guard": values["FAILURE_GUARD"],
        "speech_ad_filter": values["SPEECH_AD_FILTER"],
        "chunk_tokens": 350, "overlap_tokens": 40, "batch_size": 32,
        "speech_chunk_tokens": 180, "speech_maximum_seconds": 50.0,
        "include_generated_descriptions": values["INCLUDE_GENERATED_DESCRIPTIONS"],
        "generated_semantic_only": values["GENERATED_SEMANTIC_ONLY"],
    }


class TokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text, add_special_tokens=False, **kwargs):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, tokens, skip_special_tokens=True):
        text = self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        # PreTrainedTokenizerBase.clean_up_tokenization, enabled by production XLM-R.
        for before, after in [(" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","),
                              (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"),
                              (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")]:
            text = text.replace(before, after)
        return text


class ModelAdapter:
    """Only the SentenceTransformer interface consumed by run_indexing, backed by real ONNX."""

    max_seq_length = 512

    def __init__(self, encoder):
        from tokenizers import Tokenizer
        self.encoder = encoder
        # Chunk splitting must not inherit the query encoder's 512-token truncation.
        tokenizer = Tokenizer.from_str(encoder.tokenizer.to_str())
        tokenizer.no_truncation()
        self.tokenizer = TokenizerAdapter(tokenizer)

    def _first_module(self):
        return types.SimpleNamespace(auto_model=types.SimpleNamespace(
            config=types.SimpleNamespace(_commit_hash=load_backend().E5_REVISION)))

    def encode(self, texts, batch_size=32, **kwargs):
        import numpy as np
        # embedding_text already contains exactly one production passage prefix.
        return np.concatenate([self.encoder.encode_many(texts[i:i + batch_size], prefix="")
                               for i in range(0, len(texts), batch_size)])


def encode_batches(encoder, texts, prefix, batch_size=32):
    import numpy as np
    return np.concatenate([encoder.encode_many(texts[i:i + batch_size], prefix=prefix)
                           for i in range(0, len(texts), batch_size)])
