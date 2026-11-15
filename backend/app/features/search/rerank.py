"""Local-only, lazy BGE cross-encoder scoring. Importing this module loads no model."""

from __future__ import annotations

import importlib.util
import math
import os
import threading
from functools import lru_cache
from pathlib import Path

RERANK_REPO = "BAAI/bge-reranker-v2-m3"
# TODO A3: fill the reviewed model commit hash before resolving a Hub cache snapshot.
# Never silently resolve "main". An explicitly provisioned local directory also works.
RERANK_REVISION = "TODO_FILL_REVIEWED_COMMIT_HASH"
MAX_TOKENS = 512
BATCH_SIZE = 8
_LOAD_LOCK = threading.Lock()


class RerankerUnavailable(RuntimeError):
    """Installed inference libraries or locally provisioned model files are unavailable."""


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _model_directory() -> Path:
    explicit = os.environ.get("LECTURE_RERANK_MODEL_DIR")
    if explicit:
        directory = Path(explicit).expanduser().resolve()
        if not directory.is_dir():
            raise RerankerUnavailable("LECTURE_RERANK_MODEL_DIR must be an existing local directory")
        return directory
    if len(RERANK_REVISION) != 40 or any(c not in "0123456789abcdef" for c in RERANK_REVISION):
        raise RerankerUnavailable("Pin RERANK_REVISION or provision LECTURE_RERANK_MODEL_DIR first")
    from huggingface_hub import snapshot_download

    # This only resolves an existing cache. It must never fetch weights or metadata.
    return Path(snapshot_download(RERANK_REPO, revision=RERANK_REVISION, local_files_only=True))


def _file(directory: Path, *names: str) -> Path:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing local reranker file: {' or '.join(names)}")


class _OnnxScorer:
    def __init__(self, directory: Path):
        import onnxruntime
        from tokenizers import Tokenizer

        model = _file(directory, "onnx/model.onnx", "model.onnx")
        tokenizer = _file(directory, "tokenizer.json", "onnx/tokenizer.json")
        # External ONNX tensor data, if used, must already be beside this model.
        self.session = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS, strategy="longest_first")
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_id is None:
            self.pad_id = self.tokenizer.token_to_id("[PAD]") or 0
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.output = next((item.name for item in self.session.get_outputs() if item.name == "logits"),
                           self.session.get_outputs()[0].name)
        self.lock = threading.Lock()

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        import numpy as np

        with self.lock:
            encodings = self.tokenizer.encode_batch(pairs)
            width = max(len(e.ids) for e in encodings)
            ids = np.full((len(pairs), width), self.pad_id, dtype=np.int64)
            mask, types = np.zeros_like(ids), np.zeros_like(ids)
            for row, encoding in enumerate(encodings):
                n = len(encoding.ids)
                ids[row, :n] = encoding.ids
                mask[row, :n] = encoding.attention_mask
                types[row, :n] = encoding.type_ids
            feed = {name: value for name, value in
                    (("input_ids", ids), ("attention_mask", mask), ("token_type_ids", types))
                    if name in self.inputs}
            logits = np.asarray(self.session.run([self.output], feed)[0])
        if logits.size != len(pairs):
            raise ValueError("BGE reranker must return one logit per pair")
        return logits.reshape(-1).tolist()


class _TransformersScorer:
    def __init__(self, directory: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                                                       trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False).eval()
        self.lock = threading.Lock()

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        with self.lock, self.torch.inference_mode():
            inputs = self.tokenizer([q for q, _ in pairs], [text for _, text in pairs],
                                    padding=True, truncation=True, max_length=MAX_TOKENS,
                                    return_tensors="pt")
            logits = self.model(**inputs).logits
        if logits.numel() != len(pairs):
            raise ValueError("BGE reranker must return one logit per pair")
        return logits.reshape(-1).cpu().tolist()


class _SentenceTransformersScorer:
    def __init__(self, directory: Path):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(str(directory), max_length=MAX_TOKENS, device="cpu",
                                  local_files_only=True, trust_remote_code=False)
        self.lock = threading.Lock()

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        from torch.nn import Identity

        # CrossEncoder normally applies a sigmoid; rerank() owns that transformation.
        with self.lock:
            values = self.model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False,
                                        activation_fn=Identity(), convert_to_numpy=True)
        if values.size != len(pairs):
            raise ValueError("BGE reranker must return one logit per pair")
        return values.reshape(-1).tolist()


@lru_cache(maxsize=1)
def _load_default_scorer():
    try:
        directory = _model_directory()
    except Exception as exc:
        raise RerankerUnavailable("No locally provisioned BGE reranker; no download attempted") from exc
    failures = []
    for modules, implementation in [(("onnxruntime", "tokenizers"), _OnnxScorer),
                                    (("torch", "transformers"), _TransformersScorer),
                                    (("torch", "sentence_transformers"), _SentenceTransformersScorer)]:
        try:
            if all(_installed(name) for name in modules):
                return implementation(directory)
        except Exception as exc:
            failures.append(exc)
    error = RerankerUnavailable("No installed backend could load the local BGE reranker")
    if failures:
        raise error from failures[-1]
    raise error


def default_scorer():
    """Construct once on first nonempty use; failures remain retryable after provisioning."""
    with _LOAD_LOCK:
        return _load_default_scorer()


def _sigmoid(logit: float) -> float:
    # Avoid overflow for confident negative logits. The endpoints may round to 0 or 1.
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    value = math.exp(logit)
    return value / (1.0 + value)


def rerank(question: str, candidates: list[dict], *, scorer=None,
           top_k: int | None = None) -> list[dict]:
    """Copy candidates, batch question/text pairs, sigmoid logits and rank from one.

    Ties retain input order. Injected scorers receive raw text and return logits;
    the default backend tokenizes each pair jointly with a 512-token total limit.
    Missing local dependencies/weights raise RerankerUnavailable, never download.
    """
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
        raise ValueError("top_k must be a nonnegative integer or None")
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    for candidate in candidates:
        if "chunk_id" not in candidate or not isinstance(candidate.get("text"), str):
            raise ValueError("Every candidate needs chunk_id and string text")
    if not candidates or top_k == 0:
        return []
    scorer = default_scorer() if scorer is None else scorer
    scored = []
    for start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[start:start + BATCH_SIZE]
        logits = list(scorer([(question, c["text"]) for c in batch]))
        if len(logits) != len(batch):
            raise ValueError("Scorer must return one logit per pair")
        for candidate, value in zip(batch, logits):
            logit = float(value)
            if not math.isfinite(logit):
                raise ValueError("Scorer logits must be finite")
            scored.append({**candidate, "rerank_score": _sigmoid(logit)})
    scored.sort(key=lambda c: -c["rerank_score"])
    for rank, candidate in enumerate(scored, 1):
        candidate["rerank_rank"] = rank
    return scored if top_k is None else scored[:top_k]
