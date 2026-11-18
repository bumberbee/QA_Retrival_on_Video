"""A6: sentence support with a lazy, local-only multilingual ONNX NLI scorer.

Injected scorers return (entailment, neutral, contradiction) probabilities. The
public scoring helpers are also usable offline, independently of the serving flag.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

NLI = os.environ.get("LECTURE_NLI", "off").strip().lower() in {"1", "true", "on", "yes"}
MAX_TOKENS = 512
BATCH_SIZE = 8
_LOAD_LOCK = threading.Lock()
# Shared with verify_claims: retain its sentence/decimal and E-ID behavior.
_SENTENCES = re.compile(r"\n+|(?<=[.!?])\s+(?=[^\d])")
_CITATIONS = re.compile(r"\[(E[1-9]\d*)\]")
_STAMPS = re.compile(r"\[\d+:\d{2}(?::\d{2})?\]")


class NLIUnavailable(RuntimeError):
    """Local NLI assets or inference libraries could not be loaded."""


def enabled() -> bool:
    """Serving flag, read once at startup like LECTURE_RERANK."""
    return NLI


def _file(directory: Path, *names: str) -> Path:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    raise NLIUnavailable(f"Missing local NLI file: {' or '.join(names)}")


def _label_order(directory: Path) -> tuple[int, int, int]:
    config = json.loads(_file(directory, "config.json").read_text(encoding="utf-8"))
    labels = config.get("id2label", {})
    aliases = {"entailment": "entail", "entail": "entail", "neutral": "neutral",
               "contradiction": "contradict", "contradict": "contradict"}
    order = {aliases.get(str(label).strip().lower()): int(index) for index, label in labels.items()}
    if set(order) != {"entail", "neutral", "contradict"} or set(order.values()) != {0, 1, 2}:
        raise NLIUnavailable("config.json id2label must name entailment, neutral and contradiction at IDs 0..2")
    return order["entail"], order["neutral"], order["contradict"]


class _OnnxScorer:
    def __init__(self, directory: Path):
        import onnxruntime
        from tokenizers import Tokenizer

        self.order = _label_order(directory)
        model = _file(directory, "model.onnx", "onnx/model.onnx")
        tokenizer = _file(directory, "tokenizer.json")
        self.session = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.special_tokens = self.tokenizer.num_special_tokens_to_add(is_pair=True)
        if not 0 <= self.special_tokens < MAX_TOKENS:
            raise NLIUnavailable("Tokenizer has an invalid pair special-token budget")
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_id is None:
            self.pad_id = self.tokenizer.token_to_id("[PAD]") or 0
        self.inputs = {item.name for item in self.session.get_inputs()}
        if not self.inputs <= {"input_ids", "attention_mask", "token_type_ids"} or "input_ids" not in self.inputs:
            raise NLIUnavailable("Unsupported ONNX input names")
        outputs = self.session.get_outputs()
        self.output = next((item.name for item in outputs if item.name == "logits"), outputs[0].name)
        self.lock = threading.Lock()

    def __call__(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float, float]]:
        import numpy as np

        with self.lock:
            # Allocate space to the hypothesis first. Only an overlong hypothesis
            # is shortened after the premise has exhausted its available space.
            raw = self.tokenizer.encode_batch([text for pair in pairs for text in pair], add_special_tokens=False)
            encodings = []
            budget = MAX_TOKENS - self.special_tokens
            for premise, hypothesis in zip(raw[::2], raw[1::2]):
                if len(hypothesis.ids) > budget:
                    hypothesis.truncate(budget)
                remaining = budget - len(hypothesis.ids)
                if len(premise.ids) > remaining:
                    premise.truncate(remaining)
                encoding = self.tokenizer.post_process(premise, hypothesis, add_special_tokens=True)
                if len(encoding.ids) > MAX_TOKENS:
                    raise ValueError("NLI tokenized pair exceeds 512 tokens")
                encodings.append(encoding)
            width = max(len(encoding.ids) for encoding in encodings)
            ids = np.full((len(pairs), width), self.pad_id, dtype=np.int64)
            mask, types = np.zeros_like(ids), np.zeros_like(ids)
            for row, encoding in enumerate(encodings):
                size = len(encoding.ids)
                ids[row, :size] = encoding.ids
                mask[row, :size] = encoding.attention_mask
                types[row, :size] = encoding.type_ids
            feed = {name: value for name, value in
                    (("input_ids", ids), ("attention_mask", mask), ("token_type_ids", types)) if name in self.inputs}
            logits = np.asarray(self.session.run([self.output], feed)[0], dtype=np.float64)
        if logits.shape != (len(pairs), 3) or not np.isfinite(logits).all():
            raise ValueError("NLI model must return three finite logits per pair")
        exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        return [tuple(float(row[index]) for index in self.order) for row in probabilities]


@lru_cache(maxsize=1)
def _load_default_scorer():
    try:
        explicit = os.environ.get("LECTURE_NLI_MODEL_DIR")
        if not explicit:
            raise NLIUnavailable("Provision LECTURE_NLI_MODEL_DIR first")
        directory = Path(explicit).expanduser().resolve()
        if not directory.is_dir():
            raise NLIUnavailable("LECTURE_NLI_MODEL_DIR must be an existing local directory")
        return _OnnxScorer(directory)
    except Exception as exc:
        raise NLIUnavailable("Cannot load local ONNX NLI assets/libraries; no download attempted") from exc


def default_scorer():
    """Load once on first use; a failed load remains retryable after provisioning."""
    with _LOAD_LOCK:
        return _load_default_scorer()


def score_pairs(pairs, *, scorer=None) -> list[tuple[float, float, float]]:
    pairs = list(pairs)
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2
           or any(not isinstance(text, str) for text in pair) for pair in pairs):
        raise ValueError("NLI pairs must contain premise and hypothesis strings")
    if not pairs:
        return []
    scorer = default_scorer() if scorer is None else scorer
    result = []
    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start:start + BATCH_SIZE]
        probabilities = list(scorer(batch))
        if len(probabilities) != len(batch):
            raise ValueError("NLI scorer must return one probability triple per pair")
        for values in probabilities:
            try:
                row = tuple(float(value) for value in values)
            except (TypeError, ValueError) as exc:
                raise ValueError("NLI scorer must return probability triples") from exc
            if (len(row) != 3 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in row)
                    or not math.isclose(sum(row), 1.0, abs_tol=1e-5)):
                raise ValueError("NLI scorer must return normalized, finite probability triples")
            result.append(row)
    return result


def split_claims(answer: str) -> list[dict]:
    """Split without changing sentence text; citation IDs are unique in first-use order."""
    return [{"claim_id": f"C{index + 1}", "text": sentence,
             "evidence_ids": list(dict.fromkeys(_CITATIONS.findall(sentence)))}
            for index, sentence in enumerate(part.strip() for part in _SENTENCES.split(answer.strip()) if part.strip())]


def thresholds(*, accept=None, reject=None) -> tuple[float, float]:
    from app.features.decisions.config import load_policy

    entry = load_policy()["Q10"] if accept is None or reject is None else {}
    accept = float(entry.get("accept", 0.6) if accept is None else accept)
    reject = float(entry.get("reject_contradict", 0.5) if reject is None else reject)
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (accept, reject)):
        raise ValueError("NLI thresholds must be finite and within [0, 1]")
    return accept, reject


def status(entail: float, contradict: float, *, accept: float, reject: float) -> str:
    return "supported" if entail >= accept else "contradicted" if contradict >= reject else "unsupported"


def found_from_evidence_map(evidence_map: dict) -> dict:
    """Tool/decision text maps have no retrieval slides; adapt to evidence_texts."""
    return {"evidence_map": evidence_map,
            "context_evidence": [{"evidence_id": key, "text": source.get("text", "")}
                                 for key, source in evidence_map.items()]}


def check_claims(answer: str, found: dict, *, scorer=None, accept=None, reject=None) -> dict:
    """Max support/contradiction across cited premises; uncited/inferred use all.

    An explicit missing citation has no premise, rather than being repaired with
    unrelated evidence. ``reject`` is Q10.reject_contradict, not its zone boundary.
    ``premise_chars`` is the total source characters submitted for that claim.
    """
    from app.features.ask.postprocess import evidence_texts

    accept, reject = thresholds(accept=accept, reject=reject)
    texts = evidence_texts(found)
    inferred = {source["id"] for source in (found.get("cited_sources") or [])
                if source.get("id") and source.get("inferred")}
    inferred.update(key for key, source in (found.get("evidence_map") or {}).items() if source.get("inferred"))
    rows, pairs, spans = [], [], []
    for claim in split_claims(answer):
        cited = [key for key in claim["evidence_ids"] if key not in inferred]
        ids = [key for key in cited if key in texts] if cited else list(texts)
        hypothesis = _STAMPS.sub("", _CITATIONS.sub("", claim["text"])).strip()
        start = len(pairs)
        pairs.extend((texts[key], hypothesis) for key in ids)
        spans.append((start, len(pairs)))
        rows.append({**claim, "evidence_ids": ids, "premise_chars": sum(len(texts[key]) for key in ids)})
    scores = score_pairs(pairs, scorer=scorer)
    counts = dict(total=len(rows), supported=0, unsupported=0, contradicted=0)
    for row, (start, end) in zip(rows, spans):
        row["entail"] = max((values[0] for values in scores[start:end]), default=0.0)
        row["contradict"] = max((values[2] for values in scores[start:end]), default=0.0)
        row["status"] = status(row["entail"], row["contradict"], accept=accept, reject=reject) if end > start else "unsupported"
        counts[row["status"]] += 1
    return {"claims": rows, "counts": counts, "accept": accept, "reject_contradict": reject}
