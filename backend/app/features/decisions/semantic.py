"""Top-k cosine intent scorer; inject an already-created E5 .encode method."""

import math
from functools import lru_cache
from threading import Lock

from .config import load_config
from .types import BackendResult, NotConfigured, QTYPES


def unit(vector):
    values = tuple(float(v) for v in vector)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Encoder must return a finite, nonempty vector")
    norm = math.sqrt(sum(v * v for v in values))
    return tuple(v / norm for v in values) if norm else tuple(0.0 for v in values)


class SemanticBackend:
    def __init__(self, encode=None, examples=None, top_k: int = 3):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.encode = encode
        self.examples = examples if examples is not None else load_config("router_examples")
        if set(self.examples) != set(QTYPES) or any(not self.examples[q] for q in QTYPES):
            raise ValueError("Examples must contain all eight question types")
        self.top_k = top_k
        self._vectors = None
        self._lock = Lock()

    def run(self, name: str, state: dict) -> BackendResult:
        if name != "Q4" or self.encode is None:
            raise NotConfigured("Semantic Q4 needs an injected encode(text) callable")
        with self._lock:
            if self._vectors is None:
                self._vectors = {q: [unit(self.encode(text)) for text in examples]
                                 for q, examples in self.examples.items()}
        query = unit(self.encode(state["question"]))
        scores = {}
        for qtype, vectors in self._vectors.items():
            if any(len(v) != len(query) for v in vectors):
                raise ValueError("Encoder vector dimensions differ")
            similarities = sorted((sum(a * b for a, b in zip(query, v)) for v in vectors), reverse=True)
            top = similarities[:self.top_k]
            scores[qtype] = max(0.0, min(1.0, sum(top) / len(top)))
        label = max(QTYPES, key=scores.__getitem__)
        return BackendResult(label, scores[label], scores)


@lru_cache(maxsize=4)
def _cached_backend(encode, top_k):
    return SemanticBackend(encode, top_k=top_k)


def semantic_backend(encode, top_k: int = 3):
    """Reuse example embeddings across questions for an injected encoder."""
    try:
        hash(encode)
    except TypeError:
        return SemanticBackend(encode, top_k=top_k)
    return _cached_backend(encode, top_k)
