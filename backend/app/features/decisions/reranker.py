"""Q8/Q9 decisions from existing reranker scores; this backend never loads a model."""

import math
from collections.abc import Mapping

from .config import load_policy
from .types import BackendResult, NotConfigured


class RerankerBackend:
    def __init__(self, policy=None):
        self.policy = load_policy() if policy is None else policy

    def run(self, name: str, state: dict) -> BackendResult:
        if name == "Q8":
            score = state.get("best_rerank")
            # Pass the search result directly or nest it in retrieval for agent callers.
            if score is None and isinstance(state.get("retrieval"), Mapping):
                score = state["retrieval"].get("best_rerank")
        elif name == "Q9":
            score = state.get("rerank_score")
            if score is None and isinstance(state.get("chunk"), Mapping):
                score = state["chunk"].get("rerank_score")
        else:
            raise NotConfigured(f"Reranker does not implement {name}")
        if score is None:
            raise NotConfigured(f"No reranker score supplied for {name}")
        if isinstance(score, bool):
            raise ValueError("Reranker score must be a finite number within [0, 1]")
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Reranker score must be finite and within [0, 1]")
        entry = self.policy[name]
        if name == "Q8":
            label = ("enough" if score >= entry["accept"] else
                     "irrelevant" if score <= entry["reject"] else "insufficient")
        else:
            label = "keep" if score >= entry["accept"] else "drop"
        # Q8 p is evidence support: a sigmoid is not a calibrated class probability.
        return BackendResult(label, score)
