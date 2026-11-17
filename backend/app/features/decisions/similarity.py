"""Interim Q8 from the best E5 cosine of a search, for when no reranker score exists."""

import math
from collections.abc import Mapping

from .config import load_policy
from .types import BackendResult, NotConfigured

DEFAULTS = {"accept": 0.85, "reject": 0.84}


class SimilarityBackend:
    """Thresholds live under Q8.similarity in the policy (pipeline/notes/Q8_SIMILARITY.md).

    p is the cosine itself (evidence support), not a calibrated class probability,
    and the policy's reranker accept/reject zones do not apply to it: use the label.
    """

    def __init__(self, policy=None):
        self.policy = load_policy() if policy is None else policy

    def run(self, name: str, state: dict) -> BackendResult:
        if name != "Q8":
            raise NotConfigured(f"Similarity does not implement {name}")
        score = state.get("best_semantic")
        if score is None and isinstance(state.get("retrieval"), Mapping):
            score = state["retrieval"].get("best_semantic")
        if score is None or isinstance(score, bool):
            raise NotConfigured("No E5 similarity supplied for Q8")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("E5 similarity must be finite")
        score = max(0.0, min(1.0, score))
        thresholds = {**DEFAULTS, **self.policy["Q8"].get("similarity", {})}
        label = ("enough" if score >= thresholds["accept"] else
                 "irrelevant" if score <= thresholds["reject"] else "insufficient")
        return BackendResult(label, score)
