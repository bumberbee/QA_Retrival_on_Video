"""Policy-controlled decisions with explicit fallback for absent model backends."""

import math
from collections.abc import Mapping
from time import perf_counter

from .config import load_policy
from .rules import RulesBackend
from .reranker import RerankerBackend
from .semantic import SemanticBackend, semantic_backend
from .similarity import SimilarityBackend
from .types import BackendResult, Decision, NotConfigured

POLICY = load_policy()


class UnconfiguredBackend:
    def __init__(self, name: str):
        self.name = name

    def run(self, name: str, state: dict):
        raise NotConfigured(f"{self.name} is not configured for {name}")


BACKENDS = {"rules": RulesBackend(), "semantic": SemanticBackend(),
            "reranker": RerankerBackend(POLICY), "similarity": SimilarityBackend(POLICY),
            **{name: UnconfiguredBackend(name) for name in ("nli", "minijev", "jev")}}


def confidence(p: float, n: int, kind: str) -> float:
    if kind == "binary":
        return abs(2 * p - 1)
    if kind != "choice" or n < 2:
        raise ValueError("Choice confidence requires at least two labels")
    return max(0.0, min(1.0, (p - 1 / n) / (1 - 1 / n)))


def zone(p: float, entry: dict) -> str:
    if p >= entry.get("accept", 0.6):
        return "accept"
    if p <= entry.get("reject", 0.2):
        return "reject"
    return "uncertain"


def run_backend(backend, name: str, state: dict) -> BackendResult:
    result = backend.run(name, state)
    if isinstance(result, Mapping):
        result = BackendResult(result["label"], result["p"], dict(result.get("scores", {})))
    if not isinstance(result, BackendResult):
        raise TypeError("Backend must return BackendResult or a label/p mapping")
    p = float(result.p)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Backend probability must be finite and within [0, 1]")
    if not all(math.isfinite(float(s)) and 0 <= float(s) <= 1 for s in result.scores.values()):
        raise ValueError("Backend scores must be finite and within [0, 1]")
    return BackendResult(result.label, p, result.scores)


def decide(name: str, state: dict, *, backend: str | None = None,
           policy=None, backends=None) -> Decision:
    """Use policy, then rules, then a neutral prior when NotConfigured is raised.

    Only NotConfigured triggers fallback; programming/configuration errors surface.
    Binary zones apply to P(first label), not to confidence in a negative label.
    Choice zones apply to the selected label's p (Q8 reranker uses evidence support).
    Uniform priors have confidence 0.
    Optional keyword arguments inject policy/registry without changing global state.
    """
    started = perf_counter()
    policy = POLICY if policy is None else policy
    backends = BACKENDS if backends is None else backends
    entry = policy[name]
    selected = backend or entry["backend"]
    if selected == "reranker" and isinstance(backends.get(selected), RerankerBackend):
        backends = {**backends, "reranker": RerankerBackend(policy)}
    encode = state.get("encode")
    if encode is None and state.get("encoder") is not None:
        encode = state["encoder"].encode
    if selected == "semantic" and encode is not None:
        backends = {**backends, "semantic": semantic_backend(
            encode, entry.get("router", {}).get("semantic_top_k", 3))}
    result = None
    for candidate in dict.fromkeys((selected, "rules")):
        implementation = backends.get(candidate)
        if implementation is None:
            raise KeyError(f"Unknown backend: {candidate}")
        try:
            result = run_backend(implementation, name, state)
            selected = candidate
            break
        except NotConfigured:
            continue
    if result is None:
        selected = "fallback"
        result = BackendResult(entry["fallback_label"], 1 / len(entry["labels"]))
    if result.label not in entry["labels"]:
        raise ValueError(f"Invalid {name} label: {result.label}")
    return Decision(name, result.label, result.p,
                    confidence(result.p, len(entry["labels"]), entry["kind"]),
                    zone(result.p, entry), selected, (perf_counter() - started) * 1000)
