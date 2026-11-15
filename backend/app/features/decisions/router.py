"""A2b intent cascade. Routes are fixed Python data, never supplied by an LLM."""

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter

from .base import BACKENDS, POLICY, confidence, run_backend
from .rules import DEICTIC, contains, normalize, previous_type
from .semantic import semantic_backend
from .types import Action, BackendResult, Modality, MODALITIES, NotConfigured, QuestionType, QTYPES

ROUTES = {
    "chitchat": ("direct", "reply_template"),
    "out_of_scope": ("direct", "refusal_with_hint"),
    "definition": ("run", "plan_definition"),
    "formula": ("run", "plan_formula"),
    "explanation": ("run", "plan_explanation"),
    "comparison": ("run", "plan_comparison"),
    "locate_time": ("run", "plan_locate"),
    "summary": ("run", "plan_summary"),
}
CLARIFICATIONS = {
    "missing_reference": "Bạn đang hỏi về slide nào / phút nào?",
    "ambiguous_types": "Bạn muốn hỏi theo hướng nào? Vui lòng làm rõ yêu cầu.",
    "multiple_videos": "Bạn muốn hỏi về video nào? Phạm vi hiện tại chỉ gồm một video.",
}


@dataclass
class QuestionDecision:
    qtype: QuestionType
    modality: Modality
    action: Action
    p: float
    decided_by: str
    confidence: float = 0.0
    steps: list[str] = field(default_factory=list)
    step_timings_ms: dict[str, float] = field(default_factory=dict)
    clarify_reason: str | None = None


def ask_back(decision: QuestionDecision) -> tuple[str, str]:
    return "clarify", CLARIFICATIONS.get(decision.clarify_reason, CLARIFICATIONS["ambiguous_types"])


def resolve_route(decision: QuestionDecision) -> tuple[str, str]:
    route = ROUTES[decision.qtype]  # Unknown types always raise KeyError.
    return ask_back(decision) if decision.action == "clarify" else route


def _clarify_reason(question: str, state: dict) -> str | None:
    text = normalize(question)
    previous_question = state.get("previous_question", "")
    has_previous = bool((previous_question.strip() if isinstance(previous_question, str) else previous_question)
                        or state.get("previous_decision") or previous_type(state))
    # A complete question containing "this algorithm" is not an unanchored pointer.
    tokens = set(re.findall(r"[^\W_]+", text))
    pointer_words = set("cai nay do cho doan slide luc la gi no vay the nao this that it here there what is explain please".split())
    if (contains(text, DEICTIC) and tokens <= pointer_words
            and state.get("current_time") is None and not has_previous):
        return "missing_reference"
    single_video = state.get("scope", "single_video") == "single_video"
    mentioned = state.get("mentioned_video_ids", [])
    multiple = (len(set(mentioned)) > 1 or bool(re.search(
        r"\b(?:(?:hai|2|nhieu|cac)\s+(?:video|bai giang)|(?:both|two|multiple)\s+videos)\b", text)))
    if single_video and multiple:
        return "multiple_videos"
    return None


def _heuristic(question: str, state: dict) -> BackendResult:
    terms = state.get("lecture_terms", [])
    terms = [term for term in terms if isinstance(term, str) and term.strip()]
    if not terms:
        raise NotConfigured("Lecture heuristic needs lecture_terms")
    text = normalize(question)
    if contains(text, terms):
        definition = contains(text, ("meaning", "term", "nghia", "khai niem")) or len(text.split()) <= 8
        label = "definition" if definition else "explanation"
        return BackendResult(label, 0.8, {label: 0.8})
    rrf = state.get("top1_rrf")
    threshold = state["_router_config"]["rrf_threshold"]
    if rrf is not None and math.isfinite(float(rrf)) and 0 <= float(rrf) < threshold:
        return BackendResult("out_of_scope", 0.8, {"out_of_scope": 0.8})
    raise NotConfigured("Lecture heuristic has no clear conclusion")


def _llm_result(question: str, state: dict, llm) -> tuple[BackendResult, str]:
    default = BackendResult("explanation", 1 / len(QTYPES))
    if llm is None:
        return default, MODALITIES[default.label]
    try:
        payload = llm(question, state)
        if isinstance(payload, str):
            payload = json.loads(payload)
    except (json.JSONDecodeError, NotConfigured):
        return default, MODALITIES[default.label]
    if not isinstance(payload, Mapping) or payload.get("qtype") not in QTYPES:
        return default, MODALITIES[default.label]
    label = payload["qtype"]
    modality = payload.get("modality")
    if modality not in ("ocr", "speech", "both"):
        modality = MODALITIES[label]
    try:
        p = float(payload.get("p", 0.5))  # No claimed certainty for an uncalibrated LLM.
        if not math.isfinite(p) or not 0 <= p <= 1:
            p = 0.5
    except (TypeError, ValueError):
        p = 0.5
    return BackendResult(label, p), modality


def route_question(question: str, state: dict, *, encode=None, llm=None,
                   backends=None, policy=None) -> QuestionDecision:
    """Rules -> optional mini-Jev/Jev -> semantic -> lecture heuristic -> LLM.

    Inject encode(text) (e.g. an existing search encoder.encode), or a configured
    SemanticBackend via backends. llm(question, state) returns a JSON string/dict.
    Neither dependency is created automatically. See README.md for state keys.
    """
    state = {**state, "question": question}
    policy = POLICY if policy is None else policy
    config = {**POLICY["Q4"]["router"], **policy["Q4"].get("router", {})}
    state["_router_config"] = config
    registry = {**BACKENDS, **(backends or {})}
    encode = encode if encode is not None else state.get("encode")
    if encode is None and state.get("encoder") is not None:
        encode = state["encoder"].encode
    if encode is not None:
        registry["semantic"] = semantic_backend(encode, config["semantic_top_k"])
    llm = llm if llm is not None else state.get("llm")
    steps, timings, informative = [], {}, []

    def measured(name, callback):
        started = perf_counter()
        steps.append(name)
        try:
            return callback()
        except NotConfigured:
            return None
        finally:
            timings[name] = (perf_counter() - started) * 1000

    def gap(result):
        others = [score for label, score in result.scores.items() if label != result.label]
        return result.p - max(others, default=0.0)

    def record(result):
        if result.label not in QTYPES or any(label not in QTYPES for label in result.scores):
            raise ValueError("Q4 backend returned an unknown question type")
        if result.scores and max(result.scores.values()) > 0:
            candidates = sorted(result.scores, key=result.scores.__getitem__, reverse=True)
            close = (len(candidates) > 1 and gap(result) + 1e-12 < config["margin_threshold"]
                     and ROUTES[candidates[0]] != ROUTES[candidates[1]])
            informative.append(close)

    def finish(result, name, *, modality=None, reason=None):
        action = "clarify" if reason else ROUTES[result.label][0]
        return QuestionDecision(result.label, modality or MODALITIES[result.label], action,
                                result.p, name, confidence(result.p, 8, "choice"),
                                list(steps), dict(timings), reason)

    def rules_step():
        reason = _clarify_reason(question, state)
        if reason:
            return BackendResult("explanation", 0.95), reason
        return run_backend(registry["rules"], "Q4", state), None

    outcome = measured("rules", rules_step)
    if outcome is not None:
        result, reason = outcome
        record(result)
        if reason:
            return finish(result, "rules", reason=reason)
        if (result.p >= 0.95 and gap(result) + 1e-12 >= config["margin_threshold"]
                and confidence(result.p, 8, "choice") >= config["confidence_threshold"]):
            return finish(result, "rules")

    models = state.get("q4_backends")
    if models is None:
        selected = state.get("q4_backend", policy["Q4"]["backend"])
        models = (selected,) if selected in ("minijev", "jev") else ("minijev", "jev")
    if isinstance(models, str):
        models = (models,)
    for name in dict.fromkeys(models):
        if name not in ("minijev", "jev"):
            raise ValueError("q4_backends can only contain minijev and jev")
        result = measured(name, lambda: run_backend(registry[name], "Q4", state))
        if result is not None:
            record(result)
            if confidence(result.p, 8, "choice") >= config["confidence_threshold"]:
                return finish(result, name)

    result = measured("semantic", lambda: run_backend(registry["semantic"], "Q4", state))
    if result is not None:
        record(result)
        if (result.label in result.scores and len(result.scores) >= 2
                and result.p >= config["semantic_threshold"]
                and gap(result) + 1e-12 >= config["margin_threshold"]):
            return finish(result, "semantic")

    result = measured("heuristic", lambda: _heuristic(question, state))
    if result is not None:
        record(result)
        if confidence(result.p, 8, "choice") >= config["heuristic_threshold"]:
            return finish(result, "heuristic")

    result, modality = measured("llm", lambda: _llm_result(question, state, llm))
    # Earlier ties do not force clarify if a later scoring step separates the types.
    reason = "ambiguous_types" if informative and all(informative) else None
    return finish(result, "llm", modality=modality, reason=reason)
