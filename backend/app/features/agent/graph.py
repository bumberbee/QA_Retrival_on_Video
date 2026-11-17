"""A5: the fixed path S1 -> S2 -> S3 <-> S4 -> S5 -> S6 | S7. Code decides every step.

No model is called here. S5 (generation) stays in ``features.ask``; this module either
hands it the evidence to answer from, or returns a final text (a template reply, a
clarifying question or a refusal) that needs no generation at all. S6 (``verify``) is
run by the caller on the generated answer.

``LECTURE_AGENT=on`` enables it for single-video questions; the default is the baseline.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from functools import partial

from app.features.ask.llm import REFUSAL
from app.features.ask.postprocess import stamp
from app.features.decisions import decide, route_question
from app.features.decisions.router import CLARIFICATIONS, ROUTES
from app.features.decisions.rules import DEICTIC, contains, normalize
from app.features.search import retrieval

from .tools import Budget, BudgetLimits, ToolContext, default_registry

CHITCHAT_MAX_WORDS = 6   # a longer "chào ..." sentence is treated as a real question
REPLIES = {
    "thanks": "Rất vui được giúp bạn. Bạn cứ hỏi tiếp về bài giảng nhé.",
    "hello": "Chào bạn! Mình có thể trả lời các câu hỏi về nội dung bài giảng này.",
}
THANKS = ("cam on", "thanks", "thank you")


def enabled() -> bool:
    return os.getenv("LECTURE_AGENT", "").strip().lower() in {"1", "true", "on", "yes"}


@dataclass
class AgentResult:
    """action "answer": generate from found/context. "direct": text is the final reply."""

    action: str
    found: dict = field(default_factory=dict)
    context: str = ""
    text: str | None = None
    refusal: bool = False
    trace: dict = field(default_factory=dict)
    budget: Budget | None = None      # the question's budget, for the LLM calls of S5/S6


def _time(value) -> float | None:
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _refusal_text(found: dict) -> str:
    """The fixed refusal plus the closest part of the lecture, when a search ran."""
    slides = found.get("slides") or []
    if not slides:
        return REFUSAL
    top = slides[0]
    where = f"[{stamp(top['start_sec'])}]"
    title = " ".join((top.get("title") or "").split())
    hint = f"Phần gần nhất trong bài là «{title}» tại {where}." if title else f"Phần gần nhất mình tìm được ở {where}."
    return f"{REFUSAL} {hint}"


def grade(found: dict) -> dict:
    """Q8: reranker scores when the search has them, otherwise the E5 cosine."""
    decision = decide("Q8", found)
    if decision.backend == "fallback":
        decision = decide("Q8", found, backend="similarity")
    label = decision.label
    if found.get("temporal") and found.get("slides"):
        label = "enough"   # "this slide" is answered from the slide being watched
    return {"label": label, "p": round(decision.p, 4), "backend": decision.backend}


def run(question: str, *, video_id: str, query: str | None = None, current_time: float | None = None,
        previous_question: str | None = None, prefer: frozenset = frozenset(), rewritten: str | None = None,
        encoder=None, registry=None, searcher=None, limits: BudgetLimits | None = None) -> AgentResult:
    """question: what the user typed. query: the text the baseline would search with.

    encoder (an object with .encode) enables the router's semantic step; without it the
    router uses rules only. searcher replaces retrieval.search in tests.
    """
    started = time.perf_counter()
    budget = Budget(limits)
    trace = {"states": [], "tools": []}

    def enter(state: str) -> None:
        trace["states"].append(state)

    def finish(result: AgentResult) -> AgentResult:
        trace["budget"] = budget.snapshot()
        trace["seconds"] = round(time.perf_counter() - started, 4)
        result.trace, result.budget = trace, budget
        return result

    # ---- S1 understand --------------------------------------------------------------
    enter("S1_understand")
    current_time = _time(current_time)
    text = normalize(question)
    points_at_video = any(p in retrieval.normalize_search(question) for p in retrieval.TEMPORAL) or contains(text, DEICTIC)
    state = {"previous_question": previous_question or "", "scope": "single_video"}
    # The player always reports a time; it is a signal only when the question points at the video.
    if points_at_video and current_time is not None:
        state["current_time"] = current_time
    decision = route_question(rewritten or question, state, encode=encoder.encode if encoder is not None else None)
    trace["decision"] = {key: value for key, value in asdict(decision).items()
                         if key in {"qtype", "modality", "action", "p", "decided_by", "confidence", "steps",
                                    "step_timings_ms", "clarify_reason"}}

    # ---- S2 plan --------------------------------------------------------------------
    enter("S2_plan")
    action, target = ROUTES[decision.qtype]
    # Two close question types are not worth a question back: every run plan starts with the
    # same search here. Only a missing reference or a second video needs the user.
    uncertain = decision.clarify_reason == "ambiguous_types"
    words = re.findall(r"[^\W_]+", text)
    if decision.qtype == "chitchat" and (uncertain or len(words) > CHITCHAT_MAX_WORDS):
        action, target = "run", "plan_explanation"
    # Only a sure rule hit may refuse before searching; any other "out of scope" is checked against retrieval.
    verify_scope = decision.qtype == "out_of_scope" and (uncertain or decision.decided_by != "rules")
    if verify_scope:
        action, target = "run", "plan_scope_check"
    clarify = decision.action == "clarify" and not uncertain
    trace["plan"] = "clarify" if clarify else target
    if clarify:
        return finish(AgentResult("direct", text=CLARIFICATIONS[decision.clarify_reason]))
    if action == "direct" and target == "reply_template":
        return finish(AgentResult("direct", text=REPLIES["thanks" if contains(text, THANKS) else "hello"]))
    if action == "direct":
        enter("S7_refuse")
        return finish(AgentResult("direct", text=REFUSAL, refusal=True))

    # ---- S3 search <-> S4 grade -------------------------------------------------------
    registry = registry or default_registry()
    context = ToolContext(video_id=video_id, current_time=current_time, budget=budget,
                          searcher=partial(searcher or retrieval.search, prefer_segments=prefer))
    queries = list(dict.fromkeys(q for q in (query or rewritten or question, question) if q and q.strip()))
    best, verdict = None, None
    for round_number, text_query in enumerate(queries, 1):
        enter("S3_search")
        result = registry.run("search_hybrid", {"q": text_query}, context)
        trace["tools"].append({"name": "search_hybrid", "round": round_number, "ok": result.ok,
                               "error": result.error, "latency_ms": round(result.latency_ms, 1)})
        if not result.ok:
            break   # budget exhausted or the search failed: answer from what there is
        found = result.data["found"]
        enter("S4_grade")
        graded = grade(found)
        trace.setdefault("q8", []).append({"round": round_number, **graded})
        if best is None or graded["p"] > verdict["p"]:
            best, verdict = found, graded
        if graded["label"] == "enough":
            break
    if best is None:
        # No search result at all: let the baseline path report the failure as it always did.
        return finish(AgentResult("answer"))
    if verdict["label"] == "irrelevant" or (verify_scope and verdict["label"] != "enough"):
        enter("S7_refuse")
        return finish(AgentResult("direct", found=best, text=_refusal_text(best), refusal=True))
    enter("S5_answer")
    evidence_context, _ = retrieval.build_context_with_ids(best)
    trace["verified"] = verdict["label"] == "enough"
    return finish(AgentResult("answer", found=best, context=evidence_context))
