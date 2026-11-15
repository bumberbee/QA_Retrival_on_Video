"""Small bilingual rules. They are routing hints, not a complete safety filter."""

import re
import unicodedata

from .types import BackendResult, MODALITIES, NotConfigured, QTYPES


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def contains(text: str, phrases) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(normalize(p)) + r"(?!\w)", text)
               for p in phrases)


INJECTION = (
    r"\b(?:ignore|disregard|forget)\b.{0,50}\b(?:previous|prior|system|instructions?)\b",
    r"\b(?:bo qua|quen|phớt lờ)\b.{0,50}\b(?:lenh|huong dan|chi dan|quy tac)\b",
    r"\b(?:reveal|print|show|tiet lo|hien thi|in ra)\b.{0,50}\b(?:system prompt|api key|mat khau|khoa api)\b",
    r"\b(?:system prompt|developer message|jailbreak|dan mode)\b",
)
TOXIC = ("ngu ngoc", "do ngu", "may ngu", "khon nan", "dit me", "fuck you", "idiot", "kill yourself")
NON_ACADEMIC = ("gia bitcoin", "du bao thoi tiet", "weather forecast", "football score",
                "mua co phieu", "buy stocks", "dat do an", "order food")
ACADEMIC = ("dinh nghia", "cong thuc", "bai giang", "xac suat", "thuat toan", "dinh ly",
            "khai niem", "lecture", "theorem", "algorithm", "probability", "definition")
Q4_PHRASES = {
    "definition": ("la gi", "dinh nghia", "khai niem", "what is", "what are", "define", "definition"),
    "formula": ("bao nhieu", "tinh", "cong thuc", "how many", "how much", "calculate", "compute", "formula"),
    "explanation": ("tai sao", "vi sao", "giai thich", "nhu the nao", "how does", "how do", "why", "explain"),
    "comparison": ("khac gi", "so sanh", "khac nhau", "compare", "comparison", "difference", "versus", "vs"),
    "locate_time": ("luc nay", "slide nay", "doan nay", "cho nay", "hien tai", "phut nao",
                    "o dau", "this slide", "current slide", "just now", "which minute", "where in the video"),
    "summary": ("tom tat", "tong ket", "y chinh", "summarize", "summary", "recap", "main points"),
    "out_of_scope": NON_ACADEMIC + ("ngoai bai", "khong lien quan bai giang", "unrelated to the lecture"),
    "chitchat": ("chao", "cam on", "tam biet", "hello", "hi", "thanks", "thank you", "goodbye"),
}
TIME = re.compile(r"\b(?:phut|minute|minutes|at)\s*\d+\b|\b\d{1,3}:\d{2}(?::\d{2})?\b|\b\d+\s*(?:phut|minutes?)\b")
NUMBER = r"[+-]?\d+(?:[.,]\d+)?"
NUMERIC = re.compile(
    rf"(?:\d|\b[a-z]\b|\))\s*(?:=|≈|≃|<=|>=|≤|≥)\s*(?:{NUMBER}|\b[a-z]\b)|"
    rf"{NUMBER}\s*(?:[+*/×÷^−-]|=)\s*{NUMBER}|"
    rf"{NUMBER}\s*(?:%|phan tram|percent\b)|"
    rf"\b(?:bang|equals?|is)\s*{NUMBER}\b|"
    rf"\b(?:[pc]\s*\(\s*\d+\s*,\s*\d+\s*\)|\d+!)"
)
DEICTIC = ("cai nay", "cai do", "cho do", "cho nay", "doan nay", "slide nay", "luc nay",
           "this", "that", "it", "here", "there")


def previous_type(state: dict):
    previous = state.get("previous_decision")
    if isinstance(previous, dict):
        previous = previous.get("qtype")
    elif previous is not None:
        previous = getattr(previous, "qtype", None)
    return state.get("previous_qtype", previous)


def q4_rules(question: str, state: dict) -> BackendResult:
    from app.features.search.retrieval import is_follow_up

    prior = previous_type(state)
    if prior in QTYPES and is_follow_up(question):
        return BackendResult(prior, 0.95, {prior: 0.95})
    text = normalize(question)
    matches = {label for label, phrases in Q4_PHRASES.items() if contains(text, phrases)}
    # Numbers that are timestamps support locate_time rather than formula.
    temporal = bool(TIME.search(text))
    if temporal or state.get("current_time") is not None:
        matches.add("locate_time")
    if not temporal and re.search(r"\d", text):
        matches.add("formula")
    if len(matches) == 1:
        label = next(iter(matches))
        return BackendResult(label, 0.95, {label: 0.95})
    if matches:
        scores = {label: 1 / len(matches) for label in QTYPES if label in matches}
        label = "explanation" if "explanation" in matches else next(iter(scores))
        return BackendResult(label, scores[label], scores)
    return BackendResult("explanation", 1 / len(QTYPES))


class RulesBackend:
    def run(self, name: str, state: dict) -> BackendResult:
        question = state.get("question", "")
        text = normalize(question)
        if name == "Q1":
            if not re.search(r"[^\W_]+", text) or not re.search(r"[^\W\d_]", text):
                return BackendResult("invalid", 0.05)
            if contains(text, NON_ACADEMIC):
                return BackendResult("invalid", 0.05)
            if contains(text, ACADEMIC) or any(contains(text, Q4_PHRASES[q]) for q in QTYPES[:6]):
                return BackendResult("valid", 0.95)
            return BackendResult("valid", 0.5)
        if name == "Q2":
            injection = any(re.search(normalize(p), text) for p in INJECTION)
            toxic = contains(text, TOXIC)
            label = "toxic_injection" if toxic and injection else "toxic" if toxic else "injection" if injection else "safe"
            return BackendResult(label, 0.95)
        if name == "Q3":
            from app.features.search.retrieval import is_follow_up
            follow_up = is_follow_up(question)
            return BackendResult("follow_up" if follow_up else "standalone", 0.95 if follow_up else 0.05)
        if name == "Q4":
            return q4_rules(question, state)
        if name == "Q5":
            qtype = state.get("qtype")
            decision = state.get("question_decision")
            if qtype is None and decision is not None:
                qtype = decision.get("qtype") if isinstance(decision, dict) else decision.qtype
            if qtype is None:
                result = q4_rules(question, state)
                if result.p < 0.6:
                    return BackendResult("both", 1 / 3)
                qtype = result.label
            return BackendResult(MODALITIES[qtype], 0.95)
        if name == "Q7":
            temporal = (state.get("current_time") is not None or bool(TIME.search(text))
                        or contains(text, Q4_PHRASES["locate_time"]))
            return BackendResult("temporal" if temporal else "not_temporal", 0.95 if temporal else 0.05)
        if name == "Q12":
            claim = normalize(str(state.get("answer", "")))
            claims = state.get("claims", [])
            if isinstance(claims, str):
                claims = [claims]
            claim = " ".join([text, claim, *(normalize(str(c)) for c in claims)])
            numeric = bool(NUMERIC.search(claim))
            return BackendResult("numeric_claim" if numeric else "no_numeric_claim", 0.95 if numeric else 0.05)
        raise NotConfigured(f"No rule implemented for {name}")
