"""Deterministic repairs of a generated answer. Every repair is a switch, default off.

``ASK_STRIP_REFUSAL=on``  drop a leading refusal sentence that is followed by an answer,
                          only when retrieval found evidence that matches the question.
``ASK_ATTRIBUTE=on``      when the model cited no evidence ID, attach the closest evidence
                          block to each sentence by E5 cosine (``ASK_ATTRIBUTE_MIN``).

Inferred citations are marked ``inferred`` so they are never counted as model citations.
"""

import os
import re

STRONG_SEMANTIC = 0.85        # keep equal to the "weak match" boundary in llm._evidence_prompt
DEFAULT_ATTRIBUTE_MIN = 0.84  # TODO A8: uncalibrated; measure on dev before reporting numbers
MIN_TAIL = 40                 # characters after a refusal that count as "an answer follows"

_LEAD = " \t\r\n\"“”'*"
_CONNECTOR = re.compile(r"^(?:tuy nhiên|tuy vậy|nhưng|dù vậy|mặc dù vậy|song)\s*,?\s*", re.I)
_STAMP = re.compile(r"\[\d+:\d{2}(?::\d{2})?\]")
_SENTENCE = re.compile(r"\n+|(?<=[.!?…])\s+(?=\S)")
_TRAILING = re.compile(r"[\s.!?…:;,\"”']*$")


def flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "on", "yes"}


def stamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def strong_evidence(found: dict) -> bool:
    return bool(found.get("temporal")) or found.get("best_semantic", 0.0) >= STRONG_SEMANTIC


class RefusalStripper:
    """Streaming filter: holds a leading refusal until it is known whether an answer follows.

    A refusal with nothing (or only a few words) after it is passed through unchanged.
    """

    def __init__(self, refusal: str, active: bool):
        self.core = refusal.rstrip(".").casefold()
        self.state = "head" if active and self.core else "pass"
        self.buffer = ""
        self.stripped = False

    def _split(self) -> tuple[str, str] | None:
        """(held refusal, tail) once the buffer starts with the whole refusal, else None."""
        lead = len(self.buffer) - len(self.buffer.lstrip(_LEAD))
        end = lead + len(self.core)
        if self.buffer[lead:end].casefold() != self.core:
            return None
        tail = self.buffer[end:]
        closing = len(tail) - len(tail.lstrip(".!\"”' \t\r\n*"))
        return self.buffer[:end + closing], tail[closing:]

    def feed(self, piece: str, *, final: bool = False) -> str:
        if self.state == "pass":
            return piece
        self.buffer += piece
        parts = self._split()
        if parts is None:
            body = self.buffer.lstrip(_LEAD).casefold()
            if not final and self.core.startswith(body):
                return ""                      # still a possible refusal prefix
            return self._release(self.buffer)
        held, tail = parts
        if len("".join(tail.split())) >= MIN_TAIL:
            self.stripped = True
            tail = _CONNECTOR.sub("", tail, count=1)
            return self._release(tail[:1].upper() + tail[1:])
        return self._release(self.buffer) if final else ""

    def _release(self, text: str) -> str:
        self.state, self.buffer = "pass", ""
        return text


def evidence_texts(found: dict) -> dict[str, str]:
    """Text of each evidence ID exactly as it was put in the prompt."""
    evidence_map = found.get("evidence_map") or {}
    texts = {c["evidence_id"]: c.get("text", "") for c in found.get("context_evidence", [])
             if c.get("evidence_id") in evidence_map}
    slides = {(s.get("video_id", ""), s["segment_id"]): s for s in found.get("slides", [])}
    for evidence_id, source in evidence_map.items():
        if evidence_id in texts or source.get("kind") != "slide":
            continue
        slide = slides.get((source.get("video_id", ""), source.get("segment_id")))
        if slide:
            texts[evidence_id] = "\n".join(part for part in (
                slide.get("title") or "", slide.get("ocr_text") or "", slide.get("speech_text") or "") if part)
    return {evidence_id: text for evidence_id, text in texts.items() if text.strip()}


def attribute(answer: str, found: dict, encoder, *, refusal: str = "",
              minimum: float | None = None) -> tuple[str, list[dict]]:
    """Append [mm:ss] of the closest evidence block to each uncited sentence.

    Returns the answer unchanged when it already carries a rendered citation, is a
    refusal, or no sentence reaches ``minimum``. Sources are returned in first-use order.
    """
    if minimum is None:
        minimum = float(os.getenv("ASK_ATTRIBUTE_MIN", DEFAULT_ATTRIBUTE_MIN))
    texts = evidence_texts(found)
    if not answer.strip() or not texts or _STAMP.search(answer):
        return answer, []
    if refusal and refusal.rstrip(".").casefold() in answer.casefold():
        return answer, []
    pieces, position = [], 0
    for match in _SENTENCE.finditer(answer):
        pieces.append((answer[position:match.start()], match.group()))
        position = match.end()
    pieces.append((answer[position:], ""))
    targets = [n for n, (sentence, _) in enumerate(pieces) if len(sentence.split()) >= 4]
    if not targets:
        return answer, []
    ids = list(texts)
    passages = encoder.encode_many([texts[i] for i in ids], prefix="passage: ")
    sentences = encoder.encode_many([pieces[n][0] for n in targets])
    similarity = sentences @ passages.T
    evidence_map, used, output = found["evidence_map"], {}, list(pieces)
    for row, n in enumerate(targets):
        best = int(similarity[row].argmax())
        score = float(similarity[row][best])
        if score < minimum:
            continue
        evidence_id = ids[best]
        sentence, separator = pieces[n]
        cut = _TRAILING.search(sentence).start()
        mark = f" [{stamp(evidence_map[evidence_id]['t_start'])}]"
        output[n] = (sentence[:cut] + mark + sentence[cut:], separator)
        entry = used.setdefault(evidence_id, {"id": evidence_id, **evidence_map[evidence_id],
                                              "inferred": True, "score": round(score, 4)})
        entry["score"] = max(entry["score"], round(score, 4))
    if not used:
        return answer, []
    return "".join(sentence + separator for sentence, separator in output), list(used.values())
