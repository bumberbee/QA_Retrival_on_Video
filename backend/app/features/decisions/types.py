"""Shared types and probability conventions for decision backends."""

from dataclasses import dataclass, field
from typing import Literal

QuestionType = Literal["definition", "formula", "explanation", "comparison",
                       "locate_time", "summary", "out_of_scope", "chitchat"]
Modality = Literal["ocr", "speech", "both"]
Action = Literal["run", "clarify", "direct"]
QTYPES = ("definition", "formula", "explanation", "comparison", "locate_time",
          "summary", "out_of_scope", "chitchat")
MODALITIES = {"definition": "both", "formula": "ocr", "explanation": "speech",
              "comparison": "both", "locate_time": "both", "summary": "speech",
              "out_of_scope": "both", "chitchat": "both"}


class NotConfigured(RuntimeError):
    """This backend has no configured implementation; callers may fall back."""


@dataclass(frozen=True)
class BackendResult:
    """Binary p is P(first policy label); choice p is P(selected label).

    scores are optional per-label probabilities/similarities for router margins.
    """

    label: str
    p: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    name: str
    label: str
    p: float
    confidence: float
    zone: Literal["accept", "reject", "uncertain"]
    backend: str
    latency_ms: float
