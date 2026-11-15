"""Per-question limits and atomic duplicate-call protection (A5)."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


def _canonical_args(value):
    """JSON has one number type: 1 and 1.0 must not bypass the call guard."""
    if isinstance(value, dict):
        return {key: _canonical_args(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_args(item) for item in value]
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


@dataclass(frozen=True)
class BudgetLimits:
    max_tool_calls: int = 6
    max_search_rounds: int = 2
    max_llm_calls: int = 2
    wall_timeout_s: float = 240.0

    def __post_init__(self):
        for name in ("max_tool_calls", "max_search_rounds", "max_llm_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        value = self.wall_timeout_s
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("wall_timeout_s must be positive and finite")

    @classmethod
    def load(cls, path: str | Path | None = None) -> BudgetLimits:
        path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "agent_limits.yaml"
        if path.suffix == ".json":
            values = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                import yaml
            except ImportError:
                values = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            else:
                if path.exists():
                    values = yaml.safe_load(path.read_text(encoding="utf-8"))
                else:
                    values = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Agent limits must be an object")
        return cls(**values)


class Budget:
    """Failed executions consume a call; validation failures and refusals do not.

    Each search tool invocation consumes one search round. The future state
    machine must reserve LLM calls before calling its gateway, including retries.
    """

    def __init__(self, limits: BudgetLimits | None = None, *, clock: Callable[[], float] = time.monotonic):
        self.limits = limits if limits is not None else BudgetLimits.load()
        self.clock = clock
        self.started_at = clock()
        self.tool_calls = self.search_rounds = self.llm_calls = 0
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        self._timed_out = False

    @property
    def remaining_seconds(self) -> float:
        if self._timed_out:
            return 0.0
        return max(0.0, self.limits.wall_timeout_s - (self.clock() - self.started_at))

    def expire(self) -> None:
        with self._lock:
            self._timed_out = True

    def reserve_tool(self, name: str, args: dict, *, search: bool = False) -> str | None:
        fingerprint = json.dumps([name, _canonical_args(args)], sort_keys=True, ensure_ascii=False, allow_nan=False,
                                 separators=(",", ":"))
        with self._lock:
            if self.remaining_seconds <= 0:
                return "wall_timeout: agent wall-clock budget exhausted"
            if fingerprint in self._seen:
                return "duplicate_call: same tool and arguments already attempted"
            if self.tool_calls >= self.limits.max_tool_calls:
                return "tool_budget: maximum tool calls reached"
            if search and self.search_rounds >= self.limits.max_search_rounds:
                return "search_budget: maximum search rounds reached"
            self._seen.add(fingerprint)
            self.tool_calls += 1
            self.search_rounds += int(search)
        return None

    def reserve_llm_call(self) -> str | None:
        with self._lock:
            if self.remaining_seconds <= 0:
                return "wall_timeout: agent wall-clock budget exhausted"
            if self.llm_calls >= self.limits.max_llm_calls:
                return "llm_budget: maximum LLM calls reached"
            self.llm_calls += 1
        return None

    def snapshot(self) -> dict:
        with self._lock:
            return {"tool_calls": self.tool_calls, "search_rounds": self.search_rounds,
                    "llm_calls": self.llm_calls, "remaining_seconds": self.remaining_seconds,
                    "limits": asdict(self.limits)}
