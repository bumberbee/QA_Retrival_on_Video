"""Validated, bounded tool execution and retrieval-compatible evidence."""

from __future__ import annotations

import copy
import math
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from .budget import Budget


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolContext:
    """One context/budget per question; dependencies are loaded only on use.

    index_loader serves slide/window reads. Search delegates to retrieval.search,
    which owns its index loading; searcher can replace that boundary in tests.
    encoder and scorer are passed through without loading models here.
    """

    video_id: str = ""
    current_time: float | None = None
    index_loader: Callable[[str], Any] | None = None
    encoder: Any = None
    scorer: Callable[[list[tuple[str, str]]], list[float]] | None = None
    budget: Budget = field(default_factory=Budget)
    searcher: Callable[..., dict] | None = None
    _indexes: dict = field(default_factory=dict, init=False, repr=False)
    _index_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    def load_index(self, video_id: str | None = None):
        video_id = validate_video_id(self.video_id if video_id is None else video_id)
        with self._index_lock:
            if video_id not in self._indexes:
                loader = self.index_loader
                if loader is None:
                    from app.features.search.retrieval import load_index
                    loader = load_index
                self._indexes[video_id] = loader(video_id)
            return self._indexes[video_id]


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict

    def run(self, args: dict, ctx: ToolContext) -> ToolResult: ...


def validate_video_id(value: str) -> str:
    # Also accepts imported lecture IDs and synthetic fixtures; no path components.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("video_id must be a safe, nonempty lecture identifier")
    return value


def validate_input(value: Any, schema: dict, path: str = "args") -> Any:
    """The small JSON-schema subset used by this toolset, with strict numbers."""
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                return validate_input(value, option, path)
            except ValueError:
                pass
        raise ValueError(f"{path} does not match any allowed input type")
    kind = schema.get("type")
    valid = {"object": lambda: isinstance(value, dict), "array": lambda: isinstance(value, list),
             "string": lambda: isinstance(value, str), "integer": lambda: type(value) is int,
             "number": lambda: type(value) in (int, float) and math.isfinite(value),
             "boolean": lambda: type(value) is bool, "null": lambda: value is None}
    if kind is not None and not valid[kind]():
        raise ValueError(f"{path} must be {kind}")
    if kind == "object":
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} keys must be strings")
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ValueError(f"{path} missing required fields: {', '.join(sorted(missing))}")
        if schema.get("additionalProperties") is False and value.keys() - properties.keys():
            raise ValueError(f"{path} contains unknown fields")
        result = copy.deepcopy(value)
        for key, child in properties.items():
            if key in value:
                result[key] = validate_input(value[key], child, f"{path}.{key}")
            elif "default" in child:
                result[key] = copy.deepcopy(child["default"])
        if isinstance(schema.get("additionalProperties"), dict):
            for key in value.keys() - properties.keys():
                result[key] = validate_input(value[key], schema["additionalProperties"], f"{path}.{key}")
        return result
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            raise ValueError(f"{path} has an invalid item count")
        return [validate_input(item, schema.get("items", {}), f"{path}[{i}]") for i, item in enumerate(value)]
    if kind == "string":
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf):
            raise ValueError(f"{path} has an invalid length")
        if schema.get("minLength", 0) and not value.strip():
            raise ValueError(f"{path} must not be blank")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            raise ValueError(f"{path} has an invalid identifier")
    if kind in {"number", "integer"}:
        if not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise ValueError(f"{path} is out of range")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} has an invalid value")
    return copy.deepcopy(value)


class FunctionTool:
    def __init__(self, name: str, description: str, input_schema: dict,
                 handler: Callable[[dict, ToolContext], Any], *, search: bool = False,
                 validator: Callable[[dict], None] | None = None):
        self.name, self.description, self.input_schema = name, description, input_schema
        self.handler, self.search, self.validator = handler, search, validator

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        started = time.monotonic()

        def result(ok, data=None, error=None):
            return ToolResult(ok, data, error, (time.monotonic() - started) * 1000)

        try:
            if not isinstance(ctx, ToolContext):
                raise ValueError("A ToolContext is required")
            args = validate_input(args, self.input_schema)
            if self.validator is not None:
                self.validator(args)
            error = ctx.budget.reserve_tool(self.name, args, search=self.search)
            if error:
                return result(False, error=error)
            output = queue.Queue(maxsize=1)

            def execute():
                try:
                    output.put((True, self.handler(args, ctx)))
                except Exception as exc:
                    output.put((False, f"{type(exc).__name__}: {exc}"))

            # A daemon cannot delay shutdown; timed-out results are discarded.
            # Python cannot forcibly cancel an in-flight native model operation.
            threading.Thread(target=execute, name=f"agent-tool-{self.name}", daemon=True).start()
            try:
                ok, data = output.get(timeout=ctx.budget.remaining_seconds)
            except queue.Empty:
                ctx.budget.expire()
                return result(False, error="wall_timeout: tool exceeded agent deadline")
            if ctx.budget.remaining_seconds <= 0:
                ctx.budget.expire()
                return result(False, error="wall_timeout: tool exceeded agent deadline")
            return result(True, data=data) if ok else result(False, error=data)
        except Exception as exc:
            return result(False, error=f"{type(exc).__name__}: {exc}")


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def schemas(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": copy.deepcopy(t.input_schema)}
                for t in self._tools.values()]

    def run(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        started = time.monotonic()
        try:
            return self.get(name).run(args, ctx)
        except Exception as exc:
            return ToolResult(False, error=f"{type(exc).__name__}: {exc}",
                              latency_ms=(time.monotonic() - started) * 1000)


QUERY = {"type": "string", "minLength": 1, "maxLength": 2000}
SLIDE_ID = {"type": "string", "minLength": 1, "maxLength": 256, "pattern": r"[A-Za-z0-9_:-]+"}
VIDEO_ID = {"type": "string", "minLength": 1, "maxLength": 128, "pattern": r"[A-Za-z0-9_-]+"}


def object_schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def evidence_item(chunk: dict, video_id: str, **defaults) -> dict:
    """Keep source chunk IDs/times; E1/E2 are assigned later by retrieval."""
    item = {**defaults, **copy.deepcopy(chunk)}
    item["video_id"] = video_id
    item.setdefault("segment_id", item.get("slide_id"))
    item.setdefault("start_sec", item.get("t_start"))
    item.setdefault("end_sec", item.get("t_end"))
    if not isinstance(item.get("chunk_id"), str) or not item["chunk_id"]:
        raise ValueError("Evidence must have a stable chunk_id")
    if not isinstance(item.get("text"), str) or not isinstance(item.get("segment_id"), str):
        raise ValueError("Evidence must have text and segment_id")
    if item.get("source_type") not in {"ocr", "speech", "region"}:
        raise ValueError("Unsupported evidence source type")
    for key in ("start_sec", "end_sec"):
        if type(item.get(key)) not in (int, float) or not math.isfinite(item[key]) or item[key] < 0:
            raise ValueError("Evidence timestamps must be nonnegative and finite")
    if item["start_sec"] > item["end_sec"]:
        raise ValueError("Evidence timestamps are reversed")
    item.update(id=f"{video_id}:{item['chunk_id']}", slide_id=item["segment_id"],
                t_start=item["start_sec"], t_end=item["end_sec"])
    item.setdefault("priority", 0)
    item.setdefault("selection_reason", "matched")
    item.setdefault("title", "")
    return item


def evidence_to_found(evidence: list[dict]) -> dict:
    """Adapt any tool's evidence to build_context_with_ids without ranking it."""
    slides = {}
    for chunk in evidence:
        item = evidence_item(chunk, chunk["video_id"])
        key = (item["video_id"], item["segment_id"])
        slide = slides.setdefault(key, {"video_id": item["video_id"], "segment_id": item["segment_id"],
                                       "start_sec": item["start_sec"], "end_sec": item["end_sec"],
                                       "title": item["title"], "ocr_text": "", "speech_text": "",
                                       "figures": [], "evidence_candidates": [], "matched_chunk_ids": []})
        slide["start_sec"] = min(slide["start_sec"], item["start_sec"])
        slide["end_sec"] = max(slide["end_sec"], item["end_sec"])
        slide["evidence_candidates"].append(item)
        slide["matched_chunk_ids"].append(item["chunk_id"])
    return {"slides": list(slides.values())}
