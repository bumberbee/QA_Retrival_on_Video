from tests import _isolate  # noqa: F401
import copy
import threading
import time
import unittest
from unittest.mock import Mock

from app.features.agent.tools import Budget, BudgetLimits, Tool, ToolContext, ToolRegistry, ToolResult, default_registry
from app.features.agent.tools.base import FunctionTool, evidence_to_found, object_schema
from app.features.search import retrieval
from tests.features.agent.fixtures import context, found


class ToolBaseTests(unittest.TestCase):
    def test_registry_contains_nine_tools_with_input_schemas(self):
        registry = default_registry()
        schemas = registry.schemas()
        self.assertEqual({entry["name"] for entry in schemas},
                         {"search_hybrid", "search_ocr", "search_speech", "rerank", "get_slide",
                          "get_window", "neighbors", "calc", "verify_claims"})
        for entry in schemas:
            self.assertTrue(entry["description"])
            self.assertEqual(entry["input_schema"]["type"], "object")
            self.assertIsInstance(registry.get(entry["name"]), Tool)
        schemas[0]["input_schema"]["properties"].clear()
        self.assertTrue(registry.schemas()[0]["input_schema"]["properties"])

    def test_registered_calculator_preserves_legacy_function(self):
        from app.features.agent.tools.calc import calc
        result = default_registry().run("calc", {"expr": "C(12,4)/C(15,4)"}, context())
        self.assertTrue(result.ok)
        self.assertEqual(result.data["value_fraction"], "33/91")
        self.assertEqual(calc("0.1+0.2")["value_fraction"], "3/10")
        self.assertEqual(set(result.to_dict()), {"ok", "data", "error", "latency_ms"})
        self.assertGreaterEqual(result.latency_ms, 0)

    def test_calculation_error_is_a_tool_failure(self):
        ctx = context()
        result = default_registry().run("calc", {"expr": "1/0"}, ctx)
        self.assertFalse(result.ok)
        self.assertTrue(result.error)
        self.assertEqual(ctx.budget.tool_calls, 1)

    def test_unknown_and_invalid_tools_never_raise(self):
        registry = default_registry()
        for name, args, ctx in (("absent", {}, context()), ([], {}, context()),
                                ("calc", {"expr": 1}, context()), ("calc", {"expr": "1", "extra": 1}, context()),
                                ("calc", None, context()), ("calc", {"expr": "1"}, None)):
            result = registry.run(name, args, ctx)
            self.assertFalse(result.ok)
            self.assertIsInstance(result.error, str)

    def test_registry_refuses_duplicate_registration(self):
        registry = default_registry()
        with self.assertRaises(ValueError):
            registry.register(registry.get("calc"))

    def test_registry_contains_arbitrary_tool_failures(self):
        registry = ToolRegistry()
        tool = Mock(name="broken")
        tool.name = "broken"
        tool.run.side_effect = RuntimeError("handler error")
        registry.register(tool)
        result = registry.run("broken", {}, context())
        self.assertFalse(result.ok)
        self.assertIn("handler error", result.error)

    def test_handler_gets_a_snapshot_of_input(self):
        args = {"items": [{"a": 1}]}
        original = copy.deepcopy(args)
        tool = FunctionTool("mutate", "test", object_schema({"items": {"type": "array"}}, ["items"]),
                            lambda args, ctx: args["items"][0].update(a=2))
        self.assertTrue(tool.run(args, context()).ok)
        self.assertEqual(args, original)

    def test_timeout_returns_and_refuses_further_work(self):
        release, entered = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def block(args, ctx):
            entered.set()
            release.wait(1)
            return "late"
        budget = Budget(BudgetLimits(wall_timeout_s=0.05))
        ctx = ToolContext(budget=budget)
        tool = FunctionTool("block", "test", object_schema({}, []), block)
        started = time.monotonic()
        result = tool.run({}, ctx)
        self.assertTrue(entered.is_set())
        self.assertFalse(result.ok)
        self.assertIn("wall_timeout", result.error)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertIn("wall_timeout", tool.run({}, ctx).error)
        self.assertEqual(budget.tool_calls, 1)
        release.set()

    def test_evidence_adapter_preserves_ids_and_source_spans(self):
        chunks = found()["slides"][0]["evidence_candidates"]
        for chunk in chunks:
            chunk["video_id"] = "vid"
        original = copy.deepcopy(chunks)
        adapted = evidence_to_found(chunks)
        text, sources = retrieval.build_context_with_ids(adapted)
        self.assertIn("[E1]", text)
        self.assertEqual({tuple(item["chunk_ids"]) for item in sources.values()}, {("ocr1",), ("sp1",)})
        self.assertEqual(next(item for item in sources.values() if item["chunk_ids"] == ["sp1"])["t_start"], 4)
        self.assertEqual(chunks, original)
        self.assertEqual(retrieval.build_context_with_ids(evidence_to_found(chunks)), (text, sources))

    def test_index_loader_is_cached_and_rejects_paths(self):
        ctx = context()
        self.assertIs(ctx.load_index(), ctx.load_index("vid"))
        ctx.index_loader.assert_called_once_with("vid")
        for video_id in ("../other", "C:\\file", "", None, 2):
            if video_id is None:
                continue
            with self.assertRaises(ValueError):
                ctx.load_index(video_id)
