from tests import _isolate  # noqa: F401
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app.features.agent.tools import Budget, BudgetLimits, default_registry
from tests.features.agent.fixtures import context


class BudgetTests(unittest.TestCase):
    def test_bundled_yaml_and_json_have_identical_defaults(self):
        folder = Path(__file__).resolve().parents[3] / "app" / "features" / "agent"
        self.assertEqual(BudgetLimits.load(), BudgetLimits())
        self.assertEqual(BudgetLimits.load(folder / "agent_limits.json"), BudgetLimits())
        with patch.dict("sys.modules", {"yaml": None}):
            self.assertEqual(BudgetLimits.load(), BudgetLimits())

    def test_custom_configuration_and_missing_yaml_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent_limits.json"
            path.write_text(json.dumps({"max_tool_calls": 1, "max_search_rounds": 0,
                                        "max_llm_calls": 1, "wall_timeout_s": 2.5}), encoding="utf-8")
            limits = BudgetLimits.load(path)
            self.assertEqual(limits.max_tool_calls, 1)
            self.assertEqual(BudgetLimits.load(path.with_suffix(".yaml")), limits)

    def test_invalid_configuration_is_rejected(self):
        for args in ({"max_tool_calls": -1}, {"max_llm_calls": True}, {"max_search_rounds": 1.5},
                     {"wall_timeout_s": 0}, {"wall_timeout_s": float("inf")}, {"wall_timeout_s": "240"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                BudgetLimits(**args)

    def test_exactly_six_tools_are_allowed(self):
        ctx, registry = context(), default_registry()
        for number in range(6):
            self.assertTrue(registry.run("calc", {"expr": str(number)}, ctx).ok)
        result = registry.run("calc", {"expr": "6"}, ctx)
        self.assertIn("tool_budget", result.error)
        self.assertEqual(ctx.budget.tool_calls, 6)

    def test_two_searches_and_two_llm_calls(self):
        budget = Budget(BudgetLimits())
        self.assertIsNone(budget.reserve_tool("search_ocr", {"q": "first"}, search=True))
        self.assertIsNone(budget.reserve_tool("search_speech", {"q": "second"}, search=True))
        self.assertIn("search_budget", budget.reserve_tool("search_hybrid", {"q": "third"}, search=True))
        self.assertIsNone(budget.reserve_tool("get_slide", {"slide_id": "s1"}))
        self.assertIsNone(budget.reserve_llm_call())
        self.assertIsNone(budget.reserve_llm_call())
        self.assertIn("llm_budget", budget.reserve_llm_call())
        self.assertEqual((budget.tool_calls, budget.search_rounds, budget.llm_calls), (3, 2, 2))

    def test_duplicates_ignore_dict_order_and_include_tool_name(self):
        budget = Budget(BudgetLimits())
        self.assertIsNone(budget.reserve_tool("one", {"a": 1, "b": {"c": 2}}))
        self.assertIn("duplicate_call", budget.reserve_tool("one", {"b": {"c": 2}, "a": 1}))
        self.assertIn("duplicate_call", budget.reserve_tool("one", {"b": {"c": 2.0}, "a": 1.0}))
        self.assertIsNone(budget.reserve_tool("two", {"a": 1, "b": {"c": 2}}))
        self.assertEqual(budget.tool_calls, 2)

    def test_failed_execution_cannot_be_repeated(self):
        ctx, registry = context(), default_registry()
        self.assertFalse(registry.run("calc", {"expr": "1/0"}, ctx).ok)
        self.assertIn("duplicate_call", registry.run("calc", {"expr": "1/0"}, ctx).error)
        self.assertEqual(ctx.budget.tool_calls, 1)

    def test_expired_clock_prevents_tools_and_llm_calls(self):
        now = [10.0]
        budget = Budget(BudgetLimits(), clock=lambda: now[0])
        now[0] += 240.0
        self.assertIn("wall_timeout", budget.reserve_tool("one", {}))
        self.assertIn("wall_timeout", budget.reserve_llm_call())
        self.assertEqual(budget.snapshot()["remaining_seconds"], 0)
        self.assertEqual(budget.tool_calls, 0)

    def test_concurrent_duplicate_reservations_are_atomic(self):
        budget, barrier, outputs = Budget(BudgetLimits()), threading.Barrier(3), []
        def reserve():
            barrier.wait()
            outputs.append(budget.reserve_tool("same", {"q": "same"}))
        threads = [threading.Thread(target=reserve) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertEqual(outputs.count(None), 1)
        self.assertTrue(any(value and "duplicate_call" in value for value in outputs))
        self.assertEqual(budget.tool_calls, 1)
