from tests import _isolate  # noqa: F401
import copy
import unittest
from unittest.mock import Mock

from app.features.agent.tools import default_registry
from app.features.agent.tools.verify_tool import verify_claims
from tests.features.agent.fixtures import context


class VerifyToolTests(unittest.TestCase):
    def test_splits_claims_and_marks_all_unverified_without_inference(self):
        evidence_map = {"E1": {"chunk_ids": ["c1"], "t_start": 1}}
        original = copy.deepcopy(evidence_map)
        scorer, encoder = Mock(side_effect=AssertionError("model forbidden")), Mock()
        ctx = context(scorer=scorer, encoder=encoder)
        result = verify_claims("First fact [E1]. Second fact [E2]!\nThird fact?", evidence_map, ctx=ctx)
        self.assertTrue(result.ok)
        self.assertEqual(result.data["status"], "unverified")
        self.assertEqual([claim["claim_id"] for claim in result.data["claims"]], ["C1", "C2", "C3"])
        self.assertTrue(all(claim["status"] == "unverified" for claim in result.data["claims"]))
        self.assertEqual(result.data["claims"][0]["evidence_ids"], ["E1"])
        self.assertEqual(result.data["claims"][1]["missing_evidence_ids"], ["E2"])
        self.assertIn("TODO (A6)", result.data["todo"])
        self.assertEqual(evidence_map, original)
        scorer.assert_not_called()
        encoder.encode.assert_not_called()
        ctx.index_loader.assert_not_called()
        self.assertEqual(ctx.budget.llm_calls, 0)

    def test_list_inputs_and_decimal_numbers_are_preserved(self):
        result = verify_claims(["P = 0.5 [E1].", "Another fact. Last fact."], {"E1": {}}, ctx=context())
        self.assertTrue(result.ok)
        self.assertEqual([claim["text"] for claim in result.data["claims"]],
                         ["P = 0.5 [E1].", "Another fact.", "Last fact."])

    def test_missing_evidence_is_not_reported_as_verified(self):
        result = verify_claims("Uncited fact. Cited fact [E4].", {}, ctx=context())
        self.assertTrue(result.ok)
        self.assertEqual(result.data["claims"][0]["evidence_ids"], [])
        self.assertEqual(result.data["claims"][1]["missing_evidence_ids"], ["E4"])
        self.assertEqual(result.data["status"], "unverified")

    def test_invalid_claims_and_evidence_are_tool_failures(self):
        registry = default_registry()
        for args in ({"claims": "", "evidence_map": {}}, {"claims": [], "evidence_map": {}},
                     {"claims": [2], "evidence_map": {}}, {"claims": "Fact", "evidence_map": []},
                     {"claims": "Fact", "evidence_map": {"E1": "text"}},
                     {"claims": "Fact", "evidence_map": {}, "extra": 1}):
            with self.subTest(args=args):
                ctx = context()
                result = registry.run("verify_claims", args, ctx)
                self.assertFalse(result.ok)
                self.assertEqual(ctx.budget.tool_calls, 0)
