from tests import _isolate  # noqa: F401

import copy
import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.features.agent import graph, nli, verify
from app.features.agent.tools import Budget, BudgetLimits, ToolContext
from app.features.agent.tools.verify_tool import NLI_TODO, verify_claims
from app.features.ask import llm
from app.features.ask import router as qa
from app.main import app


def evidence():
    source = nli.found_from_evidence_map({"E1": {"text": "Biến cố chắc chắn có xác suất bằng một.", "t_start": 65},
                                         "E2": {"text": "Một bằng một.", "t_start": 65}})
    source.update(cited_sources=[{"id": "E1"}], invalid_citations=[], slides=[], best_semantic=0.9)
    return source


def step(max_llm_calls=2):
    return graph.AgentResult("answer", trace={"states": ["S5_answer"], "verified": True},
                             budget=Budget(BudgetLimits(max_llm_calls=max_llm_calls)))


def scorer(pairs):
    return [(0.05, 0.05, 0.9) if "bằng không" in hypothesis else
            (0.1, 0.8, 0.1) if "thiên văn" in hypothesis else (0.9, 0.05, 0.05)
            for _, hypothesis in pairs]


def test_flag_off_identity_for_answer_reports_trace_citations_and_tool(monkeypatch):
    monkeypatch.setattr(nli, "NLI", False)
    forbidden = Mock(side_effect=AssertionError("NLI must remain unused"))
    monkeypatch.setattr(nli, "default_scorer", forbidden)
    source, state = evidence(), step()
    source.pop("context_evidence")
    source["evidence_map"] = {"E1": {"t_start": 65}}
    renderer = llm.CitationRenderer(source["evidence_map"])
    answer = renderer.feed("Nội dung [E1].", final=True)
    source.update(renderer.metadata())
    answer, report = qa.recheck(state, source, answer, forbidden, scorer=forbidden)
    qa.conclude(state, source, report)
    actual = {"answer": answer, "metadata": renderer.metadata(), "report": report,
              "states": state.trace["states"], "s6": state.trace["s6"], "verification": state.trace["verification"]}
    # Literal pre-A6 payload: compare serialized UTF-8 bytes, including field order.
    expected = {"answer": "Nội dung [01:05].",
                "metadata": {"cited_sources": [{"id": "E1", "t_start": 65}], "invalid_citations": []},
                "report": {"claims": [], "mismatches": []}, "states": ["S5_answer", "S6_check"],
                "s6": [{"claims": [], "mismatches": []}],
                "verification": {"label": "verified", "reasons": [], "numeric_claims": 0, "numeric_checked": 0}}
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert encode(actual) == encode(expected)
    assert verify.hint({"mismatches": []}) == (
        "Lưu ý: câu trả lời trước có phép tính sai. Tính chính xác thì "
        ". Hãy trả lời lại, kiểm tra từng phép tính và chỉ dùng số liệu có trong bằng chứng.")
    tool = verify_claims("Nội dung [E1].", source["evidence_map"], ctx=ToolContext(nli_scorer=forbidden))
    assert tool.data == {"status": "unverified", "claims": [{"claim_id": "C1", "text": "Nội dung [E1].",
        "status": "unverified", "evidence_ids": ["E1"], "missing_evidence_ids": [], "todo": NLI_TODO}],
        "evidence_map": {"E1": {"t_start": 65}}, "evidence": [], "todo": NLI_TODO}
    forbidden.assert_not_called()


@pytest.mark.parametrize("error", [nli.NLIUnavailable("No local assets"), RuntimeError("Inference failed"),
                                   ValueError("Invalid probabilities")])
def test_nli_failure_preserves_answer_and_existing_verdict(monkeypatch, error):
    monkeypatch.setattr(nli, "NLI", True)
    monkeypatch.setattr(nli, "default_scorer", Mock(side_effect=error))
    source, state = evidence(), step()
    answer = "Nội dung [E1]."
    retry = Mock(side_effect=AssertionError("No retry for NLI errors"))
    kept, report = qa.recheck(state, source, answer, retry)
    qa.conclude(state, source, report)
    assert kept == answer
    assert report["nli_error"] == f"{type(error).__name__}: {error}"
    assert state.trace["verification"] == {"label": "verified", "reasons": [], "numeric_claims": 0,
                                            "numeric_checked": 0, "nli_error": report["nli_error"]}
    retry.assert_not_called()
    assert state.budget.llm_calls == 1


def test_contradiction_uses_one_retry_even_when_budget_allows_more(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    source, state = evidence(), step(max_llm_calls=5)
    answer = "Biến cố chắc chắn có xác suất bằng không [E1]."
    retry = Mock(return_value=answer)
    kept, report = qa.recheck(state, source, answer, retry, scorer=scorer)
    qa.conclude(state, source, report)
    assert kept == answer
    retry.assert_called_once()
    assert "mâu thuẫn với bằng chứng" in retry.call_args.args[0]
    assert answer in retry.call_args.args[0]
    assert state.budget.llm_calls == 2
    assert len(state.trace["s6"]) == 2
    assert state.trace["verification"]["reasons"] == ["contradicted_claim"]


def test_corrected_contradiction_is_kept_with_its_nli_report(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    source, state = evidence(), step()
    right = "Biến cố chắc chắn có xác suất bằng một [E1]."
    answer, report = qa.recheck(state, source, "Biến cố chắc chắn có xác suất bằng không [E1].",
                                Mock(return_value=right), scorer=scorer)
    qa.conclude(state, source, report)
    assert answer == right
    assert state.trace["verification"]["label"] == "verified"
    assert state.trace["verification"]["nli"]["counts"]["supported"] == 1


def test_unsupported_claim_changes_verdict_without_regeneration_or_text_edits(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    source, state, retry = evidence(), step(), Mock()
    answer = "Bài giảng nói về thiên văn [E1]."
    kept, report = qa.recheck(state, source, answer, retry, scorer=scorer)
    qa.conclude(state, source, report)
    assert kept == answer
    assert state.trace["verification"]["reasons"] == ["unsupported_claim"]
    retry.assert_not_called()


def test_numeric_mismatch_keeps_priority_and_shares_the_single_retry(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    source, state = evidence(), step(max_llm_calls=5)
    bad = "Ta có 2 + 2 = 5 [E1]. Biến cố chắc chắn có xác suất bằng không [E1]."
    improved = "Ta có 2 + 2 = 4 [E1]. Biến cố chắc chắn có xác suất bằng không [E1]."
    retry = Mock(return_value=improved)
    answer, report = qa.recheck(state, source, bad, retry, scorer=scorer)
    assert answer == improved
    assert "phép tính sai" in retry.call_args.args[0]
    assert "mâu thuẫn" not in retry.call_args.args[0]
    retry.assert_called_once()
    assert report["mismatches"] == [] and report["nli"]["counts"]["contradicted"] == 1
    assert state.budget.llm_calls == 2


def test_numeric_regression_is_rejected_even_when_contradiction_is_fixed(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    source, state = evidence(), step()
    original = "Biến cố chắc chắn có xác suất bằng không [E1]."
    retry = Mock(return_value="Ta có 2 + 2 = 5 [E1].")
    kept, report = qa.recheck(state, source, original, retry, scorer=scorer)
    assert kept == original and report["nli"]["counts"]["contradicted"] == 1


def test_failed_second_nli_check_cannot_count_as_improvement(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    source, state = evidence(), step()
    fake = Mock(side_effect=[[(0.05, 0.05, 0.9)], nli.NLIUnavailable("failed second inference")])
    answer = "Sentence [E1]."
    kept, report = qa.recheck(state, source, answer, Mock(return_value="Other [E1]."), scorer=fake)
    assert kept == answer
    assert report["nli"]["counts"]["contradicted"] == 1
    assert "nli_error" in state.trace["s6"][1]


def test_rejected_retry_restores_raw_citations_and_metadata(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    source, state = evidence(), step()
    source["nli_answer"] = "Biến cố chắc chắn có xác suất bằng không [E1]."
    before = copy.deepcopy(source)

    def regenerate(hint):
        source["nli_answer"] = "Biến cố chắc chắn có xác suất bằng không [E2]."
        source["cited_sources"] = [{"id": "E2"}]
        return "Biến cố chắc chắn có xác suất bằng không [01:05]."

    answer = "Biến cố chắc chắn có xác suất bằng không [01:05]."
    kept, _ = qa.recheck(state, source, answer, regenerate, scorer=scorer)
    assert kept == answer
    assert source.get("nli_answer") == before["nli_answer"]
    assert source["cited_sources"] == before["cited_sources"]


def test_rejected_retry_removes_raw_metadata_absent_from_original(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    source, state = evidence(), step()

    def regenerate(hint):
        source["nli_answer"] = "Biến cố chắc chắn có xác suất bằng không [E2]."
        return "Biến cố chắc chắn có xác suất bằng không [01:05]."

    original = "Biến cố chắc chắn có xác suất bằng không [E1]."
    kept, _ = qa.recheck(state, source, original, regenerate, scorer=scorer)
    assert kept == original and "nli_answer" not in source


def test_stripped_refusal_is_excluded_from_raw_claims(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    renderer = llm.CitationRenderer(evidence()["evidence_map"])
    raw = llm.REFUSAL + " Tuy nhiên, biến cố chắc chắn có xác suất bằng một theo nội dung bài giảng [E1]."
    stripper = qa.postprocess.RefusalStripper(llm.REFUSAL, active=True)
    rendered = stripper.feed(renderer.feed(raw, final=True), final=True)
    metadata = llm._citation_metadata(renderer, stripper)
    assert stripper.stripped
    assert llm.REFUSAL not in rendered and llm.REFUSAL not in metadata["nli_answer"]
    source = evidence()
    source.update(metadata)
    report = verify.check(rendered, source, scorer=scorer)
    assert len(report["nli"]["claims"]) == 1
    assert report["nli"]["claims"][0]["evidence_ids"] == ["E1"]


def test_no_retry_when_shared_budget_is_exhausted(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    state, retry = step(max_llm_calls=1), Mock()
    qa.recheck(state, evidence(), "Biến cố chắc chắn có xác suất bằng không [E1].", retry, scorer=scorer)
    retry.assert_not_called()
    assert state.budget.llm_calls == 1


def test_verdict_retains_all_existing_reasons_and_adds_both_support_codes():
    report = {"claims": [{"status": "mismatch"}], "mismatches": [{}],
              "nli": {"counts": {"unsupported": 1, "contradicted": 1}, "claims": []}}
    result = verify.verdict(report, {"invalid_citations": ["E9"], "cited_sources": []}, evidence_enough=False)
    assert result["reasons"] == ["weak_evidence", "numeric_mismatch", "invalid_citation", "no_citation",
                                 "unsupported_claim", "contradicted_claim"]


def test_verify_tool_uses_nli_statuses_and_keeps_placeholder_on_unavailability(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    source = evidence()["evidence_map"]
    claims = ["Biến cố chắc chắn có xác suất bằng một [E1].",
              "Bài giảng nói về thiên văn [E1].", "Biến cố chắc chắn có xác suất bằng không [E1]."]
    result = verify_claims(claims, source, ctx=ToolContext(nli_scorer=scorer))
    assert result.ok and result.data["status"] == "contradicted"
    assert [row["status"] for row in result.data["claims"]] == ["supported", "unsupported", "contradicted"]
    assert "todo" not in result.data
    monkeypatch.setattr(nli, "default_scorer", Mock(side_effect=nli.NLIUnavailable("absent")))
    fallback = verify_claims(claims, source, ctx=ToolContext())
    monkeypatch.setattr(nli, "NLI", False)
    legacy = verify_claims(claims, source, ctx=ToolContext())
    assert fallback.ok and fallback.data == legacy.data
    assert fallback.data["todo"] == NLI_TODO


def test_raw_ids_distinguish_sources_with_identical_timestamps(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    source = evidence()
    renderer = llm.CitationRenderer(source["evidence_map"])
    answer = renderer.feed("Première [E2]. Second [E1].", final=True)
    source.update(renderer.metadata())
    calls = Mock(return_value=[(0.9, 0.05, 0.05)] * 2)
    report = verify.check(answer, source, scorer=calls)
    assert [row["evidence_ids"] for row in report["nli"]["claims"]] == [["E2"], ["E1"]]
    assert calls.call_args.args[0][0][0] == "Một bằng một."
    assert answer == "Première [01:05]. Second [01:05]."


def test_stream_and_api_include_additive_verification_fields_and_one_retry(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    monkeypatch.setattr(nli, "default_scorer", lambda: scorer)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    source = evidence()
    plan = {"conversation": None, "history": [], "query": "q", "prefer": frozenset(), "rewritten": None,
            "follow_up": False, "previous": None, "summary": ""}
    bad = "Biến cố chắc chắn có xác suất bằng không [E1]."
    right = "Biến cố chắc chắn có xác suất bằng một [E1]."

    def evidence_for(*args):
        state = step()
        return state, copy.deepcopy(source), "[E1] Biến cố chắc chắn có xác suất bằng một."

    def complete(parts):
        return right if "mâu thuẫn với bằng chứng" in parts[0] else bad

    api = TestClient(app)
    payload = {"video_id": "abc123def456", "question": "Xác suất bằng bao nhiêu?"}
    with patch.object(qa.retrieval, "has_index", return_value=True), patch.object(qa, "prepare", return_value=plan), \
            patch.object(qa, "evidence_for", side_effect=evidence_for), patch.object(qa, "remember"), \
            patch.object(llm.gemini, "complete_parts", side_effect=complete) as completions, \
            patch.object(llm.gemini, "stream_parts", return_value=iter([bad[:30], bad[30:-3], bad[-3:]])):
        body = api.post("/api/qa", json=payload).json()
        assert completions.call_count == 2
        events = [json.loads(line) for line in api.post("/api/qa/stream", json=payload).text.splitlines()]
        assert completions.call_count == 3
    assert body["answer"] == "Biến cố chắc chắn có xác suất bằng một [01:05]."
    assert body["agent"]["verification"]["nli"]["counts"]["supported"] == 1
    assert body["agent"]["verification"]["label"] == "verified"
    assert events[-1]["answer"] == body["answer"]
    assert events[-1]["agent"]["verification"] == body["agent"]["verification"]
    assert body["agent"]["budget"]["llm_calls"] == events[-1]["agent"]["budget"]["llm_calls"] == 2
    assert "".join(event["text"] for event in events if event["type"] == "delta") == bad.replace("[E1]", "[01:05]")
    assert "nli_answer" not in body
