from tests import _isolate  # noqa: F401

import copy
from dataclasses import replace

import pytest

from app.features.decisions import POLICY, NotConfigured, QuestionDecision, resolve_route, route_question
from app.features.decisions.types import BackendResult, MODALITIES


class FakeBackend:
    def __init__(self, label="definition", p=0.9, scores=None, unavailable=False):
        self.result = BackendResult(label, p, scores or {})
        self.unavailable = unavailable
        self.calls = 0

    def run(self, name, state):
        self.calls += 1
        assert name == "Q4"
        if self.unavailable:
            raise NotConfigured("fake unavailable")
        return self.result


def test_rules_stop_before_any_model_or_encoder_or_llm():
    def never(*args):
        raise AssertionError("Later stage ran")
    model = FakeBackend()
    decision = route_question("Tóm tắt bài học", {}, encode=never, llm=never,
                              backends={"minijev": model})
    assert decision.qtype == "summary"
    assert decision.decided_by == "rules"
    assert decision.steps == ["rules"]
    assert model.calls == 0


@pytest.mark.parametrize("backend", ["minijev", "jev"])
def test_optional_model_stops_at_confidence_threshold(backend):
    def never(*args):
        raise AssertionError("Later stage ran")
    configured = FakeBackend("comparison", 0.5625)
    decision = route_question("Nội dung cần tìm", {"q4_backend": backend}, encode=never, llm=never,
                              backends={backend: configured})
    assert decision.decided_by == backend
    assert decision.confidence == 0.5
    assert decision.qtype == "comparison"
    assert decision.steps == ["rules", backend]


def test_optional_models_are_tried_in_order_until_one_passes():
    mini = FakeBackend("definition", 0.5)
    jev = FakeBackend("comparison", 0.8)
    semantic = FakeBackend()
    decision = route_question("Nội dung cần tìm", {},
                              backends={"minijev": mini, "jev": jev, "semantic": semantic})
    assert decision.steps == ["rules", "minijev", "jev"]
    assert decision.decided_by == "jev"
    assert (mini.calls, jev.calls, semantic.calls) == (1, 1, 0)


def test_optional_backend_can_be_selected_in_policy():
    policy = copy.deepcopy(POLICY)
    policy["Q4"]["backend"] = "jev"
    decision = route_question("Nội dung cần tìm", {}, policy=policy,
                              backends={"jev": FakeBackend("summary", 0.9)})
    assert decision.steps == ["rules", "jev"]
    assert decision.decided_by == "jev"


@pytest.mark.parametrize("p,runner,expected", [(0.85, 0.8, "semantic"), (0.9, 0.82, "semantic"),
                                             (0.849, 0.7, "llm"), (0.9, 0.86, "llm")])
def test_semantic_requires_both_threshold_and_margin(p, runner, expected):
    semantic = FakeBackend("definition", p, {"definition": p, "explanation": runner})
    decision = route_question("Nội dung cần tìm", {}, backends={"semantic": semantic},
                              llm=lambda q, s: {"qtype": "summary", "modality": "speech"})
    assert decision.decided_by == expected
    assert semantic.calls == 1
    assert decision.steps[:4] == ["rules", "minijev", "jev", "semantic"]
    if expected == "semantic":
        assert decision.steps[-1] == "semantic"
    else:
        assert decision.steps[-2:] == ["heuristic", "llm"]


def test_unconfigured_backends_continue_without_mislabeling_rule_fallback():
    semantic = FakeBackend(unavailable=True)
    decision = route_question("Nội dung cần tìm", {}, backends={"semantic": semantic})
    assert decision.decided_by == "llm"
    assert decision.qtype == "explanation"
    assert decision.confidence == 0
    assert decision.steps == ["rules", "minijev", "jev", "semantic", "heuristic", "llm"]
    assert set(decision.step_timings_ms) == set(decision.steps)
    assert all(ms >= 0 for ms in decision.step_timings_ms.values())


def test_semantic_needs_runner_up_scores_before_it_can_stop():
    decision = route_question("Nội dung cần tìm", {}, backends={"semantic": FakeBackend(p=0.99)})
    assert decision.decided_by == "llm"


@pytest.mark.parametrize("question,qtype", [
    ("Nội dung entropy", "definition"),
    ("Tôi muốn tìm nội dung liên quan entropy và các ứng dụng trong thực tế", "explanation"),
])
def test_lecture_terms_use_definition_or_explanation_heuristic(question, qtype):
    decision = route_question(question, {"lecture_terms": ["ENTROPY"]})
    assert decision.decided_by == "heuristic"
    assert decision.qtype == qtype
    assert decision.steps[-1] == "heuristic"


def test_lecture_heuristic_needs_terms_and_a_low_rrf_to_reject():
    question = "Một chuyến đi xa"
    assert route_question(question, {"lecture_terms": ["entropy"], "top1_rrf": 0.001}).qtype == "out_of_scope"
    assert route_question(question, {"lecture_terms": ["entropy"], "top1_rrf": 0.01}).decided_by == "llm"
    assert route_question(question, {"lecture_terms": ["entropy"]}).decided_by == "llm"
    assert route_question(question, {"top1_rrf": 0}).decided_by == "llm"
    assert route_question("Đề mục entropylike", {"lecture_terms": ["entropy"]}).decided_by == "llm"


@pytest.mark.parametrize("payload", ["not JSON", "[]", "null", '{"qtype":"invented","modality":"video"}',
                                    {}, {"qtype": ["formula"]}, {"qtype": "plan_dangerous"}])
def test_bad_llm_json_is_forced_to_valid_enums(payload):
    decision = route_question("Nội dung cần tìm", {}, llm=lambda q, s: payload)
    assert decision.decided_by == "llm"
    assert decision.qtype == "explanation"
    assert decision.modality == "speech"
    assert resolve_route(decision) == ("run", "plan_explanation")


@pytest.mark.parametrize("p", [-1, 2, "bad", None, float("nan")])
def test_bad_llm_probability_is_not_propagated(p):
    decision = route_question("Nội dung cần tìm", {},
                              llm=lambda q, s: {"qtype": "formula", "modality": "bad", "p": p})
    assert decision.qtype == "formula"
    assert decision.modality == "ocr"
    assert decision.p == 0.5


def test_llm_valid_json_cannot_create_new_routes_or_actions():
    decision = route_question("Nội dung cần tìm", {}, llm=lambda q, s:
                              '{"qtype":"chitchat","modality":"both","action":"run","route":"shell"}')
    assert decision.action == "direct"
    assert resolve_route(decision) == ("direct", "reply_template")


def test_llm_can_preserve_valid_modality_and_report_not_configured():
    decision = route_question("Nội dung cần tìm", {},
                              llm=lambda q, s: {"qtype": "definition", "modality": "ocr", "p": 0.7})
    assert decision.modality == "ocr"
    assert decision.p == 0.7
    def missing(q, s):
        raise NotConfigured("No LLM")
    assert route_question("Nội dung cần tìm", {}, llm=missing).confidence == 0


@pytest.mark.parametrize("question", ["Cái này là gì?", "Chỗ đó", "What is this?", "Explain that"])
def test_unanchored_pointer_clarifies_before_models(question):
    decision = route_question(question, {})
    assert decision.action == "clarify"
    assert decision.clarify_reason == "missing_reference"
    assert decision.steps == ["rules"]
    assert resolve_route(decision) == ("clarify", "Bạn đang hỏi về slide nào / phút nào?")


def test_timestamp_zero_or_prior_question_resolves_pointer_context():
    assert route_question("Chỗ đó", {"current_time": 0}).action == "run"
    assert route_question("Cái này là gì?", {"previous_question": "Đạo hàm"}).action == "run"
    assert route_question("What is this algorithm?", {}).action == "run"
    previous = QuestionDecision("comparison", "both", "run", 0.95, "rules")
    assert route_question("Còn cái đó?", {"previous_decision": previous}).qtype == "comparison"
    assert route_question("Còn cái đó?", {"previous_decision": {"qtype": "summary"}}).qtype == "summary"
    assert route_question("Cái này là gì?", {"previous_question": "   "}).action == "clarify"


@pytest.mark.parametrize("question,state", [("So sánh hai video", {}), ("Compare both videos", {}),
                                          ("Nội dung cần tìm", {"mentioned_video_ids": ["v1", "v2"]})])
def test_multiple_videos_in_single_scope_clarifies(question, state):
    decision = route_question(question, state)
    assert decision.action == "clarify"
    assert decision.clarify_reason == "multiple_videos"
    assert resolve_route(decision)[0] == "clarify"


def test_multiple_video_scope_is_allowed_when_provided():
    decision = route_question("So sánh hai video", {"scope": "multi_video"})
    assert decision.action == "run"
    assert decision.qtype == "comparison"


def test_persistent_ties_across_scoring_steps_clarify():
    decision = route_question("Nội dung cần tìm", {}, backends={
        "rules": FakeBackend("definition", 0.5, {"definition": 0.5, "formula": 0.5}),
        "minijev": FakeBackend("definition", 0.4, {"definition": 0.4, "formula": 0.4}),
        "semantic": FakeBackend("definition", 0.9, {"definition": 0.9, "formula": 0.88}),
    }, llm=lambda q, s: {"qtype": "definition", "modality": "both"})
    assert decision.decided_by == "llm"
    assert decision.action == "clarify"
    assert decision.clarify_reason == "ambiguous_types"


def test_a_later_clear_step_resolves_an_earlier_tie():
    decision = route_question("Nội dung cần tìm", {}, backends={
        "rules": FakeBackend("definition", 0.5, {"definition": 0.5, "formula": 0.5}),
        "semantic": FakeBackend("comparison", 0.9, {"comparison": 0.9, "formula": 0.7}),
    })
    assert decision.decided_by == "semantic"
    assert decision.action == "run"


def test_rules_collision_does_not_stop_even_with_lower_threshold():
    policy = copy.deepcopy(POLICY)
    policy["Q4"]["router"]["confidence_threshold"] = 0.3
    decision = route_question("So sánh và tóm tắt bài học", {}, policy=policy)
    assert decision.decided_by == "llm"
    assert decision.action == "clarify"


def test_router_copies_session_state_and_supports_state_callable():
    state = {"question": "original", "q4_backends": [],
             "llm": lambda q, s: {"qtype": "comparison", "modality": "both"}}
    before = dict(state)
    decision = route_question("Nội dung cần tìm", state)
    assert state == before
    assert decision.qtype == "comparison"
    assert decision.steps == ["rules", "semantic", "heuristic", "llm"]


def test_unknown_model_backend_or_model_enum_is_a_configuration_error():
    with pytest.raises(ValueError):
        route_question("Nội dung cần tìm", {"q4_backends": ["reranker"]})
    with pytest.raises(ValueError):
        route_question("Nội dung cần tìm", {}, backends={"minijev": FakeBackend("unknown", 0.9)})


@pytest.mark.parametrize("qtype,expected", [
    ("chitchat", ("direct", "reply_template")),
    ("out_of_scope", ("direct", "refusal_with_hint")),
    ("definition", ("run", "plan_definition")),
    ("formula", ("run", "plan_formula")),
    ("explanation", ("run", "plan_explanation")),
    ("comparison", ("run", "plan_comparison")),
    ("locate_time", ("run", "plan_locate")),
    ("summary", ("run", "plan_summary")),
])
def test_resolve_route_exact_spec_table(qtype, expected):
    decision = QuestionDecision(qtype, MODALITIES[qtype], expected[0], 0.95, "rules")
    assert resolve_route(decision) == expected
    assert resolve_route(replace(decision, action="clarify"))[0] == "clarify"


@pytest.mark.parametrize("action", ["run", "clarify", "direct"])
def test_unknown_question_type_raises_key_error(action):
    decision = QuestionDecision("invented", "both", action, 0.5, "llm")
    with pytest.raises(KeyError):
        resolve_route(decision)
