from tests import _isolate  # noqa: F401; prevent credential loading before app imports

import copy
import json
import sys
from unittest.mock import patch

import pytest

from app.features.decisions import BACKENDS, POLICY, Decision, NotConfigured, decide
from app.features.decisions.base import confidence, run_backend
from app.features.decisions.config import ROOT, load_config, load_policy
from app.features.decisions.types import BackendResult


class FixedBackend:
    def __init__(self, label, p):
        self.result = BackendResult(label, p)

    def run(self, name, state):
        return self.result


@pytest.mark.parametrize("p,expected", [(0, 1), (0.1, 0.8), (0.5, 0), (0.8, 0.6), (1, 1)])
def test_binary_confidence(p, expected):
    assert confidence(p, 2, "binary") == pytest.approx(expected)


@pytest.mark.parametrize("n,p,expected", [(8, 0.125, 0), (8, 0.5625, 0.5),
                                         (8, 0.95, 33 / 35), (3, 1 / 3, 0),
                                         (3, 0.8, 0.7), (2, 0.75, 0.5),
                                         (8, 1, 1), (8, 0, 0)])
def test_choice_confidence(n, p, expected):
    assert confidence(p, n, "choice") == pytest.approx(expected)


@pytest.mark.parametrize("p,zone", [(0, "reject"), (0.2, "reject"), (0.20001, "uncertain"),
                                   (0.59999, "uncertain"), (0.6, "accept"), (1, "accept")])
def test_zone_boundaries(p, zone):
    result = decide("Q1", {}, backend="fake", backends={"fake": FixedBackend("valid", p)})
    assert isinstance(result, Decision)
    assert result.name == "Q1"
    assert result.zone == zone
    assert result.p == p
    assert result.backend == "fake"
    assert result.latency_ms >= 0


def test_thresholds_are_per_decision_and_configuration_is_not_mutated():
    policy = copy.deepcopy(POLICY)
    policy["Q1"].update(accept=0.8, reject=0.3)
    registry = {"fake": FixedBackend("valid", 0.7)}
    assert decide("Q1", {}, backend="fake", policy=policy, backends=registry).zone == "uncertain"
    assert decide("Q3", {}, backend="fake", backends={"fake": FixedBackend("follow_up", 0.7)}).zone == "accept"
    assert POLICY["Q1"]["accept"] == 0.6


@pytest.mark.parametrize("backend", ["reranker", "nli", "minijev", "jev"])
def test_model_stubs_raise_and_decide_falls_back_to_rules(backend):
    with pytest.raises(NotConfigured):
        BACKENDS[backend].run("Q1", {})
    decision = decide("Q1", {"question": "Định nghĩa xác suất?"}, backend=backend)
    assert decision.label == "valid"
    assert decision.backend == "rules"
    assert decision.confidence == pytest.approx(0.9)


@pytest.mark.parametrize("name,label", [("Q6", "no"), ("Q8", "insufficient"), ("Q9", "drop"),
                                       ("Q10", "unrelated"), ("Q11", "no")])
def test_unimplemented_decisions_have_honest_neutral_fallback(name, label):
    decision = decide(name, {})
    assert decision.label == label
    assert decision.p == pytest.approx(1 / len(POLICY[name]["labels"]))
    assert decision.confidence == 0
    assert decision.zone == "uncertain"
    assert decision.backend == "fallback"


def test_unconfigured_semantic_uses_q4_rules():
    decision = decide("Q4", {"question": "Tóm tắt bài học"}, backend="semantic")
    assert decision.label == "summary"
    assert decision.backend == "rules"


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_is_not_silently_fallen_back(p):
    with pytest.raises(ValueError, match="probability"):
        decide("Q1", {}, backend="fake", backends={"fake": FixedBackend("valid", p)})


def test_invalid_label_and_unknown_configuration_surface():
    with pytest.raises(ValueError, match="label"):
        decide("Q1", {}, backend="fake", backends={"fake": FixedBackend("invented", 0.9)})
    with pytest.raises(KeyError):
        decide("Q99", {})
    with pytest.raises(KeyError, match="backend"):
        decide("Q1", {}, backend="missing")


def test_backend_programming_errors_are_not_swallowed():
    class Broken:
        def run(self, name, state):
            raise RuntimeError("broken adapter")
    with pytest.raises(RuntimeError, match="broken adapter"):
        decide("Q1", {}, backend="broken", backends={"broken": Broken()})


def test_backends_can_return_mappings():
    class MappingBackend:
        def run(self, name, state):
            return {"label": "valid", "p": 0.8, "scores": {"valid": 0.8}}
    assert run_backend(MappingBackend(), "Q1", {}).p == 0.8


def test_yaml_policy_and_json_fallback_are_equivalent():
    assert load_policy() == json.loads((ROOT / "agent_policy.json").read_text(encoding="utf-8"))
    with patch.dict(sys.modules, {"yaml": None}):
        assert load_policy() == POLICY
        assert load_config("router_examples") == json.loads(
            (ROOT / "router_examples.json").read_text(encoding="utf-8"))
    assert set(POLICY) == {f"Q{i}" for i in range(1, 13)}
    assert all(POLICY[q]["accept"] == 0.6 and POLICY[q]["reject"] == 0.2 for q in POLICY)


@pytest.mark.parametrize("changes", [{"accept": 0.2, "reject": 0.6}, {"kind": "invalid"},
                                    {"labels": ["valid", "valid"]}, {"labels": [True, False]},
                                    {"fallback_label": "unknown"}])
def test_invalid_policy_is_rejected(tmp_path, changes):
    policy = copy.deepcopy(POLICY)
    policy["Q1"].update(changes)
    (tmp_path / "agent_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    with patch.dict(sys.modules, {"yaml": None}), pytest.raises(ValueError):
        load_policy(tmp_path)
