from tests import _isolate  # noqa: F401

import copy
from unittest.mock import Mock

import pytest

from app.features.agent import nli
from app.features.decisions import BACKENDS, POLICY, NotConfigured, decide
from app.features.decisions.nli import NLIBackend


@pytest.mark.parametrize("triple,label,zone", [((0.6, 0.3, 0.1), "supported", "accept"),
                                              ((0.59999, 0.30001, 0.1), "unsupported", "uncertain"),
                                              ((0.2, 0.3, 0.5), "contradicted", "reject"),
                                              ((0.20001, 0.29999, 0.5), "contradicted", "uncertain"),
                                              ((0, 0.4, 0.6), "contradicted", "reject")])
def test_q10_labels_probability_and_generic_zones(monkeypatch, triple, label, zone):
    monkeypatch.setattr(nli, "NLI", True)
    decision = decide("Q10", {"claim": "Claim", "evidence": "Evidence", "scorer": lambda pairs: [triple] * len(pairs)})
    assert (decision.label, decision.p, decision.zone, decision.backend) == (label, triple[0], zone, "nli")


def test_q10_injected_policy_and_backend_scorer_preserved(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    policy = copy.deepcopy(POLICY)
    policy["Q10"].update(accept=0.8, reject=0.3, reject_contradict=0.25)
    original = copy.deepcopy(policy)
    registry = {**BACKENDS, "nli": NLIBackend(scorer=lambda pairs: [(0.7, 0, 0.3)] * len(pairs))}
    result = decide("Q10", {"claim": "Claim", "evidence": "Evidence"}, policy=policy, backends=registry)
    assert (result.label, result.p, result.zone) == ("contradicted", 0.7, "uncertain")
    assert policy == original


def test_multi_claim_state_uses_worst_status_and_that_claims_entailment(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    fake = Mock(return_value=[(0.9, 0.05, 0.05), (0.1, 0.8, 0.1), (0.05, 0.05, 0.9)])
    result = decide("Q10", {"claims": ["First [E1].", "Second [E1].", "Third [E1]."],
                            "evidence_map": {"E1": {"text": "Evidence"}}, "scorer": fake})
    assert (result.label, result.p) == ("contradicted", 0.05)


def test_unavailable_or_disabled_retains_original_neutral_prior(monkeypatch):
    monkeypatch.setattr(nli, "NLI", False)
    forbidden = Mock(side_effect=AssertionError("No scoring when disabled"))
    state = {"claim": "Claim", "evidence": "Evidence", "scorer": forbidden}
    result = decide("Q10", state)
    assert (result.label, result.p, result.confidence, result.zone, result.backend) == (
        "unrelated", 1 / 3, 0, "uncertain", "fallback")
    forbidden.assert_not_called()
    monkeypatch.setattr(nli, "NLI", True)
    monkeypatch.setattr(nli, "default_scorer", Mock(side_effect=nli.NLIUnavailable("absent")))
    result = decide("Q10", {"claim": "Claim", "evidence": "Evidence"})
    assert (result.label, result.p, result.backend) == ("unrelated", 1 / 3, "fallback")
    with pytest.raises(NotConfigured):
        BACKENDS["nli"].run("Q10", {})


def test_invalid_configured_scores_surface_in_decide(monkeypatch):
    monkeypatch.setattr(nli, "NLI", True)
    with pytest.raises(ValueError):
        decide("Q10", {"claim": "Claim", "evidence": "Evidence", "scorer": lambda pairs: [(float("nan"), 0, 1)]})
