from tests import _isolate  # noqa: F401

import copy

import pytest

from app.features.decisions import BACKENDS, POLICY, NotConfigured, decide


@pytest.mark.parametrize("score,label,zone", [(0, "irrelevant", "reject"),
                                             (0.2, "irrelevant", "reject"),
                                             (0.20001, "insufficient", "uncertain"),
                                             (0.59999, "insufficient", "uncertain"),
                                             (0.6, "enough", "accept"),
                                             (1, "enough", "accept")])
def test_q8_score_boundaries(score, label, zone):
    decision = decide("Q8", {"best_rerank": score})
    assert (decision.label, decision.p, decision.zone, decision.backend) == (label, score, zone, "reranker")


@pytest.mark.parametrize("score,label", [(0, "drop"), (0.2, "drop"), (0.59999, "drop"),
                                       (0.6, "keep"), (1, "keep")])
def test_q9_keeps_individual_chunks_at_policy_threshold(score, label):
    decision = decide("Q9", {"chunk": {"chunk_id": "c", "rerank_score": score}})
    assert (decision.label, decision.p, decision.backend) == (label, score, "reranker")


def test_injected_policy_controls_labels_and_zones_without_mutation():
    policy = copy.deepcopy(POLICY)
    policy["Q8"].update(accept=0.8, reject=0.3)
    policy["Q9"].update(accept=0.75)
    assert decide("Q8", {"best_rerank": 0.7}, policy=policy).label == "insufficient"
    assert decide("Q8", {"best_rerank": 0.3}, policy=policy).label == "irrelevant"
    assert decide("Q8", {"best_rerank": 0.8}, policy=policy).label == "enough"
    assert decide("Q9", {"rerank_score": 0.7}, policy=policy).label == "drop"
    assert decide("Q9", {"rerank_score": 0.75}, policy=policy).label == "keep"
    assert POLICY["Q8"]["accept"] == 0.6
    assert POLICY["Q9"]["accept"] == 0.6


def test_nested_retrieval_score_is_supported():
    assert decide("Q8", {"retrieval": {"best_rerank": 0.9}}).label == "enough"


@pytest.mark.parametrize("name,state", [("Q8", {}), ("Q8", {"best_rerank": None}),
                                       ("Q9", {"chunk": {"chunk_id": "c"}}),
                                       ("Q9", {"best_rerank": 0.9})])
def test_absent_scores_raise_not_configured_and_preserve_neutral_fallback(name, state):
    with pytest.raises(NotConfigured):
        BACKENDS["reranker"].run(name, state)
    decision = decide(name, state)
    assert decision.backend == "fallback"
    assert decision.confidence == 0
    assert decision.zone == "uncertain"


@pytest.mark.parametrize("name,key", [("Q8", "best_rerank"), ("Q9", "rerank_score")])
@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_invalid_scores_do_not_silently_fall_back(name, key, score):
    with pytest.raises(ValueError):
        decide(name, {key: score})
