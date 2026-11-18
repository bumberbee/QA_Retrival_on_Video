"""NLI metric contracts using only synthetic probabilities and a tiny UTF-8 CSV."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import nli_eval

FIXTURE = Path(__file__).parent / "fixtures/nli_tiny.csv"


def test_fake_csv_metrics_and_threshold_sweep_never_load_model(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Model loading forbidden"))
    monkeypatch.setattr(nli_eval.load_nli(), "default_scorer", forbidden)
    report = nli_eval.evaluate(nli_eval.read_csv(FIXTURE), scorer=nli_eval.fake_scorer)
    assert report["accuracy"] == report["macro_f1"] == 1
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert report["labels"] == ["supported", "unsupported", "contradicted"]
    assert report["scorer"] == "fake"
    assert len(report["threshold_sweep"]) == 25
    assert all(row["accuracy"] == 1 for row in report["threshold_sweep"])
    forbidden.assert_not_called()


def test_nonperfect_accuracy_macro_f1_and_confusion_orientation():
    report = nli_eval.metrics(["supported", "unsupported", "contradicted", "supported"],
                              ["supported", "supported", "unsupported", "contradicted"])
    assert report["accuracy"] == 0.25
    assert report["macro_f1"] == pytest.approx(1 / 6)
    assert report["confusion_matrix"] == [[1, 0, 1], [1, 0, 0], [0, 1, 0]]


def test_threshold_edges_share_production_status_and_sweep_scores_once():
    rows = [{"claim": "A", "evidence": "P", "label": "supported"},
            {"claim": "B", "evidence": "Q", "label": "contradicted"}]
    fake = Mock(return_value=[(0.6, 0.4, 0), (0.2, 0.3, 0.5)])
    report = nli_eval.evaluate(rows, scorer=fake, accept_sweep=(0.6, 0.60001), contradict_sweep=(0.5, 0.50001))
    assert report["accuracy"] == 1
    assert [row["accuracy"] for row in report["threshold_sweep"]] == [1, 0.5, 0.5, 0]
    fake.assert_called_once_with([("P", "A"), ("Q", "B")])


def test_cli_fake_mode_writes_prints_report_without_loading_model(monkeypatch, tmp_path, capsys):
    forbidden = Mock(side_effect=AssertionError("Model loading forbidden"))
    monkeypatch.setattr(nli_eval.load_nli(), "default_scorer", forbidden)
    target = tmp_path / "nli.json"
    report = nli_eval.main([str(FIXTURE), "--scorer-fake", "--output", str(target)])
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert json.loads(capsys.readouterr().out) == report
    forbidden.assert_not_called()


@pytest.mark.parametrize("content", ["claim,evidence\nA,B\n", "claim,evidence,label\nA,B,invalid\n",
                                     "claim,evidence,label\n", "claim,evidence,label\n,B,supported\n"])
def test_invalid_csv_rejected(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        nli_eval.read_csv(path)


@pytest.mark.parametrize("gold,predicted", [([], []), (["supported"], []), (["invalid"], ["supported"])])
def test_invalid_metric_inputs(gold, predicted):
    with pytest.raises(ValueError):
        nli_eval.metrics(gold, predicted)


def test_absent_classes_have_zero_f1():
    assert nli_eval.metrics(["supported"], ["supported"])["macro_f1"] == pytest.approx(1 / 3)
