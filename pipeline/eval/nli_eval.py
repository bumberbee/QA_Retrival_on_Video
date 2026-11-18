"""Offline claim/evidence CSV evaluation. No downloads; --scorer-fake loads no model."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LABELS = ("supported", "unsupported", "contradicted")


@lru_cache(maxsize=1)
def load_nli():
    # Import only the scorer source, never app.main/storage/dotenv or HTTP adapters.
    path = REPO / "backend/app/features/agent/nli.py"
    spec = importlib.util.spec_from_file_location("_lecture_nli_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_thresholds():
    # The bundled JSON mirrors YAML; backend tests enforce their equivalence.
    entry = json.loads((REPO / "backend/app/features/decisions/agent_policy.json").read_text(encoding="utf-8"))["Q10"]
    return entry.get("accept", 0.6), entry.get("reject_contradict", 0.5)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"claim", "evidence", "label"} <= set(reader.fieldnames or []):
            raise ValueError("CSV requires claim, evidence, label columns")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV must contain at least one labeled pair")
    for index, row in enumerate(rows, 2):
        if (row.get("label") not in LABELS or not (row.get("claim") or "").strip()
                or not (row.get("evidence") or "").strip()):
            raise ValueError(f"Invalid claim/evidence/label at CSV row {index}")
    return rows


def metrics(gold, predicted) -> dict:
    gold, predicted = list(gold), list(predicted)
    if not gold or len(gold) != len(predicted) or any(label not in LABELS for label in gold + predicted):
        raise ValueError("Metrics require equal nonempty lists of NLI labels")
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for actual, prediction in zip(gold, predicted):
        matrix[LABELS.index(actual)][LABELS.index(prediction)] += 1
    f1 = []
    for index in range(len(LABELS)):
        true_positive = matrix[index][index]
        denominator = sum(matrix[index]) + sum(row[index] for row in matrix)
        f1.append(2 * true_positive / denominator if denominator else 0.0)
    return {"n": len(gold), "accuracy": sum(matrix[i][i] for i in range(len(LABELS))) / len(gold),
            "macro_f1": sum(f1) / len(LABELS), "labels": list(LABELS), "confusion_matrix": matrix}


def fake_scorer(pairs):
    """Synthetic smoke test only: exact equality entails, FAKE_CONTRADICT: contradicts.

    Everything else is neutral. This does not inspect the CSV's gold labels and
    is not a multilingual model or a measurement of real answer support.
    """
    return [(0.9, 0.05, 0.05) if premise.strip() == hypothesis.strip() else
            (0.05, 0.05, 0.9) if premise.startswith("FAKE_CONTRADICT:") else (0.1, 0.8, 0.1)
            for premise, hypothesis in pairs]


def evaluate(rows, *, scorer=None, accept=None, reject_contradict=None,
             accept_sweep=(0.4, 0.5, 0.6, 0.7, 0.8), contradict_sweep=(0.3, 0.4, 0.5, 0.6, 0.7)) -> dict:
    nli = load_nli()
    defaults = default_thresholds()
    accept = defaults[0] if accept is None else accept
    reject_contradict = defaults[1] if reject_contradict is None else reject_contradict
    import math

    if any(not math.isfinite(value) or not 0 <= value <= 1
           for value in (accept, reject_contradict, *accept_sweep, *contradict_sweep)):
        raise ValueError("Thresholds must be finite and within [0, 1]")
    gold = [row["label"] for row in rows]
    if not gold or any(label not in LABELS for label in gold):
        raise ValueError("Evaluation requires nonempty labeled pairs")
    scores = nli.score_pairs([(row["evidence"], row["claim"]) for row in rows], scorer=scorer)

    def at(accept_value, contradict_value):
        predicted = [nli.status(entail, contradict, accept=accept_value, reject=contradict_value)
                     for entail, _, contradict in scores]
        return {"accept": accept_value, "reject_contradict": contradict_value, **metrics(gold, predicted)}

    return {**at(accept, reject_contradict), "scorer": "fake" if scorer is fake_scorer else
            "injected" if scorer is not None else "local_onnx",
            "threshold_sweep": [at(a, c) for a in accept_sweep for c in contradict_sweep],
            "pairs": [{"claim": row["claim"], "evidence": row["evidence"], "label": row["label"],
                       "entail": score[0], "neutral": score[1], "contradict": score[2]}
                      for row, score in zip(rows, scores)]}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--scorer-fake", action="store_true", help="Deterministic synthetic scorer; no model load")
    parser.add_argument("--accept", type=float)
    parser.add_argument("--reject-contradict", type=float)
    parser.add_argument("--accept-sweep", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--contradict-sweep", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--output", type=Path, help="Optional JSON report (also printed to stdout)")
    args = parser.parse_args(argv)
    report = evaluate(read_csv(args.csv), scorer=fake_scorer if args.scorer_fake else None,
                      accept=args.accept, reject_contradict=args.reject_contradict,
                      accept_sweep=args.accept_sweep, contradict_sweep=args.contradict_sweep)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return report


if __name__ == "__main__":
    main()
