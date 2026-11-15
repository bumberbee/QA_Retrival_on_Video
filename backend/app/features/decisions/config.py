"""Load YAML when installed, otherwise the equivalent bundled JSON."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_config(stem: str, directory: Path = ROOT) -> dict:
    try:
        import yaml
    except ImportError:
        return json.loads((directory / f"{stem}.json").read_text(encoding="utf-8"))
    return yaml.safe_load((directory / f"{stem}.yaml").read_text(encoding="utf-8"))


def load_policy(directory: Path = ROOT) -> dict:
    policy = load_config("agent_policy", directory)
    for name in (f"Q{i}" for i in range(1, 13)):
        entry = policy[name]
        if entry["kind"] not in {"binary", "choice"}:
            raise ValueError(f"Invalid decision kind: {name}")
        if (any(not isinstance(label, str) for label in entry["labels"])
                or len(set(entry["labels"])) != len(entry["labels"]) or len(entry["labels"]) < 2):
            raise ValueError(f"Invalid labels: {name}")
        if entry["kind"] == "binary" and len(entry["labels"]) != 2:
            raise ValueError(f"Binary decision needs two labels: {name}")
        if not 0 <= entry.get("reject", 0.2) < entry.get("accept", 0.6) <= 1:
            raise ValueError(f"Invalid thresholds: {name}")
        if entry["fallback_label"] not in entry["labels"]:
            raise ValueError(f"Invalid fallback label: {name}")
    return policy
