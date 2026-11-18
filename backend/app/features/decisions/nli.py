"""Q10 adapter to the shared local sentence-support scorer; importing loads no model."""

from .config import load_policy
from .types import BackendResult, NotConfigured


class NLIBackend:
    def __init__(self, policy=None, *, scorer=None):
        self.policy = load_policy() if policy is None else policy
        self.scorer = scorer

    def run(self, name: str, state: dict) -> BackendResult:
        from app.features.agent import nli

        if name != "Q10" or not nli.enabled():
            raise NotConfigured(f"NLI is not configured for {name}")
        scorer = state.get("scorer", self.scorer)
        if "claim" in state and "evidence" in state:
            answer = state["claim"]
            evidence = state["evidence"]
            evidence_map = {"E1": {"text": evidence}} if isinstance(evidence, str) else evidence
        elif "claims" in state and "evidence_map" in state:
            claims = state["claims"]
            answer = "\n".join(claims) if isinstance(claims, list) else claims
            evidence_map = state["evidence_map"]
        else:
            raise NotConfigured("Q10 requires claim/evidence or claims/evidence_map")
        try:
            scorer = nli.default_scorer() if scorer is None else scorer
            entry = self.policy[name]
            report = nli.check_claims(answer, nli.found_from_evidence_map(evidence_map), scorer=scorer,
                                      accept=entry.get("accept", 0.6), reject=entry.get("reject_contradict", 0.5))
        except nli.NLIUnavailable as exc:
            raise NotConfigured("No local NLI scorer configured for Q10") from exc
        if not report["claims"]:
            raise NotConfigured("Q10 requires at least one claim")
        # For multiple claims, choose the worst status; p is support of that claim.
        severity = {"supported": 0, "unsupported": 1, "contradicted": 2}
        worst = max(report["claims"], key=lambda row: (severity[row["status"]], -row["entail"]))
        return BackendResult(worst["status"], worst["entail"],
                             {"entail": worst["entail"], "contradict": worst["contradict"]})
