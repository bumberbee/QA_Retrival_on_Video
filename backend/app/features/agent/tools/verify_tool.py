"""Sentence support when local NLI is enabled; retain the original placeholder otherwise."""

from __future__ import annotations

import copy

from .. import nli

from .base import FunctionTool, ToolContext, ToolResult, object_schema

NLI_TODO = "TODO (A6): use multilingual NLI on each claim/evidence pair; no model verification has run."
_SENTENCES = nli._SENTENCES
_CITATIONS = nli._CITATIONS


def _verify(args: dict, ctx: ToolContext) -> dict:
    originals = [args["claims"]] if isinstance(args["claims"], str) else args["claims"]
    if nli.enabled():
        try:
            scorer = nli.default_scorer() if ctx.nli_scorer is None else ctx.nli_scorer
            report = nli.check_claims("\n".join(originals), nli.found_from_evidence_map(args["evidence_map"]),
                                      scorer=scorer)
            for row in report["claims"]:
                row["missing_evidence_ids"] = [key for key in dict.fromkeys(_CITATIONS.findall(row["text"]))
                                               if key not in args["evidence_map"]]
            overall = ("contradicted" if report["counts"]["contradicted"] else
                       "unsupported" if report["counts"]["unsupported"] else "supported")
            return {"status": overall, **report, "evidence_map": copy.deepcopy(args["evidence_map"]), "evidence": []}
        except Exception:
            pass  # Exact historical placeholder, including NLI_TODO, when unavailable.
    claims = []
    for original in originals:
        for sentence in _SENTENCES.split(original.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            citations = list(dict.fromkeys(_CITATIONS.findall(sentence)))
            claims.append({"claim_id": f"C{len(claims) + 1}", "text": sentence, "status": "unverified",
                           "evidence_ids": [cid for cid in citations if cid in args["evidence_map"]],
                           "missing_evidence_ids": [cid for cid in citations if cid not in args["evidence_map"]],
                           "todo": NLI_TODO})
    return {"status": "unverified", "claims": claims, "evidence_map": copy.deepcopy(args["evidence_map"]),
            "evidence": [], "todo": NLI_TODO}


_CLAIM = {"type": "string", "minLength": 1, "maxLength": 20000}
VERIFY_CLAIMS = FunctionTool("verify_claims", ("Check sentence support against evidence with local multilingual NLI."
                                             if nli.enabled() else
                                             "Split claims and mark them unverified; A6 NLI is not implemented."),
                             object_schema({"claims": {"anyOf": [_CLAIM, {"type": "array", "minItems": 1,
                                                                         "maxItems": 100, "items": _CLAIM}]},
                                            "evidence_map": {"type": "object", "additionalProperties": {"type": "object"}}},
                                           ["claims", "evidence_map"]), _verify)


def verify_claims(claims: str | list[str], evidence_map: dict, *, ctx: ToolContext | None = None) -> ToolResult:
    return VERIFY_CLAIMS.run({"claims": claims, "evidence_map": evidence_map}, ctx)
