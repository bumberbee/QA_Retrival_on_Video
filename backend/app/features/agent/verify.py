"""S6: arithmetic/citations and optional local multilingual sentence support.

A written equality whose left side does not evaluate to its right side is a mismatch
(``numcheck``). One regeneration may be requested with ``hint``; the final ``verdict``
is what the interface shows. LECTURE_NLI enables the additional Q10 support check.
"""

from __future__ import annotations

import re
from fractions import Fraction

from app.features.ask import numcheck, postprocess
from . import nli


# A rendered citation right after a number ("= 5 [01:05]") would be read as part of the value.
_CITATION = re.compile(r"\s*\[(?:E\d+|\d+:\d{2}(?::\d{2})?)\]")

# Vietnamese prompt constants, following ask/llm.py's codex-strings.json style.
CONTRADICTION_HINT = (
    "Lưu ý: câu trả lời trước có câu mâu thuẫn với bằng chứng: «{sentence}». "
    "Hãy trả lời lại, đối chiếu câu này với nguồn trích dẫn và chỉ nêu nội dung được bằng chứng hỗ trợ."
)


def _number(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def check(answer: str, found: dict, *, scorer=None) -> dict:
    """Arithmetic claims of the answer, each with its status and exact value."""
    try:
        claims = numcheck.check_answer(_CITATION.sub("", answer), list(postprocess.evidence_texts(found).values()))
    except Exception as error:   # a checker fault must not cost the user the answer
        report = {"claims": [], "mismatches": [], "error": f"{type(error).__name__}: {error}"}
    else:
        rows = [{"lhs": claim.lhs_expr, "rhs": claim.rhs_value, "status": claim.status,
                 "computed": _number(claim.computed_value), "operands_in_evidence": claim.operand_supported}
                for claim in claims]
        report = {"claims": rows, "mismatches": [row for row in rows if row["status"] == "mismatch"]}
    if nli.enabled():
        try:
            # S5 preserves E IDs before rendering timestamps, including in streams.
            active_scorer = nli.default_scorer() if scorer is None else scorer
            report["nli"] = nli.check_claims(found.get("nli_answer", answer), found, scorer=active_scorer)
        except Exception as error:  # unavailable assets, inference faults and bad outputs all fail open
            report["nli_error"] = f"{type(error).__name__}: {error}"
    return report


def hint(report: dict) -> str:
    """Text added to the prompt for the one regeneration."""
    if not report.get("mismatches") and report.get("nli", {}).get("counts", {}).get("contradicted", 0):
        contradicted = next(row for row in report["nli"]["claims"] if row["status"] == "contradicted")
        return CONTRADICTION_HINT.format(sentence=contradicted["text"])
    wrong = "; ".join(f"{row['lhs']} = {row['computed']}, không phải {row['rhs']}" for row in report["mismatches"])
    return ("Lưu ý: câu trả lời trước có phép tính sai. Tính chính xác thì " + wrong +
            ". Hãy trả lời lại, kiểm tra từng phép tính và chỉ dùng số liệu có trong bằng chứng.")


def verdict(report: dict, found: dict, *, evidence_enough: bool) -> dict:
    """label "verified" only when nothing below applies; reasons name what does."""
    reasons = []
    if not evidence_enough:
        reasons.append("weak_evidence")
    if report.get("mismatches"):
        reasons.append("numeric_mismatch")
    if found.get("invalid_citations"):
        reasons.append("invalid_citation")
    cited = found.get("cited_sources") or []
    if not cited:
        reasons.append("no_citation")
    elif all(source.get("inferred") for source in cited):
        reasons.append("inferred_citation")
    support = report.get("nli")
    if support:
        if support["counts"]["unsupported"]:
            reasons.append("unsupported_claim")
        if support["counts"]["contradicted"]:
            reasons.append("contradicted_claim")
    return {"label": "unverified" if reasons else "verified", "reasons": reasons,
            "numeric_claims": len(report.get("claims", [])),
            "numeric_checked": sum(row["status"] != "unparsed" for row in report.get("claims", [])),
            **({"nli": support} if support is not None else {}),
            **({"nli_error": report["nli_error"]} if "nli_error" in report else {})}
