"""S6 without a model: check the arithmetic and the citations of a generated answer.

A written equality whose left side does not evaluate to its right side is a mismatch
(``numcheck``). One regeneration may be requested with ``hint``; the final ``verdict``
is what the interface shows. Whether each sentence is supported by its source (Q10,
NLI) is not checked here.
"""

from __future__ import annotations

import re
from fractions import Fraction

from app.features.ask import numcheck, postprocess


# A rendered citation right after a number ("= 5 [01:05]") would be read as part of the value.
_CITATION = re.compile(r"\s*\[(?:E\d+|\d+:\d{2}(?::\d{2})?)\]")


def _number(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def check(answer: str, found: dict) -> dict:
    """Arithmetic claims of the answer, each with its status and exact value."""
    try:
        claims = numcheck.check_answer(_CITATION.sub("", answer), list(postprocess.evidence_texts(found).values()))
    except Exception as error:   # a checker fault must not cost the user the answer
        return {"claims": [], "mismatches": [], "error": f"{type(error).__name__}: {error}"}
    rows = [{"lhs": claim.lhs_expr, "rhs": claim.rhs_value, "status": claim.status,
             "computed": _number(claim.computed_value), "operands_in_evidence": claim.operand_supported}
            for claim in claims]
    return {"claims": rows, "mismatches": [row for row in rows if row["status"] == "mismatch"]}


def hint(report: dict) -> str:
    """Text added to the prompt for the one regeneration."""
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
    return {"label": "unverified" if reasons else "verified", "reasons": reasons,
            "numeric_claims": len(report.get("claims", [])),
            "numeric_checked": sum(row["status"] != "unparsed" for row in report.get("claims", []))}
