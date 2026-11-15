"""Exact, bounded arithmetic for agent tool calls."""

from app.features.ask.numcheck import safe_eval


def calc(expr: str) -> dict:
    """Return a JSON-compatible result; invalid or unsafe input never raises."""
    try:
        value = safe_eval(expr)
        return {
            "ok": True,
            "value_fraction": f"{value.numerator}/{value.denominator}",
            "value_float": float(value),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "value_fraction": None,
            "value_float": None,
            "error": str(exc),
        }
