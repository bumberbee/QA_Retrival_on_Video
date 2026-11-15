"""Bounded, deterministic arithmetic checks, using only the standard library.

``safe_eval`` returns a Fraction or raises ValueError. ``extract_claims`` returns
``(lhs_expr, rhs_value, (start, end))`` tuples; rhs_value is the *written string*
so trailing decimal zeroes and percentage precision are preserved. Spans use
Python string offsets, with an exclusive end. ``check_answer`` never raises for
malformed answer/evidence input and returns a list of ClaimCheck dataclasses.

Extraction covers plain arithmetic equalities, chains, and a parenthesized
percentage following an equality. It does not interpret LaTeX, prose reasoning,
or standalone numbers. Evidence support means the literal numeric operands on
the LHS occur as numbers in evidence, not that the evidence proves the formula.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, NamedTuple

__all__ = ["safe_eval", "extract_claims", "check_answer", "ClaimCheck"]

_MAX_EXPR_LENGTH = 4096
_MAX_TOKENS = 512
_MAX_AST_NODES = 512
_MAX_LITERAL_DIGITS = 100
_MAX_BITS = 4096
_MAX_EXPONENT = 100
_MAX_COMBINATORIAL_N = 1000
_MAX_TEXT_LENGTH = 100_000
_MAX_CLAIMS = 200
_MAX_EVIDENCE_LENGTH = 200_000
_FUNCTIONS = frozenset({"C", "P", "binom", "tohop"})
_FACTORIAL = "__numcheck_factorial"
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)\Z", re.ASCII)
_LEXER = re.compile(
    r"(?P<number>\d+(?:\.\d*)?|\.\d+)"
    r"|(?P<compact>C(?=\d))"
    r"|(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<symbol>\*\*|[+\-*/^(),!%])", re.ASCII,
)
_TEXT_TOKEN = re.compile(
    r"\d+(?:[.,]\d+)?|[.,]\d+|[^\W\d]\w*|\*\*|[^\s]", re.UNICODE,
)
_TEXT_NUMBER = re.compile(r"(?:\d+(?:[.,]\d+)?|[.,]\d+)\Z", re.ASCII)
_EQUALITY = re.compile(r"(?<![<>=!])(?:=|≈)(?!=)")
_ROUNDED = re.compile(
    r"[+-]?(?P<number>\d+(?:[.,]\d*)?|[.,]\d+)\s*(?P<percent>%)?\Z",
    re.ASCII,
)
_PAREN_PERCENT = re.compile(
    r"\s*\(\s*([+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*%)\s*\)", re.ASCII,
)
_OPERATORS = frozenset({"+", "-", "*", "/", "^", "**", "!", "%"})


class _Token(NamedTuple):
    text: str
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class ClaimCheck:
    lhs_expr: str
    rhs_value: str
    span: tuple[int, int]
    status: Literal["ok", "mismatch", "unparsed"]
    computed_value: Fraction | None
    claimed_value: Fraction | None
    operand_supported: bool
    operand_support: dict[str, bool]
    error: str | None = None


def _lex(expr: str) -> list[_Token]:
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("expression must be a nonempty string")
    if len(expr) > _MAX_EXPR_LENGTH:
        raise ValueError("expression is too long")
    tokens: list[_Token] = []
    pos = 0
    while pos < len(expr):
        if expr[pos].isspace():
            pos += 1
            continue
        match = _LEXER.match(expr, pos)
        if match is None:
            raise ValueError("unsupported character or syntax")
        tokens.append(_Token(match[0], pos, match.end(), match.lastgroup or ""))
        pos = match.end()
        if len(tokens) > _MAX_TOKENS:
            raise ValueError("too many expression tokens")

    # A comma is an argument separator inside combinatorial calls. Elsewhere,
    # contiguous digits on either side make it a Vietnamese decimal comma.
    result: list[_Token] = []
    frames: list[bool] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.text == "(":
            frames.append(i > 0 and tokens[i - 1].text in _FUNCTIONS)
        elif token.text == ")" and frames:
            frames.pop()
        if (not any(frames) and i + 2 < len(tokens)
                and token.kind == "number" and token.text.isdigit()
                and tokens[i + 1].text == ","
                and tokens[i + 2].kind == "number"
                and tokens[i + 2].text.isdigit()
                and token.end == tokens[i + 1].start
                and tokens[i + 1].end == tokens[i + 2].start):
            last = tokens[i + 2]
            result.append(_Token(token.text + "." + last.text,
                                 token.start, last.end, "number"))
            i += 3
            continue
        result.append(token)
        i += 1
    for token in result:
        if token.kind == "number":
            if sum(c.isdigit() for c in token.text) > _MAX_LITERAL_DIGITS:
                raise ValueError("numeric literal is too large")
        elif token.kind in {"name", "compact"} and token.text not in _FUNCTIONS:
            raise ValueError("unsupported name")
    return result


def _atom_start(parts: list[str]) -> int:
    if not parts:
        raise ValueError("postfix operator needs an operand")
    if _NUMBER.fullmatch(parts[-1]):
        return len(parts) - 1
    if parts[-1] != ")":
        raise ValueError("postfix operator needs a number or parenthesized atom")
    depth = 0
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == ")":
            depth += 1
        elif parts[i] == "(":
            depth -= 1
            if depth == 0:
                if i and parts[i - 1] in _FUNCTIONS | {_FACTORIAL}:
                    return i - 1
                return i
    raise ValueError("unbalanced parentheses")


def _normalize(tokens: list[_Token]) -> str:
    parts: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if (i + 2 < len(tokens) and token.kind == "number"
                and tokens[i + 1].text == "C" and tokens[i + 2].kind == "number"):
            parts.extend(["C", "(", token.text, ",", tokens[i + 2].text, ")"])
            i += 3
            continue
        if token.text in {"!", "%"}:
            start = _atom_start(parts)
            atom = parts[start:]
            if token.text == "!":
                parts[start:] = [_FACTORIAL, "(", *atom, ")"]
            else:
                parts[start:] = ["(", *atom, "/", "100", ")"]
        elif token.text == "^":
            parts.append("**")
        elif token.kind == "number":
            whole, dot, decimal = token.text.partition(".")
            parts.append(str(int(whole or "0")) + (dot + decimal if dot else ""))
        else:
            parts.append(token.text)
        i += 1
    return " ".join(parts)


def _bounded(value: Fraction) -> Fraction:
    if (value.numerator.bit_length() > _MAX_BITS
            or value.denominator.bit_length() > _MAX_BITS):
        raise ValueError("arithmetic result is too large")
    return value


def _natural(value: Fraction) -> int:
    if value.denominator != 1 or not 0 <= value <= _MAX_COMBINATORIAL_N:
        raise ValueError("combinatorial arguments must be integers from 0 to 1000")
    return value.numerator


def _evaluate(node: ast.AST, source: str) -> Fraction:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        literal = ast.get_source_segment(source, node) or ""
        if not _NUMBER.fullmatch(literal):
            raise ValueError("unsupported numeric literal")
        return _bounded(Fraction(literal))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand, source)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            raise ValueError("unsupported arithmetic operator")
        left, right = _evaluate(node.left, source), _evaluate(node.right, source)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        elif isinstance(node.op, ast.Div):
            value = left / right
        else:
            if right.denominator != 1 or abs(right) > _MAX_EXPONENT:
                raise ValueError("exponent must be an integer from -100 to 100")
            exponent = right.numerator
            if max(left.numerator.bit_length(), left.denominator.bit_length()) * abs(exponent) > 2 * _MAX_BITS:
                raise ValueError("power result is too large")
            value = left ** exponent
        return _bounded(value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        arity = 1 if name == _FACTORIAL else 2
        if name not in _FUNCTIONS | {_FACTORIAL} or node.keywords or len(node.args) != arity:
            raise ValueError("unsupported function or arguments")
        args = [_natural(_evaluate(arg, source)) for arg in node.args]
        if name == _FACTORIAL:
            value = math.factorial(args[0])
        elif name == "P":
            value = math.perm(*args)
        else:
            value = math.comb(*args)
        return _bounded(Fraction(value))
    raise ValueError("unsupported expression node")


def safe_eval(expr: str) -> Fraction:
    """Evaluate arithmetic exactly; reject all code and unsupported syntax.

    Supports + - * / ^ **, unary signs, parentheses, decimal comma outside
    combinatorial calls, postfix %, n!, C(n,k), numeric nCk, binom, tohop and
    P(n,k). No Python eval/exec is used. Limits: 4096 characters, 512 tokens/AST
    nodes, 100 literal digits, 4096-bit results, integer exponents +/-100, and
    combinatorial/factorial arguments 0..1000. All failures raise ValueError.
    """
    try:
        source = _normalize(_lex(expr))
        tree = ast.parse(source, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            raise ValueError("too many expression nodes")
        return _evaluate(tree.body, source)
    except (SyntaxError, ZeroDivisionError, OverflowError, RecursionError, TypeError) as exc:
        raise ValueError("invalid or unsafe arithmetic expression") from exc


def _math_name(token: str) -> bool:
    return token in _FUNCTIONS or re.fullmatch(r"C\d+", token, re.ASCII) is not None


def _left_candidate(text: str, end: int) -> tuple[str, int]:
    start = max(text.rfind("\n", 0, end) + 1, end - _MAX_EXPR_LENGTH)
    fragment = text[start:end].rstrip(" \t\r`$")
    tokens = list(_TEXT_TOKEN.finditer(fragment))
    depth = 0
    first = len(fragment)
    for i in range(len(tokens) - 1, -1, -1):
        value = tokens[i][0]
        following = tokens[i + 1][0] if i + 1 < len(tokens) else ""
        if value == ")":
            depth += 1
        elif value == "(":
            depth = max(0, depth - 1)
        elif depth:
            pass  # Keep malformed contents, rather than accepting a valid suffix.
        elif _TEXT_NUMBER.fullmatch(value) or value in _OPERATORS:
            pass
        elif (value == "." and following
              and (following[0].isalpha() or following[0] == "_")
              and tokens[i].end() == tokens[i + 1].start()):
            pass  # Preserve forbidden attributes so they are reported unparsed.
        elif value[0].isalpha() or value[0] == "_":
            if not (_math_name(value) or i == len(tokens) - 1
                    or following in _OPERATORS | {"(", "."}):
                break
        else:
            break
        first = tokens[i].start()
    return fragment[first:], start + first


def _right_candidate(text: str, start: int) -> tuple[str, int]:
    newline = text.find("\n", start)
    end = min(len(text), start + _MAX_EXPR_LENGTH, newline if newline >= 0 else len(text))
    fragment = text[start:end]
    tokens = list(_TEXT_TOKEN.finditer(fragment))
    depth = 0
    first = tokens[0].start() if tokens else 0
    last = first
    previous = ""
    for i, token in enumerate(tokens):
        value = token[0]
        following = tokens[i + 1][0] if i + 1 < len(tokens) else ""
        if value in {"=", "≈"}:
            break
        if value == "(" and not depth and previous and (
                _TEXT_NUMBER.fullmatch(previous) or previous in {")", "%", "!"}):
            if _PAREN_PERCENT.match(fragment, token.start()):
                break
        if value == "(":
            depth += 1
        elif value == ")":
            if not depth:
                break
            depth -= 1
        elif depth:
            pass
        elif _TEXT_NUMBER.fullmatch(value) or value in _OPERATORS:
            pass
        elif (value == "." and previous and following
              and (following[0].isalpha() or following[0] == "_")
              and token.end() == tokens[i + 1].start()):
            pass
        elif value in {"[", "]", "{", "}", "<", ">", "&", "|", "~", "@"}:
            pass  # Retain forbidden syntax instead of checking only its prefix.
        elif value[0].isalpha() or value[0] == "_":
            if not (_math_name(value) or i == 0 or previous in _OPERATORS | {"."}
                    or following == "(" or (previous and token.start() == last)):
                break
        else:
            break
        last = token.end()
        previous = value
    return fragment[first:last].strip(), start + last


def _parseable(expr: str) -> bool:
    try:
        safe_eval(expr)
        return True
    except ValueError:
        return False


def extract_claims(text: str) -> list[tuple[str, str, tuple[int, int]]]:
    """Extract equalities without interpreting prose; malformed ones are retained.

    In a chain each RHS is checked against the original computable LHS. If a
    chain starts with a symbolic label, its first computable RHS becomes the
    LHS for subsequent links. Continuation spans start at their equality sign.
    Invalid input returns []; scanning is bounded to 100,000 characters and
    200 claims. A naked ``= 0.148`` has an empty LHS and will be unparsed.
    """
    if not isinstance(text, str):
        return []
    text = text[:_MAX_TEXT_LENGTH]
    claims: list[tuple[str, str, tuple[int, int]]] = []
    previous_end = -1
    anchor = ""
    previous_rhs = ""
    for equality in _EQUALITY.finditer(text):
        if len(claims) >= _MAX_CLAIMS:
            break
        lhs, left_start = _left_candidate(text, equality.start())
        chain = previous_end >= 0 and not text[previous_end:equality.start()].strip(" \t\r\n`$")
        if chain:
            lhs = anchor if _parseable(anchor) else previous_rhs
            left_start = equality.start()
        rhs, right_end = _right_candidate(text, equality.end())
        claims.append((lhs, rhs, (left_start, right_end)))
        anchor, previous_rhs, previous_end = lhs, rhs, right_end
        alternate = _PAREN_PERCENT.match(text, right_end)
        if alternate and len(claims) < _MAX_CLAIMS:
            claims.append((lhs, alternate[1], (right_end, alternate.end())))
            previous_end = alternate.end()
    return claims


def _rounding_tolerance(written: str) -> Fraction:
    match = _ROUNDED.fullmatch(written.strip())
    if not match:
        return Fraction(0)
    number = match["number"].replace(",", ".")
    if "." not in number and not match["percent"]:
        return Fraction(0)  # An integer equality is exact.
    digits = len(number.partition(".")[2])
    scale = 100 if match["percent"] else 1
    return Fraction(1, 2 * 10 ** digits * scale)


def _source_numbers(text: str) -> list[tuple[str, Fraction]]:
    """Read numeric tokens in prose, distinguishing C(12,4) from 12,4."""
    numbers: list[tuple[str, Fraction]] = []
    frames: list[bool] = []
    previous = ""
    previous_end = -1
    for token in _TEXT_TOKEN.finditer(text):
        value = token[0]
        if value == "(":
            frames.append(previous in _FUNCTIONS)
        elif value == ")" and frames:
            frames.pop()
        if _TEXT_NUMBER.fullmatch(value):
            following_char = text[token.end():token.end() + 1]
            compact = re.match(r"C\d+(?!\w)", text[token.end():], re.ASCII)
            # Avoid numbers embedded in words, but permit numeric 12C4 notation.
            if following_char and (following_char.isalpha() or following_char == "_") and not compact:
                previous, previous_end = value, token.end()
                continue
            pieces = value.split(",") if any(frames) and "," in value else [value]
            for piece in pieces:
                if sum(c.isdigit() for c in piece) <= _MAX_LITERAL_DIGITS:
                    try:
                        numbers.append((piece, Fraction(piece.replace(",", "."))))
                    except ValueError:
                        pass
        elif (re.fullmatch(r"C\d+", value, re.ASCII)
              and _TEXT_NUMBER.fullmatch(previous) and previous_end == token.start()):
            digits = value[1:]
            if len(digits) <= _MAX_LITERAL_DIGITS:
                numbers.append((digits, Fraction(digits)))
        previous, previous_end = value, token.end()
    return numbers


def _evidence_values(evidence_texts: object) -> set[Fraction]:
    values: set[Fraction] = set()
    if isinstance(evidence_texts, str):
        evidence_texts = (evidence_texts,)
    remaining = _MAX_EVIDENCE_LENGTH
    try:
        for i, evidence in enumerate(evidence_texts):  # type: ignore[arg-type]
            if i >= 1000 or remaining <= 0:
                break
            if isinstance(evidence, str):
                evidence = evidence[:remaining]
                remaining -= len(evidence)
                values.update(value for _, value in _source_numbers(evidence))
    except Exception:
        # Bad/noniterable evidence cannot prevent verification of the answer.
        pass
    return values


def check_answer(text: str, evidence_texts=()) -> list[ClaimCheck]:
    """Check extracted claims; malformed arithmetic produces ``unparsed``.

    Fractions/arithmetic RHSs and integers are exact. A decimal RHS has an
    inclusive half-unit rounding tolerance in its last displayed place;
    percentages use the same rule scaled by 1/100. No floating point is used.
    ``operand_support`` maps each distinct written LHS numeric literal to its
    presence (by exact numeric value) in evidence. ``operand_supported`` is
    true only when there is at least one operand and all operands are present.
    With no evidence it is false, independently of arithmetic status.
    """
    evidence = _evidence_values(evidence_texts)
    results: list[ClaimCheck] = []
    for lhs, rhs, span in extract_claims(text):
        try:
            operands = [(lhs[t.start:t.end], Fraction(t.text))
                        for t in _lex(lhs) if t.kind == "number"]
        except ValueError:
            operands = _source_numbers(lhs)
        support = {written: value in evidence for written, value in operands}
        computed = claimed = None
        error = None
        status: Literal["ok", "mismatch", "unparsed"] = "unparsed"
        try:
            computed = safe_eval(lhs)
            claimed = safe_eval(rhs)
            status = "ok" if abs(computed - claimed) <= _rounding_tolerance(rhs) else "mismatch"
        except ValueError as exc:
            error = str(exc)
        results.append(ClaimCheck(lhs, rhs, span, status, computed, claimed,
                                  bool(support) and all(support.values()), support, error))
    return results
