from tests import _isolate  # noqa: F401 (must come before app imports)
import json
import unittest
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from app.features.ask.numcheck import check_answer, extract_claims, safe_eval


CARD_PROBABILITY = json.loads(
    (Path(__file__).parent / "fixtures" / "card_probability.json").read_text(encoding="utf-8")
)


class SafeEvalTests(unittest.TestCase):
    def test_exact_arithmetic_and_precedence(self):
        cases = {
            "0.1 + 0.2": Fraction(3, 10),
            "0.123456789012345678901234567890": Fraction("0.123456789012345678901234567890"),
            "(3 + 5) * 2 / 4 - 1": Fraction(3),
            "2^3^2": Fraction(512),
            "2**-3": Fraction(1, 8),
            "-2^2": Fraction(-4),
            "(-2)^2": Fraction(4),
            "0002 + .5": Fraction(5, 2),
            "50% + 25%": Fraction(3, 4),
            "(1 + 1)%": Fraction(1, 50),
            "14,8%": Fraction(37, 250),
            "1,5 + 2,25": Fraction(15, 4),
            "(14,8) / 100": Fraction(37, 250),
            "4!": Fraction(24),
            "(2+3)! / 4!": Fraction(5),
            "-3!": Fraction(-6),
            "2^3!": Fraction(64),
            "0!": Fraction(1),
        }
        for expr, expected in cases.items():
            with self.subTest(expr=expr):
                self.assertEqual(safe_eval(expr), expected)
                self.assertIsInstance(safe_eval(expr), Fraction)

    def test_combinatorics_verified_independently(self):
        # 12*11*10*9 / 24 = 495; 15*14*13*12 / 24 = 1365.
        self.assertEqual(safe_eval("C(12,4)"), Fraction(495))
        self.assertEqual(safe_eval("C(15,4)"), Fraction(1365))
        self.assertEqual(safe_eval("C(12,4)/C(15,4)"), Fraction(33, 91))
        self.assertEqual(safe_eval("495/1365"), Fraction(33, 91))
        self.assertEqual(safe_eval("1 - C(12,4)/C(15,4)"), Fraction(58, 91))
        for expr in ("12C4", "12 C 4", "binom(12,4)", "tohop(12, 4)"):
            with self.subTest(expr=expr):
                self.assertEqual(safe_eval(expr), Fraction(495))
        self.assertEqual(safe_eval("P(12,4)"), Fraction(11880))
        self.assertEqual(safe_eval("C(5,7) + P(5,7)"), Fraction(0))
        self.assertEqual(safe_eval("C(2+3, 2)"), Fraction(10))

    def test_rejects_code_and_every_nonallowlisted_construct(self):
        expressions = (
            "__import__('os').system('echo unsafe')", "open('file')", "eval('2+2')",
            "C.__globals__", "(1).__class__", "abs(-1)", "factorial(3)",
            "__numcheck_factorial(3)", "sum([1,2])", "[x for x in (1,2)]",
            "x+1", "True+1", "None", "lambda: 1", "1 if 1 else 0",
            "C(n,4)", "C(n=12,k=4)", "C(*(12,4))", "C(12)", "P(12,4,1)",
            "1//2", "5%2", "1<<2", "1&2", "1 or 2", "1==1", "1;2",
            "1e3", "0x10", "1_000", "1j", "(1, 2)", "(1,2,3)", "C(1,5,2)",
            "C(1.5,1)", "C(-1,0)", "(-1)!", "1/0", "0**-1",
            "2**0.5", "2^101", "2^-101", "1001!", "C(1001,1)",
            "P(1000,1000)", "9" * 101, "9" * 100 + "**100",
            "(" * 300 + "1" + ")" * 300, "+".join(["1"] * 300),
            " " * 4097 + "1", "", None, 1, [],
        )
        for expr in expressions:
            with self.subTest(expr=repr(expr)[:100]):
                with self.assertRaises(ValueError):
                    safe_eval(expr)


class ClaimTests(unittest.TestCase):
    def test_extracts_chain_and_offsets(self):
        text = "Kết quả: C(12,4)/C(15,4) = 495/1365 = 33/91\n= 0.363 = 36,3%."
        claims = extract_claims(text)
        self.assertEqual([c[0] for c in claims], ["C(12,4)/C(15,4)"] * 4)
        self.assertEqual([c[1] for c in claims], ["495/1365", "33/91", "0.363", "36,3%"])
        self.assertEqual(text[slice(*claims[0][2])], "C(12,4)/C(15,4) = 495/1365")
        self.assertEqual(text[slice(*claims[-1][2])], "= 36,3%")
        self.assertEqual([c.status for c in check_answer(text)], ["ok"] * 4)

    def test_regression_real_saved_card_explanation(self):
        checks = check_answer(CARD_PROBABILITY["wrong_explanation"],
                              CARD_PROBABILITY["evidence_texts"])
        self.assertEqual(len(checks), 3)
        self.assertEqual([c.status for c in checks], ["mismatch", "ok", "mismatch"])
        # The original also miscalculates 3*1365*3003 as 12789105.
        self.assertEqual(checks[0].computed_value, Fraction(12297285))
        self.assertEqual(checks[0].claimed_value, Fraction(12789105))
        self.assertEqual(checks[-1].computed_value, Fraction(12789105, 30045015))
        self.assertEqual(checks[-1].claimed_value, Fraction(99, 667))
        self.assertEqual(safe_eval(CARD_PROBABILITY["erroneous_expression"]),
                         Fraction(273, 667))
        source = CARD_PROBABILITY["source"]
        self.assertEqual(source["file"], "D:/thesis-data/demo-notebook/final2/quality-review.json")
        self.assertEqual(source["key"], "results[5].answer")

    def test_regression_corrected_saved_card_explanation(self):
        checks = check_answer(CARD_PROBABILITY["corrected_explanation"],
                              CARD_PROBABILITY["evidence_texts"])
        self.assertEqual(len(checks), 3)
        self.assertEqual([c.status for c in checks], ["ok"] * 3)
        self.assertEqual([c.computed_value for c in checks],
                         [Fraction(4459455), Fraction(30045015), Fraction(99, 667)])
        self.assertEqual(safe_eval(CARD_PROBABILITY["correct_expression"]), Fraction(99, 667))
        # Correct the exclusion constraint as well as its arithmetic. Keeping
        # C(15,4) and changing the final answer to 273/667 is not a valid repair.
        self.assertIn(r"C_{12}^4", CARD_PROBABILITY["corrected_explanation"])
        self.assertIn("12 tấm thẻ chẵn không chia hết cho 10", CARD_PROBABILITY["corrected_explanation"])

    def test_card_count_independently_enumerates_odd_and_even_hands(self):
        cards = range(1, 31)
        odds = [card for card in cards if card % 2]
        evens = [card for card in cards if not card % 2]
        odd_hands = sum(1 for _ in combinations(odds, 5))
        even_hands = sum(sum(card % 10 == 0 for card in hand) == 1
                         for hand in combinations(evens, 5))
        self.assertEqual(odd_hands, 3003)
        self.assertEqual(even_hands, 1485)  # 3 choices times 495, excluding all {10,20,30}.
        proof = CARD_PROBABILITY["brute_force"]
        self.assertEqual(odd_hands * even_hands, proof["favorable_hands"])
        self.assertEqual(Fraction(odd_hands * even_hands, proof["total_hands"]),
                         Fraction(CARD_PROBABILITY["correct_answer"]))

    def test_latex_numeric_chains_preserve_original_offsets(self):
        text = r"Trước: 2+2 = 4; \[ P(A) = \frac{C_3^1 \times C_{12}^4 \times C_{15}^5}{C_{30}^{10}} = \frac{99}{667}. \] Sau: 1/2 = 0.5"
        checks = check_answer(text)
        self.assertEqual([c.status for c in checks], ["ok"] * 3)
        self.assertEqual(checks[1].lhs_expr, "(C(3,1) * C(12,4) * C(15,5))/(C(30,10))")
        self.assertEqual(text[slice(*checks[1].span)],
                         r"\frac{C_3^1 \times C_{12}^4 \times C_{15}^5}{C_{30}^{10}} = \frac{99}{667}")
        text = r"\[ 1/2 = \frac{99}{667} = 0.148 \]"
        self.assertEqual([c.status for c in check_answer(text)], ["mismatch"] * 2)

    def test_latex_unsupported_expressions_remain_unparsed(self):
        cases = (
            r"\[ x + 2 = 3 \]", r"\[ P(A) = evil(12) \]",
            r"\[ 1 = \sqrt{1} \]", r"\[ C_{12}^x = 495 \]",
            r"\[ \frac{1}{0} = 2 \]", r"\[ 1 = \frac{1}{2 \]",
            r"\[ 2 = 2.__class__ \]", r"\[ 1 = __import__('os').system('bad') \]",
        )
        for text in cases:
            with self.subTest(text=text):
                checks = check_answer(text)
                self.assertEqual(len(checks), 1)
                self.assertEqual(checks[0].status, "unparsed")

    def test_latex_claims_respect_global_count_limit(self):
        text = r"\[ 1=1 \]" * 250
        self.assertEqual(len(check_answer(text)), 200)

    def test_wrong_initial_fraction_cannot_launder_later_rounding(self):
        text = "1/2 = 99/667 = 0.148 = 14,8%"
        checks = check_answer(text)
        self.assertEqual([c.lhs_expr for c in checks], ["1/2"] * 3)
        self.assertEqual([c.status for c in checks], ["mismatch"] * 3)
        self.assertEqual(check_answer("99/667 = 0.148 = 14,8%")[1].status, "ok")

    def test_vietnamese_decimals_percentages_and_rounding(self):
        cases = {
            "1/3 = 0,33": "ok", "1/3 = 33,3%": "ok",
            "1/3 = 33%": "ok", "1/3 = 0,330": "mismatch",
            "1/3 = 0.34": "mismatch", "1/3 = 34%": "mismatch",
            "37/250 = 14,8%": "ok", "37/250 = 14.80%": "ok",
            "37/250 = 14,9%": "mismatch", "1,5 + 2,25 = 3,75": "ok",
            "1/3 = 333/1000": "mismatch", "3/2 = 2": "mismatch",
            "1/3 = 0.333333333333333333333333333333": "ok",
            "-1/3 = -0,33": "ok", "201/200 = 1.00": "ok",
            "10051/10000 = 1.00": "mismatch",
        }
        for text, status in cases.items():
            with self.subTest(text=text):
                checks = check_answer(text)
                self.assertEqual(len(checks), 1)
                self.assertEqual(checks[0].status, status)

    def test_parenthesized_percentage_and_symbolic_chain(self):
        checks = check_answer("99/667 = 0.148 (14,8%)")
        self.assertEqual([c.rhs_value for c in checks], ["0.148", "14,8%"])
        self.assertEqual([c.status for c in checks], ["ok", "ok"])
        checks = check_answer("p = C(12,4)/C(15,4) = 33/91")
        self.assertEqual([c.status for c in checks], ["unparsed", "ok"])
        self.assertEqual(checks[1].lhs_expr, "C(12,4)/C(15,4)")

    def test_evidence_support_is_per_operand_and_independent_of_status(self):
        expr = "C(12,4)/C(15,4) = 33/91"
        partial = check_answer(expr, ["Có 12 và 15 phần tử."])[0]
        self.assertEqual(partial.operand_support, {"12": True, "4": False, "15": True})
        self.assertFalse(partial.operand_supported)
        self.assertEqual(partial.status, "ok")
        complete = check_answer(expr, ["C(12,4)", "C(15,4)"])[0]
        self.assertTrue(complete.operand_supported)
        self.assertTrue(check_answer(expr, "12C4; 15C4")[0].operand_supported)
        self.assertFalse(check_answer(expr)[0].operand_supported)
        mismatch = check_answer(expr.replace("33/91", "99/667"), ["12, 15, 4"])[0]
        self.assertEqual(mismatch.status, "mismatch")
        self.assertTrue(mismatch.operand_supported)
        self.assertFalse(check_answer("6 + 7 = 13", ["667"])[0].operand_supported)
        self.assertFalse(check_answer("12 = 12", ["12students"])[0].operand_supported)
        self.assertTrue(check_answer("1,5 + 2.25 = 3.75", ["1.50 và 2,25"])[0].operand_supported)
        self.assertTrue(check_answer("50% = 1/2", ["50%"])[0].operand_supported)
        # Normalization's generated /100 is not a written operand.
        self.assertEqual(check_answer("50% = 1/2", ["50%"])[0].operand_support, {"50": True})

    def test_bad_input_and_malformed_claims_never_raise(self):
        for text in (None, 4, [], {}, "No equality: 14,8%", "1 != 2; 2 == 2; 3 >= 2"):
            with self.subTest(text=text):
                self.assertEqual(extract_claims(text), [])
                self.assertEqual(check_answer(text), [])
        for text in ("= 0.148", "x + 2 = 3", "C(12,x) = 495", "1/0 = 2",
                     "1 =", "2^101 = 4", "2 = NaN", "evil(12) = 12",
                     "1 = 1e3", "2 = 2.__class__", "1 = 1[0]", "1 = 1<<3",
                     "1 = 1 & 3", "1 = 1_000",
                     "__import__('os').system('bad') = 0"):
            with self.subTest(text=text):
                checks = check_answer(text, None)
                self.assertEqual(len(checks), 1)
                self.assertEqual(checks[0].status, "unparsed")
                self.assertIsNotNone(checks[0].error)
        self.assertEqual(check_answer("1/2=0.5", [None, 3, {}, "1 and 2"])[0].status, "ok")

    def test_multiple_independent_claims_and_nonliteral_rhs(self):
        checks = check_answer("2+2 = 4; then 3*3 = 8.\n1/2 = 2/4 = 0.50")
        self.assertEqual([c.status for c in checks], ["ok", "mismatch", "ok", "ok"])
        self.assertEqual(check_answer("1+1 = 2. This is the result.")[0].status, "ok")
        checks = check_answer("1 - C(12,4)/C(15,4) = 1 - 495/1365 = 58/91")
        self.assertEqual([c.status for c in checks], ["ok", "ok"])
        self.assertEqual(checks[0].claimed_value, Fraction(58, 91))


if __name__ == "__main__":
    unittest.main()
