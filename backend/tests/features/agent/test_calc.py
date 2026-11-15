from tests import _isolate  # noqa: F401 (must come before app imports)
import unittest
from fractions import Fraction
from unittest.mock import patch

from app.features.agent.tools.calc import calc


class CalcTests(unittest.TestCase):
    def test_exact_values_and_stable_result_schema(self):
        cases = {
            "C(3,1)*C(12,4)*C(15,5)/C(30,10)": Fraction(99, 667),
            "C(3,1)*C(15,4)*C(15,5)/C(30,10)": Fraction(273, 667),
            "C(12,4)/C(15,4)": Fraction(33, 91),
            "0.1+0.2": Fraction(3, 10),
            "14,8%": Fraction(37, 250),
            "P(12,4)": Fraction(11880),
            "4!": Fraction(24),
            "2**-3": Fraction(1, 8),
            "-3/2": Fraction(-3, 2),
            "0": Fraction(0),
        }
        for expr, expected in cases.items():
            with self.subTest(expr=expr):
                self.assertEqual(calc(expr), {
                    "ok": True,
                    "value_fraction": f"{expected.numerator}/{expected.denominator}",
                    "value_float": float(expected),
                    "error": None,
                })

    def test_invalid_unsafe_and_out_of_bounds_inputs_never_raise(self):
        expressions = (
            "", " ", None, 123, [], {}, "1/0", "C(1001,1)", "2^101",
            "9" * 101, "(" * 300 + "1" + ")" * 300,
            "__import__('os').system('echo unsafe')", "open('unsafe')",
            "eval('2+2')", "exec('2+2')", "(1).__class__", "C.__globals__",
            "[x for x in (1,2)]", "lambda: 1", "C(n=12,k=4)", "abs(-1)",
        )
        with patch("os.system") as system, patch("builtins.open") as open_file:
            for expr in expressions:
                with self.subTest(expr=repr(expr)[:100]):
                    result = calc(expr)
                    self.assertEqual(set(result), {"ok", "value_fraction", "value_float", "error"})
                    self.assertIs(result["ok"], False)
                    self.assertIsNone(result["value_fraction"])
                    self.assertIsNone(result["value_float"])
                    self.assertIsInstance(result["error"], str)
                    self.assertTrue(result["error"])
            system.assert_not_called()
            open_file.assert_not_called()

    def test_unexpected_evaluator_failure_is_returned_as_error(self):
        with patch("app.features.agent.tools.calc.safe_eval", side_effect=RuntimeError("failure")):
            self.assertEqual(calc("1+1"), {
                "ok": False, "value_fraction": None, "value_float": None, "error": "failure",
            })

    def test_unrepresentable_float_is_returned_as_error(self):
        # Exact arithmetic permits 4096-bit fractions; a float may overflow.
        result = calc("*".join(["2^100"] * 11))
        self.assertIs(result["ok"], False)
        self.assertIsNone(result["value_fraction"])
        self.assertIsNone(result["value_float"])
        self.assertIsInstance(result["error"], str)


if __name__ == "__main__":
    unittest.main()
