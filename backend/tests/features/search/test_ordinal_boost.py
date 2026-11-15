import unittest

from app.features.search import retrieval as r


class OrdinalRefTests(unittest.TestCase):
    def test_extracts_numbered_items(self):
        self.assertEqual(r.ordinal_refs("Nhận định 14 nói về gì?"), [("nhan dinh", "14")])
        self.assertEqual(r.ordinal_refs("câu số 2 giải thế nào"), [("cau", "2")])
        self.assertEqual(r.ordinal_refs("bằng 2 lần"), [])

    def test_matches_only_the_same_number(self):
        refs = [("nhan dinh", "13")]
        self.assertTrue(r.mentions_ordinal("NHẬN ĐỊNH SỐ 13 Trong bài Làm thế nào", refs))
        self.assertTrue(r.mentions_ordinal("nhận định 13: thơ", refs))
        self.assertFalse(r.mentions_ordinal("NHẬN ĐỊNH 130", refs))
        self.assertFalse(r.mentions_ordinal("NHẬN ĐỊNH 1.3", [("nhan dinh", "1")]))
        self.assertFalse(r.mentions_ordinal("NHẬN ĐỊNH 16", refs))


if __name__ == "__main__":
    unittest.main()
