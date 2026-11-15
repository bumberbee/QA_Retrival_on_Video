from tests import _isolate  # noqa: F401
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.features.search import retrieval
from app.shared import supabase as supa


def chunk(cid, sid, text, start, end, kind="speech"):
    return {"chunk_id": cid, "segment_id": sid, "source_type": kind, "text": text,
            "start_sec": start, "end_sec": end, "title": ""}


def segment(sid, start, end, kind="slide"):
    return {"segment_id": sid, "start_sec": start, "end_sec": end,
            "content_kind": kind, "visual": {}, "speech": {}}


def index(chunks, segments, scores=None):
    vectors = np.array([[score] for score in (scores or [1.0] * len(chunks))], dtype=np.float32)
    return retrieval.LectureIndex(None, chunks, vectors, segments, {s["segment_id"]: s for s in segments})


class Phase2Tests(unittest.TestCase):
    def search(self, fixture, features, question="test", **kwargs):
        encoder = type("Encoder", (), {"encode": lambda self, text: np.array([1.0])})()
        with patch.object(retrieval, "load_index", return_value=fixture), patch.object(retrieval, "lexical_hits", return_value=[]):
            return retrieval.search("v", question, encoder=encoder, options=retrieval.retrieval_options(features), **kwargs)

    def test_accented_stopwords_preserve_technical_syllables(self):
        terms, pairs = retrieval.lexical_query("Tính lực từ và độ cứng của cơ học với chi phí cơ hội", phrases=True)
        for term in ("luc", "tu", "do", "cung", "co", "chi", "phi", "hoi"):
            self.assertIn(term, terms)
        for pair in (("luc", "tu"), ("do", "cung"), ("chi", "phi"), ("co", "hoi")):
            self.assertIn(pair, pairs)
        self.assertNotIn(("cung", "co"), pairs)
        self.assertNotIn("cua", terms)
        self.assertEqual(retrieval.lexical_query("của một như nào"), ([], []))
        self.assertIn("co", retrieval.lexical_query("co có cơ")[0])
        long_query = " và ".join(f"term{i}" for i in range(70)) + " late phrase"
        terms, pairs = retrieval.lexical_query(long_query, phrases=True)
        self.assertEqual(len(terms), 64)
        self.assertTrue(all(a in terms and b in terms for a, b in pairs))

    def test_sqlite_and_notebook_share_query_logic(self):
        from types import ModuleType
        root = Path(__file__).resolve().parents[4] / "pipeline"
        notebook = ModuleType("query_mirror")
        for name in ("shared/common.py", "stages/s06_index/retrieval.py"):
            exec(compile((root / name).read_text(encoding="utf-8"), str(root / name), "exec"), notebook.__dict__)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with closing(sqlite3.connect(folder / "lexical.sqlite3")) as db:
                db.execute("CREATE VIRTUAL TABLE evidence USING fts5(chunk_id UNINDEXED, text)")
                db.executemany("INSERT INTO evidence VALUES (?, ?)", [("a", "luc tu do cung"), ("b", "chi phi co hoi")])
                db.commit()
            for question in ("Lực từ có độ cứng?", "Chi phí cơ hội là gì?", "của một như nào", 'OR "x" -- lực'):
                for fixed in (True, False):
                    for phrases in (True, False):
                        self.assertEqual(retrieval.lexical_query(question, fixed, phrases), notebook.lexical_query(question, fixed, phrases))
                        options = retrieval.RetrievalOptions(vietnamese_lexical=fixed, phrases=phrases)
                        self.assertEqual(retrieval.lexical_hits(folder, question, 5, options=options),
                                         [cid for cid, score in notebook.lexical_search(folder / "lexical.sqlite3", question, 5, fixed, phrases)])

    def test_cloud_uses_safe_equivalent_phrase_operators(self):
        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, sql, params):
                self.sql, self.params = sql, params
                return self

            def fetchall(self):
                return [{"id": "a"}]

        conn = Connection()
        with patch.object(supa.db, "connection", return_value=conn):
            self.assertEqual(supa.lexical_hits("v", ["luc", "tu", "';DROP"], 5, [("luc", "tu"), ("tu", "evil")]), ["a"])
        self.assertEqual(conn.params, ("v", "luc | tu | (luc <-> tu)", "luc | tu | (luc <-> tu)", 5))
        self.assertIn("to_tsquery('simple'", conn.sql)
        with patch.object(retrieval, "lexical_query", return_value=(["luc", "tu"], [("luc", "tu")])), patch.object(supa, "lexical_hits", return_value=["a"]) as call:
            retrieval.lexical_hits(None, "question", 5, "v", retrieval.RetrievalOptions(phrases=True))
        call.assert_called_once_with("v", ["luc", "tu"], 5, phrases=[("luc", "tu")])

    def test_dedup_protects_numbers_and_equations(self):
        text = " ".join(f"word{i}" for i in range(40))
        chunks = [chunk("a", "v:a", text, 0, 1, "ocr"), chunk("b", "v:b", text + " extra", 1, 2, "ocr"),
                  chunk("c", "v:c", text.replace("word20", "word21"), 2, 3, "ocr"),
                  chunk("d", "v:d", "A+B=3", 3, 4, "ocr"), chunk("e", "v:e", "A-B=3", 4, 5, "ocr")]
        groups = retrieval.content_groups(index(chunks, []))
        self.assertEqual(groups["a"], groups["b"])
        self.assertNotEqual(groups["a"], groups["c"])
        self.assertNotEqual(groups["d"], groups["e"])

    def test_dedup_precedes_candidate_limit_and_keeps_occurrences(self):
        chunks = [chunk(f"c{i}", f"v:{i}", "identical OCR", i, i + 1, "ocr") for i in range(30)]
        chunks.append(chunk("answer", "v:answer", "useful speech", 31, 32))
        fixture = index(chunks, [segment(c["segment_id"], c["start_sec"], c["end_sec"]) for c in chunks], [1.0] * 30 + [0.95])
        baseline = self.search(fixture, [])
        found = self.search(fixture, ["dedup"])
        self.assertNotIn("v:answer", [s["segment_id"] for s in baseline["slides"]])
        self.assertIn("v:answer", [s["segment_id"] for s in found["slides"]])
        self.assertEqual(len(found["slides"][0]["evidence_candidates"][0]["occurrences"]), 30)
        preferred = self.search(fixture, ["dedup"], prefer_segments=frozenset({"v:29"}))
        self.assertEqual(preferred["slides"][0]["segment_id"], "v:29")

    def test_diversity_uses_content_groups(self):
        chunks = [chunk("a", "v:a", "duplicate OCR", 0, 1, "ocr"), chunk("b", "v:b", "duplicate OCR", 1, 2, "ocr"),
                  chunk("c", "v:c", "novel", 2, 3), chunk("d", "v:d", "different", 3, 4)]
        fixture = index(chunks, [segment(c["segment_id"], c["start_sec"], c["end_sec"]) for c in chunks])
        found = self.search(fixture, ["diversity"])
        self.assertEqual([s["segment_id"] for s in found["slides"]], ["v:a", "v:c", "v:d"])
        pinned = self.search(fixture, ["diversity"], question="this slide", current_time=1.5)
        self.assertEqual([s["segment_id"] for s in pinned["slides"]], ["v:b", "v:c", "v:d"])

    def test_matched_tail_replaces_slide_prefix_and_cites_span(self):
        seg = segment("v:s", 0, 1600)
        seg["speech"]["text"] = "padding " * 500 + "Chi phí cơ hội và rủi ro."
        fixture = index([chunk("late", "v:s", "Chi phí cơ hội và rủi ro.", 1500, 1520)], [seg])
        found = self.search(fixture, ["ranked_context"])
        context = retrieval.build_context(found)
        self.assertIn("Chi phí cơ hội", context)
        self.assertNotIn("padding", context)
        self.assertIn("25:00–25:20", context)
        self.assertEqual(found["slides"][0]["start_sec"], 0)
        self.assertEqual(found["slides"][0]["matched_spans"][0]["start_sec"], 1500)

    def test_speech_window_uses_adjacent_chunks(self):
        chunks = [chunk("a", "v:s", "previous sentence", 0, 10), chunk("b", "v:s", "matched", 10, 20),
                  chunk("c", "v:s", "next sentence", 20, 30)]
        fixture = index(chunks, [segment("v:s", 0, 30)])
        matched, candidates = retrieval.context_candidates(fixture, "v:s", ["b"], np.array([0.8, 1.0, 0.7]), retrieval.RetrievalOptions(neighbors=False))
        self.assertEqual([c["chunk_id"] for c in matched], ["b"])
        self.assertEqual({c["chunk_id"] for c in candidates}, {"a", "b", "c"})

    def test_neighbor_repairs_short_boundary_and_skips_ads_and_gaps(self):
        chunks = [chunk("a", "v:a", "OK, so let me go", 0, 10), chunk("b", "v:b", "back.", 10, 11)]
        segments = [segment("v:a", 0, 10), segment("v:b", 10, 11)]
        fixture = index(chunks, segments, [0.5, 1.0])
        options = retrieval.retrieval_options(["ranked_context", "neighbors"])
        matched, candidates = retrieval.context_candidates(fixture, "v:b", ["b"], np.array([0.5, 1.0]), options)
        self.assertEqual([(c["chunk_id"], c["selection_reason"]) for c in candidates], [("b", "matched"), ("a", "previous")])
        for change in ("ad", "gap"):
            segments[0]["content_kind"] = "ad" if change == "ad" else "slide"
            segments[0]["end_sec"] = 5 if change == "gap" else 10
            matched, candidates = retrieval.context_candidates(fixture, "v:b", ["b"], np.array([0.5, 1.0]), options)
            self.assertEqual(len(candidates), 1)

    def test_neighbor_explanation_trigger_is_conditional(self):
        chunks = [chunk("a", "v:a", "earlier explanation", 0, 10), chunk("b", "v:b", "long matched sentence " * 10, 20, 30)]
        fixture = index(chunks, [segment("v:a", 0, 10), segment("v:b", 10, 60)])
        options = retrieval.retrieval_options(["ranked_context", "neighbors"])
        for question, count in (("specific fact", 1), ("Giải thích các bước", 2)):
            matched, candidates = retrieval.context_candidates(fixture, "v:b", ["b"], np.array([0.5, 1.0]), options, question=question)
            self.assertEqual(len(candidates), count)

    def test_context_budget_deduplicates_across_slides_and_records_only_kept_text(self):
        chunks = [chunk(str(i), f"v:{i}", f"evidence {i} " + "text " * 150, i, i + 1) for i in range(5)]
        fixture = index(chunks, [segment(c["segment_id"], c["start_sec"], c["end_sec"]) for c in chunks])
        found = self.search(fixture, ["ranked_context", "neighbors"])
        context = retrieval.build_context(found, max_chars=1200)
        self.assertLessEqual(len(context), 1200)
        self.assertEqual(len(found["context_evidence"]), 1)
        self.assertIn(found["context_evidence"][0]["text"], context)
        self.assertNotIn("evidence 1", context)
        self.assertEqual(context, retrieval.build_context(found, max_chars=1200))

    def test_multi_video_context_preserves_video_identity_and_global_budget(self):
        fixtures = {v: index([chunk(v, v + ":s", "same text", 0, 10)], [segment(v + ":s", 0, 10)]) for v in ("a", "b")}
        encoder = type("Encoder", (), {"encode": lambda self, text: np.array([1.0])})()
        with patch.object(retrieval, "load_index", side_effect=lambda v: fixtures[v]), patch.object(retrieval, "lexical_hits", return_value=[]), patch.object(retrieval, "default_encoder", return_value=encoder), patch.object(retrieval, "has_index", return_value=True), patch.object(retrieval, "video_title", side_effect=lambda v: "Title " + v):
            found = retrieval.search_many(["a", "b", "a"], "question", options=retrieval.retrieval_options(["ranked_context"]))
        context = retrieval.build_context(found)
        self.assertEqual(found["videos"], 2)
        self.assertIn("Video Title a", context)
        self.assertIn("Video Title b", context)
        self.assertEqual(len(found["context_evidence"]), 2)
        self.assertLessEqual(len(context), retrieval.MAX_CONTEXT_CHARS)


if __name__ == "__main__":
    unittest.main()
