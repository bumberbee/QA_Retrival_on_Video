import _isolate  # noqa: F401  (must come before any app import)
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import storage
from app.services import retrieval


class FakeEncoder:
    """Maps a question to the vector of the chunk it is meant to find (no model download)."""

    def __init__(self, table):
        self.table = table

    def encode(self, text):
        return self.table.get(text, np.array([0.0, 0.0, 1.0]))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        data = Path(self.directory.name)
        patcher = patch.object(storage, "DATA_DIR", data)
        patcher.start()
        self.addCleanup(patcher.stop)
        retrieval.load_index.cache_clear()
        folder = data / "index" / "vid"
        (folder / "lecture").mkdir(parents=True)
        segments = [
            {"segment_id": "vid:seg_00000", "start_sec": 0, "end_sec": 60, "content_kind": "slide",
             "visual": {"title_clean": "Định nghĩa xác suất", "ocr_text_clean": "P(A) = n(A)/n(Ω)", "image_path": "a.jpg"},
             "speech": {"text": "xác suất của biến cố A"}},
            {"segment_id": "vid:seg_00001", "start_sec": 60, "end_sec": 120, "content_kind": "ad",
             "visual": {}, "speech": {"text": "tuyển sinh"}},
            {"segment_id": "vid:seg_00002", "start_sec": 120, "end_sec": 200, "content_kind": "slide",
             "visual": {"title_clean": "Câu 3", "ocr_text_clean": "40 tấm thẻ", "image_path": "b.jpg"},
             "speech": {"text": "chia hết cho 3"}},
        ]
        (folder / "lecture" / "segments.json").write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
        chunks = [
            {"chunk_id": "c0", "segment_id": "vid:seg_00000", "source_type": "ocr", "text": "Định nghĩa xác suất P(A)",
             "start_sec": 0, "end_sec": 60, "title": "Định nghĩa xác suất"},
            {"chunk_id": "c1", "segment_id": "vid:seg_00002", "source_type": "speech", "text": "các số chia hết cho 3",
             "start_sec": 120, "end_sec": 200, "title": "Câu 3"},
        ]
        (folder / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
        np.save(folder / "embeddings.npy", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
        with closing(sqlite3.connect(folder / "lexical.sqlite3")) as db:
            db.execute("CREATE VIRTUAL TABLE evidence USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61 remove_diacritics 2')")
            db.executemany("INSERT INTO evidence(chunk_id, text) VALUES (?, ?)",
                           [(c["chunk_id"], retrieval.normalize_search(c["title"] + "\n" + c["text"])) for c in chunks])
            db.commit()
        self.encoder = FakeEncoder({"Tính xác suất thế nào?": np.array([1.0, 0.0, 0.0]),
                                    "Có bao nhiêu số chia hết cho 3?": np.array([0.0, 1.0, 0.0])})

    def test_ranked_slides_and_context(self):
        self.assertTrue(retrieval.has_index("vid"))
        self.assertFalse(retrieval.has_index("other"))
        found = retrieval.search("vid", "Có bao nhiêu số chia hết cho 3?", encoder=self.encoder)
        self.assertEqual(found["slides"][0]["segment_id"], "vid:seg_00002")
        self.assertEqual(found["slides"][0]["title"], "Câu 3")
        context = retrieval.build_context(found)
        self.assertIn("40 tấm thẻ", context)
        self.assertIn("chia hết cho 3", context)

    def test_this_slide_during_an_ad_uses_the_slide_before(self):
        found = retrieval.search("vid", "Giải thích slide này", current_time=90, encoder=self.encoder)
        self.assertTrue(found["temporal"])
        self.assertEqual(found["slides"][0]["segment_id"], "vid:seg_00000")

    def test_stopwords_do_not_drive_keyword_search(self):
        folder = retrieval.index_dir("vid")
        self.assertEqual(retrieval.lexical_hits(folder, "của một như thế nào", 5), [])
        self.assertEqual(retrieval.lexical_hits(folder, "định nghĩa xác suất", 5)[0], "c0")


class FollowUpTest(unittest.TestCase):
    def test_short_questions_that_point_back_are_follow_ups(self):
        for question in ("Còn ASEAN thì sao?", "Nó được thành lập năm nào?", "Cái đó nghĩa là gì?",
                         "Thế còn EU?", "What about the eigenvalues?"):
            self.assertTrue(retrieval.is_follow_up(question), question)

    def test_standalone_questions_are_not(self):
        # "độ", "con", "thế nào" look like "đó", "còn", "thế" once the diacritics are gone
        for question in ("Nhiệt độ nóng chảy của nhôm là bao nhiêu?", "Liên hợp quốc thành lập năm nào?",
                         "Con người tiến hóa thế nào trong bài?", "Làm thế nào tính xác suất?"):
            self.assertFalse(retrieval.is_follow_up(question), question)

    def test_a_follow_up_is_searched_with_the_question_before_it(self):
        self.assertEqual(retrieval.search_question("Còn EU thì sao?", "ASEAN thành lập năm nào?"),
                         "ASEAN thành lập năm nào? Còn EU thì sao?")
        self.assertEqual(retrieval.search_question("EU thành lập năm nào?", "ASEAN thành lập năm nào?"),
                         "EU thành lập năm nào?")
        self.assertEqual(retrieval.search_question("Còn EU thì sao?", None), "Còn EU thì sao?")


if __name__ == "__main__":
    unittest.main()
