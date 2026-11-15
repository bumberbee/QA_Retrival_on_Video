"""Cloud mode end to end: sync a small local library to Postgres + pgvector and an S3 bucket, then
read it back through storage, retrieval, sessions and the file routes.

Needs a Postgres with pgvector, e.g.
    docker run -d --name lecture-rag-pg -e POSTGRES_PASSWORD=localtest -p 55432:5432 pgvector/pgvector:pg16
and TEST_DB_URL (default postgresql://postgres:localtest@localhost:55432/postgres); skipped otherwise.
R2 is replaced by moto's in-memory S3.
"""

from tests import _isolate  # noqa: F401  (must come before any app import)
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

TEST_DB_URL = os.getenv("TEST_DB_URL", "postgresql://postgres:localtest@localhost:55432/postgres")
SCHEMA = Path(__file__).resolve().parents[3] / "infra" / "supabase" / "schema.sql"


def database_ready() -> bool:
    try:
        import psycopg
        with psycopg.connect(TEST_DB_URL, connect_timeout=3) as conn:
            conn.execute("select 1")
        return True
    except Exception:
        return False


class FakeEncoder:
    def __init__(self, vector):
        self.vector = vector

    def encode(self, text):
        return self.vector


def unit(index, size=384):
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


def write_library(root: Path) -> None:
    """One video, two slides, three chunks, in the layout import_fast_pipeline.py writes."""
    (root / "videos").mkdir(parents=True)
    (root / "videos" / "abc123def456.mp4").write_bytes(b"fake mp4")
    (root / "videos" / "abc123def456.meta.json").write_text(json.dumps(
        {"original_filename": "L25_V011.mp4", "uploaded_at": "2026-10-20T00:00:00+00:00", "duration_seconds": 894.6}))
    (root / "transcripts").mkdir()
    (root / "transcripts" / "abc123def456.json").write_text(json.dumps(
        {"video_id": "abc123def456", "language": "vi",
         "segments": [{"start": 0, "end": 5, "text": "Chào các em"}, {"start": 5, "end": 9, "text": "Liên hợp quốc"}]}))
    keyframes = root / "keyframes" / "abc123def456"
    (keyframes / "fast").mkdir(parents=True)
    (keyframes / "seg_00000.jpg").write_bytes(b"jpg0")
    (keyframes / "fast" / "s_0001.jpg").write_bytes(b"jpg1")
    index = root / "index" / "abc123def456"
    (index / "lecture").mkdir(parents=True)
    segments = [
        {"segment_id": "L25_V011:seg_00000", "sequence": 0, "start_sec": 0, "end_sec": 100, "content_kind": "slide",
         "visual": {"title_clean": "Liên hợp quốc", "ocr_text_clean": "Hội nghị Yalta"},
         "speech": {"text": "Liên hợp quốc ra đời năm 1945", "word_ids": [1, 2]}},
        {"segment_id": "L25_V011:seg_00001", "sequence": 1, "start_sec": 100, "end_sec": 200, "content_kind": "slide",
         "visual": {"title_clean": "ASEAN", "ocr_text_clean": "Hiệp hội các quốc gia Đông Nam Á"},
         "speech": {"text": "ASEAN thành lập năm 1967", "word_ids": [3]}},
    ]
    (index / "lecture" / "segments.json").write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    chunks = [
        {"chunk_id": "c0", "segment_id": "L25_V011:seg_00000", "source_type": "speech", "start_sec": 0, "end_sec": 50,
         "title": "Liên hợp quốc", "text": "Liên hợp quốc ra đời năm 1945", "embedding_text": "x"},
        {"chunk_id": "c1", "segment_id": "L25_V011:seg_00001", "source_type": "ocr", "start_sec": 100, "end_sec": 200,
         "title": "ASEAN", "text": "Hiệp hội các quốc gia Đông Nam Á", "embedding_text": "x"},
        {"chunk_id": "c2", "segment_id": "L25_V011:seg_00001", "source_type": "speech", "start_sec": 100,
         "end_sec": 150, "title": "ASEAN", "text": "ASEAN thành lập năm 1967", "embedding_text": "x"},
    ]
    (index / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    np.save(index / "embeddings.npy", np.stack([unit(0), unit(1), unit(2)]))
    (index / "lexical.sqlite3").write_bytes(b"")


@unittest.skipUnless(database_ready(), "no Postgres with pgvector at TEST_DB_URL")
class CloudModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg
        with psycopg.connect(TEST_DB_URL, autocommit=True) as conn:
            conn.execute(SCHEMA.read_text(encoding="utf-8"))
            conn.execute("truncate videos, conversations, app_config cascade")

    def setUp(self):
        from moto import mock_aws

        self.mock = mock_aws()
        self.mock.start()
        self.addCleanup(self.mock.stop)
        env = {"SUPABASE_DB_URL": TEST_DB_URL, "R2_ACCOUNT_ID": "acct", "R2_ACCESS_KEY_ID": "key",
               "R2_SECRET_ACCESS_KEY": "secret", "R2_BUCKET": "lecture-rag", "R2_ENDPOINT": "https://s3.amazonaws.com",
               "R2_REGION": "us-east-1"}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        from app.shared import objects
        from app.features.search import retrieval
        objects.client.cache_clear()
        retrieval.load_index.cache_clear()
        self.addCleanup(retrieval.load_index.cache_clear)
        objects.client().create_bucket(Bucket="lecture-rag")
        self.data = tempfile.TemporaryDirectory()
        self.addCleanup(self.data.cleanup)
        write_library(Path(self.data.name))
        spec = importlib.util.spec_from_file_location(
            "sync_supabase", Path(__file__).resolve().parents[2] / "scripts" / "sync_supabase.py")
        self.sync = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.sync)

    def test_synced_library_reads_back_and_searches_like_the_files(self):
        from app.shared import storage
        from app.shared import objects
        from app.features.search import retrieval

        report = self.sync.sync_video(Path(self.data.name), "abc123def456", skip_files=False)
        self.assertIn("2 slide, 3 chunk", report)
        self.assertTrue(objects.exists("videos/abc123def456/L25_V011.mp4"))
        self.assertTrue(objects.exists("keyframes/abc123def456/fast/s_0001.jpg"))
        # re-running replaces the rows instead of failing on duplicates
        self.sync.sync_video(Path(self.data.name), "abc123def456", skip_files=False)

        videos = storage.list_videos()
        self.assertEqual([(v.video_id, v.filename, v.has_transcript, v.has_slides) for v in videos],
                         [("abc123def456", "L25_V011.mp4", True, True)])
        self.assertEqual(storage.find_video_id_by_original_filename("L25_V011.mp4"), "abc123def456")
        self.assertEqual([s.text for s in storage.load_transcript("abc123def456").segments],
                         ["Chào các em", "Liên hợp quốc"])
        self.assertEqual(storage.list_keyframes("abc123def456"), ["seg_00000.jpg"])

        self.assertTrue(retrieval.has_index("abc123def456"))
        found = retrieval.search("abc123def456", "ASEAN thành lập khi nào", encoder=FakeEncoder(unit(2)))
        self.assertEqual(found["slides"][0]["segment_id"], "L25_V011:seg_00001")
        self.assertEqual(retrieval.lexical_hits(None, "asean thanh lap", 5, "abc123def456")[0], "c2")
        self.assertIn("Hiệp hội", retrieval.build_context(found))

    def test_conversations_and_signed_file_urls(self):
        from fastapi.testclient import TestClient

        from app.main import app
        from app.features.conversations import sessions

        self.sync.sync_video(Path(self.data.name), "abc123def456", skip_files=False)
        conversation = sessions.create("browser-cloud-0001", "abc123def456")
        sessions.append_turn(conversation["id"], "browser-cloud-0001", {"question": "LHQ?", "answer": "1945"})
        self.assertEqual([c["title"] for c in sessions.list_for("browser-cloud-0001")], ["LHQ?"])
        self.assertEqual(sessions.get(conversation["id"], "someone-else-01"), None)

        api = TestClient(app)
        video = api.get("/api/videos/abc123def456/file", follow_redirects=False)
        self.assertEqual(video.status_code, 307)
        self.assertIn("videos/abc123def456/L25_V011.mp4", video.headers["location"])
        slide = api.get("/api/videos/abc123def456/slides/fast/s_0001.jpg", follow_redirects=False)
        self.assertEqual(slide.status_code, 307)
        self.assertEqual(api.get("/api/videos/abc123def456/slides/nope.jpg", follow_redirects=False).status_code, 404)
        self.assertTrue(sessions.delete(conversation["id"], "browser-cloud-0001"))


if __name__ == "__main__":
    unittest.main()
