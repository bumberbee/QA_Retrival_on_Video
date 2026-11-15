import _isolate  # noqa: F401  (must come before any app import)
import asyncio
import importlib
import io
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import quote

import httpx
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from app import main, storage
from app.models import Transcript, VideoImportStatus
from app.routers import qa, videos
from app.services import video_import
from test_storage_security import BAD_IDS, StorageFixture, VIDEO_ID


class AdminAndPathTests(StorageFixture):
    def setUp(self):
        super().setUp()
        self.api = TestClient(main.app)
        (storage.SLIDE_VARIANTS_DIR / f"{VIDEO_ID}.json").write_text(json.dumps({"variants": {"F": {"slides": []}}}))

    def writes(self):
        return (
            ("POST", "/api/videos", {"files": {"file": ("lecture.mp4", b"video", "video/mp4")}}),
            ("PUT", f"/api/videos/{VIDEO_ID}/transcript", {"json": {"video_id": VIDEO_ID, "segments": []}}),
            ("PUT", f"/api/videos/{VIDEO_ID}/slide-reviews/F/0.000", {"json": {"verdict": "good", "note": ""}}),
            ("POST", "/api/videos/import", {"json": {"url": "https://youtu.be/public123"}}),
        )

    def test_all_content_writes_require_the_configured_admin_token(self):
        with patch.dict(os.environ, {"ADMIN_TOKEN": "test-admin-secret"}), \
                patch.object(video_import, "start_import") as start:
            for method, path, options in self.writes():
                for headers in ({}, {"X-Admin-Token": "wrong"}):
                    with self.subTest(path=path, headers=headers):
                        self.assertEqual(self.api.request(method, path, headers=headers, **options).status_code, 403)
            start.assert_not_called()
        self.assertEqual(list(storage.VIDEOS_DIR.iterdir()), [])
        self.assertEqual(list(storage.TRANSCRIPTS_DIR.iterdir()), [])
        self.assertEqual(list(storage.SLIDE_REVIEWS_DIR.iterdir()), [])

    def test_local_writes_accept_the_token_and_allow_an_unset_token(self):
        job = VideoImportStatus(import_id="test-job", status="queued", progress=0)
        for token in ("", "test-admin-secret"):
            with patch.dict(os.environ, {"ADMIN_TOKEN": token}), patch.object(storage, "probe_duration_seconds", return_value=10), \
                    patch.object(video_import, "start_import", return_value=job):
                for method, path, options in self.writes():
                    with self.subTest(token=bool(token), path=path):
                        headers = {"X-Admin-Token": token} if token else {}
                        response = self.api.request(method, path, headers=headers, **options)
                        self.assertEqual(response.status_code, 202 if path.endswith("/import") else 200, response.text)

    def test_cloud_writes_without_a_configured_token_are_disabled(self):
        with patch.dict(os.environ, {"SUPABASE_DB_URL": "mock-cloud"}), patch.object(storage, "_supa") as supa:
            for method, path, options in self.writes():
                for headers in ({}, {"X-Admin-Token": "arbitrary"}):
                    with self.subTest(path=path, headers=headers):
                        self.assertEqual(self.api.request(method, path, headers=headers, **options).status_code, 403)
            supa.assert_not_called()

    def test_cloud_upload_import_and_transcript_are_rejected_even_with_the_token(self):
        with patch.dict(os.environ, {"SUPABASE_DB_URL": "mock-cloud", "ADMIN_TOKEN": "test-admin-secret"}), \
                patch.object(storage, "_supa") as supa, patch.object(video_import, "start_import") as start:
            for method, path, options in self.writes():
                if "slide-reviews" in path:
                    continue
                with self.subTest(path=path):
                    response = self.api.request(method, path, headers={"X-Admin-Token": "test-admin-secret"}, **options)
                    self.assertEqual(response.status_code, 403)
                    self.assertIn("unsupported in cloud mode", response.json()["detail"])
            supa.assert_not_called()
            start.assert_not_called()
        self.assertEqual(list(storage.VIDEOS_DIR.iterdir()), [])
        self.assertEqual(list(storage.TRANSCRIPTS_DIR.iterdir()), [])

    def test_cloud_slide_review_uses_mock_database_only_after_admin_auth(self):
        supa = Mock()
        supa.slide_variants.return_value = {"variants": {"F": {}}}
        supa.slide_reviews.return_value = {}
        with patch.dict(os.environ, {"SUPABASE_DB_URL": "mock-cloud", "ADMIN_TOKEN": "test-admin-secret"}), \
                patch.object(storage, "_supa", return_value=supa):
            response = self.api.put(f"/api/videos/{VIDEO_ID}/slide-reviews/F/0.000",
                                    headers={"X-Admin-Token": "test-admin-secret"}, json={"verdict": "good"})
        self.assertEqual(response.status_code, 200)
        supa.save_slide_reviews.assert_called_once_with(VIDEO_ID, {"F": {"0.000": {"verdict": "good", "note": ""}}})
        self.assertEqual(list(storage.SLIDE_REVIEWS_DIR.iterdir()), [])

    def test_read_routes_and_transcript_qa_remain_public(self):
        storage.video_path(VIDEO_ID, ".mp4").write_bytes(b"video")
        storage.save_transcript(Transcript(video_id=VIDEO_ID, segments=[]))
        folder = storage.keyframes_dir(VIDEO_ID) / "fast"
        folder.mkdir(parents=True)
        (folder / "state_00012.jpg").write_bytes(b"jpg")
        with patch.dict(os.environ, {"ADMIN_TOKEN": "test-admin-secret"}), \
                patch.object(qa.retrieval, "has_index", return_value=False), \
                patch.object(qa.llm, "answer_question", return_value="Public answer"):
            for path in ("/api/videos", "/api/health", f"/api/videos/{VIDEO_ID}/file",
                         f"/api/videos/{VIDEO_ID}/transcript", f"/api/videos/{VIDEO_ID}/slides",
                         f"/api/videos/{VIDEO_ID}/slides/fast/state_00012.jpg", f"/api/videos/{VIDEO_ID}/outline",
                         f"/api/videos/{VIDEO_ID}/slide-variants", f"/api/videos/{VIDEO_ID}/slide-reviews"):
                with self.subTest(path=path):
                    self.assertEqual(self.api.get(path).status_code, 200)
            response = self.api.post("/api/qa", json={"video_id": VIDEO_ID, "question": "Question?"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["answer"], "Public answer")

    def test_encoded_traversal_ids_cannot_reach_files_or_database(self):
        with patch.dict(os.environ, {"SUPABASE_DB_URL": "mock-cloud"}), patch.object(storage, "_supa") as supa:
            for value in BAD_IDS:
                if not value:
                    continue
                for suffix in ("slides/.env", "transcript", "file", "outline", "slide-reviews", "slide-variants"):
                    with self.subTest(value=value, suffix=suffix):
                        response = self.api.get(f"/api/videos/{quote(value, safe='')}/{suffix}")
                        self.assertIn(response.status_code, (400, 404, 422))
            supa.assert_not_called()
        self.assertEqual(self.api.get("/api/videos/%2e%2e%5c%2e%2e%5c%2e%2e/slides/.env").status_code, 400)

    def test_qa_validates_primary_and_cross_video_ids_before_retrieval(self):
        with patch.object(qa.retrieval, "has_index") as has_index:
            for path in ("/api/qa", "/api/qa/stream"):
                for payload in ({"video_id": r"..\..", "question": "q"},
                                {"video_id": VIDEO_ID, "video_ids": ["%2e%2e"], "question": "q"}):
                    with self.subTest(path=path, payload=payload):
                        self.assertEqual(self.api.post(path, json=payload).status_code, 400)
            has_index.assert_not_called()

    def test_direct_cloud_import_is_rejected_before_scheduling(self):
        with patch.dict(os.environ, {"SUPABASE_DB_URL": "mock-cloud"}), \
                patch.object(video_import._executor, "submit") as submit:
            with self.assertRaises(storage.CloudWriteUnsupported):
                video_import.start_import("https://youtu.be/public123")
            submit.assert_not_called()

    def test_concurrent_slide_reviews_preserve_both_edits(self):
        first_loaded, release, second_loaded = threading.Event(), threading.Event(), threading.Event()
        original = storage.load_slide_reviews
        calls = []

        def load(video_id):
            reviews = original(video_id)
            calls.append(video_id)
            if len(calls) == 1:
                first_loaded.set()
                release.wait(2)
            else:
                second_loaded.set()
            return reviews

        def save(key):
            return self.api.put(f"/api/videos/{VIDEO_ID}/slide-reviews/F/{key}", json={"verdict": "good"})

        with patch.object(storage, "load_slide_reviews", side_effect=load), ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(save, "0.000")
            try:
                self.assertTrue(first_loaded.wait(1))
                second = executor.submit(save, "1.000")
                self.assertFalse(second_loaded.wait(0.1), "second edit read stale reviews")
            finally:
                release.set()
            self.assertEqual(first.result().status_code, 200)
            self.assertEqual(second.result().status_code, 200)
        self.assertEqual(set(original(VIDEO_ID)["F"]), {"0.000", "1.000"})


class UploadTests(StorageFixture):
    def test_default_upload_limit_is_two_gib(self):
        with patch.dict(os.environ, {"MAX_UPLOAD_MB": "2048"}):
            self.assertEqual(videos.max_upload_bytes(), 2 * 1024 ** 3)

    def test_upload_at_exact_limit_is_published_atomically(self):
        payload = b"x" * (1024 * 1024)
        file = UploadFile(io.BytesIO(payload), filename="lecture.mp4")

        def probe(path):
            self.assertEqual(storage.list_videos(), [])
            self.assertEqual(path.read_bytes(), payload)
            return 12.5

        with patch.object(storage, "probe_duration_seconds", side_effect=probe):
            video = videos.upload_video(file)
        self.assertEqual(storage.find_video_file(video.video_id).read_bytes(), payload)
        self.assertEqual(video.duration_seconds, 12.5)
        self.assertEqual(sorted(p.name for p in storage.VIDEOS_DIR.iterdir()),
                         [f"{video.video_id}.meta.json", f"{video.video_id}.mp4"])
        self.assertTrue(file.file.closed)

    def test_size_limit_counts_reads_even_when_upload_size_is_unknown(self):
        file = UploadFile(io.BytesIO(b"x" * (1024 * 1024 + 1)), filename="lecture.mp4")
        with self.assertRaises(HTTPException) as raised:
            videos.upload_video(file)
        self.assertEqual(raised.exception.status_code, 413)
        self.assertTrue(file.file.closed)
        self.assertEqual(list(storage.VIDEOS_DIR.iterdir()), [])

    def test_http_size_limit_rejects_without_leaving_files(self):
        response = TestClient(main.app).post("/api/videos", files={"file": ("lecture.mp4", b"x" * (1024 * 1024 + 1))})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(list(storage.VIDEOS_DIR.iterdir()), [])

    def test_declared_oversize_is_rejected_before_multipart_spooling(self):
        with patch("starlette.formparsers.SpooledTemporaryFile") as spool:
            response = TestClient(main.app).post("/api/videos", content=b"unused",
                                                 headers={"content-length": str(2 * 1024 * 1024),
                                                          "content-type": "multipart/form-data; boundary=demo"})
        self.assertEqual(response.status_code, 413)
        spool.assert_not_called()

    def test_unauthorized_upload_is_rejected_before_multipart_spooling(self):
        with patch.dict(os.environ, {"ADMIN_TOKEN": "test-admin-secret"}), \
                patch("starlette.formparsers.SpooledTemporaryFile") as spool:
            response = TestClient(main.app).post("/api/videos", files={"file": ("lecture.mp4", b"video")})
        self.assertEqual(response.status_code, 403)
        spool.assert_not_called()

    def test_read_failure_cleans_partial_upload(self):
        class BrokenFile(io.BytesIO):
            def read(self, size):
                if self.tell():
                    raise OSError("client disconnected")
                return super().read(4)

        file = UploadFile(BrokenFile(b"unfinished"), filename="lecture.mp4")
        with self.assertRaises(OSError):
            videos.upload_video(file)
        self.assertTrue(file.file.closed)
        self.assertEqual(list(storage.VIDEOS_DIR.iterdir()), [])

    def test_persistence_failure_cleans_video_metadata_and_temporary_files(self):
        for helper in ("save_video_meta", "load_video_meta"):
            file = UploadFile(io.BytesIO(b"video"), filename="lecture.mp4")
            with self.subTest(helper=helper), patch.object(storage, "probe_duration_seconds", return_value=1), \
                    patch.object(storage, helper, side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    videos.upload_video(file)
            self.assertEqual(list(storage.VIDEOS_DIR.iterdir()), [])
            self.assertTrue(file.file.closed)


class AsyncUploadTests(StorageFixture, unittest.IsolatedAsyncioTestCase):
    async def test_chunked_request_limit_closes_partial_multipart_spool(self):
        spools = []

        def spool(*args, **kwargs):
            file = tempfile.SpooledTemporaryFile(*args, **kwargs)
            spools.append(file)
            return file

        async def body():
            yield b'--demo\r\nContent-Disposition: form-data; name="file"; filename="lecture.mp4"\r\n\r\n'
            for _ in range(18):
                yield b"x" * (64 * 1024)
            yield b"\r\n--demo--\r\n"

        with patch("starlette.formparsers.SpooledTemporaryFile", side_effect=spool):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as api:
                response = await api.post("/api/videos", content=body(),
                                          headers={"content-type": "multipart/form-data; boundary=demo"})
        self.assertEqual(response.status_code, 413)
        self.assertTrue(spools)
        self.assertTrue(all(file.closed for file in spools))
        self.assertEqual(list(storage.VIDEOS_DIR.iterdir()), [])

    async def test_slow_storage_read_does_not_block_health_requests(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()

        def slow_list():
            started.set()
            release.wait(2)
            finished.set()
            return []

        with patch.object(storage, "list_videos", side_effect=slow_list):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as api:
                listing = asyncio.create_task(api.get("/api/videos"))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 1))
                    response = await api.get("/api/health")
                    self.assertEqual(response.status_code, 200)
                    self.assertFalse(finished.is_set(), "storage work blocked the event loop")
                finally:
                    release.set()
                    self.assertEqual((await listing).status_code, 200)


class CORSSecurityTests(unittest.TestCase):
    def test_configured_origins_are_honored_and_wildcard_is_ignored_with_credentials(self):
        try:
            with patch.dict(os.environ, {"CORS_ORIGINS": " https://demo.example/ , * "}):
                api = TestClient(importlib.reload(main).app)
                for origin in ("https://demo.example", "http://localhost:5173", "http://127.0.0.1:8000", "http://[::1]:5173"):
                    response = api.get("/api/health", headers={"Origin": origin})
                    self.assertEqual(response.headers.get("access-control-allow-origin"), origin)
                    self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")
                response = api.get("/api/health", headers={"Origin": "https://attacker.example"})
                self.assertNotIn("access-control-allow-origin", response.headers)
                preflight = api.options("/api/videos", headers={"Origin": "https://demo.example",
                                        "Access-Control-Request-Method": "POST",
                                        "Access-Control-Request-Headers": "X-Admin-Token"})
                self.assertEqual(preflight.status_code, 200)
                self.assertEqual(preflight.headers["access-control-allow-origin"], "https://demo.example")
        finally:
            importlib.reload(main)
