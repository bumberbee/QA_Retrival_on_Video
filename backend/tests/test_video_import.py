import json
import io
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app import storage
from app.models import VideoImportRequest
from app.routers.videos import get_video_import, import_video
from app.services import video_import


class VideoImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data = Path(self.directory.name)
        for name in ("videos", "transcripts", "keyframes"):
            (self.data / name).mkdir()
        for name, value in {
            "DATA_DIR": self.data,
            "VIDEOS_DIR": self.data / "videos",
            "TRANSCRIPTS_DIR": self.data / "transcripts",
            "KEYFRAMES_DIR": self.data / "keyframes",
        }.items():
            self.addCleanup(patch.stopall)
            patch.object(storage, name, value).start()
        self.submit = patch.object(video_import._executor, "submit").start()
        with video_import._lock:
            video_import._jobs.clear()
        self.url = "https://www.youtube.com/watch?v=public123"

    def create_job(self):
        return video_import.start_import(self.url)

    def test_recognizes_supported_share_and_mobile_links(self):
        urls = {
            "https://youtu.be/public123": "youtube",
            "https://m.youtube.com/shorts/public123": "youtube",
            "https://www.facebook.com/reel/12345": "facebook",
            "https://fb.watch/abc123/": "facebook",
            "https://m.facebook.com/watch/?v=12345": "facebook",
            "https://www.tiktok.com/@teacher/video/12345": "tiktok",
            "https://vm.tiktok.com/abc123/": "tiktok",
            "https://vt.tiktok.com/abc123/": "tiktok",
        }
        for url, platform in urls.items():
            with self.subTest(url=url):
                self.assertEqual(video_import.validate_source_url(f" {url}#fragment "), (url, platform))

    def test_rejects_unsafe_unsupported_and_playlist_urls_before_submission(self):
        urls = (
            "file:///etc/passwd", "http://localhost/video", "http://127.0.0.1/video",
            "http://[::1]/video", "https://youtube.com.evil.example/watch?v=123",
            "https://youtube.com@127.0.0.1/video", "https://user:secret@youtube.com/watch?v=123",
            "https://youtube.com:8080/watch?v=123", "https://youtube.com/watch?v=123\nnext",
            "https://youtube.com\\@example.com/video", "https://youtube.com/playlist?list=123",
            "https://youtube.com/watch?v=123&list=456", "https://example.com/video.mp4", "",
        )
        for url in urls:
            with self.subTest(url=url), self.assertRaises(video_import.ImportValidationError):
                video_import.start_import(url)
        self.submit.assert_not_called()

    def test_public_url_guard_rejects_private_dns_results(self):
        for address in ("127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254", "::1"):
            records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]
            with self.subTest(address=address), patch.object(socket, "getaddrinfo", return_value=records):
                with self.assertRaisesRegex(video_import.VideoImportError, "nội bộ"):
                    video_import.ensure_public_url(self.url)

    def test_redirect_to_private_address_is_blocked_before_request(self):
        requested = []

        class RedirectingHTTPHandler(urllib.request.HTTPHandler):
            def http_open(self, request):
                requested.append(request.full_url)
                headers = Message()
                headers["Location"] = "http://127.0.0.1/private"
                response = urllib.response.addinfourl(io.BytesIO(b""), headers, request.full_url, 302)
                response.msg = "Found"
                return response

        def resolve(hostname, *args, **kwargs):
            address = "127.0.0.1" if hostname == "127.0.0.1" else "8.8.8.8"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))]

        opener = urllib.request.build_opener(
            video_import.PublicRequestGuard(time.monotonic() + 30), RedirectingHTTPHandler(),
        )
        with patch.object(socket, "getaddrinfo", side_effect=resolve):
            with self.assertRaisesRegex(video_import.VideoImportError, "nội bộ"):
                opener.open("http://www.youtube.com/watch?v=public123", timeout=15)
        self.assertEqual(requested, ["http://www.youtube.com/watch?v=public123"])

    def test_submission_is_queued_and_polling_returns_a_snapshot(self):
        job = self.create_job()
        self.assertEqual(job.status, "queued")
        self.assertIsNone(job.video)
        self.submit.assert_called_once_with(video_import._run_import, job.import_id, self.url, "youtube")
        snapshot = video_import.get_import(job.import_id)
        snapshot.status = "failed"
        self.assertEqual(video_import.get_import(job.import_id).status, "queued")

    def test_full_queue_rejects_more_work(self):
        with patch.object(video_import, "MAX_PENDING_IMPORTS", 1):
            self.create_job()
            with self.assertRaises(video_import.ImportBusyError):
                self.create_job()
        self.assertEqual(self.submit.call_count, 1)

    def test_success_persists_source_and_only_publishes_finished_video(self):
        job = self.create_job()

        def download(source, directory, import_id, deadline):
            self.assertEqual(storage.list_videos(), [])
            path = directory / "download.mp4"
            path.write_bytes(b"video-content")
            self.assertEqual(storage.list_videos(), [])
            return path, {"title": "Bài giảng thử nghiệm"}

        with patch.object(video_import, "_download_video", side_effect=download), patch.object(storage, "probe_duration_seconds", return_value=12.5):
            video_import._run_import(job.import_id, self.url, "youtube")
        completed = video_import.get_import(job.import_id)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.progress, 100)
        self.assertEqual(completed.video.filename, "Bài giảng thử nghiệm.mp4")
        self.assertEqual(completed.video.source_url, self.url)
        self.assertEqual(completed.video.source_platform, "youtube")
        self.assertEqual(completed.video.duration_seconds, 12.5)
        self.assertFalse(completed.video.has_transcript)
        self.assertEqual(storage.list_videos(), [completed.video])
        self.assertEqual(storage.find_video_file(completed.video.video_id).read_bytes(), b"video-content")
        self.assertEqual(list((self.data / "imports").iterdir()), [])

    def test_failed_download_removes_partials_without_publishing_metadata(self):
        job = self.create_job()

        def fail(source, directory, import_id, deadline):
            (directory / "download.mp4.part").write_bytes(b"unfinished")
            raise RuntimeError("Sign in to confirm this private video")

        with patch.object(video_import, "_download_video", side_effect=fail):
            video_import._run_import(job.import_id, self.url, "youtube")
        failed = video_import.get_import(job.import_id)
        self.assertEqual(failed.status, "failed")
        self.assertIsNone(failed.video)
        self.assertIn("đăng nhập", failed.error)
        self.assertEqual(list((self.data / "videos").iterdir()), [])
        self.assertEqual(list((self.data / "imports").iterdir()), [])

    def test_persistence_failure_cleans_video_and_metadata(self):
        job = self.create_job()

        def download(source, directory, import_id, deadline):
            path = directory / "download.webm"
            path.write_bytes(b"video")
            return path, {"title": "Test"}

        with patch.object(video_import, "_download_video", side_effect=download), patch.object(storage, "probe_duration_seconds", return_value=1), patch.object(storage, "load_video_meta", side_effect=OSError("disk failure")):
            video_import._run_import(job.import_id, self.url, "youtube")
        self.assertEqual(video_import.get_import(job.import_id).status, "failed")
        self.assertEqual(list((self.data / "videos").iterdir()), [])

    def test_rejects_live_private_playlist_and_oversize_metadata(self):
        invalid = (
            {"_type": "playlist", "entries": []}, {"_type": "multi_video"},
            {"is_live": True}, {"live_status": "is_upcoming"}, {"availability": "private"},
            {"duration": video_import.MAX_DURATION_SECONDS + 1},
            {"filesize": video_import.MAX_FILE_BYTES + 1},
        )
        for info in invalid:
            with self.subTest(info=info), self.assertRaises(video_import.VideoImportError):
                video_import._validate_info(info)

    def test_expired_download_deadline_fails(self):
        with self.assertRaisesRegex(video_import.VideoImportError, "30 phút"):
            video_import._remaining_seconds(time.monotonic() - 1)

    def test_missing_ffmpeg_returns_actionable_error(self):
        job = self.create_job()
        with patch.object(video_import.shutil, "which", return_value=None):
            with self.assertRaisesRegex(video_import.VideoImportError, "ffmpeg và ffprobe"):
                video_import._download_video(self.url, self.data, job.import_id, time.monotonic() + 30)

    def test_downloader_enforces_progress_limits_and_uses_single_video_options(self):
        job = self.create_job()
        captured = {}

        class FakeDownloader:
            def __init__(self, options):
                captured.update(options)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, source, download=False):
                return {"title": "Lesson", "duration": 20}

            def process_ie_result(self, info, download=True):
                captured["progress_hooks"][0]({"status": "downloading", "downloaded_bytes": 20, "total_bytes": 40})
                (Path(captured["outtmpl"]).parent / "download.mp4").write_bytes(b"media")
                return info

        with patch("yt_dlp.YoutubeDL", FakeDownloader), patch.object(video_import.shutil, "which", return_value="ffmpeg"), patch.object(video_import, "ensure_public_url"), patch.object(video_import, "_browser_video", side_effect=lambda path, deadline: path):
            path, info = video_import._download_video(self.url, self.data, job.import_id, time.monotonic() + 30)
        self.assertEqual(path.read_bytes(), b"media")
        self.assertEqual(info["title"], "Lesson")
        self.assertTrue(captured["noplaylist"])
        self.assertEqual(captured["playlistend"], 1)
        self.assertEqual(captured["max_filesize"], video_import.MAX_FILE_BYTES)
        self.assertEqual(captured["socket_timeout"], 15)
        self.assertEqual(video_import.get_import(job.import_id).progress, 97)
        with self.assertRaisesRegex(video_import.VideoImportError, "1 GiB"):
            captured["progress_hooks"][0]({"status": "downloading", "downloaded_bytes": video_import.MAX_FILE_BYTES + 1})

    def test_browser_verification_keeps_compatible_media_and_rejects_audio_only(self):
        path = self.data / "video.mp4"
        metadata = {"streams": [{"codec_type": "video", "codec_name": "h264"}, {"codec_type": "audio", "codec_name": "aac"}], "format": {"duration": "10.5"}}
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(metadata))) as run:
            self.assertEqual(video_import._browser_video(path, time.monotonic() + 30), path)
            self.assertEqual(run.call_count, 1)
        metadata["streams"] = [{"codec_type": "audio", "codec_name": "aac"}]
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(metadata))):
            with self.assertRaisesRegex(video_import.VideoImportError, "không chứa video"):
                video_import._browser_video(path, time.monotonic() + 30)


class VideoImportRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_url_returns_400(self):
        with self.assertRaises(HTTPException) as raised:
            await import_video(VideoImportRequest(url="http://localhost/video.mp4"))
        self.assertEqual(raised.exception.status_code, 400)

    async def test_busy_queue_returns_429(self):
        with patch.object(video_import, "start_import", side_effect=video_import.ImportBusyError("Đang bận")):
            with self.assertRaises(HTTPException) as raised:
                await import_video(VideoImportRequest(url="https://youtu.be/test"))
        self.assertEqual(raised.exception.status_code, 429)

    async def test_unknown_import_returns_404(self):
        with self.assertRaises(HTTPException) as raised:
            await get_video_import("unknown-import")
        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
