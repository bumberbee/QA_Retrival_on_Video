import _isolate  # noqa: F401  (must come before any app import)
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, Request

from app.routers.videos import get_video_file
from app.services.media import video_response


class VideoStreamingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "lecture.mp4"
        self.content = bytes(range(256)) * 5000
        self.path.write_bytes(self.content)

    def request(self, range_value=None, method="GET", if_range=None):
        headers = []
        if range_value is not None:
            headers.append((b"range", range_value.encode()))
        if if_range is not None:
            headers.append((b"if-range", if_range.encode()))
        return Request({"type": "http", "method": method, "headers": headers})

    async def read_response(self, request):
        response = video_response(self.path, request)
        messages = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        await response(request.scope, receive, send)
        body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        return response, body

    async def test_full_video_advertises_range_support(self):
        response, body = await self.read_response(self.request())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["accept-ranges"], "bytes")
        self.assertEqual(response.headers["content-type"], "video/mp4")
        self.assertEqual(int(response.headers["content-length"]), len(self.content))
        self.assertEqual(body, self.content)

    async def test_bounded_range_returns_only_requested_bytes(self):
        response, body = await self.read_response(self.request("bytes=12345-23456"))
        self.assertEqual(response.status_code, 206)
        self.assertEqual(body, self.content[12345:23457])
        self.assertEqual(
            response.headers["content-range"],
            f"bytes 12345-23456/{len(self.content)}",
        )
        self.assertEqual(int(response.headers["content-length"]), len(body))

    async def test_open_ended_range_can_cross_stream_chunks(self):
        response, body = await self.read_response(self.request("bytes=100-"))
        self.assertEqual(response.status_code, 206)
        self.assertEqual(body, self.content[100:])

    async def test_suffix_range_reads_the_end_of_the_video(self):
        response, body = await self.read_response(self.request("bytes=-1024"))
        self.assertEqual(response.status_code, 206)
        self.assertEqual(body, self.content[-1024:])

    async def test_ranges_are_clamped_to_the_file_size(self):
        _, body = await self.read_response(self.request("bytes=100-999999999"))
        self.assertEqual(body, self.content[100:])
        _, body = await self.read_response(self.request("bytes=-999999999"))
        self.assertEqual(body, self.content)

    async def test_separate_forward_and_backward_requests_read_correct_offsets(self):
        for start in [0, 900000, 200000, 1200000, 40]:
            with self.subTest(start=start):
                response, body = await self.read_response(
                    self.request(f"bytes={start}-{start + 1023}")
                )
                self.assertEqual(response.status_code, 206)
                self.assertEqual(body, self.content[start:start + 1024])

    async def test_invalid_and_unsatisfiable_ranges_report_size(self):
        for value in ["bytes=-", "bytes=-0", "bytes=20-10", "bytes=x-y", "bytes=1280000-"]:
            with self.subTest(value=value):
                with self.assertRaises(HTTPException) as raised:
                    video_response(self.path, self.request(value))
                self.assertEqual(raised.exception.status_code, 416)
                self.assertEqual(
                    raised.exception.headers["Content-Range"],
                    f"bytes */{len(self.content)}",
                )

    async def test_empty_video_has_no_satisfiable_range(self):
        self.path.write_bytes(b"")
        response, body = await self.read_response(self.request())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, b"")
        with self.assertRaises(HTTPException) as raised:
            video_response(self.path, self.request("bytes=0-"))
        self.assertEqual(raised.exception.headers["Content-Range"], "bytes */0")

    async def test_head_returns_metadata_without_a_body(self):
        response, body = await self.read_response(self.request("bytes=0-10", method="HEAD"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(int(response.headers["content-length"]), len(self.content))
        self.assertEqual(body, b"")

    async def test_if_range_uses_the_existing_file_validator(self):
        full = video_response(self.path, self.request())
        for validator in [full.headers["etag"], full.headers["last-modified"]]:
            response, body = await self.read_response(
                self.request("bytes=0-31", if_range=validator)
            )
            self.assertEqual(response.status_code, 206)
            self.assertEqual(body, self.content[:32])
        response, body = await self.read_response(
            self.request("bytes=0-31", if_range='"outdated-video"')
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, self.content)

    async def test_other_units_and_multiple_ranges_fall_back_to_full_response(self):
        for value in ["items=0-10", "bytes=0-10,20-30"]:
            response, body = await self.read_response(self.request(value))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(body, self.content)

    async def test_missing_video_keeps_the_existing_404(self):
        with patch("app.routers.videos.storage.find_video_file", return_value=None):
            with self.assertRaises(HTTPException) as raised:
                await get_video_file("missing", self.request())
            self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
