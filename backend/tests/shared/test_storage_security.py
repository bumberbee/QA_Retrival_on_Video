from tests import _isolate  # noqa: F401  (must come before any app import)
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.shared import storage
from app.shared.models import Transcript

VIDEO_ID = "abc123def456"
BAD_IDS = ("..", "../..", r"..\..\..", "%2e%2e", "%2e%2e%5c%2e%2e", "/etc/passwd",
           r"C:\Windows\win.ini", r"\server\share", "abc123def456/..", "abc123def456\\..",
           "abc123def456.json", "ABC123DEF456", "abc123def456\n", "", "*", "L25_V011")


class StorageFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name, folder in (("DATA_DIR", "data"), ("VIDEOS_DIR", "videos"),
                             ("TRANSCRIPTS_DIR", "transcripts"), ("KEYFRAMES_DIR", "keyframes"),
                             ("SLIDE_VARIANTS_DIR", "slide_variants"), ("SLIDE_REVIEWS_DIR", "slide_reviews")):
            path = self.root / folder
            path.mkdir()
            patcher = patch.object(storage, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.dict(os.environ, {"ADMIN_TOKEN": "", "SUPABASE_DB_URL": "", "MAX_UPLOAD_MB": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)


class StorageSecurityTests(StorageFixture):
    def test_existing_id_format_and_generated_ids_are_accepted(self):
        for video_id in (VIDEO_ID, "0bc38ac7ee13", "fc75a29f8f7c", storage.new_video_id()):
            self.assertEqual(storage.validate_video_id(video_id), video_id)
            self.assertTrue(storage.video_path(video_id, ".mp4").is_relative_to(storage.VIDEOS_DIR.resolve()))

    def test_every_id_helper_rejects_traversal_before_reading_or_writing(self):
        sentinel = self.root / ".env"
        sentinel.write_text("untouched")
        helpers = (
            lambda value: storage.video_path(value, ".mp4"), storage.video_meta_path,
            storage.transcript_path, storage.keyframes_dir, storage.find_video_file,
            storage.load_video_meta, storage.load_transcript, storage.list_keyframes,
            storage.load_slide_variants, storage.load_slide_reviews,
            lambda value: storage.slide_file(value, ".env"),
            lambda value: storage.save_transcript(Transcript(video_id=value, segments=[])),
            lambda value: storage.save_slide_reviews(value, {}),
            lambda value: storage.save_video_meta(value, "lecture.mp4", sentinel),
        )
        for value in BAD_IDS:
            for index, helper in enumerate(helpers):
                with self.subTest(value=value, helper=index), self.assertRaises(storage.InvalidVideoId):
                    helper(value)
        self.assertEqual(sentinel.read_text(), "untouched")
        self.assertEqual(list(storage.TRANSCRIPTS_DIR.iterdir()), [])
        self.assertEqual(list(storage.SLIDE_REVIEWS_DIR.iterdir()), [])

    def test_cloud_helpers_validate_ids_before_calling_supabase(self):
        with patch.dict(os.environ, {"SUPABASE_DB_URL": "mock-cloud"}), patch.object(storage, "_supa") as supa:
            for helper in (storage.load_video_meta, storage.load_transcript, storage.list_keyframes,
                           storage.load_slide_variants, storage.load_slide_reviews,
                           lambda value: storage.save_slide_reviews(value, {})):
                with self.assertRaises(storage.InvalidVideoId):
                    helper(r"..\..")
            supa.assert_not_called()

    def test_slide_paths_allow_nested_images_and_reject_escapes(self):
        folder = storage.keyframes_dir(VIDEO_ID) / "fast"
        folder.mkdir(parents=True)
        image = folder / "state_00012.jpg"
        image.write_bytes(b"jpg")
        self.assertEqual(storage.slide_file(VIDEO_ID, "fast/state_00012.jpg"), image)
        for name in ("../../../.env", r"..\..\.env", str(self.root / ".env"),
                     r"C:\Windows\win.ini", "/etc/passwd", "%2e%2e/.env", "state.jpg:secret"):
            with self.subTest(name=name):
                self.assertIsNone(storage.slide_file(VIDEO_ID, name))

    def test_video_extensions_cannot_inject_a_path(self):
        for ext in ("/../../.env", r"\..\.env", ".mp4/../.env", ".mp4:secret", ""):
            with self.subTest(ext=ext), self.assertRaises(storage.UnsafeStoragePath):
                storage.video_path(VIDEO_ID, ext)

    def test_resolved_keyframe_directory_cannot_escape_through_a_link(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / ".env").write_text("private")
        link = storage.KEYFRAMES_DIR / VIDEO_ID
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                raise
            # Directory junctions need no symlink privilege on Windows.
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                           capture_output=True, check=True)
        with self.assertRaises(storage.UnsafeStoragePath):
            storage.keyframes_dir(VIDEO_ID)
        self.assertIsNone(storage.slide_file(VIDEO_ID, ".env"))
        with self.assertRaises(storage.UnsafeStoragePath):
            storage.list_keyframes(VIDEO_ID)

    def test_cloud_metadata_and_transcript_writes_are_rejected_without_local_files(self):
        with patch.dict(os.environ, {"SUPABASE_DB_URL": "mock-cloud"}), patch.object(storage, "_supa") as supa:
            with self.assertRaises(storage.CloudWriteUnsupported):
                storage.save_video_meta(VIDEO_ID, "lecture.mp4", self.root / "lecture.mp4")
            with self.assertRaises(storage.CloudWriteUnsupported):
                storage.save_transcript(Transcript(video_id=VIDEO_ID, segments=[]))
            supa.assert_not_called()
        self.assertEqual(list(storage.VIDEOS_DIR.iterdir()), [])
        self.assertEqual(list(storage.TRANSCRIPTS_DIR.iterdir()), [])

    def test_cloud_slide_reviews_keep_the_supported_database_write(self):
        supa = Mock()
        with patch.dict(os.environ, {"SUPABASE_DB_URL": "mock-cloud"}), patch.object(storage, "_supa", return_value=supa):
            storage.save_slide_reviews(VIDEO_ID, {"F": {}})
        supa.save_slide_reviews.assert_called_once_with(VIDEO_ID, {"F": {}})
        self.assertEqual(list(storage.SLIDE_REVIEWS_DIR.iterdir()), [])

    def test_failed_atomic_transcript_replace_preserves_existing_file_and_cleans_temp(self):
        old = Transcript(video_id=VIDEO_ID, language="vi", segments=[])
        storage.save_transcript(old)
        with patch.object(Path, "replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                storage.save_transcript(Transcript(video_id=VIDEO_ID, language="en", segments=[]))
        self.assertEqual(storage.load_transcript(VIDEO_ID), old)
        self.assertEqual([p.name for p in storage.TRANSCRIPTS_DIR.iterdir()], [f"{VIDEO_ID}.json"])

    def test_library_skips_unfinished_files_and_invalid_ids(self):
        (storage.VIDEOS_DIR / ".upload-example.part").write_bytes(b"unfinished")
        (storage.VIDEOS_DIR / "not-a-video.mp4").write_bytes(b"invalid")
        (storage.VIDEOS_DIR / "not-a-video.meta.json").write_text(json.dumps({"original_filename": "invalid.mp4"}))
        self.assertEqual(storage.list_videos(), [])
        self.assertIsNone(storage.find_video_id_by_original_filename("invalid.mp4"))
