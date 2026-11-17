from tests import _isolate  # noqa: F401

import importlib
import io
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

from app import main
from app.features.extract import jobs
from app.features.extract.importer import import_video
from app.features.extract.models import ExtractionStatus
from app.features.ask import llm
from app.features.videos import importer as video_import, router as videos
from app.shared import storage
from app.shared.frontend import mount_frontend
from app.shared.models import Transcript

VIDEO_ID = "abc123def456"
SECOND_ID = "abc123def457"
FAKE = Path(__file__).with_name("fake_pipeline.py")


@pytest.fixture
def data(tmp_path, monkeypatch):
    for name, folder in (("DATA_DIR", "data"), ("VIDEOS_DIR", "videos"),
                         ("TRANSCRIPTS_DIR", "transcripts"), ("KEYFRAMES_DIR", "keyframes"),
                         ("SLIDE_VARIANTS_DIR", "slide_variants"), ("SLIDE_REVIEWS_DIR", "slide_reviews")):
        directory = tmp_path / folder
        directory.mkdir()
        monkeypatch.setattr(storage, name, directory)
    monkeypatch.setenv("LECTURE_EXTRACT", "on")
    monkeypatch.setenv("ADMIN_TOKEN", "")
    monkeypatch.setenv("SUPABASE_DB_URL", "")
    monkeypatch.setenv("LECTURE_EXTRACT_LANGUAGE", "vi")
    monkeypatch.setenv("LECTURE_EXTRACT_DEVICE", "cpu")
    monkeypatch.delenv("FAKE_PIPELINE_RELEASE", raising=False)
    monkeypatch.setenv("FAKE_PIPELINE_MODE", "success")
    storage.video_path(VIDEO_ID, ".mp4").write_bytes(b"video")
    storage.video_path(SECOND_ID, ".mp4").write_bytes(b"video")
    return tmp_path


@pytest.fixture
def extraction_queue(data, monkeypatch):
    command = jobs.pipeline_command
    monkeypatch.setattr(jobs, "pipeline_command", lambda *args: [sys.executable, str(FAKE), *command(*args)[3:]])
    instance = jobs.ExtractionQueue(storage.DATA_DIR)
    yield instance
    instance.close()


def wait_for(predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("Extraction did not reach expected state")


def finished(instance, video_id=VIDEO_ID):
    return wait_for(lambda: (job if (job := instance.get(video_id)) and job.status in {"completed", "failed"} else None))


def test_contract_command_uses_separate_python_and_repository_root(data, monkeypatch):
    monkeypatch.setenv("PIPELINE_PYTHON", "pipeline-python")
    work = data / "work"
    assert jobs.pipeline_command(storage.find_video_file(VIDEO_ID), VIDEO_ID, work) == [
        "pipeline-python", "-m", "pipeline.run_video", "--video", str(storage.find_video_file(VIDEO_ID)),
        "--video-id", VIDEO_ID, "--out", str(work), "--language", "vi", "--device", "cpu",
    ]
    assert jobs.REPO_ROOT == Path(__file__).resolve().parents[4]


def test_all_transitions_progress_and_real_export_import(extraction_queue, monkeypatch):
    seen = []
    save = extraction_queue._save
    monkeypatch.setattr(extraction_queue, "_save", lambda job: (seen.append(job.model_copy()), save(job)))
    queued = extraction_queue.enqueue(VIDEO_ID)
    assert queued.status == "queued"
    assert finished(extraction_queue).status == "completed"
    assert [job.status for job in seen] == ["queued", "extracting", "extracting", "extracting", "importing", "completed"]
    assert any(job.stage == "s03_ocr" and job.progress == 0.42 for job in seen)
    assert any(job.stage == "s03_ocr" and job.overall == 0.318 and job.phase == "working" for job in seen)
    assert (seen[1].overall, seen[-1].overall, seen[-1].phase) == (0, 1, None)
    assert storage.load_transcript(VIDEO_ID).segments[0].text == "Test lecture"
    assert storage.list_keyframes(VIDEO_ID) == ["seg_00000.jpg"]
    assert storage.load_slide_variants(VIDEO_ID)["variants"]["F_fast_pipeline"]["slides"][0]["image"] == "fast/state_00000.jpg"
    assert (storage.DATA_DIR / "index" / VIDEO_ID / "embeddings.npy").read_bytes() == b"fixture-embeddings"
    work = next((extraction_queue.root / "work" / VIDEO_ID).iterdir())
    assert Path((work / "cwd.txt").read_text(encoding="utf-8")) == jobs.REPO_ROOT
    assert "diagnostic log" in (work / "pipeline.stderr.log").read_text(encoding="utf-8")
    assert json.loads(extraction_queue._state_path(VIDEO_ID).read_text(encoding="utf-8"))["status"] == "completed"


def test_only_one_video_extracts_at_a_time(extraction_queue, data, monkeypatch):
    release = data / "release"
    monkeypatch.setenv("FAKE_PIPELINE_RELEASE", str(release))
    extraction_queue.enqueue(VIDEO_ID)
    wait_for(lambda: extraction_queue.get(VIDEO_ID).stage == "s03_ocr")
    extraction_queue.enqueue(SECOND_ID)
    assert extraction_queue.get(SECOND_ID).status == "queued"
    assert not (extraction_queue.root / "work" / SECOND_ID).exists()
    release.touch()
    assert finished(extraction_queue).status == "completed"
    assert finished(extraction_queue, SECOND_ID).status == "completed"


def test_importing_is_visible_until_import_finishes(extraction_queue, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = jobs.import_video
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(jobs, "import_video", blocked)
    extraction_queue.enqueue(VIDEO_ID)
    try:
        assert entered.wait(5)
        assert extraction_queue.get(VIDEO_ID).status == "importing"
        assert extraction_queue.get(VIDEO_ID).stage == "import"
    finally:
        release.set()
    assert finished(extraction_queue).status == "completed"


@pytest.mark.parametrize("mode", ["error", "error_zero", "silent", "nonzero", "wrong_export", "missing_index"])
def test_unsuccessful_pipeline_is_failed_in_vietnamese(extraction_queue, monkeypatch, mode):
    monkeypatch.setenv("FAKE_PIPELINE_MODE", mode)
    extraction_queue.enqueue(VIDEO_ID)
    job = finished(extraction_queue)
    assert job.status == "failed"
    assert job.progress is None
    assert "thất bại" in job.error or "Không thể nạp" in job.error
    assert "sensitive internal diagnostic" not in job.error
    assert not storage.transcript_path(VIDEO_ID).exists()


def test_missing_executable_is_failed(extraction_queue, monkeypatch):
    monkeypatch.setattr(jobs, "pipeline_command", lambda *args: ["missing-pipeline-executable-xyz"])
    extraction_queue.enqueue(VIDEO_ID)
    assert "PIPELINE_PYTHON" in finished(extraction_queue).error


def test_duplicate_submission_returns_snapshot_and_failed_job_can_retry(extraction_queue, data, monkeypatch):
    monkeypatch.setenv("FAKE_PIPELINE_MODE", "error")
    release = data / "release"
    monkeypatch.setenv("FAKE_PIPELINE_RELEASE", str(release))
    queued = extraction_queue.enqueue(VIDEO_ID)
    wait_for(lambda: extraction_queue.get(VIDEO_ID).stage == "s03_ocr")
    assert extraction_queue.enqueue(VIDEO_ID).updated_at == extraction_queue.get(VIDEO_ID).updated_at
    queued.status = "completed"
    assert extraction_queue.get(VIDEO_ID).status == "extracting"
    release.touch()
    assert finished(extraction_queue).status == "failed"
    monkeypatch.setenv("FAKE_PIPELINE_MODE", "success")
    assert extraction_queue.enqueue(VIDEO_ID).status == "queued"
    assert finished(extraction_queue).status == "completed"
    assert extraction_queue.enqueue(VIDEO_ID).status == "completed"
    assert len(list((extraction_queue.root / "work" / VIDEO_ID).iterdir())) == 2


def test_restart_retains_finished_and_failed_states_and_fails_interrupted_jobs(data):
    instance = jobs.ExtractionQueue(storage.DATA_DIR)
    for video_id, status in ((VIDEO_ID, "completed"), (SECOND_ID, "failed"), ("abc123def458", "extracting"), ("abc123def459", "queued"), ("abc123def450", "importing")):
        instance._save(ExtractionStatus(video_id=video_id, status=status, updated_at=jobs._now()))
    instance.close()
    recovered = jobs.ExtractionQueue(storage.DATA_DIR)
    try:
        assert recovered.get(VIDEO_ID).status == "completed"
        assert recovered.get(SECOND_ID).status == "failed"
        for video_id in ("abc123def458", "abc123def459", "abc123def450"):
            assert recovered.get(video_id).status == "failed"
            assert "khởi động lại" in recovered.get(video_id).error
    finally:
        recovered.close()


def test_script_import_preserves_previous_variants_and_transcript(extraction_queue):
    extraction_queue.enqueue(VIDEO_ID)
    assert finished(extraction_queue).status == "completed"
    storage.save_transcript(Transcript(video_id=VIDEO_ID, segments=[{"start": 0, "end": 1, "text": "Original"}]))
    path = storage.SLIDE_VARIANTS_DIR / f"{VIDEO_ID}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["variants"]["old"] = {"slides": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    work = next((extraction_queue.root / "work" / VIDEO_ID).iterdir())
    export = work / "export" / VIDEO_ID
    with patch.object(storage, "find_video_id_by_original_filename", return_value=VIDEO_ID):
        assert import_video(export) == VIDEO_ID
    assert "old" in storage.load_slide_variants(VIDEO_ID)["variants"]
    assert storage.load_transcript(VIDEO_ID).segments[0].text == "Original"


def test_endpoints_return_202_and_status_snapshot(data, monkeypatch):
    instance = Mock()
    job = ExtractionStatus(video_id=VIDEO_ID, status="queued", progress=0, updated_at=jobs._now())
    instance.enqueue.return_value = instance.get.return_value = job
    monkeypatch.setattr(jobs, "get_queue", lambda: instance)
    api = TestClient(main.app)
    assert api.post(f"/api/videos/{VIDEO_ID}/extract").status_code == 202
    assert api.get(f"/api/videos/{VIDEO_ID}/extract").json()["status"] == "queued"
    instance.enqueue.assert_called_once_with(VIDEO_ID)


def test_disabled_endpoints_return_503_without_queue_or_admin_check(data, monkeypatch):
    monkeypatch.setenv("LECTURE_EXTRACT", "off")
    monkeypatch.setenv("ADMIN_TOKEN", "admin")
    instance = Mock()
    monkeypatch.setattr(jobs, "get_queue", instance)
    api = TestClient(main.app)
    for method in (api.get, api.post):
        response = method(f"/api/videos/{VIDEO_ID}/extract")
        assert response.status_code == 503
        assert "LECTURE_EXTRACT=on" in response.json()["detail"]
    instance.assert_not_called()


def test_extraction_write_requires_import_admin_token_and_local_mode(data, monkeypatch):
    api = TestClient(main.app)
    monkeypatch.setenv("ADMIN_TOKEN", "admin")
    with patch.object(jobs, "get_queue") as getter:
        assert api.post(f"/api/videos/{VIDEO_ID}/extract").status_code == 403
        assert api.post(f"/api/videos/{VIDEO_ID}/extract", headers={"X-Admin-Token": "wrong"}).status_code == 403
        monkeypatch.setenv("SUPABASE_DB_URL", "mock-cloud")
        assert api.post(f"/api/videos/{VIDEO_ID}/extract", headers={"X-Admin-Token": "admin"}).status_code == 403
        getter.assert_not_called()


def test_missing_jobs_videos_and_invalid_ids(data, monkeypatch):
    instance = Mock()
    instance.get.return_value = None
    instance.enqueue.side_effect = FileNotFoundError
    monkeypatch.setattr(jobs, "get_queue", lambda: instance)
    api = TestClient(main.app)
    assert api.get(f"/api/videos/{VIDEO_ID}/extract").status_code == 404
    assert api.post("/api/videos/abc123def450/extract").status_code == 404
    assert api.get("/api/videos/invalid/extract").status_code == 400


def test_existing_export_is_ready_without_queue_state(extraction_queue, monkeypatch):
    extraction_queue.enqueue(VIDEO_ID)
    assert finished(extraction_queue).status == "completed"
    instance = Mock()
    instance.get.return_value = None
    monkeypatch.setattr(jobs, "get_queue", lambda: instance)
    assert TestClient(main.app).get(f"/api/videos/{VIDEO_ID}/extract").json()["status"] == "completed"


@pytest.mark.parametrize("enabled", [True, False])
def test_upload_auto_enqueues_only_when_enabled(data, monkeypatch, enabled):
    monkeypatch.setenv("LECTURE_EXTRACT", "on" if enabled else "off")
    with patch.object(jobs, "get_queue") as getter, patch.object(storage, "probe_duration_seconds", return_value=5):
        video = videos.upload_video(UploadFile(filename="uploaded.mp4", file=io.BytesIO(b"video")))
        if enabled:
            getter.return_value.enqueue.assert_called_once_with(video.video_id)
        else:
            getter.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_url_import_auto_enqueues_after_success(data, monkeypatch, enabled):
    monkeypatch.setenv("LECTURE_EXTRACT", "on" if enabled else "off")
    def download(source, directory, import_id, deadline):
        path = directory / "download.mp4"
        path.write_bytes(b"video")
        return path, {"title": "Lesson"}
    with patch.object(video_import._executor, "submit"), patch.object(video_import, "_download_video", side_effect=download), \
            patch.object(storage, "probe_duration_seconds", return_value=5), patch.object(jobs, "get_queue") as getter:
        job = video_import.start_import("https://youtu.be/lesson")
        video_import._run_import(job.import_id, "https://youtu.be/lesson", "youtube")
        completed = video_import.get_import(job.import_id)
        assert completed.status == "completed"
        if enabled:
            getter.return_value.enqueue.assert_called_once_with(completed.video.video_id)
        else:
            getter.assert_not_called()


def test_failed_import_does_not_enqueue_and_queue_error_keeps_successful_upload(data):
    with patch.object(video_import._executor, "submit"), patch.object(video_import, "_download_video", side_effect=RuntimeError), \
            patch.object(jobs, "get_queue") as getter:
        job = video_import.start_import("https://youtu.be/lesson")
        video_import._run_import(job.import_id, "https://youtu.be/lesson", "youtube")
        assert video_import.get_import(job.import_id).status == "failed"
        getter.assert_not_called()
    with patch.object(jobs, "get_queue", side_effect=OSError), patch.object(storage, "probe_duration_seconds", return_value=5):
        video = videos.upload_video(UploadFile(filename="uploaded.mp4", file=io.BytesIO(b"video")))
        assert storage.find_video_file(video.video_id) is not None


def test_static_assets_spa_and_api_priority(data):
    dist = data / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>SPA</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('app')", encoding="utf-8")
    app = FastAPI()
    app.include_router(videos.router)
    mount_frontend(app, str(dist))
    api = TestClient(app)
    for path in ("/", "/library/lecture", "/nested/route"):
        assert api.get(path).text == "<html>SPA</html>"
    assert api.get("/assets/app.js").text == "console.log('app')"
    assert api.head("/library/lecture").status_code == 200
    assert api.get("/api/videos").json()[0]["video_id"] == VIDEO_ID
    assert api.get("/api/missing").status_code == 404
    assert api.get("/api").status_code == 404
    assert api.post("/library/lecture").status_code == 405


def test_frontend_env_mounts_build_and_is_optional(data, monkeypatch):
    dist = data / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("frontend from env", encoding="utf-8")
    monkeypatch.setenv("SERVE_FRONTEND_DIR", str(dist))
    try:
        importlib.reload(main)
        assert TestClient(main.app).get("/client/route").text == "frontend from env"
        assert TestClient(main.app).get("/api/health").json()["status"] == "ok"
    finally:
        monkeypatch.setenv("SERVE_FRONTEND_DIR", "")
        importlib.reload(main)
    assert TestClient(main.app).get("/client/route").status_code == 404


@pytest.mark.parametrize("stream", [False, True])
def test_allinone_ollama_releases_gpu_after_answer(monkeypatch, stream):
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "0")
    response = Mock()
    response.read.return_value = b'{"message":{"content":"Answer"}}'
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.__iter__ = Mock(return_value=iter([b'{"message":{"content":"Answer"},"done":true}\n']))
    with patch.object(urllib.request, "urlopen", return_value=response) as opened:
        assert (list(llm._ollama_stream("Question")) if stream else llm._ollama("Question")) == (["Answer"] if stream else "Answer")
        assert json.loads(opened.call_args.args[0].data)["keep_alive"] == "0"
