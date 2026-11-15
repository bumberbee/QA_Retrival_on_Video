"""One daemon worker; atomic JSON snapshots survive application restarts.

Run one uvicorn process on the GPU host. Interrupted jobs are failed on recovery,
so an explicit retry starts a fresh export rather than reusing partial artifacts.
"""

import json
import logging
import math
import os
import queue
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.shared import storage
from .importer import import_video
from .models import ExtractionStatus

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[4]
ACTIVE = {"queued", "extracting", "importing"}
STAGE_NAMES = {
    "s01": "tách slide", "s02": "lọc khung", "s03": "OCR",
    "s04": "nhận dạng giọng nói", "s05": "đồng bộ", "s06": "tạo chỉ mục",
}


def enabled() -> bool:
    return os.getenv("LECTURE_EXTRACT", "").strip().lower() == "on"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pipeline_command(video: Path, video_id: str, work: Path) -> list[str]:
    language = os.getenv("LECTURE_EXTRACT_LANGUAGE", "auto").strip().lower()
    device = os.getenv("LECTURE_EXTRACT_DEVICE", "").strip().lower()
    if language not in {"vi", "en", "auto"} or device not in {"", "cuda", "cpu"}:
        raise ValueError("Invalid extraction configuration")
    command = [os.getenv("PIPELINE_PYTHON") or sys.executable, "-m", "pipeline.run_video",
               "--video", str(video), "--video-id", video_id, "--out", str(work),
               "--language", language]
    if device:
        command += ["--device", device]
    return command


class ExtractionQueue:
    def __init__(self, data_dir: Path):
        self.root = storage.contained_path(Path(data_dir), "extract")
        self.states = storage.contained_path(self.root, "jobs")
        self.states.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, ExtractionStatus] = {}
        self._lock = threading.RLock()
        self._pending = queue.Queue()
        self._closed = False
        self._process: subprocess.Popen | None = None
        self._recover()
        self._worker = threading.Thread(target=self._work, name="video-extract", daemon=True)
        self._worker.start()

    def _state_path(self, video_id: str) -> Path:
        return storage.contained_path(self.states, f"{storage.validate_video_id(video_id)}.json")

    def _save(self, job: ExtractionStatus) -> None:
        storage._write_text(self._state_path(job.video_id), job.model_dump_json(indent=2))

    def _recover(self) -> None:
        for path in self.states.glob("*.json"):
            try:
                video_id = storage.validate_video_id(path.stem)
                job = ExtractionStatus.model_validate_json(self._state_path(video_id).read_text(encoding="utf-8"))
                if job.video_id != video_id:
                    raise ValueError("Mismatched job ID")
                if job.status in ACTIVE:
                    job = job.model_copy(update={"status": "failed", "progress": None,
                        "error": "Trích xuất bị gián đoạn do máy chủ khởi động lại. Vui lòng thử lại.",
                        "updated_at": _now()})
                    self._save(job)
                self._jobs[video_id] = job
            except (ValueError, OSError):
                logger.warning("Could not recover extraction state %s", path.stem)

    def get(self, video_id: str) -> ExtractionStatus | None:
        storage.validate_video_id(video_id)
        with self._lock:
            job = self._jobs.get(video_id)
            return job.model_copy(deep=True) if job else None

    def enqueue(self, video_id: str) -> ExtractionStatus:
        storage.validate_video_id(video_id)
        storage.require_local_writes()
        if not storage.find_video_file(video_id):
            raise FileNotFoundError("Video not found")
        with self._lock:
            if self._closed:
                raise RuntimeError("Extraction queue closed")
            existing = self._jobs.get(video_id)
            if existing and existing.status != "failed":
                return existing.model_copy(deep=True)
            job = ExtractionStatus(video_id=video_id, status="queued", progress=0, updated_at=_now())
            self._save(job)
            self._jobs[video_id] = job
            self._pending.put(video_id)
            return job.model_copy(deep=True)

    def _update(self, video_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs[video_id].model_copy(update={**changes, "updated_at": _now()})
            self._save(job)
            self._jobs[video_id] = job

    def _work(self) -> None:
        while True:
            video_id = self._pending.get()
            try:
                if video_id is None:
                    return
                if not self._closed:
                    self._run(video_id)
            finally:
                self._pending.task_done()

    def _run(self, video_id: str) -> None:
        stage = None
        try:
            self._update(video_id, status="extracting", stage=None, progress=0, error=None)
            video = storage.find_video_file(video_id)
            if not video:
                raise FileNotFoundError("Video not found")
            work = storage.contained_path(self.root, "work", video_id, uuid.uuid4().hex)
            work.mkdir(parents=True, exist_ok=True)
            expected_export = work / "export" / video_id
            done = False
            pipeline_error = False
            # A file for stderr avoids a full stderr pipe blocking stdout progress.
            with (work / "pipeline.stderr.log").open("w", encoding="utf-8") as stderr:
                with subprocess.Popen(pipeline_command(video, video_id, work), cwd=REPO_ROOT,
                        stdout=subprocess.PIPE, stderr=stderr, text=True, encoding="utf-8",
                        errors="replace", env={**os.environ, "PYTHONUNBUFFERED": "1"}) as process:
                    with self._lock:
                        self._process = process
                        if self._closed:
                            process.terminate()
                    try:
                        for line in process.stdout:
                            try:
                                event = json.loads(line)
                            except (ValueError, TypeError):
                                continue
                            if not isinstance(event, dict):
                                continue
                            if event.get("event") == "stage":
                                candidate = event.get("stage")
                                if isinstance(candidate, str) and candidate.split("_")[0] in STAGE_NAMES:
                                    stage = candidate
                                value = event.get("progress")
                                progress = (max(0, min(1, value)) if isinstance(value, (int, float))
                                            and math.isfinite(value) else None)
                                self._update(video_id, stage=stage, progress=progress)
                            elif event.get("event") == "done":
                                export = event.get("export_dir")
                                if isinstance(export, str):
                                    reported = Path(export)
                                    if not reported.is_absolute():
                                        reported = REPO_ROOT / reported
                                    done = reported.resolve() == expected_export.resolve()
                            elif event.get("event") == "error":
                                candidate = event.get("stage")
                                if isinstance(candidate, str) and candidate.split("_")[0] in STAGE_NAMES:
                                    stage = candidate
                                pipeline_error = True
                        returncode = process.wait()
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
                    finally:
                        with self._lock:
                            self._process = None
            if returncode != 0 or not done or pipeline_error:
                raise RuntimeError("Pipeline did not finish successfully")
            self._update(video_id, status="importing", stage="import", progress=0)
            import_video(expected_export, video_id=video_id, require_index=True, replace_transcript=True)
            self._update(video_id, status="completed", stage=None, progress=1, error=None)
        except Exception as error:
            logger.warning("Extraction %s failed: %s", video_id, type(error).__name__)
            name = STAGE_NAMES.get((stage or "").split("_")[0])
            message = ("Không thể nạp dữ liệu trích xuất. Kiểm tra nhật ký máy chủ và thử lại."
                       if self.get(video_id).status == "importing" else
                       f"Trích xuất video thất bại{f' ở bước {name}' if name else ''}. "
                       "Kiểm tra PIPELINE_PYTHON và nhật ký máy chủ, rồi thử lại.")
            try:
                self._update(video_id, status="failed", stage=stage, progress=None, error=message)
            except OSError:
                logger.error("Could not persist failed extraction %s", video_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
            self._pending.put(None)
        self._worker.join(timeout=10)


_queue: ExtractionQueue | None = None
_queue_lock = threading.Lock()


def get_queue() -> ExtractionQueue:
    global _queue
    with _queue_lock:
        if _queue is None:
            _queue = ExtractionQueue(storage.DATA_DIR)
        return _queue


def shutdown() -> None:
    global _queue
    with _queue_lock:
        if _queue is not None:
            _queue.close()
            _queue = None


def enqueue_after_import(video_id: str) -> None:
    if enabled():
        try:
            get_queue().enqueue(video_id)
        except Exception as error:
            # A queue failure must not roll back a successfully stored video.
            logger.error("Could not enqueue extraction %s: %s", video_id, type(error).__name__)
