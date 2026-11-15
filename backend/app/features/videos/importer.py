import ipaddress
import json
import logging
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

from app.shared import storage
from app.shared.models import VideoImportStatus, VideoInfo

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 1024 * 1024 * 1024
MAX_DURATION_SECONDS = 2 * 60 * 60
MAX_IMPORT_SECONDS = 30 * 60
MAX_PENDING_IMPORTS = 8
MAX_RETAINED_JOBS = 100
PLATFORM_DOMAINS = {
    "youtube": ("youtube.com", "youtu.be"),
    "facebook": ("facebook.com", "fb.watch"),
    "tiktok": ("tiktok.com",),
}

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video-import")
_jobs: dict[str, VideoImportStatus] = {}
_lock = threading.Lock()


class ImportValidationError(ValueError):
    pass


class ImportBusyError(RuntimeError):
    pass


class VideoImportError(RuntimeError):
    pass


def validate_source_url(value: str) -> tuple[str, str]:
    url = value.strip()
    if not url or len(url) > 2048 or any(character.isspace() for character in url):
        raise ImportValidationError("Vui lòng dán một liên kết video hợp lệ.")
    if "\\" in url or any(ord(character) < 32 for character in url):
        raise ImportValidationError("Liên kết video không hợp lệ.")
    try:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower()
        if (
            parsed.scheme not in {"https", "http"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443 if parsed.scheme == "https" else 80}
        ):
            raise ValueError
        query = parse_qs(parsed.query, max_num_fields=100)
    except ValueError:
        raise ImportValidationError("Vui lòng dùng liên kết http hoặc https hợp lệ.") from None

    platform = next(
        (
            name
            for name, domains in PLATFORM_DOMAINS.items()
            if any(hostname == domain or hostname.endswith(f".{domain}") for domain in domains)
        ),
        None,
    )
    if platform is None:
        raise ImportValidationError("Hiện hỗ trợ liên kết YouTube, Facebook và TikTok.")
    if "/playlist" in parsed.path.lower() or (platform == "youtube" and "list" in query):
        raise ImportValidationError("Hãy dán liên kết của một video, không phải danh sách phát.")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")), platform


def ensure_public_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise VideoImportError("Nguồn video sử dụng liên kết không được hỗ trợ.")
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        if not addresses or any(
            not ipaddress.ip_address(address[4][0]).is_global for address in addresses
        ):
            raise VideoImportError("Không thể nhập video từ địa chỉ mạng nội bộ.")
    except (ValueError, socket.gaierror):
        raise VideoImportError("Không kết nối được nền tảng video. Vui lòng thử lại.") from None


class PublicRequestGuard(urllib.request.BaseHandler):
    def __init__(self, deadline: float):
        self.deadline = deadline

    def http_request(self, request):
        remaining = _remaining_seconds(self.deadline)
        ensure_public_url(request.full_url)
        request.timeout = min(getattr(request, "timeout", 15), 15, remaining)
        return request

    https_request = http_request
    ftp_request = http_request


def get_import(import_id: str) -> VideoImportStatus | None:
    with _lock:
        job = _jobs.get(import_id)
        return job.model_copy(deep=True) if job else None


def _update(import_id: str, **changes) -> None:
    with _lock:
        _jobs[import_id] = _jobs[import_id].model_copy(update=changes)


def start_import(url: str) -> VideoImportStatus:
    storage.require_local_writes()
    source_url, platform = validate_source_url(url)
    with _lock:
        pending = sum(job.status in {"queued", "downloading"} for job in _jobs.values())
        if pending >= MAX_PENDING_IMPORTS:
            raise ImportBusyError("Hàng đợi nhập video đang đầy. Vui lòng thử lại sau.")
        for import_id in list(_jobs):
            if len(_jobs) < MAX_RETAINED_JOBS:
                break
            if _jobs[import_id].status in {"completed", "failed"}:
                del _jobs[import_id]
        job = VideoImportStatus(import_id=uuid.uuid4().hex, status="queued", progress=0)
        _jobs[job.import_id] = job
    try:
        _executor.submit(_run_import, job.import_id, source_url, platform)
    except RuntimeError:
        _update(job.import_id, status="failed", error="Dịch vụ nhập video đang khởi động lại.")
    return get_import(job.import_id)


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise VideoImportError("Nhập video quá thời gian 30 phút. Hãy thử video ngắn hơn.")
    return remaining


def _validate_info(info: dict) -> None:
    if info.get("_type") in {"playlist", "multi_video"} or "entries" in info:
        raise VideoImportError("Hãy dùng liên kết của một video, không phải kênh hay danh sách phát.")
    if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming", "post_live"}:
        raise VideoImportError("Chưa hỗ trợ livestream. Hãy nhập video đã phát hành đầy đủ.")
    if info.get("availability") in {"private", "premium_only", "subscriber_only", "needs_auth"}:
        raise VideoImportError("Video cần đăng nhập hoặc bị giới hạn. Hãy dùng liên kết công khai.")
    if (info.get("duration") or 0) > MAX_DURATION_SECONDS:
        raise VideoImportError("Video dài hơn giới hạn 2 giờ. Hãy chọn video ngắn hơn.")
    if (info.get("filesize") or info.get("filesize_approx") or 0) > MAX_FILE_BYTES:
        raise VideoImportError("Video vượt quá giới hạn 1 GiB. Hãy chọn video nhỏ hơn.")


def _browser_video(path: Path, deadline: float) -> Path:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,codec_type", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=min(30, _remaining_seconds(deadline)),
    )
    metadata = json.loads(result.stdout)
    video_streams = [stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "audio"]
    if not video_streams:
        raise VideoImportError("Liên kết không chứa video có thể phát.")
    duration = float(metadata.get("format", {}).get("duration") or 0)
    if duration > MAX_DURATION_SECONDS:
        raise VideoImportError("Video dài hơn giới hạn 2 giờ. Hãy chọn video ngắn hơn.")
    video_codec = video_streams[0].get("codec_name")
    audio_codec = audio_streams[0].get("codec_name") if audio_streams else None
    if path.suffix == ".mp4" and video_codec == "h264" and audio_codec in {None, "aac", "mp3"}:
        return path
    if path.suffix == ".webm" and video_codec in {"vp8", "vp9", "av1"} and audio_codec in {None, "opus", "vorbis"}:
        return path
    converted = path.with_name("playable.mp4")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
            "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
            "-movflags", "+faststart", "-fs", str(MAX_FILE_BYTES + 1), str(converted),
        ],
        capture_output=True,
        check=True,
        timeout=_remaining_seconds(deadline),
    )
    if converted.stat().st_size >= MAX_FILE_BYTES:
        raise VideoImportError("Video sau chuyển đổi vượt quá giới hạn 1 GiB.")
    return converted


def _download_video(source_url: str, directory: Path, import_id: str, deadline: float) -> tuple[Path, dict]:
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.networking._urllib import UrllibRH
    except ImportError:
        raise VideoImportError("Máy chủ chưa cài yt-dlp. Hãy chạy pip install -r requirements.txt.") from None

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise VideoImportError("Máy chủ cần ffmpeg và ffprobe trong PATH để nhập video từ liên kết.")

    def progress_hook(event):
        _remaining_seconds(deadline)
        downloaded = event.get("downloaded_bytes") or 0
        total = event.get("total_bytes") or event.get("total_bytes_estimate")
        actual_bytes = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
        if max(downloaded, total or 0, actual_bytes) > MAX_FILE_BYTES:
            raise VideoImportError("Video vượt quá giới hạn 1 GiB. Hãy chọn video nhỏ hơn.")
        if event.get("status") == "downloading":
            percentage = min(95, round(downloaded / total * 95, 1)) if total else None
            current = get_import(import_id)
            if percentage is not None and current and current.progress is not None:
                percentage = max(percentage, current.progress)
            _update(import_id, progress=percentage)

    def match_filter(info, *, incomplete=False):
        _remaining_seconds(deadline)
        _validate_info(info)
        return None

    class PublicUrllibRH(UrllibRH):
        RH_KEY = "Urllib"
        _SUPPORTED_URL_SCHEMES = ("http", "https")

        def _create_instance(self, proxies, cookiejar, legacy_ssl_support=None):
            opener = super()._create_instance(proxies, cookiejar, legacy_ssl_support)
            opener.add_handler(PublicRequestGuard(deadline))
            return opener

    class PublicYoutubeDL(YoutubeDL):
        def build_request_director(self, handlers, preferences=None):
            return super().build_request_director([PublicUrllibRH])

    options = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "outtmpl": str(directory / "download.%(ext)s"),
        "format": "bestvideo[height<=1080][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best[ext=webm]",
        "merge_output_format": "mp4/webm",
        "noplaylist": True,
        "playlistend": 1,
        "extract_flat": "in_playlist",
        "allowed_extractors": ["youtube.*", "facebook.*", "tiktok.*"],
        "match_filter": match_filter,
        "max_filesize": MAX_FILE_BYTES,
        "socket_timeout": 15,
        "retries": 2,
        "fragment_retries": 2,
        "extractor_retries": 2,
        "skip_unavailable_fragments": False,
        "hls_prefer_native": True,
        "concurrent_fragment_downloads": 1,
        "cachedir": False,
        "proxy": "",
        "js_runtimes": {"node": {}, "deno": {}},
        "progress_hooks": [progress_hook],
        "postprocessor_hooks": [lambda event: _remaining_seconds(deadline)],
    }
    ensure_public_url(source_url)
    with PublicYoutubeDL(options) as downloader:
        info = downloader.extract_info(source_url, download=False)
        if not info:
            raise VideoImportError("Không tìm thấy video công khai tại liên kết này.")
        _validate_info(info)
        _remaining_seconds(deadline)
        downloader.process_ie_result(info, download=True)
    _remaining_seconds(deadline)
    downloaded = next((directory / f"download{ext}" for ext in (".mp4", ".webm") if (directory / f"download{ext}").is_file()), None)
    if not downloaded:
        raise VideoImportError("Không tải được video MP4/WebM. Video có thể quá lớn hoặc bị nền tảng giới hạn.")
    if downloaded.stat().st_size == 0 or downloaded.stat().st_size > MAX_FILE_BYTES:
        raise VideoImportError("Video trống hoặc vượt quá giới hạn 1 GiB.")
    _update(import_id, progress=97)
    return _browser_video(downloaded, deadline), info


def _friendly_error(error: Exception) -> str:
    if isinstance(error, VideoImportError):
        return str(error)
    message = str(error).lower()
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) or "timed out" in message:
        return "Kết nối hoặc xử lý video quá thời gian. Vui lòng thử lại với video ngắn hơn."
    if any(word in message for word in ("private", "login", "sign in", "sign-in", "cookies", "restricted", "403", "429")):
        return "Nền tảng yêu cầu đăng nhập hoặc đang chặn lượt tải. Hãy dùng video công khai khác hoặc tải tệp từ máy."
    if "unavailable" in message or "not available" in message or "404" in message:
        return "Video không còn tồn tại hoặc không khả dụng tại khu vực này."
    if "unsupported url" in message:
        return "Không nhận diện được video. Hãy dùng liên kết trực tiếp hoặc liên kết chia sẻ của video."
    if "requested format" in message:
        return "Video chưa có định dạng có thể nhập. Hãy thử video khác hoặc tải tệp từ máy."
    if isinstance(error, (subprocess.SubprocessError, json.JSONDecodeError)):
        return "Không xử lý được định dạng video. Hãy kiểm tra ffmpeg hoặc thử video khác."
    return "Không thể nhập video lúc này. Kiểm tra liên kết công khai, thử lại hoặc tải tệp từ máy."


def _run_import(import_id: str, source_url: str, platform: str) -> None:
    _update(import_id, status="downloading", progress=None)
    destination = None
    metadata_path = None
    video_id = storage.new_video_id()
    try:
        storage.require_local_writes()
        imports_directory = storage.contained_path(storage.DATA_DIR, "imports")
        metadata_path = storage.video_meta_path(video_id)
        imports_directory.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + MAX_IMPORT_SECONDS
        with tempfile.TemporaryDirectory(prefix=f"{import_id}-", dir=imports_directory) as temporary:
            downloaded, info = _download_video(source_url, Path(temporary), import_id, deadline)
            _remaining_seconds(deadline)
            destination = storage.video_path(video_id, downloaded.suffix)
            title = str(info.get("title") or "Video đã nhập").strip()[:180]
            filename = f"{title}{downloaded.suffix}"
            storage.save_video_meta(
                video_id, filename, downloaded, source_url=source_url, source_platform=platform,
            )
            downloaded.replace(destination)
            metadata = storage.load_video_meta(video_id) or {}
            video = VideoInfo(
                video_id=video_id,
                filename=filename,
                has_transcript=False,
                duration_seconds=metadata.get("duration_seconds"),
                uploaded_at=metadata.get("uploaded_at"),
                source_url=source_url,
                source_platform=platform,
            )
        _update(import_id, status="completed", progress=100, video=video)
    except Exception as error:
        logger.warning("Video import %s failed: %s", import_id, type(error).__name__)
        if destination is not None:
            destination.unlink(missing_ok=True)
        if metadata_path is not None:
            metadata_path.unlink(missing_ok=True)
        _update(import_id, status="failed", progress=None, error=_friendly_error(error))
