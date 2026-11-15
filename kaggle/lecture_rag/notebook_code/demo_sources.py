"""Model-free source resolution for the demo; production keeps its original input API."""
from pathlib import Path
import re
import unicodedata

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.webm'}


def safe_video_id(stem):
    text = unicodedata.normalize('NFKD', stem.replace('đ', 'd').replace('Đ', 'D'))
    text = text.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^A-Za-z0-9_-]+', '_', text).strip('_-')[:80] or 'video'


def video_source_error(message, input_root):
    dataset_root = Path(input_root) / 'datasets'
    entries = []
    if dataset_root.is_dir():
        for first in sorted(dataset_root.iterdir()):
            if first.is_dir() or first.suffix.lower() in VIDEO_EXTENSIONS:
                entries.append(str(first))
            if first.is_dir():
                entries.extend(str(p) for p in sorted(first.iterdir())
                               if p.is_dir() or p.suffix.lower() in VIDEO_EXTENSIONS)
            if len(entries) > 40:
                break
    listing = '\n'.join(entries[:40]) or '(chưa có dataset/video ở đây)'
    if len(entries) > 40:
        listing += '\n… (chỉ hiện 40 đường dẫn đầu)'
    return FileNotFoundError(f'{message}\nĐường dẫn hiện có dưới {dataset_root} (2 cấp):\n{listing}'
                             '\nHãy Add Input, copy đường dẫn rồi dán vào VIDEO_SOURCES.')


def resolve_video_sources(video_sources=None, video_ids=(), languages=None, input_root='/kaggle/input'):
    """Nonempty VIDEO_SOURCES replaces VIDEO_IDS; repeated files are processed once."""
    languages = languages or {}
    sources = list(video_sources or video_ids)
    if not sources:
        raise video_source_error('VIDEO_SOURCES / VIDEO_IDS đang rỗng.', input_root)
    resolved, seen = {}, set()
    for source in sources:
        raw = str(source).strip()
        path = Path(raw).expanduser()
        is_path = path.is_absolute() or '/' in raw or '\\' in raw or path.suffix.lower() in VIDEO_EXTENSIONS
        if not raw:
            raise video_source_error('Nguồn video đang rỗng.', input_root)
        if path.is_file():
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise video_source_error(f'Tệp không phải video hỗ trợ: {path}', input_root)
            files = [path]
        elif path.is_dir():
            files = sorted((p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
                           key=lambda p: p.as_posix())
            if not files:
                raise video_source_error(f'Thư mục không có video (.mp4/.mkv/.mov/.webm): {path}', input_root)
        elif is_path:
            raise video_source_error(f'Không tồn tại đường dẫn video: {path}', input_root)
        else:
            files = sorted(p for p in Path(input_root).rglob(raw + '.mp4') if p.is_file())
            if len(files) != 1:
                raise video_source_error(f'Cần đúng một {raw}.mp4, tìm thấy {len(files)}. '
                                         'Nếu trùng tên, hãy dán đường dẫn đầy đủ.', input_root)
        for file in files:
            file = file.resolve()
            if file in seen:
                continue
            seen.add(file)
            base, number = safe_video_id(file.stem), 2
            video_id = base
            while video_id in resolved:
                video_id = f'{base}__{number}'
                number += 1
            language = languages.get(video_id, languages.get(str(file), languages.get(file.stem,
                                     languages.get(raw, 'vi'))))
            if language not in {'vi', 'en'}:
                raise ValueError(f'Ngôn ngữ không hỗ trợ cho {video_id}: {language}. Dùng vi hoặc en.')
            resolved[video_id] = {'path': file, 'language': language}
    return resolved


def find_video_file(video_id):
    """All demo stages use the same resolved file, including source checksum validation."""
    return RESOLVED_VIDEOS[video_id]['path']


def demo_validate_video_output(root, video_id):
    result = validate_video_output(root, video_id)
    if result['source_sha256'] != file_sha256(find_video_file(video_id)):
        raise ValueError(f'Checksum video nguồn không khớp: {video_id} → {find_video_file(video_id)}')
    return result
