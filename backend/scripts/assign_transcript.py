"""Gán 1 file transcript (xuất từ Kaggle, đặt tên theo video gốc, vd. L25_V001.json)
vào đúng video_id đã upload lên app, dựa theo tên file gốc đã upload.

Cách dùng:
    cd backend
    .venv/Scripts/python scripts/assign_transcript.py \
        --transcript path/to/L25_V001.json \
        --original-filename L25_V001.mp4

Nếu không truyền --original-filename, script sẽ dùng đúng field "video_id" có sẵn
trong file JSON (vd. "L25_V001") làm tên file gốc để tìm (giả định bạn upload video
với tên gốc y hệt, vd. "L25_V001.mp4").
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import storage  # noqa: E402
from app.models import Transcript  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, help="Đường dẫn file transcript JSON")
    parser.add_argument(
        "--original-filename",
        help="Tên file gốc lúc upload (vd. L25_V001.mp4). Mặc định suy ra từ video_id trong JSON + .mp4",
    )
    args = parser.parse_args()

    raw = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
    source_video_id = raw["video_id"]

    original_filename = args.original_filename or f"{source_video_id}.mp4"
    video_id = storage.find_video_id_by_original_filename(original_filename)

    if not video_id:
        uploaded = [v.filename for v in storage.list_videos()]
        print(f"Không tìm thấy video đã upload với tên gốc '{original_filename}'.")
        print(f"Các video đã upload: {uploaded or '(chưa có video nào)'}")
        print("-> Upload video này lên app trước (qua giao diện web), rồi chạy lại script.")
        raise SystemExit(1)

    raw["video_id"] = video_id
    transcript = Transcript(**raw)
    storage.save_transcript(transcript)

    print(f"OK: đã gán transcript ({len(transcript.segments)} đoạn) cho video_id={video_id}")
    print(f"File lưu tại: {storage.transcript_path(video_id)}")


if __name__ == "__main__":
    main()
