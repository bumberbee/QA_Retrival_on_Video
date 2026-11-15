"""Thêm kết quả của fast_pipeline_l25.ipynb vào app, thành một bản gộp slide để so sánh.

Với mỗi video (thư mục L25_Vxxx trong fast_pipeline_export):
  - chép ảnh trạng thái (1280px) vào data/keyframes/<id>/fast/;
  - thêm bản "F_fast_pipeline" vào data/slide_variants/<id>.json, giữ các bản cũ;
  - nếu video chưa có transcript thì dùng 04_bundle/transcript_app.json;
  - đoạn quảng cáo (content_kind "ad", hoặc ad_flagged khi bộ lọc chỉ đánh dấu) không thành slide: app ẩn chúng;
  - chép chỉ mục bước 05 vào data/index/<id>/: hỏi đáp tìm bằng chứng theo slide thay vì gửi cả transcript.

Cách dùng:
    cd backend
    .venv/Scripts/python scripts/import_fast_pipeline.py \
        --export ../pipeline/runs/<run>/outputs/export \
        --videos path/to/mp4s   # chỉ cần cho video chưa có trong app
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.features.extract.importer import import_video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--videos", type=Path, help="Thư mục chứa L25_Vxxx.mp4 cho video chưa có trong app")
    args = parser.parse_args()
    # one folder per video (L25_V008, MLS-15DA1, ...), as written by the pipeline notebooks
    for folder in sorted(p for p in args.export.iterdir() if (p / "01_slides").is_dir()):
        import_video(folder, args.videos)


if __name__ == "__main__":
    main()
