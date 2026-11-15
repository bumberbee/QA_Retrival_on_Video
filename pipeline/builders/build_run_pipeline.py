"""Generate run_pipeline.ipynb: the current pipeline (01, OCR gate, 02 OCR with Qwen3-VL, 03 ASR,
04 sync, 05 index, retrieval) on ONE video chosen by name, printing the sample output of every
stage. Same stage cells as full_pipeline_walkthrough (its V008-only answer-key check is left out);
the user only edits VIDEO_NAME and presses Run all on Kaggle.

The name is the file name without extension (L25_V008, D_kuulxKuxc) or a unique part of it,
searched under /kaggle/input, so any attached video dataset works. Run: python build_run_pipeline.py
"""

from pathlib import Path
import json

import build_full_walkthrough as walk
from build_fast_pipeline import DEPS
from build_notebooks import module_path, cell

ROOT = Path(__file__).resolve().parents[1]
NAME = "run_pipeline"
DATE = "20261005"
OWNER = "thnhtrungnguynmif"

INTRO = """# Chạy pipeline cho 1 video · in kết quả mẫu từng bước

**Cách dùng**
1. **Add Input** dataset chứa video (mp4/mkv/webm/mov). Đã gắn sẵn `l25a-lecture-videos-media`
   `vn-lecture-test12` và `m3av-maths` (toán tiếng Anh); video mới: tải lên thành dataset rồi Add Input.
2. Lọc nội dung chỉ dùng bộ gác OCR zero-shot (SigLIP), không cần model học riêng.
3. Bật **GPU T4 x2** và **Internet**.
4. Sửa `VIDEO_NAME`, `LANGUAGE` (và nếu muốn, `QUESTIONS`) ở ô **Cấu hình** → **Run all**.

| Bước | In ra |
|---|---|
| 0 · Video | thông tin file, 6 khung hình mẫu |
| 1 · Tách slide (01) | chỉ số, tham số, bảng mọi đoạn kèm ảnh, slide hiện dần đã gộp |
| 2 · Lọc quảng cáo | tắt mặc định: bộ gác OCR ở bước 4 bỏ quảng cáo và logo |
| 3 · Giọng nói (03) | cửa sổ ASR, 60 từ đầu kèm thời gian và xác suất |
| 4 · Bộ gác OCR + OCR (02) | bảng từng slide, giây/slide, chữ và mô tả hình Qwen3-VL đọc, đầu ra thô |
| 5 · Đồng bộ (04) | lời giảng gán vào slide, dòng lặp bị bỏ, từ ASR được sửa, 1 slide đầy đủ |
| 6 · Chỉ mục (05) | số đoạn theo nguồn, 30 đoạn đầu |
| 7 · Hỏi thử | 5 bằng chứng đầu cho mỗi câu hỏi |
| 8 · Tổng kết | thời gian từng bước, tổng 01→05, bộ nhớ GPU |

Mọi thứ cũng được lưu vào `walkthrough_report.html` và `walkthrough_export.zip` (tab Output).
Thời gian: ước tính 8–12 phút cho 30 phút video với OCR Qwen3-VL, cộng 5–10 phút cài đặt và tải model."""

HEAD = """# ===== SỬA Ở ĐÂY =====
VIDEO_NAME = 'MLS-15DA1'  # tên file video không có đuôi (hoặc một phần duy nhất của tên)
LANGUAGE = 'en'            # ngôn ngữ lời giảng (ASR, mô tả hình của Qwen): 'vi' (L25), 'en' (M3AV, LPM, SlideSpeech)
# Câu hỏi thử ở bước 7: (câu hỏi, None) hoặc (câu hỏi, 'đang xem'). [] = tự lấy từ tiêu đề slide.
QUESTIONS = [('What is an eigenvector?', None),
             ('How do you diagonalize a matrix?', None),
             ('What does a diagonal matrix do geometrically?', None),
             ('Explain this slide.', 'đang xem'),
             ('How do I install PyTorch?', None)]
# ======================
"""

FIND = """VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.webm', '.mov')
_all_videos = sorted(p for p in Path('/kaggle/input').rglob('*') if p.suffix.lower() in VIDEO_EXTENSIONS)
_matches = ([p for p in _all_videos if p.stem == VIDEO_NAME]
            or [p for p in _all_videos if VIDEO_NAME.lower() in p.stem.lower()])
if len({p.stem for p in _matches}) != 1:
    raise FileNotFoundError(f'"{VIDEO_NAME}": {len(_matches)} video khớp. Có trong Input: '
                            + ', '.join(p.stem for p in _all_videos[:40]))
VIDEO, VIDEO_ID = _matches[0], _matches[0].stem
print('Video:', VIDEO)
"""

AUTO_QUESTIONS = """if not QUESTIONS:
    _titles = [s['visual'].get('title_clean') or s['visual'].get('title') or '' for s in segments
               if s['content_kind'] != 'ad']
    _titles = [t for t in dict.fromkeys(' '.join(t.split()) for t in _titles) if len(t) >= 8]
    QUESTIONS = [(f'{t} là gì?', None) for t in _titles[:3]] + [('Giải thích slide này.', 'đang xem'),
                                                                 ('Cách cài đặt PyTorch?', None)]
"""


def config():
    lines = walk.CONFIG.split("# 7 · câu hỏi thử")[0].splitlines(keepends=True)
    body = "".join(line for line in lines if not line.startswith("VIDEO_ID ="))
    body = body.replace("DESCRIBE_LANGUAGE = 'vi'", "DESCRIBE_LANGUAGE = LANGUAGE")
    assert "DESCRIBE_LANGUAGE = LANGUAGE" in body
    return HEAD + body


def build():
    stage03 = walk.STAGE03.replace("language='vi'", "language=LANGUAGE")
    assert stage03 != walk.STAGE03
    video = walk.VIDEO.replace("VIDEO = next(iter(sorted(Path('/kaggle/input').rglob(VIDEO_ID + '.mp4'))))\n", "")
    assert video != walk.VIDEO
    cells = [cell("markdown", INTRO), cell("code", "%pip install -q " + DEPS + ' "matplotlib"')]
    cells += [cell("code", module_path(name).read_text(encoding="utf-8"), hidden=True) for name in walk.MODULES]
    cells += [cell("markdown", "## Cấu hình"), cell("code", config()),
              cell("code", walk.HELPERS, hidden=True), cell("code", FIND)]
    for title, source in [("0 · Video", video), ("Nạp model", walk.MODELS), ("1 · Bước 01 · Tách slide", walk.STAGE01),
                          ("2 · Lọc quảng cáo", walk.ADS), ("3 · Bước 03 · Giọng nói", stage03),
                          ("4 · Bộ gác OCR + bước 02 · OCR", walk.STAGE02), ("5 · Bước 04 · Đồng bộ", walk.STAGE04),
                          ("6 · Bước 05 · Chỉ mục", walk.STAGE05), ("7 · Hỏi thử", AUTO_QUESTIONS + walk.RETRIEVE),
                          ("8 · Tổng kết", walk.SUMMARY)]:
        cells += [cell("markdown", "## " + title), cell("code", source)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    (ROOT / "notebooks" / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    run = ROOT / "runs" / f"{DATE}-run-pipeline"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    json.dump({"id": f"{OWNER}/l25-run-pipeline", "title": "l25-run-pipeline",
               "code_file": f"{NAME}.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [f"{OWNER}/l25a-lecture-videos-media", f"{OWNER}/vn-lecture-test12",
                                   f"{OWNER}/m3av-maths"],
               "competition_sources": [], "kernel_sources": []},
              open(run / "kernel-metadata.json", "w"), indent=2)
    print(run / f"{NAME}.ipynb")


if __name__ == "__main__":
    build()
