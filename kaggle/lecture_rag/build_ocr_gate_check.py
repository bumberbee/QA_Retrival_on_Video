"""Writes notebooks/ocr_gate_check.ipynb: picks the OCR gate threshold on real keyframes.

Input on Kaggle: dataset l25-ocr-gate-keyframes (217 keyframes of 7 videos, each with the
length of the text that OCR found on it). A keyframe with < 80 characters has no readable
content. The gate (adfilter.load_ocr_gate, SigLIP zero-shot) must skip as many of those as
possible without skipping any slide that has text.

    python kaggle/lecture_rag/build_ocr_gate_check.py
"""

import json
from pathlib import Path

from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent

INTRO = """# Bộ gác OCR · chọn ngưỡng trên keyframe thật

217 keyframe của 7 video L25 (đã OCR ở lần chạy 20261002). Khung có < 80 ký tự chữ là
**không có nội dung đọc được** (giảng viên nói, logo mở đầu). Bộ gác phải bỏ được nhiều khung
loại này nhất mà **không bỏ nhầm slide nào có chữ**."""

RUN = """import csv, glob
from pathlib import Path
from PIL import Image
base = Path(glob.glob('/kaggle/input/**/keyframes.csv', recursive=True)[0]).parent
rows = list(csv.DictReader(open(base / 'keyframes.csv', encoding='utf-8')))
images = [Image.open(base / r['image']).convert('RGB') for r in rows]
empty = [int(r['ocr_chars']) < 80 for r in rows]
print(len(rows), 'keyframe ·', sum(empty), 'không có nội dung')
results = {}
for model_id in ('google/siglip-base-patch16-224', 'google/siglip-so400m-patch14-384'):
    check, _ = load_ocr_gate('cuda', model_id)
    scores = []
    for i in range(0, len(images), 32):
        scores += check(images[i:i + 32])
    results[model_id] = scores
    print('\\n' + model_id)
    for th in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        skip = [s < th for s in scores]
        good = sum(s and e for s, e in zip(skip, empty))
        bad = sum(s and not e for s, e in zip(skip, empty))
        print(f'  ngưỡng {th:.2f}: bỏ đúng {good}/{sum(empty)} · bỏ nhầm slide có chữ {bad}/{len(rows) - sum(empty)}')
    content = sorted((s, r['image'], r['ocr_chars']) for s, r, e in zip(scores, rows, empty) if not e)
    print('  slide có chữ điểm thấp nhất:', [(round(s, 3), n, c) for s, n, c in content[:6]])
    print('  khung không nội dung điểm cao nhất:',
          sorted(((round(s, 3), r['image'], r['content_kind']) for s, r, e in zip(scores, rows, empty) if e), reverse=True)[:6])
import json
json.dump({'rows': rows, 'scores': results}, open('/kaggle/working/ocr_gate_scores.json', 'w'), ensure_ascii=False)
"""


def build():
    cells = [cell("markdown", INTRO)]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in ("common.py", "adfilter.py")]
    cells += [cell("code", RUN)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "notebooks" / "ocr_gate_check.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target.name)


if __name__ == "__main__":
    build()
