"""Writes notebooks/ad_detection_run.ipynb: stage 01 with the ad filter on the gold-set videos,
to score slide boundaries and ad cuts against gold set 1 (gold/eval_ad_detection.py).

Inputs on Kaggle: dataset l25a-lecture-videos-media and the output of the ad filter
experiments kernel (ad_student.npz). Only stage 01 runs, so no OCR or ASR.

    python kaggle/lecture_rag/build_ad_detection_run.py
"""

from pathlib import Path

from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
MODULES = ["common.py", "detection.py", "adfilter.py"]

INTRO = """# Bước 01 có lọc quảng cáo · 10 video của bộ gold 1

Input: dataset **l25a-lecture-videos-media** và output của kernel thí nghiệm bộ lọc
(**ad_student.npz**). Chỉ chạy bước 01 (cùng tham số với pipeline nhanh) để chấm lại ranh giới
slide và các đoạn quảng cáo bị cắt bằng `gold/eval_ad_detection.py`."""

CONFIG = """from pathlib import Path
VIDEO_IDS = [f'L25_V{n:03d}' for n in range(1, 11)]
OUTPUT_ROOT = Path('/kaggle/working/lecture_rag')
SAMPLE_HZ, FINE_HZ, MERGE_REVEALS = 1.0, 8.0, True
AD_FILTER, AD_DEVICE = 'student', 'cuda'
"""

RUN = """import json, shutil, time
student = next(iter(sorted(Path('/kaggle/input').rglob('ad_student.npz'))))
ad_scorer = load_ad_scorer(AD_FILTER, AD_DEVICE, student)
print('Lọc quảng cáo:', ad_scorer[1])
summary = []
for video_id in VIDEO_IDS:
    video = next(iter(sorted(Path('/kaggle/input').rglob(video_id + '.mp4'))))
    started = time.monotonic()
    stage01, manifest, intervals = run_slide_detection(
        video, OUTPUT_ROOT / video_id, video_id, sample_hz=SAMPLE_HZ, fine_hz=FINE_HZ,
        merge_reveals=MERGE_REVEALS, scan='single_pass', ad_scorer=ad_scorer)
    report = json.loads((stage01 / 'ad_filter.json').read_text(encoding='utf-8'))
    row = {'video_id': video_id, 'sec': round(time.monotonic() - started, 1),
           'segments': len(intervals), 'ad_segments': manifest['ad_segment_count'],
           'ad_seconds': report['ad_seconds'], 'ad_intervals': report['ad_intervals']}
    summary.append(row)
    print('AD_RUN', json.dumps(row), flush=True)
# Gói nhẹ: intervals, báo cáo bộ lọc và manifest của từng video (không kèm ảnh).
export = Path('/kaggle/working/ad_detection_export')
for video_id in VIDEO_IDS:
    target = export / video_id
    target.mkdir(parents=True, exist_ok=True)
    for name in ('intervals.json', 'ad_filter.json', 'manifest.json'):
        shutil.copy(OUTPUT_ROOT / video_id / '01_slides' / name, target / name)
(export / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
shutil.make_archive('/kaggle/working/ad_detection_export', 'zip', export)
shutil.rmtree(OUTPUT_ROOT)
"""


def build():
    cells = [cell("markdown", INTRO),
             cell("code", '%pip install -q "av>=14,<17" "opencv-python-headless>=4.10,<4.12" "Pillow>=10"')]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in MODULES]
    cells += [cell("code", CONFIG), cell("code", RUN)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "notebooks" / "ad_detection_run.ipynb"
    import json
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target.name)


if __name__ == "__main__":
    build()
