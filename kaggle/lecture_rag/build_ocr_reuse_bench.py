"""Generate ocr_reuse_bench.ipynb: stage 02 of one L25 video, four times, to measure two speed-ups.

Input: the stage-01 output of L25_V017 from kernel l25-all-0-20261007 (21 slides, 16.5 min of OCR
in that run). Every variant starts with an empty cache and uses the same models and OCR gate:
  base    current defaults
  reuse   + reuse_near_duplicates (a keyframe almost identical to one already read reuses its OCR)
  skip    + hybrid_min_region_area 0.02 and hybrid_skip_repeated_decor (icons, logos, the
          lecturer's photo do not go to Qwen3-VL)
  both    reuse + skip
Reported: seconds of stage 02, VLM calls, slides reused, and text similarity of each segment to
base (difflib on OCR text + region text). A variant is worth turning on if it is clearly faster
and the mean similarity stays >= 0.98.

    python kaggle/lecture_rag/build_ocr_reuse_bench.py
"""

from pathlib import Path
import json

import build_fast_pipeline as fast
from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
NAME = "ocr_reuse_bench"
DATE = "20261008"
OWNER = "thnhtrungnguynmif"
SOURCE = "l25-all-0-20261007"

INTRO = """# Tăng tốc OCR · dùng lại slide gần trùng, bỏ vùng hình nhỏ/logo

Bước 02 của **L25_V017** chạy 4 lần, cache trống, cùng model và bộ gác:
**base** (hiện tại), **reuse** (dùng lại slide gần trùng), **skip** (không gửi Qwen vùng < 2% slide
và logo/ảnh giảng viên lặp lại), **both**. So thời gian, số lần gọi Qwen, và chữ giống base."""

CONFIG = """import difflib, json, statistics, time
from pathlib import Path
VIDEO_ID = 'L25_V017'
STAGE01 = next(p for p in Path('/kaggle/input').rglob('01_slides') if p.parent.name == VIDEO_ID)
DEVICES = ('cuda:0', 'cuda:1')
ocr_gate = load_ocr_gate('cuda:0', threshold=0.15, views='frame+crops')
VARIANTS = {
    'base': {},
    'reuse': {'reuse_near_duplicates': True},
    'skip': {'hybrid_min_region_area': 0.02, 'hybrid_skip_repeated_decor': True},
    'both': {'reuse_near_duplicates': True, 'hybrid_min_region_area': 0.02, 'hybrid_skip_repeated_decor': True},
}
print(STAGE01)
"""

RUN = """def segment_text(record):
    return record['ocr_text'] + '\\n' + '\\n'.join(r.get('text', '') + ' ' + r.get('description', '') for r in record['regions'])

def vlm_calls(records):
    called = skipped = 0
    for r in records:
        crops = json.loads(r['raw_ocr_response'] or '{}').get('crops', []) if r.get('raw_ocr_response') else []
        if r.get('ocr_reused_from'):
            continue
        called += sum(not c.get('skipped') for c in crops)
        skipped += sum(bool(c.get('skipped')) for c in crops)
    return called, skipped

results = {}
for name, options in VARIANTS.items():
    started = time.monotonic()
    _, _, records = run_visual_extraction(
        STAGE01, Path('/kaggle/working/bench') / name, VIDEO_ID, languages=('vi', 'en'),
        ocr_device='cuda', ocr_engine='hybrid', hybrid_devices=DEVICES, ocr_gate=ocr_gate, **options)
    called, skipped = vlm_calls(records)
    results[name] = {'sec': round(time.monotonic() - started, 1), 'records': records,
                     'read': sum(r['ocr_status'] == 'read' and not r.get('ocr_reused_from') for r in records),
                     'reused': sum(bool(r.get('ocr_reused_from')) for r in records),
                     'gate_skipped': sum(r['ocr_status'] == 'skipped_no_content' for r in records),
                     'vlm_called': called, 'vlm_skipped': skipped}
    print(name, {k: v for k, v in results[name].items() if k != 'records'}, flush=True)
"""

SCORE = """def similar(a, b):
    return difflib.SequenceMatcher(None, ' '.join(a.split()), ' '.join(b.split())).ratio()

base = {r['segment_id']: segment_text(r) for r in results['base']['records']}
summary, worst = [], {}
for name, result in results.items():
    sims = {r['segment_id']: similar(base[r['segment_id']], segment_text(r)) for r in result['records']
            if r['ocr_status'] == 'read'}
    worst[name] = sorted(sims.items(), key=lambda kv: kv[1])[:3]
    summary.append({'variant': name, 'stage02_sec': result['sec'],
                    'speedup': round(results['base']['sec'] / result['sec'], 2),
                    'ocr_read': result['read'], 'reused': result['reused'], 'gate_skipped': result['gate_skipped'],
                    'vlm_called': result['vlm_called'], 'vlm_skipped': result['vlm_skipped'],
                    'similar_mean': round(statistics.mean(sims.values()), 4), 'similar_min': round(min(sims.values()), 3)})
reused = [(r['segment_id'], r['ocr_reused_from']) for r in results['both']['records'] if r.get('ocr_reused_from')]
print('dùng lại:', reused)
print('giống base thấp nhất:', worst)
json.dump({'summary': summary, 'worst': worst, 'reused': reused,
           'texts': {name: {r['segment_id']: segment_text(r) for r in res['records']} for name, res in results.items()}},
          open('/kaggle/working/ocr_reuse_bench.json', 'w'), ensure_ascii=False, indent=1)
import pandas as pd
display(pd.DataFrame(summary))
"""


def build():
    cells = [cell("markdown", INTRO), cell("code", "%pip install -q " + fast.DEPS)]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in fast.MODULES]
    cells += [cell("markdown", "## Cấu hình"), cell("code", CONFIG),
              cell("markdown", "## Chạy 4 bản"), cell("code", RUN),
              cell("markdown", "## So sánh"), cell("code", SCORE)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    (ROOT / "notebooks" / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    run = ROOT / "runs" / f"{DATE}-ocr-reuse-bench"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    json.dump({"id": f"{OWNER}/l25-ocr-reuse-bench-{DATE}", "title": f"l25-ocr-reuse-bench-{DATE}",
               "code_file": f"{NAME}.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
               "competition_sources": [], "kernel_sources": [f"{OWNER}/{SOURCE}"]},
              open(run / "kernel-metadata.json", "w"), indent=2)
    print(run / f"{NAME}.ipynb")


if __name__ == "__main__":
    build()
