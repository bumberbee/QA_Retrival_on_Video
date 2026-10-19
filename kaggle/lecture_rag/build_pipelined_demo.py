"""Generate pipelined_demo.ipynb: stage 02 of one short L25 video, before and after the speed-up.

  before  current defaults
  after   hybrid_pipelined (Unlimited-OCR on GPU 0 reads the next slide while Qwen3-VL on GPU 1
          reads this slide's regions in one batch) + reuse_near_duplicates + small/repeated
          figure regions not sent to Qwen3-VL
Input: the stage-01 output of L25_V011 (15 min) from kernel l25-all-0-20261007. Output:
pipelined_demo.json with the seconds and each segment's OCR + region text for both runs.

An English run (--en) first detects the slides of SlideSpeech cs_0001 (stage 01), then runs
the same two variants with English region prompts, and adds small keyframes to the JSON.

    python kaggle/lecture_rag/build_pipelined_demo.py [--en]
"""

import sys

from pathlib import Path
import json

import build_fast_pipeline as fast
from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
NAME = "pipelined_demo"
DATE = "20261012"
OWNER = "thnhtrungnguynmif"
SOURCE = "l25-all-0-20261007"

INTRO = """# Demo tăng tốc bước 02 · L25_V011

**before**: OCR hiện tại. **after**: Unlimited-OCR (GPU 0) đọc slide sau trong lúc Qwen3-VL
(GPU 1) đọc các vùng hình của slide trước theo lô, dùng lại slide gần trùng, bỏ vùng hình nhỏ/logo."""

RUN = """import json, time
from pathlib import Path
VIDEO_ID = 'L25_V011'
STAGE01 = next(p for p in Path('/kaggle/input').rglob('01_slides') if p.parent.name == VIDEO_ID)
DEVICES = ('cuda:0', 'cuda:1')
ocr_gate = load_ocr_gate('cuda:0', threshold=0.15, views='frame+crops')
shared_model(('hybrid_ocr', DEVICES), lambda: load_hybrid_ocr(*DEVICES))   # loaded before timing
VARIANTS = {'before': {},
            'after': {'hybrid_pipelined': True, 'reuse_near_duplicates': True,
                      'hybrid_min_region_area': 0.02, 'hybrid_skip_repeated_decor': True}}
demo = {}
for name, options in VARIANTS.items():
    started = time.monotonic()
    _, _, records = run_visual_extraction(
        STAGE01, Path('/kaggle/working/demo') / name, VIDEO_ID, languages=('vi', 'en'),
        ocr_device='cuda', ocr_engine='hybrid', hybrid_devices=DEVICES, ocr_gate=ocr_gate, **options)
    demo[name] = {'sec': round(time.monotonic() - started, 1), 'segments': [
        {'segment_id': r['segment_id'], 'status': r['ocr_status'], 'reused_from': r.get('ocr_reused_from'),
         'ocr_text': r['ocr_text'],
         'regions': [{'kind': g['kind'], 'text': g.get('text', ''), 'description': g.get('description', '')}
                     for g in r['regions']]} for r in records]}
    print(name, demo[name]['sec'], 's', flush=True)
json.dump(demo, open('/kaggle/working/pipelined_demo.json', 'w'), ensure_ascii=False, indent=1)
print('nhanh hơn ×', round(demo['before']['sec'] / demo['after']['sec'], 2))
"""


RUN_EN = """import base64, io, json, time
from pathlib import Path
from PIL import Image
VIDEO_ID = 'cs_0001'
VIDEO = next(p for p in Path('/kaggle/input').rglob('*.mp4') if p.stem == VIDEO_ID)
DEVICES = ('cuda:0', 'cuda:1')
started = time.monotonic()
STAGE01, manifest, intervals = run_slide_detection(VIDEO, Path('/kaggle/working/demo/stage01'), VIDEO_ID,
                                                   sample_hz=1.0, fine_hz=8.0, merge_reveals=True, scan='single_pass')
stage01_sec = round(time.monotonic() - started, 1)
print('01:', len(intervals), 'đoạn,', stage01_sec, 's, video', round(manifest['source']['duration_sec'] / 60, 1), 'phút')
ocr_gate = load_ocr_gate('cuda:0', threshold=0.15, views='frame+crops')
shared_model(('hybrid_ocr', DEVICES), lambda: load_hybrid_ocr(*DEVICES))   # loaded before timing
VARIANTS = {'before': {},
            'after': {'hybrid_pipelined': True, 'reuse_near_duplicates': True,
                      'hybrid_min_region_area': 0.02, 'hybrid_skip_repeated_decor': True}}
demo = {'video': VIDEO_ID, 'minutes': round(manifest['source']['duration_sec'] / 60, 1), 'stage01_sec': stage01_sec}
for name, options in VARIANTS.items():
    started = time.monotonic()
    _, _, records = run_visual_extraction(
        STAGE01, Path('/kaggle/working/demo') / name, VIDEO_ID, languages=('en',), describe_language='en',
        ocr_device='cuda', ocr_engine='hybrid', hybrid_devices=DEVICES, ocr_gate=ocr_gate, **options)
    demo[name] = {'sec': round(time.monotonic() - started, 1), 'segments': [
        {'segment_id': r['segment_id'], 'status': r['ocr_status'], 'reused_from': r.get('ocr_reused_from'),
         'ocr_text': r['ocr_text'], 'skipped_crops': [c.get('skipped') for c in json.loads(r['raw_ocr_response'] or '{}').get('crops', []) if c.get('skipped')],
         'regions': [{'kind': g['kind'], 'text': g.get('text', ''), 'description': g.get('description', '')}
                     for g in r['regions']]} for r in records]}
    print(name, demo[name]['sec'], 's', flush=True)
thumbs = {}
for interval in intervals:
    with Image.open(relative_asset(STAGE01, interval['image_path'])) as image:
        image = image.convert('RGB'); image.thumbnail((640, 360))
        buffer = io.BytesIO(); image.save(buffer, 'JPEG', quality=75)
    thumbs[interval['segment_id']] = {'start': interval.get('start_sec'), 'jpg': base64.b64encode(buffer.getvalue()).decode()}
demo['thumbs'] = thumbs
json.dump(demo, open('/kaggle/working/pipelined_demo.json', 'w'), ensure_ascii=False, indent=1)
print('nhanh hơn ×', round(demo['before']['sec'] / demo['after']['sec'], 2))
"""


def build(english=False):
    cells = [cell("markdown", INTRO.replace("L25_V011", "SlideSpeech cs_0001 (tiếng Anh)") if english else INTRO), cell("code", "%pip install -q " + fast.DEPS)]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in fast.MODULES]
    cells += [cell("markdown", "## Chạy before / after"), cell("code", RUN_EN if english else RUN)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    name, slug = (f"{NAME}_en", "pipelined-demo-en") if english else (NAME, "pipelined-demo")
    (ROOT / "notebooks" / f"{name}.ipynb").write_text(text, encoding="utf-8")
    run = ROOT / "runs" / f"{DATE}-{slug}"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{name}.ipynb").write_text(text, encoding="utf-8")
    json.dump({"id": f"{OWNER}/l25-{slug}-{DATE}", "title": f"l25-{slug}-{DATE}",
               "code_file": f"{name}.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{OWNER}/slidespeech-train100"] if english else [],
               "competition_sources": [], "kernel_sources": [] if english else [f"{OWNER}/{SOURCE}"]},
              open(run / "kernel-metadata.json", "w"), indent=2)
    print(run / f"{name}.ipynb")


if __name__ == "__main__":
    build("--en" in sys.argv)
