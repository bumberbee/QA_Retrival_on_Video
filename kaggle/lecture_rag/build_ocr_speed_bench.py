"""Generate ocr_speed_bench.ipynb: where the hybrid OCR spends its time, and two speed-ups.

Same slides for every variant: 40 L25 keyframes (4 videos, in video order so the per-video
decoration memory works) and 20 M3AV slides with hand-checked text.
  base   current pipeline (Unlimited-OCR bf16 + Qwen3-VL per figure region)
  decor  + hybrid_skip_repeated_decor (visual.repeated_decor: logo / lecturer photo read once)
  fp16   + Unlimited-OCR in fp16 (T4 has no bf16 cores); masked_scatter_ is patched to cast its
         source, the reason bf16 was chosen. Skipped with the error if it still fails.
Reported: seconds per slide, Unlimited-OCR vs Qwen time, text similarity to base (difflib),
M3AV CER (gold/eval_ocr.py). A variant is worth turning on if similarity >= 0.98 and the CER
is no worse than base + 0.01.

    python kaggle/lecture_rag/build_ocr_speed_bench.py
"""

from pathlib import Path
import json

import build_got_ocr_bench as got
from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
NAME = "ocr_speed_bench"
DATE = "20261005"
OWNER = "thnhtrungnguynmif"

INTRO = """# Tốc độ OCR kết hợp · thời gian ở đâu, và hai cách tăng tốc

Cùng 40 keyframe L25 + 20 slide M3AV cho mọi bản: **base** (ảnh 1024, hiện tại), **size768**, **size640** (ảnh nhỏ hơn cho Unlimited-OCR). Bật một bản khi chữ
giống base ≥ 98% và CER M3AV không tệ hơn 0,01."""

CONFIG = """import csv, difflib, gc, glob, json, statistics, time
from pathlib import Path
import torch
from PIL import Image
L25_VIDEOS, PER_VIDEO, M3AV_SLIDES = ['L25_V001', 'L25_V007', 'L25_V010', 'L25_V004'], 10, 20
DEVICES = ('cuda:0', 'cuda:1')
"""

DATA = """kf_csv = next(Path('/kaggle/input').rglob('keyframes.csv'))
rows = [r for r in csv.DictReader(open(kf_csv, encoding='utf-8')) if int(r['ocr_chars']) >= 80]
l25 = []
for video in L25_VIDEOS:
    l25 += [{'set': 'L25', 'video': video, 'id': r['segment_id'], 'image': kf_csv.parent / r['image']}
            for r in rows if r['video'] == video][:PER_VIDEO]
m3 = []
for path in sorted(Path('/kaggle/input').rglob('seg_speech_ocr.json')):
    for entry in json.load(open(path, encoding='utf-8')):
        image = path.parent / 'seg_imgs' / entry['name']
        if image.exists() and len(normalize(entry.get('ocr_text') or '')) >= 20:
            m3.append({'set': 'M3AV', 'video': path.parent.parent.name, 'id': entry['name'], 'image': image, 'gold': entry['ocr_text']})
            break
m3 = m3[:M3AV_SLIDES]
items = l25 + m3
print(len(l25), 'L25 ·', len(m3), 'M3AV')
"""

RUN = """read = shared_model(('hybrid_ocr', DEVICES), lambda: load_hybrid_ocr(*DEVICES))

def run(variant):
    for key in read.timings:
        read.timings[key] = 0.0 if key.endswith('_sec') else 0
    memory, current, seconds = [], None, []
    for item in items:
        if item['video'] != current:
            memory, current = [], item['video']
        image = Image.open(item['image']).convert('RGB')
        torch.cuda.synchronize(); started = time.monotonic()
        try:
            blocks, regions, _ = read(image, decor_memory=memory if variant == 'decor' else None)
            item[variant] = prediction_text({'ocr_text': '\\n'.join(b['text'] for b in blocks), 'regions': regions})
        except Exception as error:
            item[variant] = ''; item.setdefault('errors', []).append(f'{variant}: {error}'[:200])
        torch.cuda.synchronize(); seconds.append(time.monotonic() - started)
    return {'variant': variant, 's_per_slide': round(statistics.mean(seconds), 2),
            **{k: round(v, 1) if isinstance(v, float) else v for k, v in read.timings.items()}}

timing = [run('base')]
for size in (768, 640):   # smaller Unlimited-OCR input: fewer vision tokens, faster
    read.options['base_size'] = size
    timing.append(run(f'size{size}'))
read.options['base_size'] = 1024
print(timing)
"""

SCORE = """def score(prediction, gold):   # as in got_ocr_bench
    p, g = normalize(prediction), normalize(gold)
    edits = levenshtein(p, g)
    recall = word_recall(prediction, gold) or 0.0
    return {'edits': edits, 'gold_chars': len(g), 'cer': edits / max(1, len(g)), 'word_recall': recall,
            'lost': recall < LOST_RECALL}

def similar(a, b):
    return difflib.SequenceMatcher(None, normalize(a or ''), normalize(b or '')).ratio()

summary = []
for row in timing:
    variant = row['variant']
    if 'error' in row:
        summary.append(row); continue
    sims = [similar(i['base'], i.get(variant)) for i in items]
    cer = summarize_text([score(i.get(variant) or '', i['gold']) for i in items if i['set'] == 'M3AV'])
    summary.append({**row, 'similar_to_base_mean': round(statistics.mean(sims), 4), 'similar_min': round(min(sims), 3),
                    'm3av_cer': cer['cer_micro'], 'errors': sum(variant + ':' in ' '.join(i.get('errors', [])) for i in items)})
for row in summary:
    print(row)
json.dump({'summary': summary, 'items': [{k: str(v) if k == 'image' else v for k, v in i.items()} for i in items]},
          open('/kaggle/working/ocr_speed_bench.json', 'w'), ensure_ascii=False, indent=1)
import pandas as pd
display(pd.DataFrame(summary))
"""


def build():
    cells = [cell("markdown", INTRO), cell("code", "%pip install -q " + got.DEPS)]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in ("common.py", "visual.py")]
    cells += [cell("code", got.eval_source(), hidden=True)]
    cells += [cell("markdown", "## Cấu hình"), cell("code", CONFIG),
              cell("markdown", "## Slide"), cell("code", DATA),
              cell("markdown", "## Chạy 3 bản"), cell("code", RUN),
              cell("markdown", "## So sánh"), cell("code", SCORE)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    (ROOT / "notebooks" / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    run = ROOT / "runs" / f"{DATE}-ocr-speed-bench"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    json.dump({"id": f"{OWNER}/l25-ocr-speed-bench-{DATE}", "title": f"l25-ocr-speed-bench-{DATE}",
               "code_file": f"{NAME}.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [f"{OWNER}/l25-ocr-gate-keyframes", f"{OWNER}/m3av-maths"],
               "competition_sources": [], "kernel_sources": []},
              open(run / "kernel-metadata.json", "w"), indent=2)
    print(run / f"{NAME}.ipynb")


if __name__ == "__main__":
    build()
