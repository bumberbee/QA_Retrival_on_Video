"""Generate gate_bench.ipynb: content filter phase 2, the OCR gate on frames of many channels.

Set A (~600 frames), labels taken from existing annotations, never from the gate itself:
  content = 1 (must be OCR'd)   L25 keyframes kind slide/mixed with >= 80 OCR chars (mixed = the
                                lecturer next to a slide: the hard group); M3AV slide images whose
                                hand-checked text has >= 20 chars; LPM frames in the middle of a
                                slide interval of slides.csv (author labels)
  content = 0 (nothing to read) L25 keyframes kind speaker/other with < 15 OCR chars
  info only                     L25 ad seconds (ad_labels.json: an ad may show text, so the gate
                                keeping it is not an error) and L25 speaker frames with 15-79 chars
Both gate views are scored ("frame", and "frame+crops" = max over the frame and gate_crops()),
then a threshold sweep: recall of content frames per source (target >= 0.99 on every source)
and the share of content-free frames skipped.

    python kaggle/lecture_rag/build_gate_bench.py
"""

from pathlib import Path
import json

from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
NAME = "gate_bench"
DATE = "20261005"
OWNER = "thnhtrungnguynmif"
MODULES = ["common.py", "detection.py", "adfilter.py", "visual.py"]
DEPS = ('"av>=14,<17" "opencv-python-headless>=4.10,<4.12" "Pillow>=10" '
        '"huggingface-hub>=0.25,<1" "transformers==4.57.1"')

INTRO = """# Lọc nội dung · Phase 2 · bộ gác OCR trên nhiều kênh

Bộ gác (SigLIP zero-shot) trả lời **"khung này có nội dung cần OCR không"**. Bộ A lấy nhãn từ chú
thích sẵn có (L25 keyframe, M3AV, mốc slide LPM), không từ chính bộ gác. Chấm cả hai cách nhìn
(`frame`, `frame+crops`) và quét ngưỡng: mục tiêu giữ **≥ 99% khung có nội dung ở mọi nguồn**,
rồi mới xem bỏ được bao nhiêu khung không có gì để đọc."""

CONFIG = """import csv, glob, json, random
from pathlib import Path
random.seed(0)
N_M3AV, N_LPM_PER_SPEAKER, N_AD = 150, 20, 80
THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
CURRENT = 0.10          # ngưỡng đang dùng trong pipeline
OUT = Path('/kaggle/working/gate_bench'); OUT.mkdir(parents=True, exist_ok=True)
INPUT = Path('/kaggle/input')
"""

DATA = """from PIL import Image
items = []
# L25 keyframes (l25-ocr-gate-keyframes): kind + số ký tự OCR của lần chạy 10 video
kf_csv = next(INPUT.rglob('keyframes.csv'))
for r in csv.DictReader(open(kf_csv, encoding='utf-8')):
    chars, kind = int(r['ocr_chars']), r['content_kind']
    if kind in {'slide', 'mixed'} and chars >= 80:
        label, group = 1, 'L25 · giảng viên + slide' if kind == 'mixed' else 'L25 · slide'
    elif kind in {'speaker', 'other'} and chars < 15:
        label, group = 0, 'L25 · không có chữ'
    else:
        label, group = None, 'L25 · giảng viên, ít chữ'
    items.append({'source': 'L25', 'group': group, 'label': label, 'id': r['segment_id'],
                  'image': str(kf_csv.parent / r['image'])})

# M3AV: ảnh slide + chữ đã kiểm tra tay
m3 = []
for path in INPUT.rglob('seg_speech_ocr.json'):
    for entry in json.load(open(path, encoding='utf-8')):
        image = path.parent / 'seg_imgs' / entry['name']
        if image.exists() and len((entry.get('ocr_text') or '').strip()) >= 20:
            m3.append({'source': 'M3AV', 'group': 'M3AV · slide toán', 'label': 1, 'id': entry['name'], 'image': str(image)})
items += random.sample(m3, min(N_M3AV, len(m3)))

# Khung lấy từ video: LPM giữa mỗi khoảng slide, L25 giây quảng cáo
video_files = {p.stem: p for p in INPUT.rglob('*.mp4')}
frames_dir = OUT / 'frames'; frames_dir.mkdir(exist_ok=True)
sources = {}
def grab(video_id, second, name):
    if video_id not in sources:
        sources[video_id] = probe_video(video_files[video_id])
    image, _ = frame_at_time(video_files[video_id], sources[video_id], second)
    path = frames_dir / f'{name}.jpg'; image.convert('RGB').save(path, quality=90)
    return str(path)

lpm_csv = next(p for p in INPUT.rglob('slides.csv') if 'lpm' in str(p).lower())
by_speaker = {}
for r in csv.DictReader(open(lpm_csv, encoding='utf-8')):
    if r['video_id'] in video_files:
        by_speaker.setdefault(r['speaker'], []).append(r)
for speaker, rows in sorted(by_speaker.items()):
    for r in random.sample(rows, min(N_LPM_PER_SPEAKER, len(rows))):
        middle = (float(r['start_s']) + float(r['end_s'])) / 2
        try:
            path = grab(r['video_id'], middle, f"lpm_{r['video_id']}_{r['slide_index']}")
        except Exception as error:
            print('bỏ', r['video_id'], error); continue
        items.append({'source': 'LPM', 'group': f'LPM · {speaker}', 'label': 1,
                      'id': f"{r['video_id']}#{r['slide_index']}", 'image': path})

ad_json = next(p for p in INPUT.rglob('ad_labels.json'))
ad_seconds = [(v['video'], s) for v in json.load(open(ad_json, encoding='utf-8'))['videos'] if v['video'] in video_files
              for s, flag in enumerate(v['per_second']) if flag]
for video_id, second in random.sample(ad_seconds, min(N_AD, len(ad_seconds))):
    items.append({'source': 'L25 ad', 'group': 'L25 · quảng cáo', 'label': None,
                  'id': f'{video_id}@{second}', 'image': grab(video_id, second + 0.5, f'ad_{video_id}_{second}')})

import collections
print(len(items), 'khung'); print(collections.Counter((i['group'], i['label']) for i in items))
"""

SCORE = """check, info = load_ocr_gate('cuda:0', threshold=0.0, views='frame')
for start in range(0, len(items), 32):
    batch = items[start:start + 32]
    images = [Image.open(i['image']).convert('RGB') for i in batch]
    for item, score in zip(batch, check(images)):
        item['frame'] = round(float(score), 4)
    for item, image in zip(batch, images):
        crops = check(gate_crops(image))
        item['crops'] = round(float(max(crops)), 4)
        item['frame+crops'] = max(item['frame'], item['crops'])
print('xong', len(items))
"""

SWEEP = """import html, io, base64
VIEWS = ['frame', 'frame+crops']
content = [i for i in items if i['label'] == 1]
empty = [i for i in items if i['label'] == 0]
groups = sorted({i['group'] for i in content})
rows = []
for view in VIEWS:
    for t in THRESHOLDS:
        kept = lambda group: [i[view] >= t for i in group]
        row = {'view': view, 'threshold': t,
               'recall_all': round(sum(kept(content)) / len(content), 4),
               'min_group_recall': round(min(sum(kept([i for i in content if i['group'] == g])) /
                                             len([i for i in content if i['group'] == g]) for g in groups), 4),
               'empty_skipped': round(sum(not k for k in kept(empty)) / max(1, len(empty)), 4),
               'ad_skipped': round(sum(i[view] < t for i in items if i['source'] == 'L25 ad') /
                                   max(1, sum(i['source'] == 'L25 ad' for i in items)), 4)}
        for g in groups:
            members = [i for i in content if i['group'] == g]
            row[g] = round(sum(i[view] >= t for i in members) / len(members), 3)
        rows.append(row)
# ngưỡng đề xuất: ngưỡng lớn nhất mà nhóm kém nhất vẫn giữ >= 99%
best = {view: max((r for r in rows if r['view'] == view and r['min_group_recall'] >= 0.99),
                  key=lambda r: r['threshold'], default=None) for view in VIEWS}
print(json.dumps(best, ensure_ascii=False, indent=1))
json.dump({'items': items, 'sweep': rows, 'best': best, 'current': CURRENT},
          open('/kaggle/working/gate_bench.json', 'w'), ensure_ascii=False, indent=1)
"""

REPORT = """def thumb(path, width=200):
    image = Image.open(path).convert('RGB'); image.thumbnail((width * 2, width * 2))
    buffer = io.BytesIO(); image.save(buffer, 'JPEG', quality=70)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}" width="{width}">'

def tile(i):
    return (f'<figure>{thumb(i["image"])}<figcaption>{html.escape(i["group"])}<br>{html.escape(i["id"])}<br>'
            f'frame {i["frame"]:.3f} · crops {i["crops"]:.3f}</figcaption></figure>')

cols = ['view', 'threshold', 'recall_all', 'min_group_recall', 'empty_skipped', 'ad_skipped'] + groups
table = '<table><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in cols) + '</tr>' + ''.join(
    '<tr' + (' class="cur"' if r['threshold'] == CURRENT else '') + '>' + ''.join(f'<td>{r[c]}</td>' for c in cols) + '</tr>'
    for r in rows) + '</table>'
missed = sorted((i for i in content if i['frame'] < CURRENT), key=lambda i: i['frame'])
low = sorted(content, key=lambda i: i['frame+crops'])[:24]
kept_empty = sorted((i for i in empty if i['frame'] >= CURRENT), key=lambda i: -i['frame'])
page = ('<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Gate bench</title><style>'
        'body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#fbf3ea;color:#333}'
        'table{border-collapse:collapse;background:#fff;font-size:12px}td,th{border:1px solid #e3ceb8;padding:4px 7px}'
        'tr.cur{background:#fde9dc}.row{display:flex;flex-wrap:wrap;gap:8px}figure{margin:0;font-size:11px;color:#666;width:200px}</style></head><body>'
        f'<h1>Bộ gác OCR trên {len(items)} khung</h1><p>{len(content)} khung có nội dung · {len(empty)} khung không có chữ · '
        f'tô màu: ngưỡng đang dùng {CURRENT}. Đề xuất: {html.escape(json.dumps(best, ensure_ascii=False))}</p>' + table +
        f'<h2>Khung có nội dung bị bỏ ở ngưỡng {CURRENT} (frame): {len(missed)}</h2><div class="row">' + ''.join(tile(i) for i in missed) + '</div>'
        '<h2>24 khung có nội dung điểm thấp nhất (frame+crops)</h2><div class="row">' + ''.join(tile(i) for i in low) + '</div>'
        f'<h2>Khung không có chữ nhưng vẫn được OCR ở ngưỡng {CURRENT}: {len(kept_empty)}</h2><div class="row">' + ''.join(tile(i) for i in kept_empty) + '</div>'
        '</body></html>')
Path('/kaggle/working/gate_bench_report.html').write_text(page, encoding='utf-8')
import shutil; shutil.rmtree(OUT)
import pandas as pd
display(pd.DataFrame(rows)[cols])
"""


def build():
    cells = [cell("markdown", INTRO), cell("code", "%pip install -q " + DEPS)]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in MODULES]
    cells += [cell("markdown", "## Cấu hình"), cell("code", CONFIG),
              cell("markdown", "## Dựng bộ A"), cell("code", DATA),
              cell("markdown", "## Chấm bộ gác (frame và frame+crops)"), cell("code", SCORE),
              cell("markdown", "## Quét ngưỡng"), cell("code", SWEEP),
              cell("markdown", "## Báo cáo"), cell("code", REPORT)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    (ROOT / "notebooks" / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    run = ROOT / "runs" / f"{DATE}-gate-bench"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    json.dump({"id": f"{OWNER}/l25-gate-bench-{DATE}", "title": f"l25-gate-bench-{DATE}",
               "code_file": f"{NAME}.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [f"{OWNER}/l25-ocr-gate-keyframes", f"{OWNER}/m3av-maths", f"{OWNER}/lpm-videos",
                                   f"{OWNER}/l25a-lecture-videos-media", f"{OWNER}/l25-ad-filter-frames"],
               "competition_sources": [], "kernel_sources": []},
              open(run / "kernel-metadata.json", "w"), indent=2)
    print(run / f"{NAME}.ipynb")


if __name__ == "__main__":
    build()
