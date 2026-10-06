"""Generate ad_flag_check.ipynb: content filter phases 0-1 on many channels, stage 01 only.

Runs the slide detection + ad filter in "flag" mode (adfilter.py) on L25 videos with gold ad
labels, ad-free English lectures (M3AV, LPM, SlideSpeech) and the 12 Vietnamese test videos
of new channels, then exports 01_slides of every video for gold/eval_content_filter.py and an
HTML page with the flagged segments (to check the new channels by eye; no labels yet).

    python kaggle/lecture_rag/build_ad_flag_check.py
"""

from pathlib import Path
import json

from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
NAME = "ad_flag_check"
DATE = "20261005"
OWNER = "thnhtrungnguynmif"
MODULES = ["common.py", "detection.py", "adfilter.py"]
DEPS = ('"av>=14,<17" "opencv-python-headless>=4.10,<4.12" "Pillow>=10" '
        '"huggingface-hub>=0.25,<1" "transformers==4.57.1"')

INTRO = """# Lọc nội dung · Phase 0–1 · chỉ bước 01 trên nhiều kênh

Bộ lọc quảng cáo chạy ở chế độ **`flag`**: đoạn nghi quảng cáo chỉ được đánh dấu, không bị cắt;
bộ an toàn bỏ mọi dấu khi > 6 đoạn hoặc > 30% video. Ba nhóm video:

| Nhóm | Vì sao |
|---|---|
| L25 (có nhãn quảng cáo theo giây) | chế độ mới phải giữ số đo cũ |
| M3AV, LPM, SlideSpeech (không có quảng cáo) | không được đánh dấu nhầm bài giảng |
| 12 video tiếng Việt kênh mới (chưa có nhãn) | xem bằng mắt các đoạn bị đánh dấu |

Kết quả: `ad_flag_export.zip` (01_slides của từng video, chấm bằng `gold/eval_content_filter.py`)
và `ad_flag_report.html`."""

CONFIG = """from pathlib import Path
VIDEOS = {
    'l25': ['L25_V003', 'L25_V004', 'L25_V006', 'L25_V008', 'L25_V009', 'L25_V010'],
    'ad_free': ['MLS-15DA1', 'MLS-03E08', 'MLS-2A8E1', 'MLS-C00E5',          # M3AV · Oxford
                'IIfeX44RI9k', 'dC93OcrNX6c', 'Xee8gmppUIw', '7EsHL8Sacg0',  # LPM · anat-1, bio-1, bio-4, dental
                'cs_0001', 'history_0000', 'agriculture_0002', 'music_0000'],  # SlideSpeech
    'vn_new': ['tHE1mC_n0-E', 'D_kuulxKuxc', 'bstCZZUMJYw', '_eDGTHhyQ7M', '8WnDooAH2a4', 'szRBlpJ0kkY',
               'laT0CKatosY', 'xysJI15jpAw', 'v2YTz8ToEag', 'hVU5Mw19V_c', 'xLGGWMDJRkY', 'y1CvB950VlU'],
}
OUT = Path('/kaggle/working/ad_flag')
AD_OPTIONS = {'mode': 'flag'}          # max_runs=6, max_share=0.30 (mặc định của adfilter.py)
SAMPLE_HZ, FINE_HZ = 1.0, 8.0
"""

RUN = """import base64, html, io, json, shutil, time, traceback
from PIL import Image

videos = {p.stem: p for p in Path('/kaggle/input').rglob('*.mp4')}
student = next(iter(sorted(Path('/kaggle/input').rglob('ad_student.npz'))))
scorer = load_ad_scorer('student', 'cuda:0', student)
rows = []
for group, names in VIDEOS.items():
    for name in names:
        if name not in videos:
            print('THIẾU', name, flush=True); rows.append({'group': group, 'video': name, 'error': 'missing'}); continue
        started = time.monotonic()
        try:
            stage01, manifest, intervals = run_slide_detection(
                videos[name], OUT / name, name, sample_hz=SAMPLE_HZ, fine_hz=FINE_HZ,
                merge_reveals=True, scan='single_pass', ad_scorer=scorer, ad_options=AD_OPTIONS)
            report = json.loads((stage01 / 'ad_filter.json').read_text(encoding='utf-8'))
            flagged = [i for i in intervals if i.get('ad_flagged')]
            rows.append({'group': group, 'video': name, 'minutes': round(manifest['source']['duration_sec'] / 60, 1),
                         'segments': len(intervals), 'ad_runs': len(report['ad_intervals']),
                         'ad_share': round(report['ad_seconds'] / manifest['source']['duration_sec'], 3),
                         'guard': report['guard']['reason'], 'flagged': len(flagged),
                         'seconds': round(time.monotonic() - started)})
        except Exception:
            traceback.print_exc(); rows.append({'group': group, 'video': name, 'error': 'failed'})
        print(rows[-1], flush=True)
"""

REPORT = """def thumb(path, width=210):
    image = Image.open(path).convert('RGB'); image.thumbnail((width * 2, width * 2))
    buffer = io.BytesIO(); image.save(buffer, 'JPEG', quality=70)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}" width="{width}">'

def clock(s):
    return f'{int(s) // 60}:{int(s) % 60:02d}'

cards = []
for row in rows:
    if 'error' in row:
        continue
    folder = OUT / row['video'] / '01_slides'
    report = json.loads((folder / 'ad_filter.json').read_text(encoding='utf-8'))
    intervals = json.loads((folder / 'intervals.json').read_text(encoding='utf-8'))
    pieces = [i for i in intervals if i.get('boundary_method') == 'ad_filter']
    shots = ''.join(f'<figure>{thumb(folder / i["image_path"])}<figcaption>{clock(i["start_sec"])}–{clock(i["end_sec"])}'
                    f' · điểm {i.get("ad_score")}</figcaption></figure>' for i in pieces[:12])
    state = (f'<b style="color:#c4553f">bộ an toàn bỏ dấu: {html.escape(row["guard"])}</b>' if row['guard']
             else f'{row["flagged"]} đoạn bị đánh dấu')
    cards.append(f'<div class="card"><h3>{row["group"]} · {html.escape(row["video"])} · {row["minutes"]} phút</h3>'
                 f'<p>{row["ad_runs"]} đoạn nghi · {row["ad_share"]:.0%} video · {state}</p><div class="row">{shots or "<i>không có</i>"}</div></div>')
head = ''.join(f'<th>{k}</th>' for k in ['group', 'video', 'minutes', 'ad_runs', 'ad_share', 'flagged', 'guard', 'seconds'])
body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(r.get(k, r.get("error", ""))))}</td>'
                               for k in ['group', 'video', 'minutes', 'ad_runs', 'ad_share', 'flagged', 'guard', 'seconds']) + '</tr>'
               for r in rows)
page = ('<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Ad flag check</title><style>'
        'body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#fbf3ea;color:#333}'
        'table{border-collapse:collapse;background:#fff}td,th{border:1px solid #e3ceb8;padding:5px 8px;font-size:13px}'
        '.card{background:#fff;border:1px solid #e3ceb8;border-radius:10px;padding:10px 14px;margin:12px 0}'
        '.row{display:flex;flex-wrap:wrap;gap:8px}figure{margin:0;font-size:11px;color:#777}</style></head><body>'
        '<h1>Lọc nội dung · chế độ flag trên nhiều kênh</h1><table><tr>' + head + '</tr>' + body + '</table>'
        '<h2>Các đoạn nghi quảng cáo (ảnh giữa đoạn)</h2>' + ''.join(cards) + '</body></html>')
Path('/kaggle/working/ad_flag_report.html').write_text(page, encoding='utf-8')
json.dump(rows, open('/kaggle/working/ad_flag_rows.json', 'w'), indent=1)
export = Path('/kaggle/working/ad_flag_export')
shutil.rmtree(export, ignore_errors=True)
for row in rows:
    if 'error' not in row:
        source = OUT / row['video'] / '01_slides'
        target = export / row['video'] / '01_slides'
        target.mkdir(parents=True)
        for name in ('intervals.json', 'ad_filter.json', 'manifest.json'):
            shutil.copy(source / name, target / name)
shutil.make_archive('/kaggle/working/ad_flag_export', 'zip', export)
shutil.rmtree(export); shutil.rmtree(OUT)
import pandas as pd
display(pd.DataFrame(rows))
"""


def build():
    cells = [cell("markdown", INTRO), cell("code", "%pip install -q " + DEPS)]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in MODULES]
    cells += [cell("markdown", "## Cấu hình"), cell("code", CONFIG),
              cell("markdown", "## Chạy bước 01 ở chế độ flag"), cell("code", RUN),
              cell("markdown", "## Báo cáo và gói kết quả"), cell("code", REPORT)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    (ROOT / "notebooks" / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    run = ROOT / "runs" / f"{DATE}-ad-flag-check"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    json.dump({"id": f"{OWNER}/l25-ad-flag-check-{DATE}", "title": f"l25-ad-flag-check-{DATE}",
               "code_file": f"{NAME}.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [f"{OWNER}/l25a-lecture-videos-media", f"{OWNER}/vn-lecture-test12",
                                   f"{OWNER}/lpm-videos", f"{OWNER}/m3av-maths", f"{OWNER}/slidespeech-train100"],
               "competition_sources": [], "kernel_sources": [f"{OWNER}/l25-ad-filter-e1-e4-20261002"]},
              open(run / "kernel-metadata.json", "w"), indent=2)
    print(run / f"{NAME}.ipynb")


if __name__ == "__main__":
    build()
