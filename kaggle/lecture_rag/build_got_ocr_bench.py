"""Writes notebooks/got_ocr_bench.ipynb: GOT-OCR 2.0 against the pipeline's hybrid OCR.

Two test sets, both already on Kaggle:
  m3av-maths              60 slide images with hand-checked text + LaTeX (M3AV seg_speech_ocr.json)
                          -> CER and word recall for GOT (3 modes) and the hybrid OCR, same images
  l25-ocr-gate-keyframes  217 Vietnamese L25 keyframes; the hybrid text of the 20261002 10-video
                          run is embedded here (no Vietnamese gold yet) -> length, diacritics,
                          agreement, and slides side by side to judge by eye

    python kaggle/lecture_rag/build_got_ocr_bench.py [<fast_pipeline_export dir>]
"""

import glob
import json
import sys
from pathlib import Path

from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
OWNER = "thnhtrungnguynmif"
DATE = "20261005"
NAME = "got_ocr_bench"
EXPORT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/20261002-full-10-ads/outputs/fast_pipeline_export"
DEPS = ('"transformers==4.57.1" "accelerate>=1" "einops" "addict" "easydict" '
        '"huggingface-hub>=0.25,<1" "Pillow>=10"')


def eval_source():
    """gold/eval_ocr.py without its command line entry (the notebook runs as __main__)."""
    text = (ROOT / "gold/eval_ocr.py").read_text(encoding="utf-8")
    return text.split('\nif __name__ == "__main__":')[0]


def hybrid_texts():
    """segment_id -> what the hybrid OCR read in the 10-video run (eval_ocr.prediction_text)."""
    scope = {}
    exec(eval_source(), scope)
    out = {}
    for path in glob.glob(str(EXPORT / "*/02_visual/visual.json")):
        for visual in json.loads(Path(path).read_text(encoding="utf-8")):
            out[visual["segment_id"]] = scope["prediction_text"](visual)
    return out


INTRO = """# GOT-OCR 2.0 so với OCR hiện tại của pipeline

**OCR hiện tại** (bước 02): Unlimited-OCR đọc cả slide, Qwen3-VL-4B đọc vùng hình (2 GPU, chậm).
**GOT-OCR 2.0** (580M, `stepfun-ai/GOT-OCR-2.0-hf`): một model nhỏ, ra chữ thường hoặc markdown/LaTeX.

1. **M3AV** (60 slide toán tiếng Anh, chữ + LaTeX đã người kiểm tra) → CER, độ phủ từ.
2. **L25** (217 keyframe tiếng Việt) → chưa có nhãn chuẩn: so độ dài, tỉ lệ chữ có dấu,
   độ giống nhau, và xem cạnh nhau bằng mắt.

Input: dataset `m3av-maths`, `l25-ocr-gate-keyframes`. GPU T4 x2."""

CONFIG = """import base64, csv, gc, glob, html, io, json, random, statistics, time, unicodedata
from pathlib import Path
import torch
from PIL import Image

M3AV_SLIDES = 5
VIDEO = 'L25_V002'   # chỉ 1 video nhỏ; '' = cả 217 keyframe
MAX_NEW_TOKENS = 1024
GOT_MODEL = 'stepfun-ai/GOT-OCR-2.0-hf'
GOT_DEVICE = 'cuda:0'
HYBRID_DEVICES = ('cuda:0', 'cuda:1')
MODES = [('GOT · chữ thường', {}),
         ('GOT · markdown/LaTeX', {'format': True})]
OUT = Path('/kaggle/working/got_ocr_bench'); OUT.mkdir(parents=True, exist_ok=True)
HYBRID_L25 = json.loads(base64.b64decode(HYBRID_L25_B64).decode('utf-8'))
"""

DATA = """# M3AV: ảnh slide + chữ đã người kiểm tra (mathpix+check), rải đều các bài
m3_root = Path(glob.glob('/kaggle/input/**/dataset_v1.0_noaudio', recursive=True)[0])
by_video = {}
for path in sorted(m3_root.glob('*/ocr/seg_speech_ocr.json')):
    items = []
    for entry in json.load(open(path, encoding='utf-8')):
        image = path.parent / 'seg_imgs' / entry['name']
        if image.exists() and len(normalize(entry.get('ocr_text') or '')) >= 20:
            items.append({'set': 'M3AV', 'id': entry['name'], 'image': image, 'gold': entry['ocr_text']})
    if items:
        by_video[path.parent.parent.name] = items
m3av, k = [], 0
while len(m3av) < M3AV_SLIDES and any(k < len(v) for v in by_video.values()):
    for items in by_video.values():
        if k < len(items) and len(m3av) < M3AV_SLIDES:
            m3av.append(items[k])
    k += 1
# L25: keyframe + chữ hybrid của lần chạy 10 video
l25_base = Path(glob.glob('/kaggle/input/**/keyframes.csv', recursive=True)[0]).parent
l25 = [{'set': 'L25', 'id': r['segment_id'], 'image': l25_base / r['image'], 'kind': r['content_kind'],
        'hybrid': HYBRID_L25.get(r['segment_id'], '')}
       for r in csv.DictReader(open(l25_base / 'keyframes.csv', encoding='utf-8'))]
if VIDEO:
    videos = sorted({i['id'].split(':')[0] for i in l25}, key=lambda v: sum(i['id'].startswith(v + ':') for i in l25))
    VIDEO = VIDEO if any(i['id'].startswith(VIDEO + ':') for i in l25) else videos[0]
    l25 = [i for i in l25 if i['id'].startswith(VIDEO + ':')]
print(VIDEO, len(by_video), 'bài M3AV ·', len(m3av), 'slide có nhãn ·', len(l25), 'keyframe L25')
"""

GOT = """from transformers import AutoModelForImageTextToText, AutoProcessor
processor = AutoProcessor.from_pretrained(GOT_MODEL)
got = AutoModelForImageTextToText.from_pretrained(GOT_MODEL, torch_dtype=torch.float16).to(GOT_DEVICE).eval()

def got_read(image, options):
    inputs = processor(image, return_tensors='pt', **options).to(GOT_DEVICE)
    with torch.inference_mode():
        ids = got.generate(**inputs, do_sample=False, tokenizer=processor.tokenizer,
                           stop_strings='<|im_end|>', max_new_tokens=MAX_NEW_TOKENS,
                           no_repeat_ngram_size=20)
    return processor.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

timing = {}
for mode, options in MODES:
    torch.cuda.reset_peak_memory_stats(GOT_DEVICE)
    seconds = []
    for n, item in enumerate(m3av + l25, 1):
        image = Image.open(item['image']).convert('RGB')
        torch.cuda.synchronize(); started = time.monotonic()
        try:
            item[mode] = got_read(image, options)
        except Exception as error:
            item[mode] = ''; item.setdefault('errors', []).append(f'{mode}: {error}'[:200])
        torch.cuda.synchronize(); seconds.append(time.monotonic() - started)
        print(f'{mode} {n}/{len(m3av) + len(l25)} {seconds[-1]:.1f}s', flush=True)
    timing[mode] = {'s_per_slide': round(statistics.mean(seconds), 2),
                    'vram_gb': round(torch.cuda.max_memory_allocated(GOT_DEVICE) / 2**30, 2)}
    print(mode, timing[mode], flush=True)
del got; gc.collect(); torch.cuda.empty_cache()
"""

HYBRID = """# OCR hiện tại của pipeline trên cùng 60 slide M3AV (L25 đã có chữ hybrid sẵn)
read = shared_model(('hybrid_ocr', HYBRID_DEVICES), lambda: load_hybrid_ocr(*HYBRID_DEVICES))
seconds = []
for item in m3av:
    image = Image.open(item['image']).convert('RGB')
    started = time.monotonic()
    try:
        blocks, regions, _ = read(image)
        item['hybrid'] = prediction_text({'ocr_text': '\\n'.join(b['text'] for b in blocks), 'regions': regions})
    except Exception as error:
        item['hybrid'] = ''; item.setdefault('errors', []).append(f'hybrid: {error}'[:200])
    seconds.append(time.monotonic() - started)
timing['Hybrid (hiện tại)'] = {'s_per_slide': round(statistics.mean(seconds), 2),
                               'vram_gb': round(sum(torch.cuda.max_memory_allocated(d) for d in HYBRID_DEVICES) / 2**30, 2)}
print(timing['Hybrid (hiện tại)'])
"""

SCORE = """ENGINES = ['Hybrid (hiện tại)'] + [mode for mode, _ in MODES]

def text_of(item, engine):
    return item.get('hybrid', '') if engine == ENGINES[0] else item.get(engine, '')

def score(prediction, gold):
    p, g = normalize(prediction), normalize(gold)
    edits = levenshtein(p, g)
    recall = word_recall(prediction, gold) or 0.0
    return {'edits': edits, 'gold_chars': len(g), 'cer': edits / max(1, len(g)),
            'word_recall': recall, 'lost': recall < LOST_RECALL}

def accent_share(text):
    letters = [c for c in unicodedata.normalize('NFD', text) if c.isalpha()]
    marks = sum(unicodedata.combining(c) > 0 for c in unicodedata.normalize('NFD', text))
    return marks / max(1, len(letters))

summary = []
for engine in ENGINES:
    m3 = summarize_text([score(text_of(item, engine), item['gold']) for item in m3av])
    readable = [item for item in l25 if len(normalize(item['hybrid'])) >= 80]
    texts = [text_of(item, engine) for item in readable]
    agree = ([1 - min(1, levenshtein(normalize(t), normalize(i['hybrid'])) / max(1, len(normalize(i['hybrid']))))
              for t, i in zip(texts, readable)] if engine != 'Hybrid (hiện tại)' else [1.0])
    summary.append({'engine': engine, **timing.get(engine, {}),
                    'm3av_cer': m3['cer_micro'], 'm3av_word_recall': m3['word_recall_mean'],
                    'm3av_lost': f"{m3['lost']}/{m3['slides']}",
                    'l25_chars_mean': round(statistics.mean(len(normalize(t)) for t in texts)),
                    'l25_accent_share': round(statistics.mean(accent_share(t) for t in texts), 3),
                    'l25_same_as_hybrid': round(statistics.mean(agree), 3),
                    'l25_short': sum(len(normalize(t)) < 0.4 * len(normalize(i['hybrid'])) for t, i in zip(texts, readable))})
for row in summary:
    print(row)
json.dump({'summary': summary, 'items': [{k: (str(v) if k == 'image' else v) for k, v in item.items()}
                                         for item in m3av + l25]},
          open(OUT / 'results.json', 'w'), ensure_ascii=False, indent=1)
"""

REPORT = """def thumb(path, width=520):
    image = Image.open(path).convert('RGB'); image.thumbnail((width, width))
    buffer = io.BytesIO(); image.save(buffer, 'JPEG', quality=70)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}">'

def pre(text):
    return f'<pre>{html.escape((text or "")[:1500])}</pre>'

LABEL = {'s_per_slide': 'giây/slide', 'vram_gb': 'VRAM (GB)', 'm3av_cer': 'M3AV · CER ↓',
         'm3av_word_recall': 'M3AV · độ phủ từ ↑', 'm3av_lost': 'M3AV · slide mất chữ',
         'l25_chars_mean': 'L25 · ký tự TB', 'l25_accent_share': 'L25 · tỉ lệ dấu',
         'l25_same_as_hybrid': 'L25 · giống hybrid', 'l25_short': 'L25 · ngắn hơn 40% hybrid'}
head = ''.join(f'<th>{LABEL.get(k, k)}</th>' for k in summary[0])
rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row.values()) + '</tr>' for row in summary)
best = ENGINES[1 + min(range(len(MODES)), key=lambda i: summary[i + 1]['m3av_cer'])]
pick_m3 = sorted(m3av, key=lambda i: -abs(score(i['hybrid'], i['gold'])['cer'] - score(i[best], i['gold'])['cer']))[:5]
readable = [i for i in l25 if len(normalize(i['hybrid'])) >= 80]
pick_l25 = sorted(readable, key=lambda i: levenshtein(normalize(i[best]), normalize(i['hybrid']))
                  / max(1, len(normalize(i['hybrid']))), reverse=True)
cards = ''
for item in pick_m3 + pick_l25:
    gold = f'<h4>Nhãn chuẩn</h4>{pre(item["gold"])}' if item.get('gold') else ''
    cer = (lambda t: f' · CER {score(t, item["gold"])["cer"]:.2f}' if item.get('gold') else '')
    cards += (f'<div class="card"><h3>{item["set"]} · {html.escape(item["id"])}</h3><div class="row">'
              f'<div>{thumb(item["image"])}{gold}</div>'
              f'<div><h4>Hybrid (hiện tại){cer(item["hybrid"])}</h4>{pre(item["hybrid"])}</div>'
              f'<div><h4>{html.escape(best)}{cer(item[best])}</h4>{pre(item[best])}</div></div></div>')
page = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>GOT-OCR vs hybrid</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#fbf3ea;color:#333}}
table{{border-collapse:collapse;background:#fff}}td,th{{border:1px solid #e3ceb8;padding:6px 9px;font-size:13px}}
th{{background:#f2e9dd}}.card{{background:#fff;border:1px solid #e3ceb8;border-radius:10px;padding:12px;margin:14px 0}}
.row{{display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:12px}}img{{max-width:100%;border:1px solid #ddd}}
pre{{white-space:pre-wrap;font-size:12px;background:#f7f1ea;padding:8px;border-radius:6px;max-height:420px;overflow:auto}}</style></head>
<body><h1>GOT-OCR 2.0 so với OCR hiện tại (hybrid)</h1>
<p>M3AV: {len(m3av)} slide toán tiếng Anh có nhãn chuẩn. L25: {len(l25)} keyframe tiếng Việt, chưa có nhãn chuẩn
({len(readable)} khung hybrid đọc được ≥ 80 ký tự dùng để so). CER tính sau khi chuẩn hóa (bỏ thẻ HTML, ký hiệu bảng, dấu $).</p>
<table><tr>{head}</tr>{rows}</table>
<h2>Xem cạnh nhau: slide M3AV và mọi keyframe L25 (GOT: {html.escape(best)})</h2>{cards}</body></html>'''
(OUT / 'got_ocr_report.html').write_text(page, encoding='utf-8')
from IPython.display import HTML, display
display(HTML(f'<table><tr>{head}</tr>{rows}</table>'))
"""


def build():
    import base64
    texts = hybrid_texts()
    blob = base64.b64encode(json.dumps(texts, ensure_ascii=False).encode("utf-8")).decode()
    cells = [cell("markdown", INTRO), cell("code", "%pip install -q " + DEPS),
             cell("code", f"HYBRID_L25_B64 = '{blob}'", hidden=True),
             cell("code", (CODE / "common.py").read_text(encoding="utf-8"), hidden=True),
             cell("code", (CODE / "visual.py").read_text(encoding="utf-8"), hidden=True),
             cell("code", eval_source(), hidden=True)]
    cells += [cell("code", code) for code in (CONFIG, DATA, GOT, HYBRID, SCORE, REPORT)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    (ROOT / "notebooks" / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    run = ROOT / "runs" / f"{DATE}-got-ocr-bench"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    json.dump({"id": f"{OWNER}/l25-got-ocr-bench-{DATE}", "title": f"l25-got-ocr-bench-{DATE}",
               "code_file": f"{NAME}.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [f"{OWNER}/m3av-maths", f"{OWNER}/l25-ocr-gate-keyframes"],
               "competition_sources": [], "kernel_sources": []},
              open(run / "kernel-metadata.json", "w"), indent=2)
    print(run.name, len(texts), "hybrid texts,", round(len(text) / 1e6, 2), "MB")


if __name__ == "__main__":
    build()
