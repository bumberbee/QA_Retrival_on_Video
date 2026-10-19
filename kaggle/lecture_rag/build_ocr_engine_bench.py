"""Generate ocr_engine_bench.ipynb: which OCR is fast and good on lecture slides.

Images: 20 Vietnamese slides (dataset ocr-bench-vi-slides, text typed by Claude in gt.json, not
yet human-checked) and 40 M3AV maths slides (hand-checked text of seg_speech_ocr.json). Every
engine reads the same content crop (detect_content_roi), one image at a time, after a warm-up:
  unlimited   Unlimited-OCR alone (text, LaTeX, tables), as in stage 02 without the VLM
  hybrid      stage 02 as run today (Unlimited-OCR + Qwen3-VL on figure regions)
  qwen_only   Qwen3-VL-4B reads the whole slide
  paddle_v5   PaddleOCR 3 (PP-OCRv5, lang vi / en), CPU, in its own venv
  easyocr     EasyOCR vi+en / en on GPU (the old stage-02 engine)
  got         GOT-OCR 2.0
Output ocr_engine_bench.json: text and seconds per engine and image; scored locally.

    python kaggle/lecture_rag/build_ocr_engine_bench.py
"""

from pathlib import Path
import json

import build_fast_pipeline as fast
from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
NAME = "ocr_engine_bench"
DATE = "20261013"
OWNER = "thnhtrungnguynmif"

INTRO = """# So sánh công cụ OCR trên slide bài giảng

Cùng một vùng slide cho mỗi công cụ, đo chữ đọc được và thời gian mỗi ảnh. Bộ tiếng Việt: 20 slide
L25/vn12 (đáp án Claude gõ, chưa người kiểm). Bộ tiếng Anh: 40 slide toán M3AV (chữ kiểm tay)."""

DATA = """import json, random, statistics, subprocess, sys, time
from pathlib import Path
from PIL import Image
random.seed(0)
INPUT = Path('/kaggle/input')
OUT = Path('/kaggle/working/crops'); OUT.mkdir(parents=True, exist_ok=True)
images = []   # {'set', 'name', 'path'} (content crop saved as JPEG)
vi_dir = next(p.parent for p in INPUT.rglob('gt.json'))
for path in sorted(vi_dir.glob('vi_*.jpg')):
    images.append({'set': 'vi', 'name': path.name, 'source': str(path)})
m3 = []
for path in INPUT.rglob('seg_speech_ocr.json'):
    for entry in json.load(open(path, encoding='utf-8')):
        image = path.parent / 'seg_imgs' / entry['name']
        if image.exists() and len((entry.get('ocr_text') or '').strip()) >= 20:
            m3.append({'set': 'm3av', 'name': entry['name'], 'source': str(image), 'gt': entry['ocr_text']})
images += random.sample(m3, min(40, len(m3)))
for item in images:
    with Image.open(item['source']) as opened:
        image = opened.convert('RGB')
    roi = detect_content_roi(image)
    item['path'] = str(OUT / f"{item['set']}_{Path(item['name']).stem}.jpg")
    crop_to_roi(image, roi).save(item['path'], quality=95)
print(len(images), 'ảnh', sum(i['set'] == 'vi' for i in images), 'vi')
results = {}

def run(engine, read, warmup=True):
    if warmup:
        read(Image.open(images[0]['path']).convert('RGB'), images[0]['set'])
    rows = []
    for item in images:
        image = Image.open(item['path']).convert('RGB')
        started = time.monotonic()
        try:
            text = read(image, item['set'])
        except Exception as error:
            text = ''; print(engine, item['name'], repr(error)[:200])
        rows.append({'set': item['set'], 'name': item['name'], 'sec': round(time.monotonic() - started, 2), 'text': text})
    results[engine] = rows
    print(engine, 'TB', round(statistics.mean(r['sec'] for r in rows), 2), 's/ảnh', flush=True)
"""

GPU = """DEVICES = ('cuda:0', 'cuda:1')
hybrid = shared_model(('hybrid_ocr', DEVICES), lambda: load_hybrid_ocr(*DEVICES))

def unlimited_text(image, lang):
    blocks = dedupe_blocks(parse_unlimited_blocks(hybrid.unlimited(image)))
    return '\\n'.join(b['text'] for b in blocks if b['text'] and b['text'] != '[Non-Text]' and not is_visual(b))
run('unlimited', unlimited_text)

def hybrid_text(image, lang):
    blocks, regions, _ = hybrid(image, language='vi' if lang == 'vi' else 'en')
    return ocr_lines_text(blocks) + '\\n' + '\\n'.join(r['text'] for r in regions if r['kind'] in ('table', 'formula'))
run('hybrid', hybrid_text)

WHOLE = ('Read all the text on this lecture slide in reading order. Keep Vietnamese diacritics exactly. '
         'Write formulas in LaTeX and tables in Markdown. Return only the text.')
def qwen_text(image, lang):
    processor, vlm = hybrid.processor, hybrid.vlm
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': WHOLE}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors='pt').to(vlm.device)
    import torch
    with torch.inference_mode():
        out = vlm.generate(**inputs, max_new_tokens=2048, do_sample=False, repetition_penalty=1.05)
    return processor.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
run('qwen_only', qwen_text)

import easyocr, numpy as np
readers = {'vi': easyocr.Reader(['vi', 'en'], gpu=True, verbose=False), 'm3av': easyocr.Reader(['en'], gpu=True, verbose=False)}
def easy_text(image, lang):
    return '\\n'.join(t for _, t, _ in readers[lang].readtext(np.asarray(image)))
run('easyocr', easy_text)

try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    got_processor = AutoProcessor.from_pretrained('stepfun-ai/GOT-OCR-2.0-hf')
    got = AutoModelForImageTextToText.from_pretrained('stepfun-ai/GOT-OCR-2.0-hf', torch_dtype=torch.float16,
                                                      device_map='cuda:1').eval()
    def got_text(image, lang):
        inputs = got_processor(image, return_tensors='pt', format=True).to('cuda:1')
        with torch.inference_mode():
            out = got.generate(**inputs, do_sample=False, tokenizer=got_processor.tokenizer,
                               stop_strings='<|im_end|>', max_new_tokens=4096)
        return got_processor.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    run('got', got_text)
except Exception as error:
    print('GOT lỗi', repr(error)[:300])
"""

PADDLE = """# PaddleOCR in its own venv (CPU) so its numpy/opencv pins cannot break torch
script = '''
import json, sys, time
from paddleocr import PaddleOCR
items = json.load(open(sys.argv[1]))
engines = {}
def engine(lang):
    if lang not in engines:
        engines[lang] = PaddleOCR(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False,
                                  use_textline_orientation=False)
    return engines[lang]
for lang in ('vi', 'en'):
    engine(lang).predict(items[0]['path'])   # warm-up
rows = []
for item in items:
    lang = 'vi' if item['set'] == 'vi' else 'en'
    started = time.monotonic()
    try:
        res = engine(lang).predict(item['path'])
        text = chr(10).join(t for r in res for t in r['rec_texts'])
    except Exception as error:
        text = ''; print(item['name'], repr(error)[:200])
    rows.append({'set': item['set'], 'name': item['name'], 'sec': round(time.monotonic() - started, 2), 'text': text})
json.dump(rows, open(sys.argv[2], 'w'), ensure_ascii=False)
'''
Path('/kaggle/working/paddle_run.py').write_text(script)
json.dump(images, open('/kaggle/working/items.json', 'w'))
setup = subprocess.run('python -m venv /kaggle/working/pd && /kaggle/working/pd/bin/pip install -q paddlepaddle "paddleocr>=3.1"',
                       shell=True, capture_output=True, text=True)
print(setup.stdout[-1500:], setup.stderr[-1500:])
done = subprocess.run(['/kaggle/working/pd/bin/python', '/kaggle/working/paddle_run.py', '/kaggle/working/items.json',
                       '/kaggle/working/paddle.json'], capture_output=True, text=True)
print(done.stdout[-1500:], done.stderr[-3000:])
if Path('/kaggle/working/paddle.json').exists():
    results['paddle_v5'] = json.load(open('/kaggle/working/paddle.json'))
    print('paddle_v5 TB', round(statistics.mean(r['sec'] for r in results['paddle_v5']), 2), 's/ảnh (CPU)')
"""

SAVE = """json.dump({'images': [{k: v for k, v in i.items() if k != 'source'} for i in images], 'results': results},
          open('/kaggle/working/ocr_engine_bench.json', 'w'), ensure_ascii=False, indent=1)
print({k: len(v) for k, v in results.items()})
"""


def build():
    cells = [cell("markdown", INTRO), cell("code", "%pip install -q " + fast.DEPS)]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in fast.MODULES]
    cells += [cell("markdown", "## Ảnh"), cell("code", DATA),
              cell("markdown", "## PaddleOCR v5 (CPU, venv riêng)"), cell("code", PADDLE),
              cell("markdown", "## Các công cụ trên GPU"), cell("code", GPU),
              cell("markdown", "## Lưu"), cell("code", SAVE)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    (ROOT / "notebooks" / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    run = ROOT / "runs" / f"{DATE}-ocr-engine-bench"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    json.dump({"id": f"{OWNER}/ocr-engine-bench-{DATE}", "title": f"ocr-engine-bench-{DATE}",
               "code_file": f"{NAME}.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [f"{OWNER}/ocr-bench-vi-slides", f"{OWNER}/m3av-maths"],
               "competition_sources": [], "kernel_sources": []},
              open(run / "kernel-metadata.json", "w"), indent=2)
    print(run / f"{NAME}.ipynb")


if __name__ == "__main__":
    build()
