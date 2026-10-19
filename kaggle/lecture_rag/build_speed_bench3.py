"""Generate speed_bench3.ipynb: ways to make stage 02 (OCR, ~86% of the time) faster, on 3 L25 videos,
one Kaggle kernel per video so they run in parallel.

Input: the stage-01 output of L25_V017, L25_V035, L25_V077 from kernel l25-all-0-20261007
(992 s, 482 s, 680 s of stage 02 in that run). Every variant starts with no cache:
  base          run_visual_extraction, current defaults
  reuse_skip    + reuse_near_duplicates, hybrid_min_region_area 0.02, hybrid_skip_repeated_decor
  seq           the same in this notebook's own loop (checks the loop matches the pipeline)
  batch         + the figure regions of one slide go to Qwen3-VL in one batched generate()
  overlap       + Unlimited-OCR (GPU 0) reads the next slide while Qwen3-VL (GPU 1) reads this one
  overlap_768   + Unlimited-OCR at base_size 768 instead of 1024
  qwen_only     Qwen3-VL reads the whole slide alone (no Unlimited-OCR)
  got           GOT-OCR 2.0 reads the whole slide alone (text only, no figure reading)
plus a try of Unlimited-OCR in fp16 (bf16 has no tensor cores on a T4).
Reported per video and variant: seconds, speed-up, model reads, Qwen calls, text similarity to
base (difflib on OCR + region text) and recall of base's OCR words.

    python kaggle/lecture_rag/build_speed_bench3.py
"""

from pathlib import Path
import json

import build_fast_pipeline as fast
from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
NAME = "speed_bench3"
DATE = "20261012"
OWNER = "thnhtrungnguynmif"
SOURCE = "l25-all-0-20261007"

INTRO = """# Tăng tốc bước 02 (OCR) · thử 8 cách trên 3 video L25

Bước 02 chiếm ~86% thời gian xử lý một video. Mỗi bản chạy với cache trống, cùng bộ gác OCR,
trên **L25_V017, L25_V035, L25_V077**. So thời gian, số lần gọi model, và chữ giống bản hiện tại."""

CONFIG = """import difflib, hashlib, json, queue, re, statistics, threading, time
from pathlib import Path
import torch
from PIL import Image
VIDEOS = [__VIDEO__]
STAGE01 = {v: next(p for p in Path('/kaggle/input').rglob('01_slides') if p.parent.name == v) for v in VIDEOS}
DEVICES = ('cuda:0', 'cuda:1')
SKIP = {'hybrid_min_region_area': 0.02, 'hybrid_skip_repeated_decor': True}
ocr_gate = load_ocr_gate('cuda:0', threshold=0.15, views='frame+crops')
hybrid = shared_model(('hybrid_ocr', DEVICES), lambda: load_hybrid_ocr(*DEVICES))   # loaded before timing
PROMPT = hybrid_crop_prompt('vi')
results = {v: {} for v in VIDEOS}
print(STAGE01)
"""

PIPELINE = """def pipeline_text(record):
    return record['ocr_text'], '\\n'.join(r.get('text', '') + ' ' + r.get('description', '') for r in record['regions'])

def run_pipeline(video, name, **options):
    started = time.monotonic()
    _, _, records = run_visual_extraction(
        STAGE01[video], Path('/kaggle/working/bench') / video / name, video, languages=('vi', 'en'),
        ocr_device='cuda', ocr_engine='hybrid', hybrid_devices=DEVICES, ocr_gate=ocr_gate, **options)
    sec = time.monotonic() - started
    read = [r for r in records if r['ocr_status'] == 'read']
    vlm = sum(not c.get('skipped') for r in read if not r.get('ocr_reused_from') and r['raw_ocr_response']
              for c in json.loads(r['raw_ocr_response']).get('crops', []))
    results[video][name] = {'sec': round(sec, 1), 'reads': sum(not r.get('ocr_reused_from') for r in read),
                            'vlm_calls': vlm, 'texts': {r['segment_id']: pipeline_text(r) for r in read}}
    print(video, name, results[video][name]['sec'], 's', flush=True)

for video in VIDEOS:
    run_pipeline(video, 'base')
    run_pipeline(video, 'reuse_skip', reuse_near_duplicates=True, **SKIP)
"""

CUSTOM = """def keyframes(video, reuse=True):
    \"\"\"(segment_id, slide crop or None, segment it reuses) for every keyframe the gate keeps.\"\"\"
    directory, seen, out = STAGE01[video], [], []
    for interval in read_json(directory / 'intervals.json'):
        if interval.get('content_kind') == 'ad':
            continue
        with Image.open(relative_asset(directory, interval['image_path'])) as opened:
            image = opened.convert('RGB')
        roi = detect_content_roi(image)
        if gate_keyframe(ocr_gate, image, roi)[0]:
            continue
        crop = crop_to_roi(image, roi)
        if reuse:
            thumb = slide_thumb(crop)
            source = near_duplicate(seen, thumb)
            if source is not None:
                out.append((interval['segment_id'], None, source))
                continue
            seen.append((interval['segment_id'], thumb))
        out.append((interval['segment_id'], crop, None))
    return out

def crop_box(image, box, pad=0.01):
    w, h = image.width, image.height
    return image.crop((max(0, (box[0] - pad) * w), max(0, (box[1] - pad) * h),
                       min(w, (box[2] + pad) * w), min(h, (box[3] + pad) * h)))

def parse_crop(raw):
    try:
        return raw, json.loads(raw[raw.find('{'): raw.rfind('}') + 1])
    except ValueError:
        return raw, {'kind': 'figure', 'description': raw.strip()}

def vlm_batch(images, prompt, batch_size=4, max_new_tokens=1200):
    \"\"\"Qwen3-VL on several images in one generate() (left padding).\"\"\"
    processor, vlm = hybrid.processor, hybrid.vlm
    processor.tokenizer.padding_side = 'left'
    out = []
    for i in range(0, len(images), batch_size):
        messages = [[{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
                    for image in images[i:i + batch_size]]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, padding=True,
                                               return_dict=True, return_tensors='pt').to(vlm.device)
        with torch.inference_mode():
            generated = vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
        out += processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return out

def slide_text(blocks, regions):
    return ocr_lines_text(blocks), '\\n'.join(r.get('text', '') + ' ' + r.get('description', '') for r in regions)

def run_custom(video, name, batch=True, overlap=False, base_size=1024):
    \"\"\"A slide is read in two passes of hybrid_read_slide(): the first runs Unlimited-OCR and
    collects the figure boxes (the VLM is a stub), the second gets the VLM answers. Between the two,
    the boxes of a slide go to Qwen3-VL one by one or in one batch, and (overlap) Unlimited-OCR on
    GPU 0 already reads the next slide. Decoration memory is a snapshot taken at the first pass.\"\"\"
    hybrid.options['base_size'] = base_size
    started = time.monotonic()
    frames = keyframes(video)
    decor, texts, stats = [], {}, {'vlm_calls': 0}

    def plan(segment_id, image):
        cache, boxes, snapshot = {}, [], list(decor)
        def unlimited(img):
            key = hashlib.md5(img.tobytes()).hexdigest()
            if key not in cache:
                cache[key] = hybrid.unlimited(img)
            return cache[key]
        def stub(img, box):
            boxes.append(box)
            return '', {'kind': 'pending'}
        hybrid_read_slide(image, unlimited, stub, decor_memory=list(snapshot), min_region_area=0.02)
        return segment_id, image, unlimited, boxes, snapshot

    def finish(segment_id, image, unlimited, boxes, snapshot):
        if batch:
            answers = [parse_crop(raw) for raw in vlm_batch([crop_box(image, b) for b in boxes], PROMPT)]
        else:
            answers = [hybrid.read_crop(image, b, PROMPT) for b in boxes]
        stats['vlm_calls'] += len(boxes)
        it, memory = iter(answers), list(snapshot)
        blocks, regions, _ = hybrid_read_slide(image, unlimited, lambda img, box: next(it),
                                               decor_memory=memory, min_region_area=0.02)
        decor.extend(memory[len(snapshot):])
        texts[segment_id] = slide_text(blocks, regions)

    todo = [(segment_id, image) for segment_id, image, _ in frames if image is not None]
    if overlap:
        planned = queue.Queue(maxsize=2)
        def producer():
            for item in todo:
                planned.put(plan(*item))
            planned.put(None)
        thread = threading.Thread(target=producer)
        thread.start()
        while (item := planned.get()) is not None:
            finish(*item)
        thread.join()
    else:
        for item in todo:
            finish(*plan(*item))
    for segment_id, image, source in frames:
        if image is None:
            texts[segment_id] = texts[source]
    hybrid.options['base_size'] = 1024
    results[video][name] = {'sec': round(time.monotonic() - started, 1), 'reads': len(todo),
                            'vlm_calls': stats['vlm_calls'], 'texts': texts}
    print(video, name, results[video][name]['sec'], 's', flush=True)

for video in VIDEOS:
    run_custom(video, 'seq', batch=False)
    run_custom(video, 'batch')
    run_custom(video, 'overlap', overlap=True)
    run_custom(video, 'overlap_768', overlap=True, base_size=768)
"""

ALONE = """WHOLE_PROMPT = ('Đọc toàn bộ nội dung slide bài giảng này theo thứ tự đọc. Giữ nguyên tiếng Việt có dấu. '
                'Công thức viết LaTeX, bảng viết Markdown, hình/biểu đồ mô tả ngắn một dòng [Hình: ...]. '
                'Chỉ trả về nội dung, không giải thích.')

def run_alone(video, name, read_many):
    started = time.monotonic()
    frames = keyframes(video)
    todo = [(segment_id, image) for segment_id, image, _ in frames if image is not None]
    answers = read_many([image for _, image in todo])
    texts = {segment_id: (text, '') for (segment_id, _), text in zip(todo, answers)}
    for segment_id, image, source in frames:
        if image is None:
            texts[segment_id] = texts[source]
    results[video][name] = {'sec': round(time.monotonic() - started, 1), 'reads': len(todo), 'vlm_calls': 0,
                            'texts': texts}
    print(video, name, results[video][name]['sec'], 's', flush=True)

for video in VIDEOS:
    run_alone(video, 'qwen_only', lambda images: vlm_batch(images, WHOLE_PROMPT, batch_size=2, max_new_tokens=2048))

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    got_processor = AutoProcessor.from_pretrained('stepfun-ai/GOT-OCR-2.0-hf')
    got = AutoModelForImageTextToText.from_pretrained('stepfun-ai/GOT-OCR-2.0-hf', torch_dtype=torch.float16,
                                                      device_map='cuda:0').eval()
    def got_read(images):
        out = []
        for image in images:
            inputs = got_processor(image, return_tensors='pt', format=True).to('cuda:0')
            with torch.inference_mode():
                generated = got.generate(**inputs, do_sample=False, tokenizer=got_processor.tokenizer,
                                         stop_strings='<|im_end|>', max_new_tokens=4096)
            out.append(got_processor.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))
        return out
    for video in VIDEOS:
        run_alone(video, 'got', got_read)
    del got
    torch.cuda.empty_cache()
except Exception as error:
    print('GOT-OCR lỗi:', repr(error)[:500])
"""

FP16 = """fp16 = {}
slides = [image for _, image, _ in keyframes(VIDEOS[0]) if image is not None][:4]
model = hybrid.ocr_model
def time_unlimited():
    started = time.monotonic()
    texts = [hybrid.unlimited(image) for image in slides]
    return round((time.monotonic() - started) / len(slides), 1), texts
fp16['bf16_sec_per_slide'], bf16_texts = time_unlimited()
try:
    model.half()
    fp16['fp16_sec_per_slide'], fp16_texts = time_unlimited()
    fp16['fp16_similar'] = round(statistics.mean(difflib.SequenceMatcher(None, a, b).ratio()
                                                 for a, b in zip(bf16_texts, fp16_texts)), 4)
except Exception as error:
    fp16['fp16_error'] = repr(error)[:400]
finally:
    model.to(torch.bfloat16)
print(fp16)
"""

SCORE = """def similar(a, b):
    return difflib.SequenceMatcher(None, ' '.join(a.split()), ' '.join(b.split())).ratio()

def words(text):
    return re.findall(r'\\w+', text.lower())

def recall(base, other):
    base_words, other_words = words(base), set(words(other))
    return sum(w in other_words for w in base_words) / len(base_words) if base_words else 1.0

summary = []
for video in VIDEOS:
    base = results[video]['base']
    for name, result in results[video].items():
        common = [s for s in base['texts'] if s in result['texts']]
        sims = [similar('\\n'.join(base['texts'][s]), '\\n'.join(result['texts'][s])) for s in common]
        recs = [recall(base['texts'][s][0], '\\n'.join(result['texts'][s])) for s in common]
        summary.append({'video': video, 'variant': name, 'sec': result['sec'],
                        'speedup': round(base['sec'] / result['sec'], 2), 'reads': result['reads'],
                        'vlm_calls': result['vlm_calls'], 'segments': len(common),
                        'similar_mean': round(statistics.mean(sims), 4) if sims else None,
                        'similar_min': round(min(sims), 3) if sims else None,
                        'word_recall': round(statistics.mean(recs), 4) if recs else None})
totals = {}
for row in summary:
    total = totals.setdefault(row['variant'], {'variant': row['variant'], 'sec': 0.0, 'similar': [], 'recall': []})
    total['sec'] += row['sec']
    total['similar'].append(row['similar_mean'] or 0)
    total['recall'].append(row['word_recall'] or 0)
base_sec = totals['base']['sec']
overall = [{'variant': t['variant'], 'sec': round(t['sec'], 1), 'speedup': round(base_sec / t['sec'], 2),
            'similar_mean': round(statistics.mean(t['similar']), 4), 'word_recall': round(statistics.mean(t['recall']), 4)}
           for t in totals.values()]
json.dump({'summary': summary, 'overall': overall, 'fp16': fp16,
           'texts': {v: {n: r['texts'] for n, r in results[v].items()} for v in VIDEOS}},
          open('/kaggle/working/speed_bench3.json', 'w'), ensure_ascii=False, indent=1)
import pandas as pd
display(pd.DataFrame(overall))
display(pd.DataFrame(summary))
"""


VIDEOS = ["L25_V017", "L25_V035", "L25_V077"]


def build(video):
    cells = [cell("markdown", INTRO), cell("code", "%pip install -q " + fast.DEPS)]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in fast.MODULES]
    cells += [cell("markdown", "## Cấu hình"), cell("code", CONFIG.replace("__VIDEO__", repr(video))),
              cell("markdown", "## Bản pipeline: base, reuse_skip"), cell("code", PIPELINE),
              cell("markdown", "## Vòng lặp riêng: seq, batch, overlap, overlap_768"), cell("code", CUSTOM),
              cell("markdown", "## Một model đọc cả slide: qwen_only, got"), cell("code", ALONE),
              cell("markdown", "## Thử fp16 cho Unlimited-OCR"), cell("code", FP16),
              cell("markdown", "## So sánh"), cell("code", SCORE)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    slug = video[-4:].lower()
    (ROOT / "notebooks" / f"{NAME}_{slug}.ipynb").write_text(text, encoding="utf-8")
    run = ROOT / "runs" / f"{DATE}-speed-bench3-{slug}"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    json.dump({"id": f"{OWNER}/l25-speed-bench3-{slug}-{DATE}", "title": f"l25-speed-bench3-{slug}-{DATE}",
               "code_file": f"{NAME}.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
               "competition_sources": [], "kernel_sources": [f"{OWNER}/{SOURCE}"]},
              open(run / "kernel-metadata.json", "w"), indent=2)
    print(run / f"{NAME}.ipynb")


if __name__ == "__main__":
    for video in VIDEOS:
        build(video)
