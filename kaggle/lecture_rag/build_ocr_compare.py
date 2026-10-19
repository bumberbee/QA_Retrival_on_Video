"""Generate ocr_compare.ipynb: stage 02 read two ways on 2 Vietnamese + 2 English videos.

  qwen    Qwen3-VL-4B reads the whole slide in one call (text, LaTeX, tables, one "[Hình]"/"[Figure]"
          line for figures), one model copy per GPU
  hybrid  stage 02 as configured today (Unlimited-OCR + Qwen3-VL on figure regions, with the speed-ups)
Vietnamese: stage-01 output of L25_V037 and vn12 hVU5Mw19V_c. English: SlideSpeech design_0004 and
medical_0000, stage 01 run here. Output ocr_compare.json: seconds and each segment's text for both,
plus small keyframes, for a local HTML page.

    python kaggle/lecture_rag/build_ocr_compare.py
"""

from pathlib import Path
import json

import build_fast_pipeline as fast
from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
NAME = "ocr_compare"
DATE = "20261013"
OWNER = "thivntrung"
DATASET = "thivntrung/ocr-compare-4videos"

INTRO = """# So sánh OCR bước 02 trên 4 video (2 tiếng Việt, 2 tiếng Anh)

**qwen**: Qwen3-VL-4B đọc cả slide một lần, mỗi GPU một bản. **hybrid**: bước 02 hiện tại
(Unlimited-OCR + Qwen3-VL vùng hình, đã bật tăng tốc). Cùng keyframe, cùng bộ gác."""

RUN = """import base64, gc, io, json, threading, time, zipfile, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
INPUT, WORK = Path('/kaggle/input'), Path('/kaggle/working')
for z in INPUT.rglob('*.zip'):
    zipfile.ZipFile(z).extractall(WORK / 'unzipped')
ROOTS = (INPUT, WORK / 'unzipped')
DEVICES = ('cuda:0', 'cuda:1')
PROMPT = {'vi': ('Đọc toàn bộ chữ trên slide bài giảng này theo thứ tự đọc. Giữ đúng dấu tiếng Việt, chép đúng như trên slide, không tự sửa. '
                 'Công thức viết LaTeX, bảng viết Markdown. Bỏ logo kênh, số điện thoại quảng cáo, phụ đề video. '
                 'Nếu có biểu đồ, sơ đồ hoặc hình minh họa, thêm cuối cùng một dòng "[Hình] " mô tả ngắn nội dung hình. Chỉ trả về nội dung.'),
          'en': ('Read all the text on this lecture slide in reading order, exactly as written, without correcting it. '
                 'Write formulas in LaTeX and tables in Markdown. Skip channel logos and video subtitles. '
                 'If there is a chart, diagram or picture, add one last line "[Figure] " with a short description. Return only the content.')}

videos = []   # (video_id, language, stage01 dir)
for name in ('L25_V037', 'hVU5Mw19V_c'):
    stage01 = next(p.parent for r in ROOTS for p in r.rglob('intervals.json') if p.parent.name == f'{name}__01_slides')
    videos.append((name, 'vi', stage01))
for name in ('design_0004', 'medical_0000'):
    mp4 = next(p for r in ROOTS for p in r.rglob(f'{name}.mp4'))
    stage01, manifest, _ = run_slide_detection(mp4, WORK / 'stage01' / name, name, sample_hz=1.0, fine_hz=8.0,
                                               merge_reveals=True, scan='single_pass')
    videos.append((name, 'en', stage01))
print(videos, flush=True)
ocr_gate = load_ocr_gate('cuda:0', threshold=0.15, views='frame+crops')

def keyframes(stage01):
    intervals = json.load(open(stage01 / 'intervals.json', encoding='utf-8'))
    return [(it, Image.open(relative_asset(stage01, it['image_path'])).convert('RGB')) for it in intervals]

report = {}
for name, lang, stage01 in videos:
    frames = keyframes(stage01)
    thumbs = {}
    for it, image in frames:
        small = image.copy(); small.thumbnail((640, 360)); buffer = io.BytesIO(); small.save(buffer, 'JPEG', quality=75)
        thumbs[it['segment_id']] = {'start': it.get('start_sec', 0), 'jpg': base64.b64encode(buffer.getvalue()).decode()}
    report[name] = {'lang': lang, 'thumbs': thumbs, 'order': [it['segment_id'] for it, _ in frames]}

# 1) qwen: one copy per GPU, slides split between them
processor = AutoProcessor.from_pretrained(HYBRID_VLM_MODEL)
models = [Qwen3VLForConditionalGeneration.from_pretrained(HYBRID_VLM_MODEL, torch_dtype=torch.float16,
          device_map=d, attn_implementation='sdpa').eval() for d in DEVICES]
def qwen_read(model, image, lang):
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT[lang]}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=1536, do_sample=False, repetition_penalty=1.05)
    return processor.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
warm = keyframes(videos[0][2])[0][1]
for m in models: qwen_read(m, warm, 'vi')
for name, lang, stage01 in videos:
    started = time.monotonic()
    segs, jobs = {}, []
    for it, image in keyframes(stage01):
        skip, _ = gate_keyframe(ocr_gate, image)
        if skip:
            segs[it['segment_id']] = {'status': 'skipped_no_content', 'text': ''}
        else:
            jobs.append((it['segment_id'], image))
    def worker(k):
        for sid, image in jobs[k::2]:
            t = time.monotonic(); text = qwen_read(models[k], image, lang)
            segs[sid] = {'status': 'read', 'text': text, 'sec': round(time.monotonic() - t, 1)}
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    report[name]['qwen'] = {'sec': round(time.monotonic() - started, 1), 'segments': segs}
    print(name, 'qwen', report[name]['qwen']['sec'], 's', flush=True)
json.dump(report, open(WORK / 'ocr_compare.json', 'w'), ensure_ascii=False)   # qwen results kept even if hybrid fails
del models, m, warm; gc.collect(); torch.cuda.empty_cache()
print({d: round(torch.cuda.memory_allocated(d) / 2**30, 2) for d in DEVICES}, 'GiB còn giữ', flush=True)

# 2) hybrid as configured today
shared_model(('hybrid_ocr', DEVICES), lambda: load_hybrid_ocr(*DEVICES))
for name, lang, stage01 in videos:
    started = time.monotonic()
    _, _, records = run_visual_extraction(
        stage01, WORK / 'hybrid' / name, name, languages=('vi', 'en') if lang == 'vi' else ('en',),
        describe_language=lang, ocr_device='cuda', ocr_engine='hybrid', hybrid_devices=DEVICES, ocr_gate=ocr_gate,
        hybrid_pipelined=True, reuse_near_duplicates=True, hybrid_min_region_area=0.02, hybrid_skip_repeated_decor=True)
    report[name]['hybrid'] = {'sec': round(time.monotonic() - started, 1), 'segments': {
        r['segment_id']: {'status': r['ocr_status'], 'text': chr(10).join([r['ocr_text']] + [
            f"[{g['kind']}] {g.get('text', '')} {g.get('description', '')}".strip() for g in r['regions']])} for r in records}}
    print(name, 'hybrid', report[name]['hybrid']['sec'], 's', flush=True)
    json.dump(report, open(WORK / 'ocr_compare.json', 'w'), ensure_ascii=False)
"""


def build():
    cells = [cell("markdown", INTRO), cell("code", "%pip install -q " + fast.DEPS)]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in fast.MODULES]
    cells += [cell("markdown", "## Chạy qwen / hybrid trên 4 video"), cell("code", RUN)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    (ROOT / "notebooks" / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    run = ROOT / "runs" / f"{DATE}-ocr-compare"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    json.dump({"id": f"{OWNER}/ocr-compare-{DATE}", "title": f"ocr-compare-{DATE}",
               "code_file": f"{NAME}.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET],
               "competition_sources": [], "kernel_sources": []},
              open(run / "kernel-metadata.json", "w"), indent=2)
    print(run / f"{NAME}.ipynb")


if __name__ == "__main__":
    build()
