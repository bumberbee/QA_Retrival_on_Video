"""Build one Kaggle notebook per OCR/VLM candidate, all run on the same 8 slides.

    python kaggle/lecture_rag/ocr_bench/build_ocr_bench.py <export_dir> <out_dir>

Every notebook writes /kaggle/working/ocr_bench/results.json with, per slide:
raw model output, parsed blocks [{type, bbox (0..1 of the slide), text}], seconds
and peak GPU memory. Images are embedded so no Kaggle dataset is needed.
"""

import base64
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

EXPORT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OWNER = "thnhtrungnguynmif"
SLIDES = [("L25_V088", i) for i in (5, 6, 10, 17, 18, 26)] + [("L25_V007", i) for i in (9, 16)]


DATASET = "l25-ocr-bench-slides"


def write_dataset():
    """Kernel sources must stay under 1 MB, so the slides go in a private dataset."""
    folder = OUT / "dataset"
    folder.mkdir(parents=True, exist_ok=True)
    for vid, index in SLIDES:
        interval = json.loads((EXPORT / vid / "01_slides" / "intervals.json").read_text(encoding="utf-8"))[index]
        with Image.open(EXPORT / vid / "01_slides" / interval["image_path"]) as image:
            image.convert("RGB").save(folder / f"{vid}_slide{index:02d}.png")
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": "L25 OCR bench slides", "id": f"{OWNER}/{DATASET}",
        "licenses": [{"name": "unknown"}]}, indent=2), encoding="utf-8")
    return folder


HARNESS = r'''
import base64, io, json, re, time, traceback
from pathlib import Path
import torch
from PIL import Image

OUT = Path("/kaggle/working/ocr_bench"); OUT.mkdir(parents=True, exist_ok=True)

SOURCE = next(Path("/kaggle/input").rglob("L25_V088_slide05.png")).parent
IMAGES = sorted(path.stem for path in SOURCE.glob("*.png"))

def load(name):
    return Image.open(SOURCE / f"{name}.png").convert("RGB")

def clamp_box(box):
    x1, y1, x2, y2 = [min(1.0, max(0.0, float(v))) for v in box]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

def run_all(model_name, run):
    results = []
    for name in IMAGES:
        image = load(name)
        path = OUT / f"{name}.jpg"; image.save(path, quality=95)
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        try:
            raw, blocks = run(image, path)
            error = None
        except Exception:
            raw, blocks, error = "", [], traceback.format_exc()[-3000:]
        seconds = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        results.append({"slide": name, "seconds": round(seconds, 2), "peak_gb": round(peak, 2),
                        "raw": raw, "blocks": blocks, "error": error})
        print("SLIDE", name, f"{seconds:.1f}s", f"{peak:.1f}GB", len(blocks), "blocks",
              "ERROR" if error else "", flush=True)
        if error: print(error, flush=True)
    (OUT / "results.json").write_text(json.dumps({"model": model_name, "slides": results},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print("DONE", model_name, flush=True)
'''

MODELS = {
    "qwen3vl": {
        "title": "Qwen3-VL-4B-Instruct",
        "pip": '"transformers>=4.57,<5" "accelerate>=1" qwen-vl-utils',
        "code": r'''
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa").eval()
processor = AutoProcessor.from_pretrained(MODEL)
PROMPT = """This is a slide from a Vietnamese high-school lecture video. Parse EVERY content region on it.
Return only a JSON array. Each element: {"type": one of "title","text","formula","table","chart","figure","diagram",
"bbox_2d": [x1, y1, x2, y2] with coordinates relative to the image on a 0-1000 scale, "content": ...}.
Rules for content: copy text exactly with Vietnamese diacritics; write every formula in LaTeX;
tables as Markdown; for a chart give its type, title, axis labels, legend and every data value you can read;
for a figure/diagram/graph describe what it shows (shapes, labels, key points). Do not translate. Do not invent values."""

def parse_json_array(text):
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    return json.loads(text[start:end + 1])

def run(image, path):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=3000, do_sample=False)
    raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    blocks = []
    try:
        for item in parse_json_array(raw):
            box = item.get("bbox_2d") or item.get("bbox")
            content = item.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            blocks.append({"type": item.get("type", "text"),
                           "bbox": clamp_box([v / 1000 for v in box]) if box and len(box) == 4 else None,
                           "text": content})
    except Exception:
        pass
    return raw, blocks

run_all("Qwen3-VL-4B-Instruct", run)
''',
    },
    "dotsocr": {
        "title": "dots.ocr",
        # dots.ocr's processor breaks on newer transformers (video_processor); the repo pins 4.51.3.
        "pip": '"transformers==4.51.3" "accelerate>=1" qwen-vl-utils huggingface_hub',
        "code": r'''
import sys
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor
# The repo name has a dot; trust_remote_code needs an importable folder name.
path = snapshot_download("rednote-hilab/dots.ocr", local_dir="/kaggle/working/DotsOCR")
sys.path.insert(0, "/kaggle/working")
# The vision tower casts pixels to bfloat16, so the weights must be bfloat16 too.
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="sdpa", trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

def run(image, path):
    messages = [{"role": "user", "content": [{"type": "image", "image": str(path)}, {"type": "text", "text": PROMPT}]}]
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=6000, do_sample=False)
    raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    # bbox is in the resized image the model saw (patch 14 grid).
    t, gh, gw = inputs["image_grid_thw"][0].tolist()
    height, width = gh * 14, gw * 14
    blocks = []
    try:
        data = json.loads(raw[raw.find("["): raw.rfind("]") + 1]) if raw.strip().startswith("[") else json.loads(raw)
        if isinstance(data, dict):
            data = data.get("layout", data.get("elements", list(data.values())[0]))
        for item in data:
            x1, y1, x2, y2 = item["bbox"]
            blocks.append({"type": item.get("category", "Text"),
                           "bbox": clamp_box([x1 / width, y1 / height, x2 / width, y2 / height]),
                           "text": item.get("text", "")})
    except Exception:
        pass
    return raw, blocks

run_all("dots.ocr", run)
''',
    },
    "unlimited": {
        "title": "baidu Unlimited-OCR",
        "pip": '"transformers==4.57.1" "einops" "addict" "easydict" "accelerate>=1"',
        "code": r'''
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoModel, AutoTokenizer
MODEL = "baidu/Unlimited-OCR"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
# The remote code builds image features in bfloat16 (as in the model card); float16 made
# masked_scatter_ fail with a Half/Float mismatch. T4 runs bf16, just without tensor cores.
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, use_safetensors=True,
                                  torch_dtype=torch.bfloat16, _attn_implementation="eager")
model = model.eval().cuda().to(torch.bfloat16)
DET = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(\[.*?\])<\|/det\|>", re.S)
DET2 = re.compile(r"<\|det\|>([^<\s\[]+)\s*(\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*)?\s*<\|/det\|>")

def parse(raw):
    blocks = []
    marks = list(DET.finditer(raw)) or list(DET2.finditer(raw))
    for n, mark in enumerate(marks):
        body = raw[mark.end(): marks[n + 1].start() if n + 1 < len(marks) else len(raw)].strip()
        box = None
        try:
            values = json.loads(mark.group(2)) if mark.group(2) else None
            if values and isinstance(values[0], list): values = values[0]
            if values and len(values) == 4: box = clamp_box([v / 999 for v in values])
        except Exception:
            pass
        blocks.append({"type": mark.group(1).strip(), "bbox": box, "text": body})
    return blocks

def run(image, path):
    out = OUT / "unlimited_raw" / path.stem
    out.mkdir(parents=True, exist_ok=True)
    result = model.infer(tokenizer, prompt="<image>document parsing.", image_file=str(path),
                         # "Base" mode: one 1024 view. The "gundam" crop mode ran out of
                         # memory on a 16 GB T4 (attention over all crops at once).
                         output_path=str(out), base_size=1024, image_size=1024, crop_mode=False,
                         max_length=8192, no_repeat_ngram_size=35, ngram_window=128,
                         save_results=True)
    raw = result if isinstance(result, str) and result.strip() else ""
    if not raw:
        for file in sorted(out.rglob("*")):
            if file.suffix in (".mmd", ".md", ".txt") and file.is_file():
                raw = file.read_text(encoding="utf-8", errors="replace"); break
    torch.cuda.empty_cache()
    return raw, parse(raw)

run_all("baidu/Unlimited-OCR", run)
''',
    },
    "paddlevl": {
        "title": "PaddleOCR-VL-1.5",
        "pip": '"transformers>=5.0.0" "accelerate>=1"',
        "code": r'''
from transformers import AutoModelForImageTextToText, AutoProcessor
MODEL = "PaddlePaddle/PaddleOCR-VL-1.5"
model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
processor = AutoProcessor.from_pretrained(MODEL)
LOC = re.compile(r"<\|LOC_(\d+)\|>")

def ask(image, task):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": task}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    return processor.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)

def run(image, path):
    # Spotting = text with location tokens; OCR / Formula / Chart give the recognised content.
    raw = {task: ask(image, task) for task in ("Spotting:", "OCR:", "Formula Recognition:", "Chart Recognition:")}
    blocks = []
    for line in raw["Spotting:"].split("\n"):
        values = [int(v) for v in LOC.findall(line)]
        text = LOC.sub("", line).replace("</s>", "").strip()
        if text and len(values) >= 8:   # 4 corner points on a 0-1000 grid
            xs, ys = values[0::2], values[1::2]
            blocks.append({"type": "text", "bbox": clamp_box([min(xs) / 1000, min(ys) / 1000, max(xs) / 1000, max(ys) / 1000]), "text": text})
        elif text and len(values) == 4:
            blocks.append({"type": "text", "bbox": clamp_box([v / 1000 for v in values]), "text": text})
    for task, kind in (("Formula Recognition:", "formula"), ("Chart Recognition:", "chart")):
        content = raw[task].replace("</s>", "").strip()
        if content:
            blocks.append({"type": kind, "bbox": None, "text": content})
    return json.dumps(raw, ensure_ascii=False), blocks

run_all("PaddleOCR-VL-1.5", run)
''',
    },
}


sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_model import HYBRID  # noqa: E402

MODELS["hybrid"] = HYBRID


def cell(kind, source):
    value = {"cell_type": kind, "metadata": {}, "source": source.strip("\n").splitlines(keepends=True),
             "id": hashlib.sha256((kind + source).encode()).hexdigest()[:12]}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def main():
    print(write_dataset())
    for key, spec in MODELS.items():
        folder = OUT / key
        folder.mkdir(parents=True, exist_ok=True)
        notebook = {"cells": [
            cell("markdown", f"# OCR bench · {spec['title']}\n\n8 slide (6 toán V088, 2 biểu đồ V007), GPU T4."),
            cell("code", f"%pip install -q {spec['pip']}"),
            cell("code", HARNESS),
            cell("code", spec["code"]),
        ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}
        (folder / f"ocr_bench_{key}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
        slug = f"l25-ocr-bench-{key}-20260929"
        (folder / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{OWNER}/{slug}", "title": slug, "code_file": f"ocr_bench_{key}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{OWNER}/{DATASET}"], "competition_sources": [],
            "kernel_sources": []}, indent=2), encoding="utf-8")
        print(folder)


if __name__ == "__main__":
    main()
