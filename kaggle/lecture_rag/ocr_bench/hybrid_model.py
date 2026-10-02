"""Hybrid entry for the OCR bench: Unlimited-OCR reads the whole slide (boxes, text,
LaTeX, tables); every image/chart region it finds is cropped and read by Qwen3-VL-4B.
Imported by build_ocr_bench.py and added to MODELS as "hybrid"."""

HYBRID = {
    "title": "Unlimited-OCR + Qwen3-VL (kết hợp)",
    "pip": '"transformers==4.57.1" "einops" "addict" "easydict" "accelerate>=1" qwen-vl-utils',
    "code": r'''
import contextlib, io, os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoModel, AutoTokenizer, AutoProcessor, Qwen3VLForConditionalGeneration

# Two T4s: Unlimited-OCR on the first, Qwen3-VL on the second (together they need ~17 GB).
OCR_GPU, VLM_GPU = "cuda:0", ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
ocr_tokenizer = AutoTokenizer.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True)
ocr_model = AutoModel.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True, use_safetensors=True,
                                      torch_dtype=torch.bfloat16, _attn_implementation="eager")
ocr_model = ocr_model.eval().to(OCR_GPU).to(torch.bfloat16)
vlm = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", torch_dtype=torch.float16, device_map=VLM_GPU, attn_implementation="sdpa").eval()
vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

LINE = re.compile(r"<\|det\|>(\S+)\s*\[([^\]]*)\]\s*<\|/det\|>")
VISUAL = ("image", "chart", "figure", "picture", "diagram")
CROP_PROMPT = """This image is one region cropped from a Vietnamese high-school lecture slide.
Decide what it is and return ONLY one JSON object:
- a chart: {"kind": "chart", "chart_type": ..., "title": ..., "axes": ..., "legend": [...], "data": [...every value you can read...]}
- a table (e.g. a variation / sign table, bảng biến thiên): {"kind": "table", "markdown": "...", "notes": "..."}
  (notes: only if arrows or curves are drawn inside the table, say where they go; otherwise "")
- a figure, graph of a function, map or diagram: {"kind": "figure", "description": "...", "labels": [...]}
- a photo of a person, a logo or decoration with no lesson content: {"kind": "decorative"}
Write every description and note in Vietnamese. Write formulas in LaTeX, keep Vietnamese
diacritics, never invent values you cannot read."""


def unlimited_raw(path):
    out = OUT / "hybrid_raw" / path.stem
    out.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    # infer() streams the raw output (with <|det|> boxes) to stdout; keep that copy.
    with contextlib.redirect_stdout(buffer):
        ocr_model.infer(ocr_tokenizer, prompt="<image>document parsing.", image_file=str(path),
                        output_path=str(out), base_size=1024, image_size=1024, crop_mode=False,
                        max_length=8192, no_repeat_ngram_size=35, ngram_window=128, save_results=True)
    text = buffer.getvalue()
    return text[text.find("<|det|>"):text.find("===============save results")] if "<|det|>" in text else text


def parse(raw):
    marks, blocks = list(LINE.finditer(raw)), []
    for n, mark in enumerate(marks):
        body = raw[mark.end(): marks[n + 1].start() if n + 1 < len(marks) else len(raw)].strip()
        try:
            values = [float(v) for v in mark.group(2).split(",")]
            box = clamp_box([v / 999 for v in values]) if len(values) == 4 else None
        except ValueError:
            box = None
        blocks.append({"type": mark.group(1), "bbox": box, "text": body})
    return blocks


def read_crop(image, box):
    w, h = image.size
    pad = 0.01
    crop = image.crop((max(0, (box[0] - pad) * w), max(0, (box[1] - pad) * h),
                       min(w, (box[2] + pad) * w), min(h, (box[3] + pad) * h)))
    messages = [{"role": "user", "content": [{"type": "image", "image": crop}, {"type": "text", "text": CROP_PROMPT}]}]
    inputs = vlm_processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(vlm.device)
    with torch.inference_mode():
        generated = vlm.generate(**inputs, max_new_tokens=1200, do_sample=False, repetition_penalty=1.05)
    raw = vlm_processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    try:
        return raw, json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        return raw, {"kind": "figure", "description": raw.strip()}


def run(image, path):
    raw = unlimited_raw(path)
    blocks, crops = parse(raw), []
    for block in blocks:
        label = block["type"].lower()
        # Captions are text: Unlimited-OCR already read them.
        if block["bbox"] and "caption" not in label and any(word in label for word in VISUAL):
            started = time.monotonic()
            crop_raw, result = read_crop(image, block["bbox"])
            kind = result.get("kind", "figure")
            if kind == "figure" and not str(result.get("description", "")).strip():
                kind = "decorative"  # nothing to describe, e.g. the lecturer's photo
            block["source_type"] = block["type"]
            block["type"] = {"chart": "chart", "table": "table", "decorative": "decorative"}.get(kind, "figure")
            if kind == "chart":
                block["text"] = json.dumps({k: v for k, v in result.items() if k != "kind"}, ensure_ascii=False)
            elif kind == "table":
                block["text"] = result.get("markdown", "") + ("\n" + result["notes"] if result.get("notes") else "")
            elif kind == "figure":
                block["text"] = result.get("description", "") + (
                    "\nNhãn: " + ", ".join(map(str, result["labels"])) if result.get("labels") else "")
            else:
                block["text"] = ""
            crops.append({"bbox": block["bbox"], "kind": kind, "seconds": round(time.monotonic() - started, 1),
                          "raw": crop_raw})
    return json.dumps({"unlimited": raw, "crops": crops}, ensure_ascii=False), blocks

run_all("Unlimited-OCR + Qwen3-VL", run)
''',
}
