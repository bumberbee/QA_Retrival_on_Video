"""Generate chapter_bench.ipynb: proposal (b) of the redesign, tested on the 31 content-filter videos.

  03a chapters   60 s transcript blocks (as the reference repo) embedded with E5 and bge-m3; a new
                 chapter starts where the cosine of two neighbouring blocks is in the video's lowest
                 CUT_SHARE (relative threshold, every channel has its own scale); chapters < 2 blocks merge.
  03b LLM        Qwen3-4B-Instruct reads every chapter: title, 1-2 sentence summary and a label
                 lesson / ad / channel / off_topic (the whole transcript is read, not only suspects).
Output chapter_bench.json: per video and embedding, the chapters with times, label, title, summary and
the seconds spent (embedding, LLM). Scored by gold/eval_chapters.py.

    python kaggle/lecture_rag/build_chapter_bench.py
"""

from pathlib import Path
import base64
import json
import zlib

from build_notebooks import cell

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RUN = RUNS / "20261006-chapter-bench"
OWNER = "thnhtrungnguynmif"
BLOCK_SEC = 60


def transcripts():
    """{video: {"language", "blocks": [{start, end, text}]}} from every export with a transcript."""
    sources = [RUNS / "20261006-full-10-v2/outputs/lecture_rag", RUNS / "20261005-run-pipeline/export", RUNS / "asr-all"]
    wanted = {p.stem for p in (RUNS / "20261005-signal-dump/outputs/signals").glob("*.npz")}
    out = {}
    for source in sources:
        for path in sorted(source.glob("*/03_audio/transcript.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data["video_id"]
            if name not in wanted or name in out:
                continue
            blocks = {}
            for word in data["words"]:
                blocks.setdefault(int(word["start_sec"] // BLOCK_SEC), []).append(word["text"].strip())
            out[name] = {"language": data.get("language"),
                         "blocks": [{"start": k * BLOCK_SEC, "end": (k + 1) * BLOCK_SEC, "text": " ".join(v)}
                                    for k, v in sorted(blocks.items())]}
    return out


CONFIG = """import base64, json, re, time, zlib, numpy as np, torch
VIDEOS = json.loads(zlib.decompress(base64.b64decode(VIDEOS_B64)).decode('utf-8'))
CUT_SHARE = 0.2       # neighbour pairs in the video's lowest 20 % of cosine start a chapter
MIN_BLOCKS = 2
EMBEDDERS = {'e5': ('intfloat/multilingual-e5-small', 'passage: '), 'bge_m3': ('BAAI/bge-m3', '')}
LLM = 'Qwen/Qwen3-4B-Instruct-2507'
print(len(VIDEOS), 'video')
"""

CHAPTERS = """from sentence_transformers import SentenceTransformer

def chapters_from(vectors):
    sims = (vectors[:-1] * vectors[1:]).sum(1)
    if len(sims) == 0:
        return [[0, len(vectors)]]
    cut = np.quantile(sims, CUT_SHARE)
    starts = [0] + [i + 1 for i, s in enumerate(sims) if s <= cut]
    spans = [[a, b] for a, b in zip(starts, starts[1:] + [len(vectors)])]
    merged = []
    for span in spans:                       # a chapter of < MIN_BLOCKS blocks joins the previous one
        if merged and span[1] - span[0] < MIN_BLOCKS:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    return merged

SEGMENTATION = {}
for name, (model_id, prefix) in EMBEDDERS.items():
    model = SentenceTransformer(model_id, device='cuda:0')
    for video, data in VIDEOS.items():
        started = time.monotonic()
        vectors = model.encode([prefix + b['text'] for b in data['blocks']], normalize_embeddings=True, convert_to_numpy=True)
        SEGMENTATION[(video, name)] = {'spans': chapters_from(vectors), 'embed_sec': round(time.monotonic() - started, 2)}
    print(name, 'xong', flush=True)
    del model; torch.cuda.empty_cache()
"""

LLM_CELL = """from transformers import AutoModelForCausalLM, AutoTokenizer
PROMPT = '''Below is one chapter ({start}-{end} s) of the transcript of a lecture video (speech recognition, may contain errors; Vietnamese or English).
Return ONE JSON object only:
{{"title": "<short chapter title in the language of the speech>",
  "summary": "<1-2 sentences in the language of the speech>",
  "label": "lesson | ad | channel | off_topic"}}
label: lesson = teaches or discusses the subject (examples, exercises, informal comments included);
ad = promotes a course, school, university, product, service or event (testimonials praising a school count as ad);
channel = greeting/goodbye of the channel or show, like/subscribe, sponsor read, without teaching;
off_topic = unrelated talk. If the chapter mixes lesson and ad, choose the label of most of its time.

Transcript:
\"\"\"{text}\"\"\"'''
tok = AutoTokenizer.from_pretrained(LLM)
llm = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.float16, device_map='cuda:0').eval()
results = []
for (video, embedder), seg in SEGMENTATION.items():
    blocks = VIDEOS[video]['blocks']
    chapters, started = [], time.monotonic()
    for a, b in seg['spans']:
        text = ' '.join(x['text'] for x in blocks[a:b])[:6000]
        msgs = [{'role': 'user', 'content': PROMPT.format(start=blocks[a]['start'], end=blocks[b - 1]['end'], text=text)}]
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=True).to('cuda:0')
        with torch.inference_mode():
            out = llm.generate(**inputs, max_new_tokens=160, do_sample=False)
        raw = tok.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        match = re.search(r'\\{.*\\}', raw, re.S)
        try:
            answer = json.loads(match.group(0)) if match else {}
        except ValueError:
            answer = {}
        chapters.append({'start_sec': blocks[a]['start'], 'end_sec': blocks[b - 1]['end'], 'blocks': b - a,
                         'label': answer.get('label', 'unparsed'), 'title': answer.get('title', ''), 'summary': answer.get('summary', raw[:200])})
    results.append({'video': video, 'embedder': embedder, 'embed_sec': seg['embed_sec'],
                    'llm_sec': round(time.monotonic() - started, 1), 'chapters': chapters})
    print(video, embedder, len(chapters), 'chương', results[-1]['llm_sec'], 's', [c['label'] for c in chapters], flush=True)
json.dump(results, open('/kaggle/working/chapter_bench.json', 'w'), ensure_ascii=False, indent=1)
"""


def build():
    videos = transcripts()
    blob = base64.b64encode(zlib.compress(json.dumps(videos, ensure_ascii=False).encode("utf-8"), 9)).decode()
    cells = [cell("markdown", "# Đề xuất (b): chia chương + LLM tóm tắt và gán nhãn trên toàn bộ transcript"),
             cell("code", '%pip install -q "transformers==4.57.1" "accelerate>=1" "sentence-transformers>=3.4,<6"'),
             cell("code", f"VIDEOS_B64 = '{blob}'", hidden=True), cell("code", CONFIG),
             cell("markdown", "## 03a · Chia chương (E5 và bge-m3)"), cell("code", CHAPTERS),
             cell("markdown", "## 03b · LLM: tên, tóm tắt, nhãn từng chương"), cell("code", LLM_CELL)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "chapter_bench.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    json.dump({"id": f"{OWNER}/l25-chapter-bench-20261006", "title": "l25-chapter-bench-20261006",
               "code_file": "chapter_bench.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
              open(RUN / "kernel-metadata.json", "w"), indent=2)
    print(len(videos), "video ·", sum(len(v["blocks"]) for v in videos.values()), "block ·", round(len(blob) / 1e6, 2), "MB")


if __name__ == "__main__":
    build()
