"""Generate pipeline_bench.ipynb: which retrieval + answer setup to keep, scored on time and quality.

Questions: gold/qa/questions.json (L25_V008, MLS-15DA1; answer ranges in seconds, null = not in the
lecture). Indexes come from the 10-video rerun (l25-full-10-v2: E5 and E5 + dedupe_previous) and the
MLS-15DA1 run (l25-run-pipeline walkthrough export).
  retrieval   E5 (current) · E5 + previous-slide dedupe · bge-m3 re-embedding of the same chunks
              -> hit@1, hit@3 of the anchor slides, ms per question, index build seconds
  answer      Qwen3-4B-Instruct (fp16) and Qwen2.5-7B-Instruct (4-bit) on the E5 context
              -> answers to grade by hand, seconds per answer
Results: /kaggle/working/pipeline_bench.json

    python kaggle/lecture_rag/build_pipeline_bench.py
"""

from pathlib import Path
import base64
import json

from build_fast_pipeline import DEPS
from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "20261006-pipeline-bench"
OWNER = "thnhtrungnguynmif"
MODULES = ["common.py", "retrieval.py"]

SETUP = """import base64, copy, json, time, zipfile, numpy as np, torch
from pathlib import Path
QUESTIONS = json.loads(base64.b64decode(QUESTIONS_B64).decode('utf-8'))['videos']
inp = Path('/kaggle/input')
v008 = next(p for p in inp.rglob('L25_V008') if (p / '05_index').is_dir())
mls_zip = next(inp.rglob('walkthrough_export.zip'))
mls_dir = Path('/kaggle/working/mls'); zipfile.ZipFile(mls_zip).extractall(mls_dir)
# the walkthrough export leaves out the .wav clips: skip the asset checksum check (search needs no audio)
_load_stage = load_stage
def load_stage(directory, stage, video_id, verify=True):
    return _load_stage(directory, stage, video_id, verify=False)
INDEXES = {('L25_V008', 'e5'): v008 / '05_index', ('L25_V008', 'e5_dedupe'): v008 / 'dedupe' / '05_index',
           ('MLS-15DA1', 'e5'): mls_dir / '05_index'}
print({k: v.exists() for k, v in INDEXES.items()})
"""

RETRIEVE = """from sentence_transformers import SentenceTransformer
bge = SentenceTransformer('BAAI/bge-m3', device='cuda:0')
bge.max_seq_length = 512

def bge_engine(engine):
    started = time.monotonic()
    texts = [c.get('embedding_text') or c['text'] for c in engine['chunks']]
    vectors = bge.encode(texts, normalize_embeddings=True, batch_size=32, convert_to_numpy=True).astype('float32')
    manifest = copy.deepcopy(engine['manifest']); manifest['parameters']['query_prefix'] = ''
    return {**engine, 'model': bge, 'vectors': vectors, 'manifest': manifest}, round(time.monotonic() - started, 1)

def hits(engine, result, answer):
    by_id = {c['chunk_id']: c for c in engine['chunks']}
    seg = {s['segment_id']: s for s in engine['segments']}
    slides = []
    for cid in result['anchor_chunk_ids']:
        sid = by_id[cid]['segment_id']
        if sid not in slides:
            slides.append(sid)
    inside = [seg[s]['start_sec'] < answer[1] and seg[s]['end_sec'] > answer[0] for s in slides]
    return bool(inside[:1] and inside[0]), any(inside[:3])

rows, contexts = [], {}
engines = {}
for (video, name), path in INDEXES.items():
    engines[(video, name)] = load_retrieval_engine(path, video)
for video in ['L25_V008', 'MLS-15DA1']:
    engines[(video, 'bge_m3')], build = bge_engine(engines[(video, 'e5')])
    rows.append({'video': video, 'setup': 'bge_m3', 'index_build_sec': build})
for (video, name), engine in engines.items():
    h1 = h3 = n = 0; ms = []; out_scores = []
    for item in QUESTIONS[video]:
        current = item.get('current_time')
        started = time.monotonic()
        result = retrieve_question(engine, item['q'], current_time=current)
        ms.append(1000 * (time.monotonic() - started))
        if name == 'e5':
            contexts[(video, item['q'])] = result['context']
        if item['answer'] is None:
            out_scores.append(round(result['best_semantic_score'], 3)); continue
        a, b = hits(engine, result, item['answer']); h1 += a; h3 += b; n += 1
    rows.append({'video': video, 'setup': name, 'hit1': f'{h1}/{n}', 'hit3': f'{h3}/{n}',
                 'ms_per_q': round(sum(ms) / len(ms)), 'out_of_lecture_best_cos': out_scores})
    print(rows[-1], flush=True)
del bge; torch.cuda.empty_cache()
"""

ANSWER = """import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
del engines; _SHARED_MODELS.clear(); gc.collect(); torch.cuda.empty_cache()   # free the E5 models cached by shared_model
print('GPU trống:', [round(torch.cuda.mem_get_info(i)[0] / 2**30, 1) for i in range(torch.cuda.device_count())], 'GB')
SYSTEM = ('Bạn là trợ giảng. Chỉ dùng BẰNG CHỨNG được cho (trích từ slide và lời giảng, có mốc thời gian) để trả lời. '
          'Trả lời bằng đúng ngôn ngữ của câu hỏi, ngắn gọn, kèm mốc thời gian dạng mm:ss. '
          'Nếu bằng chứng không chứa câu trả lời, nói rõ là bài giảng không đề cập. '
          'Answer in the language of the question.')
MODELS = [('qwen3-4b', 'Qwen/Qwen3-4B-Instruct-2507', None),
          ('qwen2.5-7b-4bit', 'Qwen/Qwen2.5-7B-Instruct', BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16))]
MAX_CONTEXT = 4000   # T4: attention over ~9k tokens needs 5 GB at once; keep the first 4000 tokens of evidence
answers = []
for label, model_id, quant in MODELS:
    tok = AutoTokenizer.from_pretrained(model_id)

    def evidence(text):
        ids = tok(text, add_special_tokens=False)['input_ids']
        return text if len(ids) <= MAX_CONTEXT else tok.decode(ids[:MAX_CONTEXT]) + ' [...]'

    model =AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto', attn_implementation='sdpa',
                                                 max_memory={0: '4GiB', 1: '14GiB'}, quantization_config=quant).eval()
    for video, items in QUESTIONS.items():
        for item in items:
            msgs = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': f"BẰNG CHỨNG:\\n{evidence(contexts[(video, item['q'])])}\\n\\nCÂU HỎI: {item['q']}"}]
            inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(model.device)
            torch.cuda.synchronize(); started = time.monotonic()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
            torch.cuda.synchronize()
            answers.append({'model': label, 'video': video, 'q': item['q'], 'answer_range': item['answer'],
                            'seconds': round(time.monotonic() - started, 1), 'context_tokens': int(inputs['input_ids'].shape[1]),
                            'answer': tok.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()})
            print(label, answers[-1]['seconds'], 's |', item['q'][:50], '|', answers[-1]['answer'][:120].replace('\\n', ' '), flush=True)
            del out, inputs; torch.cuda.empty_cache()
    del model; gc.collect(); torch.cuda.empty_cache()
json.dump({'retrieval': rows, 'answers': answers}, open('/kaggle/working/pipeline_bench.json', 'w'), ensure_ascii=False, indent=1)
"""


def build():
    blob = base64.b64encode((ROOT / "gold" / "qa" / "questions.json").read_bytes()).decode()
    cells = [cell("markdown", "# So sánh pipeline · tìm kiếm + trả lời (thời gian và chất lượng)"),
             cell("code", "%pip install -q " + DEPS + ' "bitsandbytes>=0.46.1"')]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in MODULES]
    cells += [cell("code", f"QUESTIONS_B64 = '{blob}'", hidden=True), cell("code", SETUP),
              cell("markdown", "## Tìm kiếm: E5 · E5 bỏ trùng · bge-m3"), cell("code", RETRIEVE),
              cell("markdown", "## Trả lời: Qwen3-4B · Qwen2.5-7B 4-bit"), cell("code", ANSWER)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "pipeline_bench.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    json.dump({"id": f"{OWNER}/l25-pipeline-bench-20261006", "title": "l25-pipeline-bench-20261006",
               "code_file": "pipeline_bench.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
               "kernel_sources": [f"{OWNER}/l25-full-10-v2-20261006", f"{OWNER}/l25-run-pipeline"]},
              open(RUN / "kernel-metadata.json", "w"), indent=2)
    print(RUN / "pipeline_bench.ipynb")


if __name__ == "__main__":
    build()
