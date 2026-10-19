"""Generate llm_judge.ipynb: content filter, LLM judge step 2. An instruction LLM on a Kaggle T4
reads every suspect stretch of runs/20261006-llm-judge/candidates.json (gold/build_llm_candidates.py)
next to the lecture's most typical speech and labels it lesson / ad / channel / off_topic.

    python kaggle/lecture_rag/build_llm_judge.py
"""

from pathlib import Path
import base64
import json

from build_notebooks import cell

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "20261006-llm-judge"
OWNER = "thnhtrungnguynmif"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

PROMPT = '''You check stretches of speech from a lecture video. Decide if the stretch is part of the teaching.

Typical speech of this lecture (for its topic):
"""{topic}"""

Stretch {start}-{end} s:
"""{text}"""

Labels:
- lesson: explains or discusses the subject, examples, exercises, the lecturer's comments about the lesson (even informal or in passing)
- ad: promotes a course, school, university, product, service or event; asks to register, buy, call, visit a link.
  Promotion that talks about education is still an ad: a student or alumnus praising a school ("Để có được thành
  công như hôm nay ... bắt đầu từ Trường Đại học ..."), a school's programs, campus, scholarships or careers
  ("đào tạo đa dạng", "cơ hội nghề nghiệp"), "đăng ký ngay", "hotline", "ưu đãi học phí".
  Teaching the lecture's own subject is never an ad.
- channel: greeting or goodbye of the channel/show, asks to like/subscribe/follow, sponsor read, without teaching
- off_topic: unrelated talk (music only, chit-chat with no lesson content)
The speech may be Vietnamese or English and contains speech-recognition errors.
Answer with ONE JSON object only: {{"label": "...", "reason": "<= 15 words"}}'''

CODE = """import base64, json, re, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
videos = json.loads(base64.b64decode(CANDIDATES_B64).decode('utf-8'))
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map='cuda:0').eval()
started, results = time.monotonic(), []
for video in videos:
    for item in video['candidates']:
        prompt = PROMPT.format(topic=video['topic'], start=round(item['start_sec']), end=round(item['end_sec']), text=item['text'] or '(no speech)')
        inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True,
                                               return_tensors='pt', return_dict=True).to('cuda:0')
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        raw = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        match = re.search(r'\\{.*\\}', raw, re.S)
        try:
            answer = json.loads(match.group(0)) if match else {}
        except ValueError:
            answer = {}
        results.append({'video': video['video'], 'start_sec': item['start_sec'], 'end_sec': item['end_sec'],
                        'label': answer.get('label', 'unparsed'), 'reason': answer.get('reason', raw[:120])})
        print(results[-1], flush=True)
json.dump({'model': MODEL, 'seconds': round(time.monotonic() - started), 'results': results},
          open('/kaggle/working/llm_judge.json', 'w'), ensure_ascii=False, indent=1)
"""


def build():
    blob = base64.b64encode((RUN / "candidates.json").read_bytes()).decode()
    cells = [cell("markdown", "# Lọc nội dung · LLM đọc các đoạn nghi vấn\n\nMỗi đoạn được gán nhãn lesson / ad / channel / off_topic."),
             cell("code", '%pip install -q "transformers==4.57.1" "accelerate>=1"'),
             cell("code", f"CANDIDATES_B64 = '{blob}'", hidden=True),
             cell("code", f"MODEL = {MODEL!r}\nPROMPT = {PROMPT!r}"),
             cell("code", CODE)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    (RUN / "llm_judge.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    json.dump({"id": f"{OWNER}/l25-llm-judge-20261006", "title": "l25-llm-judge-20261006",
               "code_file": "llm_judge.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
              open(RUN / "kernel-metadata.json", "w"), indent=2)
    print(RUN / "llm_judge.ipynb")


if __name__ == "__main__":
    build()
