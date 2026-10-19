"""Generate asr_dump notebooks: stage 03 (ASR) only, for the content-filter videos that have a
picture-signal dump (signal_dump) but no transcript, split into shards run as separate Kaggle
sessions (2 per account at a time). Each session runs two videos at once, one per T4.

Output per shard: asr_<shard>.zip with <video>/03_audio/{transcript,manifest}.json, read by
gold/eval_slide_speech.py together with the signal dump.

    python kaggle/lecture_rag/build_asr_dump.py
"""

from pathlib import Path
import json

from build_fast_pipeline import DEPS
from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
DATE = "20261006"
MODULES = ["common.py", "transcription.py"]
ACCOUNTS = ["thnhtrungnguynmif", "thnhtrungnguynmif", "tuvo6804", "tuvo6804"]
DATASETS = ["l25a-lecture-videos-media", "vn-lecture-test12", "lpm-videos", "m3av-maths", "slidespeech-train100"]
# (video, language, minutes) -- the 23 videos of ad_flag_check without a transcript
VIDEOS = [("MLS-03E08", "en", 37.7), ("MLS-2A8E1", "en", 17.3), ("MLS-C00E5", "en", 15.8),
          ("IIfeX44RI9k", "en", 28.9), ("dC93OcrNX6c", "en", 29.5), ("Xee8gmppUIw", "en", 19.9),
          ("7EsHL8Sacg0", "en", 16.8), ("cs_0001", "en", 21.8), ("history_0000", "en", 16.8),
          ("agriculture_0002", "en", 18.8), ("music_0000", "en", 20.1),
          ("tHE1mC_n0-E", "vi", 43.9), ("D_kuulxKuxc", "vi", 31.6), ("bstCZZUMJYw", "vi", 45.0),
          ("_eDGTHhyQ7M", "vi", 43.5), ("8WnDooAH2a4", "vi", 45.0), ("szRBlpJ0kkY", "vi", 45.0),
          ("laT0CKatosY", "vi", 39.9), ("xysJI15jpAw", "vi", 45.0), ("v2YTz8ToEag", "vi", 45.0),
          ("hVU5Mw19V_c", "vi", 30.6), ("xLGGWMDJRkY", "vi", 45.0), ("y1CvB950VlU", "vi", 13.3)]


def shards(count):
    """Longest video first into the lightest shard, so every session ends at about the same time."""
    groups = [[] for _ in range(count)]
    for video in sorted(VIDEOS, key=lambda v: -v[2]):
        min(groups, key=lambda g: sum(v[2] for v in g)).append(video)
    return groups

RUN = """import json, shutil, threading, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

OUT = Path('/kaggle/working/asr')
videos = {p.stem: p for p in Path('/kaggle/input').rglob('*.mp4')}
rows, lock = [], threading.Lock()

def work(item, device_index):
    name, language, _ = item
    started = time.monotonic()
    try:
        run_transcription(None, videos[name], OUT / name, name, model_id=ASR_MODEL, revision=ASR_REVISION,
                          language=language, transcription_mode='windowed', device_index=device_index)
        row = {'video': name, 'gpu': device_index, 'seconds': round(time.monotonic() - started)}
    except Exception:
        traceback.print_exc(); row = {'video': name, 'error': True}
    with lock:
        rows.append(row); print(row, flush=True)

def lane(device_index):
    for item in VIDEOS[device_index::2]:   # GPU 0 takes videos 0, 2, 4 ...; GPU 1 the others
        work(item, device_index)

with ThreadPoolExecutor(2) as pool:
    list(pool.map(lane, [0, 1]))
export = Path('/kaggle/working/export')
for row in rows:
    if 'error' not in row:
        for name in ('transcript.json', 'manifest.json'):
            target = export / row['video'] / '03_audio'; target.mkdir(parents=True, exist_ok=True)
            shutil.copy(OUT / row['video'] / '03_audio' / name, target / name)
shutil.make_archive(f'/kaggle/working/asr_{SHARD}', 'zip', export)
print(rows)
"""


def build():
    for number, group in enumerate(shards(len(ACCOUNTS))):
        owner = ACCOUNTS[number]
        config = (f"SHARD = {number}\nVIDEOS = {group!r}\n"
                  "ASR_MODEL = 'Systran/faster-whisper-medium'\n"
                  "ASR_REVISION = '08e178d48790749d25932bbc082711ddcfdfbc4f'\n")
        cells = [cell("markdown", f"# ASR cho lọc nội dung · nhóm {number} ({sum(v[2] for v in group):.0f} phút video)"),
                 cell("code", "%pip install -q " + DEPS)]
        cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in MODULES]
        cells += [cell("code", config), cell("code", RUN)]
        notebook = {"cells": cells,
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python"}},
                    "nbformat": 4, "nbformat_minor": 5}
        run = ROOT / "runs" / f"{DATE}-asr-dump-{number}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "asr_dump.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        json.dump({"id": f"{owner}/l25-asr-dump-{number}-{DATE}", "title": f"l25-asr-dump-{number}-{DATE}",
                   "code_file": "asr_dump.ipynb", "language": "python", "kernel_type": "notebook",
                   "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                   "machine_shape": "NvidiaTeslaT4",
                   "dataset_sources": [f"thnhtrungnguynmif/{d}" for d in DATASETS],
                   "competition_sources": [], "kernel_sources": []},
                  open(run / "kernel-metadata.json", "w"), indent=2)
        print(run.name, owner, f"{sum(v[2] for v in group):.0f} phút", [v[0] for v in group])


if __name__ == "__main__":
    build()
