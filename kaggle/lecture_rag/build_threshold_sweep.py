"""Generate threshold_sweep notebooks: stage 01 (slide detection) on the 10 L25 videos for a grid of
change_threshold x stable_samples, exactly as the pipeline runs it (single-pass scan, 8 Hz refine,
reveal merging, no ad filter). CPU only, split over three Kaggle accounts; every config's
intervals.json is exported and scored locally with gold/eval_ad_detection.py (gold set 1:
choose on the 4 tuning videos, report on the 6 held-out ones).

    python kaggle/lecture_rag/build_threshold_sweep.py
"""

from pathlib import Path
import json

from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
DATE = "20261006"
CHANGE = [0.035, 0.045, 0.055, 0.07, 0.09]
STABLE = [2, 3, 4]          # stable_samples must be >= 2 (detection.py)
ACCOUNTS = ["thnhtrungnguynmif", "tuvo6804", "thivntrung"]
MODULES = ["common.py", "detection.py"]
DEPS = '"av>=14,<17" "opencv-python-headless>=4.10,<4.12" "Pillow>=10"'

# L25: gold set 1 (hand-checked boundaries). LPM: the test speakers' author slide times (slides.csv);
# all 3 anat-2 videos and the 5 shortest bio-3 videos (the other bio-3 lectures are 60-84 min).
VIDEO_SETS = {"l25": ("thnhtrungnguynmif/l25a-lecture-videos-media", [f"L25_V{n:03d}" for n in range(1, 11)]),
              "lpm": ("thnhtrungnguynmif/lpm-videos", ["ALRJCeVT0fQ", "BsXUWddl-as", "DM4MWxnXOzM", "QPA4yLCbMw4",
                                                      "_YHDxyO9W8Q", "uS8F-qRUeMM", "gsDGPyBzrQM", "V-4SZ4f7kMc"])}

RUN = """import json, shutil, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
files = {p.stem: p for p in Path('/kaggle/input').rglob('*.mp4') if p.stem in VIDEOS}
OUT = Path('/kaggle/working/sweep')

def one(job):
    (change, stable), video = job
    name = f'c{change}_s{stable}'
    started = time.monotonic()
    try:
        stage01, manifest, intervals = run_slide_detection(
            files[video], Path('/tmp/sweep') / name / video, video, sample_hz=1.0, fine_hz=8.0,
            merge_reveals=True, scan='single_pass', change_threshold=change, stable_samples=stable)
        target = OUT / name / video
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(stage01 / 'intervals.json', target / 'intervals.json')
        shutil.rmtree(Path('/tmp/sweep') / name / video, ignore_errors=True)
        return {'config': name, 'video': video, 'segments': len(intervals), 'seconds': round(time.monotonic() - started)}
    except Exception:
        return {'config': name, 'video': video, 'error': traceback.format_exc()[-300:]}

jobs = [(config, video) for config in CONFIGS for video in VIDEOS]
with ProcessPoolExecutor(4) as pool:
    for row in pool.map(one, jobs):
        print(row, flush=True)
shutil.make_archive(f'/kaggle/working/sweep_{SHARD}', 'zip', OUT)
"""


def build(video_set="l25"):
    dataset, videos = VIDEO_SETS[video_set]
    suffix = "" if video_set == "l25" else f"-{video_set}"
    configs = [(c, s) for c in CHANGE for s in STABLE]
    shards = [configs[i::len(ACCOUNTS)] for i in range(len(ACCOUNTS))]
    for number, (owner, group) in enumerate(zip(ACCOUNTS, shards)):
        cells = [cell("markdown", f"# Quét ngưỡng bước 01 · nhóm {number}: {group}"),
                 cell("code", "%pip install -q " + DEPS)]
        cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in MODULES]
        cells += [cell("code", f"SHARD = {number}\nCONFIGS = {group!r}\nVIDEOS = {videos!r}"), cell("code", RUN)]
        notebook = {"cells": cells,
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python"}},
                    "nbformat": 4, "nbformat_minor": 5}
        run = ROOT / "runs" / f"{DATE}-threshold-sweep{suffix}-{number}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "threshold_sweep.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        json.dump({"id": f"{owner}/l25-threshold-sweep{suffix}-{number}-{DATE}", "title": f"l25-threshold-sweep{suffix}-{number}-{DATE}",
                   "code_file": "threshold_sweep.ipynb", "language": "python", "kernel_type": "notebook",
                   "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
                   "dataset_sources": [dataset],
                   "competition_sources": [], "kernel_sources": []},
                  open(run / "kernel-metadata.json", "w"), indent=2)
        print(run.name, owner, group)


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "l25")
