"""Generate the 10-video L25 rerun (item D): fast_pipeline_l25 with the new defaults (ad filter in
"flag" mode, OCR gate frame+crops 0.15, speech relevance recorded) on L25_V001..V010, plus a
second stage-05 index per video with dedupe_previous=True (<video>/dedupe/05_index) so the app
search can be scored with and without repeated slide lines.

    python kaggle/lecture_rag/build_full10_v2.py
"""

from pathlib import Path
import json

import build_fast_pipeline as fast

ROOT = Path(__file__).resolve().parent
DATE = "20261006"
OWNER = "thnhtrungnguynmif"
VIDEOS = [f"L25_V{n:03d}" for n in range(1, 11)]

DEDUPE = """# Chỉ mục thứ hai: chỉ dòng OCR chưa có ở slide liền trước (so hỏi đáp có / không bỏ trùng)
for video_id in VIDEO_IDS:
    stage04 = OUTPUT_ROOT / video_id / '04_bundle'
    if (stage04 / 'segments.json').exists():
        run_indexing(stage04, OUTPUT_ROOT / video_id / 'dedupe', video_id,
                     embedding_model=EMBEDDING_MODEL, revision=EMBEDDING_REVISION, dedupe_previous=True)
"""


def build():
    fast.build()
    notebook = json.loads((ROOT / "notebooks" / f"{fast.NAME}.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if source.startswith("from pathlib import Path\n\nVIDEO_IDS"):
            old = source.split("\n")[2]
            cell["source"] = source.replace(old, f"VIDEO_IDS = {VIDEOS!r}").splitlines(keepends=True)
    inspect = next(i for i, c in enumerate(notebook["cells"]) if "".join(c["source"]).startswith("import pandas"))
    notebook["cells"].insert(inspect, {"cell_type": "code", "execution_count": None, "metadata": {},
                                       "outputs": [], "source": DEDUPE.splitlines(keepends=True)})
    assert any("L25_V010" in "".join(c["source"]) for c in notebook["cells"])
    run = ROOT / "runs" / f"{DATE}-full-10-v2"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{fast.NAME}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    json.dump({"id": f"{OWNER}/l25-full-10-v2-{DATE}", "title": f"l25-full-10-v2-{DATE}",
               "code_file": f"{fast.NAME}.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [f"{OWNER}/l25a-lecture-videos-media"],
               "competition_sources": [], "kernel_sources": [f"{OWNER}/l25-ad-filter-e1-e4-20261002"]},
              open(run / "kernel-metadata.json", "w"), indent=2)
    print(run)


if __name__ == "__main__":
    build()
