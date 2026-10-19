"""Run fast_pipeline_l25 with the current defaults on the 78 L25 videos without hand labels
(L25_V011 .. L25_V088), so the web app can answer over the whole 88-video course. Split into 6
notebooks of 13 videos (~3.5 h each on T4 x2), two per Kaggle account.

    python kaggle/lecture_rag/build_l25_all.py
"""

from pathlib import Path
import json

import build_fast_pipeline as fast

ROOT = Path(__file__).resolve().parent
DATE = "20261007"
ACCOUNTS = ["thnhtrungnguynmif", "tuvo6804", "thivntrung"]
VIDEOS = [f"L25_V{n:03d}" for n in range(11, 89)]
PARTS = 6


def build():
    fast.build()
    base = (ROOT / "notebooks" / f"{fast.NAME}.ipynb").read_text(encoding="utf-8")
    for number in range(PARTS):
        owner, group = ACCOUNTS[number % len(ACCOUNTS)], VIDEOS[number::PARTS]
        notebook = json.loads(base)
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if source.startswith("from pathlib import Path\n\nVIDEO_IDS"):
                old = source.split("\n")[2]
                cell["source"] = source.replace(old, f"VIDEO_IDS = {group!r}").splitlines(keepends=True)
        assert any(group[0] in "".join(c["source"]) for c in notebook["cells"])
        run = ROOT / "runs" / f"{DATE}-l25-all-{number}"
        run.mkdir(parents=True, exist_ok=True)
        (run / f"{fast.NAME}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        json.dump({"id": f"{owner}/l25-all-{number}-{DATE}", "title": f"l25-all-{number}-{DATE}",
                   "code_file": f"{fast.NAME}.ipynb", "language": "python", "kernel_type": "notebook",
                   "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                   "machine_shape": "NvidiaTeslaT4",
                   "dataset_sources": ["thnhtrungnguynmif/l25a-lecture-videos-media"],
                   "competition_sources": [], "kernel_sources": []},
                  open(run / "kernel-metadata.json", "w"), indent=2)
        print(run.name, owner, len(group), group[0], "…", group[-1])


if __name__ == "__main__":
    build()
