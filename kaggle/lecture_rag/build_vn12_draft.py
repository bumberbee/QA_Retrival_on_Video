"""Generate the gold set 2 draft run: fast_pipeline_l25 with the current defaults on the 12 Vietnamese
test videos (vn-lecture-test12), split over the three Kaggle accounts (4 videos each, T4 x2). Its
slide boundaries, OCR text and transcript are the machine draft that gold/build_gold2_draft.py
turns into files for hand correction (F1, CER, WER on channels never used for tuning).

    python kaggle/lecture_rag/build_vn12_draft.py
"""

from pathlib import Path
import json

import build_fast_pipeline as fast

ROOT = Path(__file__).resolve().parent
DATE = "20261006"
ACCOUNTS = ["thnhtrungnguynmif", "tuvo6804", "thivntrung"]
VIDEOS = ["tHE1mC_n0-E", "D_kuulxKuxc", "bstCZZUMJYw", "_eDGTHhyQ7M", "8WnDooAH2a4", "szRBlpJ0kkY",
          "laT0CKatosY", "xysJI15jpAw", "v2YTz8ToEag", "hVU5Mw19V_c", "xLGGWMDJRkY", "y1CvB950VlU"]


def build():
    fast.build()
    base = (ROOT / "notebooks" / f"{fast.NAME}.ipynb").read_text(encoding="utf-8")
    for number, owner in enumerate(ACCOUNTS):
        group = VIDEOS[number::len(ACCOUNTS)]
        notebook = json.loads(base)
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if source.startswith("from pathlib import Path\n\nVIDEO_IDS"):
                old = source.split("\n")[2]
                cell["source"] = source.replace(old, f"VIDEO_IDS = {group!r}").splitlines(keepends=True)
        assert any(group[0] in "".join(c["source"]) for c in notebook["cells"])
        run = ROOT / "runs" / f"{DATE}-vn12-draft-{number}"
        run.mkdir(parents=True, exist_ok=True)
        (run / f"{fast.NAME}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        json.dump({"id": f"{owner}/vn12-draft-{number}-{DATE}", "title": f"vn12-draft-{number}-{DATE}",
                   "code_file": f"{fast.NAME}.ipynb", "language": "python", "kernel_type": "notebook",
                   "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                   "machine_shape": "NvidiaTeslaT4",
                   "dataset_sources": ["thnhtrungnguynmif/vn-lecture-test12"],
                   "competition_sources": [], "kernel_sources": []},
                  open(run / "kernel-metadata.json", "w"), indent=2)
        print(run.name, owner, group)


if __name__ == "__main__":
    build()
