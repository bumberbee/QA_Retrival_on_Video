"""Small subprocess fixture using the run_video CLI and stdout contract."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--video")
parser.add_argument("--video-id")
parser.add_argument("--out")
parser.add_argument("--language")
parser.add_argument("--device")
args = parser.parse_args()
mode = os.getenv("FAKE_PIPELINE_MODE", "success")
work = Path(args.out)
(work / "cwd.txt").write_text(str(Path.cwd()), encoding="utf-8")


def emit(event):
    print(json.dumps(event), flush=True)


print("diagnostic log", file=sys.stderr, flush=True)
print("not JSON", flush=True)
emit({"event": "stage", "stage": "s03_ocr", "status": "running", "progress": 0.42,
      "overall": 0.318, "phase": "working"})
release = os.getenv("FAKE_PIPELINE_RELEASE")
if release:
    deadline = time.monotonic() + 5
    while not Path(release).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
if mode in {"error", "error_zero"}:
    emit({"event": "error", "stage": "s03_ocr", "message": "sensitive internal diagnostic"})
    sys.exit(0 if mode == "error_zero" else 1)
if mode == "silent":
    sys.exit(0)

export = work / "export" / args.video_id
slides = export / "01_slides"
states = slides / "keyframes" / "states"
states.mkdir(parents=True)
(states / "state_00000.jpg").write_bytes(b"fake-slide")
(slides / "intervals.json").write_text(json.dumps([{
    "segment_id": "seg_00000", "start_sec": 0.0, "end_sec": 5.0,
    "content_kind": "slide", "reveal_states": [{
        "image_path": "states/state_00000.jpg", "start_sec": 0.0, "source_state": "state_00000",
    }],
}]), encoding="utf-8")
(slides / "reveal_audit.json").write_text("[]", encoding="utf-8")
bundle = export / "04_bundle"
bundle.mkdir()
(bundle / "segments.json").write_text("[]", encoding="utf-8")
(bundle / "transcript_app.json").write_text(json.dumps({
    "video_id": args.video_id, "language": args.language,
    "segments": [{"start": 0, "end": 5, "text": "Test lecture"}],
}), encoding="utf-8")
index = export / "05_index"
index.mkdir()
(index / "chunks.jsonl").write_text('{}\n', encoding="utf-8")
if mode != "missing_index":
    (index / "embeddings.npy").write_bytes(b"fixture-embeddings")
    (index / "lexical.sqlite3").write_bytes(b"fixture-lexical")
emit({"event": "stage", "stage": "s06_index", "status": "completed", "progress": 1})
emit({"event": "done", "export_dir": str(export if mode != "wrong_export" else work.parent)})
if mode == "nonzero":
    sys.exit(1)
