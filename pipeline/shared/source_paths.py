"""Locate source files embedded verbatim by notebook builders and offline tools."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_GOLD = ROOT / "runs/legacy-gold" / "kaggle" / "lecture_rag" / "gold"
MODULE_PATHS = {
    "common.py": "shared/common.py",
    "production_runtime.py": "shared/production_runtime.py",
    "production_queue.py": "shared/production_queue.py",
    "detection.py": "stages/s01_segment/detection.py",
    "adfilter.py": "stages/s02_gate/adfilter.py",
    "visual.py": "stages/s03_ocr/visual.py",
    "transcription.py": "stages/s04_asr/transcription.py",
    "alignment.py": "stages/s05_align/alignment.py",
    "textclean.py": "stages/s05_align/textclean.py",
    "indexing.py": "stages/s06_index/indexing.py",
    "retrieval.py": "stages/s06_index/retrieval.py",
    "demo_answer.py": "demo/demo_answer.py",
    "demo_attribution.py": "demo/demo_attribution.py",
    "demo_capture.py": "demo/demo_capture.py",
    "demo_chat.py": "demo/demo_chat.py",
    "demo_evidence.py": "demo/demo_evidence.py",
    "demo_runtime.py": "demo/demo_runtime.py",
    "demo_sources.py": "demo/demo_sources.py",
}


def module_path(filename):
    return ROOT / MODULE_PATHS[filename]
