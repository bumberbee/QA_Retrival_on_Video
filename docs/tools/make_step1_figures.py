"""Figures for the slide-detection slides of the progress report, made with the pipeline's own
functions (pipeline/stages/s01_segment/detection.py) on a local copy of L25_V008.mp4.

    python docs/tools/make_step1_figures.py <L25_V008.mp4>
Writes docs/figs/step1_*.jpg and docs/figs/step1_curve.json.
"""
import json, shutil, subprocess, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "pipeline"))
from shared.source_paths import module_path
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)
VIDEO = Path(sys.argv[1])
ns = {}
for name in ("common.py", "detection.py"):
    exec(module_path(name).read_text(encoding="utf-8"), ns)
FONT = ImageFont.truetype("arial.ttf", 22) if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default()


def frames_at(seconds):
    """Exact frames at these seconds (the pipeline's own PyAV seek, not cv2's approximate one)."""
    source = ns["probe_video"](VIDEO)
    return {second: ns["frame_at_time"](VIDEO, source, second)[0] for second in seconds}


def label(image, text, color=(21, 56, 152)):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), text, font=FONT)
    draw.rectangle((0, 0, box[2] + 16, box[3] + 10), fill=color)
    draw.text((8, 4), text, fill="white", font=FONT)
    return image


def row(images, width=320, gap=8):
    images = [im.resize((width, int(width * im.height / im.width))) for im in images]
    canvas = Image.new("RGB", (len(images) * (width + gap) - gap, max(im.height for im in images)), "white")
    for index, im in enumerate(images):
        canvas.paste(im, (index * (width + gap), 0))
    return canvas


# 1. one picture per second around the cut at 1:38 (lecturer -> definition slide)
strip_seconds = [95, 96, 97, 98, 99, 100]
shots = frames_at(strip_seconds)
row([label(shots[s], f"{s // 60}:{s % 60:02d}") for s in strip_seconds], 300).save(OUT / "step1_strip.jpg", quality=85)

# 2. the signature of one frame: grey 240x135 blurred + edges (what the code compares)
frame = frames_at([115])[115]
signature = ns["frame_signature"](frame)
blurred = Image.fromarray((signature[0] * 255).astype("uint8")).convert("RGB").resize((480, 270), Image.NEAREST)
edges = Image.fromarray((signature[1] * 255).astype("uint8")).convert("RGB").resize((480, 270), Image.NEAREST)
row([label(frame, "Khung gốc"), label(blurred, "Xám 240×135, làm mờ"), label(edges, "Cạnh (nét chữ)")], 420).save(
    OUT / "step1_signature.jpg", quality=85)

# 3. distance of every second to the current slide's reference, with the code's boundaries
source = ns["probe_video"](VIDEO)
samples = list(ns["sampled_frames"](VIDEO, source, 1.0))
boundaries, candidates, _ = ns["detect_stable_boundaries"](samples, source["duration_sec"])
reference, curve, cut_index = samples[0]["signature"], [], 1
for sample in samples:
    while cut_index < len(boundaries) - 1 and sample["time_sec"] >= boundaries[cut_index]:
        reference = next(c["signature"] for c in candidates if c["time_sec"] == boundaries[cut_index])
        cut_index += 1
    curve.append([round(sample["time_sec"], 2), round(ns["signature_distance"](reference, sample["signature"]), 4)])
json.dump({"curve": curve, "boundaries": [round(b, 2) for b in boundaries]},
          open(OUT / "step1_curve.json", "w"), indent=0)

print("boundaries", [round(b) for b in boundaries][:14])
