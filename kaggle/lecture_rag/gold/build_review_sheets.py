"""Review sheets for the pipeline's draft boundaries (Claude's first pass).

    python kaggle/lecture_rag/gold/build_review_sheets.py <stage01_root> <out_dir>

For every video: draft.json lists each state boundary (time, whether the pipeline
merged it as a reveal or kept it as a new slide) and sheets/*.jpg show 8 boundaries
per image as before | after keyframes. Long slides also get a strip of their
states so missed changes inside them can be spotted.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
THUMB = (300, 169)
PER_SHEET = 8


def clock(seconds):
    return f"{int(seconds) // 60}:{seconds % 60:04.1f}"


for folder in sorted(ROOT.glob("L25_V*")):
    stage = folder / "01_slides"
    intervals = json.loads((stage / "intervals.json").read_text(encoding="utf-8"))
    states = sorted((s | {"slide": i["sequence"]} for i in intervals for s in i["reveal_states"]),
                    key=lambda s: s["source_state"])
    events = []
    for previous, current in zip(states, states[1:]):
        events.append({"id": len(events), "time_sec": round(current["start_sec"], 2),
                       "pipeline": "slide" if current["state"] == 0 else "reveal",
                       "before": previous["image_path"], "after": current["image_path"]})
    out = OUT / folder.name
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    (out / "draft.json").write_text(json.dumps({"video": folder.name, "events": events,
                                                "slides": len(intervals), "states": len(states)},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    for start in range(0, len(events), PER_SHEET):
        chunk = events[start:start + PER_SHEET]
        rows = (len(chunk) + 1) // 2
        sheet = Image.new("RGB", (2 * (2 * THUMB[0] + 30), rows * (THUMB[1] + 22)), "white")
        draw = ImageDraw.Draw(sheet)
        for n, event in enumerate(chunk):
            x = (n % 2) * (2 * THUMB[0] + 30)
            y = (n // 2) * (THUMB[1] + 22)
            for k, key in enumerate(("before", "after")):
                with Image.open(stage / event[key]) as image:
                    image = image.convert("RGB")
                    image.thumbnail(THUMB)
                    sheet.paste(image, (x + k * (THUMB[0] + 4), y + 20))
            colour = "red" if event["pipeline"] == "slide" else "blue"
            draw.text((x + 2, y + 3), f"#{event['id']} {clock(event['time_sec'])} {event['pipeline']}", fill=colour)
        sheet.save(out / "sheets" / f"{start // PER_SHEET:03d}.jpg", quality=80)
    print(folder.name, "events", len(events), "sheets", (len(events) + PER_SHEET - 1) // PER_SHEET)
