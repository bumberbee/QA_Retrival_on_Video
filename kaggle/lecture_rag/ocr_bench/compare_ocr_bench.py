"""Side-by-side comparison page for the OCR bench.

    python kaggle/lecture_rag/ocr_bench/compare_ocr_bench.py <bench_dir> <slides_dir> <out.html>

<bench_dir> holds one folder per model with results.json (downloaded kernel outputs);
<slides_dir> holds the 8 slide PNGs. Boxes are drawn on each slide and coloured by kind.
"""

import base64
import html
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

BENCH, SLIDES, TARGET = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ORDER = ["easyocr", "paddlevl", "dotsocr", "unlimited", "qwen3vl"]
COLOURS = {"formula": (214, 48, 49), "table": (9, 132, 227), "chart": (0, 148, 50),
           "figure": (230, 126, 34), "text": (120, 120, 120), "title": (108, 52, 131)}


def kind(label):
    label = (label or "").lower()
    if "formula" in label or "equation" in label or "latex" in label:
        return "formula"
    if "table" in label:
        return "table"
    if "chart" in label or "plot" in label or "graph" in label:
        return "chart"
    if any(word in label for word in ("figure", "picture", "image", "diagram")):
        return "figure"
    if "title" in label or "header" in label:
        return "title"
    return "text"


def overlay(slide, blocks):
    with Image.open(SLIDES / f"{slide}.png") as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for block in blocks:
        if not block.get("bbox"):
            continue
        x1, y1, x2, y2 = block["bbox"]
        colour = COLOURS[kind(block.get("type"))]
        draw.rectangle((x1 * width, y1 * height, x2 * width, y2 * height), outline=colour,
                       width=4 if kind(block.get("type")) != "text" else 2)
    image.thumbnail((640, 640))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


models = {}
for key in ORDER:
    path = BENCH / key / "results.json"
    if not path.exists():
        found = list((BENCH / key).rglob("results.json"))
        path = found[0] if found else None
    if path:
        models[key] = json.loads(path.read_text(encoding="utf-8"))

slides = [item["slide"] for item in next(iter(models.values()))["slides"]]
summary, sections = [], []
for key, data in models.items():
    rows = data["slides"]
    ok = [row for row in rows if not row.get("error")]
    counts = {name: sum(1 for row in rows for block in row["blocks"] if kind(block.get("type")) == name)
              for name in ("formula", "table", "chart", "figure")}
    boxed = sum(1 for row in rows for block in row["blocks"] if block.get("bbox"))
    total = sum(len(row["blocks"]) for row in rows)
    seconds = [row["seconds"] for row in ok if row.get("seconds") is not None]
    peaks = [row["peak_gb"] for row in ok if row.get("peak_gb")]
    summary.append(
        f"<tr><td><b>{html.escape(data['model'])}</b></td><td class=num>{len(ok)}/{len(rows)}</td>"
        f"<td class=num>{total}</td><td class=num>{boxed}</td>"
        + "".join(f"<td class=num>{counts[name]}</td>" for name in ("formula", "table", "chart", "figure"))
        + f"<td class=num>{(sum(seconds) / len(seconds)):.1f}</td>" if seconds else "<td class=num>—</td>")
    summary[-1] += f"<td class=num>{max(peaks):.1f}</td></tr>" if peaks else "<td class=num>—</td></tr>"

for slide in slides:
    cards = []
    for key, data in models.items():
        row = next(item for item in data["slides"] if item["slide"] == slide)
        blocks = row["blocks"]
        listed = "".join(
            f"<li><span class='tag t-{kind(b.get('type'))}'>{html.escape(str(b.get('type')))}</span>"
            f"<code>{html.escape(str(b.get('text', ''))[:400])}</code></li>"
            for b in blocks if kind(b.get("type")) != "text")[:12000]
        text_blocks = [b for b in blocks if kind(b.get("type")) == "text"]
        texts = html.escape(" · ".join(str(b.get("text", "")) for b in text_blocks)[:1500])
        error = f"<pre class=err>{html.escape(row['error'][-600:])}</pre>" if row.get("error") else ""
        timing = f"{row['seconds']:.0f} s" if row.get("seconds") is not None else "đã có sẵn"
        cards.append(
            f"<div class=card><h4>{html.escape(data['model'])} <small>{timing} · {len(blocks)} khối</small></h4>"
            f"<img src='{overlay(slide, blocks)}' alt=''>{error}"
            f"<details open><summary>Công thức / bảng / biểu đồ / hình ({len(blocks) - len(text_blocks)})</summary><ul>{listed or '<li>—</li>'}</ul></details>"
            f"<details><summary>Chữ thường ({len(text_blocks)})</summary><p class=txt>{texts or '—'}</p></details>"
            f"<details><summary>Kết quả thô</summary><pre>{html.escape(str(row['raw'])[:6000])}</pre></details></div>")
    sections.append(f"<section id='{slide}'><h2>{slide}</h2><div class=grid>{''.join(cards)}</div></section>")

legend = "".join(f"<span style='--c:rgb{colour}'>{name}</span>" for name, colour in COLOURS.items())
TARGET.write_text(f"""<!DOCTYPE html><html lang=vi><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>So sánh OCR slide</title>
<style>
body{{margin:0;background:#fbf3ea;color:#3a3a3a;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:20px 18px 60px}}
h1{{margin:0 0 6px}} h2{{margin:34px 0 10px;border-bottom:2px solid #f2ceba;padding-bottom:4px}}
table{{border-collapse:collapse;background:#fffaf3;font-size:13px}} td,th{{border:1px solid #e3ceb8;padding:6px 10px}}
th{{background:#f2e9dd;font-size:11px;text-transform:uppercase}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}}
.card{{background:#fffaf3;border:1px solid #f2ceba;border-radius:12px;padding:10px 12px;min-width:0}}
.card h4{{margin:0 0 8px}} .card small{{color:#8a7f74;font-weight:400}} .card img{{width:100%;border-radius:6px}}
ul{{padding-left:0;list-style:none;margin:6px 0}} li{{margin:4px 0}} code{{white-space:pre-wrap;word-break:break-word;font-size:12px}}
.tag{{display:inline-block;font-size:10px;font-weight:700;color:#fff;border-radius:4px;padding:1px 6px;margin-right:6px}}
.t-formula{{background:rgb{COLOURS['formula']}}} .t-table{{background:rgb{COLOURS['table']}}} .t-chart{{background:rgb{COLOURS['chart']}}}
.t-figure{{background:rgb{COLOURS['figure']}}} .t-title{{background:rgb{COLOURS['title']}}}
pre{{white-space:pre-wrap;word-break:break-word;font-size:11px;background:#f2e9dd;padding:8px;border-radius:6px;max-height:320px;overflow:auto}}
.err{{background:#fbe4de;color:#a33}} .txt{{font-size:12px}} summary{{cursor:pointer;font-weight:600;font-size:12px;margin-top:6px}}
.legend span{{margin-right:14px;font-size:12px}} .legend span::before{{content:"";display:inline-block;width:14px;height:10px;border:3px solid var(--c);margin-right:5px;vertical-align:-2px}}
</style></head><body>
<h1>So sánh OCR / VLM trên 8 slide</h1>
<p>6 slide toán (L25_V088) và 2 slide biểu đồ (L25_V007), GPU T4. Khung được vẽ theo tọa độ mỗi model trả về.</p>
<p class=legend>{legend}</p>
<table><tr><th>Model</th><th>Chạy được</th><th>Khối</th><th>Có tọa độ</th><th>Công thức</th><th>Bảng</th><th>Biểu đồ</th><th>Hình</th><th>Giây/slide</th><th>GPU GB</th></tr>
{''.join(summary)}</table>
{''.join(sections)}</body></html>""", encoding="utf-8")
print(TARGET, TARGET.stat().st_size // 1024, "KB")
