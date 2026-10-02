"""Readable OCR-bench report (docs/ocr-bench-report.html).

    python kaggle/lecture_rag/ocr_bench/write_report.py

Reads every kaggle/lecture_rag/runs/ocr-bench-20260929/out/<model>/**/results.json
that exists; models without results are listed with their status.
"""

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "kaggle/lecture_rag/runs/ocr-bench-20260929"
TARGET = ROOT / "docs/ocr-bench-report.html"
COLOURS = {"formula": (214, 48, 49), "table": (9, 132, 227), "chart": (0, 148, 50),
           "figure": (230, 126, 34), "title": (108, 52, 131), "text": (150, 150, 150)}
MODELS = [  # key, name, status when missing
    ("hybrid", "Kết hợp: Unlimited-OCR + Qwen3-VL", ""),
    ("qwen3vl", "Qwen3-VL-4B", "đang chạy"),
    ("unlimited", "Unlimited-OCR (Baidu)", "đang chạy"),
    ("dotsocr", "dots.ocr", ""),
    ("paddlevl", "PaddleOCR-VL-1.5 (Baidu)",
     "bị hủy vì quá chậm khi chạy bằng transformers v5 trên T4: 19.351 s (~5,4 giờ) cho slide đầu, "
     "10.423 s cho slide thứ hai; mới xong 3/8 slide nên chưa có kết quả để so. Cần thử lại bằng "
     "pipeline chính thức của Paddle hoặc vLLM"),
    ("easyocr", "EasyOCR (đang dùng)", ""),
]


def kind(label):
    label = (label or "").lower()
    if any(w in label for w in ("formula", "equation")):
        return "formula"
    if "table" in label:
        return "table"
    if any(w in label for w in ("chart", "plot", "graph")):
        return "chart"
    if any(w in label for w in ("figure", "picture", "image", "diagram")) and "caption" not in label:
        return "figure"
    if any(w in label for w in ("title", "header")):
        return "title"
    return "text"


def load(key):
    found = list((BENCH / "out" / key).rglob("results.json"))
    return json.loads(found[0].read_text(encoding="utf-8")) if found else None


def overlay(slide, blocks, width=760):
    with Image.open(BENCH / "dataset" / f"{slide}.png") as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for block in blocks:
        if block.get("bbox"):
            x1, y1, x2, y2 = block["bbox"]
            k = kind(block.get("type"))
            draw.rectangle((x1 * w, y1 * h, x2 * w, y2 * h), outline=COLOURS[k], width=2 if k == "text" else 5)
    image.thumbnail((width, width))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def esc(text):
    return html.escape(str(text))


results = {key: load(key) for key, *_ in MODELS}
available = [(key, name) for key, name, _ in MODELS if results[key]]
by_slide = {key: {row["slide"]: row for row in data["slides"]} for key, data in results.items() if data}


def stats(key):
    rows = results[key]["slides"]
    count = lambda k: sum(1 for r in rows for b in r["blocks"] if kind(b.get("type")) == k)
    filled = lambda k: sum(1 for r in rows for b in r["blocks"] if kind(b.get("type")) == k and str(b.get("text", "")).strip())
    seconds = [r["seconds"] for r in rows if r.get("seconds")]
    return {"ok": sum(1 for r in rows if r["blocks"]), "n": len(rows),
            "formula": count("formula"), "table": count("table"),
            "chart": f"{count('chart')} ({filled('chart')} có nội dung)",
            "figure": f"{count('figure')} ({filled('figure')} có nội dung)",
            "sec": f"{sum(seconds) / len(seconds):.0f}" if seconds else "~2–3",
            "gb": f"{max(r['peak_gb'] for r in rows if r.get('peak_gb')):.1f}" if any(r.get("peak_gb") for r in rows) else "—"}


def card(key, name, slide):
    row = by_slide[key].get(slide)
    if not row:
        return ""
    specials = [b for b in row["blocks"] if kind(b.get("type")) not in ("text", "title")]
    items = "".join(f"<li><span class='tag t-{kind(b.get('type'))}'>{esc(b.get('type'))}</span>"
                    f"<code>{esc(str(b.get('text', ''))[:500]) or '<i>(không có nội dung)</i>'}</code></li>"
                    for b in specials)
    seconds = f"{row['seconds']:.0f} s" if row.get("seconds") else "có sẵn"
    return (f"<div class='card'><h4>{esc(name)} <small>{seconds} · {len(row['blocks'])} khối</small></h4>"
            f"<img src='{overlay(slide, row['blocks'])}' alt=''>"
            f"<ul class='items'>{items or '<li>(không khoanh được công thức / bảng / biểu đồ / hình)</li>'}</ul></div>")


EXAMPLES = [
    ("L25_V088_slide26", "Ví dụ 4 — slide bạn khoanh đỏ",
     "Qwen3-VL khoanh công thức y′ và cả bảng biến thiên, chuyển bảng thành Markdown kèm mô tả đồ thị. "
     "Unlimited-OCR viết LaTeX gọn và đúng trong câu nhưng coi bảng biến thiên là \"image\" và không đọc bên trong. "
     "Cả hai chưa tách công thức giữa câu thành khung riêng."),
    ("L25_V088_slide10", "Chuỗi biến đổi công thức",
     "Qwen3-VL và Unlimited-OCR đều tách từng dòng biến đổi thành khung công thức với LaTeX đúng; Unlimited-OCR khung khớp hơn."),
    ("L25_V088_slide17", "Bảng xét dấu",
     "Qwen3-VL ra bảng Markdown; Unlimited-OCR ra bảng HTML — cả hai đọc đúng dấu +, −, 0."),
    ("L25_V007_slide09", "Biểu đồ tròn",
     "Chỉ Qwen3-VL đọc được số liệu trong biểu đồ; Unlimited-OCR khoanh vùng hình nhưng để trống."),
    ("L25_V007_slide16", "Biểu đồ cột",
     "Qwen3-VL bị lặp chữ \"THANH NIÊN\" nên không ra kết quả; Unlimited-OCR khoanh 3 vùng biểu đồ nhưng không đọc số liệu."),
]

summary_rows = ""
for key, name in available:
    s = stats(key)
    summary_rows += (f"<tr><td><b>{esc(name)}</b></td><td class=num>{s['ok']}/{s['n']}</td><td class=num>{s['formula']}</td>"
                     f"<td class=num>{s['table']}</td><td class=num>{s['chart']}</td><td class=num>{s['figure']}</td>"
                     f"<td class=num>{s['sec']}</td><td class=num>{s['gb']}</td></tr>")
missing = "".join(f"<li><b>{esc(name)}</b>: {esc(status)}</li>" for key, name, status in MODELS if not results[key])

example_html = ""
for slide, title, note in EXAMPLES:
    cards = "".join(card(key, name, slide) for key, name in available if key != "easyocr")
    easy = esc(by_slide["easyocr"][slide]["raw"][:400]) if "easyocr" in by_slide else ""
    example_html += (f"<section class='example'><h3>{esc(title)} <small>{slide}</small></h3><p>{esc(note)}</p>"
                     f"<div class='grid'>{cards}</div><details><summary>EasyOCR (đang dùng) đọc được</summary><pre>{easy}…</pre></details></section>")

page = f"""<!DOCTYPE html><html lang=vi><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Báo cáo so sánh OCR</title>
<style>
:root{{--cream:#fbf3ea;--card:#fffaf3;--ink:#3a3a3a;--sub:#8a7f74;--orange:#e8734a;--soft:#f2ceba;--dash:#e3ceb8;--chip:#f2e9dd}}
body{{margin:0;background:var(--cream);color:var(--ink);font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1240px;margin:0 auto;padding:28px 20px 70px}}
h1{{font-size:1.7rem;margin:0 0 6px}} h2{{font-size:1.25rem;margin:36px 0 10px;padding-top:24px;border-top:1px dashed var(--dash)}}
h3{{font-size:1.05rem;margin:0 0 4px}} h3 small,h4 small{{color:var(--sub);font-weight:400}} h4{{margin:0 0 6px;font-size:.92rem}}
p{{max-width:85ch}} .lead{{color:var(--sub)}}
table{{border-collapse:collapse;background:var(--card);width:100%}} td,th{{border:1px solid var(--dash);padding:8px 10px;text-align:left;vertical-align:top;font-size:.88rem}}
th{{background:var(--chip);font-size:.72rem;text-transform:uppercase}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
.example{{background:var(--card);border:1px solid var(--soft);border-radius:14px;padding:14px 16px;margin:18px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}}
.card{{min-width:0}} .card img{{width:100%;border-radius:8px;border:1px solid var(--dash)}}
.items{{list-style:none;padding:0;margin:6px 0 0;max-height:300px;overflow:auto}} .items li{{margin:4px 0}}
code{{white-space:pre-wrap;word-break:break-word;font-size:.78rem}} pre{{white-space:pre-wrap;word-break:break-word;font-size:.78rem;background:var(--chip);padding:8px;border-radius:6px}}
.tag{{display:inline-block;color:#fff;font-size:.66rem;font-weight:700;border-radius:4px;padding:1px 6px;margin-right:6px}}
.t-formula{{background:rgb{COLOURS['formula']}}} .t-table{{background:rgb{COLOURS['table']}}} .t-chart{{background:rgb{COLOURS['chart']}}} .t-figure{{background:rgb{COLOURS['figure']}}}
.legend span{{margin-right:14px;font-size:.82rem}} .legend span::before{{content:"";display:inline-block;width:14px;height:10px;border:3px solid var(--c);margin-right:5px;vertical-align:-2px}}
.callout{{background:var(--card);border-left:4px solid var(--orange);border-radius:8px;padding:10px 14px;margin:14px 0}}
summary{{cursor:pointer;font-size:.85rem;margin-top:8px}}
</style></head><body><main>
<h1>Báo cáo so sánh OCR cho slide bài giảng</h1>
<p class="lead">Mục tiêu: trích từ slide không chỉ chữ mà cả <b>công thức (LaTeX)</b>, <b>bảng</b>, <b>biểu đồ</b>, <b>hình vẽ</b>, mỗi phần kèm <b>tọa độ khung</b>, để LLM hiểu slide. Thử trên cùng 8 slide: 6 slide toán (L25_V088) và 2 slide biểu đồ (L25_V007), GPU Kaggle T4. Một người chấm (Claude) bằng mắt; chưa có nhãn chuẩn.</p>

<h2>So sánh tổng quát</h2>
<table><tr><th>Model</th><th>Slide ra kết quả</th><th>Khung công thức</th><th>Khung bảng</th><th>Khung biểu đồ</th><th>Khung hình</th><th>Giây / slide</th><th>GPU (GB)</th></tr>{summary_rows}</table>
<table style="margin-top:14px"><tr><th>Yêu cầu</th><th>Kết hợp (đề xuất)</th><th>Qwen3-VL-4B</th><th>Unlimited-OCR</th><th>dots.ocr</th><th>EasyOCR (đang dùng)</th></tr>
<tr><td>Tọa độ từng vùng</td><td>✅ Khung của Unlimited-OCR</td><td>✅ Có; đôi chỗ lệch nửa dòng</td><td>✅ Khớp tốt; phân loại header/title/text/equation/table/image</td><td>✅ Khớp tốt; 11 loại vùng (Section-header, List-item, Formula, Table, Picture…)</td><td>Chỉ theo dòng chữ</td></tr>
<tr><td>Công thức → LaTeX</td><td>✅ Rất tốt</td><td>✅ Rất tốt</td><td>✅ Rất tốt, mỗi công thức một khung</td><td>✅ Rất tốt; công thức giữa câu viết $…$ trong khung chữ</td><td>❌ Rác</td></tr>
<tr><td>Bảng</td><td>✅ HTML; bảng biến thiên có hình → Markdown + mô tả đường cong</td><td>✅ Markdown, đôi khi sai 1 ô</td><td>✅ HTML đúng; bảng biến thiên có hình vẽ bị coi là ảnh</td><td>✅ HTML; bảng biến thiên bị coi là Picture</td><td>❌</td></tr>
<tr><td>Biểu đồ / hình</td><td>✅ <b>Đọc số liệu biểu đồ, mô tả hình bằng tiếng Việt</b>; ảnh giảng viên/logo đánh dấu trang trí</td><td>✅ Đọc số liệu biểu đồ tròn, mô tả đồ thị</td><td>⚠️ Chỉ khoanh vùng, <b>không đọc nội dung</b></td><td>⚠️ Chỉ khoanh vùng (Picture), <b>không đọc nội dung</b></td><td>❌</td></tr>
<tr><td>Tiếng Việt</td><td>Như Unlimited-OCR (sai dấu ở slide màu); vùng hình do Qwen đọc thì đúng</td><td>✅ Gần như đúng hết</td><td>Sai nhiều dấu ở slide địa nhiều màu ("co cứu", "CHỘN ĐƯNG")</td><td>Khá, sai vài dấu ("CƠ CÁU", "SÂN XUÁT")</td><td>Hay sai dấu</td></tr>
<tr><td>Ổn định</td><td>8/8</td><td>1/8 slide lặp chữ, hỏng</td><td>8/8</td><td>8/8</td><td>8/8</td></tr>
<tr><td>Tốc độ trên T4</td><td><b>~28 s/slide</b> (15 s slide chữ, 40–60 s slide biểu đồ)</td><td>~88 s/slide</td><td><b>~18 s/slide</b></td><td>~37 s/slide</td><td>~2–3 s/slide</td></tr>
</table>
<div class="callout"><b>Kết luận:</b> dùng <b>bản kết hợp</b> cho bước 02 — đủ cả khung, chữ, công thức LaTeX, bảng, <b>số liệu biểu đồ và mô tả hình</b>, ổn định 8/8 slide, ~28 s/slide (nhanh gấp 3 Qwen chạy riêng). Còn lại: sai dấu tiếng Việt ở slide nhiều màu (thừa hưởng từ Unlimited-OCR) và vài vùng ảnh giảng viên chưa được đánh dấu trang trí.</div>
<div class="callout"><b>So với từng model riêng:</b> không model nào tự làm đủ cả. <b>Unlimited-OCR</b> nhanh (~5 lần), ổn định, khung và LaTeX tốt nhất cho chữ + công thức + bảng, nhưng bỏ trống biểu đồ và hình. <b>Qwen3-VL-4B</b> là model duy nhất <b>hiểu</b> biểu đồ và hình vẽ, nhưng chậm và có lúc lặp. Vì vậy bản kết hợp dùng <b>Unlimited-OCR cho cả slide</b> và <b>Qwen3-VL chỉ cho các vùng ảnh/biểu đồ</b>. dots.ocr cho kết quả gần giống Unlimited-OCR (tốt hơn chút ở tiếng Việt, chậm gấp đôi) nên là phương án dự phòng. PaddleOCR-VL chưa đánh giá được vì quá chậm trong cách chạy thử này.</div>
<p>Chưa có kết quả:</p><ul>{missing or '<li>—</li>'}</ul>
<p class="legend"><span style="--c:rgb{COLOURS['formula']}">công thức</span><span style="--c:rgb{COLOURS['table']}">bảng</span><span style="--c:rgb{COLOURS['chart']}">biểu đồ</span><span style="--c:rgb{COLOURS['figure']}">hình</span><span style="--c:rgb{COLOURS['title']}">tiêu đề</span><span style="--c:rgb{COLOURS['text']}">chữ</span></p>

<h2>Ví dụ từng slide</h2>
{example_html}
<p>Trang chi tiết (kết quả thô của mọi model): <a href="ocr-bench.html">ocr-bench.html</a>.</p>
</main></body></html>"""
TARGET.write_text(page, encoding="utf-8")
print(TARGET, TARGET.stat().st_size // 1024, "KB")
