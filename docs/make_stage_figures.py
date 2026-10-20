"""Result figures for the progress report, from the 10-video L25 rerun (runs/20261006-full-10-v2).

    python docs/make_stage_figures.py gate <L25_V008 export dir>
        -> figs/gate_grid.jpg: every segment's keyframe, its OCR-gate score and decision
    python docs/make_stage_figures.py retrieval <L25_V008 export dir> <ranking.json>
        -> figs/retrieval_demo.jpg: a question and the slides the search returned (ranking.json is
           written by the backend's ONNX E5, see the `rank` command run with backend/.venv)
    backend/.venv/Scripts/python docs/make_stage_figures.py rank <L25_V008 export dir> <ranking.json>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs"
QUESTIONS = ["Xác suất của một biến cố được tính như thế nào?", "Từ 1 đến 40 có bao nhiêu số chia hết cho 3?"]


def clock(sec):
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def font(size):
    from PIL import ImageFont
    for name in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def gate(folder):
    from PIL import Image, ImageDraw
    visual = {r["segment_id"]: r for r in json.loads((folder / "02_visual/visual.json").read_text(encoding="utf-8"))}
    segments = json.loads((folder / "04_bundle/segments.json").read_text(encoding="utf-8"))
    W, H, cols, pad = 300, 169, 4, 10
    rows = (len(segments) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (W + pad) + pad, rows * (H + 52 + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for k, seg in enumerate(segments):
        record = visual[seg["segment_id"]]
        read = record["ocr_status"] == "read"
        score = max([record.get("ocr_gate_score") or 0] + list(record.get("ocr_gate_crop_scores") or []))
        x, y = pad + (k % cols) * (W + pad), pad + (k // cols) * (H + 52 + pad)
        image = Image.open(folder / "04_bundle" / seg["visual"]["image_path"]).convert("RGB").resize((W, H))
        sheet.paste(image, (x, y))
        color = (47, 125, 74) if read else (196, 85, 63)
        draw.rectangle((x - 3, y - 3, x + W + 2, y + H + 2), outline=color, width=5)
        draw.text((x, y + H + 6), f"#{seg['sequence']}  {clock(seg['start_sec'])}–{clock(seg['end_sec'])}", fill=(30, 36, 51), font=font(17))
        draw.text((x, y + H + 27), f"điểm {score:.2f} · " + ("đọc OCR" if read else "bỏ OCR"), fill=color, font=font(17))
    OUT.mkdir(exist_ok=True)
    sheet.save(OUT / "gate_grid.jpg", quality=84)
    print("gate_grid.jpg", len(segments), "đoạn")


def rank(folder, out):
    import numpy as np
    sys.path.insert(0, str(HERE.parent / "backend"))
    from app.services.retrieval import default_encoder
    encoder = default_encoder()
    index = folder / "05_index"
    chunks = [json.loads(line) for line in (index / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    vectors = np.load(index / "embeddings.npy")
    result = []
    for question in QUESTIONS:
        scores = vectors @ encoder.encode(question)
        slides = []
        for i in np.argsort(-scores):
            if chunks[i]["segment_id"] not in [s["segment_id"] for s in slides]:
                slides.append({"segment_id": chunks[i]["segment_id"], "score": round(float(scores[i]), 3),
                               "source": chunks[i]["source_type"], "start_sec": chunks[i]["start_sec"],
                               "text": " ".join(chunks[i]["text"].split())[:90]})
            if len(slides) == 3:
                break
        result.append({"question": question, "slides": slides})
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)


def retrieval(folder, ranking):
    from PIL import Image, ImageDraw
    segments = {s["segment_id"]: s for s in json.loads((folder / "04_bundle/segments.json").read_text(encoding="utf-8"))}
    data = json.loads(Path(ranking).read_text(encoding="utf-8"))
    W, H, pad = 330, 186, 14
    sheet = Image.new("RGB", (pad + 3 * (W + pad) + 260, len(data) * (H + 70 + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(data):
        y = pad + r * (H + 70 + pad)
        draw.rounded_rectangle((pad, y, pad + 240, y + H), 12, fill=(238, 241, 250), outline=(21, 56, 152), width=2)
        words, line, lines = row["question"].split(), "", []
        for w in words:
            if len(line + " " + w) > 22:
                lines.append(line); line = w
            else:
                line = (line + " " + w).strip()
        lines.append(line)
        draw.text((pad + 12, y + 12), "Câu hỏi", fill=(197, 138, 0), font=font(16))
        for n, text in enumerate(lines):
            draw.text((pad + 12, y + 40 + n * 24), text, fill=(14, 31, 92), font=font(18))
        for k, slide in enumerate(row["slides"]):
            x = pad + 260 + k * (W + pad)
            seg = segments[slide["segment_id"]]
            image = Image.open(folder / "04_bundle" / seg["visual"]["image_path"]).convert("RGB").resize((W, H))
            sheet.paste(image, (x, y))
            color = (47, 125, 74) if k == 0 else (90, 106, 138)
            draw.rectangle((x - 3, y - 3, x + W + 2, y + H + 2), outline=color, width=4)
            draw.text((x, y + H + 6), f"hạng {k + 1} · slide {clock(seg['start_sec'])}–{clock(seg['end_sec'])}", fill=(14, 31, 92), font=font(16))
            draw.text((x, y + H + 28), f"cosine {slide['score']:.2f} · từ {slide['source']}", fill=color, font=font(15))
            draw.text((x, y + H + 48), slide["text"][:44], fill=(90, 106, 138), font=font(13))
    sheet.save(OUT / "retrieval_demo.jpg", quality=84)
    print("retrieval_demo.jpg")


def wrap(text, width):
    lines, line = [], ""
    for w in text.split():
        if len(line + " " + w) > width:
            lines.append(line); line = w
        else:
            line = (line + " " + w).strip()
    return lines + [line]


def sync(folder, picks=(4, 5, 6)):
    """Keyframes of consecutive slides, the recognised words on one time axis coloured by the slide they
    were assigned to, and the start of each slide's speech with the ASR fixes (asr_corrections.json)."""
    from PIL import Image, ImageDraw
    segments = [s for s in json.loads((folder / "04_bundle/segments.json").read_text(encoding="utf-8")) if s["sequence"] in picks]
    words = json.loads((folder / "04_bundle/words.json").read_text(encoding="utf-8"))
    fixes = json.loads((folder / "04_bundle/asr_corrections.json").read_text(encoding="utf-8"))
    colors = [(21, 56, 152), (197, 138, 0), (47, 125, 74)]
    W, H, pad, axis = 400, 225, 16, 300
    sheet = Image.new("RGB", (pad + 3 * (W + pad), 560), "white")
    draw = ImageDraw.Draw(sheet)
    t0, t1 = segments[0]["start_sec"], segments[-1]["end_sec"]
    x_of = lambda t: pad + (t - t0) / (t1 - t0) * (3 * (W + pad) - pad)
    for k, seg in enumerate(segments):
        x = pad + k * (W + pad)
        image = Image.open(folder / "04_bundle" / seg["visual"]["image_path"]).convert("RGB").resize((W, H))
        sheet.paste(image, (x, pad))
        draw.rectangle((x - 3, pad - 3, x + W + 2, pad + H + 2), outline=colors[k], width=5)
        draw.text((x, pad + H + 8), f"Slide #{seg['sequence']}  {clock(seg['start_sec'])}–{clock(seg['end_sec'])}", fill=colors[k], font=font(19))
        # this slide's span on the shared time axis
        draw.rectangle((x_of(seg["start_sec"]), axis, x_of(seg["end_sec"]) - 2, axis + 10), fill=colors[k])
        speech = seg["speech"]["text"]
        n_words = sum(seg["start_sec"] <= w["start_sec"] < seg["end_sec"] for w in words)
        draw.text((x, axis + 52), f"{n_words} từ được gán vào slide này", fill=(30, 36, 51), font=font(16))
        for n, line in enumerate(wrap("“" + " ".join(speech.split()[:42]) + " …”", 44)[:8]):
            draw.text((x, axis + 80 + n * 22), line, fill=(60, 70, 90), font=font(15))
    for w in words:  # one tick per recognised word
        if t0 <= w["start_sec"] < t1:
            k = next(i for i, s in enumerate(segments) if s["start_sec"] <= w["start_sec"] < s["end_sec"])
            draw.line((x_of(w["start_sec"]), axis + 14, x_of(w["start_sec"]), axis + 34), fill=colors[k], width=2)
    draw.text((pad, axis - 26), "Trục thời gian: mỗi vạch là 1 từ Whisper nhận dạng, tô theo slide đang hiện lúc từ đó được nói",
              fill=(90, 106, 138), font=font(15))
    shown = [f for f in fixes if t0 - 15 <= f["at_sec"] < t1] or fixes[:3]
    text = "Sửa từ nghe nhầm theo chữ trên slide (" + str(len(fixes)) + " chỗ cả video):  " + "   ·   ".join(
        f"{clock(f['at_sec'])} “{f['heard']}” → “{f['fixed']}”" for f in shown[:3])
    draw.rounded_rectangle((pad, 500, sheet.width - pad, 544), 8, fill=(238, 241, 250))
    draw.text((pad + 12, 511), text, fill=(14, 31, 92), font=font(17))
    sheet.save(OUT / "sync_demo.jpg", quality=84)
    print("sync_demo.jpg", len(shown), "chỗ sửa trong khoảng")


def sweep(runs):
    """F1 of every change_threshold x stable_samples of the stage-01 sweep: L25 held-out and LPM (±1 s)
    (runs/sweep-l25 and runs/sweep-lpm sweep_table.json, written by the gold/eval_threshold_sweep*.py)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = [("L25 · 6 video test (tiếng Việt)", Path(runs) / "sweep-l25/sweep_table.json", "held-out_F1"),
            ("LPM · 8 video (tiếng Anh, chưa dùng để chỉnh)", Path(runs) / "sweep-lpm/sweep_table.json", "F1@1")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (title, path, key) in zip(axes, sets):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for stable, color in [(2, "#2f7d4a"), (3, "#c58a00"), (4, "#c4553f")]:
            pts = sorted((r["change"], r[key]) for r in rows if r["stable"] == stable)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=color, lw=2.5, label=f"đứng yên ≥ {stable} s")
        current = next(r[key] for r in rows if (r["change"], r["stable"]) == (0.055, 2))
        ax.scatter([0.055], [current], s=260, facecolors="none", edgecolors="#153898", linewidths=2.5, zorder=5)
        ax.annotate(f"đang dùng\nF1 {current:.3f}", (0.055, current), (0.062, current - 0.09 if current > 0.95 else current + 0.035), color="#153898", fontsize=11,
                    arrowprops={"arrowstyle": "->", "color": "#153898"})
        ax.set_title(title, fontsize=13, color="#0e1f5c"); ax.set_xlabel("ngưỡng khác (change_threshold)")
        ax.set_xticks([0.035, 0.045, 0.055, 0.07, 0.09]); ax.grid(alpha=.3)
    axes[0].set_ylabel("F1 tách slide"); axes[0].set_ylim(0.65, 1.0); axes[0].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "sweep_f1.jpg", dpi=110); print("sweep_f1.jpg")


def gate_examples(folder, picks=((5, "slide chữ"), (2, "giảng viên + slide nhỏ"), (9, "quảng cáo"), (0, "logo"))):
    """Four V008 keyframes, each with its OCR-gate score as a bar against the 0.15 threshold."""
    from PIL import Image, ImageDraw
    visual = {r["segment_id"]: r for r in json.loads((folder / "02_visual/visual.json").read_text(encoding="utf-8"))}
    segments = {s["sequence"]: s for s in json.loads((folder / "04_bundle/segments.json").read_text(encoding="utf-8"))}
    W, H, pad = 300, 169, 14
    sheet = Image.new("RGB", (pad + len(picks) * (W + pad), H + 120), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (seq, label) in enumerate(picks):
        seg, x = segments[seq], pad + k * (W + pad)
        record = visual[seg["segment_id"]]
        score = max([record.get("ocr_gate_score") or 0] + list(record.get("ocr_gate_crop_scores") or []))
        read = record["ocr_status"] == "read"
        color = (47, 125, 74) if read else (196, 85, 63)
        sheet.paste(Image.open(folder / "04_bundle" / seg["visual"]["image_path"]).convert("RGB").resize((W, H)), (x, pad))
        draw.text((x, pad + H + 6), label, fill=(14, 31, 92), font=font(18))
        top = pad + H + 36
        draw.rectangle((x, top, x + W, top + 18), fill=(232, 236, 245))
        draw.rectangle((x, top, x + max(3, int(W * score)), top + 18), fill=color)
        tx = x + int(W * 0.15)
        draw.line((tx, top - 5, tx, top + 23), fill=(14, 31, 92), width=3)
        draw.text((x, top + 26), f"điểm {score:.2f} → " + ("đọc OCR" if read else "bỏ OCR"), fill=color, font=font(18))
    draw.text((pad + int(W * 0.15) - 10, pad + H + 84), "", fill=(14, 31, 92))
    sheet.save(OUT / "gate_examples.jpg", quality=86)
    print("gate_examples.jpg")


def gate_crops_figure(folder, seq=2, out_name="gate_crops.jpg"):
    """A keyframe the whole-frame score would skip (L25_V002 #2, 0.13), its five gate crops and their scores."""
    from PIL import Image, ImageDraw
    visual = {r["segment_id"]: r for r in json.loads((folder / "02_visual/visual.json").read_text(encoding="utf-8"))}
    seg = next(s for s in json.loads((folder / "04_bundle/segments.json").read_text(encoding="utf-8")) if s["sequence"] == seq)
    record = visual[seg["segment_id"]]
    crops = record["ocr_gate_crop_scores"][-5:]  # a stage-02 slide region, when found, comes first
    boxes = [(0, 0, .5, .5), (.5, 0, 1, .5), (0, .5, .5, 1), (.5, .5, 1, 1), (.25, .25, .75, .75)]
    names = ["góc trên trái", "góc trên phải", "góc dưới trái", "góc dưới phải", "phần giữa"]
    image = Image.open(folder / "04_bundle" / seg["visual"]["image_path"]).convert("RGB")
    W, H, pad, cw, ch = 520, 293, 16, 190, 107
    sheet = Image.new("RGB", (pad * 3 + W + 3 * (cw + pad), H + 2 * pad + 60), "white")
    draw = ImageDraw.Draw(sheet)
    sheet.paste(image.resize((W, H)), (pad, pad))
    red = (196, 85, 63)
    draw.rectangle((pad - 3, pad - 3, pad + W + 2, pad + H + 2), outline=red, width=5)
    draw.text((pad, pad + H + 10), f"Cả khung: điểm {record['ocr_gate_score']:.2f} < 0,15 → định bỏ…", fill=red, font=font(19))
    draw.text((pad, pad + H + 36), "…nên nhìn thêm 5 vùng cắt →", fill=(14, 31, 92), font=font(19))
    for k, (box, name, score) in enumerate(zip(boxes, names, crops)):
        x = 2 * pad + W + (k % 3) * (cw + pad)
        y = pad + (k // 3) * (ch + 64)
        crop = image.crop((int(box[0] * image.width), int(box[1] * image.height), int(box[2] * image.width), int(box[3] * image.height)))
        sheet.paste(crop.resize((cw, ch)), (x, y))
        color = (47, 125, 74) if score >= 0.15 else (90, 106, 138)
        draw.rectangle((x - 2, y - 2, x + cw + 1, y + ch + 1), outline=color, width=4)
        draw.text((x, y + ch + 6), name, fill=(14, 31, 92), font=font(15))
        draw.text((x, y + ch + 26), f"điểm {score:.2f}", fill=color, font=font(17))
    x = 2 * pad + W + 2 * (cw + pad)
    y = pad + ch + 64
    keep = max(crops) >= 0.15
    color = (47, 125, 74) if keep else red
    draw.rounded_rectangle((x, y, x + cw, y + ch + 40), 10, fill=(232, 244, 236) if keep else (250, 236, 232), outline=color, width=2)
    draw.text((x + 12, y + 14), f"cao nhất {max(crops):.2f}", fill=color, font=font(19))
    draw.text((x + 12, y + 44), "≥ 0,15" if keep else "< 0,15 (cả 6 điểm)", fill=color, font=font(19))
    draw.text((x + 12, y + 80), "→ vẫn đọc OCR" if keep else "→ bỏ OCR", fill=color, font=font(19))
    sheet.save(OUT / out_name, quality=86)
    print(out_name, record["ocr_gate_score"], crops)


def reveal(folder, chain=(9, 10, 11, 18, 19), pair=(11, 12)):
    """Stage-01 reveal merging on V008, from 01_slides/keyframes/states and reveal_audit.json.
    reveal_chain.jpg: consecutive states, the decision between each pair and its three numbers.
    reveal_mask.jpg: for one merged pair, the content masks the pipeline compares (detection.py,
    content_mask / reveal_evidence): old strokes still in place (blue), strokes added (green)."""
    import numpy as np
    from PIL import Image, ImageDraw
    ns = {}
    code = HERE.parent / "kaggle/lecture_rag/notebook_code"
    for name in ("common.py", "detection.py"):
        exec((code / name).read_text(encoding="utf-8"), ns)
    states = folder / "01_slides/keyframes/states"
    audit = {r["current_state"]: r for r in json.loads((folder / "01_slides/reveal_audit.json").read_text(encoding="utf-8"))}
    image = lambda n: Image.open(states / f"state_{n:05d}.jpg").convert("RGB")
    green, red, ink = (47, 125, 74), (196, 85, 63), (14, 31, 92)
    # chain
    W, H, gap, pad = 300, 169, 70, 12
    sheet = Image.new("RGB", (pad + len(chain) * W + (len(chain) - 1) * gap + pad, H + 150), "white")
    draw = ImageDraw.Draw(sheet)
    for k, n in enumerate(chain):
        x = pad + k * (W + gap)
        sheet.paste(image(n).resize((W, H)), (x, pad))
        draw.text((x, pad + H + 6), f"trạng thái {n}", fill=ink, font=font(15))
        if k == 0:
            continue
        r = audit[n]
        merged = r["decision"].startswith("merge")
        color = green if merged else red
        if chain[k - 1] != n - 1:  # states skipped in the picture
            draw.text((x - gap + 18, pad + H // 2 - 10), "· · ·", fill=ink, font=font(20))
            continue
        draw.text((x - gap + 6, pad + H // 2 - 22), "→", fill=color, font=font(34))
        draw.text((x - gap + 4, pad + H // 2 + 18), "GỘP" if merged else "TÁCH", fill=color, font=font(16))
        lines = [f"còn đúng chỗ {r['content_retained']:.0%}", f"hình đổi {r['image_change']:.1%}",
                 f"nội dung ×{r['content_growth']:.2f}"]
        for j, text in enumerate(lines):
            draw.text((x, pad + H + 30 + j * 22), text, fill=color, font=font(15))
    sheet.save(OUT / "reveal_chain.jpg", quality=86)
    # masks of one merged pair
    views = [ns["reveal_view"](image(n)) for n in range(len(list(states.iterdir())))]
    template = ns["persistent_template"](views)
    a, b = pair
    old = ns["content_mask"](views[a]) & ~template
    new = ns["content_mask"](views[b]) & ~template
    evidence = ns["reveal_evidence"](views[a], views[b], template)
    overlay = np.full(views[b].shape + (3,), 255, np.uint8)
    overlay[new & ~old] = green
    overlay[old] = (21, 56, 152)
    panels = [(image(a), f"Trước (trạng thái {a})"), (image(b), f"Sau (trạng thái {b})"),
              (Image.fromarray(overlay), "Nét chữ: xanh dương = cũ, xanh lá = mới thêm")]
    W, H = 400, 225
    sheet = Image.new("RGB", (pad + 3 * (W + pad), H + 120), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (im, title) in enumerate(panels):
        x = pad + k * (W + pad)
        sheet.paste(im.resize((W, H)), (x, pad + 26))
        draw.rectangle((x - 1, pad + 25, x + W, pad + 26 + H), outline=(200, 205, 220), width=1)
        draw.text((x, pad), title, fill=ink, font=font(16))
    text = (f"Còn đúng chỗ {evidence['content_retained']:.0%} (≥ 85%)  ·  hình đổi {evidence['image_change']:.1%} (≤ 25%)  ·  "
            f"nội dung ×{evidence['content_growth']:.2f} (≥ 0,9)  →  GỘP")
    draw.rounded_rectangle((pad, H + 50, sheet.width - pad, H + 94), 8, fill=(232, 244, 236))
    draw.text((pad + 12, H + 61), text, fill=green, font=font(18))
    sheet.save(OUT / "reveal_mask.jpg", quality=86)
    print("reveal_chain.jpg reveal_mask.jpg", {k: round(v, 3) for k, v in evidence.items()}, audit[b])


def datasets(runs):
    """One sample picture per dataset for the data slides (YouTube covers for videos not run locally)."""
    import csv
    import io
    import urllib.request
    from PIL import Image, ImageDraw
    cover = lambda vid: Image.open(io.BytesIO(urllib.request.urlopen(f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg").read())).convert("RGB").crop((0, 45, 480, 315))
    frame = lambda path: Image.open(path).convert("RGB")

    def grid(tiles, cols, name, w=480, h=270):
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * w + (cols - 1) * 8, rows * (h + 34) - 4), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (image, label, color) in enumerate(tiles):
            x, y = (k % cols) * (w + 8), (k // cols) * (h + 34)
            sheet.paste(image.resize((w, h)), (x, y))
            draw.rectangle((x, y + h, x + w, y + h + 30), fill=color)
            draw.text((x + 8, y + h + 3), label, font=font(19), fill="white")
        sheet.save(OUT / name, quality=85)

    v008 = runs / "20261002-full-10-ads/outputs/fast_pipeline_export/L25_V008/04_bundle/keyframes"
    grid([(frame(v008 / "seg_00004.jpg"), "Bài giảng · 1:38", "#153898"), (frame(v008 / "ad_00009.jpg"), "Quảng cáo xen giữa bài (có nhãn giây)", "#c0392b")], 2, "ds_l25.jpg")
    vn = runs / "20261006-vn12-draft-0/output/lecture_rag"
    grid([(cover("D_kuulxKuxc"), "VTV7 · Toán lớp 2", "#153898"), (cover("tHE1mC_n0-E"), "HTV · giải đề Vật lí", "#153898"),
          (frame(vn / "_eDGTHhyQ7M/01_slides/keyframes/seg_00008.jpg"), "Slide bài tập Hóa", "#153898"),
          (frame(vn / "hVU5Mw19V_c/01_slides/keyframes/seg_00006.jpg"), "Viết bảng phấn · Xác suất", "#153898")], 2, "ds_vn12.jpg")
    spans = [(float(r["start_s"]), float(r["end_s"])) for r in csv.DictReader(open(HERE.parent / "kaggle/lecture_rag/gold/lpm/slides.csv", encoding="utf-8")) if r["video_id"] == "IIfeX44RI9k"]
    sheet = Image.new("RGB", (968, 400), "white")
    sheet.paste(cover("IIfeX44RI9k").resize((480, 270)), (0, 0))
    draw = ImageDraw.Draw(sheet)
    draw.text((500, 10), "Mốc slide do tác giả LPM gán (28 slide):", font=font(20), fill="#153898")
    for k, (a, b) in enumerate(spans[:8]):
        draw.text((500, 44 + k * 27), f"slide {k:>2}:  {clock(a)} – {clock(b)}", font=font(19), fill="#1d2330")
    draw.text((500, 44 + 8 * 27), "…", font=font(19), fill="#1d2330")
    end = spans[-1][1]
    for k, (a, b) in enumerate(spans):
        draw.rectangle((a / end * 960, 300, b / end * 960 - 2, 340), fill=("#153898", "#f6b21b")[k % 2])
    draw.text((0, 350), "Trục thời gian cả video: mỗi khối màu là một slide → so với ranh giới máy cắt (±1 s / ±3 s)", font=font(18), fill="#5a6a8a")
    sheet.save(OUT / "ds_lpm.jpg", quality=85)
    m3 = runs / "20261003-ds-m3av-maths/outputs/m3av-maths/dataset_v1.0_noaudio/MLS-03E08/ocr/seg_imgs/MLS-03E08-0669000.jpg"
    frame(m3).save(OUT / "ds_m3av.jpg", quality=85)
    grid([(cover("-6sEcMzz0ic"), "design_0000 · 17 phút", "#153898"), (cover("7HOFhLm29XI"), "design_0001 · 18 phút", "#153898")], 2, "ds_slidespeech.jpg")


if __name__ == "__main__":
    command, folder = sys.argv[1], Path(sys.argv[2])
    {"gate": lambda: gate(folder), "rank": lambda: rank(folder, sys.argv[3]),
     "retrieval": lambda: retrieval(folder, sys.argv[3]), "sync": lambda: sync(folder), "sweep": lambda: sweep(folder), "gate_examples": lambda: gate_examples(folder),
     "gate_crops": lambda: gate_crops_figure(folder),
     "gate_crops_skip": lambda: gate_crops_figure(folder, 9, "gate_crops_skip.jpg"), "reveal": lambda: reveal(folder), "datasets": lambda: datasets(folder)}[command]()
