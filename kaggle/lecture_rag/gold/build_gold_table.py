"""One HTML table of the whole gold set 1, for reading it through.

    python kaggle/lecture_rag/gold/build_gold_table.py kaggle/lecture_rag/gold/slide_boundaries <videos_dir>
"""

import html
import json
import subprocess
import sys
from pathlib import Path

OUT, VIDEOS = Path(sys.argv[1]), Path(sys.argv[2])
gold = json.loads((OUT / "gold.json").read_text(encoding="utf-8"))
draft = {v["video"]: v["events"] for v in json.loads((OUT / "draft_labels.json").read_text(encoding="utf-8"))["videos"]}
metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
legend = gold["legend"]


def frame(video, seconds):
    name = f"thumbs/{video}_t{seconds:08.2f}.jpg"
    if not (OUT / name).exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{seconds:.2f}", "-i", str(VIDEOS / f"{video}.mp4"),
                        "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "5", str(OUT / name)], check=True)
    return name


def verdict(pipeline, label):
    if label == "L":
        return "", "đổi khung (không tính)"
    if pipeline == "slide":
        return ("ok", "đúng: tách") if label in {"S", "C", "SC"} else ("bad", "tách thừa")
    if pipeline == "missed":
        return "bad", "bỏ sót"
    return ("bad", "gộp nhầm") if label in {"S", "C"} else ("ok", "đúng: gộp")


rows, counts = [], {}
for video in gold["videos"]:
    v = video["video"]
    for event in video["events"]:
        n, source = event["index"], draft[v][event["index"]]
        before, after = source["before"], source["after"]
        if event["pipeline"] == "missed" and abs(event["time_sec"] - source["time_sec"]) > 0.05:
            before, after = frame(v, max(0.0, event["time_sec"] - 1.5)), frame(v, event["time_sec"] + 1.5)
        cls, text = verdict(event["pipeline"], event["label"])
        counts.setdefault(v, {}).setdefault(text, 0)
        counts[v][text] += 1
        t = event["time_sec"]
        pending = "to confirm" in event["source"]
        rows.append(f"""<tr data-v="{v}" data-l="{event['label']}" data-c="{cls}">
<td>{v[4:]}<br><small>{'tuning' if video['split'] == 'tuning' else 'test'}</small></td><td>{n}</td>
<td>{int(t // 60)}:{t % 60:04.1f}</td>
<td><span class="p {event['pipeline']}">{ {'slide': 'tách slide', 'reveal': 'gộp', 'missed': 'bỏ sót'}[event['pipeline']] }</span></td>
<td><b>{event['label']}</b><br><small>{html.escape(legend[event['label']])}</small>{'<br><small class="pend">nhãn chờ xác nhận</small>' if pending else ''}</td>
<td class="{cls}">{text}</td>
<td><img loading="lazy" src="{before}"></td><td><img loading="lazy" src="{after}"></td></tr>""")

summary = []
for policy, title in (("scroll_is_slide", "Chính · cuộn sang nội dung mới là slide mới"), ("strict", "Phụ · chỉ tính slide PowerPoint")):
    for group in ("tuning", "held-out"):
        m = metrics[policy][f"_{group}"]
        summary.append(f"<tr><td>{title}</td><td>{'tuning' if group == 'tuning' else 'test'}</td><td>{m['precision']:.3f}</td>"
                       f"<td>{m['recall']:.3f}</td><td><b>{m['f1']:.3f}</b></td><td>{m['tp']}</td><td>{m['fp']}</td><td>{m['fn']}</td></tr>")
kinds = ["đúng: tách", "đúng: gộp", "tách thừa", "gộp nhầm", "bỏ sót", "đổi khung (không tính)"]
per_video = "".join(f"<tr><td>{v}</td>" + "".join(f"<td>{c.get(k, 0)}</td>" for k in kinds) + f"<td>{sum(c.values())}</td></tr>"
                    for v, c in counts.items())
videos = "".join(f'<option value="{v["video"]}">{v["video"]}</option>' for v in gold["videos"])
labels = "".join(f'<option value="{k}">{k}: {html.escape(t)}</option>' for k, t in legend.items() if k != "C")

page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Bộ gold ranh giới slide</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#667085; --line:#d9dde5; }}
body {{ margin:0; font:14px/1.45 system-ui,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }}
main {{ max-width:1250px; margin:auto; padding:16px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:20px 0 6px; }}
table {{ border-collapse:collapse; background:var(--card); width:100%; }}
td,th {{ border:1px solid var(--line); padding:4px 7px; vertical-align:middle; text-align:left; }}
th {{ background:#eef1f6; position:sticky; top:0; }}
.box {{ overflow-x:auto; }}
img {{ width:220px; display:block; border-radius:3px; cursor:zoom-in; }}
small {{ color:var(--muted); }} .pend {{ color:#b45309; }}
.ok {{ color:#1f7a45; font-weight:600; }} .bad {{ color:#c53030; font-weight:600; }}
.p {{ padding:1px 7px; border-radius:9px; font-size:12px; white-space:nowrap; }}
.slide {{ background:#fde4e4; }} .reveal {{ background:#e1ecff; }} .missed {{ background:#fff1d6; }}
.bar {{ display:flex; flex-wrap:wrap; gap:8px; margin:10px 0; position:sticky; top:0; background:var(--bg); padding:6px 0; z-index:2; }}
select {{ font:inherit; padding:3px 6px; }}
#zoom {{ position:fixed; inset:0; background:rgba(0,0,0,.85); display:none; align-items:center; justify-content:center; }}
#zoom img {{ width:auto; max-width:96vw; max-height:92vh; }}
</style></head><body><main>
<h1>Bộ gold 1: ranh giới slide (L25_V001–V010)</h1>
<p><small>{sum(len(v["events"]) for v in gold["videos"])} ranh giới (đã bỏ quảng cáo, logo, ảnh trường; bản đầy đủ ở gold_with_ads.json). Claude gán nhãn cho từng chỗ pipeline báo và tìm thêm chỗ pipeline bỏ sót; người đã kiểm tra hết.
V001, V002, V005, V007 đã dùng để chỉnh pipeline (tuning); 6 video còn lại là test.</small></p>
<h2>Điểm của pipeline</h2><div class="box"><table><tr><th>Cách tính</th><th>Nhóm</th><th>Precision</th><th>Recall</th><th>F1</th><th>Đúng</th><th>Tách thừa</th><th>Sót / gộp nhầm</th></tr>{''.join(summary)}</table></div>
<h2>Theo video</h2><div class="box"><table><tr><th>Video</th>{''.join(f'<th>{k}</th>' for k in kinds)}<th>Tổng</th></tr>{per_video}</table></div>
<h2>Toàn bộ ranh giới</h2>
<div class="bar"><select id="fv"><option value="">Tất cả video</option>{videos}</select>
<select id="fl"><option value="">Tất cả nhãn</option>{labels}</select>
<select id="fc"><option value="">Mọi kết quả</option><option value="bad">Chỉ lỗi của pipeline</option><option value="ok">Chỉ đúng</option></select>
<span id="n"></span></div>
<div class="box"><table id="t"><tr><th>Video</th><th>#</th><th>Thời điểm</th><th>Pipeline</th><th>Nhãn gold</th><th>Kết quả</th><th>Trước</th><th>Sau</th></tr>
{''.join(rows)}</table></div></main><div id="zoom"><img alt=""></div>
<script>
const f = () => {{ const v = fv.value, l = fl.value, c = fc.value; let n = 0;
  document.querySelectorAll("#t tr[data-v]").forEach(r => {{ const s = (!v || r.dataset.v === v) && (!l || r.dataset.l === l) && (!c || r.dataset.c === c);
    r.style.display = s ? "" : "none"; n += s; }}); document.getElementById("n").textContent = n + " dòng"; }};
[fv, fl, fc].forEach(e => e.onchange = f);
f();
document.getElementById("t").onclick = e => {{ if (e.target.tagName === "IMG") {{ zoom.querySelector("img").src = e.target.src; zoom.style.display = "flex"; }} }};
zoom.onclick = () => zoom.style.display = "none";
</script></body></html>"""
(OUT / "gold_table.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "gold_table.html", len(rows), "rows")
