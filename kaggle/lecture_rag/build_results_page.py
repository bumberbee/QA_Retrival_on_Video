"""Browse page of fast-pipeline results: one row per video, and every segment of a video with its
keyframe, OCR text, figure regions and speech.

    python kaggle/lecture_rag/build_results_page.py <out_dir> <export_dir> [<export_dir> ...]
        [--timings <fast_pipeline_timings.json> ...] [--meta <folder of L25_Vxxx.json>]

<export_dir> holds <video>/04_bundle/segments.json + keyframes/ (fast_pipeline_export unpacked, or a
run's lecture_rag/). Writes <out_dir>/index.html, data.js and thumbs/<video>/seg_xxxxx.jpg.
"""

import argparse
import json
from pathlib import Path

from PIL import Image


def clean(text, limit):
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " …"


def video_rows(folder, out, timing, meta):
    segments = json.loads((folder / "04_bundle" / "segments.json").read_text(encoding="utf-8"))
    video = folder.name
    (out / "thumbs" / video).mkdir(parents=True, exist_ok=True)
    rows = []
    for segment in segments:
        visual = segment.get("visual") or {}
        name = f"seg_{segment['sequence']:05d}.jpg"
        source = folder / "04_bundle" / "keyframes" / name
        if not source.exists():
            source = folder / "04_bundle" / "keyframes" / f"ad_{segment['sequence']:05d}.jpg"
        thumb = out / "thumbs" / video / name
        if source.exists() and not thumb.exists():
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((360, 360))
                image.save(thumb, quality=72)
        status = visual.get("ocr_status") or ("skipped_ad" if segment.get("content_kind") == "ad" else "")
        rows.append({
            "seq": segment["sequence"], "start": round(segment["start_sec"], 1), "end": round(segment["end_sec"], 1),
            "kind": segment.get("content_kind", ""), "status": status,
            "ad": segment.get("content_kind") == "ad" or bool(segment.get("ad_flagged")),
            "gate": visual.get("ocr_gate_score"), "reused": visual.get("ocr_reused_from"),
            "title": clean(visual.get("title_clean") or visual.get("title"), 160),
            "ocr": clean(visual.get("ocr_text_clean") or visual.get("ocr_text"), 3000),
            "regions": [{"kind": r.get("kind", ""), "text": clean((r.get("description") or "") + " " + (r.get("text") or ""), 400)}
                        for r in visual.get("regions") or []],
            "speech": clean((segment.get("speech") or {}).get("text"), 1500),
            "thumb": f"thumbs/{video}/{name}" if thumb.exists() else "",
        })
    info = meta.get(video, {})
    return {"video": video, "title": info.get("title", ""), "url": info.get("watch_url", ""),
            "duration": timing.get("duration_sec") or (segments[-1]["end_sec"] if segments else 0),
            "states": timing.get("visual_states"), "slides": len(rows),
            "read": sum(r["status"] == "read" for r in rows),
            "skipped": sum(r["status"] == "skipped_no_content" for r in rows),
            "ads": sum(r["ad"] for r in rows),
            "t01": timing.get("01_sec"), "t02": timing.get("02_sec"), "t03": timing.get("03_sec"),
            "t04": timing.get("04_sec"), "t05": timing.get("05_sec"), "segments": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--timings", nargs="*", default=[])
    ap.add_argument("--meta", default="")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    for path in a.timings:
        for row in json.loads(Path(path).read_text(encoding="utf-8"))["videos"]:
            timings[row["video_id"]] = row
    meta = {}
    if a.meta:
        for path in Path(a.meta).glob("*.json"):
            meta[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    folders = sorted({p.parent.parent for export in a.exports for p in Path(export).glob("*/04_bundle/segments.json")},
                     key=lambda p: p.name)
    videos = []
    for folder in folders:
        videos.append(video_rows(folder, out, timings.get(folder.name, {}), meta))
        print(folder.name, videos[-1]["slides"], "đoạn")
    (out / "data.js").write_text("const VIDEOS = " + json.dumps(videos, ensure_ascii=False) + ";\n", encoding="utf-8")
    (out / "index.html").write_text(PAGE, encoding="utf-8")
    print(out / "index.html", len(videos), "video")


PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kết quả trích xuất</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#1c2230;--muted:#667085;--line:#dde1e8;--brand:#153898;--ok:#1f9d55;--warn:#c27c0e;--bad:#c0392b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#11141a;--card:#1a1f28;--ink:#e6e9ef;--muted:#98a2b3;--line:#2c3340;--brand:#7d9cff}}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 system-ui,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}
header{padding:16px;border-bottom:1px solid var(--line);background:var(--card)}
h1{margin:0 0 4px;font-size:20px}.muted{color:var(--muted)}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}
.kpi{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 12px;min-width:120px}
.kpi b{display:block;font-size:20px}
main{display:grid;grid-template-columns:minmax(0,560px) minmax(0,1fr);gap:16px;padding:16px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.tools{display:flex;gap:8px;padding:10px;border-bottom:1px solid var(--line)}
input,select{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
input{flex:1;min-width:0}
.list{max-height:calc(100vh - 230px);overflow:auto}
table{width:100%;border-collapse:collapse}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
th{position:sticky;top:0;background:var(--card);cursor:pointer;font-size:12px;color:var(--muted)}
tr.v{cursor:pointer}tr.v:hover,tr.v.on{background:color-mix(in srgb,var(--brand) 10%,transparent)}
.detail{padding:14px;max-height:calc(100vh - 170px);overflow:auto}
.vhead h2{margin:0;font-size:17px}
.bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:8px 0}
.bar span{display:block}
.filters{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}
.filters button{font:inherit;font-size:12px;padding:3px 10px;border:1px solid var(--line);border-radius:14px;background:var(--card);color:var(--ink);cursor:pointer}
.filters button.on{background:var(--brand);color:#fff;border-color:var(--brand)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.seg{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--card)}
.seg img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;background:#000;cursor:zoom-in}
.seg .b{padding:8px}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px;color:#fff;margin-right:4px}
.read{background:var(--ok)}.skipped_no_content{background:var(--warn)}.skipped_ad{background:var(--bad)}.reused{background:#6b5bd2}
details{margin-top:4px}summary{cursor:pointer;color:var(--muted);font-size:12px}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.45 ui-monospace,Consolas,monospace;background:var(--bg);padding:6px;border-radius:6px;margin:4px 0;max-height:260px;overflow:auto}
.zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9}
.zoom img{max-width:95vw;max-height:92vh}
</style></head><body>
<header><h1>Kết quả trích xuất · <span id="count"></span></h1>
<div class="muted">Pipeline nhanh (01 → 05) trên Kaggle T4 x2. Bấm một video để xem từng đoạn: keyframe, chữ OCR, vùng hình, lời giảng.</div>
<div class="kpis" id="kpis"></div></header>
<main>
<section class="panel"><div class="tools"><input id="q" placeholder="Tìm mã hoặc tên video…"></div>
<div class="list"><table><thead><tr>
<th data-k="video">Video</th><th data-k="duration">Phút</th><th data-k="slides">Đoạn</th><th data-k="read">OCR</th><th data-k="skipped">Bỏ</th><th data-k="total">Xử lý (phút)</th>
</tr></thead><tbody id="rows"></tbody></table></div></section>
<section class="panel"><div class="detail" id="detail"><p class="muted">Chọn một video bên trái.</p></div></section>
</main>
<div class="zoom" id="zoom"><img alt=""></div>
<script src="data.js"></script>
<script>
const $ = s => document.querySelector(s);
const clock = s => Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const num = (x, d = 1) => x == null ? '–' : x.toLocaleString('vi-VN', {maximumFractionDigits: d});
for (const v of VIDEOS) v.total = [v.t01, v.t02, v.t04, v.t05].every(x => x == null) ? null : ((v.t01||0) + (v.t02||0) + (v.t04||0) + (v.t05||0)) / 60;
const short = t => (t || '').replace(/^BÍ QUYẾT ÔN THI THPT 2024\s*/i, '').replace(/\s+/g, ' ').slice(0, 44);
const sum = k => VIDEOS.reduce((a, v) => a + (v[k] || 0), 0);
const timed = VIDEOS.filter(v => v.total != null);
$('#count').textContent = VIDEOS.length + ' video';
$('#kpis').innerHTML = [
  ['Giờ video', num(sum('duration') / 3600)], ['Đoạn (slide)', num(sum('slides'), 0)],
  ['Đọc OCR', num(sum('read'), 0)], ['Bộ gác bỏ OCR', num(sum('skipped'), 0) + ' (' + num(100 * sum('skipped') / Math.max(1, sum('read') + sum('skipped'))) + '%)'],
  ['Quảng cáo ẩn', num(sum('ads'), 0)],
  ['Xử lý TB / video', timed.length ? num(timed.reduce((a, v) => a + v.total, 0) / timed.length) + ' phút' : '–'],
].map(([k, v]) => `<div class="kpi"><span class="muted">${k}</span><b>${v}</b></div>`).join('');
let sortKey = 'video', sortDir = 1, current = null, filter = 'all';
function renderRows() {
  const q = $('#q').value.toLowerCase();
  const list = VIDEOS.filter(v => (v.video + ' ' + v.title).toLowerCase().includes(q))
    .sort((a, b) => ((a[sortKey] ?? -1) > (b[sortKey] ?? -1) ? 1 : (a[sortKey] ?? -1) < (b[sortKey] ?? -1) ? -1 : 0) * sortDir);
  $('#rows').innerHTML = list.map(v => `<tr class="v${v === current ? ' on' : ''}" data-id="${v.video}"><td><b>${v.video}</b><br><span class="muted">${esc(short(v.title))}</span></td>
    <td>${num(v.duration / 60)}</td><td>${v.slides}</td><td>${v.read}</td><td>${v.skipped}</td><td>${num(v.total)}</td></tr>`).join('');
}
document.querySelectorAll('th').forEach(th => th.onclick = () => { const k = th.dataset.k; sortDir = sortKey === k ? -sortDir : (k === 'video' ? 1 : -1); sortKey = k; renderRows(); });
$('#q').oninput = renderRows;
$('#rows').onclick = e => { const tr = e.target.closest('tr'); if (!tr) return; current = VIDEOS.find(v => v.video === tr.dataset.id); filter = 'all'; history.replaceState(null, '', '#' + current.video); renderRows(); renderDetail(); };
function renderDetail() {
  const v = current, steps = [['01 tách slide', v.t01, '#153898'], ['02 OCR', v.t02, '#f6b21b'], ['04 đồng bộ', v.t04, '#1f9d55'], ['05 chỉ mục', v.t05, '#6b5bd2']];
  const total = steps.reduce((a, s) => a + (s[1] || 0), 0);
  const segs = v.segments.filter(s => filter === 'all' || (filter === 'reused' ? s.reused : filter === 'skipped_ad' ? s.ad : s.status === filter));
  $('#detail').innerHTML = `<div class="vhead"><h2>${v.video} · ${esc(v.title)}</h2>
    <div class="muted">${num(v.duration / 60)} phút video · ${v.states ?? '–'} trạng thái hình → ${v.slides} đoạn · OCR ${v.read}, bỏ ${v.skipped}${v.ads ? ', quảng cáo ' + v.ads : ''}
    ${v.url ? ` · <a href="${esc(v.url)}" target="_blank" rel="noopener">YouTube</a>` : ''}</div>
    ${total ? `<div class="bar">${steps.map(s => `<span title="${s[0]}: ${num((s[1]||0)/60)} phút" style="width:${100 * (s[1]||0) / total}%;background:${s[2]}"></span>`).join('')}</div>
    <div class="muted">${steps.map(s => `${s[0]} ${num((s[1]||0)/60)}′`).join(' · ')} · 03 lời giảng ${num((v.t03||0)/60)}′ (chạy song song)</div>` : ''}</div>
    <div class="filters">${[['all','Tất cả'],['read','Đọc OCR'],['skipped_no_content','Bị bỏ OCR'],['skipped_ad','Quảng cáo'],['reused','Dùng lại']]
      .map(([k, l]) => `<button data-f="${k}" class="${filter === k ? 'on' : ''}">${l}</button>`).join('')}</div>
    <div class="grid">${segs.map(s => `<div class="seg">${s.thumb ? `<img loading="lazy" src="${s.thumb}" alt="">` : ''}
      <div class="b">${s.ad ? '<span class="tag skipped_ad">quảng cáo (ẩn)</span>' : ''}<span class="tag ${s.status}">${({read:'đọc OCR',skipped_no_content:'bỏ OCR',skipped_ad:'quảng cáo'})[s.status] || s.status || s.kind}</span>${s.reused ? '<span class="tag reused">dùng lại</span>' : ''}
      <b>#${s.seq}</b> <span class="muted">${clock(s.start)}–${clock(s.end)}${s.gate != null ? ' · gác ' + num(s.gate, 2) : ''}</span>
      ${s.title ? `<div><b>${esc(s.title)}</b></div>` : ''}
      ${s.ocr ? `<details><summary>Chữ OCR (${s.ocr.length} ký tự)</summary><pre>${esc(s.ocr)}</pre></details>` : ''}
      ${s.regions.length ? `<details><summary>Vùng hình (${s.regions.length})</summary>${s.regions.map(r => `<pre>[${esc(r.kind)}] ${esc(r.text)}</pre>`).join('')}</details>` : ''}
      ${s.speech ? `<details><summary>Lời giảng</summary><pre>${esc(s.speech)}</pre></details>` : ''}</div></div>`).join('')}</div>`;
  $('#detail').querySelectorAll('.filters button').forEach(b => b.onclick = () => { filter = b.dataset.f; renderDetail(); });
  $('#detail').querySelectorAll('.seg img').forEach(img => img.onclick = () => { $('#zoom img').src = img.src; $('#zoom').style.display = 'flex'; });
}
$('#zoom').onclick = () => $('#zoom').style.display = 'none';
renderRows();
if (location.hash) { current = VIDEOS.find(v => v.video === location.hash.slice(1)) || null; if (current) { renderRows(); renderDetail(); } }
</script></body></html>
"""

if __name__ == "__main__":
    main()
