"""Review page for gold set 2 (speech), built from stage-03 transcripts of the 12 Vietnamese test videos.

    python kaggle/lecture_rag/gold/build_asr_review.py <asr_root> <videos.csv> <out_dir> [--per-video 2] [--sec 60] [--seed 0]
        [--video-dir D:/thesis-data/vn12-videos]

<asr_root>/<video>/03_audio/transcript.json (runs/asr-all). videos.csv is the vn-lecture-test12 list
(video_id, youtube_url, ...). Picks --per-video clips of --sec seconds per video, one per equal slice
of the video at a random offset (not chosen by looking at the transcript), each with the ASR text
pre-filled. The labeller plays the clip (YouTube embed, start/end) and corrects the text to what was
said; the page exports asr_gold.json for eval_asr.py. With --video-dir the clip plays from the local
<video_id>.mp4 instead (a YouTube embed refuses to play in a page opened from a file: error 153).
"""

import argparse
import csv
import html
import json
import random
from pathlib import Path


def clips(transcript, per_video, sec, rng):
    words = transcript["words"]
    duration = max(w["end_sec"] for w in words)
    out = []
    for k in range(per_video):
        lo, hi = duration * k / per_video, duration * (k + 1) / per_video - sec
        start = round(rng.uniform(lo, max(lo, hi)))
        text = " ".join(w["text"].strip() for w in words if start <= w["start_sec"] < start + sec)
        out.append({"start": start, "end": start + sec, "asr": text})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asr_root"); ap.add_argument("videos_csv"); ap.add_argument("out_dir")
    ap.add_argument("--per-video", type=int, default=2); ap.add_argument("--sec", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-dir", default="", help="folder of <video_id>.mp4 to play clips locally")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    items = []
    for row in csv.DictReader(open(a.videos_csv, encoding="utf-8")):
        path = Path(a.asr_root) / row["video_id"] / "03_audio" / "transcript.json"
        if not path.exists():
            print("thiếu transcript", row["video_id"]); continue
        for n, clip in enumerate(clips(json.loads(path.read_text(encoding="utf-8")), a.per_video, a.sec, rng)):
            items.append({"id": f"{row['video_id']}#{n}", "video_id": row["video_id"], "title": row.get("title", ""), **clip})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "asr_review_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    video_dir = Path(a.video_dir).resolve().as_uri() if a.video_dir else ""
    page = TEMPLATE.replace("__VIDEO_DIR__", json.dumps(video_dir)).replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    (out / "asr_review.html").write_text(page, encoding="utf-8")
    print(out / "asr_review.html", len(items), "đoạn")


TEMPLATE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gán nhãn lời giảng</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#667085;--line:#d9dde5;--ok:#1f9d55}
body{margin:0;font:14px/1.5 system-ui,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:17px;margin:0 0 4px}.note{color:var(--muted);font-size:12.5px}
main{max-width:1100px;margin:auto;padding:12px 16px 80px}
.card{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--line);border-radius:8px;padding:10px;margin:10px 0}
.card.done{border-left-color:var(--ok)}
.top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.body{display:grid;grid-template-columns:minmax(0,420px) minmax(0,1fr);gap:10px}
@media(max-width:800px){.body{grid-template-columns:1fr}}
video{width:100%;border-radius:6px;background:#000}
iframe{width:100%;aspect-ratio:16/9;border:0;border-radius:6px;background:#000}
textarea{width:100%;min-height:200px;font:inherit;padding:8px;border:1px solid var(--line);border-radius:6px;box-sizing:border-box}
button{font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
button.on{background:var(--ok);color:#fff;border-color:var(--ok)}
</style></head><body>
<header><h1>Gán nhãn lời giảng (WER) · <span id="count"></span></h1>
<div class="note">Bấm phát đoạn video (tự dừng ở cuối đoạn; bấm phát lại để nghe lại từ đầu đoạn), sửa ô chữ cho <b>đúng từng từ đã nói</b>: thêm từ thiếu, xóa từ thừa,
sửa từ nghe nhầm. Không cần dấu câu, viết hoa. Chỗ nghe không rõ ghi <code>[?]</code>. Xong bấm <b>Đã kiểm</b>.
Dữ liệu tự lưu trong trình duyệt; cuối cùng bấm <button id="export">Xuất asr_gold.json</button></div></header>
<main id="list"></main>
<script>
const ITEMS = __ITEMS__;
const VIDEO_DIR = __VIDEO_DIR__;
const KEY = 'asr_review_v1';
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} count(); };
const count = () => document.getElementById('count').textContent =
  ITEMS.filter(i => state[i.id] && state[i.id].checked).length + ' / ' + ITEMS.length + ' đã kiểm';
const clock = s => Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
const list = document.getElementById('list');
for (const it of ITEMS) {
  const st = state[it.id] || (state[it.id] = {text: it.asr, checked: false});
  const card = document.createElement('div');
  card.className = 'card' + (st.checked ? ' done' : '');
  card.innerHTML = `<div class="top"><b>${it.video_id} · ${clock(it.start)}–${clock(it.end)}</b><span class="note"></span></div>
    <div class="body"><div><button class="play">▶ Tải đoạn video</button><div class="player"></div></div>
    <div><textarea></textarea><div style="margin-top:6px;display:flex;gap:6px">
    <button class="chk">Đã kiểm</button><button class="reset">Khôi phục chữ máy</button></div></div></div>`;
  card.querySelector('.note').textContent = it.title;
  const ta = card.querySelector('textarea'), chk = card.querySelector('.chk');
  ta.value = st.text; chk.classList.toggle('on', st.checked);
  ta.oninput = () => { st.text = ta.value; save(); };
  chk.onclick = () => { st.checked = !st.checked; chk.classList.toggle('on', st.checked); card.classList.toggle('done', st.checked); save(); };
  card.querySelector('.reset').onclick = () => { ta.value = st.text = it.asr; save(); };
  card.querySelector('.play').onclick = () => {
    const box = card.querySelector('.player');
    if (VIDEO_DIR) {
      box.innerHTML = `<video controls autoplay src="${VIDEO_DIR}/${it.video_id}.mp4#t=${it.start},${it.end}"></video>`;
      const v = box.querySelector('video');
      v.ontimeupdate = () => { if (v.currentTime >= it.end) v.pause(); };
      v.onerror = () => { box.textContent = 'Không mở được ' + VIDEO_DIR + '/' + it.video_id + '.mp4'; };
      return;
    }
    box.innerHTML =
    `<iframe src="https://www.youtube-nocookie.com/embed/${it.video_id}?start=${it.start}&end=${it.end}&autoplay=1&rel=0" allow="autoplay; encrypted-media" allowfullscreen></iframe>`; };
  list.appendChild(card);
}
count();
document.getElementById('export').onclick = () => {
  const out = ITEMS.map(i => ({id: i.id, video_id: i.video_id, start: i.start, end: i.end, asr: i.asr,
                               gold: state[i.id].text, checked: state[i.id].checked}));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = 'asr_gold.json'; a.click();
};
</script></body></html>
"""

if __name__ == "__main__":
    main()
