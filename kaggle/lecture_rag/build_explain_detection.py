"""Writes explain_slide_detection.ipynb: stage 01 run step by step on one YouTube video,
printing every intermediate output. It calls the same functions as run_slide_detection
(notebook_code/detection.py) and checks at the end that the result is identical.

    python kaggle/lecture_rag/build_explain_detection.py
    jupyter nbconvert --to notebook --execute --inplace kaggle/lecture_rag/notebooks/explain_slide_detection.ipynb
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Tách slide (stage 01) từng bước

Notebook chạy thuật toán tách slide của pipeline trên **một video YouTube** và in ra kết quả của
**từng bước** để quan sát. Mỗi bước gọi đúng hàm trong `notebook_code/detection.py`; ô cuối chạy lại
`run_slide_detection` (hàm pipeline dùng thật) và kiểm tra kết quả trùng khớp.

Các bước:

1. Tải video, đọc thông tin
2. Quét video một lần: lấy mẫu 1 Hz, và 8 Hz để tinh chỉnh
3. "Chữ ký" của một khung hình: ảnh xám mờ + đường biên
4. Khoảng cách giữa các mẫu liên tiếp
5. Tìm ranh giới ổn định (thay đổi ≥ 0.055, sau đó giữ yên ≤ 0.035 trong 2 mẫu)
6. Tinh chỉnh ranh giới bằng các khung 8 Hz
7. Mặt nạ chuyển động: bỏ vùng luôn động (giảng viên, nền động) rồi tìm thêm ranh giới
8. Tách thay đổi "ẩn": tĩnh → động, động → lắng lại
9. Chọn ảnh nét nhất cho mỗi trạng thái
10. Gộp slide hiện dần (nội dung cũ được giữ ≥ 85% và có thêm nội dung)
11. Kết quả cuối: danh sách slide
12. Kiểm chứng với `run_slide_detection`
""")

code(r'''
VIDEO_URL = "https://youtu.be/vk93aEGb9Bo"
VIDEO_ID = "YT_vk93aEGb9Bo"
MAX_HEIGHT = 720          # tải bản 720p là đủ cho tách slide

# Tham số mặc định của pipeline (giống fast_pipeline_l25)
SAMPLE_HZ, FINE_HZ = 1.0, 8.0
CHANGE, STABLE, STABLE_SAMPLES, MIN_INTERVAL = 0.055, 0.035, 2, 2.0
ROI, IGNORED = (0, 0, 1, 1), ()
''')

code(r'''
import json, shutil, subprocess, sys, time, types
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import HTML, display

# chạy từ notebooks/, từ kaggle/lecture_rag/ hoặc từ thư mục gốc repo
ROOT = next(p for p in (Path.cwd().parent, Path.cwd(), Path.cwd() / "kaggle" / "lecture_rag")
            if (p / "notebook_code").exists())
WORK = ROOT / "runs" / f"explain_{VIDEO_ID}"
WORK.mkdir(parents=True, exist_ok=True)

d = types.ModuleType("detection")
for name in ("common.py", "detection.py"):
    exec(compile((ROOT / "notebook_code" / name).read_text(encoding="utf-8"), name, "exec"), d.__dict__)

def table(rows):
    """Small HTML table (list of dicts), so the notebook needs no pandas."""
    if not rows:
        return print("(trống)")
    head = "".join(f"<th>{key}</th>" for key in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{row.get(key, '')}</td>" for key in rows[0]) + "</tr>" for row in rows)
    display(HTML(f"<table><tr>{head}</tr>{body}</table>"))

def clock(seconds):
    return f"{int(seconds) // 60}:{seconds % 60:04.1f}"

def grid(images, titles, columns=4, width=4.2):
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * width, rows * width * 0.62), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, title in zip(axes.flat, images, titles):
        ax.imshow(image, cmap="gray" if np.ndim(image) == 2 else None)
        ax.set_title(title, fontsize=9)
    plt.tight_layout(); plt.show()
print("Thư mục làm việc:", WORK)
''')

md("## Bước 1 — Tải video và đọc thông tin")
code(r'''
VIDEO = WORK / f"{VIDEO_ID}.mp4"
if not VIDEO.exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"], check=True)
    command = [sys.executable, "-m", "yt_dlp", "--no-playlist", "-f",
               f"bv*[height<={MAX_HEIGHT}][ext=mp4]+ba[ext=m4a]/b[height<={MAX_HEIGHT}]",
               "--merge-output-format", "mp4", "-o", str(VIDEO), VIDEO_URL]
    if shutil.which("node"):
        command[3:3] = ["--js-runtimes", "node"]
    subprocess.run(command, check=True)
source = d.probe_video(VIDEO)
print(json.dumps(source, indent=1, ensure_ascii=False))
''')

md("""
## Bước 2 — Quét video một lần

Giải mã video **một lần duy nhất** (không tua), bỏ các khung không tham chiếu (NONREF) để nhanh gấp đôi.

- Mỗi giây giữ **1 mẫu**: chữ ký, độ nét, ảnh lưu lại để chọn keyframe.
- Giữa hai mẫu giữ thêm **chữ ký 8 Hz** (`fine`) để sau này tinh chỉnh thời điểm ranh giới mà không cần tua lại.
""")
code(r'''
records, samples = [], []
started = time.monotonic()
for sample in d.scan_frames(VIDEO, source, WORK, SAMPLE_HZ, FINE_HZ, ROI, IGNORED, skip_nonref=True, records=records):
    samples.append(sample)
print(f"{len(samples)} mẫu 1 Hz, {sum(len(s['fine']) for s in samples)} khung 8 Hz, {time.monotonic() - started:.1f} s")
table(([{"t": round(s["time_sec"], 2), "độ nét": round(s["sharpness"], 1),
                       "số khung 8 Hz trước đó": len(s["fine"]), "ảnh": s["image_path"]} for s in samples[:12]]))
step = max(1, len(samples) // 16)
picks = samples[::step][:16]
grid([Image.open(WORK / s["image_path"]) for s in picks], [f"mẫu t={clock(s['time_sec'])}" for s in picks])
''')

md("""
## Bước 3 — "Chữ ký" của một khung hình

Mỗi khung được thu về 240×135, gồm 2 kênh: **ảnh xám làm mờ** (bắt đổi màu, đổi nền) và **đường biên Canny**
(bắt chữ mới trên nền trắng giống nhau). Khoảng cách giữa hai chữ ký là max của hai kênh; kênh xám bỏ 3% điểm
thay đổi nhiều nhất để không bị con trỏ chuột làm nhiễu.
""")
code(r'''
example = samples[len(samples) // 3]
fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
axes[0].imshow(Image.open(WORK / example["image_path"])); axes[0].set_title(f"khung t={clock(example['time_sec'])}")
axes[1].imshow(example["signature"][0], cmap="gray"); axes[1].set_title("kênh 1: xám, làm mờ (240×135)")
axes[2].imshow(example["signature"][1], cmap="gray"); axes[2].set_title("kênh 2: đường biên")
for ax in axes: ax.axis("off")
plt.show()
''')

md("""
## Bước 4 — Khoảng cách giữa các mẫu liên tiếp

Đỉnh cao là lúc hình thay đổi. Đường đỏ là ngưỡng **0.055**: vượt ngưỡng mới được xét là ranh giới.
""")
code(r'''
times = np.array([s["time_sec"] for s in samples])
steps = np.array([0.0] + [d.signature_distance(a["signature"], b["signature"]) for a, b in zip(samples, samples[1:])])
fig, ax = plt.subplots(figsize=(16, 3.5))
ax.plot(times, steps, lw=0.8); ax.axhline(CHANGE, color="red", lw=1, label="ngưỡng thay đổi 0.055")
ax.set_xlabel("giây"); ax.set_ylabel("khoảng cách"); ax.legend(); plt.show()
print(f"{(steps >= CHANGE).sum()} / {len(steps)} bước vượt ngưỡng")
''')

md("""
## Bước 5 — Tìm ranh giới ổn định

Một ranh giới được **chấp nhận** khi:

1. mẫu mới khác **trạng thái tham chiếu** ≥ 0.055;
2. mẫu kế tiếp giống mẫu mới (≤ 0.035), tức hình đã **đứng yên** ở trạng thái mới trong 2 mẫu;
3. cách ranh giới trước ≥ 2 giây.

Nhờ vậy, chuyển cảnh mờ dần hay video đang chạy sẽ không sinh ra hàng loạt ranh giới.
""")
code(r'''
stable_bounds, candidates, _ = d.detect_stable_boundaries(
    iter(samples), source["duration_sec"], CHANGE, STABLE, STABLE_SAMPLES, MIN_INTERVAL)
print(f"{len(candidates)} ranh giới ổn định")
table(([{"#": i, "thời điểm (1 Hz)": clock(c["time_sec"]),
                       "khác tham chiếu": round(d.signature_distance(c["old_signature"], c["signature"]), 3)}
                      for i, c in enumerate(candidates)]))
fig, ax = plt.subplots(figsize=(16, 3.5))
ax.plot(times, steps, lw=0.8)
for c in candidates:
    ax.axvline(c["time_sec"], color="green", lw=0.8, alpha=0.7)
ax.axhline(CHANGE, color="red", lw=1); ax.set_title("xanh lá = ranh giới được chấp nhận"); plt.show()
''')

md("""
## Bước 6 — Tinh chỉnh ranh giới bằng khung 8 Hz

Mẫu 1 Hz có thể trễ tới 1 giây. Trong các khung 8 Hz ngay trước ranh giới, chọn khung **sớm nhất** đã khác
trạng thái cũ và đã giống trạng thái mới.
""")
code(r'''
refined = [0.0]
rows = []
for c in candidates:
    better = d.refine_from_buffer(c, CHANGE, STABLE)
    refined.append(max(refined[-1] + MIN_INTERVAL, min(better, c["time_sec"])))
    rows.append({"1 Hz": round(c["time_sec"], 2), "sau tinh chỉnh": round(refined[-1], 2),
                 "sớm hơn (s)": round(c["time_sec"] - refined[-1], 2)})
boundaries = refined + [source["duration_sec"]]
table((rows))
''')

md("""
## Bước 7 — Mặt nạ chuyển động

Điểm ảnh **thay đổi trong hơn 30% số giây** của video (giảng viên cử động, nền động) bị coi là "vùng bận".
Chạy lại bước 5 với vùng bận bị che để bắt những slide chỉ đổi ở phần còn lại. Ranh giới mới chỉ được thêm
nếu cách mọi ranh giới cũ ≥ 2 giây.
""")
code(r'''
mask = d.motion_mask(records)
if mask is None:
    print("Không có vùng bận rõ ràng (0% hoặc > 85% khung hình): bỏ qua bước này.")
    masked_extra = []
else:
    masked = d.masked_boundaries(records, source["duration_sec"], mask, CHANGE, STABLE, STABLE_SAMPLES, MIN_INTERVAL)
    merged = d.union_boundaries(boundaries, masked, MIN_INTERVAL)
    masked_extra = sorted(set(merged) - set(boundaries))
    boundaries = merged
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    axes[0].imshow(Image.open(WORK / samples[len(samples) // 2]["image_path"])); axes[0].set_title("khung mẫu")
    axes[1].imshow(mask, cmap="Reds"); axes[1].set_title(f"vùng bận: {mask.mean():.1%} khung hình")
    for ax in axes: ax.axis("off")
    plt.show()
print("Ranh giới thêm nhờ mặt nạ:", [clock(t) for t in masked_extra] or "không có")
''')

md("""
## Bước 8 — Tách thay đổi "ẩn"

Bước 5 cần hình đứng yên sau khi đổi, nên bỏ lỡ hai trường hợp:

- **tĩnh → động**: slide đứng yên rồi chuyển sang đoạn phim chạy liên tục (quảng cáo, cảnh quay);
- **động → lắng lại**: đoạn phim chạy rồi dừng ở một cảnh chỉ nhúc nhích nhẹ.

Mỗi đoạn giữa hai ranh giới được kiểm tra hai trường hợp này.
""")
code(r'''
checked, hidden = [boundaries[0]], []
for start, end in zip(boundaries, boundaries[1:]):
    splits = d.hidden_change_splits(records, start, end, CHANGE, STABLE, MIN_INTERVAL)
    hidden += splits
    checked += splits + [end]
boundaries = checked
print("Ranh giới thêm ở bước này:", [clock(t) for t in hidden] or "không có")
print(f"Tổng cộng {len(boundaries) - 1} trạng thái hình ảnh")
''')

md("""
## Bước 9 — Ảnh nét nhất cho mỗi trạng thái

Trong mỗi trạng thái, chọn mẫu có **độ nét (phương sai Laplacian) cao nhất**, ưu tiên mẫu không sát hai đầu đoạn.
""")
code(r'''
states, views = [], []
for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
    pick = d.pick_scanned_keyframe(records, start, end)
    if pick is not None:
        image = Image.open(WORK / pick["image_path"]).convert("RGB")
        at, sharp = pick["time_sec"], pick["sharpness"]
    else:
        image, at, sharp = d.select_keyframe(VIDEO, source, start, end)
    states.append({"state": index, "start": round(start, 2), "end": round(end, 2), "dài (s)": round(end - start, 1),
                   "keyframe": round(at, 2), "độ nét": round(sharp, 1), "image": image})
    views.append(d.reveal_view(image, ROI, IGNORED))
table(([{k: v for k, v in s.items() if k != "image"} for s in states]))
grid([s["image"] for s in states], [f"trạng thái {s['state']} · {clock(s['start'])}–{clock(s['end'])}" for s in states])
''')

md("""
## Bước 10 — Gộp slide hiện dần

So từng trạng thái với trạng thái trước đó (chỉ xét **nét chữ và hình vẽ**; phần có mặt ở hầu hết các trạng thái như
logo, khung tiêu đề bị bỏ qua). **Gộp** khi cả ba điều kiện đúng:

- nội dung cũ còn giữ ≥ **85%** (`content_retained`);
- ảnh thay đổi ≤ **25%** (`image_change`);
- nội dung mới ≥ 0.9 lần nội dung cũ (`content_growth`).

Ngoài ra, trạng thái mới còn phải giữ ≥ 60% nội dung của trạng thái **đầu nhóm** (`anchor_retained`), để các lần
gộp không nối dây sang một slide khác.
""")
code(r'''
groups, audit = d.group_reveal_states(views)
names = {"merge_content_kept": "GỘP (hiện thêm)", "split_content_changed": "tách: nội dung đổi",
         "split_little_content": "tách: slide cũ gần như trống", "split_anchor_lost": "tách: mất nội dung đầu nhóm"}
from collections import Counter
rows = [{"trước": a["previous_state"], "sau": a["current_state"], "quyết định": names.get(a["decision"], a["decision"]),
         **{k: a[k] for k in ("content_retained", "image_change", "content_growth", "anchor_retained", "old_content_pixels") if k in a}}
        for a in audit]
table(rows)
print(Counter(r["quyết định"] for r in rows))
for number, group in enumerate(groups):
    if len(group) > 1:
        grid([states[i]["image"] for i in group], [f"slide {number} · trạng thái {i}" for i in group], columns=min(4, len(group)))
''')

md("""
## Bước 11 — Kết quả cuối: danh sách slide

Mỗi nhóm là một slide. Ảnh đại diện là **trạng thái cuối** của nhóm, tức lúc slide đã hiện đầy đủ nhất.
""")
code(r'''
slides = [{"slide": n, "start": clock(states[g[0]]["start"]), "end": clock(states[g[-1]]["end"]),
           "số trạng thái": len(g), "image": states[g[-1]]["image"]} for n, g in enumerate(groups)]
print(f"{len(states)} trạng thái → {len(slides)} slide")
table(([{k: v for k, v in s.items() if k != "image"} for s in slides]))
grid([s["image"] for s in slides], [f"slide {s['slide']} · {s['start']}–{s['end']}" for s in slides])
''')

md("""
## Bước 12 — Kiểm chứng với hàm của pipeline

Chạy `run_slide_detection` (hàm pipeline gọi thật, cùng tham số) và so sánh ranh giới từng slide với kết quả
từng bước ở trên.
""")
code(r'''
shutil.rmtree(WORK / "_scan", ignore_errors=True)
started = time.monotonic()
_, manifest, intervals = d.run_slide_detection(VIDEO, WORK / "pipeline", VIDEO_ID)
print(f"Pipeline: {manifest['visual_state_count']} trạng thái, {len(intervals)} slide, {time.monotonic() - started:.0f} s")
ours = [(round(states[g[0]]["start"], 2), round(states[g[-1]]["end"], 2)) for g in groups]
theirs = [(round(i["start_sec"], 2), round(i["end_sec"], 2)) for i in intervals]
print("Trùng khớp hoàn toàn:", ours == theirs)
if ours != theirs:
    table([{"từng bước": a, "pipeline": b} for a, b in zip(ours, theirs)])
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                "language_info": {"name": "python"}})
target = HERE / "notebooks" / "explain_slide_detection.ipynb"
nbf.write(nb, target)
print("wrote", target)
