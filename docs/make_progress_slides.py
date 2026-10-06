"""Build docs/progress-report.html: progress report for the supervisor, same school template and
styles as the defense deck (make_thesis_slides.py), one step per slide with real outputs.

    python docs/make_progress_slides.py <template media dir> <runs dir>
<template media dir>: ppt/media of the unzipped "Template Powerpoint v1.pptx";
<runs dir>: kaggle/lecture_rag/runs (exports of 20261002-full-10-ads and 20261005-run-pipeline).
"""
import base64, io, re, sys
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
MEDIA, RUNS = Path(sys.argv[1]), Path(sys.argv[2])
source = (HERE / "make_thesis_slides.py").read_text(encoding="utf-8")
head, tail = source.split("\nS = []\n", 1)[0], source.split("\nCSS = f'''", 1)[1]
ns = {"__name__": "progress"}
sys.argv = [sys.argv[0], str(MEDIA), str(HERE / "progress-report.html")]
exec(head, ns)
ns["SECTIONS"] = ["Tổng quan & mục tiêu", "Dữ liệu", "Xử lý video", "Truy xuất & hỏi đáp",
                  "Kết quả hiện tại", "Kế hoạch tiếp theo"]
content, section, card, table = ns["content"], ns["section"], ns["card"], ns["table"]
ns["PIPELINE_SVG"] = ns["PIPELINE_SVG"].replace("E5 + BM25", "bge-m3 + BM25").replace("quảng cáo · khung trống", "bộ gác: có chữ không?")
ns["RAG_SVG"] = ns["RAG_SVG"].replace("E5 · top 20", "bge-m3 · top 20")

V008 = RUNS / "20261002-full-10-ads/outputs/fast_pipeline_export/L25_V008"
MLS = RUNS / "20261005-run-pipeline/export/MLS-15DA1"


def photo(path, width=900, quality=78, top_share=1.0):
    image = Image.open(path).convert("RGB")
    if top_share < 1:   # keep the first rows of a tall contact sheet
        image = image.crop((0, 0, image.width, int(image.height * top_share)))
    image.thumbnail((width, width))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def report_png(html_path, index=0):
    """index-th PNG chart embedded in a run report (the ad-score plot of the walkthrough)."""
    found = re.findall(r'src="(data:image/png;base64,[^"]+)"', Path(html_path).read_text(encoding="utf-8"))
    return found[index]


def fig(src, caption, cls=""):
    return f'<figure class="fig {cls}"><img src="{src}" alt=""><figcaption>{caption}</figcaption></figure>'


S = []
S.append(f'''<section class="slide title-slide">
  <div class="logo-white"></div>
  <div class="cover-top">TRƯỜNG ĐẠI HỌC CÔNG NGHIỆP THÀNH PHỐ HỒ CHÍ MINH<br>KHOA CÔNG NGHỆ THÔNG TIN</div>
  <div class="cover-major">GRADUATION THESIS<br>MAJOR: COMPUTER SCIENCE</div>
  <div class="title-box">
    <p class="kicker">BÁO CÁO TIẾN ĐỘ KHÓA LUẬN</p>
    <h1>MULTIMODAL LECTURE RAG</h1>
    <p class="subtitle">Hệ thống hỏi đáp đa phương thức trên video bài giảng</p>
    <div class="people">
      <div><span>Sinh viên thực hiện:</span><b>NGUYỄN THÀNH TRUNG</b> 22696701<br><b>THÁI VĂN TRUNG</b> 22696241</div>
      <div><span>Giảng viên hướng dẫn:</span><b>TS. ĐẶNG THỊ PHÚC</b></div>
    </div>
  </div>
  <div class="cover-date">Hồ Chí Minh, 10/2026</div>
</section>''')
S.append(content(0, "Nội dung báo cáo", '<ol class="agenda">' + "".join(
    f'<li><span>{n:02d}</span>{name}</li>' for n, name in enumerate(ns["SECTIONS"], 1)) + "</ol>"))

# 1 overview
S.append(section(1))
S.append(content(1, "Yêu cầu đề tài và tiến độ chung", '''<div class="cols2">
  <div class="panel"><h3>Yêu cầu (framework cơ bản)</h3><ol>
    <li><b>Xử lý video:</b> phát hiện slide + keyframe; OCR tiêu đề, văn bản, công thức, bảng, hình; STT lời giảng theo slide; đồng bộ theo thời gian.</li>
    <li><b>Multimodal Lecture RAG:</b> câu hỏi SV → truy xuất slide + transcript + ngữ cảnh → LLM → trả lời.</li></ol></div>
  <div><h3>Tiến độ</h3>''' + table(["Phần", "Trạng thái"], [
      ["Phát hiện slide", "<span class='ok'>✔ xong</span> · F1 0,970"],
      ["Lọc nội dung (bộ gác OCR)", "<span class='ok'>✔ chốt</span> · giữ 100% slide có chữ"],
      ["OCR đa thành phần", "<span class='ok'>✔ chạy tốt</span>"],
      ["Speech-to-Text", "<span class='ok'>✔ chạy tốt</span>"],
      ["Đồng bộ slide ↔ transcript", "<span class='ok'>✔ xong</span>"],
      ["Chỉ mục + truy xuất", "<span class='ok'>✔ chốt bge-m3</span> · Hit@3 22/22"],
      ["LLM trả lời + web app", "<span class='wip'>◐ chốt Qwen, chưa nạp app</span>"]]) + '''</div>
</div>'''))
S.append(card(1, "Luồng xử lý đã xây dựng", ns["PIPELINE_SVG"] + '<p class="note">Hình 1: Mọi khối đã chạy được từ đầu đến cuối trên Kaggle (GPU T4 x2). Video 30 phút xử lý trong ~21 phút.</p>'))

# 2 data
S.append(section(2))
S.append(content(2, "Dữ liệu đã thu thập", table(
    ["Bộ dữ liệu", "Video", "Giờ", "Ngôn ngữ", "Vai trò"],
    [["L25 · kênh ôn thi THPT", "10", "4,2", "vi", "Chỉnh tham số, test tách slide & quảng cáo"],
     ["vn-lecture-test12 · 11 kênh mới", "12", "7,9", "vi", "Test chính trên kênh chưa thấy"],
     ["LPM", "244", "117", "en", "Tách slide (có sẵn mốc slide)"],
     ["M3AV · Oxford Mathematics", "51", "27", "en", "Chấm OCR công thức (LaTeX)"],
     ["SlideSpeech", "141", "90", "en", "Lọc nội dung, test ASR"]],
    "Bảng 1: Tổng 458 video, 246 giờ; lưu trên Kaggle") + '''<p class="note">Nhãn gán tay đã có: 549 ranh giới slide và 2.068 giây quảng cáo trên L25; 26 câu hỏi có đáp án (V008, MLS-15DA1).</p>'''))

# 3 video processing
S.append(section(3))
def change_chart(npz, intervals_path, until=340, cap=0.3):
    """SVG of the real picture change per second (signal_dump) with the cuts stage 01 made."""
    import json as _json
    import numpy as np
    data = np.load(npz)
    times, change = data["times"], np.minimum(data["change"], cap)
    keep = times <= until
    W, H, left, top = 1000, 175, 40, 10
    x = lambda t: left + (W - left - 10) * t / until
    y = lambda v: top + (H - top - 34) * (1 - v / cap)
    points = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(times[keep], change[keep]))
    cuts = [i for i in _json.load(open(intervals_path, encoding="utf-8")) if 0 < i["start_sec"] <= until]
    marks = "".join(
        f'<line x1="{x(i["start_sec"]):.1f}" y1="{top}" x2="{x(i["start_sec"]):.1f}" y2="{H - 34}" stroke="#153898" stroke-width="2" stroke-dasharray="4 3"/>'
        for i in cuts)
    ads = "".join(f'<rect x="{x(i["start_sec"]):.1f}" y="{top}" width="{x(i["end_sec"]) - x(i["start_sec"]):.1f}" height="{H - 34 - top}" fill="#f6b21b" opacity=".18"/>'
                  for i in cuts if i["content_kind"] == "ad")
    ticks = "".join(f'<text x="{x(t):.1f}" y="{H - 14}" text-anchor="middle" font-size="12" fill="#5a6a8a">{t // 60}:{t % 60:02d}</text>'
                    for t in range(0, until + 1, 60))
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Độ khác nhau giữa hai giây liên tiếp">{ads}'
            f'<line x1="{left}" y1="{y(0.055):.1f}" x2="{W - 10}" y2="{y(0.055):.1f}" stroke="#c4553f" stroke-width="1.5" stroke-dasharray="6 4"/>'
            f'<text x="{W - 12}" y="{y(0.055) - 5:.1f}" text-anchor="end" font-size="12" fill="#c4553f">ngưỡng "khác" 0,055</text>'
            f'<polyline points="{points}" fill="none" stroke="#5a6a8a" stroke-width="1.3"/>{marks}{ticks}'
            f'<line x1="{left}" y1="{H - 34}" x2="{W - 10}" y2="{H - 34}" stroke="#d8deee"/></svg>')


FIGS = HERE / "figs"


def code(text):
    import html as _html
    return f'<pre class="code">{_html.escape(text.strip())}</pre>'


def embed(name):
    return "data:image/jpeg;base64," + base64.b64encode((FIGS / name).read_bytes()).decode()


def distance_chart(path, start=85, until=165, cap=0.3):
    """SVG of the real distance of every second to the current slide's reference (step1_curve.json)."""
    import json as _json
    data = _json.load(open(path, encoding="utf-8"))
    W, H, left, top = 1000, 140, 40, 8
    x = lambda t: left + (W - left - 10) * (t - start) / (until - start)
    y = lambda v: top + (H - top - 32) * (1 - min(v, cap) / cap)
    pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in data["curve"] if start <= t <= until)
    cuts = "".join(f'<line x1="{x(b):.1f}" y1="{top}" x2="{x(b):.1f}" y2="{H - 32}" stroke="#153898" stroke-width="2" stroke-dasharray="4 3"/>'
                   for b in data["boundaries"] if start < b < until)
    ticks = "".join(f'<text x="{x(t):.1f}" y="{H - 12}" text-anchor="middle" font-size="12" fill="#5a6a8a">{t // 60}:{t % 60:02d}</text>'
                    for t in range(start - start % 10 + 10, until + 1, 10))
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Độ khác với ảnh tham chiếu">'
            f'<line x1="{left}" y1="{y(0.055):.1f}" x2="{W - 10}" y2="{y(0.055):.1f}" stroke="#c4553f" stroke-width="1.5" stroke-dasharray="6 4"/>'
            f'<text x="{W - 12}" y="{y(0.055) - 5:.1f}" text-anchor="end" font-size="12" fill="#c4553f">ngưỡng khác 0,055</text>'
            f'<polyline points="{pts}" fill="none" stroke="#5a6a8a" stroke-width="1.6"/>{cuts}{ticks}'
            f'<line x1="{left}" y1="{H - 32}" x2="{W - 10}" y2="{H - 32}" stroke="#d8deee"/></svg>')


S.append(content(3, "Bước 1 · Phát hiện slide: tổng quan", '''<div class="steps4">
  <div><i class="n">1</i><h4>Lấy ảnh</h4><p>Mỗi giây lấy 1 ảnh từ video (20 phút → 1.200 ảnh).</p></div>
  <div><i class="n">2</i><h4>Tóm tắt ảnh</h4><p>Thu mỗi ảnh thành một <b>“chữ ký”</b> nhỏ: ảnh xám + nét chữ.</p></div>
  <div><i class="n">3</i><h4>So với slide hiện tại</h4><p>Đo ảnh mới <b>khác ảnh tham chiếu</b> của slide đang chiếu bao nhiêu.</p></div>
  <div><i class="n">4</i><h4>Cắt & chọn ảnh</h4><p>Khác ≥ 0,055 <b>và đứng yên ≥ 2 giây</b> → slide mới; giữ ảnh nét nhất.</p></div>
</div>
<p class="note" style="margin-top:18px">Code: <code>kaggle/lecture_rag/notebook_code/detection.py</code> · hàm chính <code>run_slide_detection()</code>.
Các slide sau giải thích từng bước bằng code thật và hình chạy trên video L25_V008.</p>'''))

S.append(content(3, "1.1 · Lấy một ảnh mỗi giây", '<div class="cols2 code-cols">' + code("""
def sampled_frames(video_path, source, sample_hz=1.0):
    next_time = 0.0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            t = float(frame.pts * frame.time_base)
            if t + 1e-6 < next_time:
                continue                 # bỏ các khung giữa hai mốc giây
            yield {"time_sec": t,
                   "signature": frame_signature(frame.to_image())}
            next_time = t + 1.0 / sample_hz
""") + '''<div><ul class="small">
  <li>Giải mã video <b>một lượt</b> bằng PyAV, không tua đi tua lại.</li>
  <li>Cứ 1 giây lấy 1 khung (<code>sample_hz = 1</code>); giữ thêm khung 8 Hz để sau này chỉnh ranh giới chính xác ~0,1 s.</li>
  <li>Mỗi khung chỉ giữ lại <b>chữ ký</b> (bước 1.2), nên tốn rất ít bộ nhớ.</li></ul></div></div>''' +
    fig(embed("step1_strip.jpg"), "Hình 2: 6 giây liên tiếp của V008 (1:35–1:40). Tới 1:38 hình chuyển từ giảng viên + slide nhỏ sang slide toàn màn hình.", "strip")))

S.append(content(3, "1.2 · “Chữ ký” của mỗi ảnh", '<div class="cols2 code-cols">' + code("""
def frame_signature_array(pixels):
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (240, 135))            # thu nhỏ
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)    # lớp 1: màu, bố cục
    edges = cv2.GaussianBlur(
        cv2.Canny(gray, 50, 150), (3, 3), 0)       # lớp 2: nét chữ
    return np.stack([blurred, edges]) / 255.0
""") + '''<div><ul class="small">
  <li>Ảnh 1280×720 được thu về <b>240×135</b>, chuyển sang xám: đủ thấy bố cục, bỏ qua nhiễu nhỏ.</li>
  <li><b>Lớp làm mờ:</b> bắt thay đổi màu nền, hình lớn.</li>
  <li><b>Lớp cạnh (Canny):</b> bắt thay đổi <b>chữ</b>. Hai slide nền trắng chỉ khác vài dòng chữ vẫn được nhận ra là khác.</li></ul></div></div>''' +
    fig(embed("step1_signature.jpg"), "Hình 3: Slide “Định nghĩa xác suất” (1:55) và hai lớp chữ ký mà code thực sự so sánh.", "strip")))

S.append(content(3, "1.3 · Đo hai ảnh khác nhau bao nhiêu", '<div class="cols2 code-cols">' + code("""
def signature_distance(left, right):
    diff = np.abs(left[0] - right[0]).reshape(-1)
    keep = int(len(diff) * 0.97)          # bỏ 3% điểm khác nhất
    color_change = np.partition(diff, keep - 1)[:keep].mean()
    edge_change = np.abs(left[1] - right[1]).mean()
    edge_density = max(0.08, left[1].mean() + right[1].mean())
    return min(1.0, max(color_change,
                        edge_change / edge_density))
""") + '''<div><ul class="small">
  <li><b>Đổi màu:</b> trung bình độ lệch điểm ảnh, sau khi <b>bỏ 3% điểm lệch nhất</b> (con trỏ chuột, logo nhấp nháy).</li>
  <li><b>Đổi chữ:</b> độ lệch lớp cạnh, chia cho mật độ chữ. Slide ít chữ chỉ đổi một dòng vẫn ra số lớn.</li>
  <li>Lấy <b>giá trị lớn hơn</b> của hai cái: chỉ cần màu <i>hoặc</i> chữ đổi là đủ.</li>
  <li>Kết quả từ 0 (giống hệt) đến 1 (khác hoàn toàn).</li></ul>
  <div class="kpis one"><div><b>0,055</b><span>ngưỡng “khác” · 0,035 ngưỡng “đứng yên”</span></div></div></div></div>'''))

S.append(content(3, "1.4 · Khi nào cắt slide mới", '<div class="cols2 code-cols tight">' + code("""
for sample in samples:                         # mỗi giây
    d = signature_distance(reference, sample)  # so với slide hiện tại
    if d >= 0.055 and t - last_cut >= 2.0:     # khác đủ nhiều
        if candidate and distance(candidate, sample) <= 0.035:
            candidate.count += 1               # ảnh mới đang đứng yên
        else:
            candidate = new_candidate(sample)
        if candidate.count >= 2:               # yên >= 2 giây
            boundaries.append(candidate.time)  # -> cắt slide mới
            reference = candidate.signature    # đổi ảnh tham chiếu
    else:
        candidate = None                       # chỉ là rung lắc
""") + '''<div><ul class="small">
  <li>Mỗi giây được so với <b>ảnh tham chiếu</b> của slide đang chiếu, không phải với giây ngay trước.</li>
  <li>Khác ≥ 0,055 thì ảnh mới thành <b>ứng viên</b>.</li>
  <li>Ứng viên phải <b>đứng yên</b> (lệch ≤ 0,035) <b>2 giây liên tiếp</b> mới được cắt. Hiệu ứng chuyển cảnh, giảng viên cử động bị loại.</li>
  <li>Hai lần cắt cách nhau ít nhất 2 giây.</li></ul></div></div>
<div class="chart">''' + distance_chart(FIGS / "step1_curve.json") + '''
<p class="note">Hình 4: Độ khác so với ảnh tham chiếu, V008 1:25–2:45 (chạy bằng chính code trên). Nét đứt xanh = chỗ cắt (trạng thái thô, bước 1b sẽ gộp).</p></div>'''))

S.append(content(3, "1.5 · Chỉnh ranh giới & chọn keyframe", '<div class="cols2 code-cols">' + code("""
# Chọn ảnh nét nhất trong đoạn: phương sai Laplacian
sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

# Vùng luôn chuyển động (đổi > 30% số giây) bị bỏ qua
busy = changed_seconds / total_seconds > 0.30
mask = busy if 0 < busy.mean() < 0.85 else None
""") + '''<div><ul class="small">
  <li><b>Chỉnh ranh giới:</b> dùng khung 8 Hz đã giữ ở bước 1.1 để đặt mốc cắt chính xác ~0,1 giây.</li>
  <li><b>Chọn keyframe:</b> trong mỗi đoạn, lấy khung <b>nét nhất</b> (phương sai Laplacian lớn nhất), tránh khung đang chuyển cảnh bị mờ.</li>
  <li><b>Mặt nạ chuyển động:</b> điểm ảnh đổi trong hơn 30% số giây (nền động, khung giảng viên quay liên tục) bị bỏ qua khi so. Trên V008 <b>không có</b> vùng nào như vậy nên mặt nạ không bật.</li>
  <li><b>Đầu ra:</b> <code>intervals.json</code>: thời gian bắt đầu/kết thúc, keyframe, các trạng thái đã gộp.</li></ul></div></div>'''))
S.append(content(3, "Bước 1 · Phát hiện slide: kết quả", '''<div class="split">
  <div><h3>Cách đọc hình bên</h3><ul class="small">
    <li>Mỗi ô = <b>một đoạn</b> hệ thống tìm được, ảnh là keyframe của đoạn.</li>
    <li>Dòng dưới ô: <b>số thứ tự | thời gian</b>, ví dụ <code>005 | 01:37–02:10</code>.</li>
    <li>Ô có <b>slide chữ nền vàng</b> = nội dung bài giảng → đưa sang OCR.</li>
    <li>Ô <b>người thật, cảnh trường, món ăn</b> = quảng cáo → chỉ đánh dấu.</li>
    <li>Ô <b>giảng viên + slide nhỏ</b> = vẫn là bài giảng.</li></ul>
    <div class="kpis one"><div><b>67 → 16</b><span>trạng thái hình → 12 slide bài giảng + 4 đoạn quảng cáo (V008, 20 phút)</span></div></div></div>
  ''' + fig(photo(V008 / "01_slides/contact_sheet.jpg", 1100), "Hình 5: Keyframe của 16 đoạn trong video L25_V008") + "</div>"))
S.append(content(3, "1.6 · Vì sao chọn ngưỡng 0,055 và 2 s?", '''<div class="cols2">
  <div class="panel"><h3>Hai con số quyết định khi nào cắt slide</h3><ul class="small">
    <li><b>Ngưỡng khác</b> (<code>change_threshold</code>): hai ảnh khác nhau hơn mức này mới coi là <b>có thay đổi</b>.
        Thấp quá → giảng viên cử động cũng bị cắt (báo nhầm); cao quá → slide đổi ít bị bỏ qua (sót).</li>
    <li><b>Đứng yên</b> (<code>stable_samples</code>): sau khi đổi, hình phải đứng yên ≥ N giây mới chốt slide mới.
        Lâu quá → slide chỉ hiện ngắn bị bỏ; ngắn quá → cảnh chuyển động bị tách vụn.</li></ul></div>
  <div class="panel"><h3>Cách chọn: quét ngưỡng (threshold sweep)</h3><ul class="small">
    <li>Thử đủ <b>5 × 3 = 15 cấu hình</b>: khác 0,035 – 0,09 × đứng yên 2, 3, 4 s.</li>
    <li>Mỗi cấu hình chạy lại bước 1 trên <b>18 video</b>, so mốc cắt với <b>đáp án do người đánh dấu</b>
        (lệch ≤ 1 s là đúng) → tính <b>P</b> (ít báo nhầm), <b>R</b> (ít sót), <b>F1</b>.</li>
    <li><b>Chọn</b> trên 4 video chỉnh L25, <b>báo cáo</b> trên 6 video test L25 và 8 video LPM tiếng Anh
        (không dùng để chọn) → biết ngưỡng có hợp với kênh khác không.</li>
    <li>Chạy trên CPU Kaggle, chia 6 notebook / 3 tài khoản (~1 giờ).</li></ul></div></div>'''))
S.append(content(3, "1.6 · Kết quả quét ngưỡng", fig(embed("sweep_f1.jpg"),
    "Mỗi đường là một mức “đứng yên”, trục ngang là “ngưỡng khác”. Vòng tròn: cấu hình đang dùng.", "wide")))
S.append(content(3, "1.6 · Bảng quét ngưỡng (cho luận văn)", table(
    ["Ngưỡng khác", "Đứng yên", "L25 chỉnh · F1", "L25 test · P / R / F1", "LPM ±1 s · F1", "LPM ±3 s · F1"],
    [["0,035", "2 s", "0,853", "0,960 / 0,951 / 0,956", "0,904", "0,932"],
     ["0,045", "2 s", "0,908", "0,960 / 0,931 / 0,945", "0,908", "0,934"],
     ["<b>0,055</b>", "<b>2 s</b>", "<b>0,973</b>", "<b>0,990 / 0,951 / 0,970</b>", "<b>0,909</b>", "<b>0,935</b>"],
     ["0,07", "2 s", "0,956", "0,970 / 0,951 / 0,960", "0,909", "0,935"],
     ["0,09", "2 s", "0,939", "0,958 / 0,902 / 0,929", "0,909", "0,935"],
     ["0,055", "3 s", "0,784", "0,833 / 0,833 / 0,833", "0,861", "0,890"],
     ["0,055", "4 s", "0,682", "0,768 / 0,745 / 0,756", "0,820", "0,838"]],
    "Bảng: F1 tách slide theo ngưỡng (đủ 15 cấu hình trong kaggle/lecture_rag/runs/sweep-*/sweep_table.json). "
    "Dòng đậm = cấu hình đang dùng: cao nhất trên L25, bằng mức cao nhất trên LPM. LPM: đáp án là mốc tác giả căn tay nên chấm thêm ±3 s.") + '''
  <p class="note" style="margin-top:14px"><b>Đọc hình:</b> đường xanh (2 s) luôn cao nhất → đứng yên 2 s là đúng;
  lên 3–4 s F1 tụt 0,10–0,20 vì sót slide hiện ngắn. Trên đường xanh, 0,055 là đỉnh của L25; LPM gần như phẳng từ 0,055.</p>''', "dense"))
S.append(content(3, "Bước 1b · Gộp slide hiện dần", '''<div class="split">
  <div><ul class="small">
    <li>Giảng viên bấm cho <b>từng dòng hiện ra</b> → một slide có nhiều trạng thái.</li>
    <li>Gộp khi nội dung cũ <b>còn đúng chỗ</b> (≥ 85%), hình đổi ít (≤ 25%) và chỉ <b>thêm</b> nội dung.</li>
    <li>Giữ trạng thái cuối (đầy đủ nhất) cho OCR.</li></ul>
    <div class="kpis one"><div><b>0,970</b><span>F1 tách slide · 6 video test L25 (P 0,990 · R 0,951)</span></div></div></div>
  ''' + fig(photo(V008 / "01_slides/reveal_sheet.jpg", 1100, top_share=0.42), "Hình 6: Mỗi hàng là các trạng thái được gộp thành một slide") + "</div>"))
S.append(content(3, "1c.1 · Bộ gác OCR: khung này có chữ để đọc không?", '''<div class="cols2">
  <div class="panel"><h3>Vì sao cần</h3><ul class="small">
    <li>Sau bước 1, mỗi đoạn có 1 keyframe. Nhưng <b>không phải keyframe nào cũng là slide</b>:
        giảng viên nói vào camera, logo kênh, quảng cáo.</li>
    <li>OCR (Unlimited-OCR + Qwen3-VL) là bước <b>chậm nhất</b>, ~85% thời gian xử lý.
        Đọc khung không có chữ vừa <b>tốn thời gian</b> vừa sinh <b>chữ rác</b> vào chỉ mục.</li>
    <li>Bộ gác đứng <b>trước</b> OCR: chấm mỗi keyframe một điểm 0 → 1 = “khả năng có nội dung đọc được”.</li></ul></div>
  <div class="panel"><h3>Quy tắc</h3><ul class="small">
    <li>Điểm ≥ <b>0,15</b> → đưa vào OCR.</li>
    <li>Điểm &lt; 0,15 → <b>bỏ OCR</b>. Đoạn vẫn được giữ, <b>lời giảng vẫn vào tìm kiếm</b>
        (không mất nội dung thầy nói).</li>
    <li>Ngưỡng đặt <b>thấp</b> có chủ ý: thà đọc thừa một khung còn hơn bỏ sót một slide.</li></ul></div></div>'''))
S.append(content(3, "1c.1 · Ví dụ: 4 loại keyframe và điểm bộ gác", fig(embed("gate_examples.jpg"),
    "Kết quả thật trên V008. Thanh = điểm bộ gác (0 → 1), vạch đứng = ngưỡng 0,15.", "wide") + '''
  <p class="note" style="margin-top:10px">Slide chữ gần 1; giảng viên kèm slide nhỏ vẫn 0,79; quảng cáo và logo gần 0 → không tốn thời gian OCR.</p>'''))
S.append(content(3, "1c.2 · Điểm được tính thế nào (SigLIP, không cần huấn luyện)", '<div class="cols2 code-cols">' + code("""
CONTENT_PROMPTS = [            # 4 câu "có nội dung"
  "a slide of a lecture presentation with text",
  "a lecture slide with a diagram, chart, table or formula",
  "a page of a document or an exam with questions",
  "a whiteboard or blackboard with handwriting"]
NO_CONTENT_PROMPTS = [         # 4 câu "không có nội dung"
  "a person talking to the camera",
  "a teacher speaking in a studio",
  "a logo or a title animation",
  "a blank screen"]

def score(images):
    sims = embed(images) @ embed(PROMPTS).T   # giống câu nào?
    probs = softmax(sims * scale + bias)      # 8 xác suất, tổng = 1
    return probs[:, :4].sum(1)                # phần của 4 câu đầu
""") + '''<div><ul class="small">
    <li><b>SigLIP</b> (Google) là model đã học trên hàng trăm triệu cặp <b>ảnh – câu mô tả</b>:
        nó biến ảnh và câu thành vector, ảnh nào khớp câu nào thì hai vector gần nhau.</li>
    <li>Ta viết sẵn <b>8 câu mô tả</b>: 4 câu “có nội dung”, 4 câu “không có nội dung”.</li>
    <li>Với mỗi keyframe, SigLIP cho biết ảnh <b>giống câu nào nhất</b> → chia thành 8 xác suất cộng lại bằng 1.</li>
    <li><b>Điểm = tổng xác suất của 4 câu “có nội dung”</b>.
        Ví dụ logo khớp “a logo …” gần như 100% → điểm ≈ 0.</li>
    <li><b>Zero-shot</b>: không học từ video nào, nên không bị “học vẹt” theo kênh L25 như bộ lọc quảng cáo cũ.</li></ul></div></div>'''))
S.append(content(3, "1c.3 · Nhìn thêm 5 vùng cắt để không bỏ sót slide nhỏ", fig(embed("gate_crops.jpg"),
    "Kết quả thật (L25_V002, 0:35): cả khung chỉ 0,13 nhưng góc dưới phải 0,88 → vẫn đọc OCR.", "wide") + '''
  <p class="note" style="margin-top:8px"><b>Quy tắc <code>frame+crops</code>:</b> chỉ khi <b>cả khung</b> dưới 0,15 mới chấm thêm
  4 góc + phần giữa; chỉ bỏ OCR khi <b>tất cả 6 điểm</b> đều dưới 0,15. </p>'''))
S.append(content(3, "1c.3 · Ví dụ bị bỏ OCR: quảng cáo món ăn", fig(embed("gate_crops_skip.jpg"),
    "Kết quả thật (L25_V008, 7:44–9:07): cả khung 0,01, 5 vùng cắt cao nhất 0,02 → cả 6 điểm dưới 0,15 → bỏ OCR.", "wide") + '''
  <p class="note" style="margin-top:8px">Không vùng nào giống slide, bảng hay chữ viết → đoạn 83 giây này <b>không tốn thời gian OCR</b>;
  đoạn vẫn được giữ trong dòng thời gian.</p>'''))
S.append(content(3, "1c.4 · Bộ gác OCR: kết quả đo", '''<div class="cols2">
  <div><div class="kpis one"><div><b>100%</b><span>slide có chữ được giữ lại · 481 khung có nhãn (L25 + M3AV + LPM)</span></div></div>
    <div class="kpis one" style="margin-top:12px"><div><b>34 + 16</b><span>đoạn quảng cáo + khung trống tự bị bỏ OCR (10 video L25)</span></div></div></div>
  <div class="panel"><h3>Ý nghĩa</h3><ul class="small">
    <li><b>Không bỏ sót slide nào</b> trên 3 bộ dữ liệu khác nhau → bộ gác an toàn.</li>
    <li>Quảng cáo và logo bị bỏ OCR <b>tự động</b>, dù bộ gác không được dạy “quảng cáo là gì”.</li>
    <li>Đã bỏ bộ lọc quảng cáo học từ L25: trên kênh khác nó coi 47–100% video là quảng cáo.</li></ul></div></div>'''))
S.append(content(3, "Bước 1c · Kết quả demo trên video V008", fig(embed("gate_grid.jpg"),
    "Khung xanh = đọc OCR · khung đỏ = bỏ OCR (điểm &lt; 0,15): bỏ đúng 3 quảng cáo (#3, #9, #12) + 2 logo; giảng viên kèm slide nhỏ vẫn được đọc.", "wide")))
seg = V008 / "04_bundle" / "keyframes"
S.append(content(3, "Bước 2 · OCR đa thành phần", '''<div class="split">
  ''' + fig(photo(V008 / "04_bundle/keyframes/seg_00004.jpg", 900), "Hình 5: Keyframe slide “Định nghĩa xác suất” (V008, 1:38)") + '''
  <div><h3>Kết quả OCR (Unlimited-OCR)</h3><pre class="ocr">ĐỊNH NGHĨA XÁC SUẤT
Xác suất của biến cố A được tính bởi công thức
\\[ P(A) = \\frac{n(A)}{n(\\Omega)} \\]
Trong đó n(A) là số kết quả thuận lợi của biến cố A;
n(Ω) là số kết quả có thể xảy ra của phép thử.</pre>
  <p class="note">Đọc đúng tiếng Việt có dấu và công thức dạng LaTeX. Vùng hình/biểu đồ được Qwen3-VL đọc riêng.</p></div></div>'''))
S.append(content(3, "Bước 2b · Công thức và hình vẽ (bài giảng tiếng Anh)", '''<div class="cols2">
  <div>''' + fig(photo(MLS / "04_bundle/keyframes/seg_00007.jpg", 800), "Hình 6: Slide viết tay MLS-15DA1 (9:27)") + '''<pre class="ocr small">Def: A vector v ∈ V is an eigenvector of T if Tv = λv
for some λ ∈ R.
Proof. λ is an eigenvalue of T ⇔ Ker(T − λ) ≠ ∅</pre></div>
  <div>''' + fig(photo(MLS / "04_bundle/regions/027c538c510c8a9ba390_000.jpg", 700), "Hình 7: Vùng hình được Qwen3-VL mô tả") + '''<pre class="ocr small">“A hand-drawn diagram showing a cube transforming
into a skewed rectangular prism …”</pre>
  <p class="note">Mô tả hình theo ngôn ngữ của bài (tiếng Anh). Chữ viết tay đôi khi đọc sai ký hiệu.</p></div></div>'''))
S.append(content(3, "Bước 3–4 · Speech-to-Text và đồng bộ", '''<div class="cols2">
  <div class="panel"><h3>Speech-to-Text</h3><ul class="small">
    <li><b>faster-whisper-medium</b>, lọc khoảng lặng, có <b>thời gian từng từ</b>.</li>
    <li>Chạy song song với tách slide và OCR (~3 phút / 30 phút video).</li>
    <li>V008: 3.599 từ trong 27 cửa sổ.</li></ul>
    <h3 style="margin-top:14px">Đồng bộ</h3><ul class="small">
    <li>Từ nói lúc slide nào hiện → gán vào slide đó.</li>
    <li>Làm sạch OCR: bỏ logo, tên GV, dòng lặp.</li>
    <li><b>Sửa từ nghe nhầm</b> theo chữ trên slide: 13 chỗ trên V008.</li></ul></div>
  <div><h3>Một slide sau đồng bộ (V008 · 1:38–2:10)</h3>
    <div class="pair"><b>Chữ trên slide</b><p>ĐỊNH NGHĨA XÁC SUẤT · P(A) = n(A)/n(Ω) …</p>
    <b>Lời giảng gán vào slide</b><p>“P A sẽ bằng với lại N A chia cho N Omega. Trong đó N A là số kết quả thuận lợi của biến cố A …”</p>
    <b>Sửa ASR</b><p><s>xác xuất</s> → <b>xác suất</b> · <s>sát xuất</s> → <b>xác suất</b></p></div></div></div>'''))
S.append(content(3, "Bước 3–4 · Kết quả demo trên video V008", fig(embed("sync_demo.jpg"),
    "Mỗi từ Whisper nghe được (vạch) có thời gian → gán vào slide đang hiện; lời đã sửa theo chữ trên slide.", "wide")))

# 4 retrieval
S.append(section(4))
S.append(card(4, "Bước 5 · Chỉ mục và truy xuất đa phương thức", ns["RAG_SVG"] + '''<ul class="small">
  <li>Mỗi slide thành các mẩu: chữ OCR, lời giảng, vùng hình → vector <b>bge-m3</b> + chỉ mục từ khóa <b>BM25</b>.</li>
  <li>Gộp 2 kết quả bằng RRF; câu "slide này" ưu tiên slide đang xem; thêm slide trước/sau để liền mạch.</li>
  <li>LLM <b>Qwen3-4B</b> (hoặc Qwen2.5-7B 4-bit) nhận bằng chứng + ≤ 3 ảnh slide, trả lời kèm mốc thời gian.</li></ul>'''))
S.append(content(4, "Bước 5 · Kết quả demo: câu hỏi → slide tìm được (V008)", fig(embed("retrieval_demo.jpg"),
    "Hình: 3 slide điểm cao nhất cho mỗi câu hỏi (chỉ mục E5, chạy 06/10): hạng 1 đều đúng slide chứa đáp án.", "wide")))
S.append(content(4, "Ví dụ hỏi đáp (Qwen3-4B, bằng chứng từ bge-m3/E5)", table(
    ["Câu hỏi", "Trả lời của hệ thống", ""],
    [["Xác suất của một biến cố được tính như thế nào?", "P(A) = n(A)/n(Ω), n(A): số kết quả thuận lợi …", "✔"],
     ["Không gian mẫu khi lấy 2 quả cầu từ hộp 15 quả?", "C(15,2) = 105 phần tử", "✔"],
     ["Tổng hai số là số chẵn khi nào?", "Cả hai cùng lẻ hoặc cả hai cùng chẵn", "✔ (qwen2:7b từng sai)"],
     ["What is the leading coefficient of the characteristic polynomial?", "(−1)ⁿ, nêu ở phần chứng minh 34:31", "✔"],
     ["Cách cài đặt PyTorch? (ngoài bài)", "Bài giảng không đề cập", "✔"]],
    "Bảng 2: 26 câu hỏi có đáp án · Qwen3-4B 24/26 đúng, Qwen2.5-7B 4-bit 25/26, ~15 s/câu trên T4")))

# 5 results
S.append(section(5))
S.append(content(5, "Kết quả đo được (pipeline đã chốt)", table(
    ["Thành phần", "Tập đo", "Kết quả"],
    [["Phát hiện slide", "6 video test L25", "F1 <b>0,970</b> · LPM 0,909 (ngưỡng tốt nhất sau quét 15 cấu hình)"],
     ["Bộ gác OCR (lọc nội dung)", "481 khung L25 · M3AV · LPM", "giữ slide có chữ <b>100%</b>"],
     ["OCR công thức", "slide toán M3AV", "CER <b>0,36</b> (GOT-OCR 2.0: 0,52)"],
     ["Truy xuất bge-m3", "22 câu hỏi có đáp án", "Hit@1 <b>21/22</b> · Hit@3 <b>22/22</b> (E5: 19/22)"],
     ["Câu trả lời LLM", "26 câu (4 câu ngoài bài)", "Qwen3-4B <b>24/26</b> · Qwen2.5-7B <b>25/26</b> (qwen2:7b: 9/13)"],
     ["Thời gian", "video 30 phút, Kaggle T4 x2", "<b>~21 phút</b> xử lý · ~15 s/câu trả lời"]],
    "Bảng 3: Kết quả của pipeline đã chốt")))
S.append(content(5, "So sánh các phương án (thời gian + kết quả)", table(
    ["Phương án", "Thời gian", "Kết quả", "Quyết định"],
    [["Bộ lọc quảng cáo học từ L25", "~17 s/video", "L25 99,8% · kênh khác coi 47–100% video là QC", "<span class='bad'>bỏ</span>"],
     ["Hình + lời lạc chủ đề (E5)", "vài giây", "video tệ nhất chỉ giữ 84–86% bài giảng", "<span class='bad'>bỏ</span>"],
     ["LLM chia chương + gán nhãn", "32 s / 30 phút", "chỉ bắt 5–12% quảng cáo; tìm 2 tầng kém hơn", "<span class='bad'>bỏ</span>"],
     ["Bộ slide chuẩn (bỏ trùng trước OCR)", "—", "chỉ bớt 2% thời gian OCR", "<span class='bad'>bỏ</span>"],
     ["Ảnh OCR 768 / 640", "chậm ×3 / nhanh 20%", "hỏng chữ / mất 1/3 nội dung", "<span class='bad'>bỏ</span>"],
     ["bge-m3 thay E5", "+3–5 s lập chỉ mục", "MLS hit@1 8 → 10/11; tách câu ngoài bài", "<span class='ok'>chọn</span>"],
     ["Qwen3-4B / Qwen2.5-7B thay qwen2:7b", "~15 s/câu (T4)", "24–25/26 đúng (trước 9/13)", "<span class='ok'>chọn</span>"]],
    "Bảng 4: Các phương án đã thử ngày 05–06/10/2026"), "dense"))
# 6 plan
S.append(section(6))
S.append(content(6, "Kế hoạch tiếp theo", '''<div class="road">''' + "".join(
    f'<div class="step {cls}"><b>{n}</b><span>{t}</span><i>{s}</i></div>'
    for n, t, s, cls in [(1, "Chạy 12 video kênh mới", "đang chạy trên Kaggle → bản nháp gold set 2", "now"),
                         (2, "Gold set 2: nhóm sửa nháp", "CER (OCR), WER (lời giảng), F1 (tách slide)", "now"),
                         (3, "Nạp pipeline mới vào web app", "bge-m3 + Qwen, mốc giờ mm:ss", "next"),
                         (4, "LLM đọc đoạn bị bộ gác bỏ", "loại lời quảng cáo khỏi tìm kiếm", "next"),
                         (5, "Viết khóa luận", "chương thực nghiệm, kết quả", "next")]) + '''</div>
<div class="key"><span class="k now">đang làm</span><span class="k next">tiếp theo</span></div>
<p class="note" style="text-align:center;margin-top:22px">Xin ý kiến cô: ưu tiên thứ tự các việc và tiêu chí đánh giá cho chương thực nghiệm.</p>'''))
S.append(f'''<section class="slide end-slide">
  <div class="logo-white end-logo"></div>
  <h2 class="end-title">Cảm ơn cô<br>đã lắng nghe</h2>
  <p class="end-sub">Xin ý kiến góp ý</p>
</section>''')

EXTRA = '''
.split{{display:grid;grid-template-columns:0.95fr 1.25fr;gap:26px;align-items:start}}
.fig{{margin:0}} .fig img{{width:100%;max-height:330px;object-fit:contain;background:#fff;border:1px solid #d8deee;border-radius:10px;display:block}}
.cols2 .fig img{{max-height:250px}}
.fig.wide img{{max-height:400px}}
.fig figcaption{{font-size:13px;color:var(--sub);text-align:center;margin-top:6px;font-style:italic}}
.ocr{{background:#f6f8fd;border-left:5px solid var(--gold);border-radius:8px;padding:12px 14px;font-size:15px;white-space:pre-wrap;margin:6px 0;font-family:Consolas,monospace}}
.ocr.small{{font-size:13px;padding:8px 12px}}
.pair p{{margin:2px 0 12px;font-size:16px}} .pair b{{color:var(--blue)}}
.ok{{color:#2f7d4a;font-weight:700}} .wip{{color:#c58a00;font-weight:700}}
.kpis.one{{grid-template-columns:1fr}}
.dense table{{font-size:14px}} .dense td,.dense th{{padding:5px 9px}}
.steps4{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.steps4 div{{background:var(--soft);border-radius:14px;padding:12px 14px;border-top:5px solid var(--blue)}}
.steps4 .n{{font-style:normal;display:inline-grid;place-items:center;width:30px;height:30px;border-radius:50%;background:var(--gold);color:var(--navy);font-weight:800}}
.steps4 h4{{margin:6px 0 4px;font-size:18px;color:var(--navy)}} .steps4 p{{margin:0;font-size:15px;line-height:1.4}}
.chart{{margin-top:14px}} .chart svg{{width:100%;height:auto;background:#fff;border:1px solid #d8deee;border-radius:10px}}
.chart .note{{font-size:13.5px;margin-top:6px}}
.code{{background:#0e1f5c;color:#e8edff;border-radius:10px;padding:12px 14px;font:12px/1.4 Consolas,monospace;margin:0;white-space:pre;overflow:hidden}}
.code-cols{{grid-template-columns:1.15fr 1fr;gap:22px}} .code-cols .small li{{font-size:15px}}
.tight .code{{font-size:11px;line-height:1.3;padding:8px 12px}} .tight .small li{{font-size:14px;margin:3px 0}}
.fig.strip{{margin-top:12px}} .fig.strip img{{max-height:150px}}
'''
exec(compile("CSS = f'''" + tail.replace("\n'''\n", "\n" + EXTRA + "'''\n", 1), "tail", "exec"), {**ns, "S": S})
