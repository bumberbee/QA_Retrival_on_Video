"""Build docs/progress-report.html: progress report for the supervisor, using the school template,
styles and helpers of docs/deck_template.py, one step per slide with real outputs.

    python docs/make_progress_slides.py <template media dir> <runs dir>
<template media dir>: ppt/media of the unzipped "Template Powerpoint v1.pptx";
<runs dir>: retained for CLI compatibility; run images are restored under docs/figs.
"""
import base64, io, re, sys
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent

# Confirmation run dated 03/11. Fill ONLY this dict after its results are audited.
# None is rendered as "đang chờ"; do not substitute cached ablation results.
CONFIRM = {
    "kernel": "lecture-rag-fast-10-confirm-20261102",
    # D:/thesis-data/run-20261102/confirm-eval/confirm-eval-summary.json (CPU kernel
    # lecture-rag-confirm-cpu-eval-20261104, "fixed" = 12 "%" speech chunks removed; same metrics as-is).
    "dev_hit1": 0.627, "dev_hit3": 0.818, "dev_mrr": 0.714, "dev_context_recall": 0.791,
    "legacy_hit1": 0.795, "legacy_hit3": 0.955, "legacy_mrr": 0.871, "legacy_time_overlap": 0.909,
    "strict_hit1": 0.727, "strict_hit3": 0.886, "strict_mrr": 0.795,
    # gpu-ledger.json (2 x T4 x 4,698.93 log seconds); progress.json wall time; 1 failed / 3 retried OCR groups.
    "t4_hours": 2.61, "wall_seconds": 4494, "ocr_failures": 1, "ocr_retries": 3,
}

# Dev: RUN_20261102.md and audit/reproduced-summary.json (rounded to 3 decimals).
# Legacy here uses ORIGINAL any-positive group-time overlap:
# RUN_20261102.md, ABLATION_20261102.md, audit/reproduced-summary-time.json.
# Do not mix it with default-scoring reproduced-summary.json or minimum-overlap.
ABLATION = [
    ("Old reference", (0.591, 0.782, 0.674, 0.764), (0.818, 0.955, 0.871, 0.909)),
    ("New + medium", (0.591, 0.764, 0.673, 0.745), (0.795, 0.864, 0.826, 0.818)),
    ("New + turbo (bỏ)", (0.564, 0.773, 0.662, 0.727), (0.705, 0.864, 0.773, 0.795)),
    ("<b>Principled (báo cáo)</b>", (0.600, 0.782, 0.679, 0.755), (0.818, 0.955, 0.871, 0.909)),
    ("Recommended-final (lạc quan)", (0.636, 0.818, 0.718, 0.809), (0.818, 0.955, 0.875, 0.909)),
]


def metric_line(values):
    return " / ".join(f"{value:.3f}" for value in values)


def confirm_value(key):
    value = CONFIRM[key]
    if value is None:
        return '<span class="wip">đang chờ</span>'
    return f"{value:.3f}" if isinstance(value, float) else str(value)


MEDIA, RUNS = Path(sys.argv[1]), Path(sys.argv[2])
source = (HERE / "deck_template.py").read_text(encoding="utf-8")
head, tail = source.split("\nS = []\n", 1)[0], source.split("\nCSS = f'''", 1)[1]
ns = {"__name__": "progress"}
sys.argv = [sys.argv[0], str(MEDIA), str(HERE / "progress-report.html")]
exec(head, ns)
ns["SECTIONS"] = ["Bài toán & mục tiêu", "Dữ liệu", "Hệ thống xử lý video", "Truy xuất & web app",
                  "Kết quả & đánh giá", "Kế hoạch & xin ý kiến", "Phụ lục"]
content, section, card, table = ns["content"], ns["section"], ns["card"], ns["table"]
ns["PIPELINE_SVG"] = (ns["PIPELINE_SVG"].replace("quảng cáo · khung trống", "bộ gác: có chữ không?")
                      .replace(">OCR đa thành phần<", ">OCR vùng nội dung<")
                      .replace("Unlimited-OCR + Qwen", "Qwen3-VL-4B · crop")
                      )
ns["RAG_SVG"] = (ns["RAG_SVG"].replace("≤ 4 slide neo", "3 slide tốt nhất")
                 .replace("+ slide trước/sau", "chữ slide ×1,05"))

FIGS = HERE / "figs"


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
  <div class="cover-date">Hồ Chí Minh, 11/2026</div>
</section>''')
S.append(content(0, "Nội dung báo cáo", '<ol class="agenda">' + "".join(
    f'<li><span>{n:02d}</span>{name}</li>' for n, name in enumerate(ns["SECTIONS"], 1)) + "</ol>"))
S.append(content(0, "Tóm tắt: đề tài làm gì, đã làm được đến đâu", '''<div class="cols2">
  <div class="panel"><h3>Đề tài</h3><ul class="small">
    <li>Sinh viên xem <b>video bài giảng có slide</b> và hỏi bằng tiếng Việt (hoặc tiếng Anh).</li>
    <li>Đọc <b>chữ trên slide</b> (OCR) + <b>lời giảng</b> (Speech-to-Text), ghép theo thời gian; LLM trả lời <b>kèm mốc thời gian</b> bấm được.</li>
    <li>RAG: tìm bằng chứng trong bài trước, LLM chỉ trả lời từ đó → hạn chế bịa.</li></ul>
    <h3 style="margin-top:12px">Đã làm được</h3><ul class="small">
    <li>Đủ <b>5 bước xử lý video</b> chạy trên Kaggle + <b>web app</b> hỏi đáp chạy được.</li>
    <li>5 bộ dữ liệu (536 video), nhãn tay L25. <b>Dev-120:</b> 110 câu có đáp án, 10 ngoài bài; AI soạn, chưa được người kiểm.</li></ul></div>
  <div class="kpis two">
    <div><b>0,970</b><span>F1 tách slide (6 video test L25)</span></div>
    <div><b>94%</b><span>từ tiếng Việt OCR đọc đúng (Qwen3-VL, 20 slide)</span></div>
    <div><b>0.600</b><span>Hit@1, principled trên 110 câu có đáp án của dev-120</span></div>
    <div><b>0.782</b><span>Hit@3 dev-120, MRR 0.679, context recall 0.755</span></div></div>
</div>''', "dense"))

# 1 overview
S.append(section(1))
S.append(content(1, "Yêu cầu đề tài và tiến độ chung", '''<div class="cols2">
  <div class="panel"><h3>Yêu cầu (framework cơ bản)</h3><ol>
    <li><b>Xử lý video:</b> phát hiện slide + keyframe; OCR tiêu đề, văn bản, công thức, bảng, hình; STT lời giảng theo slide; đồng bộ theo thời gian.</li>
    <li><b>Multimodal Lecture RAG:</b> câu hỏi SV → truy xuất slide + transcript + ngữ cảnh → LLM → trả lời.</li></ol></div>
  <div><h3>Tiến độ</h3>''' + table(["Phần", "Trạng thái"], [
      ["Phát hiện slide", "<span class='ok'>✔ đã đo</span> · F1 0,970 (6 video test L25)"],
      ["Lọc nội dung (bộ gác OCR)", "<span class='ok'>✔ chốt</span> · bỏ 81% khung trống, bỏ nhầm 7 slide / 2.917 khung"],
      ["OCR đa thành phần", "Qwen3-VL, content crop, kiểm EOS/lặp, tối đa một retry"],
      ["Speech-to-Text", "Giữ medium; turbo bị loại sau đối chiếu truy xuất"],
      ["Đồng bộ slide ↔ transcript", "<span class='ok'>✔ xong</span>"],
      ["Chỉ mục + truy xuất", "E5 + BM25 · principled dev-120: Hit@3 0.782"],
      ["LLM trả lời + web app", "Vercel + Supabase + R2 hoạt động; backend GPU còn chờ"],
      ["Đánh giá tự động (RAGAS)", "<span class='wip'>◐ chưa làm</span> · cần GPU"]]) + '''</div>
</div>''', "dense"))
S.append(card(1, "Luồng xử lý đã xây dựng", ns["PIPELINE_SVG"] + '<p class="note">Lần chạy riêng Kaggle 02/11: 10 video, wall time pipeline 4.950,79 s. Bộ chỉ mục mới có hồi quy nên chưa được chấp nhận thay bản tham chiếu. Nguồn: RUN_20261102.md.</p>'))

S.append(content(1, "Tổng hợp hệ thống: mỗi bước làm gì, ví dụ, kết quả", table(
    ["Bước", "Làm gì", "Công cụ", "Ví dụ thật", "Kết quả đo"],
    [["01 · Tách slide", "Tìm lúc đổi slide, lấy 1 ảnh đại diện mỗi slide", "So “chữ ký” ảnh 1 lần/giây (CPU)",
      "V008 (20 phút): 67 trạng thái → 16 đoạn", "F1 <b>0,970</b> (L25) · 0,909 (LPM)"],
     ["Bộ gác", "Bỏ OCR khung không có chữ (giảng viên, logo, quảng cáo)", "SigLIP, 14 câu mô tả, ngưỡng 0,15",
      "Logo điểm 0,00 → bỏ; slide chữ 0,99 → đọc", "Bỏ 81% khung trống, bỏ nhầm 7 slide / 2.917 khung"],
     ["02 · OCR", "Đọc chữ, công thức (LaTeX), bảng, mô tả hình", "Qwen3-VL-4B, content crop + kiểm EOS/lặp",
      "V037: “1.HOÀN CẢNH LỊCH SỬ · 2.DIỄN BIẾN …” + [Hình]", "02/11: 1 OCR lỗi, 4 retry (10 video)"],
     ["03 · Lời giảng", "Chuyển giọng nói thành chữ, có thời gian từng từ", "faster-whisper-medium + lọc khoảng lặng",
      "V008: 3.599 từ; sửa “xác xuất” → “xác suất”", "~3 phút / 30 phút video"],
     ["04 · Đồng bộ", "Gán lời giảng vào slide đang hiện, làm sạch OCR", "Mốc trạng thái hình; thời gian gốc từng từ",
      "OCR lỗi bị loại; lời giảng hợp lệ được giữ", "Mốc OCR theo visual state, lời giảng theo từ"],
     ["05 · Chỉ mục & tìm", "Cắt thành mẩu, tìm theo nghĩa + từ khóa, lấy 3 slide", "E5 + BM25, gộp RRF, chữ slide ×1,05",
      "“Xác suất tính thế nào?” → slide 1:37–2:10 hạng 1", "Principled dev-120: H1 <b>0.600</b> · H3 <b>0.782</b>"],
     ["Trả lời", "LLM trả lời từ bằng chứng, kèm mốc thời gian", "Qwen2.5-7B (Ollama), stream",
      "“Liên hợp quốc ra đời thế nào?” → … ▶ 02:10–02:38", "Trích mốc 4/5 · ngoài bài: “không đề cập”"]],
    "Chi tiết từng bước: phần 03–04 và Phụ lục"), "dense tiny"))

# 2 data
S.append(section(2))
S.append(content(2, "Dữ liệu đã thu thập", table(
    ["Bộ dữ liệu", "Video", "Giờ", "Ngôn ngữ", "Vai trò"],
    [["L25 · Báo Thanh Niên, ôn thi THPT", "88 (10 gán nhãn)", "36,2 (4,2)", "vi", "Chỉnh tham số, test tách slide & quảng cáo"],
     ["vn-lecture-test12 · 11 kênh mới", "12", "7,9", "vi", "Test chính trên kênh chưa thấy"],
     ["LPM", "244", "117", "en", "Tách slide (có sẵn mốc slide)"],
     ["M3AV · Oxford Mathematics", "51", "27", "en", "Chấm OCR công thức (LaTeX)"],
     ["SlideSpeech", "141", "90", "en", "Lọc nội dung, test ASR"]],
    "Bảng 1: Tổng 536 video, ~278 giờ (nếu bỏ 78 video L25 chưa gán nhãn: 458 video, 246 giờ); lưu trên Kaggle") + '''<p class="note">Nhãn gán tay đã có: 549 ranh giới slide và 2.068 giây quảng cáo trên L25; 74 câu hỏi có đáp án trên 6 video (48 câu soạn tự động, nhóm đang kiểm lại).</p>'''))

# 2b one slide per dataset
S.append(content(2, 'Dữ liệu 1 · L25 – kênh ôn thi THPT (tiếng Việt)', '''<div class="split">
  ''' + fig(photo(FIGS / 'ds_l25.jpg', 1000, 85), 'V008: slide bài giảng và một đoạn quảng cáo chen giữa') + '''
  <ul class="small"><li><b>Gồm:</b> 88 video (36,2 giờ) “Bí quyết ôn thi THPT 2024” của Báo Thanh Niên, 9 môn; slide + giảng viên, có quảng cáo chen giữa bài</li><li><b>Đã qua pipeline cả 88 video:</b> 5.853 trạng thái hình → 2.332 slide; bộ gác bỏ OCR 345 slide (14,8%); trung bình 15,4 phút xử lý/video (3,7–59,6) trên Kaggle T4 x2</li><li><b>Vì sao chỉ 10 video có nhãn:</b> V001–V010 là 10 video mới nhất, phủ đủ 9 môn; mỗi video phải xem hết để gán nhãn từng lần đổi hình và từng giây quảng cáo. 88 video cùng một mẫu dựng, nên thêm video cùng kênh ít giá trị hơn thêm kênh mới (→ vn-lecture-test12)</li><li><b>Nhãn gán tay:</b> 549 ranh giới slide, 2.068 giây quảng cáo, 26 câu hỏi có đáp án (V008)</li><li><b>Công dụng:</b> chỉnh ngưỡng tách slide (0,055 / 2 s), gộp slide hiện dần, bộ lọc quảng cáo, thử tìm kiếm</li><li><b>Ví dụ:</b> V008: 67 trạng thái → 16 đoạn, F1 tách slide 0,970</li></ul></div>'''))
S.append(content(2, 'Cấu trúc L25: video + nhãn gán tay', '<div class="cols2"><div><h3>Thư mục</h3><pre class="ocr small">l25/video/L25_V001.mp4 … L25_V010.mp4\ngold/slide_boundaries/gold.json   ← ranh giới slide\ngold/slide_boundaries/ad_labels.json ← giây quảng cáo\ngold/qa/questions.json            ← câu hỏi + đáp án</pre><h3>Bản ghi mẫu</h3><pre class="ocr small">{&quot;video&quot;: &quot;L25_V001&quot;, &quot;split&quot;: &quot;tuning&quot;,\n &quot;events&quot;: [{&quot;time_sec&quot;: 25.76, &quot;label&quot;: &quot;S&quot;}, …]}\n{&quot;video&quot;: &quot;L25_V001&quot;, &quot;ad_intervals&quot;: [[64,123], [761,845]]}\n{&quot;q&quot;: &quot;Xác suất của một biến cố được tính\n       như thế nào?&quot;, &quot;answer&quot;: [89, 131]}</pre></div><div><h3>Ý nghĩa các trường</h3><ul class="small"><li><code>split</code>: tuning = dùng để chỉnh ngưỡng, test = chỉ để chấm</li><li><code>events</code>: mỗi lần hình đổi máy tìm được, người xem lại và gán nhãn</li><li><code>label</code>: S slide mới · R hiện thêm cùng slide · SC cuộn trang · C quảng cáo/chuyển cảnh · L đổi khung · X không đổi nội dung. Ranh giới đúng = S + SC</li><li><code>ad_intervals</code>: các khoảng [giây đầu, giây cuối] là quảng cáo</li><li><code>answer</code>: khoảng giây trong video chứa câu trả lời; null = bài không nói tới</li></ul></div></div>'))
S.append(content(2, 'Dữ liệu 2 · vn-lecture-test12 – 11 kênh mới', '''<div class="split">
  ''' + fig(photo(FIGS / 'ds_vn12.jpg', 1000, 85), '4 trong 12 video: truyền hình, slide, bảng phấn') + '''
  <ul class="small"><li><b>Gồm:</b> 12 video, 7,9 giờ, 11 kênh chưa từng dùng khi chỉnh (VTV7, HTV, …); Toán, Lý, Hóa, Python…</li><li><b>Nhãn:</b> gold set 2: máy chạy bản nháp (12/12 xong), người sửa lại ranh giới slide, chữ OCR, lời giảng</li><li><b>Công dụng:</b> bài test cuối cùng: F1 tách slide, CER OCR, WER lời giảng; <b>không</b> chỉnh tham số trên bộ này</li><li><b>Ví dụ:</b> video bảng phấn Xác suất: không có slide in, chữ viết tay</li></ul></div>'''))
S.append(content(2, 'Cấu trúc vn-lecture-test12', '<div class="cols2"><div><h3>Thư mục</h3><pre class="ocr small">vn-lecture-test12/video/&lt;video_id&gt;.mp4   (12 file)\nvn-lecture-test12/videos.csv\ngold/vn12/  ← nhãn đang sửa: ranh giới, OCR, lời giảng</pre><h3>Bản ghi mẫu</h3><pre class="ocr small">video_id, channel, subject, title, duration, width, height\ntHE1mC_n0-E, HTV - Đài Hà Nội, Vật lí,\n  Giải đề thi môn Vật lí · tốt nghiệp THPT 2025, 2633.47, 1280, 720</pre></div><div><h3>Ý nghĩa các trường</h3><ul class="small"><li><code>video_id</code>: mã YouTube, cũng là tên file video</li><li><code>channel / subject</code>: kênh và môn: chọn 11 kênh, nhiều môn để test độ tổng quát</li><li><code>duration</code>: độ dài (giây); 12 video = 7,9 giờ</li><li><code>width, height</code>: độ phân giải đã tải (720p)</li><li><code>gold/vn12</code>: máy chạy bản nháp → người sửa lại trên trang review → xuất JSON để chấm F1, CER, WER</li></ul></div></div>'))
S.append(content(2, 'Dữ liệu 3 · LPM – bài giảng tiếng Anh có mốc slide', '''<div class="split">
  ''' + fig(photo(FIGS / 'ds_lpm.jpg', 1000, 85), 'IIfeX44RI9k (Giải phẫu): mốc slide có sẵn của bộ dữ liệu') + '''
  <ul class="small"><li><b>Nguồn:</b> Lee et al., ICCV 2023; tải lại từ YouTube</li><li><b>Gồm:</b> 244 video, 117 giờ (sinh học, giải phẫu, nha khoa)</li><li><b>Nhãn có sẵn:</b> 6.277 mốc slide (giây bắt đầu – kết thúc)</li><li><b>Công dụng:</b> kiểm ngưỡng tách slide trên dữ liệu ngoài, không cần gán tay</li><li><b>Kết quả:</b> F1 0,909 (lệch ±1 s), 0,935 (±3 s)</li></ul></div>'''))
S.append(content(2, 'Cấu trúc LPM: một dòng = một slide', '<div class="cols2"><div><h3>Thư mục</h3><pre class="ocr small">lpm-videos/video/&lt;video_id&gt;.mp4   (244 file)\nlpm-videos/videos.csv   ← thông tin video, split\nlpm-videos/slides.csv   ← 6.277 mốc slide</pre><h3>Bản ghi mẫu</h3><pre class="ocr small">speaker, course, lecture, video_id, slide_index, start_s, end_s, split\nanat-1, AnatomyPhysiology, 01, IIfeX44RI9k, 0, 25.47, 100.00, train\nanat-1, AnatomyPhysiology, 01, IIfeX44RI9k, 1, 100.00, 150.00, train</pre></div><div><h3>Ý nghĩa các trường</h3><ul class="small"><li><code>speaker / course</code>: giảng viên và khóa học (giải phẫu, sinh học, nha khoa)</li><li><code>slide_index</code>: thứ tự slide trong video</li><li><code>start_s, end_s</code>: giây slide bắt đầu và kết thúc, do tác giả LPM gán</li><li><code>cách chấm</code>: ranh giới thật = start_s của mọi slide trừ slide đầu; so với mốc máy cắt, đúng nếu lệch ≤ 1 s hoặc ≤ 3 s</li><li><code>split</code>: train / test theo giảng viên; chỉ dùng để kiểm, không huấn luyện</li></ul></div></div>'))
S.append(content(2, 'Dữ liệu 4 · M3AV – Oxford Mathematics', '''<div class="split">
  ''' + fig(photo(FIGS / 'ds_m3av.jpg', 1000, 85), 'MLS-03E08 (9:30): slide toán dày công thức') + '''
  <ul class="small"><li><b>Gồm:</b> 51 video, 27 giờ, bài giảng toán đại học</li><li><b>Nhãn có sẵn:</b> chữ trên slide dạng LaTeX + lời giảng, ví dụ: <code>d_{X \\times Y}((x_1,y_1),(x_2,y_2)) = \sqrt{d_X(x_1,x_2)^2 + d_Y(y_1,y_2)^2}</code></li><li><b>Công dụng:</b> chấm OCR công thức: so chữ máy đọc với LaTeX gốc (CER)</li><li><b>Vì sao cần:</b> L25 ít công thức phức tạp; RAG phải tìm được cả ký hiệu toán</li></ul></div>'''))
S.append(content(2, 'Cấu trúc M3AV: ảnh slide + chữ chuẩn + lời giảng', '<div class="cols2"><div><h3>Thư mục</h3><pre class="ocr small">dataset_v1.0_noaudio/MLS-03E08/\n  raw/MLS-03E08.mp4.ytbUrl        ← link YouTube\n  ocr/seg_imgs/MLS-03E08-0669000.jpg ← ảnh slide\n  ocr/seg_speech_ocr.json         ← chữ slide + lời giảng\n  speech/final+timestamps.json    ← lời giảng có mốc giờ</pre><h3>Bản ghi mẫu</h3><pre class="ocr small">{&quot;name&quot;: &quot;MLS-03E08-0669000.jpg&quot;,\n &quot;ocr_text&quot;: &quot;PRODUCT SPACES Recall that if $(X, d_{X})$ and $(Y, d_{Y})$ are metric spaces …&quot;,\n &quot;speech_text&quot;: &quot;Now we&#x27;ll turn to a different topic…&quot;,\n &quot;start&quot;: 570.0, &quot;end&quot;: 669.0}</pre></div><div><h3>Ý nghĩa các trường</h3><ul class="small"><li><code>name</code>: tên ảnh slide; số cuối = mốc thời gian (669000 ms)</li><li><code>ocr_text</code>: chữ trên slide do người kiểm, công thức viết bằng LaTeX ($…$) → đáp án để chấm OCR</li><li><code>speech_text</code>: lời giảng trong lúc slide hiện</li><li><code>start, end</code>: khoảng giây slide xuất hiện</li><li><code>final+timestamps</code>: từng câu nói có mốc thời gian (dạng 0012700_0022080 = 12,7 s → 22,08 s)</li></ul></div></div>'))
S.append(content(2, 'Dữ liệu 5 · SlideSpeech – bài nói có slide', '''<div class="split">
  ''' + fig(photo(FIGS / 'ds_slidespeech.jpg', 1000, 85), 'Hai video trong tập test (chủ đề design)') + '''
  <ul class="small"><li><b>Gồm:</b> 141 video, 90 giờ, bài nói tiếng Anh có slide, nhiều chủ đề</li><li><b>Nhãn có sẵn:</b> lời nói chuẩn (transcript) theo từng câu</li><li><b>Công dụng:</b> đo WER của bước Speech-to-Text; 12 video <b>không có quảng cáo</b> để kiểm bộ lọc không xóa nhầm bài giảng (giữ ≥ 99%)</li><li><b>Ví dụ:</b> video bài nói có minh họa vẽ tay: bộ gác vẫn phải giữ lại để OCR</li></ul></div>'''))
S.append(content(2, 'Cấu trúc SlideSpeech', '<div class="cols2"><div><h3>Thư mục</h3><pre class="ocr small">slidespeech-devtest/video/dev|test/&lt;wavid&gt;.mp4\nslidespeech-devtest/related_files/  ← đoạn, lời nói chuẩn, OCR slide\nslidespeech-devtest/videos.csv</pre><h3>Bản ghi mẫu</h3><pre class="ocr small">wavid, youtube_id, category, split, file, status, duration\ndesign_0000, -6sEcMzz0ic, design, test,\n  video/test/design_0000.mp4, ok, 1016.11</pre></div><div><h3>Ý nghĩa các trường</h3><ul class="small"><li><code>wavid</code>: mã video của SlideSpeech, khớp với file lời nói chuẩn</li><li><code>category</code>: chủ đề bài nói (design, medical, education…)</li><li><code>split</code>: dev / test (bộ train để riêng, chưa dùng)</li><li><code>status</code>: ok = tải được; failed = video YouTube đã bị ẩn/xóa</li><li><code>related_files</code>: lời nói chuẩn từng đoạn → đáp án để tính WER cho Speech-to-Text</li></ul></div></div>'))

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
    <li>Ô <b>người thật, cảnh trường, món ăn</b> = quảng cáo; <b>giảng viên + slide nhỏ</b> vẫn là bài giảng.</li></ul>
    <div class="kpis one"><div><b>67 → 16</b><span>trạng thái hình → 12 slide + 4 đoạn quảng cáo (V008, 20 phút)</span></div></div></div>
  ''' + fig(embed("v008_contact_sheet.jpg"), "Hình 5: Keyframe của 16 đoạn trong video L25_V008") + "</div>"))
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
S.append(content(3, "1b.1 · Vấn đề: một slide hiện ra từng dòng", '''<div class="cols2">
  <div class="panel"><h3>Chuyện gì xảy ra</h3><ul class="small">
    <li>Giảng viên bấm cho <b>từng dòng lời giải hiện ra</b>. Mỗi lần bấm, hình đổi → bước 1 tưởng là <b>slide mới</b>.</li>
    <li>V008 có slide “Câu 1” hiện dần qua <b>10 trạng thái</b> (2:33 → 5:19). Không gộp thì có 10 slide gần giống nhau:
        OCR 10 lần, chỉ mục lặp chữ, câu trả lời trích nhầm trạng thái thiếu.</li></ul></div>
  <div class="panel"><h3>Ý tưởng</h3><ul class="small">
    <li>So <b>từng cặp trạng thái liền nhau</b>: nếu nội dung cũ <b>vẫn nằm nguyên chỗ cũ</b> và chỉ <b>có thêm</b> → cùng một slide → <b>GỘP</b>.</li>
    <li>Nếu nội dung cũ biến mất → <b>TÁCH</b> thành slide mới.</li>
    <li>Gộp xong, chỉ giữ <b>trạng thái cuối</b> (đầy đủ nhất) để OCR.</li></ul></div></div>'''))
S.append(content(3, "1b.1 · Ví dụ thật: slide “Câu 1” của V008", fig(embed("reveal_chain.jpg"),
    "Trạng thái 9 → 18 (10 lần bấm) giữ nguyên chữ cũ, chỉ thêm dòng → GỘP thành 1 slide. Trạng thái 19 là “Câu 2”: chữ cũ chỉ còn 31% → TÁCH.", "wide")))
S.append(content(3, "1b.2 · Code: ba con số quyết định gộp hay tách", '<div class="cols2 code-cols">' + code("""
def content_mask(view):              # đâu là nét chữ/hình?
    return abs(view - medianBlur(view, 41)) > 28

def reveal_evidence(prev, cur, template):
    old = content_mask(prev) & ~template   # bỏ logo, khung cố định
    new = content_mask(cur)  & ~template
    preserved = |prev - cur| nhỏ (cho lệch 1 px)
    return {
      "content_retained": (old & preserved).sum() / old.sum(),
      "content_growth":   new.sum() / old.sum(),
      "image_change":     mean(|prev - cur| > 32)}

if (retained >= 0.85 and change <= 0.25
        and growth >= 0.9):
    GỘP        # nội dung cũ còn, chỉ thêm
else:
    TÁCH       # slide mới
""") + '''<div><ul class="small">
    <li><b>Nét chữ</b> = điểm ảnh khác hẳn nền xung quanh (so với ảnh làm mờ). Logo, khung có trong hầu hết video bị bỏ qua.</li>
    <li><b>Còn đúng chỗ</b> (retained): trong các nét chữ cũ, bao nhiêu % vẫn còn đúng vị trí ở trạng thái sau. <b>≥ 85%</b>.</li>
    <li><b>Hình đổi</b> (change): bao nhiêu % điểm ảnh của cả khung thay đổi. Hiện thêm 1 dòng chỉ đổi ~1%. <b>≤ 25%</b>.</li>
    <li><b>Nội dung ×</b> (growth): lượng nét chữ sau / trước. <b>≥ 0,9</b> = không bị xóa bớt.</li>
    <li>Cả 3 đạt → GỘP. Thêm một kiểm tra: trạng thái mới vẫn phải giữ ≥ 60% nội dung của <b>trạng thái đầu</b>, để không gộp dây chuyền A~B~C khi C đã khác A.</li></ul></div></div>'''))
S.append(content(3, "1b.3 · Máy “nhìn” thấy gì khi so hai trạng thái", fig(embed("reveal_mask.jpg"),
    "Tính lại bằng chính code pipeline trên trạng thái 11 → 12 của V008 (khớp số trong reveal_audit.json).", "wide") + '''
  <p class="note" style="margin-top:8px">Toàn bộ nét xanh dương (chữ cũ) vẫn nằm đúng chỗ; phần xanh lá là dòng “TH1 …” vừa hiện thêm → GỘP.</p>'''))
S.append(content(3, "Bước 1 · Gộp slide hiện ra từng dòng (V008)", '''<div class="split">
  <div><ul class="small">
    <li>V008: <b>67 trạng thái hình → 16 đoạn</b>; 7 slide được gộp từ 2–12 trạng thái.</li>
    <li>Mỗi hàng bên phải là một slide sau khi gộp, từ trạng thái đầu đến cuối.</li>
    <li>Hàng 2 (giảng viên) là trường hợp cảnh quay gần như đứng yên cũng được gộp.</li></ul>
    <div class="kpis one"><div><b>0,970</b><span>F1 tách slide · 6 video test L25 (P 0,990 · R 0,951)</span></div></div></div>
  ''' + fig(embed("v008_reveal_sheet.jpg"), "Mỗi hàng là các trạng thái được gộp thành một slide (V008)") + "</div>"))
S.append(content(3, "Bộ gác OCR: khung này có chữ để đọc không?", '''<div class="cols2">
  <div class="panel"><h3>Vì sao cần</h3><ul class="small">
    <li>Sau bước 1, mỗi đoạn có 1 keyframe. Nhưng <b>không phải keyframe nào cũng là slide</b>:
        giảng viên nói vào camera, logo kênh, quảng cáo.</li>
    <li>OCR là bước <b>chậm nhất</b>, ~85% thời gian xử lý.
        Đọc khung không có chữ vừa <b>tốn thời gian</b> vừa sinh <b>chữ rác</b> vào chỉ mục.</li>
    <li>Bộ gác đứng <b>trước</b> OCR: chấm mỗi keyframe một điểm 0 → 1 = “khả năng có nội dung đọc được”.</li></ul></div>
  <div class="panel"><h3>Quy tắc</h3><ul class="small">
    <li>Điểm ≥ <b>0,15</b> → đưa vào OCR.</li>
    <li>Điểm &lt; 0,15 → <b>bỏ OCR</b>. Đoạn vẫn được giữ, <b>lời giảng vẫn vào tìm kiếm</b>
        (không mất nội dung thầy nói).</li>
    <li>Ngưỡng đặt <b>thấp</b> có chủ ý: thà đọc thừa một khung còn hơn bỏ sót một slide.</li></ul></div></div>'''))
S.append(content(3, "Bộ gác: ví dụ 4 loại keyframe và điểm", fig(embed("gate_examples.jpg"),
    "Kết quả thật trên V008. Thanh = điểm bộ gác (0 → 1), vạch đứng = ngưỡng 0,15.", "wide") + '''
  <p class="note" style="margin-top:10px">Slide chữ gần 1; giảng viên kèm slide nhỏ vẫn 0,79; quảng cáo và logo gần 0 → không tốn thời gian OCR.</p>'''))
S.append(content(3, "1c.2 · Điểm được tính thế nào (SigLIP, không cần huấn luyện)", '<div class="cols2 code-cols">' + code("""
CONTENT_PROMPTS = [            # 8 câu "có nội dung"
  "a slide of a lecture presentation with text",
  "a lecture slide with a diagram, chart, table or formula",
  "a page of a document or an exam with questions",
  "a whiteboard or blackboard with handwriting",
  "a chart or graph with axes and labels",          # 4 câu thêm sau
  "a scientific figure, diagram or illustration in a lesson",
  "math exercises or equations on a board or card",
  "computer code in a text editor or terminal"]
NO_CONTENT_PROMPTS = [         # 6 câu "không có nội dung"
  "a person talking to the camera",
  "a teacher speaking in a studio",
  "a logo or a title animation",
  "a blank screen",
  "a building or a landscape photo",
  "a group of people on a stage or outdoors"]

def score(images):
    sims = embed(images) @ embed(PROMPTS).T   # giống câu nào?
    probs = softmax(sims * scale + bias)      # 14 xác suất, tổng = 1
    return probs[:, :8].sum(1)                # phần của 8 câu "có nội dung"
""") + '''<div><ul class="small">
    <li><b>SigLIP</b> (Google) là model đã học trên hàng trăm triệu cặp <b>ảnh – câu mô tả</b>:
        nó biến ảnh và câu thành vector, ảnh nào khớp câu nào thì hai vector gần nhau.</li>
    <li>Ta viết sẵn <b>14 câu mô tả</b>: 8 câu “có nội dung”, 6 câu “không có nội dung” (6 câu sau thêm từ lần quét 2.917 khung, slide 1c.5).</li>
    <li>Với mỗi keyframe, SigLIP cho biết ảnh <b>giống câu nào nhất</b> → chia thành 14 xác suất cộng lại bằng 1.</li>
    <li><b>Điểm = tổng xác suất của 8 câu “có nội dung”</b>.
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
S.append(content(3, "Bộ gác: quét lại trên 2.917 khung (90 video)", '''<div class="cols2">
  <div>''' + table(["Cách", "Slide có nội dung bị bỏ nhầm", "Khung trống bỏ được"], [
    ["Cũ: 8 câu, ngưỡng 0,15", "34 (+9 thẻ tiêu đề)", "405/530 (76%)"],
    ["Cũ, hạ ngưỡng 0,05", "9", "269/530 (51%)"],
    ["<b>Mới: 14 câu, ngưỡng 0,15 (chốt)</b>", "<b>7</b> (+9 thẻ tiêu đề)", "<b>427/530 (81%)</b>"],
    ["Chỉ bộ dò chữ CRAFT", "1–16", "12–22%"],
    ["SigLIP <i>và</i> CRAFT cùng đồng ý", "0", "12–19%"],
    ["Sigmoid thay softmax", "≈ 2.300", "—"]]) + '''</div>
  <div class="panel"><h3>Kết luận</h3><ul class="small">
    <li>742 khung được xem và gán nhãn bằng mắt; slide bị bỏ sót chủ yếu ở kênh mới: thẻ toán lớp 2, màn hình code CS50.</li>
    <li><b>Thêm 6 câu mô tả</b> (biểu đồ, hình khoa học, bài tập toán, code; ảnh phong cảnh, nhóm người): bỏ nhầm 34 → 7, khung trống bỏ được còn tăng 76% → 81%.</li>
    <li><b>Không thêm bộ dò chữ:</b> CRAFT thấy “chữ” ở logo, phụ đề, tên giảng viên trên hầu hết khung trống → gần như không bỏ được gì.</li>
    <li>Giữ <b>softmax</b>, ngưỡng <b>0,15</b>, <code>frame+crops</code>. 9 thẻ tiêu đề vẫn bị bỏ: ít thông tin, chấp nhận.</li></ul></div></div>''', "dense"))
S.append(content(3, "Bước 1c · Kết quả demo trên video V008", fig(embed("gate_grid.jpg"),
    "Khung xanh = đọc OCR · khung đỏ = bỏ OCR (điểm &lt; 0,15): bỏ đúng 3 quảng cáo (#3, #9, #12) + 2 logo; giảng viên kèm slide nhỏ vẫn được đọc.", "wide")))
S.append(content(3, "Bước 2 · OCR: Qwen3-VL-4B đọc nội dung slide", '''<div class="split">
  ''' + fig(embed("v037_keyframe.jpg"), "Keyframe L25_V037 (Cao trào cách mạng 1930–1931)") + '''
  <div><h3>Kết quả OCR (Qwen3-VL, nguyên văn)</h3><pre class="ocr small">-Khung hoàng kinh tế thế giới 1929-1933
làm kinh tế Việt Nam suy thoái, khủng hoảng nặng nề.
-Các tầng lớp xã hội bị tác động, đời sống khó khăn bản cùng …
1.HOÀN CẢNH LỊCH SỬ · 2.DIỄN BIẾN · 3.Ý NGHĨA LỊCH SỬ
[Hình] Sơ đồ ba phần: Hoàn cảnh lịch sử, Diễn biến, Ý nghĩa lịch sử …</pre>
  <h3 style="margin-top:8px">Cách cũ (hybrid) trên cùng slide</h3><pre class="ocr small">THANK NIEN
-Khùng hoàng kinh tế thế giới … khùng hoàng nặng nè.
2.DIỂN BIẾN · 3.ý NGHĨA LICH SỰ · Thầy NGUYÊN VIỆT ĐẦNG DU …</pre>
  <p class="note">Ví dụ lưu từ 10/2026, còn lỗi “Khung hoàng”, “bản cùng”. Bản 11/2026 đọc vùng nội dung và kiểm EOS/lặp trước khi dùng làm bằng chứng.</p></div></div>'''))
S.append(content(3, "Cải tiến trích xuất (11/2026)", '''<div class="cols2">
  <div><h3>Qwen OCR</h3><ul class="small">
    <li><b>Content crops:</b> ưu tiên ROI đã cấu hình, rồi vùng tài liệu hoặc vùng hiển thị. Giữ cả khung nếu không có crop phù hợp.</li>
    <li><b>Kiểm đầu ra:</b> phải có token EOS thật, không rỗng và không có vòng lặp kéo dài.</li>
    <li><b>Tối đa một retry:</b> prompt đọc nguyên văn. Lưu cả hai lần thử và lý do lỗi.</li>
    <li>Retry vẫn lỗi: loại OCR khỏi bằng chứng, giữ lời giảng hợp lệ.</li></ul></div>
  <div><h3>Mốc thời gian và nhóm nội dung</h3><ul class="small">
    <li><b>State onsets:</b> lưu thời điểm bắt đầu từng trạng thái hình. OCR trích mốc của trạng thái đại diện.</li>
    <li>Lời giảng vẫn theo thời gian gốc từng từ và nhóm sở hữu. Không suy ra thời gian cho từng dòng OCR.</li>
    <li><b>Same-content merge:</b> gộp nhóm có cùng nội dung. Principled chỉ thêm onsets + merge vào cấu hình legacy.</li>
    <li>Bỏ tùy chọn riêng cho bảng viết tay. Tùy chọn editor cho <code>y1CvB950VlU</code> vẫn cần kiểm chứng.</li></ul></div>
</div><p class="note">Nguồn: PORT_V3.md; ABLATION_20261102.md. OCR mặc định là Qwen, ASR được giữ ở medium sau đánh giá.</p>''', "dense"))
S.append(content(3, "Bước 2b · Công thức và hình vẽ (bài giảng tiếng Anh)", '''<div class="cols2">
  <div>''' + fig(embed("mls15da1_keyframe.jpg"), "Hình 6: Slide viết tay MLS-15DA1 (9:27)") + '''<pre class="ocr small">Def: A vector v ∈ V is an eigenvector of T if Tv = λv
for some λ ∈ R.
Proof. λ is an eigenvalue of T ⇔ Ker(T − λ) ≠ ∅</pre></div>
  <div>''' + fig(embed("mls15da1_region.jpg"), "Hình 7: Vùng hình được Qwen3-VL mô tả") + '''<pre class="ocr small">“A hand-drawn diagram showing a cube transforming
into a skewed rectangular prism …”</pre>
  <p class="note">Mô tả hình theo ngôn ngữ của bài (tiếng Anh). Chữ viết tay đôi khi đọc sai ký hiệu.</p></div></div>'''))
S.append(content(3, "Bước 2 · Vì sao chọn Qwen3-VL (1/2): tốc độ trên 4 video", '''<div class="cols2"><div>''' + table(
    ["Video", "Slide", "Qwen3-VL", "Hybrid", "Nhanh hơn"], [
    ["L25_V037 · tiếng Việt", "10", "95 s", "199 s", "×2,1"],
    ["hVU5Mw19V_c · toán viết tay", "18", "227 s", "705 s", "×3,1"],
    ["design_0004 · tiếng Anh", "20", "175 s", "1.255 s", "×7,2"],
    ["medical_0000 · tiếng Anh", "33", "208 s", "371 s", "×1,8"]],
    "Benchmark 10/2026: bước 02 trên Kaggle 2×T4, cùng keyframe và cùng bộ gác") + '''</div>
  <div class="panel"><h3>So sánh hai cách</h3><ul class="small">
    <li><b>Hybrid</b> (cách cũ): Unlimited-OCR đọc chữ, rồi Qwen3-VL đọc riêng từng vùng hình → nhiều lần gọi model.</li>
    <li><b>Qwen3-VL cả slide</b>: một lần gọi cho cả slide, <b>mỗi GPU một bản</b> → nhanh hơn <b>×1,8–7</b>.</li>
    <li>Chữ sạch hơn: hybrid lặp 32–202 dòng mỗi video, có slide bịa vòng lặp; Qwen gần như không lặp.</li>
    <li>Hybrid vẫn giữ làm tùy chọn (có tọa độ từng vùng hình).</li></ul></div></div>'''))
S.append(content(3, "Bước 2 · Vì sao chọn Qwen3-VL (2/2): độ chính xác", '''<div class="cols2"><div>''' + table(
    ["Công cụ OCR", "Đúng từ (VI)", "Đúng dấu (VI)", "Đúng từ (M3AV)"],
    [["<b>Qwen3-VL cả slide</b>", "<b>94%</b>", "<b>94%</b>", "83%"],
     ["Hybrid", "80%", "76%", "84%"],
     ["Unlimited-OCR", "76%", "70%", "80%"],
     ["EasyOCR (0,6 s/slide)", "74%", "71%", "61%"],
     ["GOT-OCR 2.0", "24%", "3%", "52%"]],
    "Benchmark 10/2026: 20 slide tiếng Việt + 40 slide toán M3AV có đáp án") + '''</div>
  <div class="panel"><h3>Đọc bảng thế nào</h3><ul class="small">
    <li><b>Đúng từ</b>: tỉ lệ từ trong đáp án mà công cụ đọc ra được.</li>
    <li><b>Đúng dấu</b>: chỉ tính các từ có dấu tiếng Việt, đo việc giữ dấu.</li>
    <li>Tiếng Việt: Qwen3-VL hơn hẳn; toán tiếng Anh: Qwen3-VL và hybrid ngang nhau.</li>
    <li>Còn yếu: số viết tay, đôi khi để trống dòng [Hình].</li>
    <li>Đáp án tiếng Việt do Claude gõ, nhóm đang kiểm lại.</li></ul></div></div>'''))
S.append(content(3, "Bước 3–4 · Speech-to-Text và đồng bộ", '''<div class="cols2">
  <div class="panel"><h3>Speech-to-Text</h3><ul class="small">
    <li><b>faster-whisper-medium</b>, lọc khoảng lặng, có <b>thời gian từng từ</b>.</li>
    <li>Giữ medium sau đối chiếu với turbo trên dev-120 và legacy-49.</li>
    <li>Làm sạch bằng <code>glossary_code</code>, giữ thời gian gốc và provenance của từ.</li></ul>
    <h3 style="margin-top:14px">Đồng bộ</h3><ul class="small">
    <li>Lời giảng theo thời gian từng từ và nhóm slide. OCR theo onset của trạng thái đại diện.</li>
    <li>Làm sạch OCR, loại đầu ra lỗi khỏi bằng chứng truy xuất.</li>
    <li>Không loại lời giảng hợp lệ chỉ vì OCR lỗi.</li></ul></div>
  <div><h3>Ví dụ đã lưu (V008 · 10/2026)</h3>
    <div class="pair"><b>Chữ trên slide</b><p>ĐỊNH NGHĨA XÁC SUẤT · P(A) = n(A)/n(Ω) …</p>
    <b>Lời giảng gán vào slide</b><p>“P A sẽ bằng với lại N A chia cho N Omega. Trong đó N A là số kết quả thuận lợi của biến cố A …”</p>
    <b>Sửa ASR</b><p><s>xác xuất</s> → <b>xác suất</b> · <s>sát xuất</s> → <b>xác suất</b></p></div></div></div>'''))
S.append(content(3, "Bước 3–4 · Kết quả demo trên video V008", fig(embed("sync_demo.jpg"),
    "Ví dụ 10/2026: mỗi từ Whisper (vạch) có thời gian và được gán vào slide đang hiện.", "wide")))

S.append(content(3, "Lần chạy riêng Kaggle 02/11", '''<p><b>10 video</b>, kernel private <code>lecture-rag-fast-10-20261102</code> đã hoàn tất.</p>''' + table(
    ["Đo thực tế", "Kết quả"],
    [["Wall time pipeline", "<b>4.950,79 s</b> (các bước chạy chồng nhau)"],
     ["GPU cấp phát (2 T4/session)", "Full: <b>2,8665 T4-hours</b>; canary + full: <b>3,0784 T4-hours</b>, không tính hàng đợi"],
     ["OCR trên 10 video", "<b>1 lỗi / 4 retry</b>; 28 cờ cần xem lại. OCR lỗi bị loại, lời giảng được giữ"],
     ["ASR, dev-120 (H1 / H3 / MRR / CR)", "Medium: <b>0.591 / 0.764 / 0.673 / 0.745</b><br>Turbo: 0.564 / 0.773 / 0.662 / 0.727"],
     ["Quyết định", "<b>Loại turbo, giữ medium.</b> Cả hai bộ chỉ mục mới vẫn hồi quy so với old reference"]],
    "Nguồn: RUN_20261102.md. So sánh CPU giữ cùng stage-01/02, dùng transcript medium đã cache và turbo của run.") + '''
<p class="note">Validator gốc báo lỗi đường dẫn ảnh nên run chưa đạt acceptance. Audit tài sản offline đã đạt 10/10, không đồng nghĩa chất lượng truy xuất đã đạt.</p>''', "dense eval"))

# 4 retrieval
S.append(section(4))
S.append(card(4, "Bước 5 · Chỉ mục và truy xuất đa phương thức", ns["RAG_SVG"] + '''<ul class="small">
  <li>Mỗi slide thành các mẩu: chữ OCR, lời giảng, vùng hình → vector <b>E5</b> + chỉ mục từ khóa <b>BM25</b>, chữ slide được ưu tiên nhẹ (×1,05).</li>
  <li>Gộp 2 kết quả bằng RRF (xếp hạng hợp nhất), lấy <b>3 slide</b>; câu "slide này" lấy slide đang xem; câu hỏi nối tiếp ("còn … thì sao?") được tìm cùng câu hỏi trước.</li>
  <li>LLM <b>Qwen2.5-7B</b> (Ollama) chỉ trả lời từ bằng chứng, trích mốc [mm:ss]; không có thì nói "bài giảng không đề cập".</li></ul>'''))
S.append(content(4, "Bước 5 · Kết quả demo: câu hỏi → slide tìm được (V008)", fig(embed("retrieval_demo.jpg"),
    "Hình: 3 slide điểm cao nhất cho mỗi câu hỏi (chỉ mục E5, chạy 06/10): hạng 1 đều đúng slide chứa đáp án.", "wide")))
S.append(content(4, "Web app sau khi nâng cấp", '''<div class="split">''' + fig(embed("webapp_answer.jpg"),
    "Hỏi trên L25_V011 (chạy thật, 13/10): “Liên hợp quốc ra đời trong hoàn cảnh nào?”") + '''
  <div class="panel"><h3>Người dùng thấy gì</h3><ul class="small">
    <li><b>Chữ hiện dần</b> khi LLM đang viết (stream), không phải chờ màn hình trống.</li>
    <li>Mốc <b>▶ 02:10–02:38</b> trong câu trả lời: bấm là video tua tới đúng đoạn.</li>
    <li>Dòng <b>“Nguồn”</b>: 3 slide dùng làm bằng chứng, cũng bấm được.</li>
    <li>Hỏi tiếp <b>“Còn ASEAN thì sao?”</b> cùng ngữ cảnh câu trước.</li></ul>
    <h3 style="margin-top:12px">Trạng thái triển khai</h3><ul class="small">
    <li><b>Vercel frontend + Supabase pgvector + Cloudflare R2</b> đang hoạt động.</li>
    <li><b>Backend GPU trên Vast.ai chưa hoàn tất.</b> LLM trên CPU còn chậm, demo đã lưu mất ~170 s tới chữ đầu.</li></ul>
    <p class="note">Kiến trúc: docs/deploy.md. Ảnh là demo 13/10, không phải ảnh backend GPU đã triển khai.</p></div></div>''', "dense"))
S.append(content(4, "Ví dụ hỏi đáp (bộ 26 câu đầu, chạy trên Kaggle T4)", table(
    ["Câu hỏi", "Trả lời của hệ thống", ""],
    [["Xác suất của một biến cố được tính như thế nào?", "P(A) = n(A)/n(Ω), n(A): số kết quả thuận lợi …", "✔"],
     ["Không gian mẫu khi lấy 2 quả cầu từ hộp 15 quả?", "C(15,2) = 105 phần tử", "✔"],
     ["Tổng hai số là số chẵn khi nào?", "Cả hai cùng lẻ hoặc cả hai cùng chẵn", "✔ (qwen2:7b từng sai)"],
     ["What is the leading coefficient of the characteristic polynomial?", "(−1)ⁿ, nêu ở phần chứng minh 34:31", "✔"],
     ["Cách cài đặt PyTorch? (ngoài bài)", "Bài giảng không đề cập", "✔"]],
    "Demo 10/2026 trên bộ 26 câu cũ: Qwen3-4B 24/26 đúng, Qwen2.5-7B 4-bit 25/26, ~15 s/câu trên T4. Đây không phải đánh giá câu trả lời trên dev-120.")))

# 5 results
S.append(section(5))
S.append(content(5, "Tiêu chí đánh giá (1/2) · từng bước pipeline", table(
    ["Bước", "Chỉ số", "Đo cái gì", "Dữ liệu", "Hiện có"],
    [["01 · Tách slide", "Precision, Recall, <b>F1</b> (lệch ±1 s / ±3 s)", "ranh giới máy cắt có trùng ranh giới thật không", "L25 · LPM · vn12", "L25 0,970 · LPM 0,909 / 0,935"],
     ["Lọc nội dung", "% giây QC bị loại · % bài giảng giữ lại", "bỏ quảng cáo mà không xóa nhầm bài (mục tiêu ≥ 90% / ≥ 99%)", "L25 · video không QC · vn12", "giữ slide có chữ 100%"],
     ["02 · OCR", "<b>CER</b> = ký tự sai / tổng ký tự · giây/slide", "đọc đúng chữ, công thức; tốc độ", "M3AV · vn12", "M3AV 0,36 · vn12 đang gán"],
     ["03 · Lời giảng", "<b>WER</b> = (thay + xóa + chèn) / số từ", "nhận dạng đúng lời giảng", "vn12 (24 đoạn) · SlideSpeech", "đang gán"]],
    "Bảng: CER, WER càng thấp càng tốt; F1, Recall càng cao càng tốt")))
S.append(content(5, "Tiêu chí đánh giá (2/2) · hỏi đáp và hiệu năng", table(
    ["Phần", "Chỉ số", "Đo cái gì", "Dữ liệu"],
    [["05 · Tìm kiếm", "<b>Hit@1/3</b>, <b>MRR</b>, <b>context recall</b>", "hạng slide có đáp án và bằng chứng thực sự còn trong ngữ cảnh", "dev-120: 110 có đáp án / 10 ngoài bài; legacy-49: 44 / 5"],
     ["Câu trả lời (LLM)", "LLM chấm: <b>đúng</b> · <b>bám nguồn</b> · <b>dẫn đúng mốc</b> · từ chối đúng", "trả lời đúng bài, không bịa, chỉ đúng slide/thời điểm, biết nói “bài không đề cập”", "cùng bộ câu hỏi + câu ngoài bài"],
     ["Hiệu năng", "wall time · T4-hours · độ trễ trả lời", "thời gian pipeline và GPU cấp phát, không cộng các bước chạy chồng", "lần chạy riêng 10 video ngày 02/11"]],
    "Dev-120 chưa được người kiểm và không phải test độc lập. Legacy không có nhãn từ khóa thống nhất: dùng retained-time-overlap, không gọi là context recall.")))
S.append(content(5, "Kết quả đo được (dev-120 và benchmark cũ)", table(
    ["Thành phần", "Tập đo", "Kết quả"],
    [["Phát hiện slide", "6 video test L25", "F1 <b>0,970</b> · LPM 0,909 (ngưỡng sau quét 15 cấu hình)"],
     ["Bộ gác OCR", "481 khung; quét lại 2.917 khung (78 L25 + 12 kênh khác)", "481 khung: giữ slide có chữ <b>100%</b>; quét lại: bỏ nhầm <b>7</b> slide, bỏ <b>81%</b> khung trống"],
     ["OCR công thức", "slide toán M3AV", "CER <b>0,36</b> (GOT-OCR 2.0: 0,52)"],
     ["Truy xuất: old reference", "dev-120 (110 câu có đáp án)", "Hit@1 0.591 · Hit@3 0.782 · MRR 0.674 · CR 0.764"],
     ["<b>Truy xuất: principled</b>", "<b>cùng dev-120, chưa human review</b>", "<b>Hit@1 0.600 · Hit@3 0.782<br>MRR 0.679 · context recall 0.755</b>"],
     ["Câu trả lời LLM", "dev-120", "Chưa đánh giá RAGAS; điểm truy xuất không đo độ đúng hoặc từ chối của LLM"],
     ["Thời gian pipeline", "10 video, lần chạy riêng 02/11", "<b>4.950,79 s</b>; canary + full <b>3,0784 T4-hours</b>"]],
    "Benchmark tách slide, bộ gác và OCR giữ từ 10/2026, không phải đo lại ngày 02/11. Dev: PHASE3.md; audit/reproduced-summary.json. Thời gian: RUN_20261102.md."), "dense"))
S.append(content(5, "Ablation trích xuất: dev-120 và legacy-49", '''<p>Dev: <b>110 có đáp án / 10 ngoài bài</b>. Legacy: <b>44 / 5</b>, cùng tập giao cố định.</p>''' + table(
    ["Cấu hình", "Dev: H1 / H3 / MRR / CR", "Legacy: H1 / H3 / MRR / time"],
    [[label, metric_line(dev), metric_line(legacy)] for label, dev, legacy in ABLATION],
    "Legacy ở bảng này dùng chấm gốc: giao thời gian nhóm dù rất ngắn. Time là retained-time-overlap, không phải CR.") + '''
<p class="note"><b>Principled là số báo cáo.</b> Recommended-final chọn override theo từng video trên dev, nên kết quả lạc quan và chưa chứng minh khả năng tổng quát.</p>
<p class="note">Nguồn: RUN_20261102.md; ABLATION_20261102.md; audit/reproduced-summary*.json. Chi tiết scoring ở Phụ lục.</p>''', "dense eval"))
S.append(content(5, "Độ tin cậy và hồi quy trích xuất", '''<div class="cols2">
  <div><h3>Hồi quy có nguyên nhân</h3><ul class="small">
    <li>Tùy chọn board riêng cho <code>hVU5Mw19V_c</code> làm <b>18 thành 117 nhóm</b>, 73 thành 240 chunk.</li>
    <li>Medium ASR giữ nguyên 5.113 từ. Nhóm ngắn cạnh tranh top-3 và làm mất lời giảng lân cận hữu ích.</li>
    <li>Đổi nhóm, ảnh đại diện và OCR tương tác với nhau. Chưa tách riêng được tác động của crop.</li>
    <li>Onset timing không đổi xếp hạng, nhưng có thể đổi phần bằng chứng được chấm.</li></ul></div>
  <div><h3>Giới hạn của số cải thiện</h3><ul class="small">
    <li><b>Principled:</b> legacy + onsets + merge, cùng cached medium. Dev H1 0.600, CR 0.755 so với 0.764 cũ.</li>
    <li><b>Recommended-final:</b> H1 0.636 có override board, editor, chart và L25_V008 được chọn trên dev.</li>
    <li><b>Strict minimum-overlap, legacy-49:</b> old / principled / recommended-final đều <b>0.750 / 0.864 / 0.792</b> (H1 / H3 / MRR).</li>
    <li>Kết quả cached OCR không bảo đảm một lần trích xuất GPU mới sẽ lặp lại kết quả đó.</li></ul></div>
</div><p class="note">Nguồn: ABLATION_20261102.md; audit/boundary-diagnostic.json. Strict scoring là chẩn đoán riêng trên ba cấu hình đã kiểm toán.</p>''', "dense"))

S.append(content(5, "Lần chạy kiểm chứng 03/11", f'''<p><b>Chạy lại từ đầu trên GPU, cả 10 video hoàn tất</b> · <code>{CONFIRM['kernel']}</code></p>
<p>Bỏ override merge riêng của <b>L25_V008</b>; giữ medium, state onsets, same-content merge. hVU5Mw19V_c về lại 18 nhóm.
Dev H1 0.627 cao hơn principled cached (0.600) nhưng vẫn chấm trên cùng bộ dev.</p>''' + table(
    ["Tập chấm", "Hit@1", "Hit@3", "MRR", "CR / time"],
    [["Dev-120", *[confirm_value(f"dev_{k}") for k in ("hit1", "hit3", "mrr", "context_recall")]],
     ["Legacy-49, chấm gốc", *[confirm_value(f"legacy_{k}") for k in ("hit1", "hit3", "mrr", "time_overlap")]],
     ["Legacy-49, strict", *[confirm_value(f"strict_{k}") for k in ("hit1", "hit3", "mrr")], "không áp dụng"]]) + table(
    ["T4-hours cấp phát", "Wall time pipeline (s)", "OCR lỗi", "OCR retry"],
    [[confirm_value(k) for k in ("t4_hours", "wall_seconds", "ocr_failures", "ocr_retries")]]) + '''
<p class="note">Nguồn: confirm-eval-summary.json (kernel CPU lecture-rag-confirm-cpu-eval-20261104, đã bỏ 12 đoạn speech chỉ có “%”, điểm không đổi); gpu-ledger.json.</p>''', "dense eval"))

S.append(content(5, "Hạn chế hiện tại (nói thẳng)", '''<div class="cols2">
  <div class="panel"><h3>Số liệu cần kiểm thêm</h3><ul class="small">
    <li><b>120 câu dev do AI soạn, chưa được người kiểm.</b> Chưa có test độc lập để kết luận khả năng tổng quát.</li>
    <li>Recommended-final có override chọn trên dev. Principled cũng chỉ là kết quả phát triển, chưa phải kiểm chứng mới.</li>
    <li><b>RAGAS chưa thực hiện</b>, chưa so với LLM không RAG. Điểm truy xuất không đo độ đúng câu trả lời.</li>
    <li>Benchmark OCR tiếng Việt cũ dùng đáp án AI soạn, còn cần người kiểm.</li></ul></div>
  <div class="panel"><h3>Hệ thống</h3><ul class="small">
    <li><b>LLM trên CPU còn chậm.</b> Backend GPU Vast.ai chưa hoàn tất dù Vercel, Supabase và R2 đã hoạt động.</li>
    <li>Run 02/11 có <b>1 OCR lỗi, 4 retry</b> và cờ cần xem lại. Chữ viết tay, số và mã nguồn còn sai.</li>
    <li>Tùy chọn editor riêng cho <code>y1CvB950VlU</code> vẫn còn, cần ablation và kiểm tra trên video mới.</li>
    <li>Kết quả run kiểm chứng 03/11 chưa có trong báo cáo.</li></ul></div></div>''', "dense"))
# 6 plan
S.append(section(6))
S.append(content(6, "Kế hoạch tiếp theo", '''<div class="road">''' + "".join(
    f'<div class="step {cls}"><b>{n}</b><span>{t}</span><i>{s}</i></div>'
    for n, t, s, cls in [(1, "Audit run kiểm chứng 03/11", "điền CONFIRM, đối chiếu dev-120 / legacy-49", "now"),
                         (2, "Người kiểm câu hỏi + OCR", "kiểm 120 câu dev, tạo test độc lập, giữ gold cố định", "next"),
                         (3, "RAGAS + so với LLM không RAG", "đúng, bám nguồn, mốc dẫn và từ chối", "next"),
                         (4, "Hỏi trên nhiều video", "cả khóa học, trích [video · mm:ss]", "next"),
                         (5, "Hoàn tất backend Vast.ai", "nối frontend và đo độ trễ LLM thực tế", "next"),
                         (6, "Ablation editor + viết luận", "giới hạn override, tách dev và test", "next")]) + '''</div>
<div class="key"><span class="k now">đang làm</span><span class="k next">tiếp theo</span></div>'''))
S.append(content(6, "Xin ý kiến cô", '''<div class="asks">
  <div><b>01</b><p>Với <strong>dev-120 do AI soạn</strong>, cần bao nhiêu người kiểm và cách tạo test độc lập thế nào?</p></div>
  <div><b>02</b><p>Báo cáo <strong>principled</strong> làm kết quả chính, để cấu hình có override ở phần ablation có phù hợp không?</p></div>
  <div><b>03</b><p>RAGAS cần kết hợp người chấm thế nào để đánh giá <strong>bám nguồn, mốc dẫn và từ chối</strong>?</p></div>
  <div><b>04</b><p>Ưu tiên <strong>hoàn tất backend GPU Vast.ai</strong> cho demo hay mở rộng hỏi đáp cả khóa học trước?</p></div></div>'''))

S.append(content(5, "Phụ lục · Recipe OCR và chính sách cấu hình", table(
    ["Phần", "Cấu hình / bằng chứng"],
    [["Qwen OCR", "Qwen3-VL-4B-Instruct, FP16/SDPA, content crop; lưu crop, ROI và provenance"],
     ["Kiểm đầu ra và retry", "EOS thật; không rỗng; dưới 8 lần lặp liên tiếp (cụm 1–32 từ). Lần đầu 1.536 token, penalty 1.05; một retry 2.048 token, penalty 1.10"],
     ["Thí nghiệm OCR trước production", "29 ảnh: CER 0.33755 thành 0.31033; lỗi/lặp 4/29 thành 0/29. Không suy ra mức tăng tốc toàn pipeline"],
     ["Principled", "Legacy + state onsets + same-content merge, cached medium, cùng cleaning và retrieval. Không thêm tùy chọn board/editor riêng"],
     ["Recommended-final (cached)", "Board: 18 nhóm + OCR cũ; L25_V008: merge OFF; editor: nhóm an toàn + ảnh/OCR tương thích; chart: bảo vệ OCR của một ảnh đã kiểm"],
     ["Builder dùng cho kiểm chứng", "Medium, onsets + merge ON; bỏ override board và L25_V008; còn editor y1CvB950VlU. Chưa chứng minh fresh OCR lặp lại cached hybrid"]],
    "Nguồn: PORT_V3.md; ABLATION_20261102.md. Recipe OCR, lựa chọn cache và builder kiểm chứng là các cấu hình khác nhau."), "dense"))
S.append(content(5, "Phụ lục · Legacy-49: đối chiếu cách chấm", table(
    ["Cấu hình", "Audit mặc định: H1 / H3 / MRR / time", "Chấm gốc theo thời gian nhóm"],
    [["Old reference", "0.795 / 0.932 / 0.848 / 0.886", "0.818 / 0.955 / 0.871 / 0.909"],
     ["Principled", "0.795 / 0.932 / 0.848 / 0.886", "0.818 / 0.955 / 0.871 / 0.909"],
     ["Recommended-final", "0.795 / 0.955 / 0.864 / 0.909", "0.818 / 0.955 / 0.875 / 0.909"]],
    "Cùng 49 câu (44 có đáp án). Cột trái: reproduced-summary.json. Cột phải: reproduced-summary-time.json và bảng ablation gốc.") + '''
<ul class="small" style="margin-top:14px">
  <li>Chấm gốc chấp nhận mọi giao thời gian nhóm dương. Hai trường hợp board chỉ giao ranh giới 0,08 s nên có thể được cộng điểm.</li>
  <li><b>Strict minimum-overlap:</b> cả ba cấu hình đã audit đều H1 <b>0.750</b>, H3 <b>0.864</b>, MRR <b>0.792</b>. Không gộp số này vào bảng chấm gốc.</li>
  <li>Legacy time là bằng chứng thời gian còn trong ngữ cảnh, không đo key-term context recall. Strict của new+medium/turbo chưa có trong các nguồn audit này.</li></ul>
<p class="note">Nguồn: audit/reproduced-summary*.json; boundary-diagnostic.json; ABLATION_20261102.md.</p>''', "dense"))
S.append(f'''<section class="slide end-slide">
  <div class="logo-white end-logo"></div>
  <h2 class="end-title">Cảm ơn cô<br>đã lắng nghe</h2>
  <p class="end-sub">Xin ý kiến góp ý</p>
</section>''')

APPENDIX = ("Dữ liệu 1 ·", "Cấu trúc L25", "Dữ liệu 2 ·", "Cấu trúc vn-lecture", "Dữ liệu 3 ·", "Cấu trúc LPM",
            "Dữ liệu 4 ·", "Cấu trúc M3AV", "Dữ liệu 5 ·", "Cấu trúc SlideSpeech",
            "1.1 ·", "1.2 ·", "1.3 ·", "1.4 ·", "1.5 ·", "1.6 ·", "1b.1 ·", "1b.2 ·", "1b.3 ·",
            "1c.2 ·", "1c.3 ·", "1c.4 ·", "Bước 1c · Kết quả demo", "Bước 2b ·",
            "Tiêu chí đánh giá (1/2)", "Tiêu chí đánh giá (2/2)",
            "Bước 2 · Vì sao chọn Qwen3-VL", "Ví dụ hỏi đáp (bộ 26", "Phụ lục ·")
detail = [s for s in S if any(f"<h2>{key}" in s for key in APPENDIX)]
S[:] = [s for s in S if s not in detail] + [section(7)] + detail
EXTRA = '''
.split{{display:grid;grid-template-columns:0.95fr 1.25fr;gap:26px;align-items:start}}
.fig{{margin:0}} .fig img{{width:100%;max-height:330px;object-fit:contain;background:#fff;border:1px solid #d8deee;border-radius:10px;display:block}}
.cols2 .fig img{{max-height:250px}}
.fig.wide img{{max-height:385px}}
.fig figcaption{{font-size:13px;color:var(--sub);text-align:center;margin-top:6px;font-style:italic}}
.ocr{{background:#f6f8fd;border-left:5px solid var(--gold);border-radius:8px;padding:12px 14px;font-size:15px;white-space:pre-wrap;margin:6px 0;font-family:Consolas,monospace}}
.ocr.small{{font-size:13px;padding:8px 12px}}
.pair p{{margin:2px 0 12px;font-size:16px}} .pair b{{color:var(--blue)}}
.ok{{color:#2f7d4a;font-weight:700}} .wip{{color:#c58a00;font-weight:700}}
.kpis.one{{grid-template-columns:1fr}}
.kpis.two{{grid-template-columns:1fr 1fr;gap:12px}} .kpis.two b{{font-size:30px}} .kpis.two span{{font-size:13px}}
.dense table{{font-size:14px}} .dense td,.dense th{{padding:5px 9px}}
.eval table{{font-size:17px}} .eval p{{margin:6px 0 10px}}
.eval .note,.eval .tcap{{font-size:15px;line-height:1.4}}
.eval table + table{{margin-top:14px}}
.tiny table{{font-size:13px}} .tiny td,.tiny th{{padding:4px 8px;line-height:1.3}}
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
.asks{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.asks div{{display:flex;gap:14px;align-items:flex-start;background:var(--soft);border-radius:14px;padding:16px 18px}}
.asks b{{flex:none;display:grid;place-items:center;width:42px;height:42px;border-radius:50%;background:var(--blue);color:#fff;font-size:17px}}
.asks p{{margin:0;font-size:19px;line-height:1.45}}
'''
exec(compile("CSS = f'''" + tail.replace("\n'''\n", "\n" + EXTRA + "'''\n", 1), "tail", "exec"), {**ns, "S": S})
