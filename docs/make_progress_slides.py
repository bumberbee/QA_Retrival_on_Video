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
      ["Lọc quảng cáo / khung trống", "<span class='wip'>◐ ~70%</span>"],
      ["OCR đa thành phần", "<span class='ok'>✔ chạy tốt</span>"],
      ["Speech-to-Text", "<span class='ok'>✔ chạy tốt</span>"],
      ["Đồng bộ slide ↔ transcript", "<span class='ok'>✔ xong</span>"],
      ["Chỉ mục + truy xuất", "<span class='ok'>✔ xong</span> · Hit@3 22/22"],
      ["LLM trả lời + web app", "<span class='wip'>◐ ~60%</span>"]]) + '''</div>
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
S.append(content(3, "Bước 1 · Phát hiện slide", '''<div class="split">
  <div><ul class="small">
    <li>Quét video <b>1 lần</b>: mẫu 1 khung/giây + khung 8 Hz để chỉnh ranh giới.</li>
    <li>"Chữ ký" mỗi khung: ảnh xám thu nhỏ + ảnh cạnh (bắt cả thay đổi chữ).</li>
    <li>Slide mới khi khung <b>khác</b> (≥ 0,055) và <b>đứng yên</b> ≥ 2 giây.</li>
    <li>Bỏ qua vùng luôn chuyển động (khung giảng viên).</li>
    <li>Chọn khung <b>nét nhất</b> làm keyframe.</li></ul>
    <p class="note">V008 (20 phút): 67 trạng thái hình → 12 slide + 4 đoạn quảng cáo.</p></div>
  ''' + fig(photo(V008 / "01_slides/contact_sheet.jpg", 1100), "Hình 2: Keyframe các đoạn của video L25_V008") + "</div>"))
S.append(content(3, "Bước 1b · Gộp slide hiện dần", '''<div class="split">
  <div><ul class="small">
    <li>Giảng viên bấm cho <b>từng dòng hiện ra</b> → một slide có nhiều trạng thái.</li>
    <li>Gộp khi nội dung cũ <b>còn đúng chỗ</b> (≥ 85%), hình đổi ít (≤ 25%) và chỉ <b>thêm</b> nội dung.</li>
    <li>Giữ trạng thái cuối (đầy đủ nhất) cho OCR.</li></ul>
    <div class="kpis one"><div><b>0,970</b><span>F1 tách slide · 6 video test L25 (P 0,990 · R 0,951)</span></div></div></div>
  ''' + fig(photo(V008 / "01_slides/reveal_sheet.jpg", 1100, top_share=0.42), "Hình 3: Mỗi hàng là các trạng thái được gộp thành một slide") + "</div>"))
S.append(content(3, "Bước 1c · Lọc quảng cáo", '''<div class="split">
  <div><ul class="small">
    <li>SigLIP chấm <b>từng giây</b> "giống quảng cáo không" (model học từ L25).</li>
    <li>Đổi từ <b>cắt bỏ</b> sang <b>chỉ đánh dấu</b>: đoạn nghi vẫn OCR, giữ lời giảng, chỉ không đưa vào tìm kiếm.</li>
    <li><b>Ngưỡng an toàn:</b> &gt; 6 đoạn hoặc &gt; 30% video → bỏ mọi dấu (kênh lạ).</li>
    <li>L25 loại <b>99,8%</b> quảng cáo; 12 video tiếng Anh không quảng cáo giữ <b>100%</b> bài giảng (bản cũ cắt mất 47–100%).</li></ul></div>
  ''' + fig(report_png(HERE / "run-pipeline-L25_V008.html"), "Hình 4: Điểm quảng cáo theo giây trên V008; vùng cam = đoạn bị đánh dấu", "wide") + "</div>"))
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

# 4 retrieval
S.append(section(4))
S.append(card(4, "Bước 5 · Chỉ mục và truy xuất đa phương thức", ns["RAG_SVG"] + '''<ul class="small">
  <li>Mỗi slide thành các mẩu: chữ OCR, lời giảng, vùng hình → vector <b>E5</b> + chỉ mục từ khóa <b>BM25</b>.</li>
  <li>Gộp 2 kết quả bằng RRF; câu "slide này" ưu tiên slide đang xem; thêm slide trước/sau để liền mạch.</li>
  <li>LLM nhận ≤ 6.000 token bằng chứng + ≤ 3 ảnh slide, trả lời kèm mốc thời gian.</li></ul>'''))
S.append(content(4, "Ví dụ hỏi đáp", table(
    ["Câu hỏi", "Slide tìm được", "Trả lời của hệ thống"],
    [["Xác suất của một biến cố được tính như thế nào?", "1:38 – 2:10 ✔", "P(A) = n(A)/n(Ω) … ✔"],
     ["Không gian mẫu khi lấy 2 quả cầu từ 15 quả?", "2:33 – 5:20 ✔", "105 phần tử ✔"],
     ["What is the definition of an eigenvector?", "4:33 – 6:34 ✔", "Tv = λv, v ≠ 0 ✔"],
     ["Tổng hai số là số chẵn khi nào?", "2:33 – 5:20 ✔", "“một lẻ một chẵn” ✘ (LLM sai)"],
     ["Cách cài đặt PyTorch? (ngoài bài)", "—", "“Bài giảng không đề cập” ✔"]],
    "Bảng 2: Ví dụ trên web app (LLM: qwen2:7b chạy trên máy)")))

# 5 results
S.append(section(5))
S.append(content(5, "Kết quả đo được", table(
    ["Thành phần", "Tập đo", "Kết quả"],
    [["Phát hiện slide", "6 video test L25", "F1 <b>0,970</b>"],
     ["Lọc quảng cáo (đánh dấu)", "6 video L25 · 12 video tiếng Anh", "loại QC <b>99,8%</b> · giữ bài giảng <b>100%</b>"],
     ["Bộ gác OCR", "481 khung L25 · M3AV · LPM", "giữ slide có chữ <b>100%</b>"],
     ["OCR công thức", "slide toán M3AV", "CER <b>0,36</b> (GOT-OCR 2.0: 0,52)"],
     ["Lời giảng thuộc bài (E5)", "9 video L25", "AUC tách quảng cáo <b>0,94</b>"],
     ["Truy xuất", "22 câu hỏi có đáp án", "Hit@1 <b>18/22</b> · Hit@3 <b>22/22</b>"],
     ["Câu trả lời LLM", "13 câu", "<b>9/13</b> đúng (qwen2:7b)"],
     ["Thời gian", "video 30 phút, Kaggle T4 x2", "<b>~21 phút</b> (OCR ~85%)"]],
    "Bảng 3: Tổng hợp kết quả hiện tại")))
S.append(content(5, "Đã thử nhưng không chọn", table(
    ["Thử nghiệm", "Kết quả", "Quyết định"],
    [["GOT-OCR 2.0 thay OCR hiện tại", "Nhẹ hơn (3 GB) nhưng mất dấu tiếng Việt, CER kém hơn", "Giữ OCR kết hợp"],
     ["Lọc quảng cáo “tự so với số đông” trong video", "Bắt 83% QC nhưng đánh dấu nhầm 11% bài giảng", "Chưa dùng, để làm tín hiệu phụ"],
     ["fp16 cho Unlimited-OCR", "Chậm hơn 11%, có slide đọc sai", "Giữ bf16"],
     ["Bỏ vùng trang trí lặp cho Qwen", "Chỉ nhanh hơn ~1%", "Không bật"]],
    "Bảng 4: Các thử nghiệm đã loại")))

# 6 plan
S.append(section(6))
S.append(content(6, "Kế hoạch tiếp theo", '''<div class="road">''' + "".join(
    f'<div class="step {cls}"><b>{n}</b><span>{t}</span><i>{s}</i></div>'
    for n, t, s, cls in [(1, "Chạy lại 10 video L25", "cấu hình mới, nạp web app", "now"),
                         (2, "LLM mạnh hơn", "Gemini · trả lời theo ngôn ngữ", "next"),
                         (3, "Gold set 2", "12 video tiếng Việt: CER, WER, F1", "next"),
                         (4, "Lớp quyết định", "gộp 3 tín hiệu lọc nội dung", "next"),
                         (5, "Tăng tốc OCR", "giảm độ phân giải đầu vào", "next"),
                         (6, "Viết khóa luận", "chương thực nghiệm, kết quả", "next")]) + '''</div>
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
.fig figcaption{{font-size:13px;color:var(--sub);text-align:center;margin-top:6px;font-style:italic}}
.ocr{{background:#f6f8fd;border-left:5px solid var(--gold);border-radius:8px;padding:12px 14px;font-size:15px;white-space:pre-wrap;margin:6px 0;font-family:Consolas,monospace}}
.ocr.small{{font-size:13px;padding:8px 12px}}
.pair p{{margin:2px 0 12px;font-size:16px}} .pair b{{color:var(--blue)}}
.ok{{color:#2f7d4a;font-weight:700}} .wip{{color:#c58a00;font-weight:700}}
.kpis.one{{grid-template-columns:1fr}}
'''
exec(compile("CSS = f'''" + tail.replace("\n'''\n", "\n" + EXTRA + "'''\n", 1), "tail", "exec"), {**ns, "S": S})
