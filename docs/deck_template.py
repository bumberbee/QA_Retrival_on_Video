"""Deck template shared by docs/make_progress_slides.py: the school look (Template Powerpoint
v1.pptx: backgrounds, logos, divider, embedded as base64), the CSS and the slide helpers
(content, section, card, table, diagrams). The progress report runs everything above `S = []`.
Run directly, it still builds the old 37-slide defense draft (no longer kept in docs/), a starting
point for the real defense deck."""
import base64, io, sys
from pathlib import Path
from PIL import Image

MEDIA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
NAMES = "NGUYỄN THÀNH TRUNG, THÁI VĂN TRUNG"
SECTIONS = ["Giới thiệu & xác định vấn đề", "Mục tiêu nghiên cứu", "Nghiên cứu liên quan", "Kiến trúc hệ thống",
            "Thực nghiệm", "Kết quả", "Kết luận", "Tài liệu tham khảo"]


def img(name, width, fmt="PNG", quality=85):
    image = Image.open(MEDIA / name)
    image = image.convert("RGBA" if fmt == "PNG" else "RGB")
    image.thumbnail((width, width * 4))
    buffer = io.BytesIO()
    image.save(buffer, fmt, **({"quality": quality} if fmt == "JPEG" else {"optimize": True}))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()


BG_TITLE = img("image2.jpg", 1920, "JPEG", 80)
LOGO_WHITE = img("image3.png", 900)
BG_CONTENT = img("image4.png", 1920, "JPEG", 82)
DIVIDER = img("image5.png", 1200)
LOGO = img("image6.png", 900)
BG_CARD = img("image7.png", 1920, "JPEG", 80)
BG_END = img("image8.png", 1920, "JPEG", 82)


def content(section_no, title, body, extra_class=""):
    crumb = f"{section_no}. {SECTIONS[section_no - 1]}" if section_no else ""
    return f'''<section class="slide content {extra_class}">
  <div class="logo"></div>
  <div class="divider"></div>
  <div class="crumb">{crumb}</div>
  <h2>{title}</h2>
  <div class="body">{body}</div>
  <div class="names">{NAMES}</div>
</section>'''


def section(number):
    return f'''<section class="slide content section">
  <div class="logo"></div>
  <div class="divider"></div>
  <div class="sec-num">{number:02d}</div>
  <h2 class="sec-title">{SECTIONS[number - 1]}</h2>
  <div class="names">{NAMES}</div>
</section>'''


def card(section_no, title, inner):
    return f'''<section class="slide card-slide">
  <div class="logo-white small"></div>
  <div class="crumb light">{section_no}. {SECTIONS[section_no - 1]}</div>
  <h2 class="card-title">{title}</h2>
  <div class="card">{inner}</div>
</section>'''


def table(head, rows, caption=""):
    out = "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
    for row in rows:
        out += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return out + "</table>" + (f'<p class="tcap">{caption}</p>' if caption else "")


def metric(name, purpose, definition, formula, meaning):
    return f'''<div class="metric">
  <p><b>Mục đích:</b> {purpose}</p>
  <p><b>Định nghĩa:</b> {definition}</p>
  <div class="formula">{formula}</div>
  <p><b>Ý nghĩa:</b> {meaning}</p></div>'''


PIPELINE_SVG = '''<svg viewBox="0 0 1000 300" role="img" aria-label="Sơ đồ kiến trúc">
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#5a6a8a"/></marker></defs>
  <g font-family="Be Vietnam Pro, sans-serif" font-size="13" fill="#5a6a8a">
  <rect x="10" y="115" width="110" height="70" rx="12" fill="#eef1fa" stroke="#153898" stroke-width="2"/>
  <text x="65" y="146" text-anchor="middle" font-weight="700" font-size="17" fill="#0e1f5c">Video</text>
  <text x="65" y="168" text-anchor="middle">vi / en</text>
  <rect x="150" y="30" width="150" height="90" rx="12" fill="#fff" stroke="#153898" stroke-width="2"/>
  <text x="165" y="56" font-weight="700" font-size="12" fill="#c58a00">01 · CPU</text>
  <text x="165" y="80" font-weight="700" font-size="17" fill="#0e1f5c">Phát hiện slide</text>
  <text x="165" y="102">keyframe + thời gian</text>
  <rect x="320" y="30" width="160" height="90" rx="12" fill="#fff8e6" stroke="#f6b21b" stroke-width="2.5"/>
  <text x="335" y="56" font-weight="700" font-size="12" fill="#c58a00">SigLIP</text>
  <text x="335" y="80" font-weight="700" font-size="17" fill="#0e1f5c">Lọc nội dung</text>
  <text x="335" y="102">quảng cáo · khung trống</text>
  <rect x="500" y="30" width="170" height="90" rx="12" fill="#fff" stroke="#153898" stroke-width="2"/>
  <text x="515" y="56" font-weight="700" font-size="12" fill="#c58a00">02 · GPU</text>
  <text x="515" y="80" font-weight="700" font-size="17" fill="#0e1f5c">OCR đa thành phần</text>
  <text x="515" y="102">Unlimited-OCR + Qwen</text>
  <rect x="150" y="190" width="520" height="80" rx="12" fill="#fff" stroke="#153898" stroke-width="2" stroke-dasharray="6 4"/>
  <text x="165" y="216" font-weight="700" font-size="12" fill="#c58a00">03 · GPU · chạy song song</text>
  <text x="165" y="240" font-weight="700" font-size="17" fill="#0e1f5c">Speech-to-Text</text>
  <text x="165" y="260">faster-whisper-medium · thời gian từng từ</text>
  <rect x="700" y="105" width="130" height="90" rx="12" fill="#fff" stroke="#153898" stroke-width="2"/>
  <text x="715" y="131" font-weight="700" font-size="12" fill="#c58a00">04</text>
  <text x="715" y="155" font-weight="700" font-size="17" fill="#0e1f5c">Đồng bộ</text>
  <text x="715" y="177">slide ↔ transcript</text>
  <rect x="855" y="30" width="135" height="90" rx="12" fill="#fff" stroke="#153898" stroke-width="2"/>
  <text x="870" y="56" font-weight="700" font-size="12" fill="#c58a00">05</text>
  <text x="870" y="80" font-weight="700" font-size="17" fill="#0e1f5c">Chỉ mục</text>
  <text x="870" y="102">E5 + BM25</text>
  <rect x="855" y="180" width="135" height="90" rx="12" fill="#153898"/>
  <text x="870" y="212" font-weight="700" font-size="17" fill="#fff">RAG + LLM</text>
  <text x="870" y="236" fill="#dfe6ff">trả lời + mốc giờ</text></g>
  <g fill="none" stroke="#5a6a8a" stroke-width="1.8">
  <path d="M120,140C135,140 135,75 148,75" marker-end="url(#a)"/><path d="M120,160C135,160 135,230 148,230" marker-end="url(#a)" stroke-dasharray="5 4"/>
  <path d="M300,75L318,75" marker-end="url(#a)"/><path d="M480,75L498,75" marker-end="url(#a)"/>
  <path d="M670,75C688,75 688,135 698,135" marker-end="url(#a)"/><path d="M670,230C688,230 688,165 698,165" marker-end="url(#a)"/>
  <path d="M830,140C843,140 843,75 853,75" marker-end="url(#a)"/><path d="M922,120L922,178" marker-end="url(#a)"/></g>
</svg>'''

RAG_SVG = '''<svg viewBox="0 0 1000 170" role="img" aria-label="Luồng truy xuất">
  <defs><marker id="b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#5a6a8a"/></marker></defs>
  <g font-family="Be Vietnam Pro, sans-serif" font-size="13" fill="#5a6a8a" text-anchor="middle">
  <rect x="5" y="55" width="120" height="60" rx="12" fill="#eef1fa" stroke="#153898" stroke-width="2"/><text x="65" y="82" font-weight="700" font-size="16" fill="#0e1f5c">Câu hỏi SV</text><text x="65" y="101">+ giây đang xem</text>
  <rect x="150" y="10" width="150" height="58" rx="12" fill="#fff" stroke="#153898" stroke-width="2"/><text x="225" y="36" font-weight="700" font-size="15" fill="#0e1f5c">Theo nghĩa</text><text x="225" y="55">E5 · top 20</text>
  <rect x="150" y="102" width="150" height="58" rx="12" fill="#fff" stroke="#153898" stroke-width="2"/><text x="225" y="128" font-weight="700" font-size="15" fill="#0e1f5c">Theo từ khóa</text><text x="225" y="147">BM25 · top 20</text>
  <rect x="325" y="55" width="120" height="60" rx="12" fill="#fff8e6" stroke="#f6b21b" stroke-width="2.5"/><text x="385" y="82" font-weight="700" font-size="15" fill="#0e1f5c">Gộp RRF</text><text x="385" y="101">nghĩa 1 · từ 0,5</text>
  <rect x="470" y="55" width="150" height="60" rx="12" fill="#fff" stroke="#153898" stroke-width="2"/><text x="545" y="82" font-weight="700" font-size="15" fill="#0e1f5c">≤ 4 slide neo</text><text x="545" y="101">"slide này" → đang xem</text>
  <rect x="645" y="55" width="140" height="60" rx="12" fill="#fff" stroke="#153898" stroke-width="2"/><text x="715" y="82" font-weight="700" font-size="15" fill="#0e1f5c">+ slide trước/sau</text><text x="715" y="101">≤ 6.000 token</text>
  <rect x="810" y="55" width="185" height="60" rx="12" fill="#153898"/><text x="902" y="82" font-weight="700" font-size="15" fill="#fff">LLM → trả lời</text><text x="902" y="101" fill="#dfe6ff">kèm mốc mm:ss</text></g>
  <g fill="none" stroke="#5a6a8a" stroke-width="1.8">
  <path d="M125,75L148,45" marker-end="url(#b)"/><path d="M125,95L148,125" marker-end="url(#b)"/>
  <path d="M300,40L323,70" marker-end="url(#b)"/><path d="M300,130L323,100" marker-end="url(#b)"/>
  <path d="M445,85L468,85" marker-end="url(#b)"/><path d="M620,85L643,85" marker-end="url(#b)"/><path d="M785,85L808,85" marker-end="url(#b)"/></g>
</svg>'''

S = []
# ---- cover
S.append(f'''<section class="slide title-slide">
  <div class="logo-white"></div>
  <div class="cover-top">TRƯỜNG ĐẠI HỌC CÔNG NGHIỆP THÀNH PHỐ HỒ CHÍ MINH<br>KHOA CÔNG NGHỆ THÔNG TIN</div>
  <div class="cover-major">GRADUATION THESIS<br>MAJOR: COMPUTER SCIENCE</div>
  <div class="title-box">
    <p class="kicker">KHÓA LUẬN TỐT NGHIỆP</p>
    <h1>MULTIMODAL LECTURE RAG</h1>
    <p class="subtitle">Hệ thống hỏi đáp đa phương thức trên video bài giảng</p>
    <div class="people">
      <div><span>Sinh viên thực hiện:</span><b>NGUYỄN THÀNH TRUNG</b> 22696701<br><b>THÁI VĂN TRUNG</b> 22696241</div>
      <div><span>Giảng viên hướng dẫn:</span><b>TS. ĐẶNG THỊ PHÚC</b></div>
    </div>
  </div>
  <div class="cover-date">Hồ Chí Minh, 10/2026</div>
</section>''')
# ---- agenda
S.append(content(0, "Nội dung trình bày", '<ol class="agenda">' + "".join(
    f'<li><span>{n:02d}</span>{name}</li>' for n, name in enumerate(SECTIONS, 1)) + "</ol>"))
# ---- 1 intro
S.append(section(1))
S.append(content(1, "Bối cảnh", '''<ul class="lead">
  <li>Video bài giảng trực tuyến (ôn thi, đại học, MOOC) ngày càng nhiều; một bài thường dài <b>20–45 phút</b>.</li>
  <li>Kiến thức nằm ở <b>nhiều phương thức</b>: chữ, công thức, bảng, biểu đồ trên slide và lời giảng của giảng viên.</li>
  <li>Mô hình ngôn ngữ lớn (LLM) trả lời tốt nhưng <b>không “xem” được video</b> và dễ bịa nội dung khi thiếu căn cứ.</li>
  <li>Retrieval-Augmented Generation (RAG) [1] giúp LLM trả lời dựa trên tài liệu, nhưng chủ yếu cho văn bản thuần.</li></ul>'''))
S.append(content(1, "Vấn đề thực tiễn", '''<div class="cols2">
  <div class="panel"><h3>Khó khăn</h3><ul>
    <li>Tìm lại một ý trong video phải <b>tua thủ công</b>.</li>
    <li>Nội dung slide và lời giảng <b>lệch nhau</b>, cần đồng bộ theo thời gian.</li>
    <li>Video có <b>quảng cáo, đoạn giới thiệu</b> chen giữa.</li>
    <li>Tiếng Việt có dấu: OCR và ASR dễ sai (“xác xuất”).</li></ul></div>
  <div class="panel gold"><h3>Mục tiêu tổng quát</h3>
    <p>Xây dựng hệ thống <b>Multimodal Lecture RAG</b>: từ video bài giảng tạo dữ liệu slide + transcript có thời gian, rồi trả lời câu hỏi của sinh viên kèm <b>bằng chứng và mốc thời gian</b> trong video.</p></div>
</div>'''))
# ---- 2 objectives
S.append(section(2))
S.append(content(2, "Mục tiêu hệ thống", '''<div class="cols2">
  <div><h3>1 · Xử lý video → dữ liệu đầu vào</h3><ul>
    <li>Phát hiện slide, thời gian xuất hiện, trích keyframe đại diện.</li>
    <li>OCR tiêu đề, văn bản, công thức, bảng, hình/đồ thị.</li>
    <li>Speech-to-Text lời giảng theo từng khoảng slide.</li>
    <li>Đồng bộ slide ↔ transcript, giữ thứ tự bài giảng.</li></ul></div>
  <div><h3>2 · Multimodal Lecture RAG</h3><ul>
    <li>Đầu vào: câu hỏi của sinh viên (tiếng Việt / tiếng Anh).</li>
    <li>Truy xuất slide + transcript liên quan, thêm đoạn trước/sau.</li>
    <li>LLM kết hợp câu hỏi và bằng chứng để trả lời, giải thích.</li>
    <li>Web app: xem video, slide, hỏi đáp, bấm tới đúng thời điểm.</li></ul></div>
</div>
<p class="flow">Câu hỏi SV → Multimodal Retrieval → Slide + Transcript + Context → LLM → Trả lời</p>'''))
S.append(content(2, "Khoảng trống nghiên cứu", '''<ul class="lead">
  <li>Các bộ dữ liệu bài giảng đa phương thức (LPM [2], SlideSpeech [3], M3AV [4]) chủ yếu <b>tiếng Anh</b> và phục vụ nhận dạng, chưa phải hỏi đáp.</li>
  <li>Thiếu hệ thống hỏi đáp trên <b>video bài giảng tiếng Việt</b> có slide, công thức, quảng cáo chen giữa.</li>
  <li>Bộ lọc học từ một kênh <b>không tổng quát</b> sang kênh khác (Chương 6: cắt nhầm 47–100% video tiếng Anh).</li>
  <li>Cần trả lời có <b>căn cứ kiểm chứng được</b>: mốc thời gian và slide nguồn.</li></ul>'''))
# ---- 3 related work
S.append(section(3))
S.append(content(3, "Nghiên cứu liên quan", table(
    ["Hướng", "Công trình", "Hạn chế so với bài toán"],
    [["RAG", "Lewis et al. [1]", "Truy xuất văn bản thuần, chưa có slide / thời gian video"],
     ["Bộ dữ liệu bài giảng", "LPM [2], SlideSpeech [3], M3AV [4]", "Tiếng Anh; tập trung nhận dạng slide, ASR, OCR"],
     ["Speech-to-Text", "Whisper [5]", "Đa ngôn ngữ nhưng nhầm thuật ngữ tiếng Việt"],
     ["Hiểu ảnh – văn bản", "SigLIP [6], Qwen3-VL [9], GOT-OCR 2.0 [10]", "Mô hình riêng lẻ, chưa ghép thành pipeline"],
     ["Truy xuất lai", "E5 [7], BM25 [11], RRF [8]", "Cần điều chỉnh cho bằng chứng slide + lời giảng"]])))
# ---- 4 architecture
S.append(section(4))
S.append(card(4, "Kiến trúc tổng thể", PIPELINE_SVG + '<p class="note">Hình 1: Kiến trúc Multimodal Lecture RAG. Speech-to-Text chỉ cần âm thanh nên chạy song song với phát hiện slide và OCR.</p>'))
S.append(content(4, "Phát hiện slide & lọc nội dung", '''<div class="cols2">
  <div class="panel"><h3>Phát hiện slide (01)</h3><ul>
    <li>Quét video 1 Hz + khung 8 Hz, tìm ranh giới hình ổn định.</li>
    <li>Gộp <b>slide hiện dần</b> (thêm dòng trên nền cũ), giữ ảnh cuối.</li>
    <li>Bỏ vùng chuyển động (khung giảng viên), chọn ảnh nét nhất.</li></ul></div>
  <div class="panel gold"><h3>Lọc nội dung</h3><ul>
    <li><b>Đánh dấu</b> quảng cáo (SigLIP + hồi quy logistic), không cắt bỏ.</li>
    <li><b>Ngưỡng an toàn:</b> &gt; 6 đoạn hoặc &gt; 30% video → bỏ mọi dấu.</li>
    <li><b>Bộ gác OCR:</b> khung không có chữ (giảng viên nói, logo) không OCR nhưng giữ lời giảng.</li></ul></div>
</div>'''))
S.append(content(4, "OCR đa thành phần & Speech-to-Text", '''<div class="cols2">
  <div class="panel"><h3>OCR kết hợp (02)</h3><ul>
    <li><b>Unlimited-OCR</b>: chữ, công thức LaTeX, bảng, kèm tọa độ.</li>
    <li><b>Qwen3-VL</b> [9]: đọc từng vùng hình / biểu đồ (số liệu, mô tả).</li>
    <li>Khối hình lớn ≥ 40% slide được đọc lại để cứu chữ.</li></ul></div>
  <div class="panel"><h3>Speech-to-Text (03)</h3><ul>
    <li><b>faster-whisper-medium</b> (Whisper [5]), lọc khoảng lặng VAD.</li>
    <li>Thời gian <b>từng từ</b> để gán lời vào đúng slide.</li>
    <li>Ngôn ngữ theo video (vi / en).</li></ul></div>
</div>'''))
S.append(content(4, "Đồng bộ & chỉ mục", '''<div class="cols2">
  <div class="panel"><h3>Đồng bộ (04)</h3><ul>
    <li>Từ được nói lúc slide nào hiện thì gán vào slide đó.</li>
    <li>Làm sạch OCR: bỏ logo, tên giảng viên, dòng lặp.</li>
    <li>Sửa từ ASR nghe nhầm theo chữ trên slide (“xác xuất” → “xác suất”).</li>
    <li>Chấm lời giảng có thuộc chủ đề bài không (E5 [7]).</li></ul></div>
  <div class="panel"><h3>Chỉ mục (05)</h3><ul>
    <li>Chunk theo slide: chữ OCR, lời giảng, vùng hình.</li>
    <li>Vector <b>multilingual-E5</b> [7] + chỉ mục từ khóa <b>BM25</b> [11] (SQLite FTS5).</li>
    <li>Đoạn bị đánh dấu quảng cáo không lập chỉ mục.</li></ul></div>
</div>'''))
S.append(card(4, "Multimodal Retrieval → LLM", RAG_SVG + '''<ul class="small">
  <li>Gộp hai danh sách bằng Reciprocal Rank Fusion [8]; câu hỏi “slide này” ưu tiên slide đang xem.</li>
  <li>Ngữ cảnh gửi LLM: tiêu đề, chữ OCR, lời giảng, mô tả hình của ≤ 4 slide neo + slide lân cận, kèm ≤ 3 ảnh slide.</li>
  <li>LLM trả lời và trích nguồn dạng <b>mm:ss</b> để sinh viên bấm tới đúng đoạn video.</li></ul>'''))
# ---- 5 experiments
S.append(section(5))
S.append(content(5, "5.1 Dữ liệu", table(
    ["Bộ dữ liệu", "Video", "Giờ", "Ngôn ngữ", "Dùng cho"],
    [["L25 (kênh ôn thi THPT)", "10", "4,2", "vi", "Chỉnh, test tách slide & lọc quảng cáo"],
     ["vn-lecture-test12", "12", "7,9", "vi", "Test chính trên 11 kênh mới"],
     ["LPM [2]", "244", "117", "en", "Tách slide (có mốc slide)"],
     ["M3AV [4]", "51", "27", "en", "Chấm OCR công thức (LaTeX)"],
     ["SlideSpeech [3]", "141", "90", "en", "Train lọc nội dung, test ASR"]],
    "Bảng 1: Dữ liệu thực nghiệm (tổng 458 video, 246 giờ)") + '''<p class="note">Nhãn gán tay: 549 ranh giới slide, 2.068 giây quảng cáo trên L25; 26 câu hỏi có đáp án.</p>'''))
S.append(content(5, "5.2 Thang đo tách slide & lọc nội dung", metric(
    "F1",
    "Đánh giá hệ thống tìm đúng thời điểm chuyển slide và giữ đúng nội dung bài giảng.",
    "Ranh giới dự đoán đúng nếu cách ranh giới gán tay ≤ 1 giây. Lọc quảng cáo chấm theo từng giây.",
    "P = TP / (TP + FP) · R = TP / (TP + FN) · F1 = 2PR / (P + R)<br>Giữ bài giảng = số giây bài giảng còn trong chỉ mục / tổng số giây bài giảng",
    "F1 cao: slide được tách đúng; giữ bài giảng ≈ 100%: không mất kiến thức khi lọc.")))
S.append(content(5, "5.3 Thang đo OCR", metric(
    "CER",
    "Đo lỗi đọc chữ và công thức trên slide.",
    "So chữ OCR với chữ gán tay (M3AV: chữ + LaTeX đã kiểm tra) sau chuẩn hóa.",
    "CER = (S + D + I) / N<br>S, D, I: số ký tự thay thế, xóa, chèn; N: số ký tự của nhãn",
    "CER càng thấp càng tốt. Kèm tỉ lệ ký tự có dấu để đo khả năng giữ dấu tiếng Việt.")))
S.append(content(5, "5.4 Thang đo truy xuất", metric(
    "Hit@k",
    "Kiểm tra hệ thống có tìm được slide chứa đáp án trong top-k bằng chứng hay không.",
    "Mỗi câu hỏi có khoảng thời gian chứa đáp án (gán tay). Đúng nếu một trong k slide trả về giao với khoảng đó.",
    "Hit@k = (1/N) · Σ 𝟙[ slide đúng ∈ top-k ]",
    "Hit@1 cao: bằng chứng đầu tiên gửi cho LLM đã đúng chỗ.")))
S.append(content(5, "5.5 Thang đo câu trả lời & tín hiệu lời giảng", '''<div class="cols2">
  <div>''' + metric("Answer", "Đánh giá câu trả lời cuối của LLM.",
                     "Câu trả lời đúng nếu đủ ý chính so với nội dung bài giảng và không bịa; câu ngoài bài phải trả lời “không có trong bài”.",
                     "Accuracy = số câu trả lời đúng / tổng số câu hỏi", "Đo chất lượng toàn hệ thống từ góc nhìn sinh viên.") + '''</div>
  <div>''' + metric("AUC", "Đo tín hiệu “lời giảng có thuộc chủ đề bài không”.",
                     "Điểm lệch chủ đề (E5, chuẩn hóa trong video) cho từng cửa sổ 45 s; nhãn quảng cáo theo giây.",
                     "AUC = P( điểm cửa sổ quảng cáo &gt; điểm cửa sổ bài giảng )", "AUC = 1: tách hoàn hảo; 0,5: ngẫu nhiên.") + '''</div>
</div>''', "dense"))
# ---- 6 results
S.append(section(6))
S.append(content(6, "6.1 Phát hiện slide & lọc nội dung", table(
    ["Thành phần", "Tập đo", "Chỉ số", "Kết quả"],
    [["Phát hiện slide", "6 video test L25", "P / R / F1", "0,990 / 0,951 / <b>0,970</b>"],
     ["Lọc quảng cáo (đánh dấu)", "6 video test L25", "Loại quảng cáo · giữ bài giảng", "<b>99,8%</b> · 99,9%"],
     ["Lọc quảng cáo (cắt bỏ, cũ)", "12 video tiếng Anh không QC", "Thời lượng bị coi là QC", "<span class='bad'>47–100%</span>"],
     ["Lọc quảng cáo (đánh dấu + an toàn)", "12 video tiếng Anh không QC", "Giữ bài giảng", "<b>100%</b>"],
     ["Bộ gác OCR", "481 khung L25 · M3AV · LPM", "Giữ slide có chữ", "<b>100%</b>"]],
    "Bảng 2: Kết quả phát hiện slide và lọc nội dung")))
S.append(content(6, "6.2 OCR: mô hình đề xuất và GOT-OCR 2.0", table(
    ["Chỉ số", "Hybrid (đề xuất)", "GOT-OCR 2.0 [10]", "Nhận xét"],
    [["CER trên slide toán M3AV ↓", "<b>0,36</b>", "0,52", "Hybrid đọc công thức tốt hơn"],
     ["Tỉ lệ ký tự có dấu (tiếng Việt) ↑", "<b>0,31</b>", "0,02 – 0,14", "GOT mất dấu tiếng Việt"],
     ["Thời gian / slide", "7,6 s", "5,5 s", "GOT nhanh hơn ~1,4 lần"],
     ["Bộ nhớ GPU", "17 GB", "3 GB", "GOT nhẹ hơn"]],
    "Bảng 3: So sánh OCR (5 slide M3AV có nhãn, 12 keyframe L25)") + '<p class="note">→ Giữ OCR kết hợp vì độ chính xác tiếng Việt quan trọng hơn tốc độ.</p>'))
S.append(content(6, "6.3 Truy xuất & hỏi đáp", table(
    ["Video", "Ngôn ngữ", "Hit@1", "Hit@3", "Câu ngoài bài"],
    [["L25_V008 · Xác suất", "vi", "10/11", "11/11", "trả lời “không có trong bài” 2/2"],
     ["MLS-15DA1 · Đại số tuyến tính", "en", "8/11", "11/11", ""]],
    "Bảng 4: Kết quả truy xuất trên 22 câu hỏi có đáp án và 4 câu ngoài bài") + '''<div class="kpis">
  <div><b>0,94</b><span>AUC tín hiệu lời giảng tách quảng cáo (9 video L25)</span></div>
  <div><b>9/13</b><span>câu trả lời đúng với LLM nhỏ chạy trên máy (qwen2:7b)</span></div>
  <div><b>15–150 ms</b><span>thời gian tìm bằng chứng mỗi câu</span></div></div>'''))
S.append(content(6, "6.4 Thời gian xử lý", '''<div class="cols2"><div><h3>Video 30 phút · Kaggle T4 x2</h3>''' + "".join(
    f'<div class="bar"><span>{a}</span><div class="track"><i class="{c}" style="width:{w}%"></i></div><b>{v}</b></div>'
    for a, w, c, v in [("01 Phát hiện slide", 16, "c1", "3,0 ph"), ("02 OCR", 100, "c2", "18,3 ph"),
                       ("03 STT (song song)", 16, "c3", "3,0 ph"), ("04 + 05", 1, "c1", "0,1 ph")]) + '''</div>
  <div class="panel"><h3>Nhận xét</h3><ul>
    <li>Tổng ≈ <b>21 phút</b> cho 30 phút video; OCR chiếm ~85%.</li>
    <li>Thử fp16 và bỏ vùng trang trí lặp: không nhanh hơn đáng kể.</li>
    <li>Chi phí: 0 USD trên Kaggle; ~0,13–0,21 USD/video trên GPU thuê.</li></ul></div></div>'''))
S.append(content(6, "6.5 Tổng hợp kết quả", table(
    ["Thành phần", "Chỉ số chính", "Kết quả", "Ý nghĩa"],
    [["Phát hiện slide", "F1", "0,970", "Tách đúng slide và thời gian"],
     ["Lọc nội dung", "Giữ bài giảng · loại QC", "100% · 99,8%", "Không mất kiến thức, vẫn loại quảng cáo"],
     ["OCR", "CER M3AV", "0,36", "Đọc tốt chữ, công thức, giữ dấu"],
     ["Truy xuất", "Hit@3", "22/22", "Luôn có slide đúng trong top 3"],
     ["Hỏi đáp", "Accuracy", "9/13", "Giới hạn do LLM nhỏ, cần LLM mạnh hơn"]],
    "Bảng 5: Tổng hợp kết quả hệ thống")))
S.append(content(6, "6.6 Chạy thử hệ thống", '''<div class="demo">
  <div class="shot">[Chèn ảnh chụp web app: video + danh sách slide + khung hỏi AI]</div>
  <div class="panel"><h3>Ví dụ</h3><p><b>Hỏi:</b> “Xác suất của một biến cố được tính như thế nào?”</p>
  <p><b>Trả lời:</b> P(A) = n(A) / n(Ω), trong đó n(A) là số kết quả thuận lợi, n(Ω) là số kết quả có thể của phép thử.</p>
  <p><b>Nguồn:</b> slide 01:38 – 02:10 (bấm để tua tới).</p></div></div>'''))
# ---- 7 conclusion
S.append(section(7))
S.append(content(7, "7.1 Tóm tắt nội dung", '''<ul class="lead">
  <li><b>Bài toán:</b> hỏi đáp trên video bài giảng đa phương thức, ưu tiên tiếng Việt.</li>
  <li><b>Giải pháp:</b> pipeline 5 bước (slide → OCR → STT → đồng bộ → chỉ mục) + truy xuất lai + LLM.</li>
  <li><b>Thực nghiệm:</b> 458 video, 246 giờ; bộ nhãn gán tay cho slide, quảng cáo, câu hỏi.</li>
  <li><b>Kết quả:</b> F1 tách slide 0,970; giữ 100% bài giảng trên kênh mới; Hit@3 22/22.</li></ul>'''))
S.append(content(7, "7.2 Đóng góp", '''<ol class="lead">
  <li>Pipeline <b>Multimodal Lecture RAG</b> hoàn chỉnh cho video bài giảng tiếng Việt, chạy trên GPU miễn phí.</li>
  <li>Cơ chế <b>lọc nội dung an toàn</b> (đánh dấu + ngưỡng an toàn + bộ gác OCR) tổng quát sang kênh chưa từng thấy.</li>
  <li>Đồng bộ slide ↔ lời giảng và <b>sửa lỗi ASR bằng chữ trên slide</b>.</li>
  <li>Bộ đánh giá: nhãn ranh giới slide, quảng cáo theo giây, câu hỏi có đáp án.</li></ol>'''))
S.append(content(7, "7.3 Lộ trình phát triển", '''<div class="road">''' + "".join(
    f'<div class="step {cls}"><b>{n}</b><span>{t}</span><i>{s}</i></div>'
    for n, t, s, cls in [(1, "Dữ liệu & nhãn", "458 video, gold set 1", "done"), (2, "Pipeline trích xuất", "slide · OCR · STT · đồng bộ", "done"),
                         (3, "RAG + web app", "truy xuất lai, LLM", "done"), (4, "Lọc nội dung đa kênh", "Phase 0–2 xong", "now"),
                         (5, "Đánh giá kênh mới", "gold set 2 · 12 video", "next"), (6, "LLM mạnh hơn & triển khai", "Gemini · tối ưu OCR", "next")]) + '''</div>
<div class="key"><span class="k done">đã xong</span><span class="k now">đang làm</span><span class="k next">tiếp theo</span></div>'''))
S.append(content(7, "7.4 Hướng phát triển", '''<ul class="lead">
  <li><b>Lớp quyết định</b> kết hợp điểm quảng cáo, lời có thuộc bài, khung có nội dung.</li>
  <li><b>LLM mạnh hơn</b> (Gemini) và trả lời theo ngôn ngữ câu hỏi.</li>
  <li><b>Gold set 2:</b> đo CER, WER, F1 trên 12 video tiếng Việt của 11 kênh mới.</li>
  <li><b>Tăng tốc OCR</b> bằng cách giảm độ phân giải đầu vào; mở rộng cho slide cuộn.</li></ul>'''))
# ---- 8 references
S.append(section(8))
S.append(content(8, "Tài liệu tham khảo", '''<ol class="refs">
  <li>P. Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” <i>NeurIPS</i>, 2020.</li>
  <li>D. W. Lee, C. Ahuja, P. P. Liang, S. Natu, L.-P. Morency, “Lecture Presentations Multimodal Dataset: Towards Understanding Multimodality in Educational Videos,” <i>ICCV</i>, 2023.</li>
  <li>H. Wang et al., “SlideSpeech: A Large Scale Slide-Enriched Audio-Visual Corpus,” <i>ICASSP</i>, 2024.</li>
  <li>Z. Chen et al., “M3AV: A Multimodal, Multigenre, and Multipurpose Audio-Visual Academic Lecture Dataset,” <i>ACL</i>, 2024.</li>
  <li>A. Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision,” <i>ICML</i>, 2023.</li>
  <li>X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer, “Sigmoid Loss for Language Image Pre-Training,” <i>ICCV</i>, 2023.</li>
  <li>L. Wang et al., “Multilingual E5 Text Embeddings: A Technical Report,” <i>arXiv:2402.05672</i>, 2024.</li>
  <li>G. V. Cormack, C. L. A. Clarke, S. Büttcher, “Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods,” <i>SIGIR</i>, 2009.</li>
  <li>Qwen Team, “Qwen3-VL Technical Report,” <i>arXiv</i>, 2025.</li>
  <li>H. Wei et al., “General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model,” <i>arXiv:2409.01704</i>, 2024.</li>
  <li>S. Robertson, H. Zaragoza, “The Probabilistic Relevance Framework: BM25 and Beyond,” <i>Foundations and Trends in IR</i>, 2009.</li></ol>''', "dense"))
S.append(f'''<section class="slide end-slide">
  <div class="logo-white end-logo"></div>
  <h2 class="end-title">Thank you<br>for listening</h2>
  <p class="end-sub">Cảm ơn thầy cô đã lắng nghe · Hỏi & đáp</p>
</section>''')

CSS = f'''
:root{{--blue:#153898;--navy:#0e1f5c;--gold:#f6b21b;--ink:#1d2433;--sub:#5a6a8a;--soft:#eef1fa}}
*{{box-sizing:border-box}}
html,body{{margin:0;height:100%;background:#0b0f2a;font-family:"Be Vietnam Pro",Segoe UI,Arial,sans-serif;color:var(--ink)}}
.deck{{position:fixed;inset:0;display:grid;place-items:center}}
.slide{{position:absolute;width:1280px;height:720px;overflow:hidden;background:#fff;display:none;transform-origin:center}}
.slide.active{{display:block}}
.title-slide{{background:url({BG_TITLE}) center/cover;color:#fff}}
.logo-white{{position:absolute;left:3.6%;top:4.9%;width:18.5%;aspect-ratio:10281/2930;background:url({LOGO_WHITE}) left top/contain no-repeat}}
.logo-white.small{{left:5%;top:2%;width:15%}}
.cover-top{{position:absolute;left:24%;top:4%;width:52%;text-align:center;font-size:19px;font-weight:700;line-height:1.5}}
.cover-major{{position:absolute;right:2.5%;top:3%;text-align:right;font-size:13px;font-weight:700;color:var(--gold);line-height:1.5}}
.title-box{{position:absolute;left:8%;top:21%;width:84%;text-align:center}}
.kicker{{font-size:24px;font-weight:700;letter-spacing:.06em;margin:0 0 10px;color:#dfe6ff}}
.title-slide h1{{font-size:64px;line-height:1.1;margin:0 0 12px;font-weight:800}}
.subtitle{{font-size:26px;margin:0 0 42px;color:#dfe6ff}}
.people{{display:flex;justify-content:center;gap:70px;text-align:left;font-size:20px;line-height:1.7}}
.people span{{display:block;color:var(--gold);font-weight:700;font-size:17px}}
.people b{{margin-right:10px}}
.cover-date{{position:absolute;bottom:11%;width:100%;text-align:center;font-size:19px;color:#dfe6ff}}
.content{{background:url({BG_CONTENT}) center/cover}}
.logo{{position:absolute;left:3.2%;top:3.1%;width:18.8%;aspect-ratio:10281/2930;background:url({LOGO}) left top/contain no-repeat}}
.divider{{position:absolute;left:63.7%;top:9%;width:29.4%;aspect-ratio:4167/285;background:url({DIVIDER}) left top/contain no-repeat}}
.crumb{{position:absolute;left:63.7%;top:4%;width:29.4%;text-align:right;font-size:14px;font-weight:700;color:var(--blue)}}
.crumb.light{{left:auto;right:5%;top:3%;color:#dfe6ff}}
.content h2{{position:absolute;left:6.9%;top:15.3%;width:86.2%;margin:0;font-size:36px;font-weight:800;color:var(--navy)}}
.content .body{{position:absolute;left:6.9%;top:26%;width:86.2%;height:58%;font-size:20px;line-height:1.5}}
.content.dense .body{{font-size:17px;top:24.5%}}
.names{{position:absolute;left:2.5%;bottom:3.2%;color:#fff;font-size:13px;font-weight:700;letter-spacing:.03em;z-index:2}}
.sec-num{{position:absolute;left:6.8%;top:22%;font-size:110px;font-weight:800;color:var(--gold);line-height:1}}
.section .sec-title{{top:44%;font-size:58px;color:var(--blue)}}
h3{{font-size:21px;margin:0 0 8px;color:var(--blue)}}
ul,ol{{margin:0;padding-left:26px}} li{{margin:6px 0}}
.lead li{{margin:14px 0;font-size:22px}}
.agenda{{list-style:none;padding:0;display:grid;grid-template-columns:1fr 1fr;gap:14px 50px;margin-top:6px}}
.agenda li{{display:flex;align-items:center;gap:16px;font-size:23px;font-weight:600;margin:0;padding:10px 16px;border-radius:12px;background:var(--soft)}}
.agenda span{{display:inline-grid;place-items:center;width:48px;height:48px;border-radius:50%;background:var(--blue);color:#fff;font-weight:800;flex:none}}
.cols2{{display:grid;grid-template-columns:1fr 1fr;gap:34px}}
.panel{{align-self:start;background:var(--soft);border-left:6px solid var(--blue);border-radius:14px;padding:16px 22px}}
.panel.gold{{border-left-color:var(--gold);background:#fff8e6}}
.flow{{margin-top:22px;text-align:center;font-weight:700;color:#fff;background:var(--blue);border-radius:999px;padding:10px 20px;font-size:19px}}
table{{width:100%;border-collapse:collapse;font-size:17px}}
th{{background:var(--blue);color:#fff;text-align:left;padding:9px 12px;font-weight:700}}
td{{padding:8px 12px;border-bottom:1px solid #d8deee;vertical-align:top}}
tr:nth-child(even) td{{background:#f6f8fd}}
.tcap{{text-align:center;font-style:italic;color:var(--sub);font-size:15px;margin:8px 0 0}}
.bad{{color:#c4553f;font-weight:700}}
.metric p{{margin:6px 0}}
.formula{{margin:12px 0;padding:14px 18px;background:#fff8e6;border:2px solid var(--gold);border-radius:12px;font-family:"Cambria Math",Cambria,serif;font-size:21px;text-align:center}}
.dense .formula{{font-size:17px;padding:10px}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:22px}}
.kpis div{{background:var(--blue);color:#fff;border-radius:14px;padding:14px 18px;border-bottom:6px solid var(--gold)}}
.kpis b{{display:block;font-size:34px;font-weight:800}} .kpis span{{font-size:15px;color:#dfe6ff}}
.bar{{display:grid;grid-template-columns:180px 1fr 80px;gap:12px;align-items:center;font-size:17px;margin:12px 0}}
.track{{height:22px;background:#e6eaf4;border-radius:11px;overflow:hidden}}
.track i{{display:block;height:100%;border-radius:11px}}
.c1{{background:var(--blue)}} .c2{{background:var(--gold)}} .c3{{background:#6f8fd0}}
.bar b{{text-align:right}}
.note{{font-size:16px;color:var(--sub);margin-top:12px}}
.small li{{font-size:17px;margin:6px 0}}
.demo{{display:grid;grid-template-columns:1.25fr 1fr;gap:28px}}
.shot{{height:330px;border:3px dashed #9fb0d8;border-radius:14px;display:grid;place-items:center;text-align:center;color:var(--sub);padding:20px;background:#f6f8fd}}
.road{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-top:20px}}
.step{{border-radius:14px;padding:16px 12px;text-align:center;border:2px solid #d8deee;position:relative}}
.step b{{display:inline-grid;place-items:center;width:44px;height:44px;border-radius:50%;background:#c9d3ea;color:var(--navy);font-size:20px;margin-bottom:8px}}
.step span{{display:block;font-weight:700;font-size:17px;color:var(--navy)}} .step i{{display:block;font-style:normal;font-size:14px;color:var(--sub);margin-top:6px}}
.step.done b{{background:var(--blue);color:#fff}} .step.now{{border-color:var(--gold);background:#fff8e6}} .step.now b{{background:var(--gold);color:var(--navy)}}
.key{{display:flex;gap:18px;justify-content:center;margin-top:18px;font-size:15px}}
.k::before{{content:"";display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:6px;vertical-align:-2px}}
.k.done::before{{background:var(--blue)}} .k.now::before{{background:var(--gold)}} .k.next::before{{background:#c9d3ea}}
.refs{{font-size:15.5px;line-height:1.45}} .refs li{{margin:5px 0}}
.card-slide{{background:url({BG_CARD}) center/cover}}
.card-title{{position:absolute;left:6.9%;top:12%;margin:0;font-size:34px;font-weight:800;color:var(--navy);z-index:2}}
.card{{position:absolute;left:5%;top:23%;width:90%;height:70%;padding:6px 22px}}
.card svg{{width:100%;height:auto}}
.end-slide{{background:url({BG_END}) center/cover;color:#fff}}
.end-logo{{width:27.1%}}
.end-title{{position:absolute;left:6.9%;top:30%;margin:0;font-size:66px;line-height:1.1;font-weight:800}}
.end-sub{{position:absolute;left:6.9%;top:60%;font-size:26px;color:var(--gold);font-weight:700}}
.nav{{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);display:flex;gap:10px;align-items:center;
  background:rgba(14,31,92,.85);color:#fff;border-radius:999px;padding:6px 14px;font-size:14px;z-index:9}}
.nav button{{background:none;border:0;color:#fff;font-size:20px;cursor:pointer;padding:2px 8px}}
.pageno{{position:absolute;right:3%;bottom:2.5%;color:#fff;font-size:15px;font-weight:700}}
.title-slide .pageno,.end-slide .pageno{{display:none}}
@media print{{.nav{{display:none}} .slide{{display:block;position:relative;transform:none!important;page-break-after:always}} .deck{{position:static;display:block}}}}
'''

html = f'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multimodal Lecture RAG</title>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<div class="deck">
{chr(10).join(S)}
</div>
<div class="nav"><button id="prev" aria-label="Slide trước">‹</button><span id="count"></span><button id="next" aria-label="Slide sau">›</button></div>
<script>
const slides=[...document.querySelectorAll('.slide')];let i=0;
slides.forEach((s,n)=>{{const p=document.createElement('div');p.className='pageno';p.textContent=n+1;s.appendChild(p);}});
function fit(){{const k=Math.min(innerWidth/1280,innerHeight/720);slides.forEach(s=>s.style.transform='scale('+k+')');}}
function show(n){{i=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,k)=>s.classList.toggle('active',k===i));
  document.getElementById('count').textContent=(i+1)+' / '+slides.length;try{{history.replaceState(null,'','#'+(i+1))}}catch(e){{}}}}
addEventListener('keydown',e=>{{if(['ArrowRight','PageDown',' '].includes(e.key))show(i+1);if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1);if(e.key==='f')document.documentElement.requestFullscreen?.();}});
document.getElementById('prev').onclick=()=>show(i-1);document.getElementById('next').onclick=()=>show(i+1);
addEventListener('resize',fit);fit();show((parseInt(location.hash.slice(1))||1)-1);
</script></body></html>'''
OUT.write_text(html, encoding="utf-8")
print(OUT, round(len(html) / 1e6, 2), "MB", len(S), "slides")
