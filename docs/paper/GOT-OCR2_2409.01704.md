# General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model (GOT, 2024, arXiv:2409.01704)

Tác giả: Haoran Wei, Chenglong Liu, Jinyue Chen và cộng sự (StepFun, Megvii, UCAS, Tsinghua).

## Tóm tắt
- **Bài toán:** OCR-1.0 là chuỗi nhiều module (phát hiện → cắt → nhận dạng), dễ kẹt ở tối ưu cục bộ và tốn công bảo trì. LVLM lớn thì tốn token ảnh và chi phí triển khai cao. Bài báo đề xuất "OCR-2.0": một mô hình end-to-end đọc mọi loại "ký tự" nhân tạo, gồm chữ, công thức toán và hóa học, bảng, biểu đồ, bản nhạc, hình học.
- **Phương pháp:** GOT có 580M tham số, gồm encoder nén mạnh (ảnh 1024×1024 thành 256 token) và decoder Qwen-0.5B có ngữ cảnh tối đa 8K token. Huấn luyện 3 giai đoạn:
  1. Pretrain encoder với OPT-125M trên khoảng 5M cặp ảnh–chữ.
  2. Huấn luyện chung với dữ liệu tổng hợp (Mathpix markdown, TikZ, SMILES, Kern...).
  3. Post-train chỉ phần decoder để thêm OCR theo vùng (chỉ định bằng tọa độ hoặc màu), độ phân giải động (multi-crop) và OCR nhiều trang.
  
  Đầu ra có thể là văn bản thuần hoặc có định dạng, chọn bằng prompt. Ngôn ngữ chính là tiếng Anh và tiếng Trung.
- **Kết quả:**
  - Tài liệu dày chữ (Fox): edit distance 0,035 (en) / 0,038 (zh), tốt hơn Fox-1.8B (0,046/0,061), Qwen-VL-Max (0,057/0,091) và Nougat (0,255).
  - Chữ trong ảnh tự nhiên (400 ảnh): 0,112/0,096.
  - Markdown có định dạng: tổng chữ đạt 0,097 khi dùng một tỉ lệ ảnh và 0,086 khi multi-crop. Công thức cải thiện từ 0,269 xuống 0,159 khi multi-crop, bảng từ 0,254 xuống 0,220.
  - OCR theo vùng: tốt hơn Fox, ví dụ tiếng Anh theo vùng đạt 0,041 so với 0,059.
  - Biểu đồ: ChartQA-SE AP@strict 0,747, cao hơn GPT-4V (0,504) và Qwen-VL (0,586).

## Điểm mạnh
- Nhỏ (0,58B) mà vượt nhiều LVLM 7B–72B về OCR thuần, nên rẻ khi triển khai và nhanh hơn VLM lớn.
- Một mô hình dùng chung cho chữ, công thức, bảng và biểu đồ. Có OCR theo vùng chỉ định bằng box, rất hợp với pipeline cắt vùng.
- Multi-crop giúp rõ rệt với chữ nhỏ trong công thức và bảng.

## Điểm yếu / hạn chế
- Chủ yếu hỗ trợ tiếng Anh và tiếng Trung. Bài báo không đánh giá tiếng Việt và tự nhận còn cần mở rộng thêm ngôn ngữ.
- Chỉ nhận dạng, không suy luận hay mô tả sơ đồ thành lời.
- Nhiều benchmark do nhóm tác giả tự dựng và quy mô nhỏ (90 trang markdown, 100 bản nhạc, 180 hình học). Bài báo không báo cáo tốc độ.

## So sánh với đề tài của tui
- **Giống:** cùng hướng OCR end-to-end bằng mô hình sinh, cùng tách vùng công thức/bảng và xuất Markdown/LaTeX.
- **Khác:** đề tài dùng hybrid hai mô hình (Unlimited-OCR + Qwen3-VL), có cổng SigLIP lọc khung hình và sinh mô tả sơ đồ. GOT chỉ là một bộ OCR thuần.
- **Đề tài hơn:** mô tả được ngữ nghĩa sơ đồ để phục vụ RAG, hỗ trợ tiếng Việt qua Qwen3-VL.
- **Đề tài kém:** nặng và chậm hơn (khoảng 35 s/slide). Chưa tận dụng OCR theo vùng bằng prompt tọa độ trên một mô hình nhỏ.

## Có áp dụng được vào đề tài không?
- **Bước 02, nhánh công thức/bảng/biểu đồ (Cao):** thử GOT chỉ cho các vùng công thức, bảng và biểu đồ (ít dấu tiếng Việt), dùng chế độ `OCR with format` kèm box. Giữ Qwen3-VL cho chữ tiếng Việt và mô tả sơ đồ. Đo tốc độ và edit distance trên block công thức của M3AV. Đây là cách trực tiếp nhất để giảm 35 s/slide.
- **Bước 02, multi-crop (Trung bình):** áp dụng chia ảnh thành nhiều phần cho slide có chữ nhỏ trước khi OCR. Có thể giúp giảm lỗi dấu do độ phân giải thấp.
- **Đánh giá (Trung bình):** báo cáo edit distance, F1, BLEU, METEOR theo cùng giao thức của GOT để dễ so sánh với tài liệu.
- **Thay toàn bộ OCR bằng GOT (Thấp):** không nên khi chưa kiểm chứng được khả năng đọc dấu tiếng Việt.
