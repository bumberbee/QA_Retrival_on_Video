# M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding (UNC Chapel Hill & Bloomberg, 2024, arXiv:2411.04952)

## Tóm tắt
- **Bài toán:** DocVQA thực tế có hai khó khăn: (a) câu hỏi cần thông tin trải trên nhiều trang hoặc nhiều tài liệu, mô hình đa phương thức (MLM) không đọc nổi nhiều tài liệu dài; (b) thông tin quan trọng nằm trong hình, bảng, biểu đồ mà OCR/trích xuất văn bản bỏ qua.
- **Phương pháp:** M3DocRAG gồm ba bước: (1) nhúng mọi trang (dạng ảnh) bằng ColPali; (2) truy hồi top-K trang theo câu hỏi, có thể dùng chỉ mục xấp xỉ (FAISS IVFFlat, IVFPQ); (3) MLM (Qwen2-VL 7B) đọc các trang truy hồi được để trả lời. Hỗ trợ cả miền đóng (một tài liệu) lẫn miền mở (nhiều tài liệu), câu hỏi một bước và nhiều bước.
- **Dữ liệu:** benchmark mới **M3DocVQA** (2,000+ câu hỏi, 3,000+ PDF, 40,000+ trang, xây từ Wikipedia/MultimodalQA), cùng MMLongBench-Doc và MP-DocVQA. Tất cả chạy trên một GPU H100 80GB, tối đa 4 trang làm đầu vào.
- **Kết quả chính:** trên M3DocVQA (F1), M3DocRAG với 4 trang đạt 36.5 so với 23.7 của Text RAG (ColBERT v2 + Llama 3.1 8B); khác biệt lớn nhất ở bằng chứng dạng ảnh (24.7 so với 7.8). Trên MP-DocVQA đạt ANLS 0.8444 và R@1 truy hồi trang 81.05, là state-of-the-art, hơn Text RAG (0.5603) và các MLM như Arctic-TILT (0.8122). Về tốc độ, tìm kiếm chính xác FlatIP mất 21 s/câu hỏi, còn IVFFlat/IVFPQ giữ gần như nguyên độ chính xác với dưới 2 s/câu hỏi.

## Điểm mạnh
- Khung linh hoạt, thay được từng thành phần (retriever, MLM); xử lý được tình huống đa trang, đa tài liệu.
- Benchmark miền mở đầu tiên cho DocVQA đa phương thức.
- Phân tích thực tế về chỉ mục xấp xỉ và đánh đổi tốc độ/độ chính xác.

## Điểm yếu / hạn chế
- Tác giả thừa nhận retriever và MLM huấn luyện thiên tiếng Anh nên có thể kém với tài liệu không phải tiếng Anh.
- Số trang đưa vào MLM bị giới hạn bởi bộ nhớ GPU (4 trang); thêm trang không luôn tăng điểm.
- Mô hình có thể sinh thông tin sai, cần giám sát của con người. Không có kênh âm thanh.

## So sánh với đề tài của tui
- **Giống:** truy hồi mức trang/slide trên kho nhiều tài liệu (đề tài: 88 video L25), sau đó đưa ngữ cảnh cho mô hình sinh; đều quan tâm câu hỏi cần ghép nhiều trang (đề tài dùng mở rộng slide lân cận).
- **Khác:** đề tài truy hồi bằng văn bản (OCR + transcript, e5 + BM25) và sinh bằng LLM văn bản; M3DocRAG truy hồi và sinh trên ảnh.
- **Đề tài hơn:** có lời giảng và mốc thời gian; mở rộng ngữ cảnh theo thứ tự slide khai thác cấu trúc tuần tự của bài giảng, điều mà PDF rời rạc không có.
- **Đề tài kém:** thua rõ ở câu hỏi có bằng chứng trong hình (bài này cho thấy text RAG chỉ đạt 7.8 F1 với bằng chứng ảnh); chưa có benchmark câu hỏi đa slide/đa video.

## Có áp dụng được vào đề tài không?
- **Đánh giá (Cao):** xây tập câu hỏi theo phân loại của M3DocVQA: theo loại bằng chứng (chữ, bảng, hình, lời giảng) và số bước (một/nhiều slide), báo cáo EM/F1 riêng từng nhóm để thấy điểm yếu của OCR.
- **Module 05 – chỉ mục (Trung bình):** khi mở rộng toàn bộ 88 video hoặc thêm nhánh nhúng ảnh, dùng FAISS IVF/PQ để giữ độ trễ thấp.
- **Module 05 – sinh đa phương thức (Trung bình):** cấu hình "ColPali + Qwen2-VL 7B" là mẫu tham chiếu cho nhánh trả lời bằng ảnh slide, chạy được trên một GPU.
