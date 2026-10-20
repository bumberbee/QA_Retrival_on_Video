# Enhancing classroom teaching with LLMs and RAG (SIGITE '24, poster/2 trang, arXiv:2411.04341)

## Tóm tắt
- **Bài toán:** liệu RAG trên tài liệu bên ngoài có giúp LLM trả lời câu hỏi cập nhật cho bối cảnh K-12 hay không. Thử với chủ đề mối đe dọa an ninh mạng.
- **Phương pháp:**
  - Cào 500 bài top cùng câu trả lời được upvote nhiều nhất từ các subreddit về cybersecurity, lưu vào Chroma.
  - LLM là Llama.
  - 30 câu hỏi kèm ground truth soạn từ thông tin CISA.
  - Chấm *answer correctness* bằng RAGAS.
  - Thử chunk size 250, 500, 1000, 2000, 4000 và 8000 ký tự.
- **Kết quả:**
  - Answer correctness trung bình ở mọi chunk size đều quanh 50% và không vượt 50%. Chunk 250 và 4000 cao nhất.
  - Chunk size hầu như không tạo khác biệt.
  - Kết luận: Reddit là nguồn nhiễu, không phù hợp. Phương pháp đánh giá nguồn tài liệu bằng RAGAS thì dùng được, kể cả để giáo viên kiểm tra học liệu.

## Điểm mạnh
- Quy trình đơn giản, có thể tái lập: RAG + RAGAS + quét chunk size.
- Gợi ý hay: dùng chính câu hỏi kiểm tra của giáo viên làm ground truth để đánh giá học liệu có "đủ" để trả lời hay không.

## Điểm yếu / hạn chế
- Chỉ 2 trang, chỉ 30 câu hỏi, không ghi rõ phiên bản Llama hay embedding.
- Không có số liệu chi tiết theo từng chunk size ngoài hình vẽ.
- Nguồn dữ liệu (Reddit) không phải học liệu thật nên kết luận về giáo dục còn suy diễn.
- Không có retrieval metric, không có người dùng thật.

## So sánh với đề tài của tui
- **Giống:** cùng RAG trên kho tài liệu cố định để trả lời người học, cùng quan tâm đến chunk size.
- **Khác:** đề tài dùng học liệu chuẩn (video bài giảng, slide), đa phương thức, tiếng Việt, hybrid retrieval. Bài này chỉ có text Reddit.
- **Đề tài hơn:** nguồn sạch, pipeline đầy đủ, có web app.
- **Đề tài kém:** chưa có đánh giá câu trả lời kiểu RAGAS. Đây chính là điểm yếu đã biết của đề tài, và bài này cho thấy đánh giá RAGAS là bước tối thiểu cần có.

## Có áp dụng được vào đề tài không?
- **Đánh giá RAGAS (Cao):** dựng bộ 50–100 câu hỏi kèm đáp án chuẩn cho 10 video L25 đã gán nhãn. Chạy RAGAS (answer correctness, faithfulness, context recall) với qwen2:7b. Có thể dùng một LLM mạnh hơn làm judge.
- **Quét chunk size / độ dài segment (Trung bình):** thử chunk theo 1 slide, theo slide ± lân cận, và theo cửa sổ token cố định. Đo answer correctness để xử lý vấn đề segment quá ngắn hoặc quá dài.
- **Câu hỏi kiểm tra của giảng viên làm ground truth (Trung bình):** xin đề kiểm tra hoặc câu hỏi ôn tập của môn tương ứng để có tập test thực tế.
