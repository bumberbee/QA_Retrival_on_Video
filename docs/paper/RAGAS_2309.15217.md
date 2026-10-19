# Ragas: Automated Evaluation of Retrieval Augmented Generation (EACL 2024 System Demonstrations, arXiv:2309.15217)

## Tóm tắt
- **Bài toán:** đánh giá hệ RAG khó vì có nhiều khía cạnh (truy hồi có đúng và gọn không, LLM có bám ngữ cảnh không, câu trả lời có tốt không) và thường thiếu đáp án chuẩn do người viết. Perplexity không phản ánh hiệu năng thực tế, còn QA trích xuất ngắn không đại diện cho cách dùng thật.
- **Phương pháp:** Ragas đề xuất ba chỉ số **không cần đáp án tham chiếu**, tính tự động bằng cách gọi LLM (gpt-3.5-turbo-16k trong thí nghiệm):
  - **Faithfulness:** LLM tách câu trả lời thành các mệnh đề ngắn, rồi kiểm từng mệnh đề có suy ra được từ ngữ cảnh không; F = số mệnh đề được hỗ trợ / tổng số mệnh đề.
  - **Answer relevance:** LLM sinh n câu hỏi từ câu trả lời, tính cosine (embedding text-embedding-ada-002) với câu hỏi gốc rồi lấy trung bình. Phạt câu trả lời thiếu hoặc thừa, không xét tính đúng.
  - **Context relevance:** LLM trích các câu trong ngữ cảnh thực sự cần để trả lời; CR = số câu trích được / tổng số câu ngữ cảnh.
- **Dữ liệu:** bộ **WikiEval** tự xây: 50 trang Wikipedia về sự kiện từ năm 2022 (sau mốc dữ liệu của mô hình), câu hỏi do ChatGPT sinh, người chú thích so sánh cặp câu trả lời/ngữ cảnh theo từng khía cạnh.
- **Kết quả chính:** độ đồng thuận với người chú thích (accuracy so sánh cặp) của Ragas là 0.95 (faithfulness), 0.78 (answer relevance), 0.70 (context relevance), cao hơn hẳn baseline GPT Score (0.72/0.52/0.63) và GPT Ranking (0.54/0.40/0.52).

## Điểm mạnh
- Không cần đáp án chuẩn, đánh giá nhanh, phù hợp vòng lặp phát triển.
- Tách riêng lỗi truy hồi (context relevance) và lỗi sinh (faithfulness, answer relevance), dễ chẩn đoán.
- Có thư viện mã nguồn mở, tích hợp llama-index và LangChain.

## Điểm yếu / hạn chế
- Phụ thuộc chất lượng LLM giám khảo và API trả phí của OpenAI; kết quả có thể thay đổi theo mô hình và prompt.
- WikiEval nhỏ (50 trang), tiếng Anh, câu hỏi do ChatGPT sinh.
- Context relevance kém đồng thuận nhất (0.70); answer relevance không đo tính đúng sự thật.

## So sánh với đề tài của tui
- **Giống:** cùng mục tiêu giảm ảo giác, trả lời có căn cứ trên ngữ cảnh truy hồi.
- **Khác:** Ragas là công cụ đánh giá, không phải hệ thống; đề tài hiện mới đánh giá các module đầu (ranh giới slide F1 0.970, OCR, ASR) chứ chưa đánh giá câu trả lời cuối.
- **Đề tài hơn:** có gold do người gán nhãn cho các bước tiền xử lý, điều Ragas không bao phủ.
- **Đề tài kém:** thiếu đúng phần Ragas giải quyết: chưa đo faithfulness hay độ liên quan của câu trả lời, nên chưa định lượng được ảo giác và lỗi trộn ngôn ngữ của qwen2:7b.

## Có áp dụng được vào đề tài không?
- **Đánh giá end-to-end (Cao):** chạy Ragas (faithfulness, answer relevance, context precision/relevance) trên tập câu hỏi của 10 video L25 có nhãn và vn12. Dùng LLM giám khảo mạnh hơn bộ sinh, và embedding đa ngôn ngữ (multilingual-e5) thay ada-002 cho tiếng Việt. Kiểm tra lại độ đồng thuận với người trên một mẫu nhỏ như cách làm với WikiEval.
- **Module 05 – so sánh cấu hình (Cao):** dùng điểm Ragas để so các phương án: chỉ e5 với e5 + BM25, có/không mở rộng slide lân cận, kích thước chunk ngắn/dài, prompt ép trả lời tiếng Việt.
- **Ứng dụng web (Trung bình):** dùng faithfulness làm bộ lọc trực tuyến: khi điểm thấp thì cảnh báo hoặc yêu cầu LLM trả lời lại chỉ dựa trên ngữ cảnh.
