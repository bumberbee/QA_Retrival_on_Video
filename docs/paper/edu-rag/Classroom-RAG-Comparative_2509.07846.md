# Aligning LLMs for the Classroom with Knowledge-Based Retrieval: A Comparative RAG Study (preprint nộp IEEE, 2025, arXiv:2509.07846)

## Tóm tắt
- **Bài toán:** với QA trong lớp học, nên chọn vector RAG hay graph RAG? Bài xét theo môn học, loại câu hỏi, cách bám giáo trình khi kiến thức thay đổi, và chi phí.
- **Phương pháp:** so sánh OpenAI Vector File Search với Microsoft GraphRAG (chế độ Local và Global). Cả index lẫn truy vấn đều dùng GPT-4.1-mini. Chấm bằng LLM-as-judge (GPT-4.1-nano), so từng cặp theo kiểu AB-BA để giảm thiên lệch vị trí, với 4 tiêu chí: Comprehensiveness, Directness, Faithfulness, Learnability.
- **Dữ liệu:**
  - **EduScopeQA (mới):** 3.176 câu hỏi trên 4 môn (Văn, Sử, Khoa học, Khoa học máy tính), khoảng 1,58 triệu từ. Câu hỏi chia 3 loại: Specific, Sectional, Thematic.
  - **KnowShiftQA:** 3.005 câu trắc nghiệm trên sách giáo khoa đã bị sửa có chủ đích một số dữ kiện, thử ở 3 phạm vi corpus: short (~315 từ), medium (~9,5K từ) và full (cả sách).
- **Kết quả:**
  - **Câu specific:** OpenAI RAG thắng ở Directness (ví dụ 0,780–0,875) và Faithfulness (ví dụ Văn 0,815).
  - **Câu sectional/thematic:** GraphRAG Global thắng rõ (Faithfulness Sử 0,851/0,868). Learnability Global đạt 0,72–0,99.
  - **KnowShiftQA:** với sách lớn (Sinh 258K từ, Sử, Địa), GraphRAG Local chính xác nhất. Với sách nhỏ (Hóa, Lý), OpenAI RAG ngang hoặc hơn.
  - **Chi phí (CS1):** OpenAI index 11,4 s và truy vấn 4,7 s. GraphRAG index khoảng 2.142 s với khoảng 4.025 lần gọi LLM, truy vấn 36,5 s (Local) và 70,1 s (Global).
  - **Bộ định tuyến:** một lời gọi LLM để định tuyến câu hỏi cho Faithfulness tổng thể cao nhất ở CS1.

## Điểm mạnh
- Phân loại câu hỏi theo phạm vi (cụ thể, theo chương, chủ đề) rất sát nhu cầu học tập.
- Đo cả chi phí và độ trễ, không chỉ đo chất lượng. Có công bố dataset.
- Cách chấm AB-BA giảm được thiên lệch của LLM-judge.

## Điểm yếu / hạn chế
- Phụ thuộc hoàn toàn vào API của OpenAI. Câu hỏi do GPT-4.1 sinh và cũng do GPT-4.1-nano chấm, nên có thể thiên lệch.
- Corpus là văn bản dài thuần chữ, không có slide/video. Chỉ tiếng Anh. Không có người dùng thật.
- Không thử hybrid hay BM25 dù có nhắc đến.

## So sánh với đề tài của tui
- **Giống:** cùng hướng đến QA bám học liệu, giảm ảo giác.
- **Khác:** đề tài dùng hybrid e5 + BM25 kèm mở rộng slide lân cận. Đây là dạng "local context" rẻ, gần với ý tưởng GraphRAG Local nhưng không cần xây đồ thị.
- **Đề tài hơn:** chạy local, rẻ, đa phương thức, tiếng Việt.
- **Đề tài kém:** chưa phân tích theo loại câu hỏi. Câu hỏi kiểu tổng quan cả bài giảng ("bài này nói về những gì?") có khả năng làm hybrid top-k trả lời kém.

## Có áp dụng được vào đề tài không?
- **Gắn nhãn Specific / Sectional / Thematic cho bộ câu hỏi đánh giá (Cao):** báo cáo kết quả theo từng loại. Việc này rẻ và làm luận văn thuyết phục hơn.
- **Đánh giá theo cặp AB-BA với LLM-judge (Trung bình):** dùng để so các biến thể, ví dụ có hoặc không có mở rộng lân cận, chỉ OCR so với aligned.
- **Định tuyến câu hỏi (Trung bình):**
  - Câu cụ thể: dùng retrieval hiện tại.
  - Câu tổng quan: dùng bản tóm tắt từng bài giảng hoặc chương, sinh sẵn khi index. Đây là một dạng "global summary" nhẹ, thay cho GraphRAG.
- **GraphRAG đầy đủ (Thấp):** quá tốn so với LLM local 7B và phần cứng hiện có.
