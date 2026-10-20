# Learning Engagement Assistant (LEA): Cross-Course Scalability and Classroom Evaluation of an Agentic AI Tutoring System (bản mở rộng của bài ICAART 2026, arXiv 2607.13370)

## Tóm tắt
- **Bài toán:** Các gia sư AI dựa trên LLM đang được đưa vào lớp học nhanh hơn tốc độ có bằng chứng thực nghiệm. Bài ICAART 2026 trước đó chỉ kiểm chứng LEA bằng mô phỏng (learner tổng hợp) trên một môn học. Bài này kiểm tra hai điều: (RQ1) sinh viên thật trải nghiệm LEA ra sao và khác mô phỏng ở đâu; (RQ2) kiến trúc có dùng được cho nhiều môn mà không phải sửa hay không.
- **Phương pháp:** LEA là hệ thống agentic có 3 chế độ: Chat, Tutor (hỏi gợi mở kiểu Socratic) và Quiz. Nó kết hợp RAG theo từng môn (ChromaDB, chunk 500 token, overlap 100, cắt theo ranh giới câu bằng NLTK, embedding OpenAI text-embedding-3-small; video được Whisper phiên âm) với mô hình Knowledge Component (KC) gồm Learning Objective (LO) và Granular Objective (GO). Agent Orchestrator ước lượng Cognitive Load bằng công thức tuyến tính, ZPD bằng độ chính xác trung bình của 4 câu trả lời gần nhất (nhắm vùng 50–80%) và trạng thái động lực, rồi chọn mức scaffolding phù hợp. Muốn thêm môn mới chỉ cần nạp thư viện RAG và mô hình KC, có giáo viên duyệt lại (human-in-the-loop).
- **Dữ liệu:** 3 môn ở Abertay University: CMP511 (ML, thạc sĩ; 1.266 chunk), CMP202 (Cấu trúc dữ liệu và giải thuật, đại học; 1.230 chunk) và PSY555 (Tâm lý học; 691 chunk). Khảo sát pilot n=8 trên tổng 41 sinh viên CMP511. Đánh giá RAGAS trên 660 câu hỏi sinh từ mô hình KC.
- **Kết quả:** Mô phỏng cũ cho Chat retrieval success 100% và chất lượng câu trả lời 0,880. Trong pilot, 8/8 sinh viên đánh giá Chat chính xác và sát môn học, nhưng chỉ 3/8 thấy Tutor thích ứng tốt khi họ nói mình chưa hiểu. Một số người thấy lời khen động viên "giả tạo", và có lỗi Quiz lặp câu hỏi. Kết quả RAGAS theo thứ tự CMP511/CMP202/PSY555:
  - Faithfulness: 0,685 / 0,609 / 0,499
  - Answer Relevancy: 0,933 / 0,935 / 0,875
  - Context Precision: 0,901 / 0,881 / 0,877
  - Context Recall: 0,609 / 0,404 / 0,556

  Recall của CMP202 thấp, có thể do chunk cắt theo câu làm vỡ các khối code.

## Điểm mạnh
- Tách lớp orchestration (không phụ thuộc môn học) khỏi lớp kiến thức (RAG + KC), và đã thử thật trên 3 môn.
- Dùng RAGAS với kiểm định phi tham số (Kruskal–Wallis, Mann–Whitney với hiệu chỉnh Holm, bootstrap CI). Thống kê làm cẩn thận.
- So sánh thẳng thắn giữa mô phỏng và thực tế, và nêu rõ các lỗi mà mô phỏng không thể phát hiện.
- Có ý tưởng hay: dùng câu hỏi sinh từ KC để kiểm tra kho tài liệu có phủ hết giáo trình hay không.

## Điểm yếu / hạn chế
- Pilot chỉ có n=8 (tỉ lệ phản hồi 20%), ít phiên dùng, không có thống kê suy luận.
- Context Precision/Recall dùng chính câu trả lời của hệ thống làm tham chiếu vì không có ground truth do người viết.
- Giả thuyết về chunking chưa được kiểm chứng bằng ablation. Các hệ số của công thức Cognitive Load được đặt bằng mô phỏng Monte Carlo.
- Phụ thuộc API OpenAI và chỉ làm với tài liệu tiếng Anh.

## So sánh với đề tài của tui
- **Giống:** cả hai đều dùng RAG theo khóa học và Whisper phiên âm bài giảng.
- **Khác:**
  - LEA xử lý video chỉ bằng transcript, không đọc slide từ video. Đề tài có detect slide (F1 0,970), OCR lai và căn transcript theo slide, nên mạnh hơn hẳn ở phần dữ liệu đa phương thức.
  - LEA có mô hình người học (KC, mastery, ZPD) cùng chế độ Tutor/Quiz. Đề tài chỉ có hỏi đáp.
  - LEA đã có đánh giá RAGAS và khảo sát người dùng thật. Đề tài chưa có hai thứ này.
- **Đề tài hơn:** pipeline video → slide → OCR, retrieval lai (e5 + BM25) cùng mở rộng sang slide lân cận, chạy LLM cục bộ, hỗ trợ tiếng Việt.
- **Đề tài kém:** phần đánh giá câu trả lời và đánh giá với người dùng, cùng các tính năng sư phạm.

## Có áp dụng được vào đề tài không?
- **Đánh giá (Cao):** chạy RAGAS (Faithfulness, Answer Relevancy, Context Precision/Recall) trên bộ câu hỏi L25/vn12, báo cáo bootstrap CI giống Bảng 6. Nên có tham chiếu do người viết để tránh lỗi tự tham chiếu mà LEA gặp.
- **User study (Cao):** dùng bảng hỏi Likert 7 mức có câu đảo chiều, kèm phân tích chủ đề (thematic analysis). Mẫu nhỏ thì nói rõ đây là pilot.
- **Module 05 – Chunking (Trung bình):** kiểm tra đoạn quá ngắn/quá dài và tránh cắt vỡ công thức, bảng, code trong OCR slide. Bài cho thấy chunking ảnh hưởng mạnh đến recall.
- **Kiểm tra độ phủ giáo trình (Thấp):** sinh câu hỏi từ đề cương môn học để tìm chủ đề kho bài giảng còn thiếu.
- **Chế độ Quiz/Tutor (Thấp):** nằm ngoài phạm vi chính, chỉ nên làm nếu còn thời gian.
