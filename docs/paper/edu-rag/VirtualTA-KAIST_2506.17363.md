# A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant (KAIST, 2025, arXiv 2506.17363)

## Tóm tắt
- **Bài toán:** Còn thiếu bằng chứng thực tế, ở quy mô lớn, về hiệu quả và mức chấp nhận của trợ giảng ảo (VTA) dùng LLM.
- **Phương pháp (kiến trúc VTA):**
  - Xây bằng LangChain (RAG), Streamlit (giao diện) và LangSmith (lưu log).
  - Tài liệu gồm PDF slide, notebook .ipynb và ghi âm bài giảng (Whisper-1). Cắt chunk 2.048 token (overlap 256), mỗi chunk có tiền tố ngày và tên bài. Embedding bằng text-embedding-3-large, lưu trong FAISS, cập nhật sau mỗi buổi học. Tổng cộng 59 tài liệu, 1.502 chunk.
  - gpt-4o-mini viết lại câu hỏi dựa trên lịch sử hội thoại (context-aware query), lấy top-5 tài liệu, rồi sinh câu trả lời. System prompt có thông tin hành chính của lớp và ngày giờ hiện tại.
  - Mã nguồn được công khai.
- **Dữ liệu:** môn nhập môn lập trình AI (PyTorch), học kỳ thu 2024, 14 tuần, 477 học viên sau đại học (472 đồng ý tham gia), dạy bằng tiếng Anh. Có 3 vòng khảo sát (trước, giữa, sau) và 3.869 cặp hỏi–đáp trong 916 cuộc hội thoại.
- **Kết quả:**
  - Chi phí khoảng 180 USD. Gần 50% sinh viên đã dùng ít nhất một lần.
  - Người chưa từng code dùng trung bình 62,2 lượt, so với 4,5 lượt ở nhóm nâng cao.
  - Số câu hỏi gửi VTA nhiều hơn khoảng 25 lần so với số câu hỏi gửi TA người năm trước (144 câu). Tỉ lệ câu hỏi lý thuyết ML tăng từ 8,3% lên 35,0%.
  - Trustworthiness tăng từ 3,27 lên 3,51 nhưng vẫn thấp hơn người (4,38). Helpfulness tổng thể giảm nhẹ từ 3,64 xuống 3,54, nhưng tăng có ý nghĩa ở nhóm dùng nhiều (p=0,043). Comfort tăng từ 0,58 lên 0,65.
  - 58% sinh viên từng ngại hỏi người.
  - Vấn đề hay gặp nhất là chậm (22 lượt báo, do không stream). Tiếp theo là không làm đúng chỉ dẫn (11), bịa hoặc sai (10) và truy xuất không ra (8). Tác giả gợi ý dùng hybrid BM25 + dense.

## Điểm mạnh
- Quy mô thật lớn, kéo dài cả học kỳ, có IRB, khảo sát theo chiều dọc và so sánh với TA người.
- Phân tích log tương tác chi tiết theo nhóm tần suất dùng, theo nền tảng kiến thức và theo loại câu hỏi.
- Rẻ, dễ tái lập, có mã nguồn.

## Điểm yếu / hạn chế
- Chỉ đo cảm nhận, không đo kết quả học tập hay độ chính xác câu trả lời bằng metric.
- Chỉ một môn lập trình, dạy bằng tiếng Anh. Retrieval chỉ dùng dense, chunk rất lớn.
- So sánh với năm trước không cùng điều kiện (khác nền tảng, khác cohort).
- Video chỉ dùng phần audio, không đọc slide trong video.

## So sánh với đề tài của tui
- **Giống:** RAG trên slide và ghi âm bài giảng (Whisper), web app cho sinh viên hỏi đáp.
- **Khác:** VTA dùng gpt-4o-mini trên cloud, chunk theo token. Đề tài dùng qwen2:7b cục bộ, đoạn theo slide và căn theo timestamp.
- **Đề tài hơn:**
  - Đã có hybrid e5 + BM25, đúng hướng mà bài này đề xuất cho tương lai.
  - OCR slide trích từ video và mở rộng slide lân cận.
  - Hỗ trợ tiếng Việt và có gold đo từng bước.
- **Đề tài kém:**
  - Chưa triển khai thật, chưa có log hay khảo sát.
  - Chưa xử lý hội thoại nhiều lượt (viết lại câu hỏi theo lịch sử).
  - Chưa stream câu trả lời.

## Có áp dụng được vào đề tài không?
- **Module truy vấn (Cao):** viết lại câu hỏi theo lịch sử hội thoại trước khi retrieve. Nên làm vì câu hỏi tiếp nối kiểu "còn cái kia thì sao?" rất phổ biến. Có thể dùng chính qwen2:7b với prompt ngắn.
- **Web app (Cao):** stream token. Bài cho thấy "chậm" là phàn nàn số 1, và với LLM cục bộ thì stream càng quan trọng. Ghi log đầy đủ (câu hỏi, chunk truy xuất, câu trả lời, thời gian) để phân tích sau.
- **User study (Cao):** khảo sát trước và sau theo 4 tiêu chí của bài (Helpfulness, Trustworthiness, Appropriateness, Comfortableness), so với TA người. Phân loại câu hỏi theo nhóm (lý thuyết, bài tập, hành chính). Quy mô có thể chỉ một lớp ở IUH.
- **Chunk metadata (Trung bình):** gắn tiền tố tên bài, buổi và ngày vào chunk để cải thiện truy xuất các câu hỏi có yếu tố thời gian.
