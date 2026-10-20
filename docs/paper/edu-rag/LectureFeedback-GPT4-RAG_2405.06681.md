# Leveraging Lecture Content for Improved Feedback: Explorations with GPT-4 and Retrieval Augmented Generation (IEEE 2024, arXiv 2405.06681)

## Tóm tắt
- **Bài toán:** Trên nền tảng luyện lập trình Tutor Kai (University of Siegen), sinh viên xin phản hồi về lời giải của mình. Nhóm tác giả muốn phản hồi do GPT-4 sinh ra không lộ đáp án, ít bịa, dùng đúng thuật ngữ của bài giảng và trỏ được tới đúng đoạn video bài giảng.
- **Phương pháp:**
  - *Indexing:* phiên âm video bài giảng bằng Whisper ra định dạng .SRT có timestamp, cắt chunk 512 ký tự (overlap 64), embedding bằng text-embedding-ada-002 và lưu vào Postgres + pgvector cùng timestamp và tên file video.
  - *Sinh phản hồi bằng prompt chain hai lượt:* lượt 1, GPT-4 (temperature 0, function calling) đọc đề bài, code, kết quả compiler và unit test để tìm các khái niệm còn thiếu, rồi đặt tối đa 2 câu hỏi truy vấn. Mỗi câu lấy top-4 chunk, tổng tối đa 8 chunk. Lượt 2 sinh phản hồi (tối đa 6 câu, 3 đoạn) có trích dẫn kiểu footnote Markdown. Bấm vào footnote sẽ mở video tại đúng timestamp.
- **Dữ liệu và đánh giá:** workshop ôn thi với 15 sinh viên: 2.192 lượt nộp code, 574 phản hồi (478 không dùng RAG, 96 có RAG). Các phản hồi RAG liên kết 160 đoạn video (trung bình 1,67 đoạn), thuộc 57 đoạn khác nhau.
- **Kết quả:** sinh viên thấy link video hữu ích và giúp nhận ra khái niệm cần dùng. Điểm SUS là 74,8. Nhược điểm là phản hồi RAG mất khoảng 18 giây mới ra token đầu tiên, so với 1–2 giây khi không dùng RAG. Vì vậy loại không RAG được chọn nhiều hơn khoảng 4 lần. Nhiều sinh viên tin phản hồi mà không đặt câu hỏi lại.

## Điểm mạnh
- Ý tưởng hay: trích dẫn bằng timestamp để mở video đúng chỗ, giúp người học kiểm chứng câu trả lời.
- Lượt sinh truy vấn giải quyết được tình huống không có câu hỏi rõ ràng (đầu vào chỉ là code và lỗi).
- Triển khai thật với sinh viên, có đo SUS và độ trễ.

## Điểm yếu / hạn chế
- Mẫu nhỏ (n=15), chỉ thấy được xu hướng. Không đo hiệu quả học tập.
- Chỉ dùng transcript, bỏ qua slide và hình ảnh. Chunk cắt theo số ký tự.
- Độ trễ cao do phải chạy hai lượt LLM. Phụ thuộc GPT-4 API.
- Không đánh giá tự động chất lượng hay độ trung thực của phản hồi.

## So sánh với đề tài của tui
- **Giống:** dùng Whisper với timestamp, lưu metadata thời gian để dẫn về video, và dùng RAG để giảm bịa.
- **Khác:** bài này chỉ làm phản hồi bài tập lập trình bằng tiếng Đức với GPT-4 trên cloud. Đề tài làm QA bài giảng tiếng Việt, chạy LLM cục bộ (qwen2:7b).
- **Đề tài hơn:** có kênh slide (detect, OCR công thức/bảng), retrieval lai e5 + BM25, chunk theo khoảng slide thay vì 512 ký tự cố định, và có gold để đo từng bước.
- **Đề tài kém:** chưa có user study, chưa có SUS, chưa đo độ trễ từ góc nhìn người dùng.

## Có áp dụng được vào đề tài không?
- **Web app / sinh câu trả lời (Cao):** bắt LLM trích dẫn dạng `[video, mm:ss]` hoặc theo số slide, rồi render thành link nhảy tới video. Bài cho thấy sinh viên đánh giá cao tính năng này. Có thể thêm few-shot cách trích dẫn để qwen2:7b làm đúng định dạng.
- **User study (Cao):** áp dụng SUS cùng vài câu hỏi riêng (link có hữu ích không, có quá chậm không, có tin tưởng câu trả lời không). Cần nhóm khoảng 15–30 sinh viên IUH.
- **Truy vấn (Trung bình):** viết lại câu hỏi của sinh viên thành truy vấn theo khái niệm trước khi retrieve. Cần cân nhắc vì sẽ tăng độ trễ trên LLM cục bộ.
- **UX (Trung bình):** stream câu trả lời và hiện cảnh báo rằng nội dung do AI sinh ra.
