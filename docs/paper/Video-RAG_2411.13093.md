# Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension (NeurIPS 2025, arXiv:2411.13093)

## Tóm tắt
- **Bài toán:** các LVLM khó hiểu video dài vì giới hạn ngữ cảnh. Hai hướng phổ biến đều tốn kém: tinh chỉnh LVLM ngữ cảnh dài (cần nhiều dữ liệu và GPU) hoặc agent dựa trên GPT (phụ thuộc mô hình đóng như GPT-4o). Tăng số khung hình cũng không chắc tốt hơn: LongVA giảm từ 52.6% xuống 51.8% trên Video-MME khi tăng từ 128 lên 384 khung hình.
- **Phương pháp:** pipeline không cần huấn luyện, cắm vào bất kỳ LVLM nào. Dùng công cụ mã nguồn mở để trích ba loại "văn bản phụ trợ" bám sát hình ảnh từ video: **OCR**, **ASR (Whisper)** và **phát hiện đối tượng** (chỉ chạy trên keyframe chọn bằng độ tương đồng CLIP). Văn bản được chia chunk, mã hóa bằng Contriever, lưu vào FAISS; khi có câu hỏi, chỉ truy hồi một lượt các đoạn liên quan rồi đưa cùng khung hình và câu hỏi vào LVLM.
- **Dữ liệu:** Video-MME, MLVU, LongVideoBench.
- **Kết quả chính:** áp dụng lên LLaVA-Video 72B đạt 77.4% trên Video-MME, nhỉnh hơn GPT-4o (77.2%). Trung bình trên 7 LVLM tăng 2.8% so với khi chỉ dùng phụ đề, tăng rõ nhất ở video dài. Trên LongVideoBench, bản 72B vượt Gemini-1.5-Pro 1.4%, chỉ kém GPT-4o 1.3%; bản 7B tăng 2.1%. Ablation theo số khung hình (8–256) cho thấy cải thiện ở mọi mức, và lợi ích lớn hơn khi dùng ít khung hình.

## Điểm mạnh
- Không cần huấn luyện, chỉ dùng công cụ mở, dễ triển khai và tương thích mọi LVLM.
- Chi phí thấp nhờ truy hồi một lượt; thay token thị giác đắt đỏ bằng văn bản ngắn gọn.
- Chứng minh OCR + ASR là nguồn thông tin bổ trợ rất mạnh cho hiểu video dài.

## Điểm yếu / hạn chế
- Tác giả thừa nhận chất lượng phụ thuộc vào công cụ trích xuất (OCR, detector) đã chọn và chưa có cơ chế thích nghi.
- Bài toán là trắc nghiệm hiểu **một** video, không phải truy hồi trên kho nhiều video.
- Chạy mô hình 72B cần tài nguyên lớn; không xét tiếng Việt.

## So sánh với đề tài của tui
- **Giống:** triết lý gần như trùng: chuyển video thành văn bản (OCR + ASR), chia chunk, nhúng vector, truy hồi rồi đưa cho mô hình ngôn ngữ. Đây là bằng chứng học thuật mạnh cho hướng thiết kế của đề tài.
- **Khác:** đề tài OCR theo từng slide đã phát hiện ranh giới (thay vì khung hình lấy mẫu đều), có OCR lai cho công thức/bảng/sơ đồ, căn chỉnh OCR với transcript theo thời gian, truy hồi lai e5 + BM25 trên kho nhiều bài giảng.
- **Đề tài hơn:** OCR có cấu trúc và theo slide nên ít trùng lặp; hỗ trợ đa video và tiếng Việt; chạy được với LLM 7B cục bộ.
- **Đề tài kém:** chưa dùng khung hình làm đầu vào khi sinh câu trả lời; chưa đánh giá trên benchmark chuẩn như Video-MME.

## Có áp dụng được vào đề tài không?
- **Viết luận văn / phần liên quan (Cao):** trích dẫn Video-RAG làm căn cứ cho lựa chọn "video → văn bản phụ trợ → RAG" và cho việc dùng mô hình nhỏ thay vì mô hình đóng.
- **Module 05 – lọc ngữ cảnh (Trung bình):** áp dụng cách tách cơ sở dữ liệu theo loại nguồn (OCR, ASR) và truy hồi riêng từng loại rồi gộp, có ngưỡng tương đồng để loại đoạn nhiễu, giúp giảm ngữ cảnh lạc đề và ảo giác của qwen2:7b.
- **Module 02 – phát hiện đối tượng (Thấp):** bài giảng chủ yếu là slide chữ, detector đối tượng ít giá trị so với mô tả hình bằng Qwen3-VL đã có.
