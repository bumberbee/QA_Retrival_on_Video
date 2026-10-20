# Towards Retrieval Augmented Generation over Large Video Libraries (Moments Lab Research, 2024, arXiv 2406.14938)

## Tóm tắt
- **Bài toán:** Đề xuất bài toán Video Library Question Answering (VLQA): trả lời yêu cầu bằng ngôn ngữ tự nhiên trên một kho video lớn, kèm tham chiếu tới đúng các đoạn (moment) video. Ví dụ ứng dụng là người làm nội dung cần tìm cảnh quay cho phóng sự.
- **Phương pháp:** kiến trúc đơn giản, không phụ thuộc thành phần cụ thể.
  - *Tách và index:* cắt video thành moment bằng PySceneDetect (theo biến thiên màu HSV). Mỗi moment có metadata gồm diarization (pyannote), phiên âm (Whisper) và caption cho 3 khung hình lấy đều (BLIP-2), lưu thành tài liệu văn bản.
  - *Retriever:* LLM sinh ít nhất 5 truy vấn từ khóa, gửi tới search engine văn bản (OpenSearch), giữ tối đa 50 moment.
  - *Sinh câu trả lời:* LLM nhận câu hỏi và metadata, trả lời kèm tham chiếu dạng `[video_id](timestamp_in;timestamp_out)`, sau đó được thay bằng hyperlink. Các LLM cho kết quả tốt: Mistral Large, Mixtral 8x7B, Command-R.
- **Dữ liệu:** thư viện NASA gồm 9.388 video (từ 10 giây đến hơn 2 giờ), tách thành 372.181 moment.
- **Kết quả:** chỉ có kết quả sơ bộ định tính qua 2 ví dụ (astronaut ăn trên ISS, trailer Apollo). Mỗi câu trả lời mất khoảng 15 giây. Bài không có metric định lượng.

## Điểm mạnh
- Kiến trúc gọn, dễ thay LLM hay search engine, mở rộng được tới hàng nghìn giờ video mà không cần LLM có ngữ cảnh dài.
- Caption hình ảnh giúp tìm được cả những đoạn không có lời nói.
- Thẳng thắn nêu hạn chế: LLM bịa link YouTube thay vì dùng moment đã truy xuất, không đếm được sự kiện, và thiếu benchmark.

## Điểm yếu / hạn chế
- Không có đánh giá định lượng hay benchmark. Bài chỉ là định hướng (position/demo).
- Chỉ trả lời được những gì có trong metadata. Caption BLIP-2 nông, không đọc được chữ hay công thức.
- Cắt cảnh theo màu không phù hợp với video bài giảng, nơi khung hình gần như tĩnh.
- Không có reranker, nên nguy cơ lẫn moment không liên quan là cao.

## So sánh với đề tài của tui
- **Giống:** cùng hướng "chuyển video thành metadata văn bản rồi làm RAG": tách đoạn, ASR có timestamp, mô tả hình ảnh, rồi trả lời kèm mốc thời gian.
- **Khác:** bài này dùng video đa dạng (NASA) và tìm theo từ khóa. Đề tài chuyên cho video bài giảng.
- **Đề tài hơn:**
  - Tách đoạn theo slide với gold F1 0,970, thay vì cắt theo màu không được đánh giá.
  - OCR lai đọc được chữ, công thức và bảng, thay vì caption chung chung.
  - Retrieval lai dense + BM25 có mở rộng sang slide lân cận.
  - Có số liệu đo trên dữ liệu thật (L25, vn12, LPM, M3AV, SlideSpeech).
- **Đề tài kém:** chưa thử diarization, và quy mô kho nhỏ hơn (88 video, 36 giờ so với 9.388 video).

## Có áp dụng được vào đề tài không?
- **Sinh câu trả lời / web app (Cao):** ép định dạng trích dẫn cố định kiểu `[video_id](in;out)` rồi hậu xử lý thành link. Thêm bước kiểm tra trích dẫn: loại mọi link hoặc ID không nằm trong ngữ cảnh đã truy xuất, vì bài ghi nhận LLM hay bịa link.
- **Truy vấn (Trung bình):** cho LLM sinh nhiều truy vấn từ khóa cho nhánh BM25 (multi-query), hữu ích với câu hỏi tiếng Việt diễn đạt khác từ trong slide.
- **Retrieval (Trung bình):** bổ sung reranker (ví dụ cross-encoder đa ngữ) sau bước hybrid. Đây cũng là hướng tương lai mà bài đề xuất.
- **Module 02 (Thấp):** caption cho frame không phải slide (giảng viên viết bảng, demo) thay vì bỏ qua hẳn qua SigLIP gate.
