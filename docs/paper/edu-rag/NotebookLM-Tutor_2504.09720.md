# NotebookLM as a Socratic physics tutor: Design and preliminary observations of a RAG-based tool (arXiv 2025, physics.ed-ph, arXiv 2504.09720)

## Tóm tắt
- **Bài toán:** Dùng một công cụ RAG có sẵn (Google NotebookLM, chạy Gemini 2.5 Flash vào thời điểm viết) làm gia sư vật lý theo lối hỏi gợi mở (Socratic), không đưa đáp án trực tiếp, và bám vào tài liệu do giáo viên cung cấp để giảm bịa.
- **Phương pháp:**
  - Viết một "Training Manual" và nạp vào như một tài liệu nguồn. Manual quy định cách hội thoại, gợi ý tăng dần, cách đánh giá câu trả lời của học sinh và yêu cầu không tiết lộ chỉ dẫn. Manual được chỉnh qua nhiều vòng, ví dụ thêm luật "adaptive scaffolding" (hỗ trợ nhiều hơn khi học sinh sai nhiều lần) sau khi thấy người dùng nản.
  - Hướng dẫn có 3 lớp: custom chat instruction, ghi chú sau lời chào, và Training Manual.
  - Chia sẻ cho học sinh ở chế độ chỉ chat (cần gói NotebookLM Plus), để học sinh không xem được lời giải ẩn.
  - Bài tập lấy từ giáo trình *College Physics: Explore and Apply*, đã được đổi cách diễn đạt.
- **Quan sát về định dạng:** cùng một đồ thị v–t, NotebookLM đọc đúng khi ở trong Google Docs nhưng báo "no-image-detected" khi ở PDF export mặc định. Lý do được nêu là PDF thiếu thẻ cấu trúc nên hệ thống phải dựa vào OCR. NotebookLM cũng không render LaTeX.
- **Đánh giá:** chỉ định tính. Tác giả tự đóng vai học sinh để thử, sau đó demo cho giáo viên tiền công vụ (một số người nản vì không được đáp án) và webinar với khoảng 95 giáo viên đang dạy (phản hồi rất tích cực). Bài không có số liệu định lượng về độ chính xác hay hiệu quả học tập.

## Điểm mạnh
- Cách làm thực tế, dễ nhân rộng: giáo viên không cần lập trình vẫn tạo được gia sư có grounding và chia sẻ bằng một link.
- Mô tả rõ quy trình chỉnh prompt qua nhiều vòng và nêu được mâu thuẫn giữa mục tiêu sư phạm và động lực của người học.
- Phát hiện thực tế: định dạng tài liệu ảnh hưởng lớn đến việc hệ thống hiểu hình và đồ thị.

## Điểm yếu / hạn chế
- Proof-of-concept, không có user study chính thức, không có số liệu. Phần tiếp nhận tích cực có nhiều yếu tố gây nhiễu.
- Phụ thuộc nền tảng đóng và gói trả phí (khoảng €20/tháng), không kiểm soát được retrieval hay chunking.
- Chỉ tương tác bằng văn bản, yếu với đồ thị phức tạp và công thức. Video phụ thuộc chất lượng transcript.

## So sánh với đề tài của tui
- **Giống:** cùng mục tiêu dùng RAG trên tài liệu môn học để giảm bịa và có trích dẫn nguồn.
- **Khác:** bài này dùng hệ thống đóng làm sẵn, tập trung vào thiết kế sư phạm. Đề tài tự xây toàn bộ pipeline từ video.
- **Đề tài hơn:** xử lý video bài giảng đầy đủ (slide, OCR công thức/bảng/sơ đồ bằng Qwen3-VL), đúng điểm mà bài này nói NotebookLM yếu (hình trong PDF, không có LaTeX, video chỉ dựa vào transcript). Đề tài cũng chạy cục bộ, có số liệu định lượng từng bước và hỗ trợ tiếng Việt.
- **Đề tài kém:** chưa có chế độ gợi mở sư phạm và chưa có buổi demo hay khảo sát với người dùng.

## Có áp dụng được vào đề tài không?
- **Baseline so sánh (Trung bình):** nạp cùng video hoặc tài liệu của vài bài L25 vào NotebookLM rồi so câu trả lời với hệ thống của đề tài. Lưu ý VideoRAG cũng dùng NotebookLM làm baseline.
- **Prompt hệ thống (Trung bình):** áp dụng ý tưởng Training Manual. Viết bộ luật rõ ràng cho qwen2:7b: chỉ trả lời bằng tiếng Việt, luôn trích slide/timestamp, nói "không có trong bài giảng" khi thiếu ngữ cảnh. Sau đó chỉnh nhiều vòng dựa trên log lỗi (lẫn ngôn ngữ, bịa).
- **Chế độ gợi mở (Thấp):** có thể thêm tùy chọn Socratic hoặc gợi ý tăng dần, chưa cần ưu tiên.
- **Lập luận trong luận văn (Thấp):** dùng phát hiện về đồ thị trong PDF để biện minh cho module OCR lai theo vùng.
