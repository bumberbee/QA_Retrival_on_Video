# VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos (arXiv preprint 2025, HKU & Baidu, arXiv 2502.01549)

## Tóm tắt
- **Bài toán:** Làm RAG trên kho nhiều video rất dài (hàng chục đến hàng trăm giờ) và hỏi đáp xuyên nhiều video, điều mà các mô hình ngôn ngữ–thị giác cho video ngắn (LLaMA-VID, VideoAgent) làm không tốt.
- **Phương pháp:** kiến trúc hai kênh.
  1. *Graph-based textual grounding:* cắt video thành clip 30 giây. Mỗi clip có caption từ VLM (MiniCPM-V, 5 frame, có đưa transcript vào prompt) và ASR (Distil-Whisper). Văn bản được gom thành chunk, LLM trích thực thể và quan hệ, rồi ghép thành knowledge graph xuyên video (gộp thực thể, tổng hợp mô tả). Chunk được embedding bằng text-embedding-3-small.
  2. *Multi-modal context encoding:* encode clip bằng ImageBind.
  - *Truy xuất:* nhánh văn bản viết lại câu hỏi, khớp thực thể rồi chọn chunk kiểu GraphRAG. Nhánh hình ảnh viết lại câu hỏi thành mô tả cảnh rồi so cosine với embedding của clip. LLM lọc nhị phân các clip giao nhau giữa hai nhánh, sau đó caption lại với 15 frame theo từ khóa của câu hỏi. GPT-4o-mini dùng cho toàn bộ pipeline.
- **Dữ liệu:** benchmark LongerVideos gồm 164 video, khoảng 134,6 giờ, 602 câu hỏi. Nhóm bài giảng chiếm 135 video, khoảng 64,3 giờ, 376 câu hỏi. Câu hỏi được sinh bằng NotebookLM từ transcript.
- **Kết quả:** đánh giá bằng LLM-judge (GPT-4o-mini) theo 5 tiêu chí, mỗi lần chạy lặp 5 lần và có đổi vị trí câu trả lời.
  - Tỉ lệ thắng tổng thể so với NaiveRAG: 53,26%. So với GraphRAG-l, GraphRAG-g và LightRAG: lần lượt 53,87%, 55,78% và 57,14%.
  - Chấm điểm thang 5 so với baseline NaiveRAG: VideoRAG đạt 4,45 tổng thể, trong khi NotebookLM 3,37, LLaMA-VID 2,44 và VideoAgent 1,98.
  - Ablation bỏ graph hoặc bỏ thị giác đều làm tỉ lệ thắng giảm.

## Điểm mạnh
- Mở rộng được tới số video và độ dài không giới hạn. Kết hợp graph với embedding thị giác. Mã nguồn và benchmark được công khai.
- Có nhóm video bài giảng lớn, phù hợp để tham khảo.
- Có thiết kế giảm thiên vị khi chấm bằng LLM (đổi vị trí, lặp 5 lần).

## Điểm yếu / hạn chế
- Chỉ dùng LLM-judge, không có ground truth hay metric retrieval, và câu hỏi do NotebookLM sinh ra. Mức chênh so với NaiveRAG chỉ khoảng 53/47.
- Chi phí index cao vì gọi LLM và VLM cho mọi clip để dựng graph. Phụ thuộc GPT-4o-mini.
- Cắt clip cố định 30 giây, không theo ranh giới nội dung. VLM caption không chuyên đọc công thức hay bảng.
- Ablation chỉ trình bày bằng biểu đồ, mức chênh nhỏ (khoảng 48–52%).

## So sánh với đề tài của tui
- **Giống:** cùng chuyển video thành văn bản (ASR + mô tả hình ảnh) rồi index và truy xuất. Cả hai đều có thành phần lai.
- **Khác:** VideoRAG dùng clip cố định cộng knowledge graph cộng embedding thị giác. Đề tài dùng đoạn theo slide, OCR lai theo vùng và kết hợp e5 + BM25.
- **Đề tài hơn:**
  - Phân đoạn theo slide có gold (F1 0,970), sát cấu trúc bài giảng.
  - OCR công thức và bảng chính xác hơn caption chung.
  - Chạy cục bộ, chi phí thấp, làm cho tiếng Việt.
  - Đánh giá theo từng bước có nhãn.
- **Đề tài kém:**
  - Chưa liên kết kiến thức xuyên video (câu hỏi gộp nhiều buổi học).
  - Chưa có kênh embedding hình ảnh.
  - Chưa có đánh giá chất lượng câu trả lời kiểu win-rate.

## Có áp dụng được vào đề tài không?
- **Đánh giá (Cao):** áp dụng giao thức win-rate bằng LLM-judge (5 tiêu chí, đổi vị trí, lặp nhiều lần) để so các cấu hình của đề tài, ví dụ có hoặc không mở rộng slide lân cận, có hoặc không BM25. Nên kết hợp với RAGAS và tham chiếu do người viết.
- **Baseline (Trung bình):** chạy VideoRAG (mã nguồn mở) hoặc NaiveRAG trên tập con L25/vn12, hoặc chạy pipeline của đề tài trên phần bài giảng của LongerVideos.
- **Module 05 (Trung bình):** viết lại câu hỏi và dùng LLM lọc ngữ cảnh trước khi sinh câu trả lời, giúp giảm bịa cho qwen2:7b. Cần cân nhắc độ trễ.
- **Liên kết khái niệm xuyên video (Thấp):** dựng graph khái niệm theo môn học (LightRAG/GraphRAG). Chi phí cao, để làm sau.
