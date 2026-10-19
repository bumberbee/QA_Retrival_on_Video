# Revolutionizing Undergraduate Learning: CourseGPT and Its Generative AI Advancements (2024, arXiv:2407.18310)

## Tóm tắt
- **Bài toán:** trợ lý hỏi đáp cho một môn đại học (pilot là môn CPR E 431, bảo mật hệ thống thông tin, Iowa State University), trả lời dựa trên học liệu của môn.
- **Phương pháp:**
  - RAG với embedding `gte-large-en-v1.5` (context 8192).
  - Kho tri thức gồm giáo trình, slide, đề cương, lịch học và tài liệu bổ sung, đã làm sạch.
  - Tìm theo tiêu đề mục rồi lấy nguyên cả mục.
  - So sánh hai LLM: Mistral-7B và Mixtral-8x7B.
- **Đánh giá:** 50 câu hỏi do công cụ đánh giá sinh ra, chấm bằng LLM theo các metric kiểu RAGAS:
  - Answer correctness = 0,25 × cosine + 0,75 × F1 trên các mệnh đề.
  - Context recall.
  - Faithfulness.
- **Kết quả:**

  | Metric | Mixtral-8x7B | Mistral-7B |
  |---|---|---|
  | Correctness | 88,0% | 78,2% |
  | Faithfulness | 66,6% | 60,2% |
  | Context recall | 99,6% | 99,0% |

  - Mistral-7B ngang Mixtral ở một số chủ đề (Authentication, IDS, DoS, …).
  - Khảo sát 6 người (cựu sinh viên và 1 trợ giảng), thang Likert 12 câu: về mức hữu ích, 4 người cho 5 điểm và 2 người cho 4 điểm. Về độ chính xác có 1 người cho 2 điểm. Về mức hài lòng tổng thể, phần chữ của bài ghi kết quả lẫn lộn (2 người cho 2 điểm, 2 người cho 5 điểm), nhưng Hình 6 lại ghi 4 người cho 4 điểm và 2 người cho 5 điểm. Bài tự mâu thuẫn ở điểm này.

## Điểm mạnh
- Có số liệu định lượng rõ ràng và so sánh được ảnh hưởng của kích thước LLM khi giữ nguyên kho tri thức.
- Dùng LLM mã nguồn mở và tự host. Có phân tích theo chủ đề.
- Các công thức metric được trình bày cụ thể, dễ tái lập.

## Điểm yếu / hạn chế
- Chỉ 50 câu hỏi tự sinh, 6 người khảo sát. Phần mô tả khảo sát mâu thuẫn với hình vẽ.
- Gọi là "fine-tuned" nhưng thực chất là RAG.
- Context recall gần 100% gợi ý tập test dễ, hoặc bị rò rỉ từ chính kho tri thức.
- Chỉ có text, chưa xử lý hình hay biểu đồ (tác giả tự nêu). Faithfulness chỉ khoảng 60–67%, vẫn còn nhiều ảo giác.

## So sánh với đề tài của tui
- **Giống:** RAG trên học liệu của môn, LLM mở chạy local cỡ 7B, có giao diện cho sinh viên.
- **Khác:** đề tài lấy tri thức từ video bài giảng (OCR + ASR) chứ không có sẵn text sạch. Đề tài có hybrid retrieval và mở rộng lân cận, và làm cho tiếng Việt.
- **Đề tài hơn:** pipeline đa phương thức, có đánh giá phát hiện slide trên tập gold.
- **Đề tài kém:** chưa có correctness/faithfulness/context recall và chưa có khảo sát người dùng, mà CourseGPT đã có cả hai dù ở mức nhỏ.

## Có áp dụng được vào đề tài không?
- **Bộ 3 metric correctness / faithfulness / context recall (Cao):** áp dụng theo đúng công thức của bài cho qwen2:7b. Đây là cách trực tiếp nhất để lấp lỗ hổng "chưa có đánh giá câu trả lời".
- **So sánh kích thước LLM với cùng kho tri thức (Trung bình):** ví dụ qwen2:7b với một model 14B, hoặc một model khác hỗ trợ tiếng Việt tốt hơn. Mục tiêu là định lượng tỉ lệ ảo giác và trộn ngôn ngữ.
- **Khảo sát Likert nhỏ (Trung bình):** 6–15 sinh viên IUH, chấm mức hữu ích, độ chính xác và mức hài lòng. Báo cáo trung thực, tránh mâu thuẫn như bài này.
- **Phân tích lỗi theo chủ đề / bài giảng (Thấp):** chỉ ra bài giảng nào OCR dấu tiếng Việt kém thì kéo theo câu trả lời kém.
