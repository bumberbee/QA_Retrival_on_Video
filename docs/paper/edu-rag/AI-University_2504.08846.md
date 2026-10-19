# AI University: An LLM-Powered Learning Assistant for Engineering—A Finite Element Method Case Study (preprint, v2 2026, arXiv:2504.08846)

## Tóm tắt
- **Bài toán:** trợ lý học tập cho một môn cao học (Phương pháp phần tử hữu hạn, FEM, USC). Trợ lý phải trả lời đúng **phong cách giảng dạy** của môn chứ không chỉ đúng kiến thức.
- **Phương pháp:**
  1. **Sinh dữ liệu:** chuyển giáo trình Hughes sang LaTeX bằng Mathpix, mất khoảng 1 giờ công mỗi chương. Lấy transcript tự động của video bài giảng và bài tập code. Chunk theo ranh giới tự nhiên (mục sách, đoạn video). GPT-4o sinh 4.648 cặp hỏi–đáp (sách 1.053, code 286, transcript 3.309). Chuyên gia chấm 200 cặp, điểm trung bình 2,64/3 và 2,74/3.
  2. **Fine-tune LoRA** Llama-3.2-11B-Vision thành LLaMA-TOMMI-1.0, tìm siêu tham số bằng Optuna (rank 45, alpha 65, lr 5e-5), huấn luyện trên 2 GPU A40.
  3. **Tổng hợp:** câu trả lời của expert model được đưa cùng context RAG cho mô hình suy luận DeepSeek-R1-0528-Qwen3-8B. Mô hình này kiểm tra xem thông tin có đủ không, nếu thiếu thì dừng. Nó sửa hoặc làm giàu câu trả lời, gắn nhãn độ tin cậy High/Medium/Low, và trích dẫn kèm link video theo timestamp.
- **Kết quả:**
  - **Expert model trên 465 câu test:**
    - Cosine 0,879 so với 0,818 của base, thắng 86,02% số câu.
    - GPT-4o judge: 43,44% so với 8,39% (judge 1) và 43,23% so với 26,88% (judge 2).
  - **Người dùng nâng cao, 100 câu hỏi thật của sinh viên:** TOMMI 47%, base 25%.
  - **Khi đưa vào nền tảng:** khoảng cách thu hẹp (34% so với 30%, 32% hòa). Giảng viên vẫn chọn TOMMI 55% so với 28%.
  - **Độ "đủ" khi chạy độc lập:** GPT-5 92,5%, DeepSeek-R1-8B 82,5%, TOMMI 52%.
  - **Về bám sát môn học:** giảng viên ưu tiên nền tảng TOMMI hơn GPT-5 (37,5% so với 17,5%).

## Điểm mạnh
- Pipeline đầy đủ, có công bố mã và dữ liệu. Toàn bộ chạy được bằng model mở, triển khai local được.
- Bước synthesis có cơ chế "từ chối khi thiếu ngữ cảnh" và nhãn độ tin cậy, giúp giảm ảo giác.
- Trích dẫn kèm link video theo timestamp. Đánh giá nhiều lớp: cosine, LLM-judge, người dùng nâng cao, giảng viên, inter-annotator agreement.

## Điểm yếu / hạn chế
- Người chấm đều là tác giả (4 người dùng nâng cao và 1 giảng viên). Chưa có nghiên cứu người dùng là sinh viên thật (tác giả tự thừa nhận).
- So với GPT-5 thì việc che danh tính bị lộ do khác biệt phong cách.
- GPT-4o vừa sinh dữ liệu vừa chấm. Bỏ qua hình ảnh trong sách. Video chỉ dùng transcript, không dùng nội dung slide.
- Expert model một mình chỉ đạt 52% "đủ".

## So sánh với đề tài của tui
- **Giống:** dùng transcript video bài giảng làm nguồn tri thức, RAG trích dẫn có link tới thời điểm trong video, LLM chạy local, web app.
- **Khác:**
  - Đề tài khai thác thêm phần thị giác (OCR slide, công thức, bảng qua Qwen3-VL), phân đoạn theo slide, và dùng hybrid retrieval.
  - AI-U fine-tune LLM để bắt phong cách giảng viên, và thêm một mô hình suy luận để tổng hợp.
- **Đề tài hơn:** khai thác đa phương thức, hỗ trợ tiếng Việt, không tốn chi phí fine-tune.
- **Đề tài kém:** chưa có bước kiểm tra "đủ ngữ cảnh / độ tin cậy", chưa có đánh giá của chuyên gia hay giảng viên, chưa fine-tune.

## Có áp dụng được vào đề tài không?
- **Prompt hai bước (Cao):** (a) kiểm tra context có đủ không, nếu không thì trả lời "không có trong bài giảng". (b) Trả lời kèm nhãn độ tin cậy và trích dẫn slide/timestamp. Cách này giảm ảo giác của qwen2:7b gần như không tốn chi phí.
- **Sinh cặp QA từ chunk slide + transcript (Trung bình):** dùng LLM mạnh rồi cho người duyệt một mẫu khoảng 100 cặp (thang 0–3 như bài). Kết quả vừa làm tập đánh giá vừa có thể làm dữ liệu fine-tune.
- **Đánh giá mù có giảng viên / GVHD (Trung bình):** so sánh từng cặp hệ thống, ví dụ có hoặc không có OCR, trên khoảng 50 câu hỏi.
- **LoRA fine-tune một LLM tiếng Việt 7B trên QA của L25 (Thấp):** giúp giảm trộn ngôn ngữ, nhưng tốn GPU và thời gian.
