# Reducing Barriers to Academic Support: Evaluating a Course-Specific RAG System for Addressing Help-Seeking Disparities in Higher Education (bài báo tạp chí Wiley, 2026, arXiv:2609.21600)

## Tóm tắt
- **Bài toán:** nhiều sinh viên ngại hỏi giảng viên vì lo âu, sợ bị đánh giá hoặc thiếu tự tin. Trong khi đó ChatGPT lại trả lời vượt phạm vi môn học, thậm chí dẫn tới vi phạm học thuật. Tác giả xây **Beacon**, một hệ RAG chỉ dựa trên học liệu đã được duyệt của các môn lập trình (Bath Spa University, Anh).
- **Phương pháp (Design-based research, 2 vòng):**
  - **Kho tri thức:** slide viết bằng MARP/Markdown, bài lab, handbook và hướng dẫn đánh giá. Chunk kèm metadata (buổi học, số slide, loại tài liệu).
  - **Lưu trữ và truy hồi:** Nomic Embed Text, PostgreSQL + pgvector, lấy top-7 theo cosine, không đặt ngưỡng.
  - **Hội thoại nhiều lượt:** câu hỏi tiếp theo được viết lại thành truy vấn độc lập dựa trên lịch sử hội thoại. Bản viết lại chỉ dùng để truy hồi, prompt vẫn giữ câu gốc.
  - **Prompt:** chỉ dùng tài liệu đã truy hồi, không suy đoán, ưu tiên gợi ý và pseudocode thay vì đưa lời giải hoàn chỉnh. Câu trả lời hiển thị kèm nguồn.
  - **Mô hình:** GPT-OSS-20B ở bản 1 (Streamlit), lên GPT-OSS-120B ở bản 2 (Flask).
- **Đánh giá:** pilot với 9 phản hồi. Nghiên cứu chính gồm 15 sinh viên làm 6 kịch bản, bảng hỏi trước và sau, 5 phỏng vấn sinh viên và 4 phỏng vấn giảng viên.
- **Kết quả:**
  - **Pilot:** 89% đánh giá câu trả lời bám sát môn học.
  - **Nghiên cứu chính, về độ tin cậy và mức bám môn:**
    - Bám học liệu: 93,3%.
    - Nguồn trích dẫn làm tăng niềm tin: 86,7%.
  - **Về việc hỏi trợ giúp:**
    - Dễ hỏi riêng tư hơn: 86,7%.
    - Có lợi cho sinh viên thiếu tự tin: 93,3%.
  - **Về việc học:**
    - Giúp tự học: 80%.
    - Hiểu khái niệm tốt hơn: 53,3%.
  - **Về khả năng dùng:**
    - Thấy trực quan: 73,3%.
    - Sẽ dùng lại: 46,6% (33% trung lập).
  - **Phàn nàn:** chậm, câu trả lời quá dài, đôi khi ảo giác hoặc kém hiệu quả khi hỏi ngoài phạm vi.

## Điểm mạnh
- Đánh giá người dùng hỗn hợp (định lượng + phỏng vấn) với cả sinh viên lẫn giảng viên. Nêu được các "design tension" thực tế.
- Viết lại truy vấn trong hội thoại nhiều lượt là chi tiết kỹ thuật đơn giản mà hữu ích.
- Metadata số slide giúp sinh viên kiểm chứng câu trả lời. Toàn bộ là mã nguồn mở.

## Điểm yếu / hạn chế
- Mẫu nhỏ (15 sinh viên), chỉ một trường và một bối cảnh. Mỗi người chỉ dùng một buổi, sau khi môn học đã kết thúc.
- Không đo độ chính xác câu trả lời một cách độc lập, ảo giác chỉ ghi nhận qua giai thoại.
- Phân tích định tính có hỗ trợ của Copilot. Lợi ích giảm vi phạm học thuật chưa được chứng minh.
- Học liệu vốn đã là Markdown sạch, không phải trích xuất từ video.

## So sánh với đề tài của tui
- **Giống:** RAG theo môn học, trích dẫn tới slide, web chat, model mở.
- **Khác:** Beacon có sẵn slide dạng text (MARP). Đề tài phải tự trích xuất slide từ video qua phát hiện slide, OCR và ASR. Beacon thiên về đánh giá sư phạm và trải nghiệm người dùng, còn đề tài thiên về kỹ thuật pipeline.
- **Đề tài hơn:** xử lý video thật, có metric kỹ thuật (F1 phát hiện slide), có hybrid retrieval.
- **Đề tài kém:** chưa có nghiên cứu người dùng, chưa xử lý hội thoại nhiều lượt, câu trả lời chưa được định hướng sư phạm.

## Có áp dụng được vào đề tài không?
- **Viết lại câu hỏi tiếp theo thành truy vấn độc lập trước khi chạy hybrid search (Cao):** rẻ, cải thiện rõ các câu như "giải thích thêm đi".
- **Nghiên cứu người dùng nhỏ theo mẫu của bài (Cao):** 10–15 sinh viên IUH, khoảng 6 kịch bản, bảng hỏi Likert trước và sau (bám bài giảng, tin cậy, dễ hỏi, sẽ dùng lại), cộng 3–5 phỏng vấn. Đây là cách trực tiếp lấp điểm yếu "chưa có user study".
- **Prompt sư phạm và giới hạn độ dài (Trung bình):** câu ngắn thì trả lời ngắn, chỉ dẫn tới slide hoặc timestamp. Mục tiêu là tránh câu trả lời dài lê thê, cũng là phàn nàn chính trong bài.
- **Tải trước slide gốc (PDF/PPTX) nếu giảng viên cung cấp (Thấp):** bỏ qua bước OCR chậm (~35 s/slide) cho những môn đã có file.
