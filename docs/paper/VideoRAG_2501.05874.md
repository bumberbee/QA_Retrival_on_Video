# VideoRAG: Retrieval-Augmented Generation over Video Corpus (KAIST/DeepAuto.ai, 2025, arXiv:2501.05874)

## Tóm tắt
- **Bài toán:** RAG hiện tại chủ yếu dùng văn bản hoặc ảnh tĩnh; các nghiên cứu dùng video thì hoặc gán sẵn video liên quan cho câu hỏi (không truy hồi), hoặc chuyển video thành mô tả văn bản nên mất thông tin thị giác.
- **Phương pháp:** VideoRAG truy hồi động các video liên quan tới câu hỏi trong một kho video, rồi đưa trực tiếp khung hình và văn bản (phụ đề) vào một Large Video Language Model (LVLM) để sinh câu trả lời. Có hai cơ chế bổ sung: (1) **chọn khung hình** (frame selection): một mô hình được huấn luyện để chọn tập con khung hình giàu thông tin nhất, vì LVLM không đủ ngữ cảnh cho mọi khung hình; (2) **văn bản phụ trợ**: khi video không có phụ đề thì dùng ASR (Whisper) để tạo transcript, phục vụ cả truy hồi lẫn sinh.
- **Dữ liệu:** WikiHowQA (câu hỏi và đáp án do người viết) ghép với kho video hướng dẫn HowTo100M, cộng thêm bộ câu hỏi–đáp án tổng hợp tự động trên HowTo100M. Đánh giá bằng ROUGE-L, BLEU-4, BERTScore và G-Eval (GPT-4o-mini chấm thang Likert 5 điểm). Bộ sinh gồm LLaVA-Video 7B, InternVL 2.5 8B, Qwen-2.5-VL 3B; bộ mã hóa truy hồi dùng InternVideo2.
- **Kết quả chính (LLaVA-Video 7B, WikiHowQA):** ROUGE-L của NAÏVE là 14.08, TextRAG (BM25) 17.22, TextImageRAG 22.43, TextVideoRAG 22.81, VideoRAG-V 24.95, VideoRAG-VT 24.93 (Oracle-V 26.19). Trên bộ tổng hợp, VideoRAG-VT đạt ROUGE-L 29.74 so với 14.70 của TextRAG (BM25). Mọi biến thể RAG đều hơn baseline không truy hồi, và VideoRAG tốt nhất trong các phương pháp không dùng Oracle.

## Điểm mạnh
- Là một trong những công trình đầu tiên chạy trọn pipeline RAG trên kho video: truy hồi động rồi sinh câu trả lời có căn cứ.
- Phân tích rõ đóng góp của đặc trưng thị giác và văn bản; cho thấy văn bản (transcript) rất quan trọng cho chất lượng truy hồi.
- Khoảng cách tới Oracle nhỏ, nghĩa là bước truy hồi đã khá tốt.

## Điểm yếu / hạn chế
- Tác giả thừa nhận thiếu benchmark chuẩn có đủ bộ ba (câu hỏi, video liên quan, đáp án); phải tự ghép dữ liệu và sinh thêm dữ liệu tổng hợp.
- Đơn vị truy hồi là **cả video**, không phải đoạn nhỏ; video hướng dẫn WikiHow ngắn và khác xa bài giảng dài nhiều slide.
- Chỉ đánh giá bằng tiếng Anh; các chỉ số n-gram có giá trị tuyệt đối thấp (BLEU-4 khoảng 2–8).

## So sánh với đề tài của tui
- **Giống:** cùng làm RAG trên video, dùng ASR họ Whisper để có văn bản, đều kết hợp thông tin thị giác và lời nói.
- **Khác:** đề tài chuyển thị giác thành văn bản có cấu trúc (OCR slide, mô tả công thức/bảng/hình bằng Qwen3-VL) rồi truy hồi lai e5 + BM25 ở mức từng slide/chunk; VideoRAG giữ khung hình thô và đưa thẳng vào LVLM ở mức cả video.
- **Đề tài hơn:** truy hồi mịn hơn (theo khoảng slide, có mở rộng slide lân cận), có trích dẫn thời điểm, hỗ trợ tiếng Việt, có gold về ranh giới slide (F1 0.970). Bài giảng có slide nên OCR rất giàu thông tin, phù hợp hơn với cách chuyển sang văn bản.
- **Đề tài kém:** mất một phần thông tin thị giác khi chỉ còn văn bản; chưa có chỉ số đánh giá câu trả lời như ROUGE/BERTScore/G-Eval.

## Có áp dụng được vào đề tài không?
- **Đánh giá (Cao):** dùng bộ chỉ số ROUGE-L, BERTScore và một chấm điểm kiểu G-Eval (LLM chấm thang 1–5) cho câu trả lời trên tập câu hỏi của L25/vn12; so sánh các cấu hình NAÏVE, chỉ transcript, chỉ OCR, OCR + transcript giống cách bài tách VideoRAG-V/VT.
- **Module 05 – sinh câu trả lời (Trung bình):** với câu hỏi về hình vẽ/sơ đồ, đính kèm keyframe của slide truy hồi được vào một VLM (Qwen-VL) thay vì chỉ đưa văn bản.
- **Module 01/02 – chọn khung hình (Thấp):** ý tưởng chọn khung hình giàu thông tin đã được đề tài giải quyết bằng phát hiện slide + cổng SigLIP, không cần mô hình chọn khung hình học được.
