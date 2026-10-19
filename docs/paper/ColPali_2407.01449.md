# ColPali: Efficient Document Retrieval with Vision Language Models (ICLR 2025, arXiv:2407.01449)

## Tóm tắt
- **Bài toán:** hệ truy hồi tài liệu hiện nay dựa vào văn bản trích từ trang (PDF parser, OCR, phát hiện bố cục, chunking, captioning), quy trình dài, dễ lỗi và bỏ qua tín hiệu thị giác (hình, bảng, bố cục). Tác giả nhận thấy tối ưu khâu nạp dữ liệu quan trọng hơn tối ưu mô hình embedding văn bản.
- **Đóng góp 1 – ViDoRe:** benchmark truy hồi mức trang gồm nhiều tác vụ: học thuật (DocVQA, InfoVQA, TAT-DQA, arXivQA, TabFQuAD tiếng Pháp) và thực tế (Energy, Government, Healthcare, AI, Shift Project tiếng Pháp; mỗi tác vụ 100 câu hỏi trên 1000 trang). Chỉ số chính: nDCG@5, kèm độ trễ truy vấn và tốc độ lập chỉ mục.
- **Đóng góp 2 – ColPali:** dùng VLM (PaliGemma) nhúng thẳng **ảnh trang** thành đa vector (mỗi patch một vector), so khớp với câu hỏi bằng late interaction kiểu ColBERT.
- **Kết quả chính:** ColPali đạt trung bình nDCG@5 = 81.3 trên ViDoRe, hơn baseline tốt nhất 22.5 điểm. Lập chỉ mục chỉ 0.39 s/trang so với 7.22 s/trang của pipeline PDF parser (chưa tính OCR/captioning). Với baseline văn bản, BM25 và BGE-M3 cho kết quả gần nhau, cho thấy nút thắt nằm ở thông tin thị giác chứ không ở mô hình embedding. Ablation cho thấy đa vector và late interaction là yếu tố then chốt (so với bản một vector BiSigLIP/BiPali).

## Điểm mạnh
- Bỏ được toàn bộ pipeline OCR/layout/captioning ở khâu lập chỉ mục, nhanh và đơn giản, huấn luyện đầu-cuối.
- Benchmark và mô hình công khai, trở thành chuẩn tham chiếu cho truy hồi tài liệu thị giác.
- Tận dụng tốt hình, bảng, infographic mà văn bản thuần bỏ sót.

## Điểm yếu / hạn chế
- Đa vector tốn bộ nhớ lưu trữ hơn nhiều so với một vector/chunk.
- Chỉ huấn luyện/đánh giá chủ yếu tiếng Anh (và một ít tiếng Pháp); chưa rõ hiệu quả với tiếng Việt có dấu.
- Chỉ giải quyết truy hồi; vẫn cần VLM để đọc trang khi sinh câu trả lời. Không xử lý lời giảng (âm thanh).

## So sánh với đề tài của tui
- **Giống:** đều coi trang/slide là đơn vị truy hồi và nhận ra thông tin thị giác (công thức, bảng, sơ đồ) quan trọng.
- **Khác:** đề tài đi theo hướng ColPali phê phán: OCR lai (Unlimited-OCR + Qwen3-VL) rồi nhúng văn bản bằng multilingual-e5 + BM25. ColPali nhúng thẳng ảnh.
- **Đề tài hơn:** kết hợp được transcript lời giảng; văn bản OCR đọc được, trích dẫn được, dùng được cho LLM văn bản cục bộ; tiếng Việt được hỗ trợ qua e5 đa ngôn ngữ.
- **Đề tài kém:** OCR chậm (khoảng 35 s/slide, lớn hơn nhiều so với 0.39 s/trang của ColPali) và gặp lỗi dấu tiếng Việt, đúng các điểm yếu mà bài này nêu về pipeline dựa trên OCR.

## Có áp dụng được vào đề tài không?
- **Module 05 – truy hồi (Trung bình–Cao):** thêm một nhánh truy hồi thị giác dùng ColPali/ColQwen trên keyframe slide, hợp nhất điểm với e5 + BM25 (ví dụ Reciprocal Rank Fusion). Nhánh này không bị ảnh hưởng bởi lỗi dấu OCR. Cần thử nghiệm trên L25 vì câu hỏi tiếng Việt.
- **Đánh giá (Trung bình):** báo cáo nDCG@5/Recall@K mức slide giống ViDoRe, và so sánh thời gian lập chỉ mục theo slide giữa OCR lai và nhúng ảnh.
- **Module 02 – tăng tốc (Thấp):** không thay thế OCR hoàn toàn, vì LLM qwen2:7b cần văn bản để trả lời.
