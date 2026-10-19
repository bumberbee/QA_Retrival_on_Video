# Video Enriched Retrieval Augmented Generation Using Aligned Video Captions (SIGIR MRR Workshop 2024, arXiv:2405.17706)

## Tóm tắt
- **Bài toán:** đưa nội dung video vào chatbot RAG mà không phải nhồi hàng loạt frame vào context của LLM đa phương thức.
- **Phương pháp:** dùng "aligned video captions": caption mô tả cảnh do máy sinh, ghép theo thời gian với phụ đề/ASR, thành một bản transcript dạng text. Chunk, nhúng bằng `text-embedding-3-small`, rồi truy hồi theo cosine. App demo chọn "tool" (mỗi tool ứng với một phần catalog), trả lời theo schema pydantic và kèm timestamp để mở đúng đoạn video.
- **Dữ liệu:** 29.259 video YouTube lấy mẫu từ Panda-70M (khoảng 1,48 triệu cảnh, khoảng 215 ngày video), có thêm phụ đề lấy qua YouTube API.
- **Kết quả:**
  - **Tóm tắt video:** so với bản tóm tắt GPT-4 làm từ aligned captions, BERTScore đạt 0,893 khi chỉ dùng ASR, 0,889 khi dùng frame + ASR (GPT-4V), 0,869 với visual captions và 0,862 với Gemini 1.5 Pro dùng video gốc.
  - **Chi phí:** nếu lấy mẫu 1 fps thì tốn khoảng 4,8 tỷ token, gấp khoảng 69 lần so với dùng captions.
  - **Retrieval** (1000 câu hỏi do GPT-4 sinh, GPT-4 chấm):
    - Aligned transcript: HIT@1 0,741, HIT@10 0,971, QUALITY@1 7,377.
    - ASR: HIT@1 0,741, QUALITY@1 7,424.
    - CLIP trên frame: HIT@1 0,542.
    - BLIP-2: HIT@1 0,482.

## Điểm mạnh
- Biểu diễn video thành text nên dễ đưa vào bất kỳ pipeline RAG nào, rẻ token.
- Số liệu cho thấy rõ text embedding (ASR/transcript) vượt hẳn cross-modal embedding (CLIP/BLIP-2) ở K thấp.
- Có timestamp để dẫn người dùng về đúng đoạn video. Có công bố code và dataset.

## Điểm yếu / hạn chế
- Bài workshop chỉ 5 trang. Câu hỏi và nhãn đều do GPT-4 sinh và chấm, không có đánh giá của người.
- Video là video YouTube đại trà, không phải bài giảng. Không có OCR chữ trên slide.
- Chưa giải quyết bài toán chọn đoạn clip có nghĩa để caption, tác giả tự nêu đây là hướng tương lai.
- Chỉ xét tiếng Anh.

## So sánh với đề tài của tui
- **Giống:** cùng ý tưởng ghép tín hiệu thị giác với lời nói theo thời gian thành text rồi index (bước 04 align OCR + transcript).
- **Khác:** đề tài dùng OCR/VLM trên slide (nhiều chữ, công thức, bảng) thay cho caption cảnh chung chung. Đề tài cắt đoạn theo ranh giới slide (F1 0,970), có hybrid BM25 + e5, có mở rộng sang slide lân cận, và làm cho tiếng Việt.
- **Đề tài hơn:** phân đoạn có nghĩa theo slide, nội dung chữ chính xác hơn caption, chạy LLM local.
- **Đề tài kém:** chưa có bộ đánh giá retrieval quy mô lớn kiểu HIT@K/QUALITY@1 tự động. Chưa so sánh ablation từng nguồn (chỉ ASR, chỉ OCR, aligned).

## Có áp dụng được vào đề tài không?
- **Đánh giá ablation nguồn index (Cao):** chạy retrieval với 3 cấu hình (chỉ transcript, chỉ OCR slide, aligned OCR + transcript) trên L25/vn12. Báo cáo HIT@1/5/10 để chứng minh giá trị của bước 04.
- **Sinh câu hỏi tự động từ chunk + LLM-judge (Trung bình):** dùng quy trình của bài (LLM sinh câu hỏi từ transcript, LLM chấm hit) để tạo nhanh tập test tiếng Việt. Cần người kiểm lại một mẫu nhỏ.
- **Timestamp trong câu trả lời (Trung bình):** web app trả kèm link nhảy tới giây tương ứng của slide/đoạn nói.
- **Caption cảnh cho frame không phải slide (Thấp):** các frame bị cổng SigLIP bỏ qua có thể được caption ngắn thay vì bỏ hẳn.
