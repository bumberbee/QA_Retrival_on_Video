# PhoWhisper: Automatic Speech Recognition for Vietnamese (ICLR 2024 Tiny Paper, arXiv:2406.02555)

## Tóm tắt
- **Bài toán:** nhận dạng tiếng nói (ASR) tiếng Việt, đặc biệt bền vững với nhiều giọng vùng miền.
- **Phương pháp:** fine-tune 5 kích thước Whisper đa ngôn ngữ (tiny, base, small, medium, large-v2) bằng `transformers`; 8 GPU A100 40GB, batch toàn cục 64, 48.000 bước (~5 epoch), learning rate giảm dần theo kích thước (3.75e-5 cho tiny đến 5e-6 cho large). Một nửa tập huấn luyện được trộn tiếng ồn môi trường (ESC, thư viện audiomentations).
- **Dữ liệu:** 843,79 giờ: CMV-Vi (Common Voice 14, 3,04 h), VIVOS (13,94 h), VLSP 2020 Task-1 (240,91 h) và dữ liệu riêng 585,90 h của 26K người thuộc 63 tỉnh/thành.
- **Kết quả (WER %, CMV-Vi / VIVOS / VLSP T1 / VLSP T2):** PhoWhisper-large 8,14 / 4,67 / 13,75 / 26,68; medium 8,27 / 4,97 / 14,12 / 26,85; small 11,08 / 6,33 / 15,93 / 32,96. Baseline mạnh nhất wav2vec2-large-vi-vlsp2020: 101,41 / 8,61 / 15,18 / 36,75. PhoWhisper-large đạt SOTA trên cả 4 bộ.

## Điểm mạnh
- Mô hình mở, dùng được trực tiếp với `transformers` và `openai-whisper` → thay thế Whisper gốc mà không đổi pipeline.
- Dữ liệu giọng đa vùng miền lớn, có tăng cường nhiễu → hợp với video giảng dạy thu âm phòng học.
- Có nhiều kích thước: small/medium cho tỉ lệ chất lượng/tốc độ tốt.

## Điểm yếu / hạn chế
- Chỉ là tiny paper 2 trang: không có phân tích lỗi, không đánh giá audio dài (long-form), không đo tốc độ, không báo cáo hallucination.
- 585,9 h dữ liệu riêng không công bố → khó tái lập.
- Chỉ so với wav2vec2, không so với Whisper gốc chưa fine-tune.
- Không có dữ liệu bài giảng/thuật ngữ chuyên ngành (CNTT, toán), không xử lý chen tiếng Anh (code-switching) – rất phổ biến trong giảng dạy.
- WER trên VLSP Task-2 (dữ liệu khó) vẫn 26,68%.

## So sánh với đề tài của tui
- **Giống:** đều dùng họ Whisper cho tiếng Việt; module (03) của đề tài chính là bài toán này.
- **Khác:** đề tài chạy ASR theo từng khoảng slide trên video bài giảng dài, có ngữ cảnh slide (OCR) bên cạnh; PhoWhisper chỉ đánh giá trên câu ngắn đọc/hội thoại.
- **Đề tài hơn:** có dữ liệu bài giảng thật (L25 36 h, vn12), có căn chỉnh thời gian với slide và đánh giá end-to-end.
- **Đề tài kém:** chưa fine-tune ASR cho tiếng Việt; nếu đang dùng Whisper gốc thì WER tiếng Việt có thể cao hơn đáng kể.

## Có áp dụng được vào đề tài không?
- **Module (03) ASR – Ưu tiên Cao:** thay/so sánh Whisper gốc với PhoWhisper-small/medium trên tập 10 video có nhãn L25 (đã có script `eval_asr.py`), báo cáo WER/CER và thời gian xử lý. Lưu ý PhoWhisper chủ yếu train đoạn ≤30 s → hợp với cách cắt theo khoảng slide, nhưng cần chia đoạn dài hơn 30 s (VAD).
- **Module (04)/(05) – Trung bình:** dùng OCR slide làm danh sách thuật ngữ để hậu xử lý transcript (sửa thuật ngữ, chữ tiếng Anh) vì PhoWhisper không xử lý code-switching.
- **Fine-tune thêm trên bài giảng – Thấp:** chỉ làm nếu còn thời gian, cần transcript gold.
