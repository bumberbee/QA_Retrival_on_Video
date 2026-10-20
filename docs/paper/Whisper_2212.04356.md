# Robust Speech Recognition via Large-Scale Weak Supervision (ICML 2023, arXiv:2212.04356)

## Tóm tắt
- **Bài toán:** xây hệ ASR chạy tốt "out of the box", không cần fine-tune theo từng bộ dữ liệu.
- **Phương pháp:** Transformer encoder-decoder trên log-Mel, audio cắt thành đoạn 30 giây; huấn luyện đa nhiệm (nhận dạng, dịch X→en, nhận diện ngôn ngữ, phát hiện không có tiếng nói, dự đoán timestamp) qua chuỗi token đặc biệt. Dữ liệu web lọc heuristic (loại transcript do máy sinh, khớp ngôn ngữ).
- **Dữ liệu:** 680.000 giờ, trong đó 117.000 giờ cho 96 ngôn ngữ khác tiếng Anh và 125.000 giờ dịch X→en. Tiếng Việt có khoảng 691 giờ dữ liệu nhận dạng (bảng phụ lục).
- **Kết quả chính:** zero-shot đạt WER 2,7% trên LibriSpeech clean, ngang wav2vec2-large-960h (huấn luyện có giám sát trên LibriSpeech), nhưng so với mô hình này Whisper giảm trung bình 55,2% lỗi trên các bộ ngoài phân phối → tính bền vững cao, tiệm cận con người. Long-form: cạnh tranh với hệ thương mại; các heuristic giải mã (beam search, temperature fallback, VAD, ràng buộc timestamp đầu, điều kiện văn bản trước) giảm WER từng bước (Bảng 7). Hiệu năng theo ngôn ngữ tương quan mạnh với số giờ dữ liệu huấn luyện.

## Điểm mạnh
- Mô hình mở nhiều kích thước (tiny→large-v2), đa ngôn ngữ, có timestamp mức đoạn.
- Bền vững với nhiễu, giọng nói, miền dữ liệu khác nhau; không cần fine-tune.
- Phân tích kỹ: scaling theo dữ liệu/mô hình, so sánh với con người, chiến lược giải mã long-form.

## Điểm yếu / hạn chế
- Chính tác giả nêu: lỗi long-form "phi con người" – lặp vòng, bỏ từ đầu/cuối đoạn, **hallucination hoàn toàn** không liên quan audio.
- Ngôn ngữ ít dữ liệu (như tiếng Việt) WER còn cao hơn hẳn tiếng Anh.
- Mô hình lớn chậm; timestamp chỉ mức đoạn, không có mức từ.
- Dữ liệu huấn luyện không công bố.

## So sánh với đề tài của tui
- **Giống:** đề tài dùng Whisper-family ở module (03), tận dụng timestamp để gắn transcript vào khoảng slide.
- **Khác:** Whisper là mô hình nền ASR; đề tài là hệ thống RAG đa phương thức, ASR chỉ là một khâu.
- **Đề tài hơn:** có ngữ cảnh slide để kiểm tra chéo transcript; cắt audio theo khoảng slide giúp giảm phụ thuộc vào long-form decoding.
- **Đề tài kém:** chưa có cơ chế phát hiện/loại hallucination ASR; đoạn rất ngắn/rất dài (đã nêu là điểm yếu) đúng là trường hợp Whisper dễ lỗi.

## Có áp dụng được vào đề tài không?
- **Module (03) – Cao:** áp dụng các chiến lược giải mã trong mục 4.5: VAD/no-speech threshold, temperature fallback dựa trên compression ratio & logprob, condition_on_previous_text=False khi thấy lặp. Lọc đoạn có compression ratio cao (dấu hiệu lặp) trước khi index.
- **Module (03) – Cao:** với khoảng slide >30 s, cắt theo VAD; khoảng <vài giây thì gộp với lân cận để tránh hallucination.
- **Module (03) – Trung bình:** chọn kích thước (small/medium/large-v2 hoặc PhoWhisper) bằng benchmark WER + thời gian trên L25.
- **Module (04) – Thấp:** dùng language ID của Whisper để phát hiện đoạn tiếng Anh trong bài giảng.
