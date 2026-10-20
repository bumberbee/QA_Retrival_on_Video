# SlideSpeech: A Large Scale Slide-Enriched Audio-Visual Corpus (ICASSP 2024, arXiv:2309.05396)

Tác giả: Haoxu Wang, Fan Yu, Xian Shi, Yuezhang Wang, Shiliang Zhang, Ming Li (Wuhan Univ., Duke Kunshan, Alibaba DAMO).

## Tóm tắt
- **Bài toán:** ASR đa phương thức thường chỉ dùng hình môi hoặc ngữ cảnh, ít khi tận dụng chữ trên slide chiếu đồng bộ với lời nói. Bài báo dùng slide để giúp ASR nhận đúng thuật ngữ và danh từ riêng.
- **Dữ liệu:** 1.705 video YouTube (hội nghị/online), hơn 1.000 giờ.
  - Tập train L gồm 1.659 video, 1.065,86 giờ. L95 (độ tin cậy trên 95%) còn khoảng 473 giờ. Tập con S có khoảng 206 giờ, S95 khoảng 161 giờ.
  - Dev (21 video, 5,07 giờ) và test (25 video, 8,75 giờ) được gán nhãn thủ công. 94% đoạn trong dev/test có slide.
  - Nhãn train tạo bằng VAD + ASR nội bộ, rồi căn với phụ đề YouTube bằng Smith-Waterman. Độ tin cậy được tính là 1 − WER.
  - Có 22 nhóm chủ đề. Bộ dữ liệu cung cấp sẵn kết quả OCR và từ khóa cho từng đoạn.
- **Phương pháp benchmark:** lấy khung hình giữa mỗi đoạn lời nói → phát hiện chữ + OCR (MMOCR) → sắp lại chữ theo tọa độ → trích từ khóa bằng KeyBERT (danh sách 50 từ) → contextual ASR (Conformer CTC/AED + CPP network, bias encoder BLSTM, cross-attention).
- **Kết quả:**
  - Baseline chỉ dùng âm thanh, train trên L95: WER dev/test 13,09/12,89.
  - Thêm từ khóa slide: WER còn 12,64/12,38. B-WER trên danh sách từ khóa giảm từ 16,13/12,70 xuống 12,39/9,32, recall tăng lên 87,64%/90,86%.
  - Với S95: WER từ 21,05/21,22 xuống 20,80/20,95.

## Điểm mạnh
- Quy mô lớn, slide đồng bộ theo thời gian thực với lời nói. Có OCR và từ khóa sẵn cho từng đoạn.
- Pipeline đơn giản, dễ tái hiện: khung hình giữa → OCR → từ khóa → bias ASR.
- Có các chỉ số U-WER, B-WER và recall để tách riêng hiệu quả trên thuật ngữ.

## Điểm yếu / hạn chế
- Mức cải thiện WER tổng thể nhỏ (khoảng 0,5 điểm tuyệt đối). Lợi ích tập trung ở từ hiếm.
- Nhãn train là nhãn tự động. Dev/test nhỏ (khoảng 14 giờ) và thuộc chủ đề khác với train.
- Chỉ có tiếng Anh. Không có ranh giới slide hay ảnh slide được gán nhãn. Không có tác vụ hiểu nội dung, hỏi đáp hay RAG.
- Chỉ dùng khung hình giữa đoạn, nên bỏ sót thay đổi slide bên trong đoạn. Bài báo không phân tích lỗi OCR.

## So sánh với đề tài của tui
- **Giống:** cùng kết hợp OCR slide và ASR theo thời gian, cùng xử lý video bài giảng trên YouTube.
- **Khác:** SlideSpeech dùng slide để cải thiện ASR. Đề tài dùng cả hai làm nguồn tri thức cho RAG. Đề tài phát hiện ranh giới slide thực sự (F1 0,970) thay vì lấy khung hình giữa đoạn VAD.
- **Đề tài hơn:** OCR bằng VLM xử lý được công thức, bảng và sơ đồ. Có retrieval và LLM trả lời. Xử lý tiếng Việt.
- **Đề tài kém:** ASR của đề tài chưa được hỗ trợ bởi ngữ cảnh slide. Đề tài chưa có số liệu B-WER cho thuật ngữ.

## Có áp dụng được vào đề tài không?
- **Bước 03, ASR (Cao):** sau bước 02, trích khoảng 20–50 từ khóa từ OCR của slide (KeyBERT hoặc TF-IDF, không cần huấn luyện) và đưa vào prompt hoặc hotwords của Whisper cho khoảng thời gian của slide đó. Đo WER và B-WER trước/sau trên vn12 và tập con SlideSpeech. Cách này rất hợp với thuật ngữ Anh–Việt trộn lẫn trong bài giảng.
- **Đánh giá (Trung bình):** dùng dev/test của SlideSpeech (nhãn thủ công) làm thêm một tập đo ASR tiếng Anh, báo cáo cả U-WER/B-WER theo định nghĩa của bài báo.
- **Bước 04 (Thấp):** kỹ thuật căn chuỗi Smith-Waterman có thể dùng để căn transcript ASR với phụ đề sẵn có khi xây gold cho L25.
