# Sigmoid Loss for Language Image Pre-Training (ICCV 2023, arXiv:2303.15343)

## Tóm tắt
- **Bài toán:** huấn luyện tương phản ảnh–văn bản (CLIP) dùng softmax trên toàn batch: tốn bộ nhớ, cần giao tiếp toàn cục, phụ thuộc batch size lớn.
- **Phương pháp:** thay softmax bằng **loss sigmoid theo từng cặp** ảnh–văn bản (bài toán nhị phân khớp/không khớp), có nhiệt độ và bias học được. Không cần chuẩn hoá toàn batch, triển khai "chunked" tiết kiệm bộ nhớ. Áp dụng cho CLIP (SigLIP) và LiT – khoá encoder ảnh (SigLiT). Có bản đa ngôn ngữ mSigLIP (hơn 100 ngôn ngữ).
- **Dữ liệu:** WebLI, dữ liệu LiT; đánh giá ImageNet zero-shot, COCO retrieval, XM3600 (36 ngôn ngữ).
- **Kết quả:** SigLiT g/14 đạt 84,5% ImageNet zero-shot với 4 chip TPUv4 trong 2 ngày; SigLiT B/8 79,7% trong 1 ngày. SigLIP B/16 từ đầu 73,4% với 32 TPUv4 trong 5 ngày. Sigmoid tốt hơn rõ softmax khi batch < 16k; lợi ích bão hoà, batch 32k là đủ (thử tới 1 triệu). mSigLIP ViT-B đạt SOTA XM3600: 42,6% image R@1 và 54,1% text R@1. Sigmoid bền vững hơn với nhiễu dữ liệu.

## Điểm mạnh
- Ý tưởng đơn giản, dễ cài đặt, tiết kiệm bộ nhớ, giảm chi phí huấn luyện.
- Phân tích hệ thống ảnh hưởng của batch size, tỉ lệ âm/dương, nhiễu.
- Mô hình mở (big_vision, HuggingFace), có bản đa ngôn ngữ; đầu ra sigmoid cho điểm xác suất độc lập từng cặp → hợp cho phân loại zero-shot có ngưỡng.

## Điểm yếu / hạn chế
- Tập trung ảnh tự nhiên; không đánh giá ảnh chứa nhiều chữ (slide, tài liệu) hay đọc chữ.
- Kết quả quy mô lớn phụ thuộc dữ liệu WebLI nội bộ của Google.
- Lợi thế giảm khi batch rất lớn; không cải thiện khả năng hiểu chi tiết (OCR, công thức).

## So sánh với đề tài của tui
- **Giống:** đề tài dùng SigLIP làm "cổng" zero-shot ở module (02) để bỏ qua frame không phải slide.
- **Khác:** đề tài chỉ dùng mô hình đã huấn luyện cho phân loại nhị phân slide/không-slide; bài báo về phương pháp pre-training.
- **Đề tài hơn:** có bài toán ứng dụng cụ thể và đo tác động lên tốc độ pipeline (OCR ~35 s/slide nên bỏ frame thừa rất có giá trị).
- **Đề tài kém:** ngưỡng sigmoid/prompt chọn theo kinh nghiệm; cần báo cáo precision/recall của gate trên gold.

## Có áp dụng được vào đề tài không?
- **Module (02) gate – Cao:** tận dụng đầu ra sigmoid (xác suất độc lập) để đặt ngưỡng có căn cứ; quét ngưỡng (threshold sweep) trên frame gán nhãn, báo cáo P/R/F1 và số giây OCR tiết kiệm. Thử nhiều prompt ("a presentation slide", "a person talking", "a screen with code") rồi ensemble.
- **Module (02) – Trung bình:** dùng embedding ảnh SigLIP để phát hiện slide trùng/gần trùng, tránh OCR lặp lại (giảm thời gian).
- **Truy xuất ảnh slide đa phương thức – Thấp:** mSigLIP có thể index ảnh slide để tìm bằng câu hỏi, nhưng tiếng Việt và văn bản dày chữ chưa được bài đánh giá → ưu tiên thấp.
