# Nougat: Neural Optical Understanding for Academic Documents (Meta AI, 2023, arXiv:2308.13418)

Tác giả: Lukas Blecher, Guillem Cucurull, Thomas Scialom, Robert Stojnic (Meta AI).

## Tóm tắt
- **Bài toán:** PDF khoa học làm mất ngữ nghĩa, nhất là công thức toán. OCR theo dòng như Tesseract hiểu sai chỉ số trên/dưới. Mục tiêu là chuyển ảnh trang tài liệu thành markup (Markdown + LaTeX) mà chỉ cần ảnh đầu vào.
- **Phương pháp:** encoder–decoder end-to-end theo Donut. Encoder là Swin Transformer với ảnh đầu vào 896×672 (96 DPI). Decoder kiểu BART dài tối đa 4.096 token, sinh bằng greedy decoding.
  - Bản base có 350M tham số. Bản small có 250M tham số với 4 lớp decoder.
  - Huấn luyện có augmentation mô phỏng bản scan (erosion, nhiễu, blur, biến dạng lưới...). Token đáp án bị thay ngẫu nhiên khi huấn luyện để giảm hiện tượng lặp.
- **Dữ liệu:** tự xây cặp trang PDF ↔ mã nguồn, gồm 1.748.201 bài arXiv cùng một phần PMC và IDL (IDL chỉ dùng để pretrain). Trang được căn với mã nguồn bằng chữ nhúng trong PDF, chỉ khoảng 47% số trang đạt ngưỡng.
- **Kết quả (arXiv test):** Nougat base đạt edit distance 0,071, BLEU 89,1, F1 93,1. Trích chữ nhúng PDF đạt 0,255 / 65,8 / 79,2, GROBID đạt 0,312 / 55,6 / 73,0. Trên phần công thức, Nougat đạt edit distance 0,128, F1 76,5, còn GROBID + LaTeX-OCR chỉ đạt F1 9,7. Bản small (0,073) gần bằng bản base.
- **Lặp vô hạn:** xảy ra ở 1,5% trang test và nhiều hơn với tài liệu ngoài miền. Bài báo đề xuất phát hiện lặp bằng phương sai của logit trên cửa sổ trượt (B = 15, ngưỡng 6,75). Augmentation chống lặp giảm 32% số trang chuyển đổi thất bại với tài liệu ngoài miền.

## Điểm mạnh
- Mã nguồn và mô hình được công bố. Chỉ cần ảnh trang nên dùng được với bản scan.
- Đọc công thức và bảng ra LaTeX tốt hơn hẳn các công cụ truyền thống.
- Có cách thực tế để phát hiện và cắt sớm khi mô hình bị lặp.

## Điểm yếu / hạn chế
- Dữ liệu gần như toàn tiếng Anh. Với ngôn ngữ Latin khác, ký tự đặc biệt bị thay bằng chữ Latin gần nhất. **Với tiếng Việt, dấu thanh sẽ bị mất**, đúng loại lỗi đề tài đang gặp. Chữ không phải Latin thì gây lặp ngay.
- Chậm: trên A10G 24 GB xử lý 6 trang song song, trung bình 19,5 s cho mỗi batch (khoảng 1.400 token/trang), trong khi GROBID đạt 10,6 PDF/s.
- Mỗi trang xử lý độc lập nên tài liệu ghép lại có thể thiếu nhất quán, có khi đánh số sai hoặc sinh ra nội dung không có thật.
- Huấn luyện cho trang bài báo dọc, không phải slide ngang hay khung hình video.

## So sánh với đề tài của tui
- **Giống:** cùng cần chuyển công thức và bảng thành văn bản có cấu trúc. Lỗi lặp của Nougat cũng là lỗi thường gặp ở OCR bằng VLM sinh tự hồi quy mà đề tài đang dùng.
- **Khác:** Nougat là một mô hình nhỏ chuyên cho trang PDF. Đề tài dùng hybrid Unlimited-OCR + Qwen3-VL theo từng vùng trên khung hình slide tiếng Việt.
- **Đề tài hơn:** đọc được tiếng Việt, xử lý ảnh slide từ video, có thêm mô tả sơ đồ.
- **Đề tài kém:** chậm hơn (khoảng 35 s/slide so với khoảng 3 s/trang của Nougat theo batch), chưa có cơ chế phát hiện lặp, chưa đo riêng phần công thức.

## Có áp dụng được vào đề tài không?
- **Bước 02, chống lặp (Cao):** thêm bộ phát hiện lặp cho đầu ra Qwen3-VL/Unlimited-OCR, bằng phương sai logit hoặc đơn giản hơn là đếm n-gram lặp. Khi phát hiện thì dừng sớm và gọi lại với `repetition_penalty`. Cách này vừa giảm thời gian vừa giảm rác trong chỉ mục.
- **Bước 02, vùng công thức (Trung bình):** có thể thử Nougat-small (250M) chỉ cho vùng công thức, nơi không có dấu tiếng Việt, để so sánh với Qwen3-VL về tốc độ và edit distance trên block công thức của M3AV.
- **Đánh giá (Trung bình):** dùng bộ chỉ số edit distance, BLEU, METEOR, F1 tách theo loại văn bản (chữ thường, công thức, bảng) để báo cáo OCR.
- **Thay OCR chính (Thấp):** không nên, vì Nougat làm mất dấu tiếng Việt.
