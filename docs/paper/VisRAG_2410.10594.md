# VisRAG: Vision-based Retrieval-Augmented Generation on Multi-modality Documents (ICLR 2025, arXiv:2410.10594)

## Tóm tắt
- **Bài toán:** RAG truyền thống (TextRAG) phải phân tích tài liệu thành văn bản (nhận dạng bố cục, OCR, ghép chữ). Quá trình này sinh lỗi và bỏ mất thông tin bố cục, hình ảnh trong tài liệu đa phương thức như sách giáo khoa, sổ tay.
- **Phương pháp:** VisRAG gồm hai phần. **VisRAG-Ret** nhúng trực tiếp ảnh trang tài liệu bằng VLM (MiniCPM-V làm nền) theo kiểu dual-encoder, huấn luyện bằng dữ liệu mở và dữ liệu tổng hợp (câu hỏi sinh từ PDF thu thập trên web). **VisRAG-Gen** đưa ảnh trang truy hồi được cho VLM sinh câu trả lời; tác giả thử nhiều cách xử lý nhiều trang (ghép ảnh, chọn trang theo trọng số, VLM nhận nhiều ảnh).
- **Baseline TextRAG:** hai cách trích văn bản: "(OCR)" dùng PPOCR rồi ghép các hộp chữ gần nhau; "(Captioner)" dùng MiniCPM-V 2.0 tinh chỉnh để chép chữ từ ảnh.
- **Dữ liệu và chỉ số:** tập xây từ các bộ VQA mở (như ArxivQA, ChartQA, DocVQA, InfoVQA, PlotQA, SlideVQA) và dữ liệu tổng hợp; truy hồi đo bằng MRR@10, Recall@10; sinh đo bằng độ chính xác (exact match nới lỏng, cho phép sai số 5% với số).
- **Kết quả chính:** độ chính xác toàn pipeline tăng từ 37.97% (TextRAG) lên 53.32% (VisRAG) khi dùng MiniCPM-V làm bộ sinh (tương đối khoảng 40%), và từ 43.54% lên 52.44% với GPT-4o (khoảng 20%). VisRAG-Ret vượt các bộ truy hồi văn bản và thị giác mạnh nhất, hiệu quả dữ liệu huấn luyện tốt và tổng quát hóa tốt; lợi ích tăng khi VLM xử lý được nhiều ảnh.

## Điểm mạnh
- Loại bỏ mất mát thông tin do khâu phân tích tài liệu, cải thiện cả truy hồi lẫn sinh.
- So sánh công bằng với hai biến thể TextRAG (OCR và captioner) cùng điều kiện.
- Mã nguồn và dữ liệu công khai; phân tích độ bền trên cả tài liệu thiên về chữ và thiên về hình.

## Điểm yếu / hạn chế
- Cần VLM ở cả hai bước nên chi phí suy luận cao hơn LLM văn bản.
- Số ảnh đưa vào bộ sinh bị giới hạn; tổng hợp nhiều trang vẫn là thách thức.
- Dữ liệu chủ yếu tiếng Anh; không có thành phần âm thanh/lời nói.

## So sánh với đề tài của tui
- **Giống:** cùng xử lý tài liệu đa phương thức có slide (SlideVQA nằm trong tập đánh giá); đề tài cũng dùng VLM (Qwen3-VL) để "chép" vùng khó như công thức, bảng, sơ đồ, gần với baseline Captioner.
- **Khác:** đề tài là TextRAG có OCR lai và thêm transcript lời giảng; VisRAG giữ nguyên ảnh xuyên suốt.
- **Đề tài hơn:** có kênh lời giảng (ASR) và căn chỉnh thời gian, điều VisRAG không có; chạy được trên LLM 7B cục bộ.
- **Đề tài kém:** chịu đúng loại mất mát mà bài này đo được (lỗi OCR, đặc biệt dấu tiếng Việt), và bộ sinh không nhìn thấy slide.

## Có áp dụng được vào đề tài không?
- **Module 05 – sinh câu trả lời (Cao):** khi truy hồi được slide, đưa kèm ảnh keyframe cho một VLM (ví dụ Qwen-VL) cùng văn bản OCR + transcript. Cách này có thể giảm ảnh hưởng của lỗi dấu OCR và câu hỏi về hình vẽ. Thử nghiệm A/B trên tập câu hỏi L25.
- **Đánh giá (Trung bình):** lặp lại thí nghiệm "TextRAG (OCR) vs TextRAG (Captioner) vs ảnh" trên slide bài giảng tiếng Việt để định lượng mất mát do OCR.
- **Module 05 – truy hồi bằng ảnh (Thấp–Trung bình):** VisRAG-Ret là một lựa chọn thay ColPali cho nhánh truy hồi thị giác, nhưng cần kiểm tra khả năng tiếng Việt.
