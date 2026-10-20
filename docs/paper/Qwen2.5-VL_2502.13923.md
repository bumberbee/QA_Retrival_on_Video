# Qwen2.5-VL Technical Report (Qwen Team, Alibaba, 2025, arXiv:2502.13923)

## Tóm tắt
- **Bài toán:** LVLM hiện có còn yếu ở nhận thức chi tiết (định vị, đọc tài liệu), tốn nhiều tính toán với ảnh độ phân giải cao và chưa hiểu tốt video dài. Qwen2.5-VL là thế hệ mới của dòng Qwen-VL nhắm vào các điểm này.
- **Phương pháp:**
  1. ViT huấn luyện lại từ đầu, dùng độ phân giải gốc (cạnh ảnh làm tròn lên bội số của 28, patch 14) với 2D-RoPE và **window attention**. Chỉ 4 lớp (7, 15, 23, 31) dùng full attention, nhờ đó giảm chi phí tính toán.
  2. Bộ trộn MLP gom 4 patch kề nhau để giảm số token đưa vào LLM.
  3. Video: lấy mẫu FPS động. **MRoPE gắn với thời gian tuyệt đối** giúp xác định sự kiện tới cấp giây trong video dài nhiều giờ.
  4. Dữ liệu pretrain tăng từ 1,2 lên khoảng 4,1 nghìn tỷ token. Có định dạng "QwenVL HTML" để phân tích toàn bộ bố cục tài liệu. Dữ liệu OCR đa ngôn ngữ có **tiếng Việt**. Định vị đối tượng dùng tọa độ tuyệt đối.
  
  Có ba kích cỡ: 3B, 7B, 72B, khởi tạo từ Qwen2.5 LLM.
- **Kết quả (72B / 7B / 3B):**
  - CC-OCR: 79,8 / 77,8 / 74,5, so với GPT-4o 73,0.
  - OmniDocBench edit distance (en/zh): 0,226/0,324 cho bản 72B và 0,308/0,398 cho bản 7B.
  - DocVQA: 96,4 / 95,7 / 93,9.
  - OCRBench: 885 / 864 / 797.
  - ChartQA: 89,5 / 87,3 / 84,0.
  - OCRBench_v2 en/zh: 61,5/63,7 cho bản 72B.
  - MMMU (72B): 70,2.
  - Video-MME không/có phụ đề (72B): 73,3/79,1.
  - LVBench (72B): 47,3, so với GPT-4o 30,8.
  - Charades-STA mIoU: 50,9 (72B) và 43,6 (7B).
  - MTVQA đa ngôn ngữ (72B): 31,7.

## Điểm mạnh
- Mô hình mở. Bản 7B đã rất mạnh về OCR và tài liệu: OCRBench 864, gần bằng bản 72B.
- Có tiếng Việt trong dữ liệu OCR. Đọc được bảng, công thức, biểu đồ và xuất HTML có bố cục.
- Định vị bằng tọa độ tuyệt đối, phù hợp để tách vùng. Gắn thời gian tuyệt đối cho video, phù hợp với video bài giảng dài.

## Điểm yếu / hạn chế
- Báo cáo kỹ thuật không có phần hạn chế riêng. Không có số liệu riêng cho OCR tiếng Việt hay cho slide.
- Bản 72B quá nặng để chạy cục bộ. Các bản nhỏ giảm rõ ở bài khó, ví dụ CharXiv RQ của bản 3B chỉ đạt 31,3.
- Là mô hình sinh nên vẫn có thể bịa nội dung. Báo cáo không nêu số liệu tốc độ suy luận cụ thể.

## So sánh với đề tài của tui
- **Giống:** đề tài dùng Qwen3-VL, thế hệ kế tiếp của dòng này, cho OCR theo vùng và mô tả sơ đồ. Kỹ thuật định vị, phân tích tài liệu và mô hình tiếng Việt là nền tảng chung.
- **Khác:** đề tài tách video bằng pipeline riêng (phát hiện slide → OCR → ASR → căn chỉnh) thay vì đưa thẳng video vào VLM. Đề tài cũng thêm retrieval và LLM qwen2:7b để trả lời.
- **Đề tài hơn:** chi phí thấp hơn (không cần mô hình 72B), có chỉ mục tìm kiếm được, trích dẫn được slide và thời điểm.
- **Đề tài kém:** VLM trong đề tài chưa được dùng cho định vị hay phân tích bố cục HTML trọn trang. LLM trả lời (qwen2:7b) cũ hơn và dễ trộn ngôn ngữ.

## Có áp dụng được vào đề tài không?
- **Bước 02, layout + OCR một lượt (Cao):** dùng prompt định vị hoặc QwenVL HTML của Qwen2.5/3-VL-7B để tìm và đọc mọi vùng trong một lần gọi thay vì gọi riêng cho từng vùng. Chỉ gọi thêm cho vùng sơ đồ cần mô tả. Mục tiêu là giảm 35 s/slide. Giới hạn `max_pixels` để kiểm soát số token.
- **Bước 05, LLM trả lời (Cao):** thay qwen2:7b bằng Qwen2.5-7B-Instruct (cùng họ, mạnh hơn), kèm system prompt bắt buộc trả lời bằng tiếng Việt và chỉ dựa trên ngữ cảnh. Việc này nhắm thẳng vào lỗi bịa nội dung và trộn ngôn ngữ.
- **Bước 01/04, kiểm tra chéo (Thấp):** dùng khả năng định vị thời gian trong video của bản 7B để kiểm tra ranh giới slide trên vài video. Chi phí cao nên chỉ dùng để đối chứng, không đưa vào pipeline chính.
