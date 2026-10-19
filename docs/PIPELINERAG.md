# Pipeline RAG hội thoại nhiều lượt: bài báo và hướng giải quyết

## Hiện trạng

- Chỉ nhớ **câu hỏi liền trước**: câu ngắn có từ chỉ trỏ ("nó", "còn … thì sao") được ghép với câu trước để
  tìm kiếm; các lượt xa hơn và câu trả lời trước **không** đưa cho LLM.
- **Không lưu lịch sử**: tin nhắn chỉ ở bộ nhớ trình duyệt, tải lại trang là mất, không mở lại được phiên cũ.
- LLM: Qwen2.5-7B-Instruct (Ollama, 4-bit), không LoRA/fine-tune; tìm kiếm E5 + BM25 + RRF, lấy 3 slide.

## Các bài nền tảng (đã kiểm tên, tác giả, nơi đăng trên arXiv)

| Bài | Nơi đăng | Ý chính áp dụng được |
|---|---|---|
| **LLM4CS** — *Large Language Models Know Your Contextual Search Intent* (Mao và cs.) · [arXiv:2303.06573](https://arxiv.org/abs/2303.06573) | Findings EMNLP 2023 | LLM **viết lại câu hỏi nối tiếp** thành câu độc lập (sinh vài bản rồi gộp), vượt cả câu viết lại tay trên TREC CAsT 19–21 |
| **ChatQA** — *Surpassing GPT-4 on Conversational QA and RAG* (NVIDIA) · [arXiv:2401.10225](https://arxiv.org/abs/2401.10225) | NeurIPS 2024 | Hỏi đáp hội thoại có truy hồi; bộ đánh giá ChatRAG Bench |
| **QReCC** — *Open-Domain QA Goes Conversational via Question Rewriting* (Apple) · [arXiv:2010.04898](https://arxiv.org/abs/2010.04898) | NAACL 2021 | 14 nghìn hội thoại; **viết lại câu hỏi là nút thắt** của truy hồi hội thoại (máy 19,1 F1, người 75,5) |
| **TopiOCQA** — *Open-domain Conversational QA with Topic Switching* · [arXiv:2110.00768](https://arxiv.org/abs/2110.00768) | TACL 2022 | Người dùng **đổi chủ đề** giữa hội thoại; hệ thống tốt nhất vẫn kém người 14,2 điểm F1 |
| **MTRAG** — *A Multi-Turn Conversational Benchmark for RAG* (IBM) · [arXiv:2501.03468](https://arxiv.org/abs/2501.03468) | arXiv 2025 | 110 hội thoại, 842 lượt; RAG kém rõ ở **lượt sau**, câu **không tự đứng được**, câu **không trả lời được** — mẫu để thiết kế bộ test |
| **Recursively Summarizing Enables Long-Term Dialogue Memory** · [arXiv:2308.15022](https://arxiv.org/abs/2308.15022) | Neurocomputing (arXiv 2023) | **Tóm tắt dần**: tóm tắt cũ + lượt mới → tóm tắt mới, giúp trả lời nhất quán khi phiên dài |

Cùng hướng bộ nhớ dài hạn: MemGPT (arXiv:2310.08560), MemoryBank, Mem0, LongMemEval — với phiên hỏi đáp
bài giảng (thường < 20 lượt) **chưa cần** các hệ nhớ phức tạp này.

Chưa tìm thấy bài làm trợ lý **bài giảng video** có nhớ phiên hội thoại → có thể nêu là **điểm đóng góp**:
RAG hội thoại cho video bài giảng, trích dẫn mốc thời gian.

> Số liệu chi tiết của các bài khác (CANARD, ConvDR, CONQRR, ConvGQR, InfoCQR, TREC iKAT, CORAL, MemoryBank,
> Generative Agents, LoCoMo, LongMemEval, Mem0, A-MEM) chưa được kiểm; cần đối chiếu bài gốc trước khi trích.

## Hướng giải quyết

### 1. Phân loại câu hỏi trước khi tìm (LLM4CS, TopiOCQA)
- **Câu tự đứng được** ("Liên hợp quốc ra đời khi nào?"): giữ nguyên, **không** ghép lịch sử — ghép vào làm hỏng
  khi người dùng đã đổi chủ đề.
- **Câu phụ thuộc lượt trước** ("Còn ASEAN thì sao?", "Nó gồm những nước nào?"): Qwen **viết lại thành câu độc lập**
  từ 2 lượt gần nhất, vài ví dụ mẫu tiếng Việt, temperature 0, tối đa ~40 token; không cần thì trả nguyên câu.
- **Không viết lại kịp** (quá thời gian, lỗi): quay về ghép câu trước như hiện nay.
- Phát hiện câu phụ thuộc bằng bộ nhận dạng từ chỉ trỏ có sẵn + câu ngắn ⇒ phần lớn câu hỏi không tốn thêm lời gọi LLM.

### 2. Tìm kiếm có ngữ cảnh phiên
- Tìm bằng câu đã viết lại (E5 + BM25 + RRF như hiện tại).
- Cộng nhẹ điểm cho **slide đã trích ở lượt trước** và slide gần nó về thời gian (câu tiếp thường hỏi quanh đoạn vừa nói).
- Chưa dùng sinh câu trả lời giả rồi mới tìm (HyDE): tốn thêm vài giây mỗi lượt.

### 3. Bộ nhớ phiên cho LLM (tóm tắt dần)
- Prompt gồm **2–3 cặp hỏi–đáp gần nhất** (mỗi câu trả lời cắt ~150 token) + **bản tóm tắt phiên** < 200 token cho phần cũ.
- Tóm tắt cập nhật **chạy nền sau khi stream xong**, người dùng không phải chờ.

### 4. Lưu và mở lại phiên
- Bảng `conversations`: phạm vi video, tóm tắt, thời gian.
- Bảng `messages`: câu hỏi, **câu đã viết lại**, câu trả lời, slide và `[mm:ss]` đã trích, độ trễ.
- Lưu trên Supabase khi deploy, lưu file khi chạy trên máy.
- Mở lại phiên: nạp tóm tắt + 2 lượt cuối. Giao diện: danh sách **"Lịch sử hỏi đáp"** + nút **"Phiên mới"**.

### 5. Đánh giá (theo khung MTRAG, QReCC)
- **Bộ test tiếng Việt**: 25–30 hội thoại × 4–6 lượt, mở rộng từ bộ 74 câu. 6 kiểu: đại từ, tỉnh lược, so sánh,
  **đổi chủ đề**, **không trả lời được**, xin làm rõ. Mỗi lượt có câu viết lại tay + slide đúng.
- **So 4 cấu hình**: câu gốc / ghép chuỗi (hiện tại) / LLM viết lại / viết lại tay.
- **Chỉ số**: Hit@1, Hit@3 **riêng câu đầu và câu tiếp**; độ bám nguồn; độ đúng `[mm:ss]`; tỉ lệ từ chối đúng;
  thời gian ra chữ đầu tiên. Không dùng BLEU để chấm câu viết lại.
- Kết quả là một **bảng thực nghiệm mới** cho luận văn.

## Công sức

| Phần | Thời gian |
|---|---|
| Viết lại câu hỏi + tìm có ngữ cảnh | ~1 ngày |
| Bộ nhớ phiên + lưu và mở lại phiên | ~1–1,5 ngày |
| Bộ test hội thoại + đo | ~½–1 ngày (cần khôi phục script đánh giá từ git, commit `c8f4a98`) |
