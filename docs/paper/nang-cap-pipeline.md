# Lộ trình nâng cấp pipeline từ 30 paper

Tổng hợp từ các file tóm tắt trong `docs/paper/` và `docs/paper/edu-rag/`. Các việc được xếp theo giá trị và độ dễ làm; mỗi việc ghi rõ paper nguồn.

## Tiến độ (cập nhật 13/10/2026)

| Mục | Trạng thái | Kết quả / ghi chú |
|---|---|---|
| Qwen2.5-7B thay qwen2:7b | ✔ áp dụng | 5 câu thử: cả hai đúng 5/5; Qwen2.5 trích `[mm:ss]` 4/5 (qwen2: 0/5), sửa lỗi ASR ("phủ trang" → "vũ trang"). Nhược: dài dòng → prompt thêm luật "1–3 câu, không vừa nói không đề cập vừa trả lời" (chưa đo lại) |
| Prompt chống bịa ("không đề cập") | ✔ có sẵn | Câu ngoài bài: cả hai model trả lời đúng "không đề cập" |
| Trích dẫn `[mm:ss]` | ✔ áp dụng | Mọi dạng Qwen viết (`[mm:ss]`, `[mm:ss–mm:ss]`, `[Slide … · mm:ss–mm:ss]`) thành 1 nút nhảy tới video; thêm dòng "Nguồn" (các slide bằng chứng) dưới mỗi câu trả lời. Chưa thử trên trình duyệt |
| Stream câu trả lời | ✔ áp dụng | `POST /api/qa/stream` (NDJSON, Ollama + Gemini), frontend hiện chữ dần. Trên CPU vẫn chờ 75–170 s tới chữ đầu (đọc ngữ cảnh); cần GPU/API để nhanh thật |
| Tiền tố `query:`/`passage:` E5 | ✔ đúng | |
| Viết lại câu hỏi nối tiếp | ✔ áp dụng (không gọi LLM) | Frontend gửi câu hỏi liền trước; câu ngắn có từ chỉ trỏ ("nó", "đó", "còn…", "what about", so có dấu để không nhầm "độ", "con", "thế nào") được tìm cùng câu trước, và câu trước được đưa vào prompt. 3 test mới. Chưa đo trên bộ câu hỏi nối tiếp |
| Từ khóa OCR vào `initial_prompt` Whisper | ~ thay thế | Đã có sửa chữ sau ASR bằng từ vựng OCR (`correct_words`); `initial_prompt` có tham số nhưng chưa truyền |
| VAD | ✔ có | `vad_filter=True` |
| Bộ câu hỏi | ✔ 74 câu / 6 video | +48 câu (V011, V037, V017, hVU5Mw19V_c) gắn nhãn nguồn speech/ocr/both; **do Claude soạn, chưa người kiểm**; chưa gắn nhãn mức độ |
| Ablation 3 cấu hình | ✔ đã đo | 66 câu có đáp án: chỉ lời giảng hit@1 50 · hit@3 59; chỉ slide 49 · 59; cả hai 50 · 64. Sau khi cho chữ slide trọng số ×1,05 (`SOURCE_BOOST`): **cả hai 55 · 65**, câu chỉ có trên slide hit@1 6 → 9/14. Đã thử và bỏ: ×1,1–1,5, xếp hạng riêng từng nguồn, cộng điểm theo slide (đều kém hơn) |
| RAGAS | ✗ chưa | Cần LLM nhanh (GPU/Gemini) |
| So với LLM không RAG | ✗ chưa | |
| F1 LPM / CER M3AV | ~ một phần | Bench OCR 40 slide M3AV: Qwen3-VL 83% từ, hybrid 84% |
| PhoWhisper vs Whisper | ✗ chưa | |
| BGE-M3 | ✗ đo lại, giữ E5 | Bench cũ (22 câu, V008 + MLS): bge-m3 21/22 > E5 19/22. Đo lại trên 66 câu / 6 video (ONNX, CPU): **bge-m3 hit@1 53 · hit@3 62, E5 55 · 65**; thua rõ ở hVU5Mw19V_c (5 vs 7/11) và V017. Ngưỡng "khớp yếu" 0,85 của LLM cũng chỉ đúng cho E5 (cos bge thấp hơn hẳn). Đã thêm: encoder ONNX chung (`OnnxEncoder`, chọn theo manifest chỉ mục), `scripts/reembed_index.py` (đổi/khôi phục vector), bước 05 tự bỏ tiền tố khi không phải E5 → đổi model sau này không phải sửa code. Report đã sửa về E5 + thêm slide ablation; deck sắp lại cho buổi báo cáo đầu tiên: 33 slide chính + 31 phụ lục (13/10) |
| Phát hiện chữ lặp OCR | ✔ có | `collapse_loops`, `collapse_phrase_loop` |
| GOT-OCR2 | ✗ bỏ | Bench: 24% từ tiếng Việt |
| Quét ngưỡng SigLIP | ✔ chốt | 2.917 khung: thêm 6 câu mô tả, bỏ nhầm 34 → 7, bỏ được khung trống 76% → 81%; không dùng bộ dò chữ |
| Truy xuất bằng hình (ColPali) | ✗ không làm | Quyết định giữ chữ làm trục chính |

**Ngoài lộ trình, đã làm thêm:** tăng tốc OCR bước 02 (chạy chồng 2 GPU, gom lô, dùng lại slide gần trùng: ×1,4–1,5, chữ giữ nguyên); bench 6 công cụ OCR (Qwen3-VL đọc cả slide tốt nhất tiếng Việt 94% từ, nhanh ×2 hybrid); so sánh trên 4 video (2 VI, 2 EN): Qwen nhanh hơn hybrid ×1,8–7, chữ sạch hơn → **chốt `OCR_ENGINE = 'qwen'`** (13/10), hybrid giữ làm tùy chọn.

**Việc tiếp theo:** chạy lại 10 video L25 → kiểm 48 câu hỏi → RAGAS + so với LLM không RAG → cập nhật report (OCR, LLM, stream) → deploy Vast.

## Giai đoạn 1: Làm ngay (rẻ, hiệu quả rõ, ~1 tuần)

### Bước 05, khâu LLM trả lời (đang yếu nhất)
| Nâng cấp | Paper nguồn | Lý do |
|---|---|---|
| Thay qwen2:7b bằng **Qwen2.5-7B-Instruct**, ép trả lời tiếng Việt | Qwen2.5-VL | Giảm bịa và lẫn ngôn ngữ, đúng lỗi đã thấy khi chạy thử |
| **Prompt 2 bước**: kiểm tra ngữ cảnh có đủ không, không đủ thì trả lời "bài giảng không đề cập" | AI-University, RAG | Chống bịa |
| **Trích dẫn `[video, mm:ss]`** rồi biến thành link nhảy tới đúng chỗ trong video | LectureFeedback, VideoLibrary-RAG | Sinh viên kiểm chứng được, thế mạnh riêng của RAG trên video |
| **Stream** câu trả lời trên web app | VirtualTA-KAIST | Khi khảo sát thật, "chậm" là phàn nàn số 1. LLM hiện mất 20–160 giây mỗi câu |

### Bước 05, khâu truy xuất
| Nâng cấp | Paper nguồn |
|---|---|
| Kiểm tra code dùng đúng tiền tố `query:` / `passage:` của E5, đoạn ≤ 512 token | mE5 |
| **Viết lại câu hỏi nối tiếp** ("còn cái kia thì sao?") thành câu độc lập trước khi tìm | VirtualTA, CourseRAG |

### Bước 03, ASR
| Nâng cấp | Paper nguồn |
|---|---|
| **Đưa từ khóa OCR của slide vào `initial_prompt`** của Whisper. Không cần huấn luyện, giúp nhận đúng thuật ngữ | SlideSpeech, M3AV |
| **VAD** (bỏ khoảng lặng) và gộp khoảng quá ngắn, để ASR không bịa chữ ở 194 đoạn không có lời giảng | Whisper |

## Giai đoạn 2: Bộ đánh giá (bắt buộc cho báo cáo, ~1 tuần)
Đây là điểm mà paper nào cũng có còn đề tài đang thiếu.
1. **Bộ 50–100 câu hỏi** cho 10 video L25 có nhãn. Gắn nhãn câu hỏi theo hai kiểu:
   - mức độ: chi tiết / một phần bài / cả chủ đề (Classroom-RAG-Comparative);
   - nguồn bằng chứng: chữ, bảng, hình hay lời giảng (M3DocRAG).
2. **RAGAS**: đo faithfulness, answer relevancy, context precision/recall (RAGAS, CourseGPT, LEA).
3. **Ablation 3 cấu hình**: chỉ transcript, chỉ OCR, và OCR + transcript đã ghép (AlignedVideoCaptions). Đây là cách **chứng minh giá trị "đa phương thức"** của đề tài.
4. **So với LLM không dùng RAG** (Tanner 2025, RAG).
5. Kiểm tra pipeline trên dữ liệu ngoài: **F1 tách slide trên LPM**, **CER OCR trên M3AV**, tách riêng công thức (LPM, M3AV).

## Giai đoạn 3: Nâng chất lượng các module (~2 tuần)
| Module | Nâng cấp | Paper nguồn |
|---|---|---|
| 03 ASR | So sánh **PhoWhisper với Whisper** trên L25 (WER/CER và thời gian) | PhoWhisper |
| 05 Truy xuất | Benchmark **BGE-M3** (dense + sparse) so với E5 + BM25 | BGE-M3 |
| 02 OCR | **Phát hiện chữ lặp** ở đầu ra VLM (đếm n-gram lặp) | Nougat |
| 02 OCR | Thử **GOT-OCR2 riêng cho vùng công thức và bảng** | GOT-OCR2 |
| 02 Gate | Chọn ngưỡng SigLIP bằng **quét ngưỡng trên frame có nhãn** | SigLIP |

## Giai đoạn 4: Hướng mở rộng (nếu còn thời gian, có thể đưa vào "hướng phát triển")
- **Truy xuất bằng hình**: tìm thẳng trên keyframe bằng ColPali/ColQwen, rồi trộn điểm với E5 và BM25. Cách này bỏ qua lỗi dấu của OCR (ColPali, VisRAG).
- **Đưa ảnh slide cho VLM khi trả lời**: đồ thị và sơ đồ được hiểu trực tiếp thay vì qua chữ OCR (VisRAG, Video-RAG).
- **Khảo sát 10–15 sinh viên IUH**: thang Likert/SUS và 4 tiêu chí của KAIST (hữu ích, đáng tin, phù hợp, thoải mái) (VirtualTA, LEA, CourseRAG).

## Thứ tự đề xuất
**LLM tốt hơn + trích dẫn → bộ câu hỏi + RAGAS + ablation → ASR có prompt từ OCR → PhoWhisper / BGE-M3 → khảo sát người dùng.**

Hai việc đầu đáng làm trước nhất. Chúng sửa thẳng điểm yếu đang thấy rõ (bịa, lẫn tiếng) và cho ra **con số đưa vào báo cáo** mà hội đồng chắc chắn sẽ hỏi.
