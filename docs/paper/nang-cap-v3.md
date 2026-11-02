# Nâng cấp v3: paper mới cho từng điểm yếu của pipeline

Tài liệu này nối tiếp `nang-cap-pipeline.md`. Lộ trình cũ đã làm xong phần lớn (Qwen2.5, trích dẫn, stream, đo E5 so với BGE-M3, chốt OCR Qwen…).

Bản v3 dựa trên hai nguồn:
- **Đợt kiểm tra ngày 01/11/2026**: 5 Codex đọc code và đối chiếu với đầu ra thật của lần chạy 10 video. Lỗi kèm bằng chứng nằm trong các file `result-*.md`.
- **Bộ đo mới**: `kaggle/lecture_rag/eval/` gồm 120 câu hỏi do Codex soạn, **chưa có người kiểm**.

Bản tiếng Anh chi tiết hơn (các thí nghiệm, cách đo, tiêu chí giữ hay bỏ) nằm ở `upgrade-v3-research.md`.

**Mức độ kiểm chứng của các paper mới:** tên, mã arXiv và kết quả chính lấy từ abstract hoặc kết quả tìm kiếm, **chưa đọc toàn văn**. Trước khi trích dẫn số liệu nào vào khóa luận, cần mở bản PDF ra kiểm lại.

## Kết quả gốc (baseline, bộ 120 câu)

| Hit@1 | Hit@3 | MRR | Ngữ cảnh LLM chứa đáp án |
|---:|---:|---:|---:|
| 51,8% | 75,5% | 0,621 | 60,9% |

Các loại câu yếu nhất: code (Hit@3 25%), hình (60%), chữ trên slide (64%), thuật ngữ (67%).

## 1. Tách slide và bộ gác

**Điểm yếu:**
- Video bảng đen bỏ sót nội dung: một đoạn 650 giây chỉ có 1 ảnh.
- Bước gộp slide hiện dần gộp cả trường hợp slide bị *thay*.
- SigLIP loại nhầm khung có chữ rõ.
- Quảng cáo không bị lọc.

| Paper | Mã | Ý chính |
|---|---|---|
| SliTraNet: phát hiện chuyển slide trong video bài giảng bằng CNN | arXiv 2202.03540 | ResNet 2-D so mỗi khung với khung mốc, rồi CNN 3-D lọc lại các ứng viên chuyển slide |
| Tóm tắt bảng trắng (Kenny và cộng sự) | ICDAR 2017 | Chỉ ra rằng cách dò "thay đổi đột ngột" bỏ sót khi giảng viên viết dần lên bảng; đề xuất tóm tắt theo **vùng nội dung** |
| ColQwen2 / ColPali, kiểm chứng lại | arXiv 2505.07730 | Truy xuất thẳng trên ảnh trang; ColQwen2 hơn ColPali 4,7 nDCG@5 trên ViDoRe |

**Nên thử:**
- So sánh theo vùng nội dung cho video bảng đen.
- Với đoạn dài hơn 120 giây thì chụp thêm ảnh định kỳ.
- Bước gộp slide hiện dần: kiểm tra dòng nào bị *xóa hoặc thay*, và giữ mốc thời gian dòng mới xuất hiện.
- Thêm câu mô tả "caption có chữ" cho SigLIP.
- Bật lọc quảng cáo theo cả hình và lời (`AD_FILTER='slide_speech'`), giữ lại phần giảng xen giữa.

Bật lọc quảng cáo chạy được trên CPU. Phần tách slide cần chạy lại bước 01–02 trên Kaggle.

## 2. OCR

**Điểm yếu:**
- Đọc sai số liệu (1,5 kg thành 4,5 kg).
- Lặp chữ hoặc bịa nội dung.
- Mất thụt lề code.
- Mô tả hình không được đưa vào chỉ mục.
- Tiêu đề chỉ là dòng đầu tiên.
- Bước làm sạch xóa nhầm tiêu đề, code và công thức.

| Paper | Mã | Ý chính |
|---|---|---|
| PaddleOCR-VL (0,9B) | arXiv 2510.14528 | Tách hai bước: phân tích bố cục, rồi nhận dạng từng phần tử. Mục đích là tránh lỗi bịa và mất ổn định của VLM đọc nguyên trang. Đạt 80,0 trên olmOCR-Bench |
| PaddleOCR-VL-1.5 | arXiv 2601.21957 | 94,50% trên OmniDocBench v1.5; thêm phần đánh giá ảnh chụp "ngoài đời" |
| dots.ocr | arXiv 2512.02498 | Một VLM vừa phân tích bố cục vừa đọc chữ; 79,1 trên olmOCR-Bench, phần bảng đạt 88,3 |
| Nougat (đã có trong repo) | arXiv 2308.13418 | Phát hiện vòng lặp khi sinh chữ |

**Nên thử:**
- Sửa ngay trên CPU:
  - làm sạch chữ thận trọng hơn: không xóa dòng chỉ vì nó lặp lại nhiều; bảo vệ tiêu đề, code và công thức;
  - giữ thụt lề code;
  - phát hiện vòng lặp hoặc chữ "không có chữ" *trước khi* đưa vào chỉ mục.
- Cần Kaggle:
  - đọc lại các khung lỗi trên ảnh đã cắt vùng slide, vì hiện Qwen đọc cả khung hình và `OCR_SCALE` không áp dụng cho Qwen;
  - thử PaddleOCR-VL làm bộ đọc thứ hai cho bảng và công thức, và để đối chiếu số liệu.

## 3. ASR

**Điểm yếu:**
- Sai thuật ngữ: Bayes thành "BS", PPF thành "PPS", `len` thành "LAN".
- Lặp từ ("tổ chức" lặp 49 lần).
- Bước "sửa bằng OCR" làm sai ít nhất 16% số lần sửa.
- Có mốc thời gian từng từ không tin được.

| Paper | Mã | Ý chính |
|---|---|---|
| LCB-net: thiên vị ngữ cảnh dài cho nhận dạng tiếng nói kèm hình | arXiv 2401.06390 | OCR chữ trên slide → trích từ khóa → dùng làm danh sách từ ưu tiên cho ASR. Thử trên SlideSpeech. **Gần nhất với bài toán của mình** |
| KG-Whisper | arXiv 2406.02649 | Đưa từ khóa vào decoder của Whisper để nhận đúng thuật ngữ |
| Nghiên cứu cách "prompt" Whisper | arXiv 2406.05806 | Prompt bằng danh sách từ khóa **không phải lúc nào cũng có tác dụng như mong đợi** → phải đo, không nên tin ngay |
| Nhận từ hiếm cho Whisper không cần huấn luyện | arXiv 2502.11572 | Hướng zero-shot cho từ hiếm |
| TCPGen với Whisper | arXiv 2306.01942 | Thiên vị theo danh sách từ, giảm mạnh lỗi trên các từ được ưu tiên |
| Khảo sát lỗi bịa của Whisper | arXiv 2501.11378 | Phân tích lỗi bịa trên đoạn không có tiếng nói, phát hiện và loại vòng lặp |
| Phát hiện lỗi bịa của Whisper | arXiv 2606.23060 | Dò trạng thái bên trong decoder là cách tốt nhất khi không có bản chép chuẩn |
| Cơ chế lặp trong Whisper | arXiv 2508.15882 | Việc lặp do vài đầu cross-attention điều khiển |
| VietASR | arXiv 2505.21527 | ASR tiếng Việt mức công nghiệp chỉ với 50 giờ có nhãn; Whisper large-v3 có WER trung bình 16,44 trên bộ đo của họ |
| ChunkFormer-Large-Vie | ICASSP 2025 | Model card (do tác giả tự báo): WER trung bình 8,31, PhoWhisper-large 8,85, Whisper large-v3 14,89 |
| PhoWhisper (đã có trong repo) | arXiv 2406.02555 | |
| HyPoradise; GenSEC; ClozeGER | arXiv 2309.15701; 2409.09785; 2405.10025 | Dùng LLM sửa lỗi ASR từ N giả thuyết tốt nhất (N-best); đây là cách thay cho bước "sửa bằng OCR" hiện tại |

**Nên thử:**
- Trên CPU, làm ngay: tắt bước sửa bằng OCR, hoặc chỉ cho sửa theo danh sách thuật ngữ đã duyệt; phát hiện vòng lặp và mốc thời gian bất thường.
- Cần Kaggle:
  - truyền danh sách thuật ngữ từ OCR vào `initial_prompt`, theo LCB-net và KG-Whisper; phải đo vì 2406.05806 cho thấy prompt không phải lúc nào cũng có tác dụng;
  - so sánh PhoWhisper, ChunkFormer và Whisper trên vài phút audio đã có bản chép tay.

## 4. Đồng bộ và chia chunk

**Điểm yếu:**
- Câu nói bị cắt ở ranh giới hai slide.
- 188 chunk có dưới 32 token.
- Không lấy được slide trước/sau.

| Paper | Mã | Ý chính |
|---|---|---|
| Reconstructing Context: late chunking và contextual retrieval | arXiv 2504.19754 | Contextual retrieval cộng trộn hạng và rerank cho độ chính xác cao nhất nhưng tốn chi phí; late chunking nhẹ hơn nhưng kết quả dao động theo model |
| LEA, VisRAG (đã có trong repo) | | |

**Nên thử:** chia chunk theo câu nói và cho phép tham chiếu nhiều slide; khi trả lời thì lấy thêm câu hoặc slide liền trước và liền sau. **Đây là yêu cầu của cô**, đang làm ở giai đoạn 2.

## 5. Truy xuất

**Điểm yếu:**
- Ngữ cảnh bị cắt 2.500 ký tự.
- Lọc stopword sau khi bỏ dấu nên mất thuật ngữ.
- 119 chunk OCR trùng lặp.

| Paper | Mã | Ý chính |
|---|---|---|
| ViRanker: reranker tiếng Việt dựa trên BGE-M3 | arXiv 2509.09131 | NDCG@3 0,6815, MRR@3 0,6641 trên MMARCO-VI |
| AITeamVN/Vietnamese_Reranker | Hugging Face | Bản tinh chỉnh bge-reranker-v2-m3 cho tiếng Việt, đo trên văn bản luật |
| Truy xuất văn bản luật tiếng Việt | arXiv 2409.13699, 2507.14619 | BM25 chỉ khớp được khi token giống hệt → cần tách từ tiếng Việt (pyvi) |
| AdaGReS: chọn ngữ cảnh có tính đến trùng lặp | arXiv 2512.25052 | Cách cân bằng liên quan và trùng lặp tự hiệu chỉnh, không phải chỉnh tay như MMR |
| DF-RAG: đa dạng theo câu hỏi | arXiv 2601.17212 (EACL 2026 Findings) | Dựa trên MMR, chọn mức đa dạng riêng cho từng câu hỏi |

**Nên thử:**
- Lọc stopword có dấu và khớp theo cụm từ; gộp OCR trùng rồi chọn slide đa dạng (MMR). Hai việc này **đang làm ở giai đoạn 2**.
- Sau đó mới thử thêm reranker tiếng Việt (ViRanker), vì reranker không cứu được bằng chứng đã bị loại từ bước trước.

## 6. Sinh câu trả lời

**Điểm yếu:**
- LLM chỉ đọc chữ, không nhìn ảnh slide.
- Chưa đo độ trung thực của câu trả lời và khả năng biết từ chối.

| Paper | Mã | Ý chính |
|---|---|---|
| From Slides to Chatbots | arXiv 2510.22272 | RAG đa phương thức: đưa **ảnh slide** đã truy xuất cho model sinh câu trả lời để giữ thông tin hình |
| VisRAG, Video-RAG (đã có trong repo) | | |

**Nên thử:** khi câu hỏi về hình, đồ thị hay bảng thì chuyển sang VLM (Qwen2.5-VL hoặc Qwen3-VL) kèm ảnh slide; câu thường vẫn dùng LLM chữ cho nhanh. Cần GPU.

## 7. Đánh giá

| Paper | Mã | Ý chính |
|---|---|---|
| CourseTimeQA | arXiv 2512.00360 | Bộ đo hỏi đáp video bài giảng **có mốc thời gian**: 52,3 giờ, 902 câu, 6 khóa học, có đoạn thời gian đáp án chuẩn. Phương pháp CrossFusion-RAG hơn BLIP-2 0,10 nDCG@10. **Rất giống bài toán của mình, nên dùng làm mẫu so sánh** |
| EduVidQA | EMNLP 2025 | Câu trả lời dài cho câu hỏi của sinh viên về video bài giảng, cùng cách đánh giá |
| Video-MMLU | arXiv 2504.14693 | Đo khả năng hiểu video bài giảng, khoảng 1.065 video |
| ViDoRe V2 | arXiv 2505.17166 | Bộ đo truy xuất tài liệu dạng ảnh |
| RAGAS (đã có trong repo) | arXiv 2309.15217 | |

**Nên làm:**
- Nhóm kiểm lại 120 câu hỏi.
- Chia tập dev và tập giữ lại (held-out) để tránh "chỉnh cho vừa" bộ đo.
- Đo theo cách của CourseTimeQA (nDCG@10, MRR theo đoạn thời gian).
- Chạy RAGAS khi có GPU.

## Lộ trình đề xuất

Bảng xếp theo lợi ích chia chi phí, theo lộ trình của Codex trong `upgrade-v3-research.md`.

| # | Việc | Chạy ở đâu | Trạng thái |
|---:|---|---|---|
| 1 | Giữ đúng đoạn tìm được và lấy thêm slide trước/sau | CPU | **đang làm (giai đoạn 2)** |
| 2 | Làm sạch OCR thận trọng (bảo vệ tiêu đề, code, công thức) | CPU | chờ |
| 3 | Bỏ bước sửa ASR bằng OCR, chỉ sửa theo danh sách thuật ngữ đã duyệt | CPU | chờ |
| 4 | Lọc stopword có dấu, khớp theo cụm từ | CPU | **đang làm (giai đoạn 2)** |
| 5 | Gộp OCR trùng, chọn slide đa dạng, chỉnh lại hệ số ưu tiên | CPU | **đang làm (giai đoạn 2)** |
| 6 | Lọc quảng cáo, giữ phần giảng xen giữa | CPU | chờ |
| 7 | Phát hiện OCR/ASR hỏng trước khi đưa vào chỉ mục, chỉ chạy lại phần hỏng | CPU + Kaggle | chờ |
| 8 | Tách slide theo vùng trên bảng đen, giữ mốc thời gian slide hiện dần | CPU + Kaggle | chờ |
| 9 | Tiêu đề, code và hình có cấu trúc; đưa mô tả hình đã duyệt vào chỉ mục | CPU + Kaggle | chờ |
| 10 | Truyền thuật ngữ cho Whisper; so sánh PhoWhisper và ChunkFormer | Kaggle | chờ |
| 11 | Reranker tiếng Việt (ViRanker) | CPU hoặc Kaggle | chờ |
| 12 | VLM nhìn ảnh slide cho câu hỏi về hình; đo trích dẫn và khả năng từ chối | Kaggle | chờ |

Việc 1–6 làm trên CPU với dữ liệu sẵn có, đo bằng bộ 120 câu. Việc 7–12 gom lại để chạy Kaggle **một lần**.

## Nguồn
- [SliTraNet, arXiv 2202.03540](https://arxiv.org/abs/2202.03540)
- [Kenny và cộng sự, ICDAR 2017](https://www.cs.rit.edu/~rlaz/files/Kenny_ICDAR_2017.pdf)
- [ColQwen2/ColPali, kiểm chứng lại, arXiv 2505.07730](https://arxiv.org/pdf/2505.07730)
- [PaddleOCR-VL, arXiv 2510.14528](https://arxiv.org/pdf/2510.14528)
- [PaddleOCR-VL-1.5, arXiv 2601.21957](https://arxiv.org/pdf/2601.21957)
- [dots.ocr, arXiv 2512.02498](https://arxiv.org/pdf/2512.02498)
- [LCB-net, arXiv 2401.06390](https://arxiv.org/abs/2401.06390)
- [KG-Whisper, arXiv 2406.02649](https://arxiv.org/html/2406.02649v1)
- [Nghiên cứu prompt Whisper, arXiv 2406.05806](https://arxiv.org/pdf/2406.05806v1)
- [Từ hiếm zero-shot cho Whisper, arXiv 2502.11572](https://arxiv.org/pdf/2502.11572)
- [TCPGen với Whisper, arXiv 2306.01942](https://arxiv.org/abs/2306.01942v1)
- [Khảo sát lỗi bịa của Whisper, arXiv 2501.11378](https://arxiv.org/html/2501.11378v1)
- [Phát hiện lỗi bịa của Whisper, arXiv 2606.23060](https://arxiv.org/pdf/2606.23060)
- [Giảm lỗi bịa của Whisper bằng điều hướng biểu diễn ẩn, arXiv 2606.07473](https://arxiv.org/pdf/2606.07473)
- [Cơ chế lặp trong ASR, arXiv 2508.15882](https://arxiv.org/pdf/2508.15882)
- [VietASR, arXiv 2505.21527](https://arxiv.org/pdf/2505.21527)
- [ChunkFormer-Large-Vie](https://huggingface.co/khanhld/chunkformer-large-vie)
- [HyPoradise, arXiv 2309.15701](https://arxiv.org/abs/2309.15701v2)
- [GenSEC, arXiv 2409.09785](https://arxiv.org/pdf/2409.09785)
- [ClozeGER, arXiv 2405.10025](https://arxiv.org/pdf/2405.10025)
- [Reconstructing Context, arXiv 2504.19754](https://arxiv.org/abs/2504.19754)
- [ViRanker, arXiv 2509.09131](https://arxiv.org/pdf/2509.09131)
- [AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker)
- [Truy xuất văn bản luật tiếng Việt, arXiv 2409.13699](https://arxiv.org/pdf/2409.13699)
- [Tối ưu truy xuất văn bản luật tiếng Việt, arXiv 2507.14619](https://arxiv.org/pdf/2507.14619)
- [AdaGReS, arXiv 2512.25052](https://arxiv.org/pdf/2512.25052)
- [DF-RAG, arXiv 2601.17212](https://arxiv.org/abs/2601.17212v1)
- [CourseTimeQA, arXiv 2512.00360](https://arxiv.org/abs/2512.00360v1)
- [From Slides to Chatbots, arXiv 2510.22272](https://arxiv.org/html/2510.22272v1)
- [EduVidQA, EMNLP 2025](https://preview.aclanthology.org/setup/2025.emnlp-main.1760)
- [Video-MMLU, arXiv 2504.14693](https://arxiv.org/html/2504.14693v1)
- [ViDoRe V2](https://vidore-readme.static.hf.space)
