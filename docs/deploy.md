# Kế hoạch deploy bản demo

## Kiến trúc

```
Người dùng ─► Frontend (Vercel, luôn bật)
                 │ HTTPS · Cloudflare Tunnel (link tạm *.trycloudflare.com)
                 ▼
            Máy Vast.ai (RTX 3090/4090, chỉ bật khi demo)
              ├─ Backend FastAPI
              ├─ Ollama · Qwen2.5-7B (GPU)
              └─ (bước 5) worker trích xuất video mới
                 │
                 ├─► Supabase: Postgres + pgvector (video, slide, lời giảng, chỉ mục, job, link backend)
                 └─► Cloudflare R2: file video mp4, ảnh keyframe
```

| Phần | Dịch vụ | Chi phí |
|---|---|---|
| Frontend | Vercel | miễn phí |
| Backend + LLM + (worker) | Vast.ai | ~0,2–0,4 USD/giờ, chỉ khi bật |
| Database + chỉ mục | Supabase (vùng Singapore) | miễn phí (500 MB) |
| File video, ảnh | Cloudflare R2 | miễn phí (10 GB, không tính băng thông) |
| Đường vào máy Vast | Cloudflare Tunnel (link tạm) | miễn phí |

Dữ liệu nằm ở Supabase và R2, không nằm trên máy Vast: tắt hoặc thuê máy mới vẫn giữ nguyên dữ liệu,
chỉ cần chạy lại script cài đặt.

**Link tạm đổi mỗi lần bật máy:** khi khởi động, máy Vast ghi link tunnel mới vào bảng `app_config`
của Supabase; frontend đọc link từ đó lúc mở trang, nên không phải deploy lại Vercel.

## Chuẩn bị (tài khoản đứng tên nhóm)

1. **Supabase**: tạo project (vùng Singapore), bật extension `vector` (Database → Extensions).
2. **Cloudflare**: tạo bucket R2 `lecture-rag`, tạo API token R2 quyền đọc/ghi (Account ID, Access Key ID, Secret).
3. **Vercel**: tạo tài khoản, nối với repo GitHub.
4. **Vast.ai**: tạo tài khoản, thêm SSH key, nạp ~10 USD (khi tới bước 4).
5. Không gửi key qua chat: điền vào `.env` theo mẫu `.env.example` (bước 1 tạo ra).

## Bước 1 · Schema Supabase và kết nối R2 (~½ ngày)

- File migration `infra/supabase/schema.sql`:

| Bảng | Nội dung |
|---|---|
| `videos` | id, tên, thời lượng, ngôn ngữ, trạng thái (`ready` / `processing` / `failed`), khóa file video trên R2 |
| `segments` | mỗi slide: video_id, bắt đầu/kết thúc, tiêu đề, chữ OCR sạch, lời giảng, khóa ảnh keyframe, loại (slide / quảng cáo) |
| `words` | bản chép lời từng từ có thời gian (tab "Bản chép lời") |
| `chunks` | mẩu bằng chứng: loại (speech / ocr / region), nội dung, `embedding vector(384)`, `tsv tsvector` (bỏ dấu) |
| `jobs` | hàng đợi trích xuất (bước 5): video_id, trạng thái, bước hiện tại, lỗi |
| `app_config` | link backend hiện tại (máy Vast ghi khi khởi động) |

- Chỉ mục: HNSW trên `chunks.embedding`, GIN trên `chunks.tsv`, `chunks(video_id)`.
- Lớp R2 dùng S3 API (boto3): tải lên, xóa, tạo URL ký tạm thời để trình duyệt phát video thẳng từ R2.
- `.env.example`: `SUPABASE_DB_URL`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `R2_ACCOUNT_ID`,
  `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`, `LLM_PROVIDER`, `OLLAMA_MODEL`, `CORS_ORIGINS`.

## Bước 2 · Backend đọc/ghi Supabase + R2 (~1,5 ngày)

- `backend/app/shared/storage.py`: có biến môi trường Supabase thì dùng Supabase + R2, không có thì dùng file
  như hiện tại (để chạy thử trên máy). API cho frontend giữ nguyên.
- `backend/app/features/search/retrieval.py`: tìm kiếm thành 2 truy vấn SQL
  - theo nghĩa: `ORDER BY embedding <=> :query_vector LIMIT 20` (E5 ONNX mã hóa câu hỏi như hiện tại);
  - theo từ khóa: `tsv @@ websearch_to_tsquery(...)` xếp theo `ts_rank`;
  - gộp RRF (nghĩa 1,0 · từ khóa 0,5), chữ slide ×1,05, lấy 3 slide; hỏi nhiều video = `video_id = ANY(...)`,
    gộp theo cosine như bản hiện tại.
- Video, keyframe: endpoint trả URL ký tạm thời của R2 thay cho stream file.
- Kiểm tra: test backend chạy với cả hai chế độ; so kết quả tìm kiếm bản Supabase với bản file trên cùng video.

## Bước 3 · Nạp dữ liệu (~½ ngày)

- Viết lại `backend/scripts/import_fast_pipeline.py`: đọc `fast_pipeline_export` của Kaggle, đẩy video và
  keyframe lên R2, ghi `videos` / `segments` / `words` / `chunks` (kèm vector E5 có sẵn) vào Supabase.
- Chạy `notebooks/fast_pipeline_l25.ipynb` trên Kaggle cho 5–8 video demo
  (ví dụ L25_V008, V011, V037, V017, hVU5Mw19V_c + 1–2 video tiếng Anh), rồi nạp.
- Kiểm tra trên Supabase dashboard: số slide, số chunk mỗi video; mở thử URL video trên R2.

## Bước 4 · Deploy (~1 ngày)

**Frontend (Vercel)**
- Root directory `frontend`, build `npm run build`, output `dist`.
- Biến môi trường: `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` (chỉ đọc bảng `app_config`).
- Lúc mở trang: đọc link backend từ `app_config`; không đọc được hoặc backend không trả lời thì hiện
  "Server demo đang tắt".

**Máy Vast** (`infra/vast_setup.sh`)
1. Thuê 1× RTX 3090/4090 (24 GB), ổ ~60 GB, image có CUDA, chế độ **On-demand**.
2. Script: clone repo → cài Python, `backend/requirements.txt`, FFmpeg → cài Ollama, tải `qwen2.5:7b`
   → chép `.env` (bạn tự đưa lên bằng `scp`) → chạy backend (uvicorn) → chạy `cloudflared tunnel --url`
   → ghi link tunnel vào `app_config`.
3. Backend bật CORS cho tên miền `*.vercel.app` của project.

**Thử toàn bộ**: mở link Vercel → chọn video → hỏi 1 câu → xem chữ hiện dần, nút ▶ mốc thời gian,
dòng "Nguồn"; chọn 2 video → hỏi → nguồn từ cả hai video.

**Vận hành**
- Không dùng: **Stop** máy Vast (giữ ổ đĩa, ~0,1–0,2 USD/ngày). Không **Destroy** nếu chưa cần.
- Trước demo: Start máy, chạy lại `vast_setup.sh` (nếu cần), đợi ~5 phút cho Ollama nạp model, hỏi thử 1 câu.
- Quay sẵn video màn hình demo để dự phòng mất mạng.

## Bước 5 (tùy chọn) · Upload là tự trích xuất (~2 ngày)

- Nút **Thêm video** trên web: video lên R2, tạo dòng trong `jobs`.
- Worker trên máy Vast: lấy job, chạy pipeline 01 → 05 (`pipeline/stages/` và `pipeline/shared/`, OCR Qwen3-VL, Whisper, E5)
  bằng GPU, ghi kết quả vào Supabase + R2, cập nhật bước hiện tại vào `jobs`.
- Web hiện thanh tiến độ trên thẻ video ("Đang đọc chữ slide 7/20…"); xong thì hỏi được ngay.
- Video 30 phút ≈ 4–6 phút trên RTX 4090. Cần GPU 24 GB vì worker (Qwen3-VL) và Ollama dùng chung.
- Làm sau khi bước 1–4 chạy ổn; không cần đổi kiến trúc.

## Tổng thời gian

| Phạm vi | Thời gian |
|---|---|
| Bước 1–4 (video nạp sẵn) | ~3,5 ngày |
| Thêm bước 5 | ~5,5 ngày |
| Chi phí demo | Vast ~10–15 USD cho cả quá trình thử + bảo vệ; các dịch vụ khác miễn phí |
