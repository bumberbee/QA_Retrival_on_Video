# Sáng mai làm gì (deploy demo)

Đêm qua đã xong: pipeline chạy trên Kaggle cho 10 video, dữ liệu đã import vào máy, tính năng hội thoại đã test, script deploy sẵn sàng.
Bạn chỉ cần tạo tài khoản, điền key và chạy lệnh theo đúng thứ tự dưới đây. Chi tiết thêm xem `docs/deploy.md`.

## Kết quả chạy đêm qua

- Kernel `thnhtrungnguynmif/fast-pipeline-10-mixed-20261020` chạy trên Kaggle 2×T4, **COMPLETE**:
  - 10 video, tổng thời lượng khoảng 4,6 giờ;
  - tổng thời gian chạy khoảng **96 phút**;
  - GPU dùng nhiều nhất khoảng 11,9 / 15 GB.
- Đầu ra nằm ở `D:/thesis-data/run-20261020/` (zip export, log, timings). Mp4 nằm ở `D:/thesis-data/mp4-extra/`.
- App data ở `D:/thesis-data/app-data`: 10 video, 502 slide, 1233 chunk.
- Bước OCR (Qwen3-VL) là bước chậm nhất. Video `tHE1mC_n0-E` có 203 slide nên riêng bước OCR mất khoảng 44 phút.
- Đã test hội thoại 3 lượt trên L25_V011 (Ollama qwen2.5:7b, máy CPU):
  - câu "Còn mục đích của nó là gì?" được viết lại đúng thành "Mục đích của Liên Hiệp Quốc là gì?";
  - phiên được lưu và mở lại được;
  - mỗi câu mất 100–160 giây trên CPU. Trên GPU Vast sẽ chỉ còn vài giây.
- Đã sync thử vào Postgres pgvector local: 10/10 video OK. Backend có 45/45 test pass.

## Các bước sáng mai

1. **Sửa quyền push GitHub.** `gh` đang đăng nhập bằng `aidev3-web`, tài khoản này không có quyền vào `toilaMif/thesis-video-qa`. Chạy:
   ```
   ! gh auth login
   ```
   Đăng nhập bằng tài khoản `toilaMif`, rồi chạy:
   ```
   git push -u origin upgrade/pipeline-v2
   ```
2. **Supabase** (supabase.com, gói Free):
   - tạo project, region Singapore;
   - mở SQL Editor, dán toàn bộ `deploy/supabase/schema.sql`, bấm Run;
   - Project Settings → Database → Connection string → **Session pooler**: chép vào `SUPABASE_DB_URL`;
   - Settings → API: chép `SUPABASE_URL` và `SUPABASE_ANON_KEY`.
3. **Cloudflare R2:**
   - tạo bucket `lecture-rag`;
   - vào Manage API tokens, tạo token Object Read & Write;
   - điền `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`.
4. Chép `.env.example` thành `.env` rồi điền các key trên. **Không commit `.env`.**
5. **Đẩy dữ liệu** (khoảng 10–20 phút, vì mp4 khoảng 1,5 GB):
   ```
   cd backend
   .venv/Scripts/python scripts/sync_supabase.py --data D:/thesis-data/app-data
   ```
6. **Vercel:**
   - Import repo, Root Directory là `frontend`;
   - đặt Env `VITE_SUPABASE_URL` và `VITE_SUPABASE_ANON_KEY`;
   - bấm Deploy, rồi chép link vào `CORS_ORIGINS` trong `.env`.
7. **Vast.ai:**
   - thuê 1× RTX 3090/4090, ổ đĩa 60 GB;
   - chép code và `.env` vào `/workspace/Thesis`;
   - chạy:
     ```
     bash /workspace/Thesis/deploy/vast_setup.sh
     ```
   - Script tự ghi link tunnel vào Supabase, frontend sẽ tự đọc link đó.
8. Mở link Vercel rồi hỏi thử. Xong buổi demo thì **Stop** máy trên Vast để khỏi tốn tiền.

## Còn lưu ý

- Link trycloudflare đổi mỗi lần bật lại máy Vast. Chỉ cần chạy lại `vast_setup.sh`, frontend tự nhận link mới.
- Viết lại câu hỏi bị giới hạn 40 giây (`REWRITE_TIMEOUT_SECONDS`). Trên CPU đôi khi quá giờ thì giữ nguyên câu gốc. Câu "Nó có bao nhiêu nước…" trong test chưa được viết lại, nhưng câu trả lời vẫn đúng ngữ cảnh. Trên GPU sẽ không bị.
- Nên đổi (rotate) các Kaggle token đã dán trong chat.
- Commit chưa push. Nhánh hiện tại là `upgrade/pipeline-v2`.
