# Video Q&A cho bài giảng tiếng Việt

Luận văn: hỏi đáp trên video bài giảng có slide. Pipeline trên Kaggle tách slide, đọc chữ trên
slide (OCR), nhận dạng lời giảng và đồng bộ chúng theo thời gian; web app cho xem video, slide,
lời giảng và hỏi AI kèm nguồn.

## Cấu trúc thư mục

```text
Thesis/
├─ backend/              FastAPI (cổng 8001): API video, transcript, slide, hỏi AI
│  ├─ app/               mã nguồn
│  ├─ scripts/           nạp kết quả pipeline vào app (import_fast_pipeline.py, …)
│  ├─ tests/
│  └─ data/              dữ liệu app: video, keyframe, transcript (không đưa lên git)
├─ frontend/             React + Vite (cổng 5173)
├─ kaggle/lecture_rag/   pipeline trích xuất 01 → 05, bộ gold, test (xem README trong đó)
└─ docs/
   ├─ pipeline-report.html    báo cáo: dữ liệu, các bước trích xuất, số đo, triển khai
   └─ walkthrough-v008.html   pipeline chạy từng giai đoạn trên 1 video toán
```

Tải video từ máy tính hoặc nhập liên kết YouTube, Facebook, TikTok, xem video và hỏi AI về nội
dung video.

## Chạy backend (FastAPI)

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

> Nếu port 8001 đã bị chiếm, đổi sang port khác, ví dụ `--port 8002`, và thêm
> `VITE_API_BASE_URL=http://localhost:8002/api` vào `frontend/.env.local`, sau đó
> khởi động lại Vite. Cấu hình API được đọc tại `frontend/src/lib/api.js`.

Cần điền `GEMINI_API_KEY` trong file `.env` ở thư mục gốc trước khi hỏi được câu hỏi.

### Nhập video từ liên kết

Trong màn hình thêm video, chọn nhập liên kết và dán URL video công khai từ YouTube
(gồm Shorts và youtu.be), Facebook (gồm Reels và fb.watch) hoặc TikTok (gồm liên kết
chia sẻ vm.tiktok.com/vt.tiktok.com). Video được tải về thư viện để phát và tua như
tệp tải từ máy; trạng thái tải cập nhật trong giao diện.

- `pip install -r requirements.txt` cài `yt-dlp[default]` và bộ hỗ trợ JavaScript EJS.
- Cài **FFmpeg** để có cả `ffmpeg` và `ffprobe` trong `PATH`. Windows:
  `winget install --id Gyan.FFmpeg -e`, sau đó mở lại terminal và khởi động lại backend.
  Kiểm tra bằng `ffmpeg -version` và `ffprobe -version`.
- YouTube thường cần **Node.js 20 trở lên hoặc Deno** để xử lý thử thách JavaScript.
  Backend tự dùng runtime có trong `PATH`; có thể dùng Node.js đã cài cho frontend.
- Nếu nền tảng thay đổi, cập nhật bộ tải bằng `pip install -U "yt-dlp[default]<2027"`.

Chỉ hỗ trợ video công khai có thể tải tại khu vực của máy chủ. Không hỗ trợ video
riêng tư, yêu cầu đăng nhập, trả phí, danh sách phát hoặc livestream. Nền tảng có thể
chặn tải tự động ngay cả với video công khai; khi đó ứng dụng hiển thị lỗi và vẫn cho
phép tải tệp từ máy. Không sử dụng cookie đăng nhập hay vượt qua giới hạn của nền tảng.
Chỉ nhập nội dung bạn được phép sử dụng.

Mỗi video tối đa **1 GiB**, dài **2 giờ**; tác vụ kiểm tra giới hạn **30 phút** trong
quá trình tải/xử lý. Tối đa 2 lượt tải chạy cùng lúc, 8 lượt đang chạy/chờ. Bản tải ưu
tiên MP4 H.264 đến 1080p, hoặc WebM; định dạng khác được chuyển đổi bằng FFmpeg.
Nhập liên kết chưa tự tạo transcript; tiếp tục quy trình Whisper ở dưới để hỏi AI.

API:

- `POST /api/videos/import` với JSON `{"url":"https://youtu.be/..."}` trả HTTP 202.
- `GET /api/videos/import/{import_id}` lấy tiến độ. Cả hai trả
  `{"import_id":"...","status":"queued","progress":0,"video":null,"error":null}`.
  `status` là `queued`, `downloading`, `completed` hoặc `failed`; `progress` từ 0–100
  hoặc `null` khi chưa xác định. Khi hoàn tất, `video` chứa `VideoInfo` kèm
  `source_url` và `source_platform`; khi thất bại, `error` là thông báo tiếng Việt.
- URL không hợp lệ trả HTTP 400; hàng đợi đầy trả HTTP 429; mã tác vụ không còn tồn
  tại trả HTTP 404. Trạng thái của 100 lượt gần nhất lưu trong bộ nhớ: khởi động lại
  backend sẽ mất trạng thái, nhưng video đã nhập thành công vẫn được giữ. Chạy một
  worker Uvicorn cho hàng đợi này.

Kiểm thử backend (không tải video từ Internet):

```bash
cd backend
.venv/Scripts/python -m unittest discover -s tests -v
```

### Chọn backend sinh câu trả lời

`LLM_PROVIDER` trong `.env` quyết định nơi sinh câu trả lời:

- `gemini` (mặc định) — gọi Gemini API, cần `GEMINI_API_KEY`. Free tier hiện giới
  hạn 5 request/phút cho `gemini-flash-latest`.
- `ollama` — gọi Ollama chạy local, dùng để smoke test toàn hệ thống mà không tốn
  quota. Cần `ollama serve` đang chạy và model đã được `ollama pull`.

```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2:7b
```

Lưu ý khi dùng `ollama` trên máy không GPU: sinh câu trả lời chỉ đạt khoảng
6 token/giây, nên một câu hỏi với transcript đầy đủ mất cỡ 2 phút. Model local
cũng không nhận được ảnh slide và thường bỏ qua yêu cầu trích dẫn `[mm:ss]`, nên
chỉ dùng để kiểm tra hệ thống chạy đúng, không dùng để đánh giá chất lượng trả lời.

`GET /api/health` trả về provider và model đang dùng.

## Chạy frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Mở http://localhost:5173

## Nạp kết quả pipeline vào app

Máy không có GPU nên pipeline chạy trên Kaggle (xem `kaggle/lecture_rag/README.md`). Tải
`fast_pipeline_export.zip`, giải nén rồi:

```bash
cd backend
.venv/Scripts/python scripts/import_fast_pipeline.py --export <thư mục giải nén> --videos <thư mục mp4>
```

Cũng có thể nạp riêng transcript bằng `PUT /api/videos/{video_id}/transcript` với JSON
`{"video_id": "...", "language": "vi", "segments": [{"start": 0.0, "end": 3.2, "text": "..."}]}`.
