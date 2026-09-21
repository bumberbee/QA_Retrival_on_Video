# AGENTS.md

Ghi chú ngữ cảnh cho dự án Thesis, dùng cho Claude Code / các agent làm việc trong repo này.

## Dataset

- Nguồn: Kaggle
- Dataset: `thnhtrungnguynmif/l25a-lecture-videos-media`
- URL: https://www.kaggle.com/datasets/thnhtrungnguynmif/l25a-lecture-videos-media
- Mục đích: dữ liệu chính dùng cho dự án này (video/media bài giảng L25A).

## MCP: Kaggle

- Server được khai báo trong `.mcp.json`, chạy qua `uvx kaggle-mcp`.
- Credentials nằm trong `.env` (KHÔNG commit — đã có trong `.gitignore`):
  - `KAGGLE_USERNAME=thnhtrungnguynmif`
  - `KAGGLE_KEY=...`
  - `KAGGLE_DATASET=thnhtrungnguynmif/l25a-lecture-videos-media`
- Cần restart Claude Code sau khi thêm/sửa `.mcp.json` hoặc `.env` để server nhận cấu hình mới.

## Kiến trúc ứng dụng: Video Q&A (giống nút "Đặt câu hỏi" của YouTube)

Ý tưởng: upload video từ máy tính lên web, xem video, bấm nút "✦ Đặt câu hỏi" để hỏi
AI về nội dung video (dựa trên transcript), giống tính năng Ask trên YouTube.

- **Backend**: `backend/` — FastAPI (Python)
  - `app/main.py`: khởi tạo app, CORS cho `http://localhost:5173`
  - `app/routers/videos.py`: upload video, list video, serve file, lưu/lấy transcript
  - `app/routers/qa.py`: nhận câu hỏi + video_id, trả lời dựa trên transcript qua Gemini
  - `app/services/gemini.py`: gọi Gemini API (`google-generativeai`), ghép transcript có
    timestamp vào prompt, có gợi ý mốc thời gian [mm:ss] hiện tại người dùng đang xem
  - `app/storage.py`: lưu video vào `backend/data/videos/`, transcript JSON vào
    `backend/data/transcripts/{video_id}.json` (KHÔNG commit thư mục videos)
  - Chạy: `cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload`

- **Frontend**: `frontend/` — React + Vite, giao diện tối kiểu YouTube
  - `src/App.jsx`: upload video, player, panel chat "Đặt câu hỏi"
  - Chạy: `cd frontend && npm run dev` (mặc định port 5173, gọi API ở `localhost:8000`)

- **Transcript / Whisper**: máy hiện KHÔNG có GPU, nên bước transcribe (Whisper) chạy
  trên **Kaggle Notebook** (có GPU free) với dataset `l25a-lecture-videos-media`.
  Sau khi có transcript, tải file JSON về theo format `Transcript` trong
  `backend/app/models.py`:
  ```json
  {
    "video_id": "<id>",
    "language": "vi",
    "segments": [{"start": 0.0, "end": 3.2, "text": "..."}]
  }
  ```
  rồi `PUT /api/videos/{video_id}/transcript` để nạp vào backend (hoặc bỏ thẳng file
  JSON vào `backend/data/transcripts/{video_id}.json`, đặt tên trùng `video_id` khi
  upload video qua `POST /api/videos`).

- **LLM trả lời câu hỏi**: dùng **Gemini API** (chưa dùng Claude/OpenAI cho phần này).
  Cần điền `GEMINI_API_KEY` vào `.env` (đang để trống, cần user cung cấp).

- **Lưu trữ video**: hiện tại lưu local (`backend/data/videos/`) để demo; dự kiến sau
  này chuyển sang cloud storage (S3/Cloudinary/Supabase Storage) khi deploy thật.

## Việc còn thiếu / cần làm tiếp

- [ ] User cung cấp `GEMINI_API_KEY` (điền vào `.env`)
- [ ] Viết Kaggle notebook chạy Whisper để transcribe video trong dataset, xuất ra
      JSON đúng format `Transcript` ở trên
- [ ] Kết nối MCP Kaggle (restart Claude Code) để tự động tải/kiểm tra dataset
- [ ] (Tương lai) Chuyển lưu trữ video sang cloud storage khi deploy
