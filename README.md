# Video Q&A (kiểu nút "Đặt câu hỏi" của YouTube)

Upload video từ máy tính, xem video, bấm "✦ Đặt câu hỏi" để hỏi AI (Gemini) về nội
dung video dựa trên transcript.

## Chạy backend (FastAPI)

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> Nếu port 8000 đã bị chiếm trên máy bạn, đổi sang port khác, ví dụ `--port 8001`,
> và sửa `API_BASE` trong `frontend/src/App.jsx` cho khớp.

Cần điền `GEMINI_API_KEY` trong file `.env` ở thư mục gốc trước khi hỏi được câu hỏi.

## Chạy frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Mở http://localhost:5173

## Luồng transcript (Whisper chạy trên Kaggle vì máy không có GPU)

1. Chạy notebook Whisper trên Kaggle với dataset
   [`l25a-lecture-videos-media`](https://www.kaggle.com/datasets/thnhtrungnguynmif/l25a-lecture-videos-media)
   để lấy transcript có timestamp cho từng video.
2. Xuất transcript theo format:
   ```json
   {
     "video_id": "<id giống video_id backend trả về khi upload>",
     "language": "vi",
     "segments": [{ "start": 0.0, "end": 3.2, "text": "..." }]
   }
   ```
3. Nạp vào backend bằng `PUT /api/videos/{video_id}/transcript`, hoặc copy thẳng
   file JSON vào `backend/data/transcripts/{video_id}.json`.

Xem thêm chi tiết kiến trúc trong `AGENTS.md`.
