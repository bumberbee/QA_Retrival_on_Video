# Pipeline trích xuất bài giảng (Kaggle)

Video bài giảng → **01** tách slide → bộ gác OCR → **02** OCR (Qwen3-VL-4B đọc cả slide) →
**03** nhận dạng giọng nói → **04** đồng bộ lời với slide → **05** chỉ mục tìm kiếm (E5 + BM25).

## Thư mục

```text
lecture_rag/
├─ notebook_code/        mã nguồn pipeline, sửa ở đây
│  ├─ common.py          tiện ích chung (đọc/ghi, audio PCM, manifest)
│  ├─ detection.py       01 · tách slide
│  ├─ adfilter.py        bộ gác OCR (SigLIP zero-shot, 14 câu mô tả, ngưỡng 0,15)
│  ├─ visual.py          02 · OCR: 'qwen' (mặc định), 'hybrid' (Unlimited-OCR + Qwen3-VL), 'easyocr'
│  ├─ transcription.py   03 · faster-whisper medium + VAD
│  ├─ textclean.py       04 · làm sạch OCR, sửa thuật ngữ ASR theo chữ trên slide
│  ├─ alignment.py       04 · gán từng từ lời giảng vào slide
│  ├─ indexing.py        05 · E5 + SQLite FTS5
│  └─ retrieval.py       tìm bằng chứng (dùng trong notebook)
├─ notebooks/            notebook sinh ra từ notebook_code/, không sửa tay
│  ├─ fast_pipeline_l25.ipynb         01 → 05 cho nhiều video, xuất fast_pipeline_export.zip
│  ├─ run_pipeline.ipynb              cả pipeline trên 1 video chọn theo tên, in từng bước
│  ├─ full_pipeline_walkthrough.ipynb như trên cho L25_V008, kèm chấm điểm tìm kiếm
│  └─ 01_…_l25_v001 … 06_…ipynb       mỗi bước một notebook (L25_V001)
├─ build_fast_pipeline.py    sinh fast_pipeline_l25.ipynb (cấu hình mặc định ở đây)
├─ build_run_pipeline.py     sinh run_pipeline.ipynb
├─ build_full_walkthrough.py sinh full_pipeline_walkthrough.ipynb
├─ build_notebooks.py        sinh 01 … 06 và chứa hàm dùng chung cho các builder
└─ tests/                    test không cần GPU hay tải model
```

## Chạy pipeline

```powershell
python kaggle/lecture_rag/build_fast_pipeline.py      # sinh notebooks/fast_pipeline_l25.ipynb
```

Import notebook lên Kaggle: GPU **T4 x2**, bật Internet, Add Input dataset chứa video (ví dụ
[`l25a-lecture-videos-media`](https://www.kaggle.com/datasets/thnhtrungnguynmif/l25a-lecture-videos-media)),
sửa `VIDEO_IDS` ở ô Cấu hình rồi Run All. Tải `fast_pipeline_export.zip` và nạp vào web app:

```powershell
cd backend
.venv/Scripts/python scripts/import_fast_pipeline.py --export <thư mục giải nén> --videos <thư mục mp4>
```

Xem kết quả từng bước trên 1 video: import `notebooks/run_pipeline.ipynb`, sửa `VIDEO_NAME` và
`LANGUAGE`, Run All; báo cáo nằm trong `walkthrough_report.html` (tab Output).

## Kiểm tra

```powershell
python kaggle/lecture_rag/build_notebooks.py          # sau khi sửa notebook_code/
python -m unittest discover -s kaggle/lecture_rag/tests
```

Test cần FFmpeg/FFprobe, NumPy, Pillow, OpenCV, PyAV, nbformat (`requirements-test.txt`).
Test kiểm tra wiring, mốc thời gian và tính toàn vẹn bằng video tổng hợp và model giả;
chất lượng model phải đo trên Kaggle.
