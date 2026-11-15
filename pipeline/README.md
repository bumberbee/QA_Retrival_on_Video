# Pipeline trích xuất bài giảng (Kaggle)

Video bài giảng → **01** tách slide → bộ gác OCR → **02** OCR (Qwen3-VL-4B đọc cả slide) →
**03** nhận dạng giọng nói → **04** đồng bộ lời với slide → **05** chỉ mục tìm kiếm (E5 + BM25).

## Thư mục

```text
pipeline/
├─ stages/               mã nguồn từng bước, sửa ở đây
│  ├─ s01_segment/detection.py       01 · tách slide
│  ├─ s02_gate/adfilter.py           bộ gác OCR (SigLIP zero-shot, 14 câu mô tả, ngưỡng 0,15)
│  ├─ s03_ocr/visual.py              02 · OCR: 'qwen' (mặc định), 'hybrid' (Unlimited-OCR + Qwen3-VL), 'easyocr'
│  ├─ s04_asr/transcription.py       03 · faster-whisper medium + VAD
│  ├─ s05_align/{alignment,textclean}.py  04 · đồng bộ, làm sạch OCR và sửa thuật ngữ ASR
│  └─ s06_index/{indexing,retrieval}.py   05 · E5 + SQLite FTS5, tìm bằng chứng
├─ shared/               common.py, production_runtime.py, production_queue.py
│  └─ source_paths.py     module_path(tên file) ánh xạ đến stages/, shared/ hoặc demo/
├─ notebooks/            notebook sinh ra từ mã Python, không sửa tay
│  ├─ fast_pipeline_l25.ipynb         01 → 05 cho nhiều video, xuất fast_pipeline_export.zip
│  ├─ run_pipeline.ipynb              cả pipeline trên 1 video chọn theo tên, in từng bước
│  ├─ full_pipeline_walkthrough.ipynb như trên cho L25_V008, kèm chấm điểm tìm kiếm
│  └─ 01_…_l25_v001 … 06_…ipynb       mỗi bước một notebook (L25_V001)
├─ builders/             build_fast_pipeline.py (cấu hình mặc định), build_run_pipeline.py,
│                        build_full_walkthrough.py, build_notebooks.py (01 … 06)
├─ demo/                 demo_*.py, build_demo_*.py, demo_run.py
├─ ops/production_run.py vận hành các lần chạy production
├─ eval/                 bộ đo; segmentation/ chứa giao thức và công cụ tách slide
├─ notes/                báo cáo chạy, ablation, port và thí nghiệm GPU
├─ runs/                 đầu ra và bộ gold lưu trữ của các lần chạy
└─ tests/                test không cần GPU hay tải model
```

## Chạy pipeline

```powershell
python pipeline/builders/build_fast_pipeline.py      # sinh notebooks/fast_pipeline_l25.ipynb
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
python pipeline/builders/build_notebooks.py          # sau khi sửa stages/ hoặc shared/
python -m unittest discover -s pipeline/tests
```

Test cần FFmpeg/FFprobe, NumPy, Pillow, OpenCV, PyAV, nbformat (`requirements-test.txt`).
Test kiểm tra wiring, mốc thời gian và tính toàn vẹn bằng video tổng hợp và model giả;
chất lượng model phải đo trên Kaggle.
