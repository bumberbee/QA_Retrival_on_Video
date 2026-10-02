# Pipeline trích xuất bài giảng (Kaggle)

Video bài giảng → **01** tách slide → **02** OCR kết hợp → **03** nhận dạng giọng nói
(chạy song song với 01) → **04** đồng bộ lời với slide → **05** chỉ mục tìm kiếm.
Mô tả đầy đủ, số đo và các đề xuất: [`docs/pipeline-report.html`](../../docs/pipeline-report.html).

## Thư mục

```text
lecture_rag/
├─ notebook_code/        mã nguồn pipeline, sửa ở đây
│  ├─ common.py          tiện ích chung (đọc/ghi, audio PCM, manifest)
│  ├─ detection.py       01 · tách slide
│  ├─ visual.py          02 · OCR (hybrid: Unlimited-OCR + Qwen3-VL; hoặc EasyOCR)
│  ├─ transcription.py   03 · Whisper medium + VAD
│  ├─ adfilter.py        lọc quảng cáo trong bước 01 (SigLIP, đoạn "ad")
│  ├─ alignment.py       04 · gán từ vào slide
│  └─ indexing.py        05 · E5 + SQLite FTS5 (và truy xuất cho notebook 06)
├─ notebooks/            notebook sinh ra từ notebook_code/, không sửa tay
│  ├─ fast_pipeline_l25.ipynb        bản đang dùng: 01 → 05 cho nhiều video, 1 phiên T4 x2
│  ├─ 01_…_l25_v001 … 06_…ipynb      cùng thuật toán, mỗi bước một notebook (L25_V001)
│  ├─ explain_slide_detection.ipynb  bước 01 chạy từng bước, in mọi output (đã chạy sẵn)
│  ├─ ad_filter_experiments.ipynb    thí nghiệm bộ lọc quảng cáo E1 (trò) và E4 (thầy)
│  ├─ ad_detection_run.ipynb         bước 01 có lọc quảng cáo trên 10 video gold
│  └─ lpm_video_download.ipynb       tải video bộ LPM về dataset Kaggle
├─ build_*.py            sinh các notebook trong notebooks/
├─ gold/                 bộ gold 1: nhãn ranh giới slide và nhãn quảng cáo của 10 video L25
├─ ocr_bench/            so sánh các model OCR (kết quả: docs/ocr-bench-report.html)
├─ tests/                31 test, không cần GPU hay tải model
└─ runs/                 lần chạy Kaggle đang dùng (không đưa lên git)
```

## Chạy pipeline

```powershell
python kaggle/lecture_rag/build_fast_pipeline.py      # sinh notebooks/fast_pipeline_l25.ipynb
```

Import notebook lên Kaggle: GPU **T4 x2**, bật Internet, Add Input dataset
[`l25a-lecture-videos-media`](https://www.kaggle.com/datasets/thnhtrungnguynmif/l25a-lecture-videos-media)
và output của kernel `l25-ad-filter-e1-e4-20261002` (model lọc quảng cáo `ad_student.npz`),
sửa `VIDEO_IDS` ở ô Cấu hình rồi Run All. Tải `fast_pipeline_export.zip` và nạp vào web app:

```powershell
cd backend
.venv/Scripts/python scripts/import_fast_pipeline.py --export <thư mục giải nén> --videos <thư mục mp4>
```

## Bộ gold 1 (`gold/`)

| File | Nội dung |
|---|---|
| `slide_boundaries/gold.json` | ranh giới slide đã người kiểm tra (S, R, SC, L, X; đã bỏ quảng cáo) |
| `slide_boundaries/gold_with_ads.json` | như trên, còn giữ nhãn C (chuyển cảnh, quảng cáo) |
| `slide_boundaries/metrics.json` | P/R/F1 của bước 01; cách tính chính: cuộn trang là slide mới |
| `slide_boundaries/gold_table.html` | bảng xem toàn bộ 549 ranh giới kèm ảnh |
| `slide_boundaries/ad_labels.json` | nhãn quảng cáo theo từng giây (1 = quảng cáo) cho bộ lọc quảng cáo |

Công cụ gán nhãn, theo thứ tự dùng: `extract_candidates.py` → `build_review_sheets.py` →
`build_boundary_review.py` (trang review) → `finalize_boundaries.py` → `build_gold_table.py`,
`build_slide_dataset.py` (dataset ảnh slide), `build_ad_labels.py` (nhãn quảng cáo),
`build_ad_frames.py` (khung 1 Hz kèm nhãn → dataset `l25-ad-filter-frames`),
`eval_ad_detection.py` (chấm bước 01 có lọc quảng cáo theo bộ gold).
4 video tuning: V001, V002, V005, V007; 6 video test: V003, V004, V006, V008, V009, V010.

## Bộ dữ liệu luận văn

`build_dataset_kernels.py` sinh 5 notebook Kaggle (LPM, SlideSpeech dev/test và train, 12 video
tiếng Việt, M3AV). YouTube chặn tải từ Kaggle nên phần video YouTube dùng
`relay_youtube_datasets.py D:/thesis-datasets` (tải ở máy, đưa lên Kaggle bằng CLI, giữ bản ở ổ D).
Vai trò từng bộ: mục "Bộ dữ liệu luận văn" trong `docs/pipeline-report.html`.

## Kiểm tra

```powershell
python kaggle/lecture_rag/build_notebooks.py          # sau khi sửa notebook_code/
python -m unittest discover -s kaggle/lecture_rag/tests
```

Test cần FFmpeg/FFprobe, NumPy, Pillow, OpenCV, PyAV, nbformat (`requirements-test.txt`).
Test kiểm tra wiring, mốc thời gian và tính toàn vẹn bằng video tổng hợp và model giả;
chất lượng model phải đo trên Kaggle.
