"""Generate fast_pipeline_l25.ipynb: stages 01-05 for many videos in ONE Kaggle session.

Speed comes from: single-pass stage 01 (no seeks); ASR (03) needing only the audio, so
it runs from the start in parallel with 01 and OCR (02); each stage working through
the videos in its own queue; and installing packages once. Run: python build_fast_pipeline.py
"""

from pathlib import Path
import json

from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
NAME = "fast_pipeline_l25"
MODULES = ["common.py", "detection.py", "adfilter.py", "visual.py", "transcription.py", "textclean.py",
           "alignment.py", "indexing.py"]
DEPS = ('"av>=14,<17" "opencv-python-headless>=4.10,<4.12" "Pillow>=10" '
        '"easyocr>=1.7.2,<2" "faster-whisper>=1.1,<2" "huggingface-hub>=0.25,<1" '
        '"sentence-transformers>=3.4,<6" "transformers==4.57.1" '
        # OCR kết hợp: Unlimited-OCR (remote code) + Qwen3-VL
        '"einops" "addict" "easydict" "accelerate>=1"')

INTRO = """# Pipeline nhanh · 01 → 05 trong một phiên

**Input:** dataset thnhtrungnguynmif/l25a-lecture-videos-media và output của kernel
thnhtrungnguynmif/l25-ad-filter-e1-e4-20261002 (model lọc quảng cáo `ad_student.npz`;
đặt `AD_FILTER = None` nếu không dùng). **Thiết bị:** GPU **T4 x2**.

Cùng thuật toán với notebook 01–05, chỉ khác cách chạy:

1. **Bước 01 quét video một lượt**: giải mã một lần, tinh chỉnh ranh giới từ frame 8 Hz
   đã giữ lại, chọn keyframe trong lượt quét, gộp slide hiện dần.
2. **Bước 03 (ASR) chỉ cần âm thanh**, nên chạy từ đầu song song với 01 và 02.
3. **Bước 02 dùng OCR kết hợp**: Unlimited-OCR đọc chữ, công thức và bảng kèm tọa độ;
   Qwen3-VL đọc số liệu trong từng biểu đồ/hình. Chậm hơn EasyOCR nhiều (15–57 s/slide, trung bình 28 s).
4. **Lọc quảng cáo** trong bước 01: SigLIP base chấm các mẫu 1 Hz, đoạn quảng cáo thành
   segment riêng (`content_kind: "ad"`); 02 bỏ qua OCR, 04 giữ lời của nó riêng, 05 không lập chỉ mục.
5. Mỗi bước xử lý các video theo hàng đợi riêng; cài thư viện một lần cho mọi video.

Kết quả mỗi video giống cấu trúc notebook 01–05, nên notebook 06 và web app dùng được."""

CONFIG = """from pathlib import Path

VIDEO_IDS = ['L25_V001', 'L25_V005', 'L25_V030', 'L25_V052']
OUTPUT_ROOT = Path('/kaggle/working/lecture_rag')

# 01 · slide
SAMPLE_HZ = 1.0
FINE_HZ = 8.0  # Tần số giữ frame nhỏ để tinh chỉnh ranh giới, không cần tua
MERGE_REVEALS = True
# Lọc quảng cáo: 'student' (SigLIP base + hồi quy logistic, file ad_student.npz từ thí nghiệm
# bộ lọc), 'teacher' (SigLIP so400m zero-shot) hoặc None để tắt.
AD_FILTER = 'student'   # 'student', 'teacher', 'outlier' hoặc None
# 'flag' = chỉ đánh dấu đoạn nghi quảng cáo (vẫn OCR + giữ lời, chỉ không lập chỉ mục), tự bỏ
# dấu khi > 6 đoạn hoặc > 30% video (kênh lạ); 'cut' = bản cũ, cắt hẳn đoạn quảng cáo.
AD_MODE = 'flag'
# 'outlier' = so với số đông của chính video (adfilter.outlier_scores), không học nhãn L25.
OUTLIER_OPTIONS = {}
AD_STUDENT_PATH = None  # None: tìm ad_student.npz dưới /kaggle/input
AD_DEVICE = 'cuda:0'
# Bộ gác OCR: keyframe không có nội dung đọc được (giảng viên nói, logo mở đầu) giữ đoạn
# và lời giảng nhưng không OCR. Dùng chung SigLIP base với bộ lọc quảng cáo.
OCR_GATE = True
# Phase 2 (gate_bench, 481 khung L25 + M3AV + LPM, nhãn đã sửa): frame+crops ở 0.15 giữ 100% slide,
# bỏ 13/14 khung trống; 'frame' ở 0.10 cũng giữ 100% nhưng slide thấp nhất chỉ 0.11.
OCR_GATE_THRESHOLD = 0.15
# 'frame' = bản gốc (chấm cả khung). 'frame+crops': khung dưới ngưỡng chỉ bị bỏ khi vùng slide,
# 4 góc và phần giữa cũng dưới ngưỡng -> không bao giờ bỏ nhiều hơn 'frame'.
OCR_GATE_VIEWS = 'frame+crops'

# 02 · OCR. 'hybrid' = Unlimited-OCR (chữ, công thức LaTeX, bảng, có tọa độ) trên GPU 0
# + Qwen3-VL đọc từng vùng biểu đồ/hình trên GPU 1; 'easyocr' = bản cũ, nhanh hơn nhiều.
# Viền hồng chỉ đúng với L25_V001; video khác dùng cả khung hình.
OCR_ENGINE = 'hybrid'
HYBRID_DEVICES = ('cuda:0', 'cuda:1')
OCR_LANGUAGES = ('vi', 'en')
OCR_SCALE = 1.5
OCR_BATCH_SIZE = 32
FRAME_HSV = {'L25_V001': ((135, 80, 120), (175, 255, 255))}
# Vùng "ảnh" chiếm từ tỉ lệ này của slide được Unlimited-OCR đọc lại riêng (cứu chữ bị coi là hình).
# 0.4 = bản gốc; chỉ đổi sau khi đo bằng gold set 2 (gold/eval_ocr.py) trên tập dev.
HYBRID_RESCUE_AREA = 0.4
# Không gửi lại cho Qwen vùng đã bị gọi là trang trí trong video (logo, ảnh giảng viên):
# L25 9 video bỏ 61/314 lần gọi Qwen, bỏ nhầm 2 vùng. Bật sau khi ocr_speed_bench đo được lợi.
HYBRID_SKIP_REPEATED_DECOR = False

# 03 · ASR (giữ model của dự án; turbo đã bị loại sau kiểm tra nội dung)
ASR_MODEL = 'Systran/faster-whisper-medium'
ASR_REVISION = '08e178d48790749d25932bbc082711ddcfdfbc4f'
# Whisper dùng chung GPU 1 với Qwen3-VL (~8.4 GB + ~2 GB), để GPU 0 cho Unlimited-OCR.
ASR_DEVICE_INDEX = 1 if OCR_ENGINE == 'hybrid' else 0
ASR_MODE = 'windowed'

# 04 · đoạn quảng cáo có lời nói giống bài giảng (cặp từ trùng với phần còn lại của bài/slide)
# vẫn giữ lời giảng và chỉ mục (ad_suspect). None = tắt (bản gốc). L25 dev: 0.25 -> 0/27 quảng
# cáo bị giữ oan, giữ được 103/110 đoạn bài giảng bị cắt nhầm; 0.20 -> 1/27 và 110/110.
AD_SUSPECT_OVERLAP = None
# Lọc nội dung B: chấm lời giảng có thuộc chủ đề bài không (E5, chuẩn hóa trong video). Chỉ ghi
# speech_topic vào segments.json, chưa dùng để quyết định. L25 9 video: AUC 0,94 tách quảng cáo.
SPEECH_RELEVANCE = True

# 05 · chỉ lập chỉ mục dòng OCR chưa có ở slide liền trước (L25: 32% dòng lặp lại). Tắt cho tới khi
# đo hỏi đáp (eval_retrieval.py) trên lần chạy lại 10 video.
OCR_DEDUPE_PREVIOUS = False

# 05 · chỉ mục
EMBEDDING_MODEL = 'intfloat/multilingual-e5-small'
EMBEDDING_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
"""

RUN = """import json, shutil, subprocess, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor

# Đo bộ nhớ GPU: lấy mẫu nvidia-smi mỗi 2 s, giữ đỉnh của từng GPU.
gpu_peak_mb, _gpu_stop = {}, threading.Event()
def _sample_gpu():
    while not _gpu_stop.is_set():
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                                  '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
            for line in out.strip().splitlines():
                index, used, total = (int(x) for x in line.split(','))
                gpu_peak_mb[index] = max(gpu_peak_mb.get(index, 0), used)
                gpu_peak_mb[f'{index}_total'] = total
        except Exception:
            pass
        _gpu_stop.wait(2)
threading.Thread(target=_sample_gpu, daemon=True).start()

# Mốc bắt đầu/kết thúc từng bước (giây kể từ lúc chạy) để kiểm tra 03 chạy song song với 01.
spans = []
def timed(stage, video_id, fn, *args):
    start = time.monotonic() - pipeline_started
    try:
        return fn(*args)
    finally:
        spans.append({'stage': stage, 'video_id': video_id, 'start': round(start, 1),
                      'end': round(time.monotonic() - pipeline_started, 1)})

def find_video_file(video_id):
    matches = sorted(Path('/kaggle/input').rglob(video_id + '.mp4'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Cần đúng một {video_id}.mp4, tìm thấy {matches}')
    return matches[0]

ad_scorer = None
if AD_FILTER:
    student_path = AD_STUDENT_PATH or (next(iter(sorted(Path('/kaggle/input').rglob('ad_student.npz'))), None)
                                       if AD_FILTER == 'student' else None)
    if AD_FILTER == 'student' and student_path is None:
        raise FileNotFoundError('AD_FILTER="student" cần ad_student.npz: Add Input output của thí nghiệm bộ lọc.')
    ad_scorer = (load_outlier_scorer(AD_DEVICE, **OUTLIER_OPTIONS) if AD_FILTER == 'outlier'
                 else load_ad_scorer(AD_FILTER, AD_DEVICE, student_path))
    print('Lọc quảng cáo:', ad_scorer[1])
ocr_gate = load_ocr_gate(AD_DEVICE, threshold=OCR_GATE_THRESHOLD, views=OCR_GATE_VIEWS) if OCR_GATE else None

def speech_embed_model():
    from sentence_transformers import SentenceTransformer
    model = shared_model(('e5', EMBEDDING_MODEL), lambda: SentenceTransformer(
        EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device='cuda:1' if OCR_ENGINE == 'hybrid' else 'cuda'))
    return lambda texts: model.encode(['query: ' + t for t in texts], normalize_embeddings=True,
                                      batch_size=32, convert_to_numpy=True)
speech_embed = speech_embed_model() if SPEECH_RELEVANCE else None

def detect(video_id):
    started = time.monotonic()
    video = find_video_file(video_id)
    stage01, manifest, intervals = run_slide_detection(
        video, OUTPUT_ROOT / video_id, video_id, sample_hz=SAMPLE_HZ,
        fine_hz=FINE_HZ, merge_reveals=MERGE_REVEALS, scan='single_pass',
        ad_scorer=ad_scorer, ad_options={'mode': AD_MODE},
    )
    return video, stage01, manifest, time.monotonic() - started

def ocr(video_id, stage01):
    started = time.monotonic()
    directory, _, _ = run_visual_extraction(
        stage01, OUTPUT_ROOT / video_id, video_id, languages=OCR_LANGUAGES,
        ocr_device='cuda', ocr_scale=OCR_SCALE, ocr_batch_size=OCR_BATCH_SIZE,
        frame_hsv=FRAME_HSV.get(video_id), ocr_engine=OCR_ENGINE,
        hybrid_devices=HYBRID_DEVICES, ocr_gate=ocr_gate, hybrid_rescue_area=HYBRID_RESCUE_AREA,
        hybrid_skip_repeated_decor=HYBRID_SKIP_REPEATED_DECOR,
    )
    return directory, time.monotonic() - started

def asr(video_id):
    # Stage 03 needs only the audio, so it does not wait for stage 01.
    started = time.monotonic()
    directory, _, _ = run_transcription(
        None, find_video_file(video_id), OUTPUT_ROOT / video_id, video_id,
        model_id=ASR_MODEL, revision=ASR_REVISION, language='vi',
        transcription_mode=ASR_MODE, device_index=ASR_DEVICE_INDEX,
    )
    return directory, time.monotonic() - started

def ocr_after_detect(video_id, detect_job):
    return ocr(video_id, detect_job.result()[1])

timings = []
pipeline_started = time.monotonic()
# Three queues run side by side: 01 on the CPU, 03 (Whisper) and 02 (EasyOCR) on
# the GPU. Each queue processes the videos in order; 02 of a video waits for its 01.
with ThreadPoolExecutor(1) as cpu, ThreadPoolExecutor(1) as asr_queue, ThreadPoolExecutor(1) as ocr_queue:
    asr_jobs = {video_id: asr_queue.submit(timed, '03', video_id, asr, video_id) for video_id in VIDEO_IDS}
    detect_jobs = {video_id: cpu.submit(timed, '01', video_id, detect, video_id) for video_id in VIDEO_IDS}
    ocr_jobs = {video_id: ocr_queue.submit(timed, '02', video_id, ocr_after_detect, video_id, detect_jobs[video_id])
                for video_id in VIDEO_IDS}
    for video_id in VIDEO_IDS:
        row = {'video_id': video_id}
        try:
            video, stage01, manifest01, row['01_sec'] = detect_jobs[video_id].result()
            row.update(duration_sec=manifest01['source']['duration_sec'],
                       visual_states=manifest01['visual_state_count'],
                       slides=manifest01['interval_count'] - manifest01.get('ad_segment_count', 0),
                       ad_segments=manifest01.get('ad_segment_count', 0))
            stage02, row['02_sec'] = ocr_jobs[video_id].result()
            visual_manifest = json.loads((stage02 / 'manifest.json').read_text(encoding='utf-8'))
            row['ocr_read'] = visual_manifest.get('ocr_read_count')
            row['ocr_skipped'] = visual_manifest.get('ocr_skipped_no_content')
            row['02_sec_per_slide'] = round(row['02_sec'] / max(row['ocr_read'] or row['slides'], 1), 1)
            stage03, row['03_sec'] = asr_jobs[video_id].result()
            row['ready_at_sec'] = time.monotonic() - pipeline_started
            started = time.monotonic()
            stage04, manifest04, _ = run_alignment(
                stage01, stage02, stage03, OUTPUT_ROOT / video_id, video_id,
                ad_suspect_overlap=AD_SUSPECT_OVERLAP, speech_embed=speech_embed)
            row['04_sec'] = time.monotonic() - started
            started = time.monotonic()
            run_indexing(stage04, OUTPUT_ROOT / video_id, video_id,
                         embedding_model=EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
                         dedupe_previous=OCR_DEDUPE_PREVIOUS)
            row['05_sec'] = time.monotonic() - started
        except Exception:
            traceback.print_exc()
            row['error'] = 'stage 01' if '01_sec' not in row else 'stage 02-05'
        timings.append(row)
        print('PIPELINE', json.dumps(row), flush=True)
total_sec = time.monotonic() - pipeline_started
_gpu_stop.set()
first = {stage: min((s for s in spans if s['stage'] == stage), key=lambda s: s['start'], default=None)
         for stage in ('01', '03')}
asr_parallel = bool(first['01'] and first['03'] and first['03']['start'] < first['01']['end'])
print('GPU_PEAK_MB', json.dumps(gpu_peak_mb))
print('SPANS', json.dumps(sorted(spans, key=lambda s: s['start'])))
print('ASR_PARALLEL_WITH_01', asr_parallel)
write_json(OUTPUT_ROOT / 'fast_pipeline_timings.json',
           {'total_wall_sec': total_sec, 'videos': timings, 'gpu_peak_mb': gpu_peak_mb,
            'spans': spans, 'asr_parallel_with_01': asr_parallel})
print(f'Tổng thời gian: {total_sec:.0f} s cho {len(VIDEO_IDS)} video')
"""

INSPECT = """import pandas as pd
table = pd.DataFrame(timings).set_index('video_id').round(1)
display(table)
# Gói kết quả nhẹ để tải về: bỏ WAV và clip audio (tạo lại được từ video).
export = Path('/kaggle/working/fast_pipeline_export')
shutil.rmtree(export, ignore_errors=True)
shutil.copytree(OUTPUT_ROOT, export, ignore=shutil.ignore_patterns('audio', '*.wav', '_cache'))
archive = shutil.make_archive('/kaggle/working/fast_pipeline_export', 'zip', export)
shutil.rmtree(export)
print('Tải về:', archive)
"""


def build():
    cells = [
        cell("markdown", INTRO),
        cell("markdown", "## 1. Môi trường\n\nBật **Internet** và GPU **T4**."),
        cell("code", "%pip install -q " + DEPS),
        cell("markdown", "## 2. Hàm xử lý\n\nNhúng nguyên văn từ `notebook_code/`."),
    ]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True)
              for name in MODULES]
    cells += [
        cell("markdown", "## 3. Cấu hình"),
        cell("code", CONFIG),
        cell("markdown", "## 4. Chạy 01 → 05 cho mọi video"),
        cell("code", RUN),
        cell("markdown", "## 5. Thời gian và gói kết quả"),
        cell("code", INSPECT),
    ]
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    target = ROOT / "notebooks" / f"{NAME}.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    print(target.name)


if __name__ == "__main__":
    build()
