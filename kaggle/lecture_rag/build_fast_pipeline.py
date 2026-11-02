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

# 10 video, nhiều môn: Sử, Toán, Hóa, Lý, Kinh tế, Lập trình, XSTK, Toán lớp 2, Khoa học máy tính (tiếng Anh, SlideSpeech)
VIDEO_IDS = ['L25_V011', 'L25_V037', 'L25_V008', '_eDGTHhyQ7M', 'tHE1mC_n0-E', 'v2YTz8ToEag',
             'y1CvB950VlU', 'hVU5Mw19V_c', 'D_kuulxKuxc', 'cs_0001']
# Ngôn ngữ lời giảng (ASR) và prompt OCR / mô tả hình; video không ghi là 'vi'.
VIDEO_LANGUAGES = {'cs_0001': 'en'}
OUTPUT_ROOT = Path('/kaggle/working/lecture_rag')

# 01 · slide
SAMPLE_HZ = 1.0
FINE_HZ = 8.0  # Tần số giữ frame nhỏ để tinh chỉnh ranh giới, không cần tua
MERGE_REVEALS = True
EXPORT_STATE_ONSETS = True
MERGE_SAME_CONTENT = True
# Only these audited layouts use board/editor heuristics (STAGE01.md).
STAGE01_VIDEO_OPTIONS = {
    'hVU5Mw19V_c': {
        'board_options': {'roi': (0.0, 0.08, 1.0, 0.96), 'window_samples': 5,
                          'check_every_sec': 5.0, 'min_state_sec': 15.0,
                          'snapshot_sec': 120.0, 'minimum_pixels': 24},
        'keyframe_options': {'roi': (0.0, 0.08, 1.0, 0.96), 'mode': 'board', 'complementary': True},
        'merge_reveals': False,
    },
    'y1CvB950VlU': {
        'reveal_thresholds': {'replacement_safe': True, 'directional_threshold': 0.94,
                              'deletion_minimum_pixels': 12, 'deletion_fraction': 0.20},
        'keyframe_options': {'roi': (0.26, 0.08, 0.99, 0.57)},
    },
}
# Lọc quảng cáo: 'student' (SigLIP base + hồi quy logistic, file ad_student.npz từ thí nghiệm
# bộ lọc), 'teacher' (SigLIP so400m zero-shot) hoặc None để tắt.
# Mặc định KHÔNG dùng bộ lọc quảng cáo học từ L25 (không tổng quát sang kênh khác); bộ gác OCR
# zero-shot bên dưới là lớp lọc nội dung duy nhất. Thử nghiệm: 'student', 'teacher', 'outlier', 'slide_speech'.
AD_FILTER = None
# 'slide_speech': không dùng model học từ L25. Bước 01 chấm mỗi giây "có nội dung bài giảng" (SigLIP
# zero-shot, cùng prompt bộ gác OCR); bước 04 đánh dấu giây vừa không giống slide vừa có lời lạc chủ
# đề (E5). Cần SPEECH_RELEVANCE = True. Ngưỡng: SLIDE_SPEECH.
SLIDE_SPEECH = {'not_slide': 0.5, 'topic_z': -1.5}
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

# 02 · OCR. 'qwen' = Qwen3-VL-4B đọc cả slide một lần (chữ, LaTeX, bảng, dòng [Hình] mô tả hình),
# mỗi GPU một bản: so trên 2 video tiếng Việt + 2 tiếng Anh (13/10) nhanh hơn hybrid ×1,8–7, chữ sạch
# hơn (không lặp dòng, đúng dấu hơn). 'hybrid' = Unlimited-OCR (GPU 0) + Qwen3-VL đọc từng vùng hình
# (GPU 1); 'easyocr' = bản cũ. Các mục HYBRID_* chỉ dùng cho 'hybrid'.
# Qwen crops audited content ROIs or the expanded screen ROI exported by stage 01;
# if neither is available it retains the full frame. Never use tight numeric audit crops here.
OCR_ENGINE = 'qwen'
QWEN_OCR_MODEL = 'Qwen/Qwen3-VL-4B-Instruct'
QWEN_OCR_REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
QWEN_CONTENT_CROP = True
QWEN_CONTENT_ROIS = {'hVU5Mw19V_c': (0.0, 0.08, 1.0, 0.96),
                     'y1CvB950VlU': (0.25, 0.05, 0.95, 0.76)}
QWEN_MIN_PIXELS = 256 * 32 * 32
QWEN_MAX_PIXELS = 1280 * 32 * 32
QWEN_VALIDATE = True
QWEN_RETRY_INVALID = True  # Exactly one literal-only retry: 2048 tokens, repetition penalty 1.10.
HYBRID_DEVICES = ('cuda:0', 'cuda:1')
OCR_LANGUAGES = ('vi', 'en')
OCR_SCALE = 1.5
OCR_BATCH_SIZE = 32
FRAME_HSV = {'L25_V001': ((135, 80, 120), (175, 255, 255))}
# Vùng "ảnh" chiếm từ tỉ lệ này của slide được Unlimited-OCR đọc lại riêng (cứu chữ bị coi là hình).
# 0.4 = bản gốc; chỉ đổi sau khi đo bằng gold set 2 (gold/eval_ocr.py) trên tập dev.
HYBRID_RESCUE_AREA = 0.4
# Không gửi lại cho Qwen vùng đã bị gọi là trang trí trong video (logo, ảnh giảng viên):
# L25 9 video bỏ 61/314 lần gọi Qwen, bỏ nhầm 2 vùng.
HYBRID_SKIP_REPEATED_DECOR = True
# Vùng hình nhỏ hơn tỉ lệ này của slide (icon, gạch đầu dòng) không gửi cho Qwen.
HYBRID_MIN_REGION_AREA = 0.02
# Keyframe gần như trùng một keyframe đã OCR trong video (slide chiếu lại) dùng lại kết quả.
REUSE_NEAR_DUPLICATES = True
# Unlimited-OCR (GPU 0) đọc slide sau trong lúc Qwen3-VL (GPU 1) đọc các vùng hình của slide trước
# theo lô. speed_bench3 (12/10, 3 video L25): cả 4 mục này ×1,31, giữ 100% từ OCR; demo V011 ×1,52,
# cs_0001 (tiếng Anh) ×1,40.
HYBRID_PIPELINED = True

# 03 · ASR: experiment-pinned CT2 turbo; FP16 is selected on CUDA by load_asr_model.
ASR_MODEL = 'dropbox-dash/faster-whisper-large-v3-turbo'
ASR_REVISION = '0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf'
# GPU 1: Qwen ~9.1 GB + turbo ~2.2 GB + shared E5. One ASR window and OCR image at a time.
ASR_DEVICE_INDEX = 0 if OCR_ENGINE == 'easyocr' else 1
ASR_MODE = 'windowed'
ASR_BATCH_SIZE = 1  # Conservative if opting into VAD batching; ignored in windowed mode.

# 04 · đoạn quảng cáo có lời nói giống bài giảng (cặp từ trùng với phần còn lại của bài/slide)
# vẫn giữ lời giảng và chỉ mục (ad_suspect). None = tắt (bản gốc). L25 dev: 0.25 -> 0/27 quảng
# cáo bị giữ oan, giữ được 103/110 đoạn bài giảng bị cắt nhầm; 0.20 -> 1/27 và 110/110.
AD_SUSPECT_OVERLAP = None
# Lọc nội dung B: chấm lời giảng có thuộc chủ đề bài không (E5, chuẩn hóa trong video). Chỉ ghi
# speech_topic vào segments.json, chưa dùng để quyết định. L25 9 video: AUC 0,94 tách quảng cáo.
SPEECH_RELEVANCE = True

# Phase 3 · CPU text processing; raw stage-02/03 outputs remain available for retries.
CONSERVATIVE_OCR = 'canonical_subscripts'  # Preserve layout; plain digits for formula subscripts outside code.
HEADING_TITLES = False
ASR_CORRECTION = 'glossary_code'  # Reviewed technical terms + literal local print feedback; raw ASR stays.
FAILURE_GUARD = True
SPEECH_AD_FILTER = 'school_signatures'  # Explicit SIU/Asian School opening/closing phrases; keeps mixed speech.
INCLUDE_GENERATED_DESCRIPTIONS = False
GENERATED_SEMANTIC_ONLY = False  # Generated descriptions receive only the semantic RRF branch.

# 05 · chỉ lập chỉ mục dòng OCR chưa có ở slide liền trước (L25: 32% dòng lặp lại). Tắt cho tới khi
# đo hỏi đáp (eval_retrieval.py) trên lần chạy lại 10 video.
OCR_DEDUPE_PREVIOUS = False

# 05 · chỉ mục
EMBEDDING_MODEL = 'intfloat/multilingual-e5-small'   # bge-m3 lost on the 66-question set (hit@1 53 vs 55, hit@3 62 vs 65)
EMBEDDING_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
EMBEDDING_DEVICE = 'cuda:0' if OCR_ENGINE == 'easyocr' else 'cuda:1'
"""

# Reuse the production settings in the single-video builders without copying defaults.
STAGE01_CONFIG = CONFIG.split('MERGE_REVEALS = True\n', 1)[1].split('# Lọc quảng cáo:', 1)[0]
QWEN_CONFIG = CONFIG.split("OCR_ENGINE = 'qwen'\n", 1)[1].split('HYBRID_DEVICES =', 1)[0]
ASR_CONFIG = '# 03 · ASR:' + CONFIG.split('# 03 · ASR:', 1)[1].split('# 04 ·', 1)[0]

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
if AD_FILTER and AD_FILTER != 'slide_speech':
    student_path = AD_STUDENT_PATH or (next(iter(sorted(Path('/kaggle/input').rglob('ad_student.npz'))), None)
                                       if AD_FILTER == 'student' else None)
    if AD_FILTER == 'student' and student_path is None:
        raise FileNotFoundError('AD_FILTER="student" cần ad_student.npz: Add Input output của thí nghiệm bộ lọc.')
    ad_scorer = (load_outlier_scorer(AD_DEVICE, **OUTLIER_OPTIONS) if AD_FILTER == 'outlier'
                 else load_ad_scorer(AD_FILTER, AD_DEVICE, student_path))
    print('Lọc quảng cáo:', ad_scorer[1])
ocr_gate = load_ocr_gate(AD_DEVICE, threshold=OCR_GATE_THRESHOLD, views=OCR_GATE_VIEWS) if OCR_GATE else None
content_scorer = load_ocr_gate(AD_DEVICE) if AD_FILTER == 'slide_speech' else None
if AD_FILTER == 'slide_speech' and not SPEECH_RELEVANCE:
    raise ValueError("AD_FILTER='slide_speech' cần SPEECH_RELEVANCE = True.")

def speech_embed_model():
    from sentence_transformers import SentenceTransformer
    model = shared_model(('e5', EMBEDDING_MODEL, EMBEDDING_REVISION, EMBEDDING_DEVICE), lambda: SentenceTransformer(
        EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=EMBEDDING_DEVICE))
    return lambda texts: model.encode(['query: ' + t for t in texts], normalize_embeddings=True,
                                      batch_size=32, convert_to_numpy=True)
speech_embed = speech_embed_model() if SPEECH_RELEVANCE else None

def detect(video_id):
    started = time.monotonic()
    video = find_video_file(video_id)
    options = dict(merge_reveals=MERGE_REVEALS, export_state_onsets=EXPORT_STATE_ONSETS,
                   merge_same_content=MERGE_SAME_CONTENT)
    options.update(STAGE01_VIDEO_OPTIONS.get(video_id, {}))
    stage01, manifest, intervals = run_slide_detection(
        video, OUTPUT_ROOT / video_id, video_id, sample_hz=SAMPLE_HZ,
        fine_hz=FINE_HZ, scan='single_pass', **options,
        ad_scorer=ad_scorer, ad_options={'mode': AD_MODE}, content_scorer=content_scorer,
    )
    return video, stage01, manifest, time.monotonic() - started

def ocr(video_id, stage01):
    started = time.monotonic()
    directory, _, _ = run_visual_extraction(
        stage01, OUTPUT_ROOT / video_id, video_id, languages=OCR_LANGUAGES,
        ocr_device='cuda', ocr_scale=OCR_SCALE, ocr_batch_size=OCR_BATCH_SIZE,
        frame_hsv=FRAME_HSV.get(video_id), ocr_engine=OCR_ENGINE,
        hybrid_devices=HYBRID_DEVICES, ocr_gate=ocr_gate, hybrid_rescue_area=HYBRID_RESCUE_AREA,
        hybrid_skip_repeated_decor=HYBRID_SKIP_REPEATED_DECOR, hybrid_min_region_area=HYBRID_MIN_REGION_AREA,
        reuse_near_duplicates=REUSE_NEAR_DUPLICATES, hybrid_pipelined=HYBRID_PIPELINED,
        describe_language=VIDEO_LANGUAGES.get(video_id, 'vi'),
        qwen_model=QWEN_OCR_MODEL, qwen_revision=QWEN_OCR_REVISION,
        qwen_content_crop=QWEN_CONTENT_CROP, qwen_content_rois=QWEN_CONTENT_ROIS,
        qwen_min_pixels=QWEN_MIN_PIXELS, qwen_max_pixels=QWEN_MAX_PIXELS,
        qwen_validate=QWEN_VALIDATE, qwen_retry_invalid=QWEN_RETRY_INVALID,
    )
    return directory, time.monotonic() - started

def asr(video_id):
    # Stage 03 needs only the audio, so it does not wait for stage 01.
    started = time.monotonic()
    directory, _, _ = run_transcription(
        None, find_video_file(video_id), OUTPUT_ROOT / video_id, video_id,
        model_id=ASR_MODEL, revision=ASR_REVISION, language=VIDEO_LANGUAGES.get(video_id, 'vi'),
        transcription_mode=ASR_MODE, batch_size=ASR_BATCH_SIZE, device_index=ASR_DEVICE_INDEX,
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
            row['ocr_failed'] = visual_manifest.get('ocr_failed_count', 0)
            row['02_sec_per_slide'] = round(row['02_sec'] / max(row['ocr_read'] or row['slides'], 1), 1)
            stage03, row['03_sec'] = asr_jobs[video_id].result()
            row['ready_at_sec'] = time.monotonic() - pipeline_started
            started = time.monotonic()
            stage04, manifest04, _ = run_alignment(
                stage01, stage02, stage03, OUTPUT_ROOT / video_id, video_id,
                ad_suspect_overlap=AD_SUSPECT_OVERLAP, speech_embed=speech_embed,
                slide_speech=SLIDE_SPEECH if AD_FILTER == 'slide_speech' else None,
                conservative_ocr=CONSERVATIVE_OCR, heading_titles=HEADING_TITLES,
                asr_correction=ASR_CORRECTION, failure_guard=FAILURE_GUARD,
                speech_ad_filter=SPEECH_AD_FILTER)
            row['04_sec'] = time.monotonic() - started
            started = time.monotonic()
            run_indexing(stage04, OUTPUT_ROOT / video_id, video_id,
                         embedding_model=EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
                         device=EMBEDDING_DEVICE,
                         dedupe_previous=OCR_DEDUPE_PREVIOUS,
                         include_generated_descriptions=INCLUDE_GENERATED_DESCRIPTIONS,
                         generated_semantic_only=GENERATED_SEMANTIC_ONLY)
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
