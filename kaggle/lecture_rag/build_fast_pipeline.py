"""Generate fast_pipeline_l25.ipynb: stages 01-05 for many videos in ONE Kaggle session.

Speed comes from: single-pass stage 01 (no seeks); ASR (03) needing only the audio, so
it runs from the start in parallel with 01 and OCR (02); each stage working through
the videos in its own queue; and installing packages once. Run: python build_fast_pipeline.py
"""

from pathlib import Path
import json
import ast
import hashlib

from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
NAME = "fast_pipeline_l25"
MODULES = ["common.py", "detection.py", "adfilter.py", "visual.py", "transcription.py", "textclean.py",
           "alignment.py", "indexing.py", "production_runtime.py"]
DEPS = ('"av==15.1.0" "opencv-python-headless>=4.10,<4.12" "Pillow==11.3.0" '
        '"easyocr>=1.7.2,<2" "faster-whisper==1.2.1" "ctranslate2==4.8.3" "huggingface-hub==0.36.0" '
        '"sentence-transformers>=3.4,<6" "transformers==4.57.1" '
        # OCR kết hợp: Unlimited-OCR (remote code) + Qwen3-VL
        '"einops" "addict" "easydict" "accelerate==1.11.0"')

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
RESUME_FROM = None  # Optional attached extracted full checkpoint directory.
CANARY = False

# 01 · slide
SAMPLE_HZ = 1.0
FINE_HZ = 8.0  # Tần số giữ frame nhỏ để tinh chỉnh ranh giới, không cần tua
MERGE_REVEALS = True
EXPORT_STATE_ONSETS = True
MERGE_SAME_CONTENT = True
# Retrieval ablation: keep editor safety and the first definition observation (ABLATION_20261102.md).
STAGE01_VIDEO_OPTIONS = {
    'L25_V008': {'merge_same_content': False},
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
ASR_MODEL = 'Systran/faster-whisper-medium'
ASR_REVISION = '08e178d48790749d25932bbc082711ddcfdfbc4f'
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
VISUAL_STATE_TIMING = True

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

RUN = (CODE / "production_queue.py").read_text(encoding="utf-8")

PREFLIGHT = """session_started = time.monotonic()
preflight = startup_preflight(VIDEO_IDS, VIDEO_LANGUAGES, OUTPUT_ROOT)
"""

INSPECT = """if not CANARY:
    print('ARCHIVES', json.dumps(package_production_outputs(OUTPUT_ROOT, '/kaggle/working')))
write_json(OUTPUT_ROOT / 'session.json', {
    'elapsed_seconds': time.monotonic() - session_started,
    'finished_utc': datetime.now(timezone.utc).isoformat(), 'devices': 2, 'canary': CANARY})
"""


def build(target=None, *, canary=False, overrides=None):
    config = CONFIG + '\n' + '\n'.join(f'{k} = {v!r}' for k, v in (overrides or {}).items())
    config += f'\nCANARY = {canary!r}\n'
    recipe = hashlib.sha256((''.join((CODE / n).read_text(encoding='utf-8') for n in MODULES)
                             + RUN).encode('utf-8')).hexdigest()
    config += f'CODE_RECIPE_SHA256 = {recipe!r}\n'
    names = [n.targets[0].id for n in ast.parse(CONFIG).body if isinstance(n, ast.Assign)]
    names = [n for n in names if n not in {'OUTPUT_ROOT', 'RESUME_FROM', 'CANARY'}]
    config += f'PRODUCTION_RECIPE = fingerprint({{k: str(globals()[k]) for k in {names + ["CODE_RECIPE_SHA256"]!r}}})\n'
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
        cell("code", config),
        cell("markdown", "## Startup preflight · before model loading"),
        cell("code", PREFLIGHT),
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
    target = Path(target) if target else ROOT / "notebooks" / f"{NAME}.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    print(target.name)


if __name__ == "__main__":
    build()
