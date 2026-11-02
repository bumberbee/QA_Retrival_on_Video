"""Generate self-contained Kaggle notebooks; never run or upload models."""

from pathlib import Path
import hashlib
import json
import textwrap

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "notebook_code"
EMBEDDING_DEPS = '"sentence-transformers>=3.4,<6" "transformers>=4.51.3,<5"'

SPECS = [
    {
        "name": "01_detect_slides_l25_v001", "module": "detection.py",
        "title": "01 · Phát hiện slide và trích keyframe",
        "intro": """**Input:** dataset thnhtrungnguynmif/l25a-lecture-videos-media.
**Thiết bị:** CPU. **Output:** 01_slides.

Quét frame theo PTS, so sánh màu và cạnh chữ/bố cục, xác nhận thay đổi ổn định
rồi tìm lại ranh giới với mật độ cao hơn. Chọn frame rõ trong mỗi đoạn.
Không chia đều video theo số ảnh. Đây là các đoạn hình ảnh ứng viên:
notebook 02 sẽ phân loại slide, bảng, người giảng và cảnh khác.
Điều chỉnh ROI/vùng bỏ qua nếu có người giảng, phụ đề hoặc logo động.""",
        "deps": '"av>=14,<17" "opencv-python-headless>=4.10,<4.12" "Pillow>=10"',
        "config": """VIDEO_PATH = None
SAMPLE_HZ = 1.0
SLIDE_ROI = (0.0, 0.0, 1.0, 1.0)  # left, top, right, bottom; chuẩn hóa 0..1
IGNORED_REGIONS = []  # Chỉ thêm vùng đã quan sát, tránh che nhầm nội dung bài
CHANGE_THRESHOLD = 0.055
STABILITY_THRESHOLD = 0.035
STABLE_SAMPLES = 2
MIN_INTERVAL_SEC = 2.0
MANUAL_BOUNDARIES_SEC = None  # Nếu sửa tay: [0.0, ..., đúng duration của video]
MERGE_REVEALS = True  # Gộp slide hiện dần (animation thêm dòng/hình) thành một đoạn
REVEAL_THRESHOLDS = {}  # Ví dụ {'retained_threshold': 0.9}; mặc định trong reveal_decision
USE_SLIDE_CLASSIFIER = False  # ResNet-34 Lecture2Notes: chặn gộp qua cảnh giảng viên (p_slide < 0.2)
MERGE_OCR = False  # Tesseract vie+eng: tách khi chữ bị thay, gộp khi chữ cũ còn giữ
""",
        "run": """video_path = find_video(VIDEO_PATH)
output_dir, manifest, intervals = run_slide_detection(
    video_path, OUTPUT_ROOT, VIDEO_ID, sample_hz=SAMPLE_HZ,
    roi=SLIDE_ROI, ignored_regions=IGNORED_REGIONS,
    change_threshold=CHANGE_THRESHOLD, stability_threshold=STABILITY_THRESHOLD,
    stable_samples=STABLE_SAMPLES, min_interval_sec=MIN_INTERVAL_SEC,
    manual_boundaries=MANUAL_BOUNDARIES_SEC,
    merge_reveals=MERGE_REVEALS, reveal_thresholds=REVEAL_THRESHOLDS,
    slide_classifier=load_slide_classifier() if USE_SLIDE_CLASSIFIER else None,
    merge_ocr=MERGE_OCR,
)
print('Video:', video_path)
print('Duration:', manifest['source']['duration_sec'])
print('SHA-256:', manifest['source']['sha256'])
print('Số trạng thái hình:', manifest['visual_state_count'],
      '| Số đoạn sau gộp:', len(intervals),
      '| Slide hiện dần đã gộp:', manifest['merged_reveal_count'])
""",
        "inspect": """from IPython.display import Image, display
display(Image(filename=str(output_dir / 'contact_sheet.jpg')))
if manifest['reveal_sheet']:
    print('Mỗi hàng: các trạng thái của một slide hiện dần, ảnh cuối là keyframe.')
    display(Image(filename=str(output_dir / manifest['reveal_sheet'])))
for item in intervals[:12]:
    print(item['segment_id'], f"{item['start_sec']:.3f}–{item['end_sec']:.3f}s",
          'keyframe:', item['keyframe_time_sec'])
""",
        "next": """Kiểm tra ranh giới với video gốc, đặc biệt slide có animation/bảng viết tay.
Sampling 1 Hz và đoạn tối thiểu 2 giây có thể bỏ qua nội dung xuất hiện rất ngắn.
Tăng sampling hoặc sửa MANUAL_BOUNDARIES_SEC khi cần.

**Save Version → Save & Run All**. Notebook **02 và 03**: **Add Input → Notebook
Output** của bước này. Giữ cùng một phiên bản cho cả hai nhánh.""",
    },
    {
        "name": "02_ocr_describe_slides_l25_v001", "module": "visual.py",
        "title": "02 · OCR và mô tả hình slide",
        "intro": """**Input:** Output 01. **Thiết bị:** GPU T4. **Output:** 02_visual.

Luồng nhanh OCR vùng slide trên GPU, ghép chữ theo dòng và giữ ảnh nguồn.
Mặc định hoãn mô tả sinh bằng vision, giữ ảnh để không chặn quá trình tạo chỉ mục.
Đặt VISION_MODE='selected' và VISION_SEGMENT_IDS để mô tả một số ảnh cần kiểm tra.
Chế độ on_demand chưa có worker tự nhận yêu cầu từ ứng dụng.
Nhãn cảnh từ bố cục là heuristic, không phải nhãn chuẩn do con người xác nhận.
Profile viền màu chỉ dành cho bố cục đã quan sát trong L25_V001.""",
        "deps": '"easyocr>=1.7.2,<2" "opencv-python-headless>=4.10,<4.12" '
                '"transformers>=4.51.3,<5" "accelerate>=1,<2" "Pillow>=10"',
        "config": """STAGE_01_DIR = None
OCR_LANGUAGES = ('vi', 'en')
OCR_DEVICE = 'cuda'
OCR_SCALE = 1.5
OCR_BATCH_SIZE = 32
DOCUMENT_FRAME_HSV = ((135, 80, 120), (175, 255, 255))  # Viền slide L25; None cho video khác
VISION_MODE = 'on_demand'  # 'selected' hoặc 'all' nếu cần mô tả sinh bằng AI
VISION_SEGMENT_IDS = []
VISION_MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
VISION_REVISION = 'main'  # Dùng commit SHA khi làm thực nghiệm chính thức
ALLOW_CPU_VISION = False  # CPU rất chậm và cần nhiều RAM
CONTENT_KIND_OVERRIDES = {}  # Ví dụ {'L25_V001:seg_00001': 'speaker'}
""",
        "run": """stage01_dir, _ = find_stage(1, STAGE_01_DIR, VIDEO_ID)
output_dir, manifest, visuals = run_visual_extraction(
    stage01_dir, OUTPUT_ROOT, VIDEO_ID, languages=OCR_LANGUAGES,
    vision_model=VISION_MODEL, vision_revision=VISION_REVISION,
    allow_cpu=ALLOW_CPU_VISION, content_kind_overrides=CONTENT_KIND_OVERRIDES,
    vision_mode=VISION_MODE, vision_segment_ids=VISION_SEGMENT_IDS,
    ocr_device=OCR_DEVICE, ocr_scale=OCR_SCALE, ocr_batch_size=OCR_BATCH_SIZE,
    frame_hsv=DOCUMENT_FRAME_HSV,
)
print('Số frame đã xử lý:', len(visuals))
""",
        "inspect": """from IPython.display import Image, display
by_id = {item['segment_id']: item for item in read_json(stage01_dir / 'intervals.json')}
for item in visuals[:3]:
    display(Image(filename=str(stage01_dir / by_id[item['segment_id']]['image_path']), width=850))
    print(item['segment_id'], item['content_kind'], item['title'])
    print('OCR:', item['ocr_text'][:1500])
    print('Mô tả:', item['visual_description'])
    print('Cần đối chiếu:', item['uncertainties'])
""",
        "next": """Save Version để notebook **04** dùng Output này. Nhánh **03** có thể
chạy độc lập sau 01. Kiểm tra/sửa CONTENT_KIND_OVERRIDES nếu heuristic nhầm loại
nội dung. OCR và mô hình nhỏ chưa bảo đảm đọc đúng mọi công thức/bảng.""",
    },
    {
        "name": "03_asr_audio_l25_v001", "module": "transcription.py",
        "title": "03 · Chuẩn hóa audio và ASR có timestamp từng từ",
        "intro": """**Input:** Output 01 và dataset video gốc.
**Thiết bị:** GPU khuyến nghị, hoặc CPU int8. **Output:** 03_audio.

Chuẩn hóa mono 16 kHz nhưng giữ offset audio với video. Gom core window 20–60
giây, ưu tiên khoảng lặng gần 45 giây, đệm 1,25 giây hai bên. Faster-Whisper
medium dùng VAD, xuất word timestamps; giữ từ theo midpoint trong core rồi khử trùng.
Không ngắt STT ở mọi lần chuyển slide. Vẫn xuất WAV theo mỗi visual interval
để nghe đối chiếu; notebook 04 mới gán từ về slide. Notebook cũ giữ làm baseline.
ASR_MODE='batched' là ablation dùng VAD batching, không dùng tham số cửa sổ.
INITIAL_PROMPT mặc định để trống sau khi quan sát glossary lọt vào transcript.""",
        "deps": '"faster-whisper>=1.1,<2" "huggingface-hub>=0.25,<1"',
        "config": """STAGE_01_DIR = None
VIDEO_PATH = None
ASR_MODEL = 'Systran/faster-whisper-medium'
ASR_REVISION = '08e178d48790749d25932bbc082711ddcfdfbc4f'
ASR_MODE = 'windowed'  # Cửa sổ có overlap; 'batched' là tùy chọn cần đánh giá chất lượng
ASR_BATCH_SIZE = 8  # Chỉ dùng khi ASR_MODE='batched'
LANGUAGE = 'vi'
ASR_DEVICE = 'auto'  # 'cuda' hoặc 'cpu'
INITIAL_PROMPT = None  # Chỉ thử glossary trong ablation; prompt có thể lọt vào transcript
TARGET_WINDOW_SEC = 45.0
MIN_WINDOW_SEC = 20.0
MAX_WINDOW_SEC = 60.0
OVERLAP_SEC = 1.25
SILENCE_THRESHOLD_DB = -38.0
""",
        "run": """stage01_dir, _ = find_stage(1, STAGE_01_DIR, VIDEO_ID)
video_path = find_video(VIDEO_PATH)
output_dir, manifest, transcript = run_transcription(
    stage01_dir, video_path, OUTPUT_ROOT, VIDEO_ID,
    model_id=ASR_MODEL, revision=ASR_REVISION, language=LANGUAGE,
    device=ASR_DEVICE, initial_prompt=INITIAL_PROMPT,
    target_sec=TARGET_WINDOW_SEC, minimum_sec=MIN_WINDOW_SEC,
    maximum_sec=MAX_WINDOW_SEC, overlap_sec=OVERLAP_SEC,
    silence_threshold_db=SILENCE_THRESHOLD_DB,
    transcription_mode=ASR_MODE, batch_size=ASR_BATCH_SIZE,
)
print('Số từ:', len(transcript['words']))
print('Trùng đã loại:', transcript['overlap_duplicates_removed'])
print('Word timing không hợp lệ:', transcript['invalid_word_count'])
""",
        "inspect": """for word in transcript['words'][:30]:
    print(f"{word['start_sec']:.3f}–{word['end_sec']:.3f}", word['text'])
if transcript['slide_audio_clips']:
    import wave
    import numpy as np
    from IPython.display import Audio, display
    clip = transcript['slide_audio_clips'][0]
    with wave.open(str(output_dir / clip['path']), 'rb') as audio_file:
        sample = np.frombuffer(audio_file.readframes(16000 * 10), dtype='<i2')
    print('Nghe tối đa 10 giây đầu clip từ:', clip['start_sec'], 's')
    display(Audio(sample, rate=16000))
""",
        "next": """Save Version để notebook **04** dùng Output này. Nghe câu nói ngay
trước/sau chuyển slide; timestamp phải khớp video chứ không chỉ WAV. Nếu runtime
Kaggle không tương thích CUDA/cuDNN của CTranslate2, đổi ASR_DEVICE='cpu'.
Lỗi không được đánh dấu là bước hoàn tất.""",
    },
    {
        "name": "04_sync_lecture_l25_v001", "module": "alignment.py", "extra_modules": ["textclean.py", "adfilter.py"],
        "title": "04 · Đồng bộ slide và transcript",
        "intro": """**Input:** Output 01 + 02 + 03. **Thiết bị:** CPU. **Output:** 04_bundle.

Kiểm tra checksum video và phiên bản phát hiện slide của hai nhánh. Gán mỗi từ
vào interval chứa midpoint, giữ occurrence lặp lại và previous/next IDs.
Span thiếu word timing được tham chiếu chung khi qua ranh giới; không ước lượng
thời gian từng từ. Gói kết quả chứa hình ảnh, vùng crop và audio theo slide.""",
        "deps": "",
        "config": """STAGE_01_DIR = None
STAGE_02_DIR = None
STAGE_03_DIR = None
CONSERVATIVE_OCR = 'canonical_subscripts'
HEADING_TITLES = False  # Heuristic remains opt-in; see eval/PHASE3.md
ASR_CORRECTION = 'glossary_code'
FAILURE_GUARD = True
SPEECH_AD_FILTER = 'school_signatures'
""",
        "run": """stage01_dir, _ = find_stage(1, STAGE_01_DIR, VIDEO_ID)
stage02_dir, _ = find_stage(2, STAGE_02_DIR, VIDEO_ID)
stage03_dir, _ = find_stage(3, STAGE_03_DIR, VIDEO_ID)
output_dir, manifest, segments = run_alignment(
    stage01_dir, stage02_dir, stage03_dir, OUTPUT_ROOT, VIDEO_ID,
    conservative_ocr=CONSERVATIVE_OCR, heading_titles=HEADING_TITLES,
    asr_correction=ASR_CORRECTION, failure_guard=FAILURE_GUARD,
    speech_ad_filter=SPEECH_AD_FILTER,
)
print('Số đoạn đồng bộ:', len(segments), '| Số từ:', manifest['word_count'])
""",
        "inspect": """from IPython.display import Image, display
for segment in segments[:3]:
    print(segment['segment_id'], segment['start_sec'], segment['end_sec'])
    display(Image(filename=str(output_dir / segment['visual']['image_path']), width=850))
    print('Loại:', segment['content_kind'])
    print('OCR:', segment['visual']['ocr_text'][:600])
    print('Lời giảng:', segment['speech']['text'][:1200])
    print('Trạng thái:', segment['speech']['status'])
""",
        "next": """Save Version; notebook **05** chỉ cần Output này.
transcript_app.json tương thích format transcript backend sau khi ánh xạ
video_id sang ID lúc upload. Notebook không tự sửa dữ liệu của web app.""",
    },
    {
        "name": "05_build_search_index_l25_v001", "module": "indexing.py",
        "title": "05 · Chia bằng chứng và tạo chỉ mục",
        "intro": """**Input:** Output 04. **Thiết bị:** CPU hoặc GPU. **Output:** 05_index.

Gom lời giảng cùng interval thành chunk, giữ OCR và ảnh làm bằng chứng.
Mặc định chỉ thêm mô tả ảnh/vùng nội dung đã được con người xác minh.
Đếm token bằng tokenizer E5, tính cả tiêu đề và prefix để không cắt ngầm.
Lưu vectors chuẩn hóa cùng SQLite FTS5, model revision và thứ tự chunk IDs.
Output kèm dữ liệu nguồn để bước 06 chỉ cần đính kèm một notebook.""",
        "deps": EMBEDDING_DEPS,
        "config": """STAGE_04_DIR = None
EMBEDDING_MODEL = 'intfloat/multilingual-e5-small'
EMBEDDING_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
CHUNK_TOKENS = 350
OVERLAP_TOKENS = 40
BATCH_SIZE = 32
EMBEDDING_DEVICE = None  # Tự chọn; hoặc 'cpu'/'cuda'
SPEECH_CHUNK_TOKENS = 180
SPEECH_MAXIMUM_SECONDS = 50.0
INCLUDE_GENERATED_DESCRIPTIONS = False  # Chỉ bật cho ablation; JSON hợp lệ vẫn có thể sai nội dung
GENERATED_SEMANTIC_ONLY = False  # Bật cùng mô tả tự động để chỉ dùng nhánh semantic
""",
        "run": """stage04_dir, _ = find_stage(4, STAGE_04_DIR, VIDEO_ID)
output_dir, manifest, chunks = run_indexing(
    stage04_dir, OUTPUT_ROOT, VIDEO_ID,
    embedding_model=EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
    chunk_tokens=CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS,
    batch_size=BATCH_SIZE, device=EMBEDDING_DEVICE,
    speech_chunk_tokens=SPEECH_CHUNK_TOKENS, speech_maximum_seconds=SPEECH_MAXIMUM_SECONDS,
    include_generated_descriptions=INCLUDE_GENERATED_DESCRIPTIONS,
    generated_semantic_only=GENERATED_SEMANTIC_ONLY,
)
print('Số chunk:', len(chunks), '| Chiều vector:', manifest['parameters']['dimensions'])
""",
        "inspect": """from collections import Counter
print('Nguồn:', dict(Counter(chunk['source_type'] for chunk in chunks)))
print('Token tối đa:', max(chunk['encoded_tokens'] for chunk in chunks))
for chunk in chunks[:5]:
    print(chunk['chunk_id'], chunk['start_sec'], chunk['end_sec'])
    print(chunk['text'][:450], '\\n')
""",
        "next": """Save Version; notebook **06** chỉ cần Output này. Đổi embedding
model cần tạo lại chỉ mục. Đây là baseline tìm hình qua OCR/mô tả; chưa dùng
image embeddings. Kết quả vẫn trả lại hình gốc để đối chiếu.""",
    },
    {
        "name": "06_retrieve_lecture_l25_v001", "module": "retrieval.py",
        "title": "06 · Truy xuất nội dung theo câu hỏi",
        "intro": """**Input:** Output 05. **Thiết bị:** CPU đủ dùng. **Output:** 06_retrieval.

Kết hợp cosine search với FTS5 bằng reciprocal rank fusion. Bổ sung có giới hạn
lời giảng/ảnh cùng slide và đoạn liền trước/sau, khử trùng rồi sắp theo thời gian.
Trả evidence IDs, timestamp và ảnh nguồn. Kết thúc ở **retrieval**, chưa sinh
câu trả lời bằng LLM; cần kiểm tra đúng bằng chứng trước khi nối bước trả lời.""",
        "deps": EMBEDDING_DEPS,
        "config": """STAGE_05_DIR = None
QUERY_DEVICE = None
QUESTION = 'Bài giảng phân tích nhân vật người đàn bà hàng chài như thế nào?'
CURRENT_TIME_SEC = None  # Điền nếu hỏi 'giải thích đoạn/slide này'
PREFER_CURRENT_SEGMENT = None  # None: tự nhận diện; hoặc True/False
CANDIDATE_COUNT = 20
ANCHOR_COUNT = 4
NEIGHBORS = 1  # 0 để so sánh khi không mở rộng ngữ cảnh
MAX_CONTEXT_TOKENS = 6000
MAX_IMAGES = 3
""",
        "run": """stage05_dir, _ = find_stage(5, STAGE_05_DIR, VIDEO_ID)
engine = load_retrieval_engine(stage05_dir, VIDEO_ID, device=QUERY_DEVICE)
""",
        "inspect": """# Đổi QUESTION rồi chạy lại ô này; không cần tải lại model ở ô trên.
retrieval = retrieve_question(
    engine, QUESTION, current_time=CURRENT_TIME_SEC,
    prefer_current=PREFER_CURRENT_SEGMENT,
    candidate_count=CANDIDATE_COUNT, anchor_count=ANCHOR_COUNT,
    neighbors=NEIGHBORS, maximum_tokens=MAX_CONTEXT_TOKENS,
)
output_dir, manifest, retrieval = export_retrieval(
    engine, retrieval, OUTPUT_ROOT, maximum_images=MAX_IMAGES,
)
print('Số token:', retrieval['context_tokens'])
print('Anchor:', retrieval['anchor_chunk_ids'])
print(retrieval['context'][:7000])
from IPython.display import HTML, display
display(HTML((output_dir / 'preview.html').read_text(encoding='utf-8')))
""",
        "next": """Thử câu hỏi về lời giảng, slide, cả hai và câu hỏi ngoài phạm vi.
Ghi Recall@k, nguồn đúng, timestamp và độ trễ. So sánh NEIGHBORS=0 với 1.
Điểm cosine không phải xác suất đúng; nội dung gần nhất chưa chắc trả lời được.

Câu hỏi “đoạn này” cần CURRENT_TIME_SEC. Output cập nhật theo câu hỏi vừa chạy:
retrieval.json, context.txt, preview.html và ảnh nguồn. Có thể tải preview.html
về mở trực tiếp để kiểm tra bằng chứng.""",
    },
]


def cell(kind, source, hidden=False):
    source = textwrap.dedent(source).strip() + "\n"
    value = {
        "cell_type": kind, "id": hashlib.sha256((kind + source).encode()).hexdigest()[:12],
        "metadata": {"jupyter": {"source_hidden": True}} if hidden else {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build(spec):
    cells = [
        cell("markdown", "# " + spec["title"] + "\n\n" + spec["intro"]),
        cell("markdown", "## 1. Môi trường\n\nBật **Internet** để cài thư viện/tải model. "
             "Chạy mỗi notebook trong phiên Kaggle mới, với Input và GPU như ghi ở trên."),
    ]
    if spec["deps"]:
        cells.append(cell("code", "%pip install -q " + spec["deps"]))
    else:
        cells.append(cell("markdown", "Chỉ dùng thư viện chuẩn Python."))
    cells += [
        cell("markdown", "## 2. Hàm xử lý\n\nCác hàm nhúng trực tiếp; không cần tải "
             "file Python đi kèm. Mở ô thu gọn để xem hoặc chỉnh thuật toán."),
        cell("code", (CODE / "common.py").read_text(encoding="utf-8"), hidden=True),
        *[cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in spec.get("extra_modules", [])],
        cell("code", (CODE / spec["module"]).read_text(encoding="utf-8"), hidden=True),
        cell("markdown", "## 3. Cấu hình\n\nĐường dẫn None sẽ được tìm trong Input. "
             "Nếu có nhiều phiên bản, điền đường dẫn rõ ràng để tránh chọn nhầm."),
        cell("code", "from pathlib import Path\n\nVIDEO_ID = 'L25_V001'\n"
             "OUTPUT_ROOT = Path('/kaggle/working/lecture_rag') / VIDEO_ID\n\n"
             + spec["config"]),
        cell("markdown", "## 4. Thực thi"),
        cell("code", spec["run"]),
        cell("markdown", "## 5. Kiểm tra / sử dụng kết quả"),
        cell("code", spec["inspect"]),
        cell("markdown", "## Tiếp theo\n\n" + spec["next"]),
    ]
    notebook = {
        "cells": cells, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
    }
    target = ROOT / "notebooks" / (spec["name"] + ".ipynb")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    print(target.name)


if __name__ == "__main__":
    for specification in SPECS:
        build(specification)
