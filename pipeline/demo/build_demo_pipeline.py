"""Build the student demo from production modules with isolated notebook QA."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'builders'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops'))
import build_fast_pipeline as fast
from build_notebooks import cell

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT.parent / 'backend/app'
TARGET = ROOT / 'notebooks/demo_pipeline.ipynb'


def selected_source(path, names):
    """Embed original definitions, including decorators, without executing the module."""
    source = path.read_text(encoding='utf-8')
    lines = source.splitlines(keepends=True)
    selected = []
    for node in ast.parse(source).body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], 'id', None)
        if name in names:
            start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
            selected.append(''.join(lines[start - 1:node.end_lineno]))
    return '\n\n'.join(selected)


def production_config():
    # AST removes legacy comments; all production values remain shared with the fast builder.
    excluded = {'VIDEO_IDS', 'OUTPUT_ROOT', 'VIDEO_LANGUAGES', 'RESUME_FROM', 'CANARY'}
    return '\n'.join(ast.unparse(n) for n in ast.parse(fast.CONFIG).body
                     if isinstance(n, ast.Assign) and n.targets[0].id not in excluded)


def backend_sources():
    retrieval = (BACKEND / 'features/search/retrieval.py').read_text(encoding='utf-8')
    # Reuse the ranker, with local storage and a broader demo evidence window.
    retrieval = retrieval.replace('from app.shared import storage', 'storage = DEMO_STORAGE')
    # The notebook embeds search as a standalone module, so include its local-only dependency.
    rerank = (BACKEND / 'features/search/rerank.py').read_text(encoding='utf-8')
    retrieval = retrieval.replace('from .rerank import rerank',
                                  rerank.replace('from __future__ import annotations', ''))
    # Expose the actual RRF pool, without changing its scores or ordering.
    retrieval = retrieval.replace('return {"slides": slides, "temporal": temporal,', '''return {"candidate_ranks": [
                {"chunk_id": cid, "segment_id": index.chunks[by_id[cid]]["segment_id"],
                 "source_type": kinds[by_id[cid]], "rrf_rank": rank,
                 "semantic_rank": semantic.index(cid) + 1 if cid in semantic else None,
                 "lexical_rank": lexical.index(cid) + 1 if cid in lexical[:CANDIDATES] else None,
                 "rrf_score": scores[cid], "semantic_score": float(similarities[by_id[cid]])}
                for rank, cid in enumerate(ranked, 1)],
            "query_vector": vector.tolist(), "encoder_model": index.model,
            "slides": slides, "temporal": temporal,''')
    # Copy diagnostics from the already-computed scores, without another encoder call.
    retrieval = retrieval.replace(
        'matched, evidence = context_candidates(index, sid, ranked, similarities, options, sources, question)',
        '''matched, evidence = context_candidates(index, sid, ranked, similarities, options, sources, question)
        for chunk in evidence:
            cid = chunk['chunk_id']
            chunk['rrf_score'] = scores.get(cid)
            chunk['semantic_score'] = float(similarities[by_id[cid]]) if cid in by_id else None
            parent = index.by_segment.get(chunk['segment_id'], {})
            frame = chunk.get('image_path') or parent.get('visual', {}).get('image_path')
            chunk['thumbnail_path'] = str(index.folder / 'lecture' / frame) if frame and index.folder else None''')
    attribution = fast.module_path('demo_attribution.py').read_text(encoding='utf-8')
    retrieval += '\n\n' + attribution + '\n\n' + fast.module_path('demo_evidence.py').read_text(encoding='utf-8')
    answer = ('import http.client, json, os, re, time, urllib.error, urllib.request\n\n' + selected_source(
        BACKEND / 'features/ask/llm.py', {'DEFAULT_OLLAMA_URL', 'DEFAULT_OLLAMA_MODEL', 'LLMUnavailable', 'REFUSAL',
                            'TEXT_ONLY_NOTE', 'IMAGE_NOTE', 'CITE_RULE', 'OLLAMA_TIMING_FIELDS', '_finish_timings',
                            '_ollama', 'REWRITE_INSTRUCTION', '_clip', 'rewrite_question'}))
    legacy = selected_source(BACKEND / 'features/ask/llm.py', {'EVIDENCE_INSTRUCTION', '_history_block', '_evidence_prompt'})
    legacy = legacy.replace('EVIDENCE_INSTRUCTION', 'LEGACY_EVIDENCE_INSTRUCTION').replace(
        'def _evidence_prompt(', 'def legacy_evidence_prompt(')
    answer += '\n\n' + legacy + '\n\n' + attribution + '\n\n' + fast.module_path('demo_answer.py').read_text(encoding='utf-8')
    return retrieval, answer


INTRO = '''# Trình diễn pipeline Lecture RAG hiện tại

Chạy **Run All** trong một phiên Kaggle mới, bật **Internet** và **GPU T4 x2**.
Gắn ba dataset của lần chạy xác nhận: `l25a-lecture-videos-media`, `vn-lecture-test12`,
`slidespeech-train100` của `thnhtrungnguynmif`. Không cần kernel/model input khác.

**Video mới:** Upload video → Create Dataset → Add Input → copy path → paste vào VIDEO_SOURCES → Run All.
Dán đường dẫn tệp như `/kaggle/input/datasets/<owner>/<slug>/my lecture.mp4`, hoặc đường dẫn
dataset/thư mục để chạy mọi video bên trong (.mp4, .mkv, .mov, .webm; tìm đệ quy).
`VIDEO_SOURCES` nhận cả video id cũ; để `[]` thì dùng `VIDEO_IDS`. GPU tốn khoảng **0.4 T4-hour
cho mỗi 15 phút video**. Bảng video_id → đường dẫn, thời lượng, ngôn ngữ hiện trước bước 01.

Sửa danh sách video ở ô đầu tiên. Mặc định chạy một video L25. Các bước dùng cùng mã và
cấu hình production: medium ASR, Qwen3-VL với crop, kiểm tra EOS/vòng lặp và một lần thử lại,
mốc xuất hiện trạng thái, gộp cùng nội dung, lọc lời quảng cáo, E5 + BM25.
Mỗi ô hiển thị kết quả; bảng dài được giới hạn, OCR từng nhóm nằm trong phần có thể mở rộng.

Phần hỏi đáp dùng **Qwen2.5-7B qua Ollama**, cùng model và xếp hạng E5 + BM25 RRF
của web app. Mặc định demo dùng tối đa 8 slide, ngữ cảnh 16.000 ký tự và prompt riêng
đọc cả OCR lẫn lời giảng, trả lời ngắn và trích dẫn đoạn chứa câu trả lời. Không cần khóa API.
Đặt `DEMO_ENHANCED_CONTEXT = False` để so sánh với prompt gốc và ngữ cảnh 3 slide.
Sau câu hỏi mẫu, mở ô **Trò chuyện với bài giảng** để hỏi liên tiếp trong cùng phiên.
'''

CONFIG = '''from pathlib import Path
import time
import torch

# Edit these values, then run all cells from the top.
VIDEO_IDS = ['L25_V037']
VIDEO_SOURCES = []  # When nonempty, replaces VIDEO_IDS: full file/folder paths or bare ids.
# VIDEO_SOURCES = ['/kaggle/input/datasets/<owner>/<slug>/my lecture.mp4', 'L25_V008']
VIDEO_LANGUAGES = {}  # Default Vietnamese; use {'video_id': 'en'} for English.
# Map language by resolved video_id (shown in the table), original stem or full file path.
OUTPUT_ROOT = Path('/kaggle/working/demo_pipeline')
RESUME_STAGES = {'01': True, '02': True, '03': True, '04': True, '05': True}
DEMO_ENHANCED_CONTEXT = True  # Verified 8-slide/16,000-character QA; False compares legacy QA.
MAX_TABLE_ROWS = 40
MAX_TEXT_CHARS = 800
TRANSCRIPT_LINES = 12
EXAMPLE_GROUPS = 5
CHAT_MODE = 'auto'  # 'interactive' for widgets; 'scripted' for a bounded 3-turn conversation.
SESSION_STARTED = time.monotonic()
print('DEMO 00 | videos:', VIDEO_IDS, '| output:', OUTPUT_ROOT)
print('CUDA available:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())
for device in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(device)
    print(device, props.name, f'{props.total_memory / 2**30:.1f} GiB')
'''

SECTIONS = [
    ('01', '1 · Phát hiện slide', 'Số trạng thái, nhóm slide, thời lượng và contact sheet.', 'demo_detection()'),
    ('gate', '2 · Bộ gác nội dung', 'Nhóm giữ/bỏ, điểm toàn ảnh và crop, lý do quyết định.', 'demo_gate()'),
    ('02', '3 · OCR', 'Mốc thời gian, ảnh và chữ từng nhóm; số lỗi và lần thử lại.', 'demo_ocr()'),
    ('03', '4 · Nhận dạng lời giảng', 'Thời lượng, số từ và các dòng đầu có timestamp.', 'demo_asr()'),
    ('04', '5 · Đồng bộ và đóng gói', 'Chữ trên slide bên cạnh lời giảng; thống kê lọc.', 'demo_alignment()'),
    ('05', '6 · Lập chỉ mục và kiểm tra', 'Chunk theo nguồn, ví dụ, file chỉ mục và PASS/FAIL.', 'demo_index()'),
    ('query_setup', '7 · Nạp model trả lời', 'Giải phóng model pipeline, nạp Qwen2.5-7B và tìm kiếm E5 + BM25 RRF.',
     'demo_query_setup()'),
    ('query', '7 · Hỏi đáp', 'Ba câu hỏi mẫu: chữ slide, lời giảng, ngoài phạm vi. Sau mỗi câu trả lời, **Căn cứ trả lời** hiện nhãn ASR/OCR, trích đoạn, ảnh OCR và bảng toàn bộ bằng chứng. Xem lại bằng `result = ask(...)` rồi `result["used_evidence"]`. Sửa ô này và chạy lại.', '''# Edit these questions or call ask('your question', video_id='L25_V037').
QUESTIONS = [
    'Trang tiêu đề cho biết chuyên đề cách mạng bao trùm giai đoạn nào?',
    'Cuối bài, thầy nhắc cao trào kháng Nhật cứu nước diễn ra tháng năm nào?',
    'Bài giảng hướng dẫn cài đặt PyTorch trên Windows như thế nào?',
]
for question in QUESTIONS:
    ask(question)
print('DEMO query | completed', len(QUESTIONS), 'example questions')'''),
    ('chat', '8 · Trò chuyện với bài giảng', 'Nhập câu hỏi rồi bấm **Gửi** hoặc Enter. Câu nối tiếp dùng ngữ cảnh phiên; truy vấn đã viết lại và **Căn cứ trả lời** hiện trong từng lượt. **Xoá ngữ cảnh** lưu phiên cũ và mở phiên mới; đổi video cũng mở phiên mới. **OK – kết thúc** khoá nhập, hiện số lượt/thời gian và lưu `chat_session_<timestamp>.json` dưới `/kaggle/working/demo_pipeline/`. Chạy lại ô để mở phiên khác. API: `start_chat(video_id=None)` mở phiên mới, rồi `chat("câu hỏi")` và `end_chat()`. Save Version/batch hoặc thiếu ipywidgets tự chạy 3 lượt rồi kết thúc; không chờ nhập. Nếu Kaggle không nhận diện phiên tương tác, đặt `CHAT_MODE = "interactive"` rồi chạy lại ô này.', 'demo_chat_start()'),
    ('timing', '9 · Tổng kết thời gian', 'Thời gian từng bước, tổng phiên và vị trí lưu kết quả.', 'demo_timing()'),
]


def notebook():
    backend_retrieval, backend_answer = backend_sources()
    queue = selected_source(fast.module_path('production_queue.py'),
                            {'speech_embed_model', 'detect', 'ocr', 'asr'})
    sources = fast.module_path('demo_sources.py').read_text(encoding='utf-8')
    preflight = selected_source(fast.module_path('production_runtime.py'), {'startup_preflight'})
    preflight = preflight.replace('def startup_preflight(', 'def demo_source_preflight(').replace(
        "sorted(p for p in Path(input_root).rglob(video_id + '.mp4') if p.is_file())", '[find_video_file(video_id)]')
    queue = sources + '\n\n' + preflight + '\n\n' + queue
    runtime = fast.module_path('demo_runtime.py').read_text(encoding='utf-8')
    runtime += '\n\n' + fast.module_path('demo_chat.py').read_text(encoding='utf-8')
    recipe = hashlib.sha256((''.join(fast.module_path(n).read_text(encoding='utf-8') for n in fast.MODULES)
                             + queue + production_config() + runtime + backend_retrieval + backend_answer).encode()).hexdigest()
    capture = fast.module_path('demo_capture.py').read_text(encoding='utf-8')
    headings = {code: (tag, f'## {title}\n\n{description}') for tag, title, description, code in SECTIONS
                if tag != 'query'}
    headings['for question in QUESTIONS:'] = ('query', '## Hỏi đáp mẫu')
    headings['VIDEO_IDS ='] = ('00', INTRO)
    capture_config = CONFIG.replace("SESSION_STARTED = time.monotonic()\n", "SESSION_STARTED = time.monotonic()\n"
        + f'exec({capture!r})\n'
        + f"_demo_capture = DemoNotebookCapture(_capture_ipython(), '/kaggle/working/demo_pipeline.executed.ipynb', {headings!r})\n")
    cells = [cell('markdown', INTRO), cell('code', capture_config),
             cell('markdown', '## Môi trường và mã dùng chung\n\nCác ô thu gọn được tạo từ mã Python trong repo.'),
             cell('code', '%pip install -q ' + fast.DEPS + ' "onnxruntime>=1.20,<2" "ipywidgets>=8,<9"')]
    cells[1]['metadata']['demo_stage'] = '00'
    cells += [cell('code', fast.module_path(name).read_text(encoding='utf-8'), hidden=True)
              for name in fast.MODULES]
    cells.append(cell('code', production_config() + '\n\n' + queue, hidden=True))
    cells.append(cell('code', runtime, hidden=True))
    # Isolated modules preserve the app's names, caches and dataclasses.
    cells.append(cell('code', 'app_search = embedded_module("demo_app_search", ' + repr(backend_retrieval)
                      + ', {"DEMO_STORAGE": DemoStorage()})\n'
                      + 'app_answer = embedded_module("demo_app_answer", ' + repr(backend_answer) + ')\n'
                      + f'DEMO_RECIPE = {recipe!r}\n' + 'demo_startup()', hidden=True))
    for tag, title, description, code in SECTIONS:
        cells.append(cell('markdown', f'## {title}\n\n{description}'))
        c = cell('code', code)
        c['metadata']['demo_stage'] = tag
        cells.append(c)
    return {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}


def build(target=TARGET):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(target.name)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=TARGET)
    build(parser.parse_args().output)
