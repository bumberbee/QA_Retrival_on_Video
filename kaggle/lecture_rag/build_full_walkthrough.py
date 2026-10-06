"""Writes notebooks/full_pipeline_walkthrough.ipynb: the whole current pipeline (01 with the
ad filter, OCR gate, 02 hybrid OCR, 03 ASR, 04 sync, 05 index, retrieval) on ONE video,
printing the output of every stage (tables, images, plots) so it can be inspected and tuned.
Everything shown is also saved to /kaggle/working/walkthrough_report.html.

Kaggle: GPU T4 x2, Internet on, inputs = dataset l25a-lecture-videos-media and the output of
kernel l25-ad-filter-e1-e4-20261002 (ad_student.npz).

    python kaggle/lecture_rag/build_full_walkthrough.py
"""

import json
from pathlib import Path

from build_fast_pipeline import DEPS
from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
MODULES = ["common.py", "detection.py", "adfilter.py", "visual.py", "transcription.py",
           "textclean.py", "alignment.py", "indexing.py", "retrieval.py"]

INTRO = """# Toàn bộ pipeline trên một video · in kết quả từng giai đoạn

Video **L25_V008 · Toán · Xác suất** (20 phút, có quảng cáo chèn giữa). Chạy đúng các hàm của
pipeline hiện tại (`notebook_code/`), theo thứ tự, và in ra sau mỗi giai đoạn:

| Giai đoạn | Làm gì | Xem gì để tinh chỉnh |
|---|---|---|
| 0 | Thông tin video | độ phân giải, fps, âm thanh |
| 1 | **01 · Tách slide** | số trạng thái → số slide, bảng slide kèm ảnh, slide hiện dần đã gộp |
| 2 | **Lọc quảng cáo** | biểu đồ điểm quảng cáo theo giây, các đoạn bị cắt |
| 3 | **03 · Giọng nói** | cửa sổ ASR, từ đầu tiên kèm thời gian |
| 4 | **Bộ gác OCR + 02 · OCR kết hợp** | điểm bộ gác, chữ và vùng hình đọc được từng slide, đầu ra thô của model |
| 5 | **04 · Đồng bộ** | lời giảng gán vào từng slide |
| 6 | **05 · Chỉ mục** | các đoạn bằng chứng được lập chỉ mục |
| 7 | **Hỏi thử** | kết quả tìm kiếm cho vài câu hỏi toán |
| 8 | Tổng kết | thời gian từng bước, bộ nhớ GPU |

Trong pipeline nhanh, 03 chạy song song với 01 và 02; ở đây chạy lần lượt để dễ xem.
Mọi thứ hiện ra cũng được lưu vào `walkthrough_report.html` (tải ở tab Output)."""

CONFIG = """from pathlib import Path
VIDEO_ID = 'L25_V008'
OUTPUT_ROOT = Path('/kaggle/working/walkthrough')

# 01 · tách slide (giống pipeline nhanh)
SAMPLE_HZ, FINE_HZ, MERGE_REVEALS = 1.0, 8.0, True
# lọc quảng cáo + bộ gác OCR (dùng chung SigLIP base)
AD_FILTER, AD_DEVICE = 'student', 'cuda:0'
AD_MODE = 'flag'   # 'flag': chỉ đánh dấu, không cắt; 'cut': bản cũ
OCR_GATE_THRESHOLD, OCR_GATE_VIEWS = 0.15, 'frame+crops'   # Phase 2: gate_bench
# 02 · OCR kết hợp
OCR_ENGINE, HYBRID_DEVICES = 'hybrid', ('cuda:0', 'cuda:1')
OCR_LANGUAGES, OCR_SCALE, OCR_BATCH_SIZE = ('vi', 'en'), 1.5, 32
# 03 · giọng nói
ASR_MODEL = 'Systran/faster-whisper-medium'
ASR_REVISION = '08e178d48790749d25932bbc082711ddcfdfbc4f'
ASR_DEVICE_INDEX, ASR_MODE = 1, 'windowed'
# 05 · chỉ mục
EMBEDDING_MODEL = 'intfloat/multilingual-e5-small'
EMBEDDING_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
# 7 · câu hỏi thử (None = hỏi cả bài; số = đang xem ở giây đó)
QUESTIONS = [
    ('Xác suất của một biến cố được tính như thế nào?', None),
    ('Công thức cộng xác suất là gì?', None),
    ('Hai biến cố độc lập khi nào?', None),
    ('Giải thích slide này.', 'giữa slide thứ 5'),
    ('Cách cài đặt PyTorch?', None),
]
"""

HELPERS = """import base64, html, io, json, shutil, sys, time
from IPython.display import HTML, display
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

REPORT, TIMES = [], {}

class Raw(str):
    '''Ô bảng là HTML (ảnh, chi tiết), không escape.'''

def show(markup):
    REPORT.append(markup)
    display(HTML(markup))

def heading(text, level=2):
    show(f'<h{level} style="margin:18px 0 6px">{html.escape(text)}</h{level}>')

def note(text):
    show(f'<p style="max-width:900px;color:#555">{text}</p>')

def clock(seconds):
    seconds = float(seconds)
    return f'{int(seconds) // 60}:{seconds % 60:04.1f}'

def img(source, width=220):
    image = source if isinstance(source, Image.Image) else Image.open(source)
    image = image.convert('RGB'); image.thumbnail((width * 2, width * 2))
    buffer = io.BytesIO(); image.save(buffer, 'JPEG', quality=80)
    data = base64.b64encode(buffer.getvalue()).decode()
    return Raw(f'<img src="data:image/jpeg;base64,{data}" style="width:{width}px;border:1px solid #ddd">')

def short(text, limit=220):
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit] + '…'

def table(rows, columns=None, limit=300):
    if not rows:
        show('<p><i>(trống)</i></p>'); return
    columns = columns or list(rows[0])
    def cell(value):
        return value if isinstance(value, Raw) else html.escape(str(value))
    head = ''.join(f'<th style="text-align:left;padding:4px 8px;background:#f3ece2">{html.escape(str(c))}</th>' for c in columns)
    body = ''.join('<tr>' + ''.join(f'<td style="vertical-align:top;padding:4px 8px;border-top:1px solid #eee">{cell(r.get(c, ""))}</td>'
                                    for c in columns) + '</tr>' for r in rows[:limit])
    more = f'<p><i>… và {len(rows) - limit} dòng nữa</i></p>' if len(rows) > limit else ''
    show(f'<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:12px">'
         f'<tr>{head}</tr>{body}</table></div>{more}')

def params(mapping, keys=None):
    table([{'tham số': k, 'giá trị': json.dumps(v, ensure_ascii=False)[:160]}
           for k, v in mapping.items() if keys is None or k in keys])

def figure(fig):
    buffer = io.BytesIO(); fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight'); plt.close(fig)
    show(f'<img src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}" style="max-width:100%">')

def pre(text, limit=4000):
    text = str(text)
    text = text if len(text) <= limit else text[:limit] + '\\n…'
    return Raw(f'<pre style="white-space:pre-wrap;max-width:620px;font-size:11px;margin:0">{html.escape(text)}</pre>')

def timed(name, fn, *args, **kwargs):
    started = time.monotonic()
    result = fn(*args, **kwargs)
    TIMES[name] = round(time.monotonic() - started, 1)
    note(f'⏱ {name}: <b>{TIMES[name]} s</b>')
    return result
"""

VIDEO = """heading('0 · Thông tin video')
VIDEO = next(iter(sorted(Path('/kaggle/input').rglob(VIDEO_ID + '.mp4'))))
SOURCE = probe_video(VIDEO)
params(SOURCE, ['duration_sec', 'width', 'height', 'has_audio', 'timeline_origin_sec', 'audio_start_sec', 'sha256'])
frames = [frame_at_time(VIDEO, SOURCE, SOURCE['duration_sec'] * k / 6) for k in range(6)]
table([{f'{clock(t)}': img(image, 180) for image, t in frames}])
"""

MODELS = """heading('Nạp model dùng chung (SigLIP base cho lọc quảng cáo và bộ gác OCR)')
student = next(iter(sorted(Path('/kaggle/input').rglob('ad_student.npz'))))
ad_scorer = timed('nạp SigLIP + model trò', load_ad_scorer, AD_FILTER, AD_DEVICE, student)
ocr_gate = load_ocr_gate(AD_DEVICE, threshold=OCR_GATE_THRESHOLD, views=OCR_GATE_VIEWS)
params({**ad_scorer[1], 'ocr_gate': ocr_gate[1]})
"""

STAGE01 = """heading('1 · Bước 01 · Tách slide')
note('Quét video một lượt (mẫu 1 Hz + khung 8 Hz), tìm ranh giới ổn định, bỏ vùng chuyển động, tách chuyển cảnh ẩn, '
     'chọn ảnh nét nhất, gộp slide hiện dần. Lọc quảng cáo chạy ở cuối bước này.')
stage01, manifest01, intervals = timed('01 tách slide + lọc quảng cáo', run_slide_detection,
    VIDEO, OUTPUT_ROOT, VIDEO_ID, sample_hz=SAMPLE_HZ, fine_hz=FINE_HZ,
    merge_reveals=MERGE_REVEALS, scan='single_pass', ad_scorer=ad_scorer,
    ad_options={'mode': AD_MODE})
table([{'chỉ số': k, 'giá trị': manifest01.get(k)} for k in
       ('sample_count', 'visual_state_count', 'merged_reveal_count', 'hidden_split_count',
        'masked_split_count', 'motion_mask_fraction', 'interval_count', 'ad_segment_count', 'ad_flagged_count')])
heading('Tham số bước 01', 3)
params(manifest01['parameters'], ['change_threshold', 'stability_threshold', 'stable_samples', 'min_interval_sec',
                                  'sample_hz', 'fine_hz', 'merge_reveals', 'split_hidden_changes', 'mask_motion'])
heading('Danh sách đoạn (slide và quảng cáo)', 3)
table([{'#': i['sequence'], 'loại': i['content_kind'], 'bắt đầu': clock(i['start_sec']), 'kết thúc': clock(i['end_sec']),
        'dài (s)': round(i['end_sec'] - i['start_sec'], 1), 'số trạng thái': len(i['reveal_states']),
        'cách tách': i['boundary_method'].replace('single_pass_stable_visual_change', 'ổn định'),
        'ảnh': img(stage01 / i['image_path'], 200)} for i in intervals])
heading('Slide hiện dần đã gộp (mỗi hàng: các trạng thái → giữ ảnh cuối)', 3)
merged = [i for i in intervals if len(i['reveal_states']) > 1]
table([{'#': i['sequence'], 'trạng thái': Raw(' '.join(img(stage01 / s['image_path'], 120) for s in i['reveal_states'][:6])),
        'quyết định': ', '.join(s.get('decision', '') for s in i['reveal_states'])} for i in merged[:12]])
"""

ADS = """heading('2 · Lọc quảng cáo')
report = json.loads((stage01 / 'ad_filter.json').read_text(encoding='utf-8'))
note(f'Ngưỡng {report["threshold"]}, làm mượt {report["smooth"]} mẫu, đoạn ≥ {report["min_ad_sec"]} mẫu, '
     f'lấp lỗ ≤ {report["max_gap_sec"]}. Tổng quảng cáo: <b>{report["ad_seconds"]} s</b> trong {len(report["ad_intervals"])} đoạn.')
guard = report.get('guard')
if guard:
    note(f'Chế độ <b>{report["mode"]}</b>: đoạn nghi quảng cáo vẫn được OCR và giữ lời, chỉ không lập chỉ mục. '
         + (f'<b>Bộ an toàn đã bỏ mọi dấu</b> ({html.escape(guard["reason"])}): coi cả video là bài giảng.'
            if guard['triggered'] else f'Bộ an toàn không kích hoạt (≤ {guard["max_runs"]} đoạn, ≤ {guard["max_share"]:.0%} video).'))
fig, ax = plt.subplots(figsize=(13, 2.6))
ax.plot(report['sample_times'], report['sample_scores'], lw=0.8, color='#5b7fa6', label='điểm quảng cáo (mẫu 1 Hz)')
ax.axhline(report['threshold'], color='#c85c4a', ls='--', lw=1, label='ngưỡng')
for a, b in report['ad_intervals']:
    ax.axvspan(a, b, color='#e8734a', alpha=0.25)
ax.set_xlabel('giây'); ax.set_ylim(-0.02, 1.02); ax.legend(loc='upper right', fontsize=8); ax.set_title('Vùng cam = đoạn bị cắt là quảng cáo')
figure(fig)
rows = []
for a, b in report['ad_intervals']:
    shots = [frame_at_time(VIDEO, SOURCE, t)[0] for t in (a + 1, (a + b) / 2, max(a, b - 1.5))]
    before = frame_at_time(VIDEO, SOURCE, max(0, a - 2))[0]
    after = frame_at_time(VIDEO, SOURCE, min(SOURCE['duration_sec'] - 0.1, b + 1.5))[0]
    rows.append({'từ': clock(a), 'đến': clock(b), 'dài (s)': round(b - a, 1), '2 s trước': img(before, 150),
                 'đầu': img(shots[0], 150), 'giữa': img(shots[1], 150), 'cuối': img(shots[2], 150), '1,5 s sau': img(after, 150)})
table(rows)
"""

STAGE03 = """heading('3 · Bước 03 · Nhận dạng giọng nói')
stage03, manifest03, transcript = timed('03 giọng nói', run_transcription, None, VIDEO, OUTPUT_ROOT, VIDEO_ID,
    model_id=ASR_MODEL, revision=ASR_REVISION, language='vi', transcription_mode=ASR_MODE,
    device_index=ASR_DEVICE_INDEX)
table([{'chỉ số': k, 'giá trị': v} for k, v in {
    'trạng thái': transcript['status'], 'số từ': len(transcript['words']), 'số cửa sổ': len(transcript['windows']),
    'đoạn không có giờ từng từ': len(transcript['fallback_spans'])}.items()])
heading('Cửa sổ ASR (cắt tại khoảng lặng)', 3)
table([{'cửa sổ': w['window_id'], 'từ': clock(w['core_start_sec']), 'đến': clock(w['core_end_sec']),
        'dài (s)': round(w['core_end_sec'] - w['core_start_sec'], 1), 'có lời (s)': w.get('speech_duration_sec', '')}
       for w in transcript['windows']], limit=40)
heading('60 từ đầu tiên (thời gian từng từ)', 3)
table([{'từ': w['text'], 'bắt đầu': round(w['start_sec'], 2), 'kết thúc': round(w['end_sec'], 2),
        'xác suất': round(w.get('probability') or 0, 2)} for w in transcript['words'][:60]])
"""

STAGE02 = """heading('4 · Bộ gác OCR + Bước 02 · OCR kết hợp')
note('Bộ gác (SigLIP zero-shot) chấm từng keyframe: điểm < ngưỡng thì không OCR (giảng viên nói, logo). '
     'Unlimited-OCR đọc cả slide; vùng hình/biểu đồ được Qwen3-VL đọc riêng. Đoạn quảng cáo bỏ qua.')
stage02, manifest02, records = timed('02 bộ gác + OCR', run_visual_extraction, stage01, OUTPUT_ROOT, VIDEO_ID,
    languages=OCR_LANGUAGES, ocr_device='cuda', ocr_scale=OCR_SCALE, ocr_batch_size=OCR_BATCH_SIZE,
    ocr_engine=OCR_ENGINE, hybrid_devices=HYBRID_DEVICES, ocr_gate=ocr_gate)
table([{'chỉ số': k, 'giá trị': manifest02.get(k)} for k in ('record_count', 'ocr_read_count', 'ocr_skipped_no_content')])
by_segment = {i['segment_id']: i for i in intervals}
table([{'#': by_segment[r['segment_id']]['sequence'], 'loại': r['content_kind'], 'OCR': r['ocr_status'],
        'điểm bộ gác': r.get('ocr_gate_score'), 'số ký tự': len(r['ocr_text']), 'số vùng': len(r['regions']),
        'đọc lại khối hình lớn': len(json.loads(r['raw_ocr_response']).get('second_pass', [])) if r.get('raw_ocr_response') else '',
        'tiêu đề': short(r['title'], 60), 'ảnh': img(stage01 / by_segment[r['segment_id']]['image_path'], 170)}
       for r in records])
heading('Chi tiết từng slide đã OCR', 3)
for r in records:
    if r['ocr_status'] != 'read':
        continue
    item = by_segment[r['segment_id']]
    regions = ''.join(f'<div style="margin:6px 0"><b>{html.escape(g["kind"])}</b><br>{img(stage02 / g["image_path"], 220)}'
                      f'{pre(g.get("text") or g.get("description") or "", 1500)}</div>' for g in r['regions'])
    raw = json.loads(r['raw_ocr_response']).get('unlimited', '') if r.get('raw_ocr_response') else ''
    show(f'<details style="margin:6px 0;border:1px solid #eee;padding:6px"><summary><b>#{item["sequence"]}</b> '
         f'{clock(item["start_sec"])}–{clock(item["end_sec"])} · {len(r["ocr_text"])} ký tự · {len(r["regions"])} vùng</summary>'
         f'<table><tr><td style="vertical-align:top">{img(stage01 / item["image_path"], 420)}</td>'
         f'<td style="vertical-align:top"><b>Chữ OCR</b>{pre(r["ocr_text"])}<b>Vùng hình (Qwen3-VL)</b>{regions or "<i>không có</i>"}'
         f'<details><summary>Đầu ra thô Unlimited-OCR</summary>{pre(raw, 3000)}</details></td></tr></table></details>')
"""

STAGE04 = """heading('5 · Bước 04 · Đồng bộ')
from sentence_transformers import SentenceTransformer
_e5 = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device='cuda:1')
speech_embed = lambda texts: _e5.encode(['query: ' + t for t in texts], normalize_embeddings=True, convert_to_numpy=True)
stage04, manifest04, segments = timed('04 đồng bộ', run_alignment, stage01, stage02, stage03, OUTPUT_ROOT, VIDEO_ID,
                                      speech_embed=speech_embed)
table([{'chỉ số': k, 'giá trị': manifest04.get(k)} for k in ('segment_count', 'word_count', 'ad_segment_count', 'ad_flagged_count', 'ad_word_count')])
table([{'#': s['sequence'], 'loại': s['content_kind'] + (' · nghi QC (không lập chỉ mục)' if s.get('ad_flagged') else ''),
        'thời gian': f"{clock(s['start_sec'])}–{clock(s['end_sec'])}",
        'số từ': len(s['speech']['word_ids']), 'lời giảng': short(s['speech']['text'], 260),
        'chữ OCR (sạch)': short(s['visual'].get('ocr_text_clean', s['visual'].get('ocr_text', '')), 160)} for s in segments])
heading('Lời giảng có thuộc chủ đề bài không (E5, chỉ ghi lại)', 3)
note('z = độ lệch so với chủ đề chung của bài, chuẩn hóa trong video; z < -2,5 là lạc chủ đề. Chưa dùng để quyết định.')
windows = json.loads((stage04 / 'speech_relevance.json').read_text(encoding='utf-8')) if (stage04 / 'speech_relevance.json').exists() else []
table([{'từ': clock(w['start_sec']), 'z': w['topic_z'], 'từ quảng cáo': w['ad_terms'], 'lời': short(w['text'], 160)}
       for w in sorted(windows, key=lambda w: w['topic_z'])[:12]])
heading('Làm sạch OCR và sửa thuật ngữ ASR (textclean.py)', 3)
table([{'chỉ số': 'dòng lặp bị bỏ (logo, tên GV…)', 'giá trị': ', '.join(manifest04['parameters']['ocr_cleaning']['repeated_lines']) or '—'},
       {'chỉ số': 'vùng hình nhiễu bị bỏ', 'giá trị': manifest04.get('noise_region_count')},
       {'chỉ số': 'từ ASR được sửa theo chữ trên slide', 'giá trị': manifest04.get('asr_correction_count')}])
corrections = json.loads((stage04 / 'asr_corrections.json').read_text(encoding='utf-8'))
table([{'lúc': clock(c['at_sec']), 'ASR nghe': c['heard'], 'sửa thành': c['fixed']} for c in corrections], limit=40)
table([{'#': s['sequence'], 'OCR gốc (ký tự)': len(s['visual'].get('ocr_text', '')), 'sau làm sạch': len(s['visual'].get('ocr_text_clean', '')),
        'tiêu đề sạch': short(s['visual'].get('title_clean', ''), 70)} for s in segments if s['content_kind'] != 'ad'])
example = max((s for s in segments if s['content_kind'] != 'ad'), key=lambda s: len(s['speech']['word_ids']))
heading(f'Ví dụ một slide đầy đủ: #{example["sequence"]}', 3)
table([{'ảnh': img(stage04 / example['visual']['image_path'], 420), 'chữ OCR (sạch)': pre(example['visual'].get('ocr_text_clean', ''), 2500),
        'lời giảng': pre(example['speech']['text'], 2500)}])
"""

STAGE05 = """heading('6 · Bước 05 · Chỉ mục tìm kiếm')
stage05, manifest05, chunks = timed('05 chỉ mục', run_indexing, stage04, OUTPUT_ROOT, VIDEO_ID,
    embedding_model=EMBEDDING_MODEL, revision=EMBEDDING_REVISION)
from collections import Counter
table([{'nguồn': k, 'số đoạn': v} for k, v in Counter(c['source_type'] for c in chunks).items()])
params(manifest05['parameters'], ['embedding_model', 'chunk_tokens', 'overlap_tokens', 'speech_chunk_tokens',
                                  'speech_maximum_seconds', 'dimensions'])
table([{'chunk': c['chunk_id'].split(':', 1)[1], 'nguồn': c['source_type'], 'thời gian': f"{clock(c['start_sec'])}–{clock(c['end_sec'])}",
        'token': c['encoded_tokens'], 'nội dung': short(c['text'], 200)} for c in chunks[:30]])
"""

RETRIEVE = """heading('7 · Hỏi thử')
note('Đây là bước tìm bằng chứng (E5 + FTS5, gộp bằng RRF), chưa phải câu trả lời của Gemini.')
engine = load_retrieval_engine(stage05, VIDEO_ID)
lesson = [s for s in segments if s['content_kind'] != 'ad']
by_chunk = {c['chunk_id']: c for c in chunks}
for question, when in QUESTIONS:
    current = (lesson[min(4, len(lesson) - 1)]['start_sec'] + lesson[min(4, len(lesson) - 1)]['end_sec']) / 2 if when else None
    started = time.monotonic()
    try:
        result = retrieve_question(engine, question, current_time=current)
    except ValueError as error:
        heading(f'❓ {question}', 3); note(f'Lỗi: {html.escape(str(error))}'); continue
    heading(f'❓ {question}' + (f'  (đang xem {clock(current)})' if current else '') +
            f'  · {1000 * (time.monotonic() - started):.0f} ms', 3)
    heading('Xếp hạng chung (trước khi chọn neo)', 4)
    table([{'hạng': n + 1, 'nguồn': by_chunk[h['chunk_id']]['source_type'],
            'thời gian': clock(by_chunk[h['chunk_id']]['start_sec']), 'RRF': round(h['rrf_score'], 4),
            'cosine': round(h['semantic_score'], 3), 'từ khóa (hạng)': h['lexical_rank'],
            'nội dung': short(by_chunk[h['chunk_id']]['text'], 200)} for n, h in enumerate(result['hits'][:5])])
    heading('Ngữ cảnh gửi cho LLM' + (' (ưu tiên slide đang xem)' if result['parameters']['prefer_current'] else ''), 4)
    table([{'mã': b['evidence_id'], 'lý do': b['selection_reason'], 'nguồn': b['source_type'],
            'slide': b['segment_id'].split(':')[1], 'thời gian': f"{clock(b['start_sec'])}–{clock(b['end_sec'])}",
            'nội dung': short(b['text'], 200)} for b in result['evidence']])
    if result['anchor_chunk_ids']:
        anchor = by_chunk[result['anchor_chunk_ids'][0]]
        table([{'bằng chứng chính': img(stage05 / 'lecture' / anchor['image_path'], 320) if (stage05 / 'lecture' / anchor['image_path']).exists() else '',
                'slide': anchor['segment_id'].split(':')[1], 'ngữ cảnh gửi cho AI (token)': result['context_tokens']}])
"""

EVAL = """heading('7b · Chấm điểm tìm kiếm trên bộ câu hỏi có đáp án')
note('Mỗi câu có khoảng thời gian chứa đáp án (giây). Đúng ở hạng k nếu một trong k slide neo đầu nằm trong khoảng đó. '
     'Câu ngoài bài: không có đáp án, xem điểm cosine cao nhất để đặt ngưỡng "không có trong bài".')
GOLD_QA = [
    ('Xác suất của một biến cố được tính như thế nào?', (89, 131)),
    ('Không gian mẫu khi lấy 2 quả cầu từ hộp 15 quả có bao nhiêu phần tử?', (130, 320)),
    ('Tổng hai số là số chẵn thì có những trường hợp nào?', (150, 320)),
    ('Chọn 10 tấm thẻ trong 30 thẻ sao cho có đúng 1 thẻ chia hết cho 10', (319, 465)),
    ('Từ 1 đến 40 có bao nhiêu số chia hết cho 3?', (547, 814)),
    ('Tính xác suất để tổng 3 số trên 3 thẻ chia hết cho 3', (547, 814)),
    ('Công thức tính nhanh số các số chia hết cho 3 trong một đoạn', (547, 814)),
    ('Xếp 9 học sinh thành hàng ngang sao cho không có 2 em lớp 12 đứng cạnh nhau', (813, 931)),
    ('Chia 9 học sinh thành 3 tổ, mỗi tổ có đúng một học sinh nữ', (989, 1210)),
    ('Thầy giáo dạy chuyên đề này là ai?', (0, 30)),
]
OUT_OF_SCOPE = ['Hai biến cố độc lập khi nào?', 'Cách cài đặt PyTorch?', 'Định lý Pythagore phát biểu thế nào?',
                'Công thức cộng xác suất là gì?']
by_chunk = {c['chunk_id']: c for c in chunks}
reranker = None
try:
    from sentence_transformers import CrossEncoder
    _ce = timed('nạp reranker bge-reranker-v2-m3', CrossEncoder, 'BAAI/bge-reranker-v2-m3', max_length=512, device='cuda:0')
    reranker = lambda q, texts: _ce.predict([(q, t) for t in texts], batch_size=16).tolist()
except Exception as error:
    note(f'Không nạp được reranker: {html.escape(str(error)[:200])}')
VARIANTS = [('RRF bằng nhau (cũ)', dict(semantic_weight=1.0, lexical_weight=1.0)),
            ('RRF nghĩa 1,0 / từ khóa 0,5 (mới)', dict(semantic_weight=1.0, lexical_weight=0.5))]
if reranker:
    VARIANTS.append(('mới + reranker', dict(semantic_weight=1.0, lexical_weight=0.5, reranker=reranker)))
def anchor_slides(result):
    out = []
    for cid in result['anchor_chunk_ids']:
        sid = by_chunk[cid]['segment_id']
        if sid not in out:
            out.append(sid)
    return out
seg_by_id = {s['segment_id']: s for s in segments}
summary_rows, detail = [], []
for name, kwargs in VARIANTS:
    hit1 = hit3 = 0; started = time.monotonic()
    for question, (a, b) in GOLD_QA:
        result = retrieve_question(engine, question, **kwargs)
        slides = anchor_slides(result)
        inside = [seg_by_id[sid]['start_sec'] < b and seg_by_id[sid]['end_sec'] > a for sid in slides]
        hit1 += bool(inside[:1] and inside[0]); hit3 += any(inside[:3])
        detail.append({'cách': name, 'câu hỏi': short(question, 60), 'slide đầu': clock(seg_by_id[slides[0]]['start_sec']) if slides else '',
                       'đúng hạng 1': '✅' if inside[:1] and inside[0] else '❌', 'cosine cao nhất': round(result['best_semantic_score'], 3),
                       'reranker cao nhất': round(result['best_rerank_score'], 3) if result.get('best_rerank_score') is not None else ''})
    out = [retrieve_question(engine, q, **kwargs) for q in OUT_OF_SCOPE]
    summary_rows.append({'cách': name, 'đúng hạng 1': f'{hit1}/{len(GOLD_QA)}', 'đúng trong 3': f'{hit3}/{len(GOLD_QA)}',
                         'cosine cao nhất · câu trong bài (thấp nhất)': round(min(d['cosine cao nhất'] for d in detail if d['cách'] == name), 3),
                         'cosine cao nhất · câu ngoài bài (cao nhất)': round(max(r['best_semantic_score'] for r in out), 3),
                         'reranker · ngoài bài (cao nhất)': round(max(r['best_rerank_score'] for r in out), 3) if out[0].get('best_rerank_score') is not None else '',
                         'ms/câu': round(1000 * (time.monotonic() - started) / (len(GOLD_QA) + len(OUT_OF_SCOPE)))})
table(summary_rows)
table(detail, limit=60)
"""

SPEED = """heading('7c · Thử tăng tốc OCR: fp16 thay bf16 (T4 không có nhân bf16)')
note('T4 tính bf16 bằng phần mềm, fp16 bằng tensor core. Thử đọc lại 3 slide bằng fp16, so thời gian và chữ đọc được. '
     'Nếu lỗi kiểu dữ liệu thì giữ bf16.')
import torch, difflib
sample = [r for r in records if r['ocr_status'] == 'read'][:3]
if sample:
    reader = shared_model(('hybrid_ocr', tuple(HYBRID_DEVICES)), lambda: load_hybrid_ocr(*HYBRID_DEVICES))
    rows, original_autocast = [], torch.autocast
    for dtype_name, dtype in (('bf16', torch.bfloat16), ('fp16', torch.float16)):
        try:
            reader.ocr_model.to(dtype)
            if dtype is torch.float16:   # infer() hard-codes autocast(bfloat16)
                torch.autocast = lambda device_type, dtype=None, **kw: original_autocast(device_type, dtype=torch.float16, **kw)
            for r in sample:
                item = by_segment[r['segment_id']]
                image = Image.open(stage01 / item['image_path']).convert('RGB')
                torch.cuda.synchronize(); started = time.monotonic()
                blocks, _, _ = reader(image)
                torch.cuda.synchronize()
                text = chr(10).join(b['text'] for b in blocks)
                rows.append({'kiểu': dtype_name, 'slide': item['sequence'], 'giây': round(time.monotonic() - started, 1),
                             'ký tự': len(text), 'giống lần chạy chính (%)': round(100 * difflib.SequenceMatcher(None, r['ocr_text'], text).ratio())})
        except Exception as error:
            rows.append({'kiểu': dtype_name, 'slide': '', 'giây': '', 'ký tự': '', 'giống lần chạy chính (%)': f'lỗi: {str(error)[:160]}'})
        finally:
            torch.autocast = original_autocast
            if hasattr(reader, 'ocr_model'):
                reader.ocr_model.to(torch.bfloat16)
    table(rows)
"""

SUMMARY = """heading('8 · Tổng kết')
import torch
table([{'bước': k, 'giây': v} for k, v in TIMES.items()])
table([{'GPU': i, 'bộ nhớ cao nhất (GB)': round(torch.cuda.max_memory_allocated(i) / 2**30, 2)} for i in range(torch.cuda.device_count())])
lesson_slides = [r for r in records if r['content_kind'] != 'ad']
note(f'{SOURCE["duration_sec"] / 60:.1f} phút video → {manifest01["visual_state_count"]} trạng thái hình → '
     f'{len(lesson_slides)} slide bài giảng ({manifest02["ocr_read_count"]} slide được OCR, {manifest02["ocr_skipped_no_content"]} bỏ qua vì không có nội dung) '
     f'+ {manifest01["ad_segment_count"]} đoạn quảng cáo, {manifest01.get("ad_flagged_count", 0)} đoạn nghi quảng cáo · {len(transcript["words"])} từ · {len(chunks)} đoạn chỉ mục.')
page = ('<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Pipeline · ' + VIDEO_ID + '</title>'
        '<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#fffaf3;color:#333}</style></head><body>'
        + ''.join(REPORT) + '</body></html>')
Path('/kaggle/working/walkthrough_report.html').write_text(page, encoding='utf-8')
export = Path('/kaggle/working/walkthrough_export')
shutil.copytree(OUTPUT_ROOT, export, ignore=shutil.ignore_patterns('audio', 'slide_audio', '*.wav', '_cache'), dirs_exist_ok=True)
shutil.make_archive('/kaggle/working/walkthrough_export', 'zip', export)
shutil.rmtree(export)
print('Đã lưu walkthrough_report.html và walkthrough_export.zip')
"""


def build():
    cells = [cell("markdown", INTRO),
             cell("code", "%pip install -q " + DEPS + ' "matplotlib"')]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in MODULES]
    cells += [cell("markdown", "## Cấu hình\n\nSửa ở đây để thử tham số khác."), cell("code", CONFIG),
              cell("code", HELPERS, hidden=True)]
    for title, source in [("0 · Video", VIDEO), ("Nạp model", MODELS), ("1 · Bước 01", STAGE01),
                          ("2 · Lọc quảng cáo", ADS), ("3 · Bước 03", STAGE03), ("4 · Bộ gác OCR + bước 02", STAGE02),
                          ("5 · Bước 04", STAGE04), ("6 · Bước 05", STAGE05), ("7 · Hỏi thử", RETRIEVE),
                          ("7b · Chấm điểm tìm kiếm", EVAL), ("7c · Thử tăng tốc OCR", SPEED),
                          ("8 · Tổng kết", SUMMARY)]:
        cells += [cell("markdown", "## " + title), cell("code", source)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "notebooks" / "full_pipeline_walkthrough.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target.name)


if __name__ == "__main__":
    build()
