"""Demo presentation, lifecycle and auditable notebook-only QA wiring."""
import copy
import gc
import html
import os
import subprocess
import sys
import time
import types
from collections import Counter
from IPython.display import HTML, Image as DisplayImage, display
from PIL import Image


def embedded_module(name, source, bindings=None):
    module = types.ModuleType(name)
    module.__dict__.update(bindings or {})
    sys.modules[name] = module
    exec(compile(source, name, 'exec'), module.__dict__)
    return module


class DemoStorage:
    @property
    def DATA_DIR(self):
        return OUTPUT_ROOT / 'app'

    @staticmethod
    def cloud():
        return False

    @staticmethod
    def load_video_meta(video_id):
        return {'original_filename': video_id}


def demo_table(rows, limit=None):
    rows = list(rows)
    limit = MAX_TABLE_ROWS if limit is None else limit
    if not rows:
        print('(no rows)')
        return
    columns = list(rows[0])
    esc = lambda value: html.escape(str(value))
    markup = '<table><tr>' + ''.join('<th>' + esc(k) + '</th>' for k in columns) + '</tr>'
    for row in rows[:limit]:
        markup += '<tr>' + ''.join('<td>' + esc(row.get(k, '')) + '</td>' for k in columns) + '</tr>'
    display(HTML(markup + '</table>'))
    if len(rows) > limit:
        print(f'Showing {limit}/{len(rows)} rows')


def demo_thumbnail(path, width=280):
    # Resize before embedding so long notebooks remain manageable.
    import io
    with Image.open(path) as opened:
        im = opened.convert('RGB')
        im.thumbnail((width, max(180, width)))
        buffer = io.BytesIO()
        im.save(buffer, format='JPEG', quality=75)
    return DisplayImage(data=buffer.getvalue())


def demo_startup():
    global DEMO_DATA, DEMO_TIMES, DEMO_REPORT, DEMO_DONE, RESOLVED_VIDEOS, VIDEO_IDS, VIDEO_LANGUAGES
    global ad_scorer, content_scorer, ocr_gate, speech_embed
    RESOLVED_VIDEOS = resolve_video_sources(globals().get('VIDEO_SOURCES'), VIDEO_IDS, VIDEO_LANGUAGES)
    VIDEO_IDS = list(RESOLVED_VIDEOS)
    VIDEO_LANGUAGES = {v: row['language'] for v, row in RESOLVED_VIDEOS.items()}
    DEMO_DATA, DEMO_TIMES, DEMO_REPORT = {}, {}, {}
    state = OUTPUT_ROOT / 'demo-completed-stages.json'
    DEMO_DONE = read_json(state) if state.exists() else {}
    for video_id in VIDEO_IDS:
        DEMO_DATA[video_id] = {'video': find_video_file(video_id)}
        DEMO_DATA[video_id]['source'] = probe_video(DEMO_DATA[video_id]['video'])
    print('Video đã xác định | video_id → đường dẫn | thời lượng | ngôn ngữ')
    demo_table([{'video_id': v, 'path': row['path'],
                 'duration_s': round(DEMO_DATA[v]['source']['duration_sec'], 2), 'language': row['language']}
                for v, row in RESOLVED_VIDEOS.items()], limit=len(VIDEO_IDS))
    demo_source_preflight(VIDEO_IDS, VIDEO_LANGUAGES, OUTPUT_ROOT)
    ad_scorer = content_scorer = ocr_gate = speech_embed = None
    print('DEMO setup | shared modules loaded | production recipe:', DEMO_RECIPE[:12])


def demo_stage(video_id, stage, action):
    """Resume only verified artifacts for the same source, code and configuration."""
    started = time.monotonic()
    folder = OUTPUT_ROOT / video_id / STAGE_NAMES[int(stage)]
    identity = fingerprint({'recipe': DEMO_RECIPE, 'source': DEMO_DATA[video_id]['source']['sha256'],
                            'language': VIDEO_LANGUAGES.get(video_id, 'vi')})
    key = video_id + ':' + stage
    resumed = False
    if RESUME_STAGES.get(stage) and DEMO_DONE.get(key) == identity:
        try:
            manifest = load_stage(folder, int(stage), video_id)
            require_same_source(manifest, {'video_id': video_id, 'pipeline_version': PIPELINE_VERSION,
                                           'source': DEMO_DATA[video_id]['source']})
            for parent in manifest['parents']:
                require_parent(manifest, load_stage(OUTPUT_ROOT / video_id / STAGE_NAMES[int(parent)],
                                                   int(parent), video_id))
            resumed = True
        except (OSError, ValueError):
            print('Resume validation failed; rerunning', video_id, stage)
    if not resumed:
        action()
        manifest = load_stage(folder, int(stage), video_id)
        downstream = {'01': ['02', '04', '05'], '02': ['04', '05'],
                      '03': ['04', '05'], '04': ['05'], '05': []}
        for child in downstream[stage]:
            DEMO_DONE.pop(video_id + ':' + child, None)
        DEMO_DONE[key] = identity
        write_json(OUTPUT_ROOT / 'demo-completed-stages.json', DEMO_DONE)
    seconds = time.monotonic() - started
    DEMO_TIMES[key] = {'video': video_id, 'stage': stage, 'seconds': round(seconds, 2), 'resumed': resumed}
    DEMO_DATA[video_id][stage] = folder
    print(f'DEMO {stage} | {video_id} | {seconds:.2f}s | resumed={resumed}')
    return folder, manifest


def demo_detection():
    for video_id in VIDEO_IDS:
        folder, manifest = demo_stage(video_id, '01', lambda: detect(video_id))
        intervals = read_json(folder / 'intervals.json')
        DEMO_DATA[video_id]['intervals'] = intervals
        print('States:', manifest['visual_state_count'], '| groups:', len(intervals),
              '| video seconds:', round(manifest['source']['duration_sec'], 2))
        demo_table([{'group': i['segment_id'], 'start': timestamp_label(i['start_sec']),
                     'end': timestamp_label(i['end_sec']), 'duration_s': round(i['end_sec'] - i['start_sec'], 2)}
                    for i in intervals])
        display(demo_thumbnail(folder / 'contact_sheet.jpg', width=1100))
        print('Contact sheet displayed (first 24 groups)')
        DEMO_REPORT[video_id] = {'states': manifest['visual_state_count'], 'groups': len(intervals),
                                 'duration_sec': manifest['source']['duration_sec']}


def demo_gate():
    global ocr_gate
    started = time.monotonic()
    ocr_gate = load_ocr_gate(AD_DEVICE, threshold=OCR_GATE_THRESHOLD, views=OCR_GATE_VIEWS) if OCR_GATE else None
    for video_id in VIDEO_IDS:
        rows = []
        for item in DEMO_DATA[video_id]['intervals']:
            skip, scores = True, {}
            if item.get('content_kind') != 'ad':
                with Image.open(DEMO_DATA[video_id]['01'] / item['image_path']) as opened:
                    image = opened.convert('RGB')
                rois = QWEN_CONTENT_ROIS
                roi, _ = qwen_content_roi(image, item, FRAME_HSV.get(video_id),
                                          rois.get(item['segment_id'], rois.get(video_id)))
                skip, scores = gate_keyframe(ocr_gate, image, roi)
            score = max([scores.get('frame', 0), *scores.get('crops', [])])
            rows.append({'group': item['segment_id'], 'kept': not skip, 'score': score,
                         'reason': 'ad' if item.get('content_kind') == 'ad' else
                         ('all views below threshold' if skip else 'readable content above threshold'),
                         'frame': scores.get('frame'), 'crops': scores.get('crops', [])})
        DEMO_DATA[video_id]['gate'] = rows
        kept = sum(r['kept'] for r in rows)
        print(f'DEMO gate | {video_id} | kept={kept} skipped={len(rows) - kept} | threshold={OCR_GATE_THRESHOLD}')
        demo_table(rows)
        DEMO_REPORT[video_id]['gate'] = {'kept': kept, 'skipped': len(rows) - kept}
    DEMO_TIMES['gate'] = {'video': 'all', 'stage': 'gate', 'seconds': round(time.monotonic() - started, 2), 'resumed': False}


def demo_ocr():
    for video_id in VIDEO_IDS:
        folder, manifest = demo_stage(video_id, '02', lambda: ocr(video_id, DEMO_DATA[video_id]['01']))
        records = read_json(folder / 'visual.json')
        retries = sum(max(0, len(r.get('ocr_generation', {}).get('attempts', [])) - 1) for r in records)
        print('OCR read:', manifest['ocr_read_count'], '| failed:', manifest.get('ocr_failed_count', 0),
              '| retried:', retries, '| skipped:', manifest['ocr_skipped_no_content'])
        for record, item in zip(records, DEMO_DATA[video_id]['intervals']):
            print(item['segment_id'], timestamp_label(item['start_sec']), record['ocr_status'])
            display(demo_thumbnail(DEMO_DATA[video_id]['01'] / item['image_path']))
            text = record.get('ocr_text') or '(no OCR text)'
            display(HTML('<details open><summary>OCR</summary><pre>' + html.escape(text[:MAX_TEXT_CHARS])
                         + (' … [truncated]' if len(text) > MAX_TEXT_CHARS else '') + '</pre></details>'))
        DEMO_REPORT[video_id]['ocr'] = {'read': manifest['ocr_read_count'],
                                       'failed': manifest.get('ocr_failed_count', 0), 'retried': retries}


def demo_asr():
    for video_id in VIDEO_IDS:
        folder, manifest = demo_stage(video_id, '03', lambda: asr(video_id))
        transcript = read_json(folder / 'transcript.json')
        print('Duration:', round(manifest['source']['duration_sec'], 2), 's | words:', len(transcript['words']))
        words = transcript['words']
        for offset in range(0, min(len(words), TRANSCRIPT_LINES * 18), 18):
            line = words[offset:offset + 18]
            print(f"[{timestamp_label(line[0]['start_sec'])}–{timestamp_label(line[-1]['end_sec'])}]",
                  ' '.join(w['text'] for w in line))
        print('Fallback spans:', len(transcript['fallback_spans']))
        DEMO_REPORT[video_id]['asr_words'] = len(words)


def demo_alignment():
    global speech_embed
    for video_id in VIDEO_IDS:
        def action():
            global speech_embed
            speech_embed = speech_embed_model() if SPEECH_RELEVANCE else None
            return run_alignment(DEMO_DATA[video_id]['01'], DEMO_DATA[video_id]['02'], DEMO_DATA[video_id]['03'],
                                 OUTPUT_ROOT / video_id, video_id, ad_suspect_overlap=AD_SUSPECT_OVERLAP,
                                 speech_embed=speech_embed, slide_speech=SLIDE_SPEECH if AD_FILTER == 'slide_speech' else None,
                                 conservative_ocr=CONSERVATIVE_OCR, heading_titles=HEADING_TITLES,
                                 asr_correction=ASR_CORRECTION, failure_guard=FAILURE_GUARD,
                                 speech_ad_filter=SPEECH_AD_FILTER)
        folder, manifest = demo_stage(video_id, '04', action)
        segments = read_json(folder / 'segments.json')
        demo_table([{'group': s['segment_id'], 'timestamp': timestamp_label(s['start_sec']),
                     'slide text': s['visual'].get('ocr_text_clean', '')[:MAX_TEXT_CHARS],
                     'assigned speech': s['speech']['text'][:MAX_TEXT_CHARS]} for s in segments[:EXAMPLE_GROUPS]])
        stats = {k: manifest.get(k, 0) for k in ['segment_count', 'asr_correction_count', 'noise_region_count', 'ad_word_count']}
        filter_path = folder / 'content_filter.json'
        if filter_path.exists():
            filt = read_json(filter_path)
            stats.update(filter_policy=filt.get('policy'), excluded_words=filt.get('excluded_word_count', 0),
                         excluded_spans=len(filt.get('excluded_spans', [])))
        print('Filter stats:', stats)
        DEMO_REPORT[video_id]['alignment'] = stats


def demo_index():
    for video_id in VIDEO_IDS:
        folder, _ = demo_stage(video_id, '05', lambda: run_indexing(
            DEMO_DATA[video_id]['04'], OUTPUT_ROOT / video_id, video_id,
            embedding_model=EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=EMBEDDING_DEVICE,
            dedupe_previous=OCR_DEDUPE_PREVIOUS, include_generated_descriptions=INCLUDE_GENERATED_DESCRIPTIONS,
            generated_semantic_only=GENERATED_SEMANTIC_ONLY, visual_state_timing=VISUAL_STATE_TIMING))
        chunks = [json.loads(line) for line in (folder / 'chunks.jsonl').read_text(encoding='utf-8').splitlines() if line]
        counts = dict(Counter(c['source_type'] for c in chunks))
        print('Chunks by source:', counts, '| total:', len(chunks))
        examples = [next(c for c in chunks if c['source_type'] == kind) for kind in counts]
        demo_table([{'source': c['source_type'], 'timestamp': timestamp_label(c['start_sec']),
                     'chunk': c['chunk_id'], 'text': c['text'][:MAX_TEXT_CHARS]} for c in examples])
        print('Index files:', ', '.join(p.name for p in sorted(folder.iterdir()) if p.is_file()))
        try:
            result = demo_validate_video_output(OUTPUT_ROOT / video_id, video_id)
            print('VALIDATION PASS | vectors:', result['vectors'])
        except Exception as exc:
            print('VALIDATION FAIL:', str(exc))
            raise
        target = OUTPUT_ROOT / 'app/index' / video_id
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(folder, target_is_directory=True)
        DEMO_REPORT[video_id]['index'] = {'counts': counts, 'chunks': len(chunks), 'validation': 'PASS'}


def start_demo_ollama():
    """Kaggle-only model transport, pinned to the project's reviewed Ollama recipe."""
    import hashlib
    import requests
    version = '0.12.6'
    digest = 'de82adce2ab79235115d511ff22fcb099ac53b67127870f12b80198c033ec0a1'
    archive = OUTPUT_ROOT / 'ollama.tgz'
    if not archive.exists():
        with requests.get(f'https://github.com/ollama/ollama/releases/download/v{version}/ollama-linux-amd64.tgz',
                          stream=True, timeout=(30, 60)) as reply:
            reply.raise_for_status()
            with archive.open('wb') as handle:
                for block in reply.iter_content(1 << 20):
                    handle.write(block)
    checksum = hashlib.sha256()
    with archive.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            checksum.update(block)
    if checksum.hexdigest() != digest:
        raise ValueError('Ollama archive checksum mismatch')
    install = OUTPUT_ROOT / 'ollama'
    install.mkdir(exist_ok=True)
    subprocess.run(['tar', '-xzf', str(archive), '-C', str(install)], check=True, timeout=180)
    env = dict(os.environ, OLLAMA_HOST='127.0.0.1:11434', OLLAMA_MODELS=str(OUTPUT_ROOT / 'ollama_models'),
               CUDA_VISIBLE_DEVICES='0')
    server_log = (OUTPUT_ROOT / 'ollama_server.log').open('w')
    server = subprocess.Popen([str(install / 'bin/ollama'), 'serve'], stdout=server_log, stderr=server_log, env=env)
    for _ in range(60):
        try:
            requests.get('http://127.0.0.1:11434/api/version', timeout=2).raise_for_status()
            break
        except requests.RequestException:
            if server.poll() is not None:
                raise RuntimeError('Ollama server exited; inspect ollama_server.log')
            time.sleep(1)
    else:
        raise TimeoutError('Ollama did not start')
    model = app_answer.DEFAULT_OLLAMA_MODEL
    manifest = requests.get('https://registry.ollama.ai/v2/library/qwen2.5/manifests/7b', timeout=30)
    manifest.raise_for_status()
    if hashlib.sha256(manifest.content).hexdigest() != '845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e':
        raise ValueError('Answer model alias changed since project pinning')
    requests.post('http://127.0.0.1:11434/api/pull', json={'model': model, 'stream': False}, timeout=600).raise_for_status()
    # Load on GPU now, so the query cell is ready for students.
    requests.post('http://127.0.0.1:11434/api/generate', json={'model': model, 'prompt': '',
                  'stream': False, 'keep_alive': -1}, timeout=180).raise_for_status()
    running = requests.get('http://127.0.0.1:11434/api/ps', timeout=10).json()['models']
    if not running or not all(m.get('size_vram', 0) > 0 for m in running):
        raise RuntimeError('Answer model did not load on the Kaggle GPU')
    print('Answer model loaded:', model, '| Ollama:', version, '| GPU MiB:',
          round(sum(m['size_vram'] for m in running) / 2**20))
    return server


def demo_query_setup():
    global ocr_gate, speech_embed, DEMO_SERVER
    started = time.monotonic()
    ocr_gate = speech_embed = None
    _SHARED_MODELS.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print('DEMO query_setup | pipeline model cache cleared')
    print(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader'], text=True).strip())
    os.environ['OLLAMA_MODEL'] = app_answer.DEFAULT_OLLAMA_MODEL
    os.environ['OLLAMA_BASE_URL'] = app_answer.DEFAULT_OLLAMA_URL
    enhanced = globals().get('DEMO_ENHANCED_CONTEXT', True)
    app_search.MAX_SLIDES = 8 if enhanced else 3
    os.environ['OLLAMA_NUM_CTX'] = '16384' if enhanced else '8192'
    os.environ['OLLAMA_MAX_TOKENS'] = '512'
    DEMO_SERVER = start_demo_ollama()
    print('Demo retrieval: E5 + BM25 | candidates per channel:', app_search.CANDIDATES,
          '| top-k slides:', app_search.MAX_SLIDES, '| RRF:', app_search.SEMANTIC_WEIGHT, app_search.LEXICAL_WEIGHT)
    DEMO_TIMES['query_setup'] = {'video': 'all', 'stage': 'query_setup',
                                'seconds': round(time.monotonic() - started, 2), 'resumed': False}


def demo_show_attribution(answer, found):
    report = app_answer.evidence_attribution(answer, found)
    print(app_answer.render_attribution(report, include_table=False))
    for item in report['used_evidence']:
        if item['source_type'] == 'ocr' and item.get('thumbnail_path'):
            print(f"Ảnh của [{item['evidence_label']}] — {app_answer.evidence_location(item)}")
            if os.path.isfile(item['thumbnail_path']):
                try:
                    display(demo_thumbnail(item['thumbnail_path'], 240))
                except Exception:
                    print('⚠ không hiển thị được ảnh keyframe')
            else:
                print('(không có ảnh keyframe tại đường dẫn đã lưu)')
    rows = app_answer.attribution_rows(report)
    print(f'Toàn bộ bằng chứng đã gửi LLM ({len(rows)} mục; hạng = thứ tự trong prompt; ✓ = được dẫn)')
    demo_table(rows, limit=len(rows))
    return report


def ask(question, video_id=None, *, history=None, query=None, show=True):
    started = time.monotonic()
    query = query or question
    history = (history or [])[-3:]
    selected = [video_id] if video_id else VIDEO_IDS
    if any(v not in VIDEO_IDS for v in selected):
        raise ValueError('Choose a video from VIDEO_IDS')
    enhanced = globals().get('DEMO_ENHANCED_CONTEXT', True)
    app_search.MAX_SLIDES = 8 if enhanced else 3
    found = app_search.search(selected[0], query) if len(selected) == 1 else app_search.search_many(
        selected, query, max_slides=app_search.MAX_SLIDES + 1)
    # The short fact-answer allowance cannot hold a multi-step explanation.
    found['answer_max_tokens'] = 768 if app_answer._explanation_request(question) else 512
    context = app_search.build_context(found) if enhanced else app_search._app_build_context(found, max_chars=8000)
    prompt = (app_answer._evidence_prompt(question, None, found, context, history=history, rewritten=query) if enhanced else
              app_answer.legacy_evidence_prompt(question, None, found, context, history=history, rewritten=query))
    # Preserve the exact legacy/new prompts in the captured timing output for QA.
    legacy = copy.deepcopy(found)
    legacy['slides'] = legacy['slides'][:3]
    legacy_context = app_search._app_build_context(legacy, max_chars=8000)
    diagnosis = {'question': question, 'query': query, 'candidate_ranks': found.get('candidate_ranks', []),
                 'query_vector': found.get('query_vector'), 'encoder_model': found.get('encoder_model'),
                 'legacy_slide_ids': [s.get('segment_id') for s in legacy['slides']],
                 'legacy_evidence': legacy.get('context_evidence', []),
                 'legacy_prompt': app_answer.legacy_evidence_prompt(question, None, legacy, legacy_context),
                 'slide_ids': [s.get('segment_id') for s in found['slides']],
                 'context_evidence': found.get('context_evidence', []),
                 'context_omitted': found.get('context_omitted', []),
                 'answer_max_tokens': found['answer_max_tokens'], 'prompt': prompt}
    DEMO_REPORT.setdefault('retrieval_diagnostics', []).append(diagnosis)
    evidence = found.get('context_evidence', [])
    if show:
        print('\nQUESTION:', question)
        if query != question:
            print('Truy vấn đã viết lại:', query)
        print('TOP-K EVIDENCE | slides:', len(found['slides']), '| best E5:', found['best_semantic'])
        for rank, slide in enumerate(found['slides'], 1):
            print(f"#{rank} {slide['video_id']} | [{app_search.clock(slide['start_sec'])}] | {slide['title'][:100]}")
            if slide.get('image') and os.path.isfile(slide['image']):
                display(demo_thumbnail(slide['image'], 240))
            for chunk in [c for c in evidence if c['slide_rank'] == rank - 1][:5]:
                print(f"  {chunk['source_type']} [{app_search.clock(chunk['start_sec'])}–{app_search.clock(chunk['end_sec'])}]",
                      chunk['text'][:320])
        if not evidence:
            print('(no evidence selected)')
    answer = app_answer._answer_with_citations(app_answer._ollama, prompt, found)
    if show:
        print('ANSWER:', answer[:2000])
    attribution = demo_show_attribution(answer, found) if show else app_answer.evidence_attribution(answer, found)
    elapsed = round(time.monotonic() - started, 2)
    if show:
        print('Query seconds:', elapsed)
    result = {'question': question, 'query': query, 'video_id': video_id, 'answer': answer, 'seconds': elapsed,
              'evidence_count': len(evidence), 'slides': len(found['slides']),
              'grounded_citation': bool(attribution['citation_matches']) and not attribution['unmatched_citations'],
              'used_evidence': attribution['used_evidence'],
              'citation_matches': attribution['citation_matches'],
              'unmatched_citations': attribution['unmatched_citations'],
              'all_evidence': attribution['all_evidence'],
              'attribution_text': app_answer.render_attribution(attribution),
              'attribution_summary': app_answer.attribution_summary(attribution)}
    DEMO_REPORT.setdefault('queries', []).append(result)
    DEMO_TIMES['query:' + str(len(DEMO_REPORT['queries']))] = {
        'video': ','.join(selected), 'stage': 'query', 'seconds': elapsed, 'resumed': False}
    return result


def demo_timing():
    print('DEMO timing | per-stage wall seconds (resume checks are labelled)')
    demo_table(DEMO_TIMES.values(), limit=100)
    elapsed = round(time.monotonic() - SESSION_STARTED, 2)
    print('Total session seconds:', elapsed)
    print('Stage results:', DEMO_REPORT)
    print('Output directory:', OUTPUT_ROOT)
    write_json(OUTPUT_ROOT / 'demo-summary.json', {'results': DEMO_REPORT, 'timings': list(DEMO_TIMES.values()),
                                                'elapsed_seconds': elapsed})
