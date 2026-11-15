import json, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor

def find_video_file(video_id):
    matches = sorted(Path('/kaggle/input').rglob(video_id + '.mp4'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Cần đúng một {video_id}.mp4, tìm thấy {matches}')
    return matches[0]

def speech_embed_model():
    from sentence_transformers import SentenceTransformer
    model = shared_model(('e5', EMBEDDING_MODEL, EMBEDDING_REVISION, EMBEDDING_DEVICE), lambda: SentenceTransformer(
        EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=EMBEDDING_DEVICE))
    return lambda texts: model.encode(['query: ' + t for t in texts], normalize_embeddings=True,
                                      batch_size=32, convert_to_numpy=True)

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
        ocr_device=globals().get('OCR_DEVICE', 'cuda'), ocr_scale=OCR_SCALE, ocr_batch_size=OCR_BATCH_SIZE,
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
        device=globals().get('ASR_DEVICE', 'auto'),
    )
    return directory, time.monotonic() - started

def ocr_after_detect(video_id, detect_job):
    return ocr(video_id, detect_job.result()[1])


def run_production_queue():
    global ad_scorer, ocr_gate, content_scorer, speech_embed
    restore_checkpoint(RESUME_FROM, OUTPUT_ROOT)
    progress = ProductionProgress(OUTPUT_ROOT, VIDEO_IDS, PRODUCTION_RECIPE)
    pending = []
    for video_id in VIDEO_IDS:
        result = completed_video(OUTPUT_ROOT, video_id, PRODUCTION_RECIPE, preflight['videos'][video_id])
        if result:
            progress.update(video_id, status='completed', skipped=True, validation=result,
                            ocr_failed=result['ocr_failed'], retries=result.get('retries', 0),
                            stage_timings=result.get('stage_timings', {}))
        else:
            pending.append(video_id)
    if not pending:
        return list(progress.rows.values())
    ad_scorer = None
    if AD_FILTER and AD_FILTER != 'slide_speech':
        student_path = AD_STUDENT_PATH or (next(iter(sorted(Path('/kaggle/input').rglob('ad_student.npz'))), None)
                                           if AD_FILTER == 'student' else None)
        if AD_FILTER == 'student' and student_path is None:
            raise FileNotFoundError('AD_FILTER=student requires ad_student.npz')
        ad_scorer = (load_outlier_scorer(AD_DEVICE, **OUTLIER_OPTIONS) if AD_FILTER == 'outlier'
                     else load_ad_scorer(AD_FILTER, AD_DEVICE, student_path))
    ocr_gate = load_ocr_gate(AD_DEVICE, threshold=OCR_GATE_THRESHOLD, views=OCR_GATE_VIEWS) if OCR_GATE else None
    content_scorer = load_ocr_gate(AD_DEVICE) if AD_FILTER == 'slide_speech' else None
    if AD_FILTER == 'slide_speech' and not SPEECH_RELEVANCE:
        raise ValueError('slide_speech requires SPEECH_RELEVANCE')
    speech_embed = speech_embed_model() if SPEECH_RELEVANCE else None
    with ThreadPoolExecutor(1) as cpu, ThreadPoolExecutor(1) as asr_queue, ThreadPoolExecutor(1) as ocr_queue:
        asr_jobs = {v: asr_queue.submit(progress.timed, '03', v, asr, v) for v in pending}
        detect_jobs = {v: cpu.submit(progress.timed, '01', v, detect, v) for v in pending}
        ocr_jobs = {v: ocr_queue.submit(progress.timed, '02', v, ocr_after_detect, v, detect_jobs[v]) for v in pending}
        for video_id in pending:
            # Await all three branches even on failure, so late worker updates cannot mask it.
            results, errors = [], []
            for job in (detect_jobs[video_id], ocr_jobs[video_id], asr_jobs[video_id]):
                try:
                    results.append(job.result())
                except Exception as exc:
                    errors.append(f'{type(exc).__name__}: {exc}')
            try:
                if errors:
                    raise RuntimeError('; '.join(errors))
                (video, stage01, manifest01, _), (stage02, _), (stage03, _) = results
                visual = read_json(stage02 / 'visual.json')
                manifest02 = read_json(stage02 / 'manifest.json')
                progress.update(video_id, duration_sec=manifest01['source']['duration_sec'],
                                slides=manifest01['interval_count'], visual_states=manifest01['visual_state_count'],
                                ocr_read=manifest02.get('ocr_read_count'),
                                ocr_failed=manifest02.get('ocr_failed_count', 0),
                                retries=sum(max(0, len(v.get('ocr_generation', {}).get('attempts', [])) - 1)
                                            for v in visual))
                stage04, _, _ = progress.timed('04', video_id, run_alignment,
                    stage01, stage02, stage03, OUTPUT_ROOT / video_id, video_id,
                    ad_suspect_overlap=AD_SUSPECT_OVERLAP, speech_embed=speech_embed,
                    slide_speech=SLIDE_SPEECH if AD_FILTER == 'slide_speech' else None,
                    conservative_ocr=CONSERVATIVE_OCR, heading_titles=HEADING_TITLES,
                    asr_correction=ASR_CORRECTION, failure_guard=FAILURE_GUARD,
                    speech_ad_filter=SPEECH_AD_FILTER)
                progress.timed('05', video_id, run_indexing, stage04, OUTPUT_ROOT / video_id, video_id,
                    embedding_model=EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
                    device=EMBEDDING_DEVICE, dedupe_previous=OCR_DEDUPE_PREVIOUS,
                    include_generated_descriptions=INCLUDE_GENERATED_DESCRIPTIONS,
                    generated_semantic_only=GENERATED_SEMANTIC_ONLY,
                    visual_state_timing=VISUAL_STATE_TIMING)
                validated = validate_video_output(OUTPUT_ROOT / video_id, video_id)
                write_json(OUTPUT_ROOT / video_id / 'completed.json', {
                    **validated, 'recipe': PRODUCTION_RECIPE, 'retries': progress.rows[video_id]['retries'],
                    'stage_timings': progress.rows[video_id]['stage_timings']})
                progress.update(video_id, status='completed', error=None, validation=validated)
            except Exception as exc:
                traceback.print_exc()
                progress.update(video_id, status='failed', error=f'{type(exc).__name__}: {exc}')
            print('PIPELINE', json.dumps(progress.rows[video_id]), flush=True)
    progress.flush()
    return list(progress.rows.values())


if CANARY:
    canary_result = run_memory_canary(globals(), preflight)
    if not canary_result['passed']:
        raise RuntimeError('CANARY FAILED; inspect canary.json; queue was not started')
else:
    with ProductionGPURecorder(OUTPUT_ROOT):
        timings = run_production_queue()
