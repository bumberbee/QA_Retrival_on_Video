"""Production lifecycle helpers; embedded by the builder, model-free until canary."""

import time
import zipfile
from concurrent.futures import ThreadPoolExecutor


def startup_preflight(video_ids, languages, output_root, input_root='/kaggle/input', *, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    problems, videos = [], {}
    count = torch_module.cuda.device_count()
    devices = [{'index': i, 'name': torch_module.cuda.get_device_properties(i).name,
                'total_mib': torch_module.cuda.get_device_properties(i).total_memory / 2**20}
               for i in range(count)]
    if count != 2 or not torch_module.cuda.is_available():
        problems.append(f'exactly 2 CUDA devices required; found {count}')
    tools = {name: shutil.which(name) for name in ('ffmpeg', 'ffprobe')}
    if not all(tools.values()):
        problems.append('ffmpeg and ffprobe must both be available in PATH')
    for video_id in video_ids:
        matches = sorted(p for p in Path(input_root).rglob(video_id + '.mp4') if p.is_file())
        if len(matches) != 1:
            problems.append(f'{video_id}: exactly one MP4 required; found {len(matches)}: {matches}')
        else:
            videos[video_id] = str(matches[0])
    effective = {v: languages.get(v, 'vi') for v in video_ids}
    if any(lang not in {'vi', 'en'} for lang in effective.values()):
        problems.append(f'unsupported effective language map: {effective}')
    Path(output_root).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(Path(output_root).parent).free
    if free < 2 * 2**30:
        problems.append(f'insufficient working disk: {free / 2**30:.2f} GiB (minimum 2)')
    packages = package_versions(['torch', 'transformers', 'faster-whisper', 'ctranslate2',
                                 'sentence-transformers', 'huggingface_hub'])
    if 'not-installed' in packages.values():
        problems.append(f'missing production packages: {packages}')
    report = {'devices': devices, 'tools': tools, 'videos': videos, 'languages': effective,
              'working_free_gib': free / 2**30, 'packages': packages,
              'image_environment': {k: __import__('os').environ.get(k) for k in
                                    ('KAGGLE_DOCKER_IMAGE', 'KAGGLE_IMAGE_DIGEST')},
              'passed': not problems, 'errors': problems}
    write_json(Path(output_root) / 'preflight.json', report)
    print('PREFLIGHT', json.dumps(report), flush=True)
    if problems:
        raise RuntimeError('STARTUP PREFLIGHT FAILED: ' + '; '.join(problems))
    return report


class ProductionProgress:
    """Thread-safe atomic snapshot, persisted at every stage start/end and video exit."""
    def __init__(self, root, video_ids, recipe):
        self.root, self.recipe = Path(root), recipe
        self.started, self.lock = time.monotonic(), threading.RLock()
        self.spans = []
        self.rows = {v: {'video_id': v, 'status': 'pending', 'stage_timings': {},
                         'error': None, 'ocr_failed': 0, 'retries': 0} for v in video_ids}
        self.flush()

    def flush(self):
        with self.lock:
            payload = {'recipe': self.recipe, 'total_wall_sec': time.monotonic() - self.started,
                       'updated_utc': datetime.now(timezone.utc).isoformat(),
                       'videos': list(self.rows.values()), 'spans': self.spans}
            write_json(self.root / 'progress.json', payload)
            write_json(self.root / 'fast_pipeline_timings.json', payload)

    def update(self, video_id, **fields):
        with self.lock:
            self.rows[video_id].update(fields)
            self.flush()

    def timed(self, stage, video_id, fn, *args, **kwargs):
        start = time.monotonic()
        self.update(video_id, status='running')
        error = None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            with self.lock:
                elapsed = time.monotonic() - start
                self.rows[video_id]['stage_timings'][stage] = {
                    'seconds': elapsed, 'start': start - self.started,
                    'end': time.monotonic() - self.started, 'error': error}
                self.rows[video_id][stage + '_sec'] = elapsed
                self.spans.append({'stage': stage, 'video_id': video_id,
                                   **self.rows[video_id]['stage_timings'][stage]})
                if error:
                    self.rows[video_id].update(status='failed', error=error)
                self.flush()


def validate_video_output(root, video_id, *, allow_missing_audio=False):
    """Full manifest lineage, source, vector, image and evidence-time contract."""
    import numpy as np
    root = Path(root)
    manifests = []
    for stage in range(1, 6):
        directory = root / STAGE_NAMES[stage]
        manifest = load_stage(directory, stage, video_id, verify=not allow_missing_audio)
        if allow_missing_audio:
            for name, expected in manifest['files'].items():
                path = relative_asset(directory, name)
                if not path.exists() and (name.endswith('.wav') or name.startswith('audio/')):
                    continue
                if not path.is_file() or file_sha256(path) != expected:
                    raise ValueError(f'missing/corrupt delivered asset: {path}')
        manifests.append(manifest)
    require_same_source(*manifests)
    by_stage = {str(m['stage']): m for m in manifests}
    for child in manifests:
        for stage in child['parents']:
            require_parent(child, by_stage[stage])
    chunks = [json.loads(line) for line in (root / '05_index/chunks.jsonl').read_text(
        encoding='utf-8').splitlines() if line]
    vectors = np.load(root / '05_index/embeddings.npy', allow_pickle=False)
    if not chunks or vectors.shape != (len(chunks), 384):
        raise ValueError(f'chunk/vector shape mismatch: {len(chunks)}, {vectors.shape}')
    if manifests[-1].get('chunk_count') != len(chunks):
        raise ValueError('manifest chunk_count mismatch')
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
        raise ValueError('vectors must be finite and normalized')
    duration = manifests[0]['source']['duration_sec']
    for c in chunks:
        if not 0 <= c['start_sec'] < c['end_sec'] <= duration + .03:
            raise ValueError(f'chunk timestamp out of bounds: {c["chunk_id"]}')
        if c.get('image_path') and not relative_asset(root / '05_index', c['image_path']).is_file():
            raise ValueError(f'missing chunk image: {c["image_path"]}')
    intervals = read_json(root / '01_slides/intervals.json')
    verify_intervals(intervals, duration)
    for item in intervals:
        if not relative_asset(root / '01_slides', item['image_path']).is_file():
            raise ValueError('missing interval image')
    return {'video_id': video_id, 'source_sha256': manifests[0]['source']['sha256'],
            'index_artifact_id': manifests[-1]['artifact_id'], 'chunks': len(chunks),
            'vectors': list(vectors.shape), 'ocr_failed': manifests[1].get('ocr_failed_count', 0),
            'long_dynamic_review': [i['segment_id'] for i in intervals
                                    if i.get('review_long_dynamic') or i.get('long_dynamic_review')],
            'passed': True}


def restore_checkpoint(resume_from, output_root):
    """Attached checkpoint directory only; retain partial fingerprinted OCR/ASR caches."""
    if resume_from is None:
        return
    source, target = Path(resume_from).resolve(), Path(output_root).resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError('checkpoint source and output root must be separate directories')
    if not source.is_dir():
        raise FileNotFoundError(f'RESUME_FROM checkpoint directory missing: {source}')
    shutil.copytree(source, target, dirs_exist_ok=True)


def completed_video(root, video_id, recipe, source_file):
    marker = Path(root) / video_id / 'completed.json'
    if not marker.exists():
        return None
    try:
        stored = read_json(marker)
        if stored['recipe'] != recipe or stored['source_sha256'] != file_sha256(source_file):
            return None
        validated = validate_video_output(Path(root) / video_id, video_id)
        if validated['index_artifact_id'] != stored['index_artifact_id']:
            return None
        return {**validated, **{k: stored[k] for k in ('retries', 'stage_timings') if k in stored}}
    except (ValueError, OSError, KeyError) as exc:
        print(f'RESUME validation failed for {video_id}; rebuilding: {exc}', flush=True)
        return None


def package_production_outputs(root, destination):
    """Stream both archives without duplicating the output tree or including weights/videos."""
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    forbidden = {'.mp4', '.mkv', '.safetensors', '.bin', '.pt', '.pth'}
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix not in forbidden
             and p.suffix != '.tmp']
    def archive(path, selected):
        temporary = path.with_suffix('.zip.tmp')
        try:
            with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as z:
                for p in selected:
                    z.write(p, p.relative_to(root).as_posix())
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return str(path)
    slim = [p for p in files if p.suffix != '.wav'
            and not any(part in {'audio', '_cache'} for part in p.relative_to(root).parts)]
    result = {'delivery': archive(destination / 'fast_pipeline_export.zip', slim)}
    required = sum(p.stat().st_size for p in files) + 1024**3
    free = shutil.disk_usage(destination).free
    if free >= required:
        result['checkpoint'] = archive(destination / 'fast_pipeline_checkpoint.zip', files)
    else:
        result['checkpoint'] = None
        result['warning'] = f'checkpoint skipped: free {free} bytes, conservative requirement {required}'
        print('WARNING', result['warning'], flush=True)
    write_json(root / 'packaging.json', result)
    return result


def gpu_memory_sample():
    out = run_command(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,memory.total',
                       '--format=csv,noheader,nounits'])
    return [{'index': int(a), 'used_mib': int(b), 'free_mib': int(c), 'total_mib': int(d)}
            for a, b, c, d in (line.split(',') for line in out.strip().splitlines())]


class ProductionGPURecorder:
    def __init__(self, root, interval=2):
        self.path, self.interval = Path(root) / 'gpu_memory.json', interval
        self.stop, self.peaks = threading.Event(), {}

    def sample(self):
        while not self.stop.is_set():
            try:
                for gpu in gpu_memory_sample():
                    i = str(gpu['index'])
                    previous = self.peaks.get(i, gpu)
                    self.peaks[i] = {**gpu, 'max_used_mib': max(previous.get('max_used_mib', 0), gpu['used_mib']),
                                     'min_free_mib': min(previous.get('min_free_mib', gpu['free_mib']), gpu['free_mib'])}
                write_json(self.path, self.peaks)
            except Exception as exc:
                print('GPU sampler warning:', str(exc), flush=True)
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=10)
        write_json(self.path, self.peaks)


def run_memory_canary(config, preflight, *, sample_interval=.2):
    """Production shared keys; both Qwen copies stress concurrently with CT2 and E5."""
    import numpy as np
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    root = Path(config['OUTPUT_ROOT'])
    report = {'passed': False, 'layout': {k: config[k] for k in
              ('HYBRID_DEVICES', 'ASR_DEVICE_INDEX', 'EMBEDDING_DEVICE', 'AD_DEVICE')},
              'samples': [], 'errors': [], 'forced_tokens': 2048}
    stop = threading.Event()
    started = time.monotonic()
    def sample():
        while not stop.is_set():
            try:
                report['samples'].append({'elapsed_sec': time.monotonic() - started,
                                           'gpus': gpu_memory_sample()})
            except Exception as exc:
                report['errors'].append(str(exc))
            stop.wait(sample_interval)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        if not config['OCR_GATE']:
            raise ValueError('canary requires the real SigLIP gate')
        gate = load_ocr_gate(config['AD_DEVICE'], threshold=config['OCR_GATE_THRESHOLD'],
                             views=config['OCR_GATE_VIEWS'])
        options = dict(model_id=config['QWEN_OCR_MODEL'], revision=config['QWEN_OCR_REVISION'],
                       min_pixels=config['QWEN_MIN_PIXELS'], max_pixels=config['QWEN_MAX_PIXELS'],
                       validate=config['QWEN_VALIDATE'], retry_invalid=config['QWEN_RETRY_INVALID'])
        qwen = shared_qwen_ocr(config['HYBRID_DEVICES'], **options)
        # This is the exact key used by run_transcription (including device='auto').
        asr = shared_model(('asr', config['ASR_MODEL'], config['ASR_REVISION'], 'auto',
                            config['ASR_DEVICE_INDEX']), lambda: load_asr_model(
                                config['ASR_MODEL'], config['ASR_REVISION'], 'auto', config['ASR_DEVICE_INDEX']))[0]
        e5 = shared_model(('e5', config['EMBEDDING_MODEL'], config['EMBEDDING_REVISION'],
                          config['EMBEDDING_DEVICE']), lambda: SentenceTransformer(
                              config['EMBEDDING_MODEL'], revision=config['EMBEDDING_REVISION'],
                              device=config['EMBEDDING_DEVICE']))
        video = preflight['videos']['hVU5Mw19V_c']
        frame, wav = root / 'canary_crop.png', root / 'canary_window.wav'
        run_command(['ffmpeg', '-y', '-ss', '300', '-i', video, '-frames:v', '1', str(frame)])
        run_command(['ffmpeg', '-y', '-ss', '300', '-i', video, '-t', '60', '-vn',
                     '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(wav)])
        with Image.open(frame) as im:
            crop = crop_to_roi(im.convert('RGB'), config['QWEN_CONTENT_ROIS'].get('hVU5Mw19V_c'))
            crop = crop.resize((1280, 1024))  # 1,310,720 pixels: production upper bound
        audio = read_pcm_audio(wav).astype(np.float32) / 32768
        for i in range(2):
            torch.cuda.reset_peak_memory_stats(i)
        barrier = threading.Barrier(3)
        def work(fn):
            barrier.wait(timeout=60)
            return fn()
        with ThreadPoolExecutor(3) as pool:
            jobs = [pool.submit(work, lambda: qwen([crop.copy() for _ in config['HYBRID_DEVICES']],
                                                   'vi', force_tokens=2048)),
                    pool.submit(work, lambda: transcribe_window(asr, audio, 'vi')),
                    pool.submit(work, lambda: e5.encode(['query: ' + ('Bayes xác suất PPF len print end ' * 40)] * 32,
                                                        batch_size=32, normalize_embeddings=True,
                                                        convert_to_numpy=True))]
            answers = jobs[0].result()
            jobs[1].result()
            vectors = jobs[2].result()
        report['generated_tokens'] = [a['attempts'][0]['generated_tokens'] for a in answers]
        if report['generated_tokens'] != [2048] * len(config['HYBRID_DEVICES']):
            raise ValueError('forced token stress did not generate exactly 2048 tokens per Qwen copy')
        if vectors.shape != (32, 384):
            raise ValueError('canary E5 batch shape mismatch')
        report['torch_peak_mib'] = {str(i): {'allocated': torch.cuda.max_memory_allocated(i) / 2**20,
                                          'reserved': torch.cuda.max_memory_reserved(i) / 2**20} for i in range(2)}
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        stop.set()
        sampler.join(timeout=5)
        report['memory'] = [{'index': i,
                             'max_used_mib': max((g['used_mib'] for s in report['samples'] for g in s['gpus']
                                                 if g['index'] == i), default=None),
                             'min_free_mib': min((g['free_mib'] for s in report['samples'] for g in s['gpus']
                                                 if g['index'] == i), default=0)} for i in range(2)]
        report['passed'] = not report['errors'] and all(g['min_free_mib'] >= 1536 for g in report['memory'])
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(root / 'canary.json', report)
        print('CANARY', json.dumps({k: v for k, v in report.items() if k != 'samples'}), flush=True)
    return report
