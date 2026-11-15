"""Run matched production windows; preserve raw segment diagnostics and timing."""
install('faster-whisper==1.2.1', 'ctranslate2==4.8.3', 'numpy==2.1.3',
        'huggingface-hub==0.36.0', 'av==15.1.0')
import numpy as np
import ctranslate2
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel

save(OUT / 'environment.json', environment())
root = export_root()
save(OUT / 'config.json', CONFIG)


def prepare_audio(video):
    source = json.loads((root / video / '03_audio/manifest.json').read_text())['source']
    candidates = list(Path('/kaggle/input').rglob(video + '.mp4'))
    if not candidates:
        raise FileNotFoundError(video + '.mp4')
    mp4 = candidates[0]
    if file_sha256(mp4) != source['sha256']:
        raise ValueError('Video source hash mismatch: ' + video)
    wav = Path('/kaggle/working/audio') / (video + '.wav')
    wav.parent.mkdir(exist_ok=True)
    if not wav.exists():
        normalize_audio(mp4, source, wav)
    import wave
    with wave.open(str(wav), 'rb') as f:
        a = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float32) / 32768
    return a, source, wav


def run_model(label, model_id, revision, hints=False, local_path=None):
    if hints:
        ctranslate2.set_random_seed(20261101)
    before = time.monotonic()
    if model_id=='vinai/PhoWhisper-large' and not local_path:
        install('transformers==4.57.1','accelerate==1.11.0','sentencepiece==0.2.1')
        from ctranslate2.converters import TransformersConverter
        original_path=snapshot_download(model_id,revision=revision)
        local_path='/kaggle/working/phowhisper_ct2'
        TransformersConverter(original_path,copy_files=['tokenizer.json','preprocessor_config.json']).convert(
            local_path,quantization='float16',force=True)
    model_path = local_path or snapshot_download(model_id, revision=revision,
        allow_patterns=['config.json','model.bin','tokenizer.json','vocabulary.*','preprocessor_config.json'])
    model = WhisperModel(model_path, device='cuda', device_index=0, compute_type='float16')
    loading = time.monotonic() - before
    for video in CONFIG['videos']:
        folder = OUT / label / video
        a, source, wav = prepare_audio(video)
        original = json.loads((root / video / '03_audio/transcript.json').read_text())
        windows = original['windows']
        language = 'en' if video == 'cs_0001' else 'vi'
        words, fallback, diagnostics, times, peak = [], [], [], [], 0
        invalid_total=0
        started = time.monotonic()
        for i, window in enumerate(windows):
            if time.monotonic() - START > CONFIG['soft_limit_seconds']:
                raise TimeoutError('Soft GPU budget reached; completed-video checkpoints retained')
            audio = a[window['clip_start_sample']:window['clip_end_sample']]
            kw = dict(language=language, beam_size=5, temperature=0, vad_filter=True,
                      vad_parameters={'min_silence_duration_ms':500,'speech_pad_ms':250},
                      word_timestamps=True, condition_on_previous_text=False, initial_prompt=None)
            if hints:
                glossary = ', '.join(CONFIG['glossaries'][video])
                kw.update(initial_prompt=glossary, hotwords=glossary,
                          temperature=[0.0,0.2,0.4,0.6], compression_ratio_threshold=2.4,
                          hallucination_silence_threshold=2.0)
            t = time.monotonic()
            segments, info = model.transcribe(audio, **kw)
            segments = list(segments)
            raw = collect_asr_segments(segments, info)
            global_words, invalid = globalize_window_words(raw['words'], window, source['duration_sec'])
            invalid_total+=invalid
            words.extend(global_words)
            for span in raw['fallback_spans']:
                span = dict(span)
                span['start_sec'] += window['clip_start_sec']; span['end_sec'] += window['clip_start_sec']
                span['window_id'] = window['window_id']
                if window['core_start_sec'] <= (span['start_sec']+span['end_sec'])/2 < window['core_end_sec']:
                    span.setdefault('span_id',f'fallback_{len(fallback):07d}')
                    fallback.append(span)
            for s in segments:
                diagnostics.append({'window_id':window['window_id'], 'start':s.start,'end':s.end,
                    'text':s.text,'avg_logprob':s.avg_logprob,'compression_ratio':s.compression_ratio,
                    'no_speech_prob':s.no_speech_prob,'temperature':s.temperature,
                    'raw_words':[{'start':w.start,'end':w.end,'text':w.word,'probability':w.probability}
                                 for w in (s.words or [])]})
            times.append(time.monotonic() - t)
            mem = vram(); peak = max(peak, mem[0] if mem else 0)
            if (i+1)%10 == 0 or i+1==len(windows):
                print(label, video, i+1, '/', len(windows), round(sum(times),1), 'seconds', flush=True)
        words, removed = deduplicate_overlap_words(words)
        transcript = dict(original, words=words, fallback_spans=fallback, status='completed',
                          invalid_word_count=invalid_total, overlap_duplicates_removed=removed)
        save(folder / 'transcript.json', transcript)
        save(folder / 'diagnostics.json', diagnostics)
        save(folder / 'metrics.json', {'model_id':model_id,'revision':revision,'settings':kw,
            'audio_sha256':file_sha256(wav),'duration_seconds':source['duration_sec'],
            'decode_seconds':sum(times),'video_elapsed_seconds':time.monotonic()-started,
            'rtf':sum(times)/source['duration_sec'],'peak_sampled_device_vram_mib':peak,
            'model_loading_seconds':loading,'window_seconds':times})
        if hints:
            save(folder / 'hint_provenance.json', {'random_seed':20261101,
                'glossary':CONFIG['glossaries'][video],
                'combined_ablation':'initial_prompt + hotwords + temperature fallback + hallucination silence threshold'})
    del model
    gc.collect()


for spec in CONFIG['models']:
    try:
        run_model(**spec)
    except Exception as e:
        failure(spec['label'], e)
if CONFIG['models']:finish()
