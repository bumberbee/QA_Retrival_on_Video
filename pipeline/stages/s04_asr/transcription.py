"""Audio clock normalization, pause-aware ASR windows, word timing, slide clips."""


def normalize_audio(video_path, source, output_path, sample_rate=16000):
    origin, duration = source["timeline_origin_sec"], source["duration_sec"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep original timestamps before subtracting the common video clock origin.
    # first_pts=0 fills an initial audio delay; apad retains silent video endings.
    filters = (
        f"asetpts=PTS-({origin})/TB,"
        f"aresample={sample_rate}:async=1:first_pts=0,"
        f"apad,atrim=start=0:end={duration}"
    )
    run_command([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-copyts",
        "-i", str(video_path), "-map", "0:a:0", "-vn", "-af", filters,
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(output_path),
    ])


def silence_midpoints(pcm, sample_rate=16000, threshold_db=-38.0,
                       minimum_silence_sec=0.35):
    import numpy as np

    frame_length = int(sample_rate * 0.02)
    usable = len(pcm) // frame_length * frame_length
    if not usable:
        return []
    frames = pcm[:usable].astype("float32").reshape(-1, frame_length) / 32768.0
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    quiet = 20 * np.log10(np.maximum(rms, 1e-8)) <= threshold_db
    pauses, start = [], None
    for index, is_quiet in enumerate([*quiet, False]):
        if is_quiet and start is None:
            start = index
        if not is_quiet and start is not None:
            if (index - start) * 0.02 >= minimum_silence_sec:
                pauses.append((start + index) * 0.01)
            start = None
    return pauses


def make_asr_windows(duration, intervals, pauses=(), *, target_sec=45.0,
                     minimum_sec=20.0, maximum_sec=60.0, overlap_sec=1.25,
                     sample_rate=16000):
    if not (0 < minimum_sec <= target_sec <= maximum_sec
            and 0 <= overlap_sec < minimum_sec / 2 and duration > 0):
        raise ValueError("Thông số ASR window không hợp lệ.")
    slide_edges = [item["end_sec"] for item in intervals[:-1]]
    windows = []
    start = 0.0
    while start < duration - 1e-6:
        if duration - start <= maximum_sec:
            end = duration
        else:
            target = start + target_sec
            choices = [time for time in pauses
                       if start + minimum_sec <= time <= start + maximum_sec
                       and abs(time - target) <= 12.0]
            if choices:
                # Prefer nearby pauses; a coincident slide boundary is a small bonus.
                def cost(time):
                    slide_bonus = 2.0 if any(abs(time - edge) < 0.5
                                             for edge in slide_edges) else 0.0
                    return abs(time - target) - slide_bonus
                end = min(choices, key=cost)
            else:
                end = target
        clip_start_sample = max(0, math.floor((start - overlap_sec) * sample_rate))
        clip_end_sample = min(round(duration * sample_rate),
                              math.ceil((end + overlap_sec) * sample_rate))
        windows.append({
            "window_id": f"asr_{len(windows):05d}",
            "core_start_sec": start, "core_end_sec": end,
            "clip_start_sample": clip_start_sample, "clip_end_sample": clip_end_sample,
            "clip_start_sec": clip_start_sample / sample_rate,
            "clip_end_sec": clip_end_sample / sample_rate,
            "segment_ids": [item["segment_id"] for item in intervals
                            if item["start_sec"] < end and item["end_sec"] > start],
        })
        start = end
    return windows


def load_asr_model(model_id="Systran/faster-whisper-small", revision="main",
                   device="auto", device_index=0):
    import ctranslate2
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model_path = snapshot_download(
        model_id, revision=revision,
        allow_patterns=["config.json", "model.bin", "tokenizer.json",
                        "vocabulary.*", "preprocessor_config.json"],
    )
    try:
        model = WhisperModel(model_path, device=device, compute_type=compute_type,
                             device_index=device_index if device == "cuda" else 0)
    except RuntimeError as error:
        raise RuntimeError(
            f"Không khởi tạo được ASR trên {device}. Kiểm tra GPU/CUDA của Kaggle; "
            "có thể đặt ASR_DEVICE='cpu' để chạy int8. Lỗi gốc: " + str(error)
        ) from error
    return model, Path(model_path).name, device, compute_type


def transcribe_window(model, audio, language="vi", initial_prompt=None):
    segments, info = model.transcribe(
        audio, language=language, beam_size=5, temperature=0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 250},
        word_timestamps=True, condition_on_previous_text=False,
        initial_prompt=initial_prompt or None,
    )
    return collect_asr_segments(segments, info)


def transcribe_audio_batched(model, audio, language="vi", initial_prompt=None, batch_size=8):
    from faster_whisper import BatchedInferencePipeline

    pipeline = BatchedInferencePipeline(model=model)
    segments, info = pipeline.transcribe(
        audio, language=language, beam_size=5, batch_size=batch_size,
        vad_filter=True, word_timestamps=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 250},
        initial_prompt=initial_prompt or None,
    )
    return collect_asr_segments(segments, info)


def collect_asr_segments(segments, info):
    words, fallback_spans = [], []
    for segment in segments:
        if segment.words:
            words.extend({
                "start_sec": float(word.start), "end_sec": float(word.end),
                "text": word.word.strip(), "probability": float(word.probability),
            } for word in segment.words)
        elif segment.text.strip():
            # Never fabricate individual word timestamps from a segment.
            fallback_spans.append({
                "start_sec": float(segment.start), "end_sec": float(segment.end),
                "text": segment.text.strip(), "timing_precision": "segment",
            })
    return {"words": words, "fallback_spans": fallback_spans, "language": info.language,
            "duration_sec": getattr(info, "duration", None),
            "speech_duration_sec": getattr(info, "duration_after_vad", None)}


def globalize_window_words(raw_words, window, duration):
    result, invalid = [], 0
    offset = window["clip_start_sec"]
    for raw in raw_words:
        start, end = offset + raw["start_sec"], offset + raw["end_sec"]
        if (not math.isfinite(start) or not math.isfinite(end)
                or end < start or not raw["text"].strip()):
            invalid += 1
            continue
        start, end = max(0, start), min(duration, end)
        if start >= duration or end <= 0:
            invalid += 1
            continue
        end = min(duration, max(start + 1 / 16000, end))
        midpoint = (start + end) / 2
        if not window["core_start_sec"] <= midpoint < window["core_end_sec"]:
            continue
        probability = raw.get("probability")
        result.append({
            "start_sec": round(start, 6), "end_sec": round(end, 6),
            "text": raw["text"].strip(),
            "probability": probability if probability is not None
            and math.isfinite(probability) else None,
            "window_id": window["window_id"],
        })
    return result, invalid


def deduplicate_overlap_words(words):
    words = sorted(words, key=lambda word: (word["start_sec"], word["end_sec"]))
    kept, removed = [], 0
    for word in words:
        duplicate = False
        for previous in kept[-8:]:
            if previous["window_id"] == word["window_id"]:
                continue  # Repeated spoken words in one utterance are legitimate.
            if normalize_search(previous["text"]).strip(".,!?;:") != \
                    normalize_search(word["text"]).strip(".,!?;:"):
                continue
            overlap = max(0, min(word["end_sec"], previous["end_sec"])
                          - max(word["start_sec"], previous["start_sec"]))
            shorter = max(1e-6, min(word["end_sec"] - word["start_sec"],
                                   previous["end_sec"] - previous["start_sec"]))
            if overlap / shorter > 0.5:
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            kept.append(word)
    for index, word in enumerate(kept):
        word["word_id"] = f"word_{index:07d}"
    return kept, removed


def run_transcription(stage01_dir, video_path, output_root, video_id="L25_V001", *,
                      model_id="Systran/faster-whisper-small", revision="main",
                      language="vi", device="auto", initial_prompt=None,
                      target_sec=45.0, minimum_sec=20.0, maximum_sec=60.0,
                      overlap_sec=1.25, silence_threshold_db=-38.0,
                      transcription_mode="windowed", batch_size=8, device_index=0):
    import numpy as np

    if transcription_mode not in {"windowed", "batched"} or batch_size < 1:
        raise ValueError("ASR_MODE phải là windowed/batched; batch_size >= 1.")
    if stage01_dir is None:
        # ASR needs only the audio: it can run in parallel with stage 01. Slide
        # audio clips are then cut by stage 04, which has both branches.
        parent, intervals = None, []
        source = probe_video(video_path)
        duration = source["duration_sec"]
    else:
        parent = load_stage(stage01_dir, 1, video_id)
        if file_sha256(video_path) != parent["source"]["sha256"]:
            raise ValueError("Video ASR không khớp checksum video đã dùng để trích slide.")
        intervals = read_json(Path(stage01_dir) / "intervals.json")
        source, duration = parent["source"], parent["source"]["duration_sec"]
        verify_intervals(intervals, duration)
    directory = begin_stage(output_root, 3)
    parameters = {
        "model_id": model_id, "revision": revision, "language": language,
        "initial_prompt": initial_prompt, "target_sec": target_sec,
        "minimum_sec": minimum_sec, "maximum_sec": maximum_sec,
        "overlap_sec": overlap_sec, "silence_threshold_db": silence_threshold_db,
        "vad_filter": True, "word_timestamps": True, "condition_on_previous_text": False,
        "sample_rate": 16000,
        "transcription_mode": transcription_mode, "batch_size": batch_size,
        "window_controls_used": transcription_mode == "windowed",
        "packages": package_versions(["faster-whisper", "ctranslate2", "numpy"]),
    }
    all_words, fallback_spans, windows, slide_clips = [], [], [], []
    invalid_count, dedup_count = 0, 0
    if source["has_audio"]:
        audio_path = directory / "audio/lecture_16k.wav"
        normalize_audio(video_path, source, audio_path)
        pcm = read_pcm_audio(audio_path)
        if abs(len(pcm) / 16000 - duration) > 0.03:
            raise ValueError("Audio chuẩn hóa không còn khớp thời lượng video.")
        if transcription_mode == "batched":
            # VAD batching restores word times to the original normalized WAV.
            windows = [{
                "window_id": "asr_batched_full", "core_start_sec": 0.0,
                "core_end_sec": duration, "clip_start_sample": 0,
                "clip_end_sample": len(pcm), "clip_start_sec": 0.0,
                "clip_end_sec": len(pcm) / 16000,
                "segment_ids": [item["segment_id"] for item in intervals],
                "internal_segmentation": "faster_whisper_vad_with_restored_timestamps",
            }]
        else:
            pauses = silence_midpoints(pcm, threshold_db=silence_threshold_db)
            windows = make_asr_windows(
                duration, intervals, pauses, target_sec=target_sec, minimum_sec=minimum_sec,
                maximum_sec=maximum_sec, overlap_sec=overlap_sec,
            )
        model, resolved_revision, actual_device, compute_type = shared_model(
            ("asr", model_id, revision, device, device_index),
            lambda: load_asr_model(model_id, revision, device, device_index))
        parameters.update({
            "resolved_revision": resolved_revision,
            "device": actual_device, "compute_type": compute_type,
        })
        cache_directory = directory / "_cache" / fingerprint({
            "parameters": parameters,
            "parent": parent["artifact_id"] if parent else source["sha256"],
        })[:20]
        cache_directory.mkdir(parents=True, exist_ok=True)
        for index, window in enumerate(windows):
            cache = cache_directory / f"{window['window_id']}.json"
            if cache.exists():
                raw = read_json(cache)
            else:
                audio = pcm[window["clip_start_sample"]:window["clip_end_sample"]]
                normalized = audio.astype(np.float32) / 32768.0
                if transcription_mode == "batched":
                    raw = transcribe_audio_batched(model, normalized, language, initial_prompt, batch_size)
                else:
                    raw = transcribe_window(model, normalized, language, initial_prompt)
                write_json(cache, raw)
            if raw.get("speech_duration_sec") is not None:
                window["speech_duration_sec"] = raw["speech_duration_sec"]
            words, invalid = globalize_window_words(raw["words"], window, duration)
            invalid_count += invalid
            all_words.extend(words)
            for span in raw["fallback_spans"]:
                start = max(window["core_start_sec"],
                            window["clip_start_sec"] + span["start_sec"])
                end = min(window["core_end_sec"],
                          window["clip_start_sec"] + span["end_sec"])
                if start < end:
                    fallback_spans.append({
                        **span, "start_sec": start, "end_sec": end,
                        "window_id": window["window_id"],
                        "span_id": f"fallback_{len(fallback_spans):05d}",
                        "uncertainty": "Không có word timing; nội dung có thể chạm ranh giới.",
                    })
            print(f"ASR {index + 1}/{len(windows)}: "
                  f"{window['core_start_sec']:.1f}–{window['core_end_sec']:.1f}s")
        all_words, dedup_count = deduplicate_overlap_words(all_words)
        for interval in intervals:
            start_sample = round(interval["start_sec"] * 16000)
            end_sample = min(len(pcm), round(interval["end_sec"] * 16000))
            name = f"slide_audio/seg_{interval['sequence']:05d}.wav"
            save_pcm_clip(directory / name, pcm, start_sample, end_sample)
            slide_clips.append({
                "segment_id": interval["segment_id"], "path": name,
                "start_sec": start_sample / 16000, "end_sec": end_sample / 16000,
            })
    transcript = {
        "video_id": video_id, "language": language, "words": all_words,
        "fallback_spans": fallback_spans, "windows": windows,
        "slide_audio_clips": slide_clips,
        "status": "completed" if source["has_audio"] else "no_audio",
        "invalid_word_count": invalid_count, "overlap_duplicates_removed": dedup_count,
    }
    write_json(directory / "transcript.json", transcript)
    manifest = finish_stage(directory, 3, video_id, source, parameters,
                            parents=[parent] if parent else [],
                            extra={"word_count": len(all_words),
                                   "audio_status": transcript["status"]})
    return directory, manifest, transcript
