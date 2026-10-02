"""Join the two branches without estimating slide times or inventing word timing."""


def words_to_text(words):
    text = " ".join(word["text"].strip() for word in words)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    return re.sub(r"([(\[])\s+", r"\1", text).strip()


def words_to_spans(words, segment_id, maximum_sec=18.0, maximum_words=70):
    spans, group = [], []

    def append_group():
        spans.append({
            "span_id": f"{segment_id}:speech_{len(spans):04d}",
            "start_sec": min(word["start_sec"] for word in group),
            "end_sec": max(word["end_sec"] for word in group),
            "text": words_to_text(group),
            "word_ids": [word["word_id"] for word in group],
            "timing_precision": "word",
        })

    for word in words:
        if group and (word["end_sec"] - group[0]["start_sec"] > maximum_sec
                      or len(group) >= maximum_words):
            append_group()
            group = []
        group.append(word)
        if re.search(r"[.!?]$", word["text"]) and len(group) >= 4:
            append_group()
            group = []
    if group:
        append_group()
    return spans


def align_records(intervals, visuals, transcript, duration):
    verify_intervals(intervals, duration)
    visual_by_segment = unique_records(visuals, "segment_id")
    expected = {item["segment_id"] for item in intervals}
    if set(visual_by_segment) != expected:
        raise ValueError("Nhánh OCR phải có đúng một record cho mỗi visual interval.")
    words = unique_records(transcript["words"], "word_id")
    owned = {item["segment_id"]: [] for item in intervals}
    for word in sorted(words.values(), key=lambda item: (item["start_sec"], item["end_sec"])):
        start, end = word["start_sec"], word["end_sec"]
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            raise ValueError(f"Word timing không hợp lệ: {word['word_id']}")
        interval = owner_interval(intervals, (start + end) / 2)
        owned[interval["segment_id"]].append(word)
    clips = unique_records(transcript["slide_audio_clips"], "segment_id")
    if transcript["status"] == "completed" and set(clips) != expected:
        raise ValueError("Thiếu audio clip để đối chiếu từng interval.")
    fallback = unique_records(transcript.get("fallback_spans", []), "span_id")
    for span in fallback.values():
        if not (0 <= span["start_sec"] < span["end_sec"] <= duration):
            raise ValueError(f"Fallback span không hợp lệ: {span['span_id']}")

    segments = []
    for index, interval in enumerate(intervals):
        segment_id = interval["segment_id"]
        visual = visual_by_segment[segment_id]
        spans = words_to_spans(owned[segment_id], segment_id)
        coarse_refs = [span["span_id"] for span in fallback.values()
                       if span["start_sec"] < interval["end_sec"]
                       and span["end_sec"] > interval["start_sec"]]
        segments.append({
            "segment_id": segment_id, "sequence": index,
            "start_sec": interval["start_sec"], "end_sec": interval["end_sec"],
            "previous_segment_id": intervals[index - 1]["segment_id"] if index else None,
            "next_segment_id": intervals[index + 1]["segment_id"]
            if index + 1 < len(intervals) else None,
            "content_kind": visual["content_kind"],
            "visual": {
                **visual,
                "image_path": interval["image_path"],
                "keyframe_time_sec": interval["keyframe_time_sec"],
            },
            "speech": {
                "text": words_to_text(owned[segment_id]), "spans": spans,
                "word_ids": [word["word_id"] for word in owned[segment_id]],
                "fallback_span_ids": coarse_refs,
                "status": "available" if spans or coarse_refs else
                ("no_audio" if transcript["status"] == "no_audio" else "no_speech_detected"),
            },
            "audio": clips.get(segment_id),
        })
    assigned = [word_id for segment in segments for word_id in segment["speech"]["word_ids"]]
    if len(assigned) != len(words) or set(assigned) != set(words):
        raise ValueError("Có từ bị mất hoặc bị gán lặp trong bước đồng bộ.")
    return segments


def run_alignment(stage01_dir, stage02_dir, stage03_dir, output_root,
                  video_id="L25_V001"):
    paths = [Path(stage01_dir), Path(stage02_dir), Path(stage03_dir)]
    parents = [load_stage(path, index + 1, video_id) for index, path in enumerate(paths)]
    require_parent(parents[1], parents[0])
    independent_asr = "1" not in parents[2]["parents"]
    if independent_asr:
        # Stage 03 ran in parallel with stage 01 from the same video file.
        require_same_source(parents[2], parents[0])
    else:
        require_parent(parents[2], parents[0])
    intervals = read_json(paths[0] / "intervals.json")
    visuals = read_json(paths[1] / "visual.json")
    transcript = read_json(paths[2] / "transcript.json")
    # Clean slide OCR (lines repeated on most slides, [Unreadable], lecturer/logo regions) and fix
    # lecture terms Whisper misheard using the words printed on the slides (textclean.py).
    lesson = [visual for visual in visuals if visual.get("content_kind") != "ad"]
    repeated = repeated_lines([visual.get("ocr_text", "") for visual in lesson])
    for visual in visuals:
        visual["ocr_text_clean"] = clean_ocr_text(visual.get("ocr_text", ""), repeated)
        lines = [line for line in visual["ocr_text_clean"].splitlines() if len(line.strip()) >= 12]
        visual["title_clean"] = (lines[0][:160] if lines and visual.get("content_kind") in {"slide", "mixed", "board"}
                                 else "")
        visual["regions"] = clean_regions(visual.get("regions", []))
    vocabulary = ocr_vocabulary([visual["ocr_text_clean"] for visual in lesson])
    transcript["words"], asr_corrections = correct_words(
        sorted(transcript["words"], key=lambda word: (word["start_sec"], word["end_sec"])), vocabulary)
    directory = begin_stage(output_root, 4)
    audio_root = paths[2]
    if independent_asr and transcript["status"] == "completed":
        pcm = read_pcm_audio(paths[2] / "audio/lecture_16k.wav")
        transcript["slide_audio_clips"] = []
        for interval in intervals:
            start_sample = round(interval["start_sec"] * 16000)
            end_sample = min(len(pcm), round(interval["end_sec"] * 16000))
            name = f"slide_audio/seg_{interval['sequence']:05d}.wav"
            save_pcm_clip(directory / name, pcm, start_sample, end_sample)
            transcript["slide_audio_clips"].append({
                "segment_id": interval["segment_id"], "path": name,
                "start_sec": start_sample / 16000, "end_sec": end_sample / 16000,
            })
        audio_root = directory
    segments = align_records(intervals, visuals, transcript, parents[0]["source"]["duration_sec"])
    for interval in intervals:
        name = interval["image_path"]
        target = relative_asset(directory, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative_asset(paths[0], name), target)
    for visual in visuals:
        if visual.get("content_image_path"):
            name = visual["content_image_path"]
            target = relative_asset(directory, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(relative_asset(paths[1], name), target)
        for region in visual["regions"]:
            name = region["image_path"]
            target = relative_asset(directory, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(relative_asset(paths[1], name), target)
    for clip in transcript["slide_audio_clips"] if audio_root != directory else ():
        name = clip["path"]
        target = relative_asset(directory, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative_asset(audio_root, name), target)

    write_json(directory / "segments.json", segments)
    write_json(directory / "asr_corrections.json", asr_corrections)
    write_json(directory / "words.json", transcript["words"])
    write_json(directory / "fallback_spans.json", transcript["fallback_spans"])
    write_json(directory / "asr_windows.json", transcript["windows"])
    # Compatible with the current app after mapping video_id to its uploaded ID.
    # Speech inside ad segments stays in segments.json but is not shown as lecture transcript.
    app_spans = [span for segment in segments if segment["content_kind"] != "ad"
                 for span in segment["speech"]["spans"]]
    app_spans += transcript["fallback_spans"]
    app_transcript = {
        "video_id": video_id, "language": transcript["language"],
        "segments": [
            {"start": span["start_sec"], "end": span["end_sec"], "text": span["text"]}
            for span in sorted(app_spans, key=lambda item: item["start_sec"])
        ],
    }
    write_json(directory / "transcript_app.json", app_transcript)
    manifest = finish_stage(
        directory, 4, video_id, parents[0]["source"],
        {"word_ownership": "midpoint_in_half_open_interval",
         "cross_boundary_fallback": "shared_reference_without_fabricated_word_times",
         "ocr_cleaning": {"repeated_lines": sorted(repeated)},
         "asr_correction": "slide_vocabulary_onset_s_x_ch_tr"},
        parents=parents,
        extra={"segment_count": len(segments), "word_count": len(transcript["words"]),
               "ad_segment_count": sum(segment["content_kind"] == "ad" for segment in segments),
               "asr_correction_count": len(asr_corrections),
               "noise_region_count": sum(region.get("noise", False) for visual in visuals for region in visual["regions"]),
               "ad_word_count": sum(len(segment["speech"]["word_ids"]) for segment in segments
                                    if segment["content_kind"] == "ad"),
               "bundle_status": "synchronized"},
    )
    return directory, manifest, segments
