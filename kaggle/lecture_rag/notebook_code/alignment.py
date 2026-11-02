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


def evidence_speech_spans(words, segment_id):
    """Split at excluded words; preserve original timing and ownership for later retries."""
    spans, group = [], []
    for word in [*words, None]:
        if word is None or word.get("evidence_excluded"):
            if group:
                spans.extend(words_to_spans(group, segment_id))
                group = []
        else:
            group.append(word)
    for index, span in enumerate(spans):
        span["span_id"] = f"{segment_id}:speech_{index:04d}"
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
            # ad filter in "flag" mode: kept and read, left out of the index only (stage 05)
            **({"ad_flagged": interval["ad_flagged"], "ad_score": interval.get("ad_score")}
               if "ad_flagged" in interval else {}),
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


def word_pairs(text):
    words = re.findall(r"\w+", (text or "").lower())
    return {f"{a} {b}" for a, b in zip(words, words[1:])}


# Words that promote rather than teach. A course ad shares the subject of the lesson, so the
# topic score alone can call it relevant; these terms are counted separately (content filter B).
AD_TERMS = re.compile(r"đăng ký|ưu đãi|học phí|hotline|khóa học|khoá học|giảm giá|khuyến mãi|tuyển sinh|"
                      r"liên hệ|link|subscribe|sponsor|discount|enroll|sign up|promo", re.I)


def speech_windows(words, size=45.0, step=15.0):
    """[{start_sec, end_sec, text}] windows of the transcript words (time order)."""
    if not words:
        return []
    end, windows, start = words[-1]["end_sec"], [], 0.0
    while start < end:
        inside = [w for w in words if start <= (w["start_sec"] + w["end_sec"]) / 2 < start + size]
        if inside:
            windows.append({"start_sec": start, "end_sec": min(start + size, end), "text": words_to_text(inside)})
        start += step
    return windows


def speech_relevance(windows, embed, reference_texts=(), trim=0.25):
    """Score each speech window "belongs to this lecture": cosine to the lecture's topic centre,
    standardised inside the video (median / MAD), so no threshold is shared across channels.
    embed(texts) -> unit vectors. The centre ignores the trim share of least typical windows
    (ads, chit-chat) and also averages reference_texts (the slide text). Adds topic_score,
    topic_z and ad_terms to every window (in place) and returns them."""
    import numpy as np

    if not windows:
        return windows
    vectors = np.asarray(embed([w["text"] for w in windows]), dtype="float32")
    references = np.asarray(embed(list(reference_texts)), dtype="float32") if reference_texts else None
    centre = vectors.mean(0)
    for _ in range(2):   # re-centre without the least typical windows
        scores = vectors @ (centre / (np.linalg.norm(centre) + 1e-9))
        keep = scores >= np.quantile(scores, trim)
        pool = vectors[keep] if references is None else np.vstack([vectors[keep], references])
        centre = pool.mean(0)
    scores = vectors @ (centre / (np.linalg.norm(centre) + 1e-9))
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median))) * 1.4826 or 1e-6
    for window, score in zip(windows, scores):
        window["topic_score"] = round(float(score), 4)
        window["topic_z"] = round((float(score) - median) / mad, 3)
        window["ad_terms"] = len(AD_TERMS.findall(window["text"]))
    return windows


def flag_runs(values, min_run=5, max_gap=2):
    """[start, end) index runs of True after filling gaps of <= max_gap between two runs and
    dropping runs shorter than min_run samples."""
    runs, start = [], None
    for index, value in enumerate(list(values) + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append([start, index]); start = None
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= max_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [run for run in merged if run[1] - run[0] >= min_run]


def slide_speech_flags(content, windows, not_slide=0.5, topic_z=-1.5, min_run=5, max_gap=2):
    """Content filter without a learned ad model: a 1 Hz sample is off-lecture when SigLIP says
    the frame shows no lecture content (1 - content score > not_slide) AND the speech around it is
    off the lecture topic (lowest topic_z of the covering speech windows < topic_z). A lecturer
    talking (no slide, on-topic speech) or a slide with silence stays lecture. L25 6 videos +
    MLS-15DA1: 92.5 % of ad seconds caught, 98.2 % / 98.9 % of lesson kept, no channel training.
    content: stage-01 content_scores.json; windows: speech_relevance() output.
    Returns [[start_sec, end_sec]] of off-lecture time."""
    times = content["times"]
    flags = []
    for time, score in zip(times, content["scores"]):
        near = [w["topic_z"] for w in windows if w["start_sec"] <= time < w["end_sec"]]
        flags.append(1 - score > not_slide and bool(near) and min(near) < topic_z)
    step = (times[-1] - times[0]) / max(1, len(times) - 1) if len(times) > 1 else 1.0
    return [[times[a], times[b - 1] + step] for a, b in flag_runs(flags, min_run, max_gap)]


def review_ad_speech(segments, threshold, min_pairs=20):
    """Stage 01 cuts ads from the picture alone; a lesson stretch it cut by mistake would lose
    its speech from the transcript and the index. An ad segment whose word pairs mostly come
    back in the rest of the lecture or on its slides is marked ad_suspect: its speech is kept.
    L25 (9 videos, 20261002 run): 27 real ads overlap 0.022-0.246; 110 runs of lesson slides
    (40-85 s) relabelled as ad overlap >= 0.207 (median 0.44). Threshold 0.25: 0/27 ads flagged,
    103/110 runs kept; 0.20: 1/27 and 110/110."""
    lesson = set()
    for segment in segments:
        if segment["content_kind"] != "ad":
            visual = segment.get("visual") or {}
            lesson |= word_pairs(segment["speech"]["text"])
            lesson |= word_pairs(visual.get("ocr_text_clean", visual.get("ocr_text", "")))
    for segment in segments:
        if segment["content_kind"] != "ad":
            continue
        pairs = word_pairs(segment["speech"]["text"])
        overlap = len(pairs & lesson) / len(pairs) if len(pairs) >= min_pairs else None
        segment["ad_review"] = {"lesson_overlap": None if overlap is None else round(overlap, 4),
                                "word_pairs": len(pairs), "threshold": threshold}
        segment["ad_suspect"] = overlap is not None and overlap >= threshold
    return segments


def run_alignment(stage01_dir, stage02_dir, stage03_dir, output_root,
                  video_id="L25_V001", ad_suspect_overlap=None, speech_embed=None, slide_speech=None,
                  conservative_ocr=False, heading_titles=False, asr_correction="legacy",
                  failure_guard=False, speech_ad_filter=False):
    """ad_suspect_overlap: None (off) or a threshold for review_ad_speech().
    speech_embed: None (off) or embed(texts) -> unit vectors (E5): every segment gets
    speech_topic = the lowest topic_z of the speech windows it overlaps (speech_relevance();
    recorded only, no decision yet) and speech_relevance.json keeps the windows.
    slide_speech: None (off) or slide_speech_flags() options ({} for the defaults): needs
    speech_embed and the stage-01 content_scores.json; a segment mostly inside off-lecture time
    gets ad_flagged (kept and read, left out of the index by stage 05)."""
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
    if asr_correction not in {"legacy", "off", "glossary", "glossary_lecture", "glossary_code"}:
        raise ValueError("asr_correction phải là legacy, off, glossary, glossary_lecture hoặc glossary_code.")
    if speech_ad_filter not in {False, True, "bounded_schools", "school_signatures"}:
        raise ValueError("speech_ad_filter phải là False, True, bounded_schools hoặc school_signatures.")
    if conservative_ocr not in {False, True, "canonical_subscripts"}:
        raise ValueError("conservative_ocr phải là False, True hoặc canonical_subscripts.")
    # Clean slide OCR (lines repeated on most slides, [Unreadable], lecturer/logo regions) and fix
    # lecture terms Whisper misheard using the words printed on the slides (textclean.py).
    lesson = [visual for visual in visuals if visual.get("content_kind") != "ad"]
    repeated = repeated_lines([visual.get("ocr_text", "") for visual in lesson])
    for visual in visuals:
        text = qwen_ocr_text(visual) if conservative_ocr else visual.get("ocr_text", "")
        if failure_guard:
            text, visual["ocr_quality"] = guard_ocr_text(text)
        visual["ocr_text_clean"] = clean_ocr_text(text, repeated, conservative=conservative_ocr,
                                                canonical_subscripts=conservative_ocr == "canonical_subscripts")
        if conservative_ocr:
            visual["ocr_layout_preserved"] = True
        lines = [line for line in visual["ocr_text_clean"].splitlines() if len(line.strip()) >= 12]
        visual["title_clean"] = (heading_title(visual["ocr_text_clean"]) if heading_titles
                                 and visual.get("content_kind") != "ad" else
                                 lines[0][:160] if not heading_titles and lines
                                 and visual.get("content_kind") in {"slide", "mixed", "board"} else "")
        visual["regions"] = clean_regions(visual.get("regions", []))
    previous = ""
    for visual in visuals:   # visual.json follows the interval order
        if visual.get("content_kind") == "ad":
            continue
        visual["ocr_text_new"], visual["ocr_repeat_share"] = lines_new_since(previous, visual["ocr_text_clean"])
        previous = visual["ocr_text_clean"] or previous
    vocabulary = ocr_vocabulary([visual["ocr_text_clean"] for visual in lesson])
    transcript["words"] = sorted(transcript["words"], key=lambda word: (word["start_sec"], word["end_sec"]))
    asr_loops = flag_asr_loops(transcript["words"]) if failure_guard else []
    asr_corrections = []
    if asr_correction == "legacy":
        transcript["words"], asr_corrections = correct_words(transcript["words"], vocabulary, atomic=False)
    else:
        for word in transcript["words"]:
            word.setdefault("text_asr", word["text"])
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
    if asr_correction in {"glossary", "glossary_lecture", "glossary_code"}:
        transcript["words"], asr_corrections = correct_glossary_words(
            transcript["words"], segments, lecture_support=asr_correction in {"glossary_lecture", "glossary_code"},
            code_feedback=asr_correction == "glossary_code")
        segments = align_records(intervals, visuals, transcript, parents[0]["source"]["duration_sec"])
    word_by_id = {word["word_id"]: word for word in transcript["words"]}
    retry_segments = []
    for segment in segments:
        quality = segment["visual"].get("ocr_quality", {})
        if quality.get("reasons"):
            retry_segments.append({"video_id": video_id, "segment_id": segment["segment_id"],
                                   "start_sec": segment["start_sec"], "end_sec": segment["end_sec"],
                                   "source": "ocr", "reasons": quality["reasons"],
                                   "status": quality["status"], "loops": quality["loops"],
                                   "raw_response_path": "_inputs/02_visual/visual.json"})
        owned_ids = set(segment["speech"]["word_ids"])
        for run in asr_loops:
            if owned_ids.intersection(run["word_ids"]):
                retry_segments.append({"video_id": video_id, "segment_id": segment["segment_id"],
                                       "source": "asr", "reasons": ["asr_repetition_loop"],
                                       **run, "raw_response_path": "_inputs/03_audio/transcript.json"})
        if failure_guard:
            owned = [word_by_id[word_id] for word_id in segment["speech"]["word_ids"]]
            segment["speech"]["text_raw"] = words_to_text([{**w, "text": w.get("text_asr", w["text"])} for w in owned])
            segment["speech"]["spans"] = evidence_speech_spans(owned, segment["segment_id"])
            segment["speech"]["text"] = " ".join(span["text"] for span in segment["speech"]["spans"])
            segment["speech"]["excluded_word_ids"] = [w["word_id"] for w in owned if w.get("evidence_excluded")]
            segment["speech"]["evidence_status"] = ("partial" if segment["speech"]["spans"] else "failed") if segment["speech"]["excluded_word_ids"] else "available"
    if ad_suspect_overlap is not None:
        review_ad_speech(segments, ad_suspect_overlap)
    windows = []
    if speech_embed is not None and transcript["words"]:
        slides = [visual["ocr_text_clean"] for visual in lesson if len(visual["ocr_text_clean"]) >= 40][:40]
        windows = speech_relevance(speech_windows(transcript["words"]), speech_embed, slides)
        for segment in segments:
            near = [w for w in windows if w["start_sec"] < segment["end_sec"] and w["end_sec"] > segment["start_sec"]]
            segment["speech_topic"] = ({"z": min(w["topic_z"] for w in near),
                                        "ad_terms": sum(w["ad_terms"] for w in near)} if near else None)
    filter_report = None
    if slide_speech is not None:
        content_path = paths[0] / "content_scores.json"
        if speech_embed is None or not content_path.exists():
            raise ValueError("slide_speech cần speech_embed và content_scores.json của bước 01 (content_scorer).")
        off = slide_speech_flags(read_json(content_path), windows, **slide_speech)
        for segment in segments:
            length = max(1e-6, segment["end_sec"] - segment["start_sec"])
            inside = sum(max(0.0, min(b, segment["end_sec"]) - max(a, segment["start_sec"])) for a, b in off)
            segment["ad_flagged"] = inside / length >= 0.5
            segment["ad_score"] = round(inside / length, 3)
        filter_report = {"mode": "slide_speech", "options": slide_speech, "off_lecture": [[round(a, 2), round(b, 2)] for a, b in off],
                         "off_lecture_seconds": round(sum(b - a for a, b in off), 1),
                         "flagged_segments": sum(s["ad_flagged"] for s in segments)}
    if speech_ad_filter:
        ad_spans = promotional_word_spans(transcript["words"], asian_spots="signature" if
                                         speech_ad_filter == "school_signatures" else speech_ad_filter == "bounded_schools")
        ad_ids = {word_id for span in ad_spans for word_id in span["word_ids"]}
        for word_id in ad_ids:
            word_by_id[word_id]["evidence_excluded"] = True
            word_by_id[word_id]["promotional"] = True
        for segment in segments:
            owned = [word_by_id[word_id] for word_id in segment["speech"]["word_ids"]]
            segment["speech"]["text_raw"] = words_to_text([{**w, "text": w.get("text_asr", w["text"])} for w in owned])
            segment["speech"]["spans_before_ad_filter"] = segment["speech"]["spans"]
            segment["speech"]["spans"] = evidence_speech_spans(owned, segment["segment_id"])
            segment["speech"]["text"] = " ".join(span["text"] for span in segment["speech"]["spans"])
            segment["speech"]["promotional_word_ids"] = [w["word_id"] for w in owned if w.get("promotional")]
        filter_report = {"mode": "speech_units", "policy": "explicit_school_signatures_and_cta" if
                         speech_ad_filter == "school_signatures" else "explicit_bounded_schools_and_cta" if
                         speech_ad_filter == "bounded_schools" else "explicit_bounded_spots_and_cta",
                         "excluded_spans": ad_spans, "excluded_word_count": len(ad_ids)}
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
    if windows:
        write_json(directory / "speech_relevance.json", windows)
    if filter_report:
        write_json(directory / "content_filter.json", filter_report)
    write_json(directory / "asr_corrections.json", asr_corrections)
    write_json(directory / "gpu_retry_segments.json", retry_segments)
    write_json(directory / "words.json", transcript["words"])
    write_json(directory / "fallback_spans.json", transcript["fallback_spans"])
    write_json(directory / "asr_windows.json", transcript["windows"])
    # Compatible with the current app after mapping video_id to its uploaded ID.
    # Speech inside ad segments stays in segments.json but is not shown as lecture transcript,
    # unless review_ad_speech() found it sounds like the lesson (ad_suspect).
    app_spans = [span for segment in segments if segment["content_kind"] != "ad" or segment.get("ad_suspect")
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
         "ocr_cleaning": {"repeated_lines": [] if conservative_ocr else sorted(repeated),
                          "conservative": conservative_ocr},
         "heading_titles": heading_titles, "failure_guard": failure_guard,
         "speech_ad_filter": speech_ad_filter, "asr_correction": asr_correction,
         **({"ad_suspect_overlap": ad_suspect_overlap} if ad_suspect_overlap is not None else {})},
        parents=parents,
        extra={"segment_count": len(segments), "word_count": len(transcript["words"]),
               "ad_segment_count": sum(segment["content_kind"] == "ad" for segment in segments),
               "ad_flagged_count": sum(bool(segment.get("ad_flagged")) for segment in segments),
               "asr_correction_count": len(asr_corrections),
               "noise_region_count": sum(region.get("noise", False) for visual in visuals for region in visual["regions"]),
               "ad_word_count": sum(len(segment["speech"]["word_ids"]) for segment in segments
                                    if segment["content_kind"] == "ad"),
               **({"ad_suspect_count": sum(bool(segment.get("ad_suspect")) for segment in segments)}
                  if ad_suspect_overlap is not None else {}),
               "bundle_status": "synchronized"},
    )
    return directory, manifest, segments
