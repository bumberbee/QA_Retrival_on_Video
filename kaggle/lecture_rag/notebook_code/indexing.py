"""Evidence chunks, multilingual E5 vectors, and a Vietnamese-friendly FTS5 index."""


def split_text_tokens(text, tokenizer, limit=350, overlap=40):
    if not 0 <= overlap < limit:
        raise ValueError("Cần 0 <= overlap < giới hạn token.")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    parts = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + limit)
        part = tokenizer.decode(tokens[start:end], skip_special_tokens=True).strip()
        if part:
            parts.append(part)
        if end == len(tokens):
            break
        start = end - overlap
    return parts


def split_layout_tokens(text, tokenizer, limit, overlap):
    """Keep literal OCR whitespace; overlap complete lines rather than decoded code tokens."""
    parts, pending = [], []
    for line in text.splitlines():
        proposed = "\n".join([*pending, line])
        if pending and len(tokenizer.encode(proposed, add_special_tokens=False)) > limit:
            parts.append("\n".join(pending))
            tail = []
            for previous in reversed(pending):
                if len(tokenizer.encode("\n".join([previous, *tail]), add_special_tokens=False)) > overlap:
                    break
                tail.insert(0, previous)
            pending = tail
            if len(tokenizer.encode("\n".join([*pending, line]), add_special_tokens=False)) > limit:
                pending = []
        if len(tokenizer.encode(line, add_special_tokens=False)) > limit:
            parts.extend(split_text_tokens(line, tokenizer, limit, overlap))
        else:
            pending.append(line)
    if pending:
        parts.append("\n".join(pending))
    return [part for part in parts if part.strip()]


def usable_figure_description(text):
    """Generated no-figure placeholders, punctuation and repetition are not evidence."""
    text = (text or "").strip()
    folded = unicodedata.normalize("NFD", text.lower())
    folded = "".join(c for c in folded if unicodedata.category(c) != "Mn").replace("đ", "d")
    return bool(re.search(r"[^\W\d_]", text) and not re.match(
        r"^(?:khong co (?:hinh|bieu do|so do)|no (?:figure|image|diagram)|there is no (?:figure|image|diagram))\b", folded)
        and not re.search(r"(\b\w+(?:\s+\w+){0,7}\b)(?:\s+\1){7,}", folded))


def group_speech_spans(spans, tokenizer, maximum_tokens=180, maximum_seconds=50.0):
    """Keep complete sentences together; never merge across visual intervals."""
    groups, pending = [], []

    def finish():
        if pending:
            groups.append({
                "source_type": "speech", "text": " ".join(span["text"] for span in pending),
                "start_sec": min(span["start_sec"] for span in pending),
                "end_sec": max(span["end_sec"] for span in pending),
                "source_ids": [span["span_id"] for span in pending],
                "timing_precision": "word_span",
            })

    for span in spans:
        proposed = " ".join(item["text"] for item in [*pending, span])
        if pending and (len(tokenizer.encode(proposed, add_special_tokens=False)) > maximum_tokens
                        or span["end_sec"] - pending[0]["start_sec"] > maximum_seconds
                        or span["start_sec"] - pending[-1]["end_sec"] > 3.0):
            finish()
            pending = []
        pending.append(span)
    finish()
    return groups


def make_evidence_chunks(segments, fallback_spans, tokenizer, *,
                         chunk_tokens=350, overlap_tokens=40, model_limit=512,
                         video_id="L25_V001", speech_chunk_tokens=180,
                         speech_maximum_seconds=50.0, include_generated_descriptions=False,
                         dedupe_previous=False, passage_prefix="passage: ", generated_semantic_only=False):
    """passage_prefix: "passage: " for E5, "" for bge-m3. dedupe_previous: index only the OCR lines not already on the previous slide (stage 04
    ocr_text_new), so a repeated slide does not crowd the search with copies."""
    if not 0 <= overlap_tokens < chunk_tokens <= model_limit - 32:
        raise ValueError("Giới hạn chunk không tương thích model tokenizer.")
    chunks = []
    seen_coarse = set()
    coarse = unique_records(fallback_spans, "span_id")
    for segment in segments:
        if segment.get("ad_flagged") or (segment.get("content_kind") == "ad" and not segment.get("ad_suspect")):
            # Ads are not lesson evidence; their speech is kept in stage 04 only. An ad_suspect
            # segment (its speech sounds like the lesson, stage 04) is indexed by its speech.
            # ad_flagged (ad filter in "flag" mode): OCR and speech are kept in stage 04 only.
            seen_coarse.update(segment["speech"]["fallback_span_ids"])
            continue
        visual = segment["visual"]
        visual_failed = visual.get("ocr_quality", {}).get("evidence_excluded", False)
        visual_start = visual.get("representative_content_onset_sec", segment["start_sec"])
        if not segment["start_sec"] <= visual_start < segment["end_sec"]:
            raise ValueError("Representative visual onset must be inside its interval")
        visual_timing = ("visual_state" if "representative_content_onset_sec" in visual else "visual_interval")
        # Bound the title separately so it cannot consume the encoder budget.
        title_tokens = tokenizer.encode("" if visual_failed else visual.get("title_clean", visual.get("title", "")),
                                        add_special_tokens=False)[:64]
        title = tokenizer.decode(title_tokens, skip_special_tokens=True).strip()
        header = f"{passage_prefix}{title}\n" if title else passage_prefix
        budget = min(chunk_tokens, model_limit
                     - len(tokenizer.encode(header, add_special_tokens=True)) - 8)
        units = group_speech_spans(
            segment["speech"]["spans"], tokenizer,
            min(speech_chunk_tokens, budget), speech_maximum_seconds,
        )
        for span_id in segment["speech"]["fallback_span_ids"]:
            if span_id in seen_coarse:
                continue
            seen_coarse.add(span_id)
            span = coarse[span_id]
            units.append({
                "source_type": "speech", "text": span["text"],
                "start_sec": span["start_sec"], "end_sec": span["end_sec"],
                "source_ids": [span_id], "timing_precision": "asr_segment",
                "uncertainty": span.get("uncertainty"),
            })
        for source_type, text in [
            ("ocr", visual["ocr_text_new"] if dedupe_previous and "ocr_text_new" in visual
             else visual.get("ocr_text_clean", visual.get("ocr_text", ""))),
            ("visual", visual.get("visual_description", "")),
        ]:
            if visual_failed:
                continue
            # Older stage-02 bundles may store malformed JSON or degenerate
            # generation (e.g. repeated punctuation) as visual_description.
            # Preserve those raw records in lecture/, but never index them.
            if source_type == "visual" and visual.get("description_status") != "json":
                continue
            if (source_type == "visual" and not include_generated_descriptions
                    and visual.get("description_verification") != "human_verified"):
                continue
            generated = source_type == "visual" and visual.get("description_verification") != "human_verified"
            if generated and not usable_figure_description(text):
                continue
            if text.strip():
                units.append({
                    "source_type": source_type, "text": text,
                    "start_sec": visual_start, "end_sec": segment["end_sec"],
                    "source_ids": [visual["occurrence_id"]],
                    "timing_precision": visual_timing,
                    "uncertainty": "OCR/mô tả tự động; đối chiếu hình gốc khi cần.",
                    **({"layout_preserved": True} if source_type == "ocr" and visual.get("ocr_layout_preserved") else {}),
                    **({"verification": "model_generated_unverified",
                        "retrieval_policy": "semantic_only" if generated_semantic_only else "hybrid"} if generated else {}),
                })
        for region in visual.get("regions", []):
            if visual_failed or region.get("noise"):
                continue
            if (not include_generated_descriptions and region.get("extraction_status")
                    not in {"human_verified", "hybrid_ocr_unverified"}):
                continue
            text = "\n".join(value for value in
                             [region.get("text"), region.get("description")] if value)
            generated = region.get("extraction_status") == "model_generated_unverified"
            if generated and not usable_figure_description(text):
                continue
            if text.strip():
                units.append({
                    "source_type": "region", "text": text,
                    "start_sec": visual_start, "end_sec": segment["end_sec"],
                    "source_ids": [region["region_id"]],
                    "image_path": region["image_path"],
                    "timing_precision": visual_timing,
                    "uncertainty": "Vùng và nội dung do mô hình đề xuất, chưa xác minh.",
                    **({"verification": "model_generated_unverified",
                        "retrieval_policy": "semantic_only" if generated_semantic_only else "hybrid"} if generated else {}),
                })
        for unit_index, unit in enumerate(units):
            split = split_layout_tokens if unit.get("layout_preserved") else split_text_tokens
            parts = split(unit["text"], tokenizer, budget, min(overlap_tokens, budget - 1))
            for part_index, text in enumerate(parts):
                embedding_text = header + text
                count = len(tokenizer.encode(embedding_text, add_special_tokens=True))
                if count > model_limit:
                    raise ValueError("Chunk vượt giới hạn encoder; giảm CHUNK_TOKENS.")
                chunks.append({
                    **{key: value for key, value in unit.items() if key != "text"},
                    "chunk_id": f"{segment['segment_id']}:u{unit_index:04d}:p{part_index:03d}",
                    "video_id": video_id, "segment_id": segment["segment_id"],
                    "sequence": segment["sequence"], "text": text,
                    "title": title, "embedding_text": embedding_text,
                    "encoded_tokens": count,
                    "image_path": unit.get("image_path", visual.get("content_image_path")
                                           or visual["image_path"]),
                    "keyframe_time_sec": visual["keyframe_time_sec"],
                    "occurrence_id": visual["occurrence_id"],
                })
    if not chunks:
        raise ValueError("Không có bằng chứng để lập chỉ mục; kiểm tra OCR/ASR.")
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Trùng chunk_id.")
    if seen_coarse != set(coarse):
        raise ValueError("Có fallback span chưa được đưa vào chỉ mục.")
    return chunks


def build_lexical_index(path, chunks):
    from contextlib import closing
    import sqlite3

    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DROP TABLE IF EXISTS evidence")
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE evidence USING fts5("
                "chunk_id UNINDEXED, text, tokenize='unicode61 remove_diacritics 2')"
            )
        except sqlite3.OperationalError as error:
            raise RuntimeError("Python/SQLite này không hỗ trợ FTS5.") from error
        connection.executemany(
            "INSERT INTO evidence(chunk_id, text) VALUES (?, ?)",
            [(chunk["chunk_id"], normalize_search(
                f"{chunk['title']}\n{chunk['text']}"
            )) for chunk in chunks if chunk.get("retrieval_policy") != "semantic_only"],
        )
        connection.commit()


def run_indexing(stage04_dir, output_root, video_id="L25_V001", *,
                 embedding_model="intfloat/multilingual-e5-small", revision="main",
                 chunk_tokens=350, overlap_tokens=40, batch_size=32, device=None,
                 speech_chunk_tokens=180, speech_maximum_seconds=50.0,
                 include_generated_descriptions=False, dedupe_previous=False, generated_semantic_only=False):
    import numpy as np
    # E5 was trained with "query: " / "passage: " prefixes; bge-m3 uses none.
    query_prefix, passage_prefix = ("query: ", "passage: ") if "e5" in embedding_model.lower() else ("", "")
    from sentence_transformers import SentenceTransformer

    parent = load_stage(stage04_dir, 4, video_id)
    stage04_dir = Path(stage04_dir)
    segments = read_json(stage04_dir / "segments.json")
    fallback_spans = read_json(stage04_dir / "fallback_spans.json")
    directory = begin_stage(output_root, 5)
    model = shared_model(("e5", embedding_model, revision, device),
                         lambda: SentenceTransformer(embedding_model, revision=revision, device=device))
    model_limit = min(int(model.max_seq_length), 512)
    chunks = make_evidence_chunks(
        segments, fallback_spans, model.tokenizer, chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens, model_limit=model_limit, video_id=video_id,
        speech_chunk_tokens=speech_chunk_tokens, speech_maximum_seconds=speech_maximum_seconds,
        include_generated_descriptions=include_generated_descriptions, dedupe_previous=dedupe_previous,
        passage_prefix=passage_prefix,
        generated_semantic_only=generated_semantic_only,
    )
    ad_review = []
    if any("spans_before_ad_filter" in s["speech"] for s in segments):
        before = [{**s, "speech": {**s["speech"], "spans": s["speech"].get("spans_before_ad_filter", s["speech"]["spans"])}}
                  for s in segments]
        original = make_evidence_chunks(
            before, fallback_spans, model.tokenizer, chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens, model_limit=model_limit, video_id=video_id,
            speech_chunk_tokens=speech_chunk_tokens, speech_maximum_seconds=speech_maximum_seconds,
            include_generated_descriptions=include_generated_descriptions, dedupe_previous=dedupe_previous,
            passage_prefix=passage_prefix, generated_semantic_only=generated_semantic_only)
        span_words = {span["span_id"]: set(span["word_ids"]) for s in before for span in s["speech"]["spans"]}
        ad_words = {word_id for s in segments for word_id in s["speech"].get("promotional_word_ids", [])}
        for chunk in original:
            if chunk["source_type"] != "speech":
                continue
            ids = set().union(*(span_words.get(sid, set()) for sid in chunk["source_ids"]))
            excluded = ids & ad_words
            if excluded:
                ad_review.append({**chunk, "review_status": "removed" if ids <= ad_words else "partially_filtered",
                                  "excluded_word_ids": sorted(excluded),
                                  "retained_word_ids": sorted(ids - ad_words)})
        write_json(directory / "ad_chunk_review.json", {
            "before_chunks": len(original), "after_chunks": len(chunks),
            "chunk_count_delta": len(chunks) - len(original),
            "removed_chunks": sum(c["review_status"] == "removed" for c in ad_review),
            "partially_filtered_chunks": sum(c["review_status"] == "partially_filtered" for c in ad_review),
            "chunks": ad_review})
    # Counts are checked above: no silent encoder truncation.
    vectors = model.encode(
        [chunk["embedding_text"] for chunk in chunks],
        batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    if (vectors.ndim != 2 or len(vectors) != len(chunks)
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)):
        raise ValueError("Embedding không hợp lệ hoặc chưa được chuẩn hóa.")
    np.save(directory / "embeddings.npy", vectors, allow_pickle=False)
    write_json(directory / "chunk_ids.json", [chunk["chunk_id"] for chunk in chunks])
    (directory / "chunks.jsonl").write_text(
        "\n".join(json.dumps(chunk, ensure_ascii=False, allow_nan=False)
                  for chunk in chunks) + "\n", encoding="utf-8",
    )
    build_lexical_index(directory / "lexical.sqlite3", chunks)
    # Notebook 06 needs only this stage's output: ship its evidence assets with it.
    copy_stage_assets(stage04_dir, directory / "lecture", parent)
    write_json(directory / "lecture/bundle_manifest.json", parent)
    auto_model = model._first_module().auto_model
    resolved_revision = getattr(auto_model.config, "_commit_hash", None)
    parameters = {
        "embedding_model": embedding_model, "revision": revision,
        "resolved_revision": resolved_revision,
        "query_prefix": query_prefix, "passage_prefix": passage_prefix,
        "normalize_embeddings": True, "model_limit": model_limit,
        "visual_description_policy": "structured_json_only" if include_generated_descriptions
                                     else "human_verified_only",
        "generated_semantic_only": generated_semantic_only,
        "speech_chunk_tokens": speech_chunk_tokens,
        "speech_maximum_seconds": speech_maximum_seconds,
        "chunk_tokens": chunk_tokens, "overlap_tokens": overlap_tokens,
        "dimensions": int(vectors.shape[1]),
        "packages": package_versions(["sentence-transformers", "transformers", "numpy"]),
    }
    manifest = finish_stage(
        directory, 5, video_id, parent["source"], parameters, parents=[parent],
        extra={
            "chunk_count": len(chunks), "index_status": "ready",
            "excluded_visual_description_segment_ids": [
                segment["segment_id"] for segment in segments
                if segment["visual"].get("description_status") != "json"
                or (not include_generated_descriptions and segment["visual"].get(
                    "description_verification") != "human_verified")
            ],
        },
    )
    return directory, manifest, chunks
