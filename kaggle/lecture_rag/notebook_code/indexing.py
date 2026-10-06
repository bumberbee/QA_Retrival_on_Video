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
                         dedupe_previous=False):
    """dedupe_previous: index only the OCR lines not already on the previous slide (stage 04
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
        # Bound the title separately so it cannot consume the encoder budget.
        title_tokens = tokenizer.encode(visual.get("title_clean", visual.get("title", "")),
                                        add_special_tokens=False)[:64]
        title = tokenizer.decode(title_tokens, skip_special_tokens=True).strip()
        header = f"passage: {title}\n" if title else "passage: "
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
            # Older stage-02 bundles may store malformed JSON or degenerate
            # generation (e.g. repeated punctuation) as visual_description.
            # Preserve those raw records in lecture/, but never index them.
            if source_type == "visual" and visual.get("description_status") != "json":
                continue
            if (source_type == "visual" and not include_generated_descriptions
                    and visual.get("description_verification") != "human_verified"):
                continue
            if text.strip():
                units.append({
                    "source_type": source_type, "text": text,
                    "start_sec": segment["start_sec"], "end_sec": segment["end_sec"],
                    "source_ids": [visual["occurrence_id"]],
                    "timing_precision": "visual_interval",
                    "uncertainty": "OCR/mô tả tự động; đối chiếu hình gốc khi cần.",
                })
        for region in visual.get("regions", []):
            if region.get("noise"):
                continue
            if (not include_generated_descriptions and region.get("extraction_status")
                    not in {"human_verified", "hybrid_ocr_unverified"}):
                continue
            text = "\n".join(value for value in
                             [region.get("text"), region.get("description")] if value)
            if text.strip():
                units.append({
                    "source_type": "region", "text": text,
                    "start_sec": segment["start_sec"], "end_sec": segment["end_sec"],
                    "source_ids": [region["region_id"]],
                    "image_path": region["image_path"],
                    "timing_precision": "visual_interval",
                    "uncertainty": "Vùng và nội dung do mô hình đề xuất, chưa xác minh.",
                })
        for unit_index, unit in enumerate(units):
            parts = split_text_tokens(unit["text"], tokenizer, budget,
                                      min(overlap_tokens, budget - 1))
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
            )) for chunk in chunks],
        )
        connection.commit()


def run_indexing(stage04_dir, output_root, video_id="L25_V001", *,
                 embedding_model="intfloat/multilingual-e5-small", revision="main",
                 chunk_tokens=350, overlap_tokens=40, batch_size=32, device=None,
                 speech_chunk_tokens=180, speech_maximum_seconds=50.0,
                 include_generated_descriptions=False, dedupe_previous=False):
    import numpy as np
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
    )
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
        "query_prefix": "query: ", "passage_prefix": "passage: ",
        "normalize_embeddings": True, "model_limit": model_limit,
        "visual_description_policy": "structured_json_only" if include_generated_descriptions
                                     else "human_verified_only",
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
