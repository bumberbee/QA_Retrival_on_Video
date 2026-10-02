"""Lecture-scoped hybrid search and bounded, chronological evidence expansion."""


# Function words (accents removed) that match almost every chunk and only add BM25 noise.
STOPWORDS = set("""a ai anh ay ba bang bao bi boi cac cach cai can cho chi chua chung co con cua cung da dang
de den deu di do doi duoc gi gio ha hay hon khi khong la lai lam len luc ma minh mot moi muon nao nay
neu ngay nhieu nhu nhung noi o phai qua ra rang rat roi sao se sau tai the thi thoi theo thu tren tro
trong tu va van ve vi voi vua xin the what how why which when who is are the of a an to in on for and
or do does""".split())


def lexical_search(database, question, count=20):
    from contextlib import closing
    import sqlite3

    terms = [term for term in dict.fromkeys(re.findall(r"[^\W_]+", normalize_search(question)))
             if term not in STOPWORDS] or list(dict.fromkeys(re.findall(r"[^\W_]+", normalize_search(question))))
    if not terms:
        return []
    # Quote every token; user input never becomes FTS operators or SQL.
    expression = " OR ".join(f'"{term}"' for term in terms[:64])
    with closing(sqlite3.connect(
            Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        return connection.execute(
            "SELECT chunk_id, bm25(evidence) AS score FROM evidence "
            "WHERE evidence MATCH ? ORDER BY score LIMIT ?",
            (expression, count),
        ).fetchall()


def rank_candidates(chunks, vectors, query_vector, lexical_hits, count=20, rrf_k=60,
                    semantic_weight=1.0, lexical_weight=0.5):
    """Weighted reciprocal rank fusion: meaning (E5) counts more than keywords (BM25), which
    ASR spelling errors and common words make noisy."""
    import numpy as np

    similarities = vectors @ query_vector
    order = np.argsort(-similarities, kind="stable")[:min(count, len(chunks))]
    by_id = {chunk["chunk_id"]: index for index, chunk in enumerate(chunks)}
    scores, semantic_ranks, lexical_ranks = {}, {}, {}
    for rank, index in enumerate(order, 1):
        chunk_id = chunks[index]["chunk_id"]
        semantic_ranks[chunk_id] = rank
        scores[chunk_id] = scores.get(chunk_id, 0) + semantic_weight / (rrf_k + rank)
    for rank, (chunk_id, _) in enumerate(lexical_hits, 1):
        if chunk_id not in by_id:
            raise ValueError("Chỉ mục lexical chứa chunk_id không tồn tại.")
        lexical_ranks[chunk_id] = rank
        scores[chunk_id] = scores.get(chunk_id, 0) + lexical_weight / (rrf_k + rank)
    ranked = [{
        "chunk_id": chunk_id, "rrf_score": score,
        "semantic_score": float(similarities[by_id[chunk_id]]),
        "semantic_rank": semantic_ranks.get(chunk_id),
        "lexical_rank": lexical_ranks.get(chunk_id),
    } for chunk_id, score in scores.items()]
    return sorted(ranked, key=lambda item: (-item["rrf_score"], item["chunk_id"])), similarities


def temporal_question(question):
    normalized = normalize_search(question)
    return any(phrase in normalized for phrase in (
        "slide nay", "doan nay", "luc nay", "cho nay", "hien tai",
        "this slide", "this moment", "this part", "current slide",
    ))


def choose_anchors(chunks, ranked, segments, *, count=4, current_time=None,
                   prefer_current=False):
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    anchors = []
    if prefer_current and current_time is not None:
        segment = owner_interval(segments, current_time)
        nearby = [chunk for chunk in chunks if chunk["segment_id"] == segment["segment_id"]]
        if not nearby and segment.get("content_kind") == "ad":
            # Paused on an ad (not indexed): the lesson slide shown just before it, else after it.
            indexed = {chunk["segment_id"] for chunk in chunks}
            lesson = [item for item in segments if item["segment_id"] in indexed]
            before = [item for item in lesson if item["end_sec"] <= segment["start_sec"] + 1e-6]
            after = [item for item in lesson if item["start_sec"] >= segment["end_sec"] - 1e-6]
            if before or after:
                segment = before[-1] if before else after[0]
                nearby = [chunk for chunk in chunks if chunk["segment_id"] == segment["segment_id"]]
        if not nearby:
            raise ValueError("Không có bằng chứng cho đoạn đang xem tại CURRENT_TIME_SEC.")
        ranks = {hit["chunk_id"]: rank for rank, hit in enumerate(ranked)}
        nearby.sort(key=lambda chunk: (
            max(chunk["start_sec"] - current_time, current_time - chunk["end_sec"], 0),
            0 if chunk["source_type"] == "speech" else 1,
            ranks.get(chunk["chunk_id"], len(ranked)), chunk["chunk_id"],
        ))
        # Deictic questions are about this interval; neighbors are added later.
        # Do not fill remaining anchor slots with unrelated global hits.
        return [chunk["chunk_id"] for chunk in nearby[:count]]
    # Several hits within one long interval may still answer different aspects.
    counts = {}
    for anchor in anchors:
        parent = by_id[anchor]["segment_id"]
        counts[parent] = counts.get(parent, 0) + 1
    for hit in ranked:
        chunk = by_id[hit["chunk_id"]]
        if hit["chunk_id"] in anchors or counts.get(chunk["segment_id"], 0) >= 2:
            continue
        if len(anchors) >= count:
            break
        anchors.append(hit["chunk_id"])
        counts[chunk["segment_id"]] = counts.get(chunk["segment_id"], 0) + 1
    return anchors


def expand_context_candidates(chunks, anchors, segments, similarities, neighbor_count=1):
    if neighbor_count not in {0, 1, 2}:
        raise ValueError("neighbor_count chỉ nhận 0, 1 hoặc 2 để giữ ngữ cảnh có giới hạn.")
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    by_segment = {}
    score = {}
    for chunk, similarity in zip(chunks, similarities):
        by_segment.setdefault(chunk["segment_id"], []).append(chunk)
        score[chunk["chunk_id"]] = float(similarity)
    positions = {segment["segment_id"]: index for index, segment in enumerate(segments)}
    output = [(chunk_id, "anchor") for chunk_id in anchors]
    for anchor_id in anchors:
        anchor = by_id[anchor_id]
        siblings = by_segment[anchor["segment_id"]]
        speech = sorted((chunk for chunk in siblings if chunk["source_type"] == "speech"),
                        key=lambda chunk: -score[chunk["chunk_id"]])
        visual = sorted((chunk for chunk in siblings if chunk["source_type"] != "speech"),
                        key=lambda chunk: -score[chunk["chunk_id"]])
        output += [(chunk["chunk_id"], "same_segment") for chunk in speech[:2] + visual[:3]]
    for anchor_id in anchors:
        position = positions[by_id[anchor_id]["segment_id"]]
        for delta in range(1, neighbor_count + 1):
            for index in (position - delta, position + delta):
                if not 0 <= index < len(segments):
                    continue
                neighboring = by_segment.get(segments[index]["segment_id"], [])
                speech = sorted(
                    (chunk for chunk in neighboring if chunk["source_type"] == "speech"),
                    key=lambda chunk: chunk["start_sec"], reverse=index < position,
                )
                visual = sorted(
                    (chunk for chunk in neighboring if chunk["source_type"] != "speech"),
                    key=lambda chunk: -score[chunk["chunk_id"]],
                )
                # Include only boundary narration and one visual chunk, not whole neighbors.
                output += [(chunk["chunk_id"], "previous" if index < position else "next")
                           for chunk in speech[:2] + visual[:1]]
    seen, deduplicated = set(), []
    for chunk_id, reason in output:
        if chunk_id not in seen:
            deduplicated.append((chunk_id, reason))
            seen.add(chunk_id)
    return deduplicated


def merge_overlap_text(left, right):
    left_words, right_words = left.split(), right.split()
    for count in range(min(120, len(left_words), len(right_words)), 0, -1):
        if left_words[-count:] == right_words[:count]:
            return " ".join(left_words + right_words[count:])
    return left + "\n" + right


def evidence_blocks(selected):
    ordered = sorted(selected, key=lambda chunk: (
        chunk["start_sec"], chunk["sequence"], chunk["source_type"], chunk["chunk_id"],
    ))
    blocks = []
    for chunk in ordered:
        unit, part = chunk["chunk_id"].rsplit(":p", 1)
        if (blocks and blocks[-1]["_unit"] == unit
                and blocks[-1]["_last_part"] + 1 == int(part)):
            block = blocks[-1]
            block["text"] = merge_overlap_text(block["text"], chunk["text"])
            block["chunk_ids"].append(chunk["chunk_id"])
            block["_last_part"] = int(part)
        else:
            blocks.append({
                "_unit": unit, "_last_part": int(part),
                "segment_id": chunk["segment_id"], "source_type": chunk["source_type"],
                "start_sec": chunk["start_sec"], "end_sec": chunk["end_sec"],
                "title": chunk["title"], "text": chunk["text"],
                "chunk_ids": [chunk["chunk_id"]], "image_path": chunk["image_path"],
                "keyframe_time_sec": chunk["keyframe_time_sec"],
                "timing_precision": chunk["timing_precision"],
                "selection_reason": chunk["selection_reason"],
                "uncertainty": chunk.get("uncertainty"),
            })
    for index, block in enumerate(blocks, 1):
        block["evidence_id"] = f"E{index}"
        block.pop("_unit")
        block.pop("_last_part")
    return blocks


def render_context(blocks):
    return "\n\n".join(
        f"[{block['evidence_id']}] {block['source_type']} | {block['segment_id']} | "
        f"{block['start_sec']:.3f}–{block['end_sec']:.3f}s\n"
        f"{block['title']}\n{block['text']}"
        + (f"\nGhi chú: {block['uncertainty']}" if block.get("uncertainty") else "")
        for block in blocks
    )


def budget_context(chunks, candidates, tokenizer, maximum_tokens=6000):
    if maximum_tokens < 64:
        raise ValueError("Context budget quá nhỏ.")
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    selected, seen_text = [], set()
    for chunk_id, reason in candidates:
        chunk = by_id[chunk_id]
        identity = (chunk["segment_id"], normalize_search(chunk["text"]).strip())
        if identity in seen_text:
            continue
        proposed = selected + [{**chunk, "selection_reason": reason}]
        context = render_context(evidence_blocks(proposed))
        if len(tokenizer.encode(context, add_special_tokens=False)) <= maximum_tokens:
            selected = proposed
            seen_text.add(identity)
    blocks = evidence_blocks(selected)
    if not any(chunk["selection_reason"] == "anchor" for chunk in selected):
        raise ValueError("Không đủ ngân sách cho anchor; tăng MAX_CONTEXT_TOKENS.")
    context = render_context(blocks)
    return blocks, context, len(tokenizer.encode(context, add_special_tokens=False))


def load_retrieval_engine(stage05_dir, video_id="L25_V001", device=None):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    directory = Path(stage05_dir)
    manifest = load_stage(directory, 5, video_id)
    chunks = [json.loads(line) for line in
              (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
    chunk_ids = read_json(directory / "chunk_ids.json")
    parameters = manifest["parameters"]
    if (chunk_ids != [chunk["chunk_id"] for chunk in chunks]
            or vectors.shape != (len(chunks), parameters["dimensions"])
            or not np.isfinite(vectors).all()
            or any(chunk["video_id"] != video_id for chunk in chunks)):
        raise ValueError("Chunk IDs, video scope hoặc vector index không khớp.")
    bundle = read_json(directory / "lecture/bundle_manifest.json")
    require_parent(manifest, bundle)
    segments = read_json(directory / "lecture/segments.json")
    revision = parameters["resolved_revision"] or parameters["revision"]
    model = shared_model(("e5", parameters["embedding_model"], revision, device),
                         lambda: SentenceTransformer(parameters["embedding_model"],
                                                     revision=revision, device=device))
    if model.get_sentence_embedding_dimension() != parameters["dimensions"]:
        raise ValueError("Query encoder không khớp chiều embedding.")
    return {
        "directory": directory, "manifest": manifest, "chunks": chunks,
        "vectors": vectors, "segments": segments, "model": model,
    }


def retrieve_question(engine, question, *, current_time=None, prefer_current=None,
                      candidate_count=20, anchor_count=4, neighbors=1,
                      maximum_tokens=6000, semantic_weight=1.0, lexical_weight=0.5, reranker=None):
    """reranker: None, or score(question, [texts]) -> [float] (a cross-encoder) that reorders
    the fused candidates; its best score is returned to help decide "not in the lecture"."""
    question = question.strip()
    if not question or candidate_count < 1 or anchor_count < 1:
        raise ValueError("Câu hỏi không được rỗng; số ứng viên/anchor phải >= 1.")
    model, manifest = engine["model"], engine["manifest"]
    if current_time is not None and not (
            math.isfinite(current_time) and 0 <= current_time <= manifest["source"]["duration_sec"]):
        raise ValueError("CURRENT_TIME_SEC nằm ngoài video.")
    query = manifest["parameters"]["query_prefix"] + question
    if len(model.tokenizer.encode(query, add_special_tokens=True)) > \
            manifest["parameters"]["model_limit"]:
        raise ValueError("Câu hỏi vượt giới hạn encoder; hãy rút ngắn câu hỏi.")
    vector = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    lexical = lexical_search(engine["directory"] / "lexical.sqlite3", question, candidate_count)
    ranked, similarities = rank_candidates(
        engine["chunks"], engine["vectors"], vector, lexical, candidate_count,
        semantic_weight=semantic_weight, lexical_weight=lexical_weight,
    )
    rerank_best = None
    if reranker is not None and ranked:
        by_id = {chunk["chunk_id"]: chunk for chunk in engine["chunks"]}
        top = ranked[:candidate_count]
        scores = reranker(question, [by_id[hit["chunk_id"]]["text"] for hit in top])
        for hit, score in zip(top, scores):
            hit["rerank_score"] = float(score)
        ranked = sorted(top, key=lambda hit: -hit["rerank_score"]) + ranked[candidate_count:]
        rerank_best = max(scores)
    temporal = temporal_question(question) if prefer_current is None else prefer_current
    if temporal and current_time is None:
        raise ValueError("Câu hỏi chỉ 'đoạn/slide này': hãy điền CURRENT_TIME_SEC.")
    anchors = choose_anchors(
        engine["chunks"], ranked, engine["segments"], count=anchor_count,
        current_time=current_time, prefer_current=temporal,
    )
    candidates = expand_context_candidates(
        engine["chunks"], anchors, engine["segments"], similarities, neighbors,
    )
    blocks, context, token_count = budget_context(
        engine["chunks"], candidates, model.tokenizer, maximum_tokens,
    )
    selected_ids = {chunk_id for block in blocks for chunk_id in block["chunk_ids"]}
    kept_anchors = [chunk_id for chunk_id in anchors if chunk_id in selected_ids]
    return {
        "question": question, "video_id": manifest["video_id"], "current_time_sec": current_time,
        "source_sha256": manifest["source"]["sha256"],
        "anchor_chunk_ids": kept_anchors,
        "omitted_anchor_chunk_ids": [chunk_id for chunk_id in anchors if chunk_id not in selected_ids],
        "hits": ranked,
        "evidence": blocks, "context": context, "context_tokens": token_count,
        "best_semantic_score": float(max(similarities)) if len(similarities) else None,
        "best_rerank_score": rerank_best,
        "parameters": {
            "candidate_count": candidate_count, "anchor_count": anchor_count,
            "neighbors": neighbors, "maximum_tokens": maximum_tokens,
            "prefer_current": temporal, "fusion": "weighted_reciprocal_rank_fusion",
            "semantic_weight": semantic_weight, "lexical_weight": lexical_weight,
            "reranker": reranker is not None,
        },
        "note": "Kết quả truy xuất, chưa phải câu trả lời của LLM. "
                "Điểm tương đồng không phải xác suất đúng; cần đánh giá trên câu hỏi có nhãn.",
    }


def export_retrieval(engine, result, output_root, maximum_images=3):
    import base64
    import html

    if maximum_images < 0:
        raise ValueError("maximum_images phải >= 0.")
    directory = begin_stage(output_root, 6)
    result = json.loads(json.dumps(result, ensure_ascii=False))
    images = {}
    by_chunk = {chunk["chunk_id"]: chunk for chunk in engine["chunks"]}
    requested = [by_chunk[chunk_id]["image_path"] for chunk_id in result["anchor_chunk_ids"]]
    requested += [block["image_path"] for block in result["evidence"]]
    for name in requested:
        if name in images or len(images) >= maximum_images:
            continue
        source = relative_asset(engine["directory"] / "lecture", name)
        target_name = f"images/{file_sha256(source)[:20]}{source.suffix}"
        target = relative_asset(directory, target_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        images[name] = target_name
    for block in result["evidence"]:
        block["exported_image_path"] = images.get(block["image_path"])
    result["images"] = [{"source_path": source, "path": target} for source, target in images.items()]
    write_json(directory / "retrieval.json", result)
    (directory / "context.txt").write_text(result["context"], encoding="utf-8")
    cards = []
    for block in result["evidence"]:
        image_html = ""
        if block["exported_image_path"]:
            data = base64.b64encode(
                relative_asset(directory, block["exported_image_path"]).read_bytes()
            ).decode("ascii")
            image_html = f'<img src="data:image/jpeg;base64,{data}" alt="Khung hình nguồn">'
        cards.append(
            f"<article><h2>{html.escape(block['evidence_id'])} · "
            f"{timestamp_label(block['start_sec'])}–{timestamp_label(block['end_sec'])}"
            f" · {html.escape(block['source_type'])}</h2>{image_html}"
            f"<p>{html.escape(block['text'])}</p></article>"
        )
    preview = (
        "<!doctype html><html lang='vi'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Kết quả truy xuất L25_V001</title><style>"
        "body{font:16px/1.65 system-ui;background:#f7f4ee;color:#29251f;"
        "max-width:1000px;margin:auto;padding:24px}article{background:white;"
        "padding:20px;margin:18px 0;border:1px solid #ded4c1;border-radius:12px}"
        "h2{font-size:17px}img{max-width:100%;max-height:400px}p{white-space:pre-wrap}"
        "</style><body><h1>" + html.escape(result["question"]) + "</h1><p>"
        + html.escape(result["note"]) + "</p>" + "".join(cards) + "</body></html>"
    )
    (directory / "preview.html").write_text(preview, encoding="utf-8")
    manifest = finish_stage(
        directory, 6, result["video_id"], engine["manifest"]["source"],
        {**result["parameters"], "question": result["question"],
         "current_time_sec": result["current_time_sec"], "maximum_images": maximum_images},
        parents=[engine["manifest"]], extra={"evidence_count": len(result["evidence"])},
    )
    return directory, manifest, result
