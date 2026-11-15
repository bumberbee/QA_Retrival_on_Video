"""Verified evidence packing with source-aligned display metadata."""

_app_build_context = build_context


def build_context(found, max_chars=16000):
    """Keep every selected slide's match, OCR and speech before optional windows.

    Never drop a match just because it exceeds a per-source character allowance.
    Record exactly which complete chunks fit, including explicit budget omissions.
    """
    if max_chars < 1:
        raise ValueError("Context budget must be positive.")
    candidates = []
    ranks = {c['chunk_id']: c['rrf_rank'] for c in found.get('candidate_ranks', [])}
    for rank, slide in enumerate(found['slides']):
        for chunk in slide.get('evidence_candidates', []):
            reason = chunk['selection_reason']
            tier = (0 if chunk.get('priority') == 0 else 1 if reason == 'same_slide'
                    else 2 if reason == 'matched' else 3)
            candidates.append({**chunk, 'video_id': slide.get('video_id', ''),
                               'video_title': slide.get('video_title', ''), 'slide_rank': rank,
                               'packing_tier': tier,
                               'thumbnail_path': chunk.get('thumbnail_path') or
                               (slide.get('image') if chunk.get('segment_id') == slide.get('segment_id') else None)})
    candidates.sort(key=lambda c: (c['packing_tier'],
                                   ranks.get(c['chunk_id'], c['slide_rank'])
                                   if c['packing_tier'] == 2 else c['slide_rank']))
    selected, seen, omitted, blocks = [], set(), [], []
    used = 0
    several = len({s.get('video_id') for s in found['slides']}) > 1
    for chunk in candidates:
        identity = (chunk['video_id'], chunk['source_type'], normalize_search(chunk['text']).strip())
        if identity in seen:
            continue
        block = (f"Mốc trích dẫn: [{clock(chunk['start_sec'])}]\n"
                 + evidence_block(chunk, several))
        extra = len(block) + (2 if blocks else 0)
        if used + extra > max_chars:
            if chunk['packing_tier'] == 0:
                raise ValueError('Context budget cannot fit all selected slide matches; increase it.')
            omitted.append({'chunk_id': chunk['chunk_id'], 'reason': 'context_budget'})
            continue
        selected.append(chunk)
        blocks.append(block)
        seen.add(identity)
        used += extra
    context = '\n\n'.join(blocks)
    label_evidence(selected)  # Display metadata only; the model sees the verified text.
    found.update(context_evidence=selected, context_chars=len(context), context_omitted=omitted,
                 context_selection_chars=used, context_content_chars=len(context), context_label_chars=0)
    return context
