"""Pure, model-free citation mapping and Vietnamese presentation for the demo."""
import re
import unicodedata
from fractions import Fraction


SOURCE_NAMES = {'ocr': 'Chữ trên slide/OCR', 'speech': 'Lời giảng/ASR',
                'region': 'Hình/bảng/công thức'}
SUPPORT_WARNING = '⚠ có thể không liên quan / không thấy con số trong đoạn này'
_CITATION = re.compile(r'\[[A-Za-z]\d+(?:\s*[,;]\s*[A-Za-z]\d+)*\]|'
                       r'\[\d+:\d{2}(?:[–-]\d+:\d{2})?\]'
                       r'(?:\s*[–-]\s*\[\d+:\d{2}\])?')


def label_evidence(items):
    """Labels are stable in prompt order, separate for each source, per question."""
    counts = {}
    for rank, item in enumerate(items, 1):
        item.setdefault('source_type', 'speech')
        prefix = {'ocr': 'S', 'speech': 'L', 'region': 'H'}.get(item['source_type'], 'H')
        counts[prefix] = counts.get(prefix, 0) + 1
        item.update(evidence_label=f'{prefix}{counts[prefix]}', evidence_rank=rank)
    return items


def evidence_clock(seconds):
    seconds = int(seconds)
    return f'{seconds // 60:02d}:{seconds % 60:02d}'


def evidence_location(item):
    sid = item.get('segment_id', '').split(':')[-1]
    # Segment IDs use zero-based numbering; display the lecture's slide number.
    match = re.fullmatch(r'seg_(\d+)', sid)
    number = int(match[1]) + 1 if match else None
    kind = 'slide' if item['source_type'] != 'speech' else 'đoạn'
    return f'{kind} {number} ({sid})' if number is not None else f'{kind} {sid or "?"}'


def _words(text):
    folded = unicodedata.normalize('NFD', text.casefold().replace('đ', 'd'))
    folded = ''.join(c for c in folded if not unicodedata.combining(c))
    stop = {'la', 'va', 'cua', 'co', 'cac', 'mot', 'trong', 'duoc', 'cho', 'thi',
            'nay', 'do', 'voi', 'de', 'tu', 'the', 'a', 'of', 'to', 'is'}
    return set(re.findall(r'[^\W_]+', folded)) - stop


def evidence_excerpt(text, claim='', limit=420):
    """Prefer the sentence/window overlapping the cited claim, without rewriting."""
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= limit:
        return text
    sentences = list(re.finditer(r'[^.!?]+(?:[.!?]+|$)', text))
    words = _words(_CITATION.sub('', claim))
    best = max(sentences, key=lambda m: len(words & _words(m[0])), default=None)
    start, end = 0, min(limit - 1, len(text))
    if best and words & _words(best[0]):
        start, end = best.span()
        if end - start > limit - 2:
            # ASR often has no punctuation: center on an overlapping answer term.
            hits = [m for m in re.finditer(r'\w+', best[0]) if _words(m[0]) & words]
            anchor = max(hits, key=lambda m: len(m[0]), default=None)
            center = start + (anchor.start() if anchor else 0)
            start = max(start, center - (limit - 2) // 2)
            end = min(best.end(), start + limit - 2)
    if start:
        boundary = text.find(' ', start, min(start + 25, end))
        if boundary >= 0:
            start = boundary + 1
    if end < len(text) and end > start and text[end - 1].isalnum() and text[end].isalnum():
        boundary = text.rfind(' ', start, end)
        if boundary > start:
            end = boundary
    return ('…' if start else '') + text[start:end].strip() + ('…' if end < len(text) else '')


def _citation_groups(answer):
    groups = []
    for match in _CITATION.finditer(answer):
        if groups and not answer[groups[-1][-1].end():match.start()].strip(' \t\r\n,;()'):
            groups[-1].append(match)
        else:
            groups.append([match])
    previous = 0
    for group in groups:
        yield answer[previous:group[0].start()], group
        previous = group[-1].end()


def _fractions(text):
    """Compare explicit numeric results across LaTeX, plain text and Vietnamese ASR."""
    pairs = re.findall(r'\\(?:d|t)?frac\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}', text)
    pairs += re.findall(r'(?<!\d)(\d+)\s*(?:/|phần|phan)\s*(\d+)(?!\d)', text.casefold())
    return {Fraction(int(n), int(d)) for n, d in pairs if int(d)}


def _combinations(text):
    """Normalize the lecture's C^k_n and the model's C_n^k/LaTeX notation."""
    number = r'\s*\{?\s*(\d+)\s*\}?'
    pairs = re.findall(r'\bC\s*_' + number + r'\s*\^' + number, text)
    pairs += [(n, k) for k, n in re.findall(r'\bC\s*\^' + number + r'\s*_' + number, text)]
    pairs += re.findall(r'\\binom\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}', text)
    return {(int(n), int(k)) for n, k in pairs}


def _numeric_support_issue(claim, text):
    """Conservative explicit-number check, not a general semantic entailment test.

    Multiple-choice options are not evidence that an option is the final answer.
    Do not rewrite a claim or move its citation to a different source.
    """
    # Remove only an option's numeric value, preserving an inline solution after
    # it. Entire-line removal can falsely flag a correct answer in OCR text.
    value = (r'(?:\\(?:d|t)?frac\s*\{\s*\d+\s*\}\s*\{\s*\d+\s*\}'
             r'|\d+\s*(?:/|phần|phan)\s*\d+|\d+(?:[.,]\d+)?)')
    solution = re.sub(r'(?:^|(?<=\s))(?:\([A-D]\)|[A-D][.)])\s*[$\\()\s]*'
                      + value, '', text, flags=re.M | re.I)
    combinations = _combinations(claim)
    source_combinations = _combinations(solution)
    missing = combinations - source_combinations
    if source_combinations and missing:
        values = ', '.join(f'C^{k}_{n}' for n, k in sorted(missing))
        return 'Tổ hợp ' + values + ' không xuất hiện trong lời giải của đoạn được dẫn.'
    fractions = _fractions(_CITATION.sub('', claim))
    if fractions and not fractions.issubset(_fractions(solution)):
        return 'Kết quả phân số không xuất hiện trong lời giải của đoạn được dẫn.'
    years = set(re.findall(r'\b\d{4}\b', _CITATION.sub('', claim)))
    if not years.issubset(set(re.findall(r'\b\d{4}\b', solution))):
        return 'Con số trong câu trả lời không xuất hiện trong đoạn được dẫn.'
    return None


def numeric_support_issue(claim, text):
    """Optional display annotation. An unavailable check also stays an annotation."""
    try:
        issue = _numeric_support_issue(claim, text)
        if issue:
            return SUPPORT_WARNING + ': ' + issue
        if _words(_CITATION.sub('', claim)) and not _words(claim) & _words(text):
            return SUPPORT_WARNING
    except Exception:
        return SUPPORT_WARNING + ' (không kiểm tra được)'
    return None


def _evidence_attribution(answer, found):
    """Map labels exactly; time-only citations return ALL matching starts, never a guess."""
    items = label_evidence(found.get('context_evidence', []))
    by_label = {c['evidence_label']: c for c in items}
    refusal = 'không đề cập' in answer.casefold()
    matches, used = [], {}
    if not refusal:
        for claim, group in _citation_groups(answer):
            labels = [label.upper() for m in group for label in re.findall(r'[A-Za-z]\d+', m[0])]
            for match in group:
                citation = match[0]
                cited_labels = re.findall(r'[A-Za-z]\d+', citation)
                if cited_labels:
                    selected = [by_label[label.upper()] for label in cited_labels if label.upper() in by_label]
                    unknown = [label for label in cited_labels if label.upper() not in by_label]
                    method = 'label'
                else:
                    times = re.findall(r'(\d+):(\d{2})', citation)
                    minutes, seconds = times[0]
                    end_minutes, end_seconds = times[1] if len(times) == 2 else (None, None)
                    at = int(minutes) * 60 + int(seconds)
                    until = int(end_minutes) * 60 + int(end_seconds) if end_minutes is not None else None
                    valid_time = int(seconds) < 60 and (until is None or
                                 (int(end_seconds) < 60 and until >= at))
                    pool = [by_label[label] for label in labels if label in by_label] if labels else items
                    selected = [c for c in pool if valid_time and int(c['start_sec']) == at
                                and (until is None or until <= int(c['end_sec']))]
                    unknown = []
                    method = 'label_time' if labels else 'timestamp_fallback'
                issues = {}
                for c in selected:
                    try:
                        issues[c['evidence_label']] = numeric_support_issue(claim, c.get('text', ''))
                    except Exception:
                        issues[c['evidence_label']] = SUPPORT_WARNING + ' (không kiểm tra được)'
                unsupported = [label for label, issue in issues.items() if issue]
                record = {'citation': citation, 'claim': claim.strip(), 'method': method,
                          'evidence_labels': [c['evidence_label'] for c in selected],
                          'ambiguous': method == 'timestamp_fallback' and len(selected) > 1,
                          'unmatched': bool(unknown) or not selected,
                          'unknown_labels': unknown, 'unsupported_evidence_labels': unsupported}
                matches.append(record)
                for item in selected:
                    label = item['evidence_label']
                    entry = used.setdefault(label, {**item, 'citations': [], 'mapping_methods': [],
                                                     'claims': [], 'ambiguous': False, 'support_issues': []})
                    entry['citations'].append(citation)
                    entry['mapping_methods'].append(method)
                    entry['claims'].append(claim)
                    entry['ambiguous'] |= record['ambiguous']
                    if issues[label] and issues[label] not in entry['support_issues']:
                        entry['support_issues'].append(issues[label])
    for item in used.values():
        item['excerpt'] = evidence_excerpt(item.get('text', ''), ' '.join(item.pop('claims')))
    all_items = [{**c, 'cited': c['evidence_label'] in used,
                  'support_issues': used.get(c['evidence_label'], {}).get('support_issues', []),
                  'excerpt': evidence_excerpt(c.get('text', ''), limit=90)} for c in items]
    return {'refusal': refusal, 'used_evidence': list(used.values()), 'citation_matches': matches,
            'unmatched_citations': [m['citation'] for m in matches if m['unmatched']],
            'all_evidence': all_items}


def evidence_attribution(answer, found):
    """Attribution is display-only, including when a support check fails."""
    try:
        return _evidence_attribution(answer, found)
    except Exception:
        return {'refusal': 'không đề cập' in str(answer).casefold(), 'used_evidence': [],
                'citation_matches': [], 'unmatched_citations': [], 'all_evidence': [],
                'display_warning': SUPPORT_WARNING + ' (không dựng được bảng căn cứ)'}


def attribution_summary(report):
    items = report['used_evidence']
    if not items:
        return 'Dựa vào: không có căn cứ.'
    parts = []
    for kind, title in [('ocr', 'slide OCR'), ('speech', 'đoạn lời giảng'), ('region', 'hình/bảng/công thức')]:
        selected = [c for c in items if c['source_type'] == kind]
        if not selected:
            continue
        if kind == 'ocr':
            selected = list({(c.get('video_id'), c.get('segment_id')): c for c in selected}.values())
            places = [f"{evidence_location(c).split(' (')[0]}, {evidence_clock(c['start_sec'])}" for c in selected]
        else:
            places = [f"{evidence_clock(c['start_sec'])}–{evidence_clock(c['end_sec'])}" for c in selected]
        parts.append(f"{len(selected)} {title} ({'; '.join(places)})")
    suffix = ' [khớp mốc thời gian; chưa xác định chính xác nguồn]' if any(c['ambiguous'] for c in items) else ''
    return 'Dựa vào: ' + ' + '.join(parts) + suffix


def attribution_rows(report):
    """Every prompt item, including unranked supporting windows; no display cap."""
    rows = []
    for item in report['all_evidence']:
        score = lambda name, digits: (f'{item[name]:.{digits}f}' if item.get(name) is not None else '—')
        rows.append({'Hạng': item['evidence_rank'], 'Dẫn': '✓' if item['cited'] else '',
                     'Nhãn': f"[{item['evidence_label']}]", 'Loại': SOURCE_NAMES[item['source_type']],
                     'Vị trí': evidence_location(item),
                     'Thời gian': f"{evidence_clock(item['start_sec'])}–{evidence_clock(item['end_sec'])}",
                     'RRF / E5': score('rrf_score', 6) + ' / ' + score('semantic_score', 4),
                     'Trích ngắn': item['excerpt'],
                     'Lưu ý': '; '.join(item.get('support_issues', []))})
    return rows


def render_attribution(report, include_table=True):
    """Text counterpart of the notebook display, also usable on saved diagnostics."""
    lines = ['Căn cứ trả lời', attribution_summary(report)]
    if report.get('display_warning'):
        lines.append(report['display_warning'])
    for item in report['used_evidence']:
        citations = ', '.join(dict.fromkeys(item['citations']))
        lines.append(f"[{item['evidence_label']}] ← {citations} | {SOURCE_NAMES[item['source_type']]} | "
                     f"{item.get('video_id', '')} {evidence_location(item)} | "
                     f"{evidence_clock(item['start_sec'])}–{evidence_clock(item['end_sec'])}")
        if 'timestamp_fallback' in item['mapping_methods']:
            lines.append('  Khớp theo mốc thời gian' + ('; nhiều bằng chứng cùng mốc, chưa xác định chính xác nguồn.'
                                                      if item['ambiguous'] else ' (LLM không dẫn nhãn).'))
        lines.append('  Trích: ' + item['excerpt'])
        for issue in item.get('support_issues', []):
            lines.append('  Kiểm tra căn cứ: ' + issue)
        if item['source_type'] == 'ocr':
            lines.append('  Ảnh keyframe: ' + str(item.get('thumbnail_path') or 'không có ảnh cục bộ'))
    for citation in report['unmatched_citations']:
        lines.append(f'{citation}: không khớp bằng chứng.')
    if not report['refusal'] and not report['citation_matches']:
        lines.append('Không có trích dẫn trong câu trả lời.')
    if include_table:
        rows = attribution_rows(report)
        lines.append(f"Toàn bộ bằng chứng đã gửi LLM ({len(rows)} mục; hạng = thứ tự trong prompt; ✓ = được dẫn)")
        if rows:
            lines.append(' | '.join(rows[0]))
            lines.extend(' | '.join(str(v).replace('|', '∣') for v in row.values()) for row in rows)
    return '\n'.join(lines)
