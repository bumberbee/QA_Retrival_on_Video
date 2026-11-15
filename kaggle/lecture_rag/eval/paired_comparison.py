"""Compare isolated CPU retrieval reports; no models or external services."""
import argparse
import json
from pathlib import Path


def paired(medium, turbo):
    def keyed(rows):
        result = {r['id']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate question IDs')
        return result
    a, b = keyed(medium), keyed(turbo)
    if a.keys() != b.keys():
        raise ValueError('Paired question scopes differ')
    answerable = lambda r: r.get('answerable', r.get('type') != 'unanswerable')
    if any(answerable(a[k]) != answerable(b[k]) or a[k]['video'] != b[k]['video'] for k in a):
        raise ValueError('Paired question labels differ')
    ids = [k for k in a if answerable(a[k])]
    metrics = {}
    for metric in ('hit1', 'hit3', 'hit5', 'mrr', 'context_recall', 'retained_time_overlap'):
        if not ids or metric not in a[ids[0]]:
            continue
        delta = {k: b[k][metric] - a[k][metric] for k in ids}
        metrics[metric] = {'medium': sum(a[k][metric] for k in ids) / len(ids),
                           'turbo': sum(b[k][metric] for k in ids) / len(ids),
                           'turbo_minus_medium': sum(delta.values()) / len(ids),
                           'turbo_gains': [k for k, v in delta.items() if v > 1e-12],
                           'turbo_losses': [k for k, v in delta.items() if v < -1e-12]}
    return {'questions': len(a), 'answerable': len(ids), 'metrics': metrics,
            'context_chars': {name: sum(len(rows[k].get('context', '')) for k in ids) / len(ids)
                              if ids else None for name, rows in (('medium', a), ('turbo', b))},
            'unanswerable': [{'id': k, **{name: {'semantic': rows[k]['best_semantic'],
                'rrf': rows[k]['top_rrf'], 'returned_slides': rows[k].get('returned_slides',
                    len(rows[k].get('slides', []))), 'context_chars': len(rows[k].get('context', ''))}
                for name, rows in (('medium', a), ('turbo', b))}} for k in a if k not in ids]}


def grouped(medium, turbo, field):
    labels = sorted({r[field] for r in medium})
    return {label: paired([r for r in medium if r[field] == label],
                           [r for r in turbo if r[field] == label]) for label in labels}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def against_reference(reference, current):
    comparison = paired(reference, current)
    return {'questions': comparison['questions'], 'answerable': comparison['answerable'],
            'metrics': {key: {'reference': value['medium'], 'new': value['turbo'],
                'new_minus_reference': value['turbo_minus_medium'],
                'gains': value['turbo_gains'], 'losses': value['turbo_losses']}
                for key, value in comparison['metrics'].items()}}


def audited_windows(root):
    """Apply saved human audit notes to the same source hashes and time windows."""
    audits = read(Path(__file__).resolve().parent / 'phase3_gpu_review.json')
    seen, cases = set(), []
    for audit in audits:
        key = (audit['video_id'], audit['start_sec'], audit['end_sec'], audit['source'], audit['note'])
        if key in seen:
            continue
        seen.add(key)
        video, start, end, _, _ = key
        case = dict(audit, variants={})
        for variant in ('medium', 'turbo'):
            base = root / variant / video
            manifest = read(base / '_inputs/03_audio/manifest.json')
            raw = read(base / '_inputs/03_audio/transcript.json')['words']
            cleaned = read(base / '04_bundle/words.json')
            selected = lambda words: [w for w in words if w['start_sec'] < end and w['end_sec'] > start]
            chunks = [json.loads(line) for line in (base / '05_index/chunks.jsonl').read_text(encoding='utf8').splitlines() if line]
            case['variants'][variant] = {'source_sha256': manifest['source']['sha256'],
                'raw_asr': ' '.join(w['text'] for w in selected(raw)),
                'cleaned_asr': ' '.join(w['text'] for w in selected(cleaned)),
                'index_evidence': [c for c in chunks if c['start_sec'] < end and c['end_sec'] > start]}
        if case['variants']['medium']['source_sha256'] != case['variants']['turbo']['source_sha256']:
            raise ValueError('Audited source hashes differ')
        cases.append(case)
    value = {'scope': 'Existing reviewed notes applied to identical source videos; no fresh listening claimed',
             'cases': cases}
    (root / 'audited_terms.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return len(cases)


def compare(root, reference):
    medium, turbo = (read(root / (name + '-dev') / 'retrieval.json') for name in ('medium', 'turbo'))
    if medium['questions_sha256'] != turbo['questions_sha256']:
        raise ValueError('Dev question hashes differ')
    old = read(reference)
    report = {'direction': 'Positive differences favor turbo', 'dev': paired(medium['questions'], turbo['questions']),
              'per_video': grouped(medium['questions'], turbo['questions'], 'video'),
              'per_type': grouped(medium['questions'], turbo['questions'], 'type'),
              'legacy': {}, 'old_reference': old['overall'],
              'against_old_reference': {name: against_reference(old['questions'], value['questions'])
                                       for name, value in (('medium', medium), ('turbo', turbo))}}
    if report['dev']['questions'] != 120 or report['dev']['answerable'] != 110:
        raise ValueError('Expected 120 dev questions / 110 answerable')
    ml, tl = (read(root / (name + '-legacy') / 'legacy.json') for name in ('medium', 'turbo'))
    for name in ml['ablations']:
        a, b = ml['ablations'][name]['questions'], tl['ablations'][name]['questions']
        report['legacy'][name] = {'overall': paired(a, b), 'per_video': grouped(a, b, 'video')}
        if len(a) != 49 or report['legacy'][name]['overall']['answerable'] != 44:
            raise ValueError('Expected 49 legacy questions / 44 answerable')
    cases = {
        'Bayes_probability': [r['id'] for r in medium['questions'] if r['video'] == 'hVU5Mw19V_c' and r['type'] != 'unanswerable'],
        'PPF': ['v2YTz8ToEag_q08', 'v2YTz8ToEag_q09', 'v2YTz8ToEag_q10'],
        'Python_len_print_end': [r['id'] for r in medium['questions'] if r['video'] == 'y1CvB950VlU' and r['type'] != 'unanswerable'],
        'L25_history_repetition': [r['id'] for r in medium['questions'] if r['video'] in ('L25_V011', 'L25_V037') and r['type'] != 'unanswerable'],
        'English_lecture': [r['id'] for r in medium['questions'] if r['video'] == 'cs_0001' and r['type'] != 'unanswerable']}
    report['term_cases'] = {name: paired([r for r in medium['questions'] if r['id'] in ids],
                                        [r for r in turbo['questions'] if r['id'] in ids]) for name, ids in cases.items()}
    case_rows = {name: {'medium': [r for r in medium['questions'] if r['id'] in ids],
                        'turbo': [r for r in turbo['questions'] if r['id'] in ids]} for name, ids in cases.items()}
    (root / 'paired.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (root / 'term_cases.json').write_text(json.dumps(case_rows, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    report['saved_audit_windows'] = audited_windows(root)
    (root / 'paired.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'dev': {k: {n: v[n] for n in ('medium', 'turbo', 'turbo_minus_medium')}
                             for k, v in report['dev']['metrics'].items()}}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[1] /
                        'runs/exp/phase3-a-reference-eval/retrieval.json')
    args = parser.parse_args()
    compare(args.root, args.reference)
