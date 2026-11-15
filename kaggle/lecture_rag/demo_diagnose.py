"""Replay saved-index retrieval with Kaggle query vectors; no local model loading."""
import argparse
import io
import json
from pathlib import Path
import re
import sqlite3
import sys
import types
import zipfile

import numpy as np
import build_demo_pipeline as demo
import production_run as production


def module(name, source):
    result = types.ModuleType(name)
    result.DEMO_STORAGE = types.SimpleNamespace(cloud=lambda: False)
    sys.modules[name] = result
    exec(compile(source, name, 'exec'), result.__dict__)
    return result


def replay(verification, archive, target):
    data = json.loads(Path(verification).read_text(encoding='utf-8'))
    queries = data['results']['retrieval_diagnostics']
    with zipfile.ZipFile(archive) as saved:
        prefix = 'L25_V037/05_index/'
        chunks = [json.loads(line) for line in saved.read(prefix + 'chunks.jsonl').decode().splitlines()]
        vectors = np.load(io.BytesIO(saved.read(prefix + 'embeddings.npy')), allow_pickle=False)
        segments = json.loads(saved.read(prefix + 'lecture/segments.json'))
        manifest = json.loads(saved.read(prefix + 'manifest.json'))
        lexical_bytes = saved.read(prefix + 'lexical.sqlite3')
    search_source, answer_source = demo.backend_sources()
    fixed = module('diagnostic_demo_search', search_source)
    fixed.MAX_SLIDES = 8  # Replay the enhanced mode verified on Kaggle, independent of UI defaults.
    baseline = module('diagnostic_baseline_search', (demo.BACKEND / 'retrieval.py').read_text(
        encoding='utf-8').replace('from .. import storage', 'storage = DEMO_STORAGE'))
    answer = module('diagnostic_demo_answer', answer_source)
    model = manifest.get('parameters', {}).get('embedding_model', baseline.E5_REPO)
    index = types.SimpleNamespace(folder=None, chunks=chunks, vectors=vectors, segments=segments,
                                  by_segment={s['segment_id']: s for s in segments}, model=model)
    db = sqlite3.connect(':memory:')
    db.deserialize(lexical_bytes)
    def lexical_hits(folder, question, count, video_id='', options=None):
        opts = options or fixed.RetrievalOptions()
        terms, phrases = fixed.lexical_query(question, opts.vietnamese_lexical, opts.phrases)
        if not terms:
            return []
        expression = ' OR '.join(f'"{term}"' for term in terms)
        if phrases:
            expression += ' OR ' + ' OR '.join(f'"{a} {b}"' for a, b in phrases)
        return [r[0] for r in db.execute(
            'SELECT chunk_id FROM evidence WHERE evidence MATCH ? ORDER BY bm25(evidence) LIMIT ?',
            (expression, count))]
    for search in (baseline, fixed):
        search.load_index = lambda _: index
        search.lexical_hits = lexical_hits
    report = {'index': str(archive), 'query_vectors_from': str(verification),
              'ranking': {'candidates_per_channel': 20, 'semantic_weight': 1, 'lexical_weight': .5,
                          'rrf_k': 60}, 'questions': []}
    for number, query in enumerate(queries, 1):
        if query.get('encoder_model') != model or not query.get('query_vector'):
            raise ValueError('Missing query vector or encoder/index mismatch; no local model fallback.')
        vector = np.asarray(query['query_vector'], dtype=np.float32)
        if vector.shape != (vectors.shape[1],):
            raise ValueError('Query vector dimensions differ from the saved index.')
        encoder = types.SimpleNamespace(encode=lambda _: vector)
        question = query['question']
        old = baseline.search('L25_V037', question, encoder=encoder)
        new = fixed.search('L25_V037', question, encoder=encoder)
        old_context = baseline.build_context(old)
        new_context = fixed.build_context(new)
        gold = [c for c in chunks if c['source_type'] == 'ocr' and c['segment_id'].endswith('00002')]
        if number == 2:
            gold = [c for c in chunks if c['source_type'] == 'speech' and c['start_sec'] >= 810
                    and re.search(r'tháng\s+3\s+năm\s+1945', c['text'], re.I)]
        elif number == 3:
            gold = []
        ranks = {c['chunk_id']: c for c in new['candidate_ranks']}
        raw_semantic = {chunks[i]['chunk_id']: rank for rank, i in enumerate(
            np.argsort(-(vectors @ vector), kind='stable'), 1)}
        raw_lexical = {cid: rank for rank, cid in enumerate(
            lexical_hits(None, question, len(chunks)), 1)}
        row = {'question': question, 'fair': bool(gold), 'gold': [
            {**c, 'ranks': ranks.get(c['chunk_id']),
             'raw_semantic_rank': raw_semantic.get(c['chunk_id']),
             'raw_lexical_rank': raw_lexical.get(c['chunk_id']),
             'legacy_sent': c['text'] in old_context, 'demo_sent': c['text'] in new_context}
            for c in gold], 'candidate_ranks': new['candidate_ranks'],
            'legacy_slides': [s['segment_id'] for s in old['slides']],
            'legacy_evidence': old.get('context_evidence', []),
            'legacy_prompt': answer.legacy_evidence_prompt(question, None, old, old_context),
            'demo_slides': [s['segment_id'] for s in new['slides']],
            'demo_evidence': new.get('context_evidence', []),
            'demo_prompt': answer._evidence_prompt(question, None, new, new_context),
            'verified_run': data['ref'], 'verified_prompt': query['prompt'],
            'verified_candidate_ranks': query['candidate_ranks'],
            'verified_evidence': query['context_evidence'],
            'verified_answer': data['results']['queries'][number - 1]}
        report['questions'].append(row)
        print(f'Q{number}: legacy slides ' + ', '.join(s.split(':')[-1] for s in row['legacy_slides']))
        print('  Gold:', [(c['chunk_id'], c['ranks'], c['legacy_sent'], c['demo_sent']) for c in row['gold']])
        print('  Context chars:', len(old_context), '->', len(new_context),
              '| candidates:', len(new['candidate_ranks']))
    db.close()
    production.write(target, report)
    print('Saved diagnosis:', target)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--archive', type=Path,
                        default=Path('D:/thesis-data/run-20261102/confirm/fast_pipeline_export.zip'))
    parser.add_argument('--output', type=Path, default=Path('D:/thesis-data/demo-notebook/diagnosis.json'))
    args = parser.parse_args()
    replay(args.verification, args.archive, args.output)
