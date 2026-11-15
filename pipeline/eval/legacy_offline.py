"""Frozen legacy questions with isolated current retrieval, no app/config/credential imports."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from offline_common import ROOT, VIDEOS, encode_batches, experiment_dir, load_backend, read_json, write_json
from eval_retrieval import overlap
from shared.source_paths import LEGACY_GOLD

LEGACY = LEGACY_GOLD / 'qa/questions.json'
SOURCES = {'speech': frozenset({'speech'}), 'visual': frozenset({'ocr', 'region'}), 'combined': None}


def select_questions(data, videos):
    return [{'id': f'legacy_{video}_{i+1:02d}', 'video': video, **q}
            for video, questions in data['videos'].items() if video in videos
            for i, q in enumerate(questions)]


def aggregate(rows):
    answered = [r for r in rows if r['answerable']]
    outside = [r for r in rows if not r['answerable']]
    n = len(answered)
    return {'questions': len(rows), 'answerable': n, 'unanswerable': len(outside),
            **{key: sum(r[key] for r in answered)/n if n else None for key in ('hit1','hit3','mrr','retained_time_overlap')},
            'context_recall': None,
            'context_recall_note': 'Legacy gold lacks consistent key-term labels; retained_time_overlap is a temporal proxy only.',
            'unanswerable_scores': [{'id':r['id'],'semantic':r['best_semantic'],'rrf':r['top_rrf'],
                                     'returned_slides':r['returned_slides']} for r in outside]}


def score_one(q, found, context):
    answerable = q['answer'] is not None
    hits = [overlap(s['start_sec'], s['end_sec'], [q['answer']]) for s in found['slides']] if answerable else []
    evidence = found.get('context_evidence', [])
    return {'id': q['id'], 'video':q['video'], 'question':q['q'], 'where':q.get('where'),
            'answerable':answerable, 'best_semantic':found['best_semantic'],
            'top_rrf':found['slides'][0]['score'] if found['slides'] else 0,
            'returned_slides':len(found['slides']), 'context':context,
            'hit1':int(any(hits[:1])), 'hit3':int(any(hits[:3])),
            'mrr':next((1/(i+1) for i,h in enumerate(hits) if h),0),
            'retained_time_overlap': int(answerable and any(e.get('text') and
                overlap(e['start_sec'],e['end_sec'],[q['answer']]) for e in evidence))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index-root',type=Path,required=True)
    p.add_argument('--questions',type=Path,default=LEGACY)
    p.add_argument('--videos',nargs='+',default=VIDEOS,choices=VIDEOS)
    p.add_argument('--name',required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args()
    data=read_json(args.questions)
    questions=select_questions(data,args.videos)
    output=experiment_dir(args.name,args.output_root)
    backend=load_backend()
    encoder=backend.OnnxEncoder()
    vectors=encode_batches(encoder,[q['q'] for q in questions],'query: ')
    folders={v:args.index_root/v/'05_index' for v in {q['video'] for q in questions}}
    report={'questions_sha256':hashlib.sha256(args.questions.read_bytes()).hexdigest(),
            'scope':{'full_questions':sum(len(q) for q in data['videos'].values()),
                     'intersection_questions':len(questions),
                     'intersection_answerable':sum(q['answer'] is not None for q in questions),
                     'excluded_videos':sorted(set(data['videos'])-set(args.videos))},
            'encoder':{'model':backend.E5_REPO,'revision':backend.E5_REVISION,'device':'CPU ONNX'},
            'retrieval':'current defaults; exact legacy half-open group overlap; no silent video skips',
            'ablations':{}}
    backend.load_index.cache_clear()
    with patch.object(backend,'index_dir',side_effect=lambda v:folders[v]):
        for label,sources in SOURCES.items():
            rows=[]
            for q,vector in zip(questions,vectors):
                fixed=type('EncodedQuery',(),{'encode':lambda self,text,v=vector:v})()
                found=backend.search(q['video'],q['q'],q.get('current_time'),sources=sources,encoder=fixed)
                rows.append(score_one(q,found,backend.build_context(found)))
            report['ablations'][label]={'overall':aggregate(rows),'questions':rows,
                'per_video':{v:aggregate([r for r in rows if r['video']==v]) for v in folders}}
            write_json(output/'legacy.json',report)
    backend.load_index.cache_clear()
    print(json.dumps({k:v['overall'] for k,v in report['ablations'].items()},ensure_ascii=False))


if __name__=='__main__':
    main()
