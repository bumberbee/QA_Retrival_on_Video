"""Summarize measured ASR/runtime/disagreement and controlled retrieval results."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path(__file__).resolve().parent.parent / 'runs/gpu')
    args = p.parse_args()
    data = json.loads((args.runs / 'analysis/asr.json').read_text(encoding='utf8'))
    summary = {'asr': {}, 'downstream': {}, 'pairwise_weighted': []}
    for label, videos in data['systems'].items():
        runtime = [v['runtime'] for v in videos.values() if 'runtime' in v]
        summary['asr'][label] = {
            'videos': list(videos),
            'long_loops': sum(len(v['loops']) for v in videos.values()),
            'timing_le_1ms': sum(v['timing_le_1ms'] for v in videos.values()),
            'timing_gt_2s': sum(v['timing_gt_2s'] for v in videos.values()),
            'decode_seconds': sum(v['decode_seconds'] for v in runtime),
            'audio_seconds': sum(v['duration_seconds'] for v in runtime),
            'rtf': sum(v['decode_seconds'] for v in runtime) / sum(v['duration_seconds'] for v in runtime) if runtime else None,
            'terms_by_video': {k: v['terms'] for k, v in videos.items()},
            'loops_by_video': {k: v['loops'] for k, v in videos.items()},
        }
    pairs = {}
    for row in data['pairwise']:
        if row['a'] == 'exported' or row['b'] == 'exported':
            continue
        key = tuple(sorted([row['a'], row['b']]))
        record = pairs.setdefault(key, {'systems': list(key), 'videos': [], 'distance': 0, 'tokens': 0})
        record['videos'].append(row['video'])
        record['distance'] += row['distance']
        record['tokens'] += sum(max(x['tokens_a'], x['tokens_b']) for x in row['windows'])
    for row in pairs.values():
        row['disagreement'] = row['distance'] / max(1, row['tokens'])
        summary['pairwise_weighted'].append(row)
    for path in sorted((args.runs / 'downstream').glob('*-eval/retrieval.json')):
        row = json.loads(path.read_text(encoding='utf8'))
        summary['downstream'][path.parent.name] = {'overall': row['overall'], 'per_video': row.get('per_video')}
    (args.runs / 'analysis/summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    for k, v in summary['asr'].items():
        print(k, {x: v[x] for x in ['videos', 'long_loops', 'timing_le_1ms', 'timing_gt_2s', 'rtf']})
    for row in summary['pairwise_weighted']:
        print('disagreement', row['systems'], round(row['disagreement'], 4))


if __name__ == '__main__':
    main()
