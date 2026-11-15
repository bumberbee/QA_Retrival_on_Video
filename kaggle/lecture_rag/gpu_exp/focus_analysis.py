"""Literal checks for the five visually verified audit-region crops.

These are targeted anchors, not a page-level OCR accuracy benchmark.
"""
import argparse
import json
from pathlib import Path
import re


def literal(text):
    text = re.sub(r'\\(?:d?frac)\s*\{(\d+)\}\s*\{(\d+)\}', r'\1/\2', text)
    return re.sub(r'\s+', '', text).lower().replace('.', ',')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = {'scope': 'Three denominators, one seed weight, one percentage; cropped code/physics reviewed separately.',
               'references': {'bayes_denominators': ['100/370', '120/370', '150/370'],
                              'seed_weights': ['1,5kg'], 'seed_percentage': ['46,75']}, 'systems': {}}
    for system in ['qwen_focus4', 'qwen_focus8']:
        rows = {}
        for path in (args.root / system).glob('*.json'):
            d = json.loads(path.read_text(encoding='utf8'))
            text = d['text']; anchors = results['references'].get(path.stem, [])
            rows[path.stem] = {'text': text, 'seconds': d['seconds'], 'peak_torch_vram_mib': d['peak_torch_vram_mib'],
                               'anchors': {x: literal(x) in literal(text) for x in anchors}}
            if path.stem == 'python_triple_quotes':
                rows[path.stem]['code_checks'] = {'a3_present': 'a3=' in literal(text),
                    'a4_present': 'a4=' in literal(text), 'two_triple_double_quotes': text.count('"""') >= 2,
                    'two_triple_single_quotes': text.count("'''") >= 2, 'invented_a5': 'a5=' in literal(text)}
        if rows:
            results['systems'][system] = {'cases': rows, 'numeric_anchors_correct': sum(v for x in rows.values() for v in x['anchors'].values()),
                'numeric_anchors_scored': sum(len(x['anchors']) for x in rows.values()),
                'seconds_per_crop': sum(x['seconds'] for x in rows.values()) / len(rows)}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
