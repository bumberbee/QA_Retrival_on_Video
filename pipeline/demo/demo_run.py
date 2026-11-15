"""Private demo operations with the production ledger gate and small downloads only."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import build_demo_pipeline as demo
import production_run as production

DESTINATION = Path('D:/thesis-data/demo-notebook')
LEDGER = production.OUTPUT / 'gpu-ledger.json'


def prepare(attempt=1):
    if attempt not in (1, 2):
        raise ValueError('At most two demo GPU attempts are authorized')
    slug = 'lecture-rag-demo-pipeline-test-20261104' + ('-attempt2' if attempt == 2 else '')
    folder = 'attempt' + str(attempt)
    directory = DESTINATION / folder / 'kernel'
    notebook = demo.build(directory / 'demo_pipeline.ipynb')
    meta = {'id': production.OWNER + '/' + slug, 'title': slug, 'code_file': notebook.name,
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
            'enable_tpu': False, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
            'dataset_sources': production.DATASETS, 'kernel_sources': [], 'model_sources': [],
            'competition_sources': []}
    production.validate_metadata(meta)
    production.write(directory / 'kernel-metadata.json', meta)
    return directory


def download_small(ref, destination):
    """Use output listings but never request large artifact URLs or print signed URLs."""
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
    api = KaggleApi()
    api.authenticate()
    owner, slug = ref.split('/')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    token, saved = None, []
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug, request.page_size = owner, slug, 200
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            if token is None and response.log:
                (destination / (slug + '.log')).write_text(response.log, encoding='utf-8')
                saved.append(destination / (slug + '.log'))
            for item in response.files or []:
                name = Path(item.file_name).name
                if name != 'demo_pipeline.executed.ipynb' and name != slug + '.log':
                    continue
                target = destination / name
                with requests.get(item.url, stream=True, timeout=(30, 60)) as reply:
                    reply.raise_for_status()
                    with target.open('wb') as handle:
                        for block in reply.iter_content(1 << 20):
                            handle.write(block)
                saved.append(target)
            token = response.next_page_token
            if not token:
                break
    print('Downloaded:', ', '.join(p.name for p in saved))
    return saved


def verify_example_answers(queries, diagnostics):
    """Acceptance checks for the published L25 examples, never used by retrieval/LLM."""
    if len(queries) != 3 or len(diagnostics) != 3:
        return False
    checks = [lambda text: '1930' in text and '1945' in text,
              lambda text: bool(re.search(r'(tháng\s+0?3\s+(?:năm\s+)?1945|0?3/1945)', text, re.I))]
    for query, diagnostic, fact in zip(queries[:2], diagnostics[:2], checks):
        answer = query.get('answer', '')
        if 'không đề cập' in answer.casefold() or not fact(answer):
            return False
        stamps = set(re.findall(r'\[(\d+:\d{2})(?:[–-]\d+:\d{2})?\]', answer))
        supporting = [c for c in diagnostic.get('context_evidence', [])
                      if f"{int(c['start_sec']) // 60:02d}:{int(c['start_sec']) % 60:02d}" in stamps
                      and fact(c['text'] + ' ' + c.get('title', ''))]
        if not supporting:
            return False
    return 'không đề cập' in queries[-1].get('answer', '').casefold()


def inspect_and_account(ref, destination):
    destination = Path(destination)
    log = destination / (ref.split('/')[1] + '.log')
    notebook = destination / 'demo_pipeline.executed.ipynb'
    highlights, errors, stages = {}, [], []
    if notebook.exists():
        n = json.loads(notebook.read_text(encoding='utf-8'))
        for index, cell in enumerate(n['cells']):
            outputs = cell.get('outputs', [])
            errors += [{'cell': index, 'name': o.get('ename'), 'error': o.get('evalue')}
                       for o in outputs if o['output_type'] == 'error']
            tag = cell.get('metadata', {}).get('demo_stage')
            if tag:
                plain = ''.join(''.join(o.get('text', '')) for o in outputs if o['output_type'] == 'stream')
                has_result = bool(plain.strip()) or any(o['output_type'] == 'display_data' for o in outputs)
                stages.append({'stage': tag, 'outputs': len(outputs), 'printed_result': has_result})
                highlights[tag] = plain
    expected = ['00', '01', 'gate', '02', '03', '04', '05', 'query_setup', 'query', 'timing']
    stage_passed = [s['stage'] for s in stages if s['printed_result']] == expected and not errors
    summary_line = next((line[len('Stage results: '):] for line in highlights.get('timing', '').splitlines()
                         if line.startswith('Stage results: ')), None)
    results = ast.literal_eval(summary_line) if summary_line else {}
    queries = results.get('queries', [])
    qa_passed = verify_example_answers(queries, results.get('retrieval_diagnostics', []))
    passed = stage_passed and qa_passed
    text = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
    if text.lstrip().startswith('['):
        timestamps = [float(event['time']) for event in json.loads(text) if isinstance(event.get('time'), (int, float))]
    else:
        timestamps = [float(m[1]) for m in re.finditer(r'^\s*(\d+(?:\.\d+)?)\s', text, re.M)]
    if not timestamps:
        raise RuntimeError('Missing timed Kaggle worker log; retain conservative ledger charge')
    # The last worker/conversion timestamp includes setup, inference and export; queue excluded.
    seconds = max(timestamps)
    rows = json.loads(LEDGER.read_text(encoding='utf-8'))
    row = next(r for r in rows if r['ref'] == ref)
    if row.get('status') not in {'COMPLETE', 'ERROR', 'CANCELLED'}:
        raise RuntimeError('Only terminal runs can be accounted from a complete worker log')
    row.update(actual_kernel_log_seconds=seconds, actual_allocated_t4_hours=seconds * row['devices'] / 3600,
               runtime_basis='Kaggle timed worker log through final conversion event; two allocated T4s; queue excluded',
               downloaded_to=str(destination), demo_verified=passed,
               completed_observed_utc=datetime.now(timezone.utc).isoformat())
    production.write(LEDGER, rows)
    report = {'ref': ref, 'status': row['status'], 'verified': passed, 'stages': stages, 'errors': errors,
              'stage_execution_verified': stage_passed, 'qa_verified': qa_passed,
              'results': results, 'highlights': highlights, 'worker_seconds': seconds,
              'actual_t4_hours': row['actual_allocated_t4_hours'],
              'ledger_total_t4_hours': sum(production.charged_hours(r) for r in rows if r.get('submitted'))}
    production.write(destination / 'verification.json', report)
    print('Status:', report['status'], '| every stage printed:', stage_passed,
          '| Q&A verified:', qa_passed, '| errors:', errors)
    print('Worker seconds:', seconds, '| T4 hours:', report['actual_t4_hours'],
          '| ledger total:', report['ledger_total_t4_hours'])
    for stage in stages:
        print(stage['stage'], 'outputs:', stage['outputs'], '| result:', stage['printed_result'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'launch', 'collect'])
    parser.add_argument('--attempt', choices=['1', '2'], default='1')
    args = parser.parse_args()
    attempt = int(args.attempt)
    directory = DESTINATION / ('attempt' + str(attempt)) / 'kernel'
    if args.action == 'prepare':
        print(prepare(attempt))
    elif args.action == 'launch':
        production.launch(directory, 1800)
    else:
        meta = json.loads((directory / 'kernel-metadata.json').read_text(encoding='utf-8'))
        destination = directory.parent / 'output'
        download_small(meta['id'], destination)
        inspect_and_account(meta['id'], destination)
