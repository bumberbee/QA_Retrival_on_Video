"""Prepare and operate only new PRIVATE 20261102 kernels via the authenticated CLI.

All remote downloads and the authoritative budget ledger live on D:.
The conservative budget charges the full timeout even when a kernel ends early.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import sys
import os
from datetime import datetime, timezone

from build_fast_pipeline import build

ROOT = Path(__file__).resolve().parent
OUTPUT = Path('D:/thesis-data/run-20261102')
OWNER = 'thnhtrungnguynmif'
DATASETS = [OWNER + '/' + slug for slug in
            ('l25a-lecture-videos-media', 'vn-lecture-test12', 'slidespeech-train100')]


def cli(*args):
    result = subprocess.run(['kaggle', *map(str, args)], capture_output=True, text=True,
                            encoding='utf8', errors='replace', timeout=300)
    if result.returncode:
        raise RuntimeError((result.stderr + result.stdout)[-3000:])
    # Some CLI errors print a message and return 0; never mistake them for success.
    if 'Cannot access kernel' in result.stdout or result.stdout.startswith(('403', '404')):
        raise RuntimeError(result.stdout[-3000:])
    return result.stdout


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    temporary.replace(path)


def validate_metadata(meta):
    if meta['is_private'] is not True:
        raise RuntimeError('STOP: kernel is PUBLIC or privacy is unverified')
    if meta['enable_gpu'] is not True or meta['enable_tpu'] is not False or meta['enable_internet'] is not True:
        raise RuntimeError('Invalid GPU/TPU/Internet settings')
    if meta['machine_shape'] not in {'NvidiaTeslaT4', 'nvidiaTeslaT4'}:
        raise RuntimeError('T4 x2 machine shape required')
    if sorted(meta['dataset_sources']) != sorted(DATASETS):
        raise RuntimeError('Required three dataset attachments differ')
    if not meta['id'].startswith(OWNER + '/') or '20261102' not in meta['id']:
        raise RuntimeError('Only new owner slugs containing 20261102 are permitted')


def build_bundle(kind, layout='dual', resume_from=None, checkpoint_kernel=None):
    suffix = {'full': '', 'canary': '-canary', 'canary-alt': '-canary-alt', 'resume': '-resume',
              'confirm': '-confirm'}[kind]
    slug = 'lecture-rag-fast-10' + suffix + '-20261102'
    directory = ROOT / 'runs' / ('20261102-fast-10' if kind == 'full' else '20261102-fast-10' + suffix)
    overrides = {} if layout == 'dual' else {'HYBRID_DEVICES': ('cuda:0',)}
    if resume_from:
        overrides['RESUME_FROM'] = str(resume_from)
    notebook = directory / 'fast_pipeline_l25.ipynb'
    build(notebook, canary=kind.startswith('canary'), overrides=overrides)
    meta = {'id': OWNER + '/' + slug, 'title': slug, 'code_file': notebook.name,
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
            'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': DATASETS,
            'competition_sources': [], 'kernel_sources': [checkpoint_kernel] if checkpoint_kernel else []}
    validate_metadata(meta)
    write(directory / 'kernel-metadata.json', meta)
    digest = hashlib.sha256(notebook.read_bytes()).hexdigest()
    write(directory / 'bundle.json', {'sha256': digest, 'kind': kind, 'layout': layout,
                                     'metadata': meta})
    print(json.dumps({'bundle': str(directory), 'sha256': digest, 'id': meta['id']}))
    return directory


def status(ref):
    result = cli('kernels', 'status', ref)
    match = re.search(r'has status "([^"]+)"', result)
    if not match:
        raise RuntimeError('Unrecognized kernel status: ' + result)
    return match[1].upper().split('.')[-1], result


def verify_remote(directory, *, observed=None):
    local = json.loads((directory / 'kernel-metadata.json').read_text(encoding='utf8'))
    state, raw = status(local['id']) if observed is None else observed
    dest = OUTPUT / 'metadata' / local['id'].split('/')[1]
    cli('kernels', 'pull', local['id'], '-p', dest, '-m')
    remote = json.loads((dest / 'kernel-metadata.json').read_text(encoding='utf8'))
    validate_metadata(remote)
    evidence = {'status': state, 'status_cli': raw, 'privacy_verified': True,
                'metadata': remote, 'verified_utc': datetime.now(timezone.utc).isoformat()}
    write(dest / 'verification.json', evidence)
    return evidence


def monitor(directory, destination=None, interval=300):
    """Persist five-minute observations; download terminal output without repolling."""
    if interval < 300:
        raise ValueError('Kaggle polling interval must be at least 300 seconds')
    directory = Path(directory)
    meta = json.loads((directory / 'kernel-metadata.json').read_text(encoding='utf8'))
    validate_metadata(meta)
    ledger = OUTPUT / 'gpu-ledger.json'
    rows = json.loads(ledger.read_text(encoding='utf8'))
    row = next(r for r in rows if r['ref'] == meta['id'])
    if row.get('privacy_verified') is not True:
        raise RuntimeError('STOP: monitor privacy is unverified')
    due = time.monotonic() + interval
    while True:
        while time.monotonic() < due:
            time.sleep(min(60, max(0, due - time.monotonic())))
        due = time.monotonic() + interval
        rows = json.loads(ledger.read_text(encoding='utf8'))
        row = next(r for r in rows if r['ref'] == meta['id'])
        try:
            state, raw = status(meta['id'])
        except RuntimeError as exc:
            row.setdefault('monitor_errors', []).append(str(exc)[-500:])
            write(ledger, rows)
            continue
        row.update(status=state, status_cli=raw,
                   status_observed_utc=datetime.now(timezone.utc).isoformat())
        write(ledger, rows)
        print(json.dumps({k: row[k] for k in ('ref', 'status', 'status_observed_utc')}), flush=True)
        if state in {'COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED'}:
            evidence = verify_remote(directory, observed=(state, raw))
            row.update(evidence)
            write(ledger, rows)
            if destination is not None:
                download(directory, destination, verification=evidence)
            return state


def charged_hours(row):
    # A finished kernel is charged its measured allocation; anything else keeps the full timeout.
    actual = row.get('actual_allocated_t4_hours')
    if row.get('status') in {'COMPLETE', 'ERROR', 'CANCELLED'} and isinstance(actual, (int, float)) \
            and 0 < actual <= row['timeout_seconds'] * 2 / 3600:
        return actual
    return row['timeout_seconds'] * 2 / 3600


def budget_bound(rows, timeout):
    used = sum(charged_hours(r) for r in rows if r.get('submitted'))
    bound = used + timeout * 2 / 3600
    if timeout <= 0 or bound > 8 + 1e-9:
        raise RuntimeError(f'Task GPU budget exceeded: conservative bound {bound:.6f} T4-hours')
    return bound


def launch(directory, timeout):
    directory = Path(directory)
    meta = json.loads((directory / 'kernel-metadata.json').read_text(encoding='utf8'))
    validate_metadata(meta)
    ledger_path = OUTPUT / 'gpu-ledger.json'
    rows = json.loads(ledger_path.read_text(encoding='utf8')) if ledger_path.exists() else []
    if any(r['ref'] == meta['id'] for r in rows):
        raise RuntimeError('Never overwrite an existing task kernel; choose a new slug')
    listing = cli('kernels', 'list', '--mine', '--search', meta['id'].split('/')[1], '--page-size', '100')
    if meta['id'] in listing:
        raise RuntimeError('Refusing to modify an existing Kaggle kernel: ' + meta['id'])
    bound = budget_bound(rows, timeout)
    row = {'ref': meta['id'], 'timeout_seconds': timeout, 'devices': 2,
           'conservative_hours_bound': bound, 'submitted': True,
           'notebook_sha256': hashlib.sha256((directory / meta['code_file']).read_bytes()).hexdigest(),
           'launch_utc': datetime.now(timezone.utc).isoformat()}
    # Check D: writes BEFORE allocating any GPU time.
    write(OUTPUT / 'quota-before-launch.txt.json', {'cli': cli('quota')})
    rows.append(row)
    write(ledger_path, rows)
    # Reserve before the request: an ambiguous network failure still consumes the bound.
    output = cli('kernels', 'push', '-p', directory, '-t', timeout)
    row.update(submitted=True, push_cli=output)
    version = re.search(r'version\s+(\d+)', output, re.I)
    row['version'] = int(version[1]) if version else None
    write(ledger_path, rows)
    # Mandatory immediate remote privacy check. Any public/unverified result raises.
    row.update(verify_remote(directory))
    write(ledger_path, rows)
    print(json.dumps(row, ensure_ascii=False))


def download(directory, destination=None, *, verification=None):
    directory = Path(directory)
    meta = json.loads((directory / 'kernel-metadata.json').read_text(encoding='utf8'))
    dest = Path(destination) if destination is not None else OUTPUT / meta['id'].split('/')[1]
    if not dest.resolve().is_relative_to(OUTPUT.resolve()):
        raise RuntimeError('Downloads must stay under D:/thesis-data/run-20261102')
    evidence = verify_remote(directory) if verification is None else verification
    validate_metadata(evidence['metadata'])
    if evidence.get('privacy_verified') is not True or evidence['metadata']['id'] != meta['id']:
        raise RuntimeError('STOP: download privacy is unverified')
    if evidence['status'] not in {'COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED'}:
        raise RuntimeError('Output download requires a terminal kernel')
    cli('kernels', 'output', meta['id'], '-p', dest, '--page-size', '200')
    rows = json.loads((OUTPUT / 'gpu-ledger.json').read_text(encoding='utf8'))
    row = next(r for r in rows if r['ref'] == meta['id'])
    row.update(evidence, downloaded_to=str(dest), completed_observed_utc=datetime.now(timezone.utc).isoformat())
    write(OUTPUT / 'gpu-ledger.json', rows)
    print(json.dumps({k: row[k] for k in ('ref', 'version', 'status', 'privacy_verified', 'downloaded_to')},
                     ensure_ascii=False))


def validate_outputs(root, report_path, mp4_root=Path('D:/thesis-data/mp4-extra')):
    """Run model-free validation against the exact ten local video sources."""
    root, report_path = Path(root), Path(report_path)
    if not report_path.resolve().is_relative_to(OUTPUT.resolve()):
        raise RuntimeError('Validation reports must stay under the task output root')
    sys.path.insert(0, str(ROOT / 'eval'))
    from offline_common import VIDEOS, load_notebook
    runtime = load_notebook('production_runtime')
    progress = json.loads((root / 'progress.json').read_text(encoding='utf8'))
    rows = progress['videos']
    failures = []
    if len(rows) != len(VIDEOS) or {r['video_id'] for r in rows} != set(VIDEOS):
        failures.append('Progress must contain exactly ten configured videos')
    if not json.loads((root / 'preflight.json').read_text(encoding='utf8')).get('passed'):
        failures.append('Preflight did not pass')
    results = []
    for video in VIDEOS:
        try:
            row = next(r for r in rows if r['video_id'] == video)
            if row['status'] != 'completed' or row.get('error') or any(
                    s.get('error') for s in row.get('stage_timings', {}).values()):
                raise ValueError('Video has an incomplete/error progress row')
            result = runtime.validate_video_output(root / video, video)
            if runtime.file_sha256(Path(mp4_root) / (video + '.mp4')) != result['source_sha256']:
                raise ValueError('Exact local MP4 source hash differs')
            manifest = runtime.load_stage(root / video / '01_slides', 1, video)
            duration = manifest['source']['duration_sec']
            transcript = runtime.read_json(root / video / '03_audio/transcript.json')
            words = transcript.get('words', [])
            intervals = runtime.read_json(root / video / '01_slides/intervals.json')
            visuals = runtime.read_json(root / video / '02_visual/visual.json')
            segments = runtime.read_json(root / video / runtime.STAGE_NAMES[4] / 'segments.json')
            for stage, name, count in ((2, 'record_count', len(visuals)),
                                       (3, 'word_count', len(words)),
                                       (4, 'segment_count', len(segments))):
                directory = root / video / runtime.STAGE_NAMES[stage]
                if runtime.read_json(directory / 'manifest.json').get(name) != count:
                    raise ValueError(f'Stage {stage} {name} differs from assets')
            if len(intervals) != len(visuals):
                raise ValueError('Slide/visual record count differs')
            if sum(v.get('ocr_status') == 'failed' for v in visuals) != result['ocr_failed']:
                raise ValueError('OCR failure count differs from assets')
            for segment in segments:
                if not 0 <= segment['start_sec'] < segment['end_sec'] <= duration + .03:
                    raise ValueError('Aligned segment timestamp out of bounds')
            for word in words:
                start, end = word.get('start_sec', word.get('start')), word.get('end_sec', word.get('end'))
                if start is None or end is None or not 0 <= start <= end <= duration + .03:
                    raise ValueError('ASR word timestamp out of bounds')
            result.update(words=len(words), intervals=len(intervals), segments=len(segments),
                          visual_records=len(visuals), stage_timings=row['stage_timings'],
                          retries=row.get('retries', 0), progress_ocr_failed=row.get('ocr_failed', 0))
            results.append(result)
        except Exception as exc:
            failures.append(f'{video}: {type(exc).__name__}: {exc}')
    report = {'passed': not failures, 'videos': results, 'failures': failures,
              'total_wall_sec': progress.get('total_wall_sec')}
    write(report_path, report)
    print(json.dumps({'passed': report['passed'], 'validated': len(results), 'failures': failures}), flush=True)
    if failures:
        raise RuntimeError('Production output validation failed; see report on D:')
    return report


def evaluate_when_downloaded(ref):
    """Wait on local state only; validate before six isolated CPU evaluation jobs."""
    state_path = OUTPUT / 'cpu-evaluation-state.json'
    write(state_path, {'status': 'waiting_for_download', 'ref': ref})
    while True:
        rows = json.loads((OUTPUT / 'gpu-ledger.json').read_text(encoding='utf8'))
        row = next(r for r in rows if r['ref'] == ref)
        if row.get('downloaded_to'):
            if row.get('privacy_verified') is not True:
                raise RuntimeError('STOP: downloaded kernel privacy is unverified')
            root = Path(row['downloaded_to']) / 'lecture_rag'
            break
        time.sleep(30)
    try:
        write(state_path, {'status': 'validating', 'root': str(root)})
        validate_outputs(root, OUTPUT / 'validation.json')
        out = OUTPUT / 'eval'
        out.mkdir(parents=True, exist_ok=True)
        steps = []
        for name in ('turbo', 'medium'):
            args = ['--export-root', str(root), '--output-root', str(out), '--name', name,
                    '--skip-reference-comparison']
            if name == 'medium':
                args += ['--asr-export-root', 'D:/thesis-data/run-20261020/export']
            steps.append((name + '-rebuild', 'rebuild_offline.py', args))
        for name in ('turbo', 'medium'):
            for kind, script in (('dev', 'eval_retrieval.py'), ('legacy', 'legacy_offline.py')):
                steps.append((name + '-' + kind, script,
                              ['--index-root', str(out / name), '--output-root', str(out),
                               '--name', name + '-' + kind]))
        steps.append(('paired', 'paired_comparison.py', [str(out)]))
        done = []
        environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           PYTHONDONTWRITEBYTECODE='1', TEMP=str(OUTPUT / 'temp'))
        for label, script, args in steps:
            write(state_path, {'status': 'running', 'step': label, 'completed_steps': done})
            with (out / (label + '.log')).open('w', encoding='utf8') as log:
                result = subprocess.run([sys.executable, str(ROOT / 'eval' / script), *args],
                                        stdout=log, stderr=subprocess.STDOUT, env=environment,
                                        timeout=7200)
            if result.returncode:
                raise RuntimeError(f'CPU evaluation step {label} failed; see its log on D:')
            done.append(label)
            print('CPU completed: ' + label, flush=True)
        write(state_path, {'status': 'complete', 'completed_steps': done,
                           'completed_utc': datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        write(state_path, {'status': 'failed', 'error': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('build')
    p.add_argument('kind', choices=('full', 'canary', 'canary-alt', 'resume', 'confirm'))
    p.add_argument('--layout', choices=('dual', 'single-qwen'), default='dual')
    p.add_argument('--resume-from')
    p.add_argument('--checkpoint-kernel')
    p = sub.add_parser('validate')
    p.add_argument('root', type=Path)
    p.add_argument('--report', type=Path, default=OUTPUT / 'validation.json')
    p.add_argument('--mp4-root', type=Path, default=Path('D:/thesis-data/mp4-extra'))
    p = sub.add_parser('evaluate')
    p.add_argument('ref')
    for action in ('launch', 'verify', 'download', 'monitor'):
        p = sub.add_parser(action)
        p.add_argument('directory', type=Path)
        if action == 'launch':
            p.add_argument('--timeout', type=int, required=True)
        elif action == 'download':
            p.add_argument('--destination', type=Path)
        elif action == 'monitor':
            p.add_argument('--destination', type=Path)
            p.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    if args.action == 'build':
        build_bundle(args.kind, args.layout, args.resume_from, args.checkpoint_kernel)
    elif args.action == 'validate':
        validate_outputs(args.root, args.report, args.mp4_root)
    elif args.action == 'evaluate':
        evaluate_when_downloaded(args.ref)
    elif args.action == 'launch':
        launch(args.directory, args.timeout)
    elif args.action == 'verify':
        print(json.dumps(verify_remote(args.directory)))
    elif args.action == 'monitor':
        monitor(args.directory, args.destination, args.interval)
    else:
        download(args.directory, args.destination)


if __name__ == '__main__':
    main()
