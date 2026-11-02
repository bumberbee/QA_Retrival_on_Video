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
    suffix = {'full': '', 'canary': '-canary', 'canary-alt': '-canary-alt', 'resume': '-resume'}[kind]
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
    return match[1].upper(), result


def verify_remote(directory):
    local = json.loads((directory / 'kernel-metadata.json').read_text(encoding='utf8'))
    state, raw = status(local['id'])
    dest = OUTPUT / 'metadata' / local['id'].split('/')[1]
    cli('kernels', 'pull', local['id'], '-p', dest, '-m')
    remote = json.loads((dest / 'kernel-metadata.json').read_text(encoding='utf8'))
    validate_metadata(remote)
    evidence = {'status': state, 'status_cli': raw, 'privacy_verified': True,
                'metadata': remote, 'verified_utc': datetime.now(timezone.utc).isoformat()}
    write(dest / 'verification.json', evidence)
    return evidence


def budget_bound(rows, timeout):
    used = sum(r['timeout_seconds'] * 2 / 3600 for r in rows if r.get('submitted'))
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


def download(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'kernel-metadata.json').read_text(encoding='utf8'))
    evidence = verify_remote(directory)
    if evidence['status'] not in {'COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED'}:
        raise RuntimeError('Output download requires a terminal kernel')
    dest = OUTPUT / meta['id'].split('/')[1]
    cli('kernels', 'output', meta['id'], '-p', dest, '--page-size', '200')
    rows = json.loads((OUTPUT / 'gpu-ledger.json').read_text(encoding='utf8'))
    row = next(r for r in rows if r['ref'] == meta['id'])
    row.update(evidence, downloaded_to=str(dest), completed_observed_utc=datetime.now(timezone.utc).isoformat())
    write(OUTPUT / 'gpu-ledger.json', rows)
    print(json.dumps(row, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('build')
    p.add_argument('kind', choices=('full', 'canary', 'canary-alt', 'resume'))
    p.add_argument('--layout', choices=('dual', 'single-qwen'), default='dual')
    p.add_argument('--resume-from')
    p.add_argument('--checkpoint-kernel')
    for action in ('launch', 'verify', 'download'):
        p = sub.add_parser(action)
        p.add_argument('directory', type=Path)
        if action == 'launch':
            p.add_argument('--timeout', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'build':
        build_bundle(args.kind, args.layout, args.resume_from, args.checkpoint_kernel)
    elif args.action == 'launch':
        launch(args.directory, args.timeout)
    elif args.action == 'verify':
        print(json.dumps(verify_remote(args.directory)))
    else:
        download(args.directory)


if __name__ == '__main__':
    main()
