"""Shared, credential-free helpers embedded in private Kaggle scripts."""
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

START = time.monotonic()
OUT = Path('/kaggle/working/results')
OUT.mkdir(parents=True, exist_ok=True)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf8')


def install(*packages):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], check=True, timeout=600)


def environment():
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                          '--format=csv,noheader'], capture_output=True, text=True)
    return {'python': sys.version, 'gpu': gpu.stdout.strip(),
            'packages': {d.metadata['Name']: d.version for d in metadata.distributions()}}


def vram():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
    return [int(x) for x in r.stdout.splitlines() if x.strip().isdigit()]


def export_root():
    # Prefer mounted individual exports, otherwise extract only stages needed by the experiments.
    paths = list(Path('/kaggle/input').rglob('transcript.json'))
    for p in paths:
        if p.parent.name == '03_audio' and p.parent.parent.name == 'hVU5Mw19V_c':
            return p.parents[2]
    for z in Path('/kaggle/input').rglob('*.zip'):
        with zipfile.ZipFile(z) as f:
            names = f.namelist()
            if not any('hVU5Mw19V_c/03_audio/transcript.json' in n for n in names):
                continue
            dest = Path('/kaggle/working/export_inputs')
            for n in names:
                if any('/' + s + '/' in '/' + n for s in ('01_slides', '02_visual', '03_audio')):
                    target = (dest / n).resolve()
                    if not target.is_relative_to(dest.resolve()):
                        raise ValueError('Unsafe archive path')
                    f.extract(n, dest)
            return next(dest.rglob('hVU5Mw19V_c/03_audio/transcript.json')).parents[2]
    raise FileNotFoundError('Attached fast-pipeline output has no stage-03 export')


def finish(extra=None):
    save(OUT / 'session.json', {'elapsed_seconds': time.monotonic() - START,
                              'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                              **(extra or {})})
    # Keep outputs limited to measurements; intermediate model copies are reproducible from pins.
    working=Path('/kaggle/working').resolve()
    for name in ('audio','export_inputs','phowhisper_ct2','ollama.tgz','ollama','ollama_models'):
        path=(working/name).resolve()
        if not path.is_relative_to(working) or path==working:raise ValueError('Unsafe scratch cleanup')
        if path.is_dir():shutil.rmtree(path)
        elif path.is_file():path.unlink()


def failure(label, error):
    detail = {'error': str(error), 'traceback': traceback.format_exc()}
    save(OUT / ('failure_' + label + '.json'), detail)
    print(label, 'FAILED', str(error), flush=True)
