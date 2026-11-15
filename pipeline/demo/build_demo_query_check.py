"""Zero-GPU regression check: reuse the private GPU run's index and Ollama weights."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import build_demo_pipeline as demo
from build_notebooks import cell
import demo_run
import production_run as production

REF = production.OWNER + '/lecture-rag-demo-query-cpu-20261104-evidence'
DIRECTORY = demo_run.DESTINATION / 'cpu-query-evidence'
PARENT = production.OWNER + '/lecture-rag-demo-pipeline-test-20261104-attempt2'

CONFIG = '''from pathlib import Path
import time, torch
assert torch.cuda.device_count() == 0, 'This regression check must use zero GPUs'
VIDEO_IDS = ['L25_V037']
DEMO_ENHANCED_CONTEXT = True
OUTPUT_ROOT = Path('/kaggle/working/cpu_query_check')
OUTPUT_ROOT.mkdir(exist_ok=True)
MAX_TABLE_ROWS, MAX_TEXT_CHARS = 40, 800
SESSION_STARTED = time.monotonic()
DEMO_REPORT, DEMO_TIMES = {}, {}
print('CPU query check | CUDA devices:', torch.cuda.device_count())
'''

SETUP = '''import requests, os, subprocess, shutil
from types import SimpleNamespace
sources = [p.parent for p in Path('/kaggle/input').rglob('demo-summary.json')
           if (p.parent / 'L25_V037/05_index/chunks.jsonl').exists()]
assert len(sources) == 1, f'Expected one attached GPU pipeline output; found {len(sources)}'
SOURCE_ROOT = sources[0]
print('Reused pipeline index validation:', validate_video_output(SOURCE_ROOT / VIDEO_IDS[0], VIDEO_IDS[0])['passed'])
app_search.index_dir = lambda video_id: SOURCE_ROOT / video_id / '05_index'
install = OUTPUT_ROOT / 'ollama'
install.mkdir(exist_ok=True)
subprocess.run(['tar', '-xzf', str(SOURCE_ROOT / 'ollama.tgz'), '-C', str(install)], check=True, timeout=180)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OLLAMA_HOST='127.0.0.1:11434',
           OLLAMA_MODELS=str(SOURCE_ROOT / 'ollama_models'))
server_log = (OUTPUT_ROOT / 'ollama_server.log').open('w')
server = subprocess.Popen([str(install / 'bin/ollama'), 'serve'], env=env, stdout=server_log, stderr=server_log)
for _ in range(60):
    try:
        requests.get('http://127.0.0.1:11434/api/version', timeout=2).raise_for_status()
        break
    except requests.RequestException:
        if server.poll() is not None: raise RuntimeError('CPU Ollama server exited')
        time.sleep(1)
else: raise TimeoutError('CPU Ollama start timeout')
os.environ['OLLAMA_MODEL'] = app_answer.DEFAULT_OLLAMA_MODEL
os.environ['OLLAMA_BASE_URL'] = app_answer.DEFAULT_OLLAMA_URL
os.environ['OLLAMA_NUM_CTX'] = '16384'
os.environ['OLLAMA_MAX_TOKENS'] = '192'
os.environ['OLLAMA_TIMEOUT_SECONDS'] = '240'
requests.post('http://127.0.0.1:11434/api/generate', json={'model': app_answer.DEFAULT_OLLAMA_MODEL,
              'prompt': '', 'stream': False, 'keep_alive': -1}, timeout=180).raise_for_status()
models = requests.get('http://127.0.0.1:11434/api/ps', timeout=10).json()['models']
assert models and all(m.get('size_vram', 0) == 0 for m in models)
print('Answer model:', app_answer.DEFAULT_OLLAMA_MODEL, '| CPU only | T4 hours: 0')
'''


def build():
    search, answer = demo.backend_sources()
    capture = demo.fast.module_path('demo_capture.py').read_text(encoding='utf-8')
    config = CONFIG.replace("print('CPU query check", f'exec({capture!r})\n'
        + '_demo_capture = DemoNotebookCapture(_capture_ipython(), "/kaggle/working/demo_pipeline.executed.ipynb", '
        + '{"for question in QUESTIONS:": ("query", "## Hỏi đáp từ chỉ mục đã kiểm tra")})\n'
        + "print('CPU query check")
    questions = next(code for tag, _, _, code in demo.SECTIONS if tag == 'query')
    questions = questions.replace('    ask(question)', '''    try:
        ask(question)
    except app_answer.LLMUnavailable as error:
        DEMO_REPORT['cpu_impractical'] = str(error)
        print('CPU_IMPRACTICAL:', error)
        break''')
    cells = [cell('markdown', '# Kiểm tra hồi quy hỏi đáp trên CPU\n\nDùng nguyên chỉ mục của lần chạy GPU; không chạy lại pipeline.'),
             cell('code', config),
             cell('code', '%pip install -q "onnxruntime>=1.20,<2"'),
             cell('code', demo.fast.module_path('common.py').read_text(encoding='utf-8'), hidden=True),
             cell('code', demo.fast.module_path('production_runtime.py').read_text(encoding='utf-8'), hidden=True),
             cell('code', demo.fast.module_path('demo_runtime.py').read_text(encoding='utf-8'), hidden=True),
             cell('code', 'app_search = embedded_module("cpu_app_search", ' + repr(search)
                  + ', {"DEMO_STORAGE": DemoStorage()})\napp_answer = embedded_module("cpu_app_answer", ' + repr(answer) + ')'),
             cell('code', SETUP), cell('code', questions), cell('code', 'demo_timing()')]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    target = DIRECTORY / 'query_check.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    metadata = {'id': REF, 'title': REF.split('/')[1], 'code_file': target.name, 'language': 'python',
                'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
                'enable_internet': True, 'dataset_sources': production.DATASETS, 'model_sources': [],
                'kernel_sources': [PARENT], 'competition_sources': []}
    production.write(DIRECTORY / 'kernel-metadata.json', metadata)
    print('Built CPU-only query regression check')
    return target


def launch():
    if (DIRECTORY / 'cpu-run.json').exists():
        raise RuntimeError('CPU regression check already reserved; never overwrite an existing kernel')
    listing = production.cli('kernels', 'list', '--mine', '--search', REF.split('/')[1], '--page-size', '100')
    if REF in listing:
        raise RuntimeError('CPU kernel already exists')
    record = {'ref': REF, 'devices': 0, 'timeout_seconds': 1800, 'actual_t4_hours': 0,
              'submitted': True, 'launch_utc': datetime.now(timezone.utc).isoformat()}
    production.write(DIRECTORY / 'cpu-run.json', record)
    production.cli('kernels', 'push', '-p', DIRECTORY, '-t', 1800)
    remote_dir = DIRECTORY / 'metadata'
    production.cli('kernels', 'pull', REF, '-p', remote_dir, '-m')
    remote = json.loads((remote_dir / 'kernel-metadata.json').read_text(encoding='utf-8'))
    assert remote['is_private'] is True and remote['enable_gpu'] is False and remote['enable_tpu'] is False
    record.update(privacy_verified=True, status='RUNNING')
    production.write(DIRECTORY / 'cpu-run.json', record)
    print(REF, '| PRIVATE | GPU disabled | T4 hours: 0')


def collect():
    record = json.loads((DIRECTORY / 'cpu-run.json').read_text(encoding='utf-8'))
    if record['status'] not in {'COMPLETE', 'ERROR', 'CANCELLED', 'CANCEL_ACKNOWLEDGED'}:
        raise RuntimeError('Observe and record a terminal CPU status before collecting outputs')
    destination = DIRECTORY / 'output'
    demo_run.download_small(REF, destination)
    n = json.loads((destination / 'demo_pipeline.executed.ipynb').read_text(encoding='utf-8'))
    errors = [o.get('evalue') for c in n['cells'] for o in c.get('outputs', []) if o['output_type'] == 'error']
    streams = ''.join(''.join(o['text']) for c in n['cells'] for o in c.get('outputs', []) if o['output_type'] == 'stream')
    import ast
    line = next((s[len('Stage results: '):] for s in streams.splitlines() if s.startswith('Stage results: ')), '{}')
    results = ast.literal_eval(line)
    queries = results.get('queries', [])
    passed = not errors and demo_run.verify_example_answers(queries, results.get('retrieval_diagnostics', []))
    record.update(verified=passed, errors=errors, results=results,
                  actual_t4_hours=0, stage_execution_verified=not errors and bool(results),
                  qa_verified=passed)
    production.write(DIRECTORY / 'cpu-run.json', record)
    production.write(destination / 'verification.json', record)
    print('CPU Q&A verified:', passed, '| errors:', errors, '| T4 hours: 0')
    for q in queries:
        print(q['question'], '\n', q['answer'], '\nseconds:', q['seconds'])
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'launch', 'collect'])
    action = parser.parse_args().action
    {'build': build, 'launch': launch, 'collect': collect}[action]()
