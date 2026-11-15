"""One private GPU verification, reusing the finished math index and Qwen cache."""
import argparse
import ast
import json
from pathlib import Path
import re
import time
from datetime import datetime, timezone

import build_demo_pipeline as demo
from build_notebooks import cell
import production_run as production

SLUG = 'lecture-rag-demo-final2-20261104'
REF = production.OWNER + '/' + SLUG
PARENT = production.OWNER + '/lecture-rag-demo-math-l25-v008-20261104'
DEST = Path('D:/thesis-data/demo-notebook/final2')
DIRECTORY = DEST / 'kernel'
EXPECTED = Path('D:/thesis-data/demo-notebook/math-l25-v008/expected-answers.json')
TIMEOUT_SECONDS = 720
VIDEO_PATH = '/kaggle/input/datasets/thnhtrungnguynmif/l25a-lecture-videos-media/Videos_L25_a/video/L25_V008.mp4'

SETUP = '''RESOLVED_VIDEOS = resolve_video_sources(VIDEO_SOURCES, VIDEO_IDS, VIDEO_LANGUAGES)
VIDEO_IDS = list(RESOLVED_VIDEOS)
assert VIDEO_IDS == ['L25_V008']
print('VIDEO_SOURCES resolved:', RESOLVED_VIDEOS['L25_V008']['path'])
sources = [p.parent.parent for p in Path('/kaggle/input').rglob('chunks.jsonl')
           if p.parent.name == '05_index' and p.parent.parent.name == 'L25_V008']
assert len(sources) == 1, f'Expected one finished L25_V008 index, found {len(sources)}'
SOURCE_ROOT = sources[0].parent
validation = demo_validate_video_output(SOURCE_ROOT / VIDEO_IDS[0], VIDEO_IDS[0])
assert validation['passed']
app_search.index_dir = lambda video_id: SOURCE_ROOT / video_id / '05_index'
print('Reused index:', SOURCE_ROOT, '| validation:', validation['passed'])
assert (SOURCE_ROOT / 'ollama.tgz').is_file()
assert (SOURCE_ROOT / 'ollama_models/manifests/registry.ollama.ai/library/qwen2.5/7b').is_file()
# Keep the reviewed checksum and model alias checks; reuse the read-only weights.
reused = OLLAMA_START_SOURCE.replace("archive = OUTPUT_ROOT / 'ollama.tgz'",
                                    "archive = SOURCE_ROOT / 'ollama.tgz'")
reused = reused.replace("str(OUTPUT_ROOT / 'ollama_models')", "str(SOURCE_ROOT / 'ollama_models')")
reused = reused.replace("requests.post('http://127.0.0.1:11434/api/pull', json={'model': model, 'stream': False}, timeout=600).raise_for_status()",
                        "print('Reusing attached Qwen model cache; no model pull')")
exec(reused)
demo_query_setup()
RAW_LLM_CALLS = []
_original_ollama = app_answer._ollama
def audited_ollama(prompt, **kwargs):
    answer = _original_ollama(prompt, **kwargs)
    RAW_LLM_CALLS.append({'answer': answer, 'prompt_chars': len(prompt),
                          'temperature': kwargs.get('temperature'),
                          'max_tokens': kwargs.get('max_tokens') or int(os.environ['OLLAMA_MAX_TOKENS']),
                          'num_ctx': int(os.environ['OLLAMA_NUM_CTX']),
                          'rewrite': '--- BẰNG CHỨNG ---' not in prompt,
                          'retry': 'Các nhãn hợp lệ:' in prompt})
    return answer
app_answer._ollama = audited_ollama
'''

QUERIES = '''EXAMPLE_RESULTS = []
for question in QUESTIONS:
    first_call = len(RAW_LLM_CALLS)
    try:
        result = ask(question, show=True)
    except Exception as error:
        result = {'question': question, 'error': str(error), 'answer': getattr(error, 'answer', ''),
                  'used_evidence': [], 'unmatched_citations': []}
        if result['answer'] and DEMO_REPORT.get('retrieval_diagnostics'):
            found = {'context_evidence': DEMO_REPORT['retrieval_diagnostics'][-1]['context_evidence']}
            result.update(app_answer.evidence_attribution(result['answer'], found))
    result['raw_llm_calls'] = RAW_LLM_CALLS[first_call:]
    EXAMPLE_RESULTS.append(result)
    print('QUESTION:', question, '\\nANSWER:', result['answer'])
    print('Mapped labels:', [x['evidence_label'] for x in result['used_evidence']],
          '| unmatched:', result.get('unmatched_citations'), '| error:', result.get('error'))
'''

FINISH = '''elapsed = round(time.monotonic() - SESSION_STARTED, 2)
report = {'ref': VERIFICATION_REF, 'source_kernel': SOURCE_KERNEL, 'video_sources': VIDEO_SOURCES,
          'resolved_videos': {v: {**row, 'path': str(row['path'])} for v, row in RESOLVED_VIDEOS.items()}, 'index_validation': validation,
          'stages_01_05_executed': False, 'expected_questions': EXPECTED_QUESTIONS,
          'examples': EXAMPLE_RESULTS, 'chat_transcript': str(DEMO_CHAT.saved_path),
          'chat_turns': DEMO_CHAT.turns, 'raw_llm_calls': RAW_LLM_CALLS,
          'retrieval_diagnostics': DEMO_REPORT.get('retrieval_diagnostics', []),
          'session_seconds': elapsed, 'timings': list(DEMO_TIMES.values())}
write_json(OUTPUT_ROOT / 'verification.json', report)
print('Verification saved | examples:', len(EXAMPLE_RESULTS), '| chat turns:', len(DEMO_CHAT.turns),
      '| elapsed seconds:', elapsed)
'''


def build():
    expected = json.loads(EXPECTED.read_text(encoding='utf-8'))['questions']
    capture = (demo.fast.CODE / 'demo_capture.py').read_text(encoding='utf-8')
    config = demo.CONFIG.replace("VIDEO_IDS = ['L25_V037']", 'VIDEO_IDS = []')
    config = config.replace('VIDEO_SOURCES = []', 'VIDEO_SOURCES = ' + repr([VIDEO_PATH]))
    config = config.replace("CHAT_MODE = 'auto'", "CHAT_MODE = 'scripted'")
    config += '\nOUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\nDEMO_REPORT, DEMO_TIMES = {}, {}\n'
    config += f'exec({capture!r})\n_demo_capture = DemoNotebookCapture(_capture_ipython(), '
    config += '"/kaggle/working/demo_pipeline.executed.ipynb", {})\n'
    config += f'QUESTIONS = {[q["question"] for q in expected]!r}\nEXPECTED_QUESTIONS = {expected!r}\n'
    config += f'VERIFICATION_REF = {REF!r}\nSOURCE_KERNEL = {PARENT!r}\n'
    config += 'assert torch.cuda.is_available() and torch.cuda.device_count() == 2\n'
    search, answer = demo.backend_sources()
    runtime = (demo.fast.CODE / 'demo_runtime.py').read_text(encoding='utf-8')
    runtime += '\n\n' + (demo.fast.CODE / 'demo_chat.py').read_text(encoding='utf-8')
    ollama = demo.selected_source(demo.fast.CODE / 'demo_runtime.py', {'start_demo_ollama'})
    cells = [cell('markdown', '# Real LLM attribution and chat verification\n\nReuse L25_V008; stages 01–05 are skipped.'),
             cell('code', config),
             cell('code', '%pip install -q "onnxruntime>=1.20,<2" "huggingface-hub==0.36.0" "ipywidgets>=8,<9"'),
             cell('code', (demo.fast.CODE / 'common.py').read_text(encoding='utf-8'), hidden=True),
             cell('code', (demo.fast.CODE / 'production_runtime.py').read_text(encoding='utf-8'), hidden=True),
             cell('code', (demo.fast.CODE / 'demo_sources.py').read_text(encoding='utf-8'), hidden=True),
             cell('code', runtime, hidden=True),
             cell('code', 'app_search = embedded_module("verify_app_search", ' + repr(search)
                  + ', {"DEMO_STORAGE": DemoStorage()})\napp_answer = embedded_module("verify_app_answer", '
                  + repr(answer) + ')\n' + f'OLLAMA_START_SOURCE = {ollama!r}'),
             cell('code', SETUP), cell('code', QUERIES), cell('code', 'demo_chat_start()'), cell('code', FINISH)]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    for c in cells:
        if c['cell_type'] == 'code' and not ''.join(c['source']).startswith('%'):
            ast.parse(''.join(c['source']))
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    production.write(DIRECTORY / 'verify.ipynb', notebook)
    meta = {'id': REF, 'title': SLUG, 'code_file': 'verify.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
            'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': production.DATASETS,
            'kernel_sources': [PARENT], 'model_sources': [], 'competition_sources': []}
    production.validate_metadata(meta)
    production.write(DIRECTORY / 'kernel-metadata.json', meta)
    print('Built query-only verification:', REF)
    return DIRECTORY


def collect(observed):
    """Download only the four requested artifact kinds; account from worker times."""
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
    evidence = production.verify_remote(DIRECTORY, observed=observed)
    if evidence['status'] not in {'COMPLETE', 'ERROR', 'CANCELLED'}:
        raise RuntimeError('Only collect terminal output')
    api = KaggleApi()
    api.authenticate()
    request = ApiListKernelSessionOutputRequest()
    request.user_name, request.kernel_slug, request.page_size = production.OWNER, SLUG, 200
    saved = []
    with api.build_kaggle_client() as client:
        while True:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            if not request.page_token and response.log:
                (DEST / (SLUG + '.log')).write_text(response.log, encoding='utf-8')
                saved.append(SLUG + '.log')
            for item in response.files or []:
                name = Path(item.file_name).name
                if name not in {'demo_pipeline.executed.ipynb', 'verification.json', SLUG + '.log'} \
                        and not re.fullmatch(r'chat_session_\d{8}_\d{6}_\d+\.json', name):
                    continue
                target = DEST / name
                with requests.get(item.url, stream=True, timeout=(30, 60)) as reply:
                    reply.raise_for_status()
                    with target.open('wb') as stream:
                        for block in reply.iter_content(1 << 20):
                            stream.write(block)
                if name not in saved:
                    saved.append(name)
            if not response.next_page_token:
                break
            request.page_token = response.next_page_token
    log_path = DEST / (SLUG + '.log')
    log = log_path.read_text(encoding='utf-8', errors='replace') if log_path.exists() else ''
    if log.lstrip().startswith('['):
        timestamps = [float(e['time']) for e in json.loads(log) if isinstance(e.get('time'), (int, float))]
    else:
        timestamps = [float(m[1]) for m in re.finditer(r'^\s*(\d+(?:\.\d+)?)\s', log, re.M)]
    rows = json.loads((production.OUTPUT / 'gpu-ledger.json').read_text(encoding='utf-8'))
    row = next(r for r in rows if r['ref'] == REF)
    row.update(evidence, downloaded_to=str(DEST))
    if timestamps:
        seconds = max(timestamps)
        row.update(actual_kernel_log_seconds=seconds, actual_allocated_t4_hours=seconds * row['devices'] / 3600,
                   runtime_basis='Kaggle timed worker log through final conversion event; two allocated T4s; queue excluded')
    production.write(production.OUTPUT / 'gpu-ledger.json', rows)
    path = DEST / 'verification.json'
    report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'examples': [], 'chat_turns': []}
    errors = []
    notebook_path = DEST / 'demo_pipeline.executed.ipynb'
    if notebook_path.exists():
        notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
        errors = [{'cell': i, 'name': o.get('ename'), 'error': o.get('evalue')}
                  for i, c in enumerate(notebook['cells']) for o in c.get('outputs', []) if o.get('output_type') == 'error']
    report.update(status=evidence['status'], privacy_verified=True, downloaded_files=saved, notebook_errors=errors,
                  worker_seconds=row.get('actual_kernel_log_seconds'), actual_t4_hours=row.get('actual_allocated_t4_hours'),
                  ledger_total_t4_hours=sum(production.charged_hours(r) for r in rows if r.get('submitted')),
                  collected_utc=datetime.now(timezone.utc).isoformat())
    production.write(path, report)
    print('Collected:', saved, '| status:', evidence['status'], '| errors:', errors)
    print('T4 hours:', report['actual_t4_hours'], '| ledger total:', report['ledger_total_t4_hours'])


def monitor():
    # The launch already observed QUEUED. Never repoll sooner than sixty seconds.
    while True:
        time.sleep(60)
        observed = production.status(REF)
        print(datetime.now(timezone.utc).isoformat(), observed[0], flush=True)
        rows = json.loads((production.OUTPUT / 'gpu-ledger.json').read_text(encoding='utf-8'))
        row = next(r for r in rows if r['ref'] == REF)
        row.update(status=observed[0], last_observed_utc=datetime.now(timezone.utc).isoformat())
        production.write(production.OUTPUT / 'gpu-ledger.json', rows)
        if observed[0] in {'COMPLETE', 'ERROR', 'CANCELLED'}:
            collect(observed)
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'launch', 'monitor'])
    args = parser.parse_args()
    if args.action == 'build':
        build()
    elif args.action == 'launch':
        production.launch(DIRECTORY, TIMEOUT_SECONDS)
    else:
        monitor()
