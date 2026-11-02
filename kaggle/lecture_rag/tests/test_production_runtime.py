"""Run lifecycle and canary contracts using local fixtures and stubs only."""
import ast
from pathlib import Path
import sys
import tempfile
import types
import zipfile
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from test_pipeline import load_code
from test_port_v3 import config_values, fast


def runtime():
    module = load_code('common')
    source = fast.CODE / 'production_runtime.py'
    exec(compile(source.read_text(encoding='utf8'), str(source), 'exec'), module.__dict__)
    return module


def cuda(count=2):
    return types.SimpleNamespace(cuda=types.SimpleNamespace(
        device_count=lambda: count, is_available=lambda: count > 0,
        get_device_properties=lambda i: types.SimpleNamespace(name='T4', total_memory=15360 * 2**20),
        reset_peak_memory_stats=Mock(), max_memory_allocated=lambda i: 100 * 2**20,
        max_memory_reserved=lambda i: 200 * 2**20))


def test_preflight_resolves_all_inputs_and_reports_before_model_loading(tmp_path):
    m = runtime()
    for v in ('V', 'E'):
        (tmp_path / (v + '.mp4')).write_bytes(b'fixture')
    with patch.object(m.shutil, 'which', return_value='/bin/ffmpeg'), \
         patch.object(m, 'package_versions', return_value={'torch': 'stub'}):
        result = m.startup_preflight(['V', 'E'], {'E': 'en'}, tmp_path / 'out',
                                     tmp_path, torch_module=cuda())
    assert result['passed'] and result['languages'] == {'V': 'vi', 'E': 'en'}
    assert len(result['devices']) == 2 and result['devices'][1]['total_mib'] == 15360
    assert m.read_json(tmp_path / 'out/preflight.json')['packages'] == {'torch': 'stub'}


@pytest.mark.parametrize('failure', ['cuda', 'duplicates', 'ffmpeg', 'packages', 'disk', 'language'])
def test_preflight_failures_are_clear_and_persisted(tmp_path, failure):
    m = runtime()
    (tmp_path / 'V.mp4').write_bytes(b'x')
    if failure == 'duplicates':
        (tmp_path / 'nested').mkdir()
        (tmp_path / 'nested/V.mp4').write_bytes(b'x')
    with patch.object(m.shutil, 'which', return_value=None if failure == 'ffmpeg' else '/bin/tool'), \
         patch.object(m.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=0 if failure == 'disk' else 10*2**30)), \
         patch.object(m, 'package_versions', return_value={'torch': 'not-installed' if failure == 'packages' else 'stub'}):
        with pytest.raises(RuntimeError, match='STARTUP PREFLIGHT FAILED'):
            m.startup_preflight(['V'], {'V': 'xx'} if failure == 'language' else {},
                                tmp_path / 'out', tmp_path, torch_module=cuda(1 if failure == 'cuda' else 2))
    assert not m.read_json(tmp_path / 'out/preflight.json')['passed']


def test_stage_timings_are_durable_before_video_finishes(tmp_path):
    m = runtime()
    p = m.ProductionProgress(tmp_path, ['V'], 'recipe')
    assert p.timed('01', 'V', lambda: 5) == 5
    row = m.read_json(tmp_path / 'progress.json')['videos'][0]
    assert row['status'] == 'running' and row['stage_timings']['01']['seconds'] >= 0
    with pytest.raises(ValueError):
        p.timed('02', 'V', lambda: (_ for _ in ()).throw(ValueError('fixture failure')))
    row = m.read_json(tmp_path / 'fast_pipeline_timings.json')['videos'][0]
    assert row['status'] == 'failed' and 'fixture failure' in row['error']
    p.update('V', status='completed', error=None, retries=2, ocr_failed=1)
    assert m.read_json(tmp_path / 'progress.json')['videos'][0]['retries'] == 2


def test_completed_skip_requires_recipe_source_and_valid_output(tmp_path):
    m = runtime()
    source = tmp_path / 'V.mp4'
    source.write_bytes(b'video')
    marker = {'recipe': 'r', 'source_sha256': m.file_sha256(source), 'index_artifact_id': 'index'}
    m.write_json(tmp_path / 'V/completed.json', marker)
    with patch.object(m, 'validate_video_output', return_value=marker) as validate:
        assert m.completed_video(tmp_path, 'V', 'r', source) == marker
        assert m.completed_video(tmp_path, 'V', 'changed', source) is None
        source.write_bytes(b'changed')
        assert m.completed_video(tmp_path, 'V', 'r', source) is None
        assert validate.call_count == 1
    source.write_bytes(b'video')
    with patch.object(m, 'validate_video_output', side_effect=ValueError('corrupt')):
        assert m.completed_video(tmp_path, 'V', 'r', source) is None


def test_checkpoint_restores_partial_caches_and_rejects_nested_paths(tmp_path):
    m = runtime()
    m.write_json(tmp_path / 'input/V/02_visual/_cache/answer.json', {'text': 'durable'})
    m.restore_checkpoint(tmp_path / 'input', tmp_path / 'out')
    assert m.read_json(tmp_path / 'out/V/02_visual/_cache/answer.json')['text'] == 'durable'
    with pytest.raises(ValueError, match='separate'):
        m.restore_checkpoint(tmp_path / 'out', tmp_path / 'out/nested')


def test_qwen_answer_cache_survives_mid_batch_death(tmp_path):
    m = load_code('visual')
    stage = m.begin_stage(tmp_path, 1)
    intervals = []
    for i, color in enumerate(('red', 'blue')):
        image = stage / f'{i}.jpg'
        Image.new('RGB', (64, 64), color).save(image)
        intervals.append({'segment_id': f'V:{i}', 'sequence': i, 'image_path': image.name,
                          'image_sha256': m.file_sha256(image), 'start_sec': i*10, 'end_sec': (i+1)*10})
    m.write_json(stage / 'intervals.json', intervals)
    m.finish_stage(stage, 1, 'V', {'sha256': 's', 'duration_sec': 20}, {})
    answer = {'text': 'durable answer', 'ocr_quality': {'status': 'available', 'reasons': [], 'evidence_excluded': False}}
    def die(images, language, *, on_answer):
        on_answer(0, answer)
        raise SystemExit('session death')
    with patch.object(m, 'load_qwen_ocr', return_value=die):
        with pytest.raises(SystemExit):
            m.run_visual_extraction(stage, tmp_path, 'V', ocr_engine='qwen', ocr_device='cpu')
    saved = list((tmp_path / '02_visual/_cache').rglob('*.qwen.json'))
    assert len(saved) == 1 and m.read_json(saved[0]) == answer
    m._SHARED_MODELS.clear()
    def resume(images, language, *, on_answer):
        assert len(images) == 1, 'only the unfinished image is generated'
        on_answer(0, answer)
        return [answer]
    with patch.object(m, 'load_qwen_ocr', return_value=resume):
        _, manifest, records = m.run_visual_extraction(stage, tmp_path, 'V', ocr_engine='qwen', ocr_device='cpu')
    assert len(records) == 2 and manifest['ocr_failed_count'] == 0


@pytest.mark.parametrize('enough', [True, False])
def test_delivery_and_checkpoint_contracts_and_disk_gate(tmp_path, enough):
    m = runtime()
    root = tmp_path / 'root'
    for name in ('V/01_slides/slide.jpg', 'V/03_audio/audio/lecture.wav',
                 'V/02_visual/_cache/answer.json', 'V/03_audio/_cache/window.json',
                 'V/model.safetensors', 'V/source.mp4'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    with patch.object(m.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=10*2**30 if enough else 0)):
        result = m.package_production_outputs(root, tmp_path / 'archives')
    with zipfile.ZipFile(result['delivery']) as z:
        assert z.namelist() == ['V/01_slides/slide.jpg']
    if enough:
        with zipfile.ZipFile(result['checkpoint']) as z:
            assert 'V/02_visual/_cache/answer.json' in z.namelist()
            assert 'V/03_audio/audio/lecture.wav' in z.namelist()
            assert not any(n.endswith(('.safetensors', '.mp4')) for n in z.namelist())
    else:
        assert result['checkpoint'] is None and 'checkpoint skipped' in result['warning']


@pytest.mark.parametrize('free,passed', [(1536, True), (1535, False)])
def test_canary_uses_production_keys_and_margin(tmp_path, free, passed):
    m = runtime()
    config = config_values(fast.CONFIG)
    config['OUTPUT_ROOT'] = tmp_path
    qwen = Mock(return_value=[{'attempts': [{'generated_tokens': 2048}]}]*2)
    e5 = types.SimpleNamespace(encode=Mock(return_value=np.ones((32,384))))
    def command(args):
        if str(args[-1]).endswith('.png'):
            Image.new('RGB', (1280, 1024)).save(args[-1])
        return ''
    fake_st = types.SimpleNamespace(SentenceTransformer=Mock(return_value=e5))
    with patch.dict(sys.modules, {'torch': cuda(), 'sentence_transformers': fake_st}), \
         patch.object(m, 'gpu_memory_sample', return_value=[{'index':i,'used_mib':15360-free,'free_mib':free,'total_mib':15360} for i in range(2)]), \
         patch.object(m, 'run_command', side_effect=command), \
         patch.object(m, 'read_pcm_audio', return_value=np.zeros(16000, dtype=np.int16)), \
         patch.object(m, 'load_ocr_gate', create=True, return_value='gate'), \
         patch.object(m, 'shared_qwen_ocr', create=True, return_value=qwen), \
         patch.object(m, 'load_asr_model', create=True, return_value=('asr','pin','cuda','float16')), \
         patch.object(m, 'transcribe_window', create=True, return_value={}), \
         patch.object(m, 'crop_to_roi', create=True, side_effect=lambda image, roi: image):
        result = m.run_memory_canary(config, {'videos': {'hVU5Mw19V_c': 'stub.mp4'}})
    assert result['passed'] is passed
    assert ('asr', config['ASR_MODEL'], config['ASR_REVISION'], 'auto', 1) in m._SHARED_MODELS
    assert ('e5', config['EMBEDDING_MODEL'], config['EMBEDDING_REVISION'], 'cuda:1') in m._SHARED_MODELS
    assert qwen.call_args.kwargs['force_tokens'] == 2048
    assert e5.encode.call_args.kwargs['batch_size'] == 32
    assert (tmp_path / 'canary.json').is_file()


def test_builder_canary_does_not_run_queue_and_embeds_runtime(tmp_path):
    import json
    fast.build(tmp_path / 'canary.ipynb', canary=True)
    notebook = json.loads((tmp_path / 'canary.ipynb').read_text(encoding='utf8'))
    sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    assert fast.PREFLIGHT in sources and fast.RUN in sources
    assert (fast.CODE / 'production_runtime.py').read_text(encoding='utf8') in sources
    canary = next(c for c in sources if 'VIDEO_IDS =' in c)
    values = {'fingerprint': lambda data: 'recipe'}
    exec(canary, values)
    assert values['CANARY'] is True
    branch = next(n for n in ast.parse(fast.RUN).body if isinstance(n, ast.If))
    values.update(run_memory_canary=Mock(return_value={'passed':True}), preflight={})
    exec(compile(ast.Module(body=[branch],type_ignores=[]),'canary-branch','exec'), values)
    assert 'timings' not in values


def test_output_validation_checks_actual_vectors_images_lineage_and_bounds(tmp_path):
    m = runtime()
    source = {'sha256': 'a'*64, 'duration_sec': 10}
    parents = []
    for stage in range(1, 6):
        directory = m.begin_stage(tmp_path, stage)
        if stage == 1:
            Image.new('RGB', (16, 16)).save(directory / 'image.png')
            m.write_json(directory / 'intervals.json', [{'segment_id':'V:s', 'image_path':'image.png',
                'start_sec':0, 'end_sec':10, 'long_dynamic_review':True}])
        elif stage == 5:
            Image.new('RGB', (16, 16)).save(directory / 'image.png')
            m.write_json(directory / 'chunks.jsonl', {'chunk_id':'V:c', 'image_path':'image.png',
                                                    'start_sec':1, 'end_sec':2})
            # The production chunk contract is JSONL, not indented JSON.
            (directory / 'chunks.jsonl').write_text('{"chunk_id":"V:c","image_path":"image.png","start_sec":1,"end_sec":2}\n', encoding='utf8')
            v = np.zeros((1,384), dtype=np.float32)
            v[0,0] = 1
            np.save(directory / 'embeddings.npy', v)
        else:
            m.write_json(directory / 'fixture.json', {})
        lineage = parents[:1] if stage == 2 else parents if stage == 4 else parents[-1:] if stage == 5 else []
        parents.append(m.finish_stage(directory, stage, 'V', source, {}, lineage,
                                      extra={'chunk_count':1} if stage == 5 else None))
    result = m.validate_video_output(tmp_path, 'V')
    assert result['vectors'] == [1,384] and result['long_dynamic_review'] == ['V:s']
    np.save(directory / 'embeddings.npy', np.zeros((1,384), dtype=np.float32))
    m.finish_stage(directory, 5, 'V', source, {}, [parents[3]], extra={'chunk_count':1})
    with pytest.raises(ValueError, match='normalized'):
        m.validate_video_output(tmp_path, 'V')
