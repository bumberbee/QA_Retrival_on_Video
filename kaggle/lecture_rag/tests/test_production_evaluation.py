"""CPU variant isolation, frozen legacy scope, private launch and budget unit tests."""
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from test_pipeline import load_code
from test_port_v3 import fast

sys.path.insert(0,str(fast.ROOT/'eval'))
from offline_common import experiment_dir
from rebuild_offline import prepare_inputs
from legacy_offline import LEGACY, aggregate, score_one, select_questions
from paired_comparison import paired
from production_run import DATASETS, budget_bound, validate_metadata, status, download, monitor, validate_outputs


def test_explicit_output_root_stays_isolated(tmp_path):
    assert experiment_dir('medium',tmp_path)==tmp_path/'medium'
    with pytest.raises(ValueError):
        experiment_dir('../outside',tmp_path)


def test_paired_comparison_keeps_regression_ids_and_rejects_different_scope():
    a = [{'id': 'q1', 'video': 'V', 'type': 'speech', 'hit1': 1, 'mrr': 1},
         {'id': 'q2', 'video': 'V', 'type': 'speech', 'hit1': 0, 'mrr': .5}]
    b = [{**a[0], 'hit1': 0, 'mrr': .5}, {**a[1], 'hit1': 1, 'mrr': 1}]
    result = paired(a, b)
    assert result['metrics']['hit1']['turbo_minus_medium'] == 0
    assert result['metrics']['hit1']['turbo_losses'] == ['q1']
    assert result['metrics']['hit1']['turbo_gains'] == ['q2']
    with pytest.raises(ValueError, match='scopes differ'):
        paired(a, b[:1])


def test_cpu_variants_keep_visual_parents_and_reject_wrong_source(tmp_path):
    m=load_code('alignment')
    video='V'
    export,medium=tmp_path/'new',tmp_path/'medium'
    source={'sha256':'same-video','duration_sec':10,'has_audio':False}
    parents=[]
    for stage in range(1,4):
        directory=m.begin_stage(export/video,stage)
        m.write_json(directory/'fixture.json',{'stage':stage})
        parents.append(m.finish_stage(directory,stage,video,source,{},parents=parents[:1] if stage==2 else []))
    directory=m.begin_stage(medium/video,3)
    m.write_json(directory/'transcript.json',{'words':[],'status':'no_audio'})
    m.finish_stage(directory,3,video,source,{'model':'medium'})
    inputs,provenance=prepare_inputs(export,tmp_path/'target',video,tmp_path,Path('ffmpeg'),medium)
    assert m.load_stage(inputs[0],1,video)['artifact_id']==parents[0]['artifact_id']
    assert m.load_stage(inputs[1],2,video)['artifact_id']==parents[1]['artifact_id']
    assert m.load_stage(inputs[2],3,video)['parameters']['model']=='medium'
    assert provenance[-1]['source_sha256']=='same-video'
    m.finish_stage(directory,3,video,{**source,'sha256':'other-video'},{})
    with pytest.raises(ValueError,match='cùng video'):
        prepare_inputs(export,tmp_path/'rejected',video,tmp_path,Path('ffmpeg'),medium)


def test_frozen_legacy_intersection_is_49_with_44_answerable():
    import hashlib,json
    assert hashlib.sha256(LEGACY.read_bytes()).hexdigest()=='027313f0bc4c0466cfb45fc78f6c1f0c46d1409a4f9501f5c92c64ff5e87f4f6'
    data=json.loads(LEGACY.read_text(encoding='utf8'))
    questions=select_questions(data,fast.CONFIG and ['L25_V008','L25_V011','L25_V037','hVU5Mw19V_c'])
    assert len(questions)==49 and sum(q['answer'] is not None for q in questions)==44


def test_legacy_score_keeps_exact_half_open_overlap_and_missing_term_gold():
    q={'id':'q','video':'V','q':'fixture','answer':[10,20]}
    found={'best_semantic':.5,'slides':[{'start_sec':0,'end_sec':10,'score':.02},
                                     {'start_sec':10,'end_sec':30,'score':.01}],
           'context_evidence':[{'start_sec':12,'end_sec':14,'text':'retained'}]}
    row=score_one(q,found,'context')
    assert row['hit1']==0 and row['hit3']==1 and row['mrr']==.5 and row['retained_time_overlap']==1
    assert aggregate([row])['context_recall'] is None


def test_private_metadata_and_budget_hard_gates():
    meta={'id':'thnhtrungnguynmif/new-20261102','is_private':True,'enable_gpu':True,
          'enable_tpu':False,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':DATASETS}
    validate_metadata(meta)
    with pytest.raises(RuntimeError,match='PUBLIC'):
        validate_metadata({**meta,'is_private':False})
    rows=[{'submitted':True,'timeout_seconds':1200}]*2
    assert budget_bound(rows,12000)==8
    with pytest.raises(RuntimeError,match='budget exceeded'):
        budget_bound(rows,12001)


def test_cli_status_accepts_enum_prefixed_terminal_state():
    with patch('production_run.cli', return_value='owner/slug has status "KernelWorkerStatus.COMPLETE"'):
        assert status('owner/slug')[0] == 'COMPLETE'


def test_download_rejects_destination_outside_task_before_remote_call(tmp_path):
    import json
    (tmp_path / 'kernel-metadata.json').write_text(json.dumps({'id': 'owner/new-20261102'}))
    with patch('production_run.verify_remote') as remote:
        with pytest.raises(RuntimeError, match='Downloads must stay'):
            download(tmp_path, Path('C:/outside-task'))
        remote.assert_not_called()


def test_download_rejects_unverified_privacy_before_output(tmp_path):
    import json
    meta = {'id': 'thnhtrungnguynmif/new-20261102', 'is_private': True,
            'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
            'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': DATASETS}
    (tmp_path / 'kernel-metadata.json').write_text(json.dumps(meta))
    with patch('production_run.OUTPUT', tmp_path), patch('production_run.cli') as remote:
        with pytest.raises(RuntimeError, match='privacy is unverified'):
            download(tmp_path, tmp_path / 'canary',
                     verification={'metadata': meta, 'privacy_verified': False})
        remote.assert_not_called()


def test_monitor_rejects_polling_faster_than_five_minutes():
    with pytest.raises(ValueError, match='at least 300'):
        monitor(Path('unused'), interval=299)


def test_monitor_reuses_terminal_observation_without_second_poll(tmp_path):
    import json
    meta = {'id': 'thnhtrungnguynmif/new-20261102', 'is_private': True,
            'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
            'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': DATASETS}
    (tmp_path / 'kernel-metadata.json').write_text(json.dumps(meta))
    (tmp_path / 'gpu-ledger.json').write_text(json.dumps([{'ref': meta['id'], 'privacy_verified': True}]))
    evidence = {'metadata': meta, 'privacy_verified': True, 'status': 'COMPLETE'}
    with patch('production_run.OUTPUT', tmp_path), \
            patch('production_run.time.monotonic', side_effect=[0, 300, 300]), \
            patch('production_run.status', return_value=('COMPLETE', 'terminal')) as poll, \
            patch('production_run.verify_remote', return_value=evidence) as verify, \
            patch('production_run.download') as output:
        assert monitor(tmp_path, tmp_path / 'full') == 'COMPLETE'
        poll.assert_called_once_with(meta['id'])
        verify.assert_called_once_with(tmp_path, observed=('COMPLETE', 'terminal'))
        output.assert_called_once_with(tmp_path, tmp_path / 'full', verification=evidence)


def test_production_validation_accepts_assets_and_rejects_changed_local_source(tmp_path):
    import json
    import numpy as np
    from PIL import Image
    import offline_common
    m = offline_common.load_notebook('production_runtime')
    root, media = tmp_path / 'export', tmp_path / 'media'
    media.mkdir()
    (media / 'V.mp4').write_bytes(b'exact-video')
    source = {'sha256': m.file_sha256(media / 'V.mp4'), 'duration_sec': 10}
    parents = []
    for stage in range(1, 6):
        d = m.begin_stage(root / 'V', stage)
        extras = {}
        if stage == 1:
            Image.new('RGB', (16, 16)).save(d / 'image.png')
            m.write_json(d / 'intervals.json', [{'segment_id': 'V:s', 'image_path': 'image.png',
                                               'start_sec': 0, 'end_sec': 10}])
        elif stage == 2:
            m.write_json(d / 'visual.json', [{'segment_id': 'V:s', 'ocr_status': 'read'}])
            extras = {'record_count': 1, 'ocr_failed_count': 0}
        elif stage == 3:
            m.write_json(d / 'transcript.json', {'words': [{'start_sec': 1, 'end_sec': 2}]})
            extras = {'word_count': 1}
        elif stage == 4:
            m.write_json(d / 'segments.json', [{'start_sec': 0, 'end_sec': 10}])
            extras = {'segment_count': 1}
        else:
            (d / 'lecture').mkdir()
            Image.new('RGB', (16, 16)).save(d / 'lecture/image.png')
            (d / 'chunks.jsonl').write_text(json.dumps({'chunk_id': 'c', 'start_sec': 0,
                'end_sec': 10, 'image_path': 'image.png'}) + '\n')
            vector = np.zeros((1, 384), dtype='float32'); vector[0, 0] = 1
            np.save(d / 'embeddings.npy', vector)
            extras = {'chunk_count': 1}
        parents.append(m.finish_stage(d, stage, 'V', source, {}, parents=parents, extra=extras))
    m.write_json(root / 'preflight.json', {'passed': True})
    m.write_json(root / 'progress.json', {'videos': [{'video_id': 'V', 'status': 'completed',
                   'error': None, 'stage_timings': {}}]})
    with patch('production_run.OUTPUT', tmp_path), patch.object(offline_common, 'VIDEOS', ['V']):
        report = validate_outputs(root, tmp_path / 'validation.json', media)
        assert report['passed'] and report['videos'][0]['vectors'] == [1, 384]
        (media / 'V.mp4').write_bytes(b'changed-video')
        with pytest.raises(RuntimeError, match='validation failed'):
            validate_outputs(root, tmp_path / 'rejected.json', media)
        rejected = json.loads((tmp_path / 'rejected.json').read_text())
        assert 'Exact local MP4 source hash differs' in rejected['failures'][0]
