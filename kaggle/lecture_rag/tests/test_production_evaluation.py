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
from production_run import DATASETS, budget_bound, validate_metadata


def test_explicit_output_root_stays_isolated(tmp_path):
    assert experiment_dir('medium',tmp_path)==tmp_path/'medium'
    with pytest.raises(ValueError):
        experiment_dir('../outside',tmp_path)


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
