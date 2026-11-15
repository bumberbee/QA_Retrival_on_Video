"""Fast synthetic tests for opt-in stage-01 content and temporal semantics."""
import types
import sys
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.source_paths import module_path

@pytest.fixture
def detection():
    module = types.ModuleType('stage01_test')
    for name in ('common.py','detection.py'):
        file=module_path(name)
        exec(compile(file.read_text(encoding='utf-8'),str(file),'exec'),module.__dict__)
    return module

def slide(lines):
    image=np.full((216,384),245,np.uint8)
    for index,text in enumerate(lines):
        cv2.putText(image,text,(30,50+35*index),cv2.FONT_HERSHEY_SIMPLEX,.55,20,1,cv2.LINE_AA)
    return image

def test_local_replacement_cannot_hide_behind_equal_or_increased_ink(detection):
    old=slide(['Retained title','print star 20','Retained footer'])
    new=slide(['Retained title','print Ga 100','Retained footer','more more more'])
    evidence=detection.local_content_evidence(old,new,grid=(16,9),minimum_pixels=8)
    assert evidence['deleted_regions']
    state={'views':[old,new],'mask':np.zeros(old.shape,bool)}
    decision,_=detection.reveal_decision(0,1,0,state,replacement_safe=True,deletion_minimum_pixels=8)
    assert decision=='split_deleted_or_replaced_content'

def test_addition_preserves_old_strokes_and_can_merge(detection):
    old=slide(['Retained title','First line'])
    new=slide(['Retained title','First line','Second line'])
    groups,audit=detection.group_reveal_states([old,new],replacement_safe=True)
    assert groups==[[0,1]]
    assert audit[0]['directional_retained']==1
    assert audit[0]['deleted_pixels']==0
    assert audit[0]['added_pixels']>0

def test_directional_text_veto_rejects_subset(detection):
    image=slide(['Same title','Same content'])
    state={'views':[image,image], 'mask':np.zeros(image.shape,bool),
           'texts':['alpha beta gamma delta epsilon','alpha beta gamma delta'], 'common':set()}
    decision,evidence=detection.reveal_decision(0,1,0,state,replacement_safe=True)
    assert decision=='split_text_deleted'
    assert evidence['text_directional_retained']==.8

def test_board_mask_covers_shirt_but_preserves_chalk(detection):
    pixels=np.full((216,384,3),(25,105,80),np.uint8)
    cv2.putText(pixels,'P(A)',(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(240,240,240),2)
    cv2.rectangle(pixels,(240,70),(330,210),(170,80,200),-1)
    mask=detection.board_presenter_mask(Image.fromarray(pixels))
    assert mask[120,280]
    assert not mask[50,30]
    assert .1<mask.mean()<.3

def test_motion_only_is_ignored_but_writing_and_erasure_split(detection):
    old=slide(['Board topic'])
    written=slide(['Board topic','NEW EXAMPLE'])
    erased=slide(['Board topic','REPLACED'])
    records=[]
    for t in range(90):
        view=(old if t<25 else written if t<55 else erased).copy()
        mask=np.zeros(view.shape,bool)
        x=260+(t%4)*10
        view[80:200,x:x+40]=120
        mask[75:205,x-3:x+43]=True
        records.append({'time_sec':float(t),'content_view':view,'presenter_mask':mask})
    boundaries,audit=detection.board_content_splits(records,[0,90],minimum_pixels=12,snapshot_sec=0)
    assert len(boundaries)==4
    assert 25<=boundaries[1]<=35
    assert 55<=boundaries[2]<=65
    assert all(a['reason']=='local_board_change' for a in audit)

def test_periodic_snapshot_and_short_clip_guards(detection):
    records=[{'time_sec':float(t),'content_view':slide(['Unchanged'])} for t in range(310)]
    boundaries,audit=detection.board_content_splits(records,[0,310],snapshot_sec=120)
    assert len(boundaries)==4
    assert all(a['reason']=='periodic_content_snapshot' for a in audit)
    assert min(np.diff(boundaries))>=15
    assert detection.board_content_splits(records[:3],[0,3])[0]==[0,3]

def test_board_history_survives_erasure_hidden_during_snapshot(detection):
    old=np.full((216,384),245,np.uint8)
    cv2.putText(old,'old board example',(180,80),cv2.FONT_HERSHEY_SIMPLEX,.5,20,1,cv2.LINE_AA)
    records=[]
    for t in range(65):
        view=old.copy() if t<25 else np.full(old.shape,245,np.uint8)
        mask=np.zeros(old.shape,bool)
        if 10<=t<30:
            mask[:,170:]=True
        records.append({'time_sec':float(t),'content_view':view,'presenter_mask':mask})
    _,audit=detection.board_content_splits(records,[0,65],min_state_sec=5,snapshot_sec=20,
                                          check_every_sec=5,minimum_pixels=12)
    assert any(a['reason']=='local_board_change' and 30<=a['time_sec']<=40 for a in audit)

def test_early_writing_does_not_leave_candidate_permanently_ineligible(detection):
    empty=slide(['Topic'])
    written=slide(['Topic','Early writing'])
    records=[{'time_sec':float(t),'content_view':empty if t<5 else written} for t in range(70)]
    boundaries,audit=detection.board_content_splits(records,[0,70],snapshot_sec=0,minimum_pixels=12)
    assert len(boundaries)==3
    assert 15<=boundaries[1]<=25
    assert audit[0]['reason']=='local_board_change'

def test_masked_holes_do_not_become_false_board_strokes(detection):
    records=[]
    for t in range(90):
        view=np.full((216,384),90,np.uint8)
        cv2.putText(view,'Still chalk',(10,60),cv2.FONT_HERSHEY_SIMPLEX,.5,240,1)
        mask=np.zeros(view.shape,bool)
        x=180+(t%15)*3
        mask[70:210,x:x+70]=True
        view[mask]=180
        records.append({'time_sec':float(t),'content_view':view,'presenter_mask':mask})
    boundaries,audit=detection.board_content_splits(records,[0,90],snapshot_sec=0)
    assert boundaries==[0,90]
    assert not audit

def test_screen_identity_normalizes_views_and_rejects_replacement(detection):
    content=Image.fromarray(slide(['Probability','P A = 1 / 2','Exam page'])).convert('RGB')
    full=Image.new('RGB',(768,432),'black');full.paste(content.resize((760,424)),(4,4))
    pip=Image.new('RGB',(768,432),'#303040');pip.paste(content,(375,10))
    a=detection.normalized_display_view(full)
    b=detection.normalized_display_view(pip)
    same,_=detection.same_display_evidence(a,b,retained_threshold=.94)
    assert same
    replacement=Image.fromarray(slide(['Probability','P B = 9 / 5','Different page'])).convert('RGB')
    pip.paste(replacement,(375,10))
    assert not detection.same_display_evidence(a,detection.normalized_display_view(pip))[0]
    assert not detection.same_display_evidence(a,None)[0]

def test_reveal_after_camera_switch_uses_equivalent_anchor_in_new_view(detection):
    content=Image.fromarray(slide(['Probability','First line'])).convert('RGB')
    full=Image.new('RGB',(768,432),'black');full.paste(content.resize((760,424)),(4,4))
    pip=Image.new('RGB',(768,432),'#303040');pip.paste(content,(375,10))
    added=Image.fromarray(slide(['Probability','First line','New reveal'])).convert('RGB')
    final=Image.new('RGB',(768,432),'black');final.paste(added.resize((760,424)),(4,4))
    images=[pip,full,final]
    views=[detection.reveal_view(i) for i in images]
    displays=[detection.normalized_display_view(i) for i in images]
    groups,audit=detection.group_reveal_states(views,display_views=displays)
    assert groups==[[0,1,2]]
    assert audit[0]['identity_anchor_state']==1
    assert audit[1]['decision']=='merge_content_kept'

def test_keyframe_penalizes_occlusion_and_retains_complement(detection):
    view=slide(['Left evidence','Right evidence','Board formula'])
    cv2.putText(view,'Other half',(220,150),cv2.FONT_HERSHEY_SIMPLEX,.55,20,1,cv2.LINE_AA)
    left=np.zeros(view.shape,bool);left[:,180:]=True
    right=~left
    # Each candidate exposes a different half; sharpness outside content is irrelevant.
    records=[{'time_sec':2.,'sharpness':9999,'content_view':view,'presenter_mask':left},
             {'time_sec':4.,'sharpness':1,'content_view':view,'presenter_mask':right},
             {'time_sec':6.,'sharpness':99999,'content_view':view,'presenter_mask':np.ones(view.shape,bool)}]
    selected=detection.content_keyframes(records,0,8,mode='board',complementary=True)
    assert selected[0]['time_sec']!=6
    assert len(selected)==2
    assert {r['time_sec'] for r in selected}=={2,4}

def test_keyframe_transient_overlay_penalty_and_half_open_interval(detection):
    base=slide(['Settled code','print value'])
    popup=base.copy();popup[65:165,90:280]=80
    records=[{'time_sec':float(t),'sharpness':1,'content_view':popup if t==3 else base} for t in range(1,8)]
    selected=detection.content_keyframes(records,1,7,transient_weight=50)
    assert selected[0]['time_sec']!=3
    assert 1<=selected[0]['time_sec']<7

def test_legacy_picker_and_reveal_default_unchanged(detection):
    records=[{'time_sec':2,'sharpness':2},{'time_sec':5,'sharpness':99}]
    assert detection.pick_scanned_keyframe(records,0,6)==records[1]
    views=[slide(['Title','line']),slide(['Title','line','reveal'])]
    groups,audit=detection.group_reveal_states(views)
    assert groups==[[0,1]]
    assert set(audit[0])=={'previous_state','current_state','decision',
                          'old_content_pixels','content_retained','content_growth','image_change'}

def test_export_merged_evidence_starts_at_state_not_group(detection,tmp_path):
    source={'duration_sec':20.,'timeline_origin_sec':0.,'filename':'fixture.mp4','sha256':'fixture',
            'width':384,'height':216}
    candidates=[{'time_sec':5.},{'time_sec':10.}]
    def keyframe(video,source,start,end):
        return Image.fromarray(slide(['same','reveal'])).convert('RGB'),start+1,10.
    with mock.patch.object(detection,'probe_video',return_value=source), \
         mock.patch.object(detection,'sampled_frames',return_value=[]), \
         mock.patch.object(detection,'detect_stable_boundaries',return_value=([0,5,10,20],candidates,20)), \
         mock.patch.object(detection,'refine_boundary',side_effect=lambda v,s,c,*a:c['time_sec']), \
         mock.patch.object(detection,'select_keyframe',side_effect=keyframe), \
         mock.patch.object(detection,'group_reveal_states',return_value=([[0,1,2]],[])):
        _,_,items=detection.run_slide_detection('fixture.mp4',tmp_path/'new',scan='seek',export_state_onsets=True)
        item=items[0]
        assert item['start_sec']==0
        assert item['representative_content_onset_sec']==10
        assert [s['content_onset_sec'] for s in item['reveal_states']]==[0,5,10]
        assert [s['onset_sec'] for s in item['visual_evidence_windows']]==[0,5,10]
        _,_,legacy=detection.run_slide_detection('fixture.mp4',tmp_path/'legacy',scan='seek')
        assert 'visual_evidence_windows' not in legacy[0]
        assert 'content_onset_sec' not in legacy[0]['reveal_states'][0]

def test_content_options_reject_seek_and_conflicting_rois(detection,tmp_path):
    with pytest.raises(ValueError,match='single_pass'):
        detection.run_slide_detection('x',tmp_path,scan='seek',board_options={})
    with pytest.raises(ValueError,match='same content ROI'):
        detection.run_slide_detection('x',tmp_path,board_options={'roi':(0,0,1,1)},
                                      keyframe_options={'roi':(0,0,.5,.5)})
