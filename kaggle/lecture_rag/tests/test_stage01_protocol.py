"""Boundary protocol tests: neutral cuts, tolerance and one-to-one matching."""
import importlib.util
from pathlib import Path

FILE=Path(__file__).resolve().parents[1]/'stage01/evaluate.py'
spec=importlib.util.spec_from_file_location('stage01_evaluation',FILE)
evaluation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

def test_l25_neutral_layout_ads_and_reveals():
    events=[{'time_sec':10.,'label':'S'},{'time_sec':20.,'label':'L'},
            {'time_sec':30.,'label':'C'},{'time_sec':40.,'label':'R'},
            {'time_sec':50.,'label':'SC'}]
    # A reveal split is an FP; layout/ad edges are neutral. The missing scroll
    # is an FN. The duplicate at the same S follows the thesis's legacy policy.
    assert evaluation.score_l25([9.,10.,20.,30.,40.,60.5],events,[60.])==(1,1,1)

def test_lpm_distance_matching_is_one_to_one():
    assert evaluation.score_lpm([10.,20.],[9.,10.2,23.],1.)==(1,2,1)
    assert evaluation.score_lpm([10.,20.],[9.,10.2,23.],3.)==(2,1,0)

def test_missing_video_is_rejected_instead_of_partial_score(tmp_path):
    try:
        evaluation.interval_file(tmp_path,'missing_video')
    except FileNotFoundError as error:
        assert 'never score a partial split' in str(error)
    else:
        raise AssertionError('Incomplete evaluation was accepted')
