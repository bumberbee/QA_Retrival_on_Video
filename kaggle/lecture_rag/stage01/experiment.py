"""Replayable CPU before/after experiment on the ten read-only extra videos.

System Python, with D:/FilmTools/Shotcut on PATH. One exact PTS scan is cached;
coarse signatures and only confirmed candidates' fine buffers are losslessly
stored as uint8. Cached replay reports processing separately from decode time.
No production config files or input data are modified.
"""
import argparse
import gzip
import json
import pickle
import shutil
import time
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs/stage01'
VIDEOS = Path('D:/thesis-data/mp4-extra')
OLD = Path('D:/thesis-data/run-20261020/export')

def load_detection():
    module = types.ModuleType('stage01')
    for name in ('common.py', 'detection.py'):
        file = ROOT/'notebook_code'/name
        exec(compile(file.read_text(encoding='utf-8'), str(file), 'exec'), module.__dict__)
    return module

def recommended(video):
    options = {'export_state_onsets':True, 'merge_same_content':True}
    if video == 'hVU5Mw19V_c':
        roi = (0.0, 0.08, 1.0, 0.96)
        options.update(board_options={'roi':roi, 'snapshot_sec':120.0},
                       keyframe_options={'roi':roi, 'mode':'board','complementary':True},
                       merge_reveals=False)
    elif video == 'y1CvB950VlU':
        options.update(reveal_thresholds={'replacement_safe':True},
                       keyframe_options={'roi':(0.26,0.08,0.99,0.57)})
    return options

def counts(intervals):
    lengths = [i['end_sec']-i['start_sec'] for i in intervals]
    return {'intervals':len(intervals), 'states':sum(len(i.get('reveal_states',[i])) for i in intervals),
            'under_5_sec':sum(t<5 for t in lengths), 'over_120_sec':sum(t>120 for t in lengths),
            'longest_sec':round(max(lengths),3)}

def run(video, variant='recommended'):
    m = load_detection()
    original_scan, original_detect = m.scan_frames, m.detect_stable_boundaries
    original_remove = m.shutil.rmtree
    cache_dir = RUNS/'cache'/video
    cache_file = cache_dir/'samples.pkl.gz'
    cache_dir.mkdir(parents=True,exist_ok=True)
    baseline_root = RUNS/'baseline'/video
    options = recommended(video)
    content_options = {'roi':options.get('board_options',options.get('keyframe_options',{})).get('roi',(0,0,1,1)),
                       'mode':'board' if video=='hVU5Mw19V_c' else 'generic'}
    # Every delete this interceptor suppresses is a _scan under our experiment.
    def preserve_scans(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved.name=='_scan' and resolved.is_relative_to(RUNS.resolve()):
            return
        return original_remove(path,*args,**kwargs)
    m.shutil = types.SimpleNamespace(**{n:getattr(m.shutil,n) for n in dir(m.shutil) if not n.startswith('_')})
    m.shutil.rmtree = preserve_scans
    cached = []
    def capture(*args, **kwargs):
        kwargs['content_options'] = content_options
        for sample in original_scan(*args,**kwargs):
            cached.append({k:v for k,v in sample.items() if k not in ('signature','fine')})
            cached[-1]['signature'] = (sample['signature']*255).round().astype('uint8')
            yield sample
    first = True
    def detect(samples,*args,**kwargs):
        nonlocal first
        result = original_detect(samples,*args,**kwargs)
        if first:
            by_time = {s['time_sec']:s for s in cached}
            for candidate in result[1]:
                by_time[candidate['time_sec']]['fine'] = [
                    (t,(sig*255).round().astype('uint8')) for t,sig in candidate['fine']]
            first = False
        return result
    cv2.setNumThreads(1)
    if not cache_file.exists():
        m.scan_frames, m.detect_stable_boundaries = capture, detect
        started = time.monotonic()
        _, manifest, intervals = m.run_slide_detection(VIDEOS/(video+'.mp4'),baseline_root,video)
        baseline_seconds = time.monotonic()-started
        with gzip.open(cache_file,'wb',compresslevel=1) as out:
            pickle.dump(cached,out,protocol=5)
        m.write_json(cache_dir/'timing.json',{'decode_and_baseline_sec':round(baseline_seconds,3)})
        m.write_json(cache_dir/'source.json',manifest['source'])
    else:
        with gzip.open(cache_file,'rb') as stream:
            cached=pickle.load(stream)
    # Images stay in the baseline scan directory. New runs use relative paths
    # pointing there; production keyframes are copied into their own stage dirs.
    scan_dir = baseline_root/'01_slides'
    def replay(video_path,source,directory,*args,records=None,**kwargs):
        for stored in cached:
            sample = {**stored,'signature':stored['signature'].astype('float32')/255,
                      'fine':[(t,s.astype('float32')/255) for t,s in stored.get('fine',[])]}
            # Absolute cached image paths are internal to the experiment only.
            sample['image_path'] = str(scan_dir/stored['image_path'])
            if records is not None:
                records.append({**stored,'image_path':sample['image_path']})
            yield sample
    m.scan_frames, m.detect_stable_boundaries = replay, original_detect
    source = m.read_json(cache_dir/'source.json')
    m.probe_video = lambda _:source
    config = options if variant=='recommended' else {'export_state_onsets':True,'reveal_thresholds':{'replacement_safe':True},'merge_same_content':True}
    target=(RUNS/variant/video).resolve()
    if not target.is_relative_to(RUNS.resolve()):
        raise ValueError('Unexpected experiment target')
    if target.exists():
        shutil.rmtree(target)
    started=time.monotonic()
    _, manifest, after = m.run_slide_detection(VIDEOS/(video+'.mp4'),RUNS/variant/video,video,**config)
    processing=time.monotonic()-started
    baseline = m.read_json(scan_dir/'intervals.json')
    old = m.read_json(OLD/video/'01_slides/intervals.json')
    timing=m.read_json(cache_dir/'timing.json')
    row={'video':video,'archived':counts(old),'baseline':counts(baseline),'after':counts(after),
         **timing,'replay_processing_sec':round(processing,3),'config':config,
         'source_sha256':source['sha256']}
    m.write_json(RUNS/variant/video/'measurement.json',row)
    print(json.dumps(row),flush=True)
    return row

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',nargs='*')
    parser.add_argument('--variant',default='recommended',choices=['recommended','strict'])
    parser.add_argument('--purge-cache',action='store_true',help='Delete this experiment scan/cache after each verified output')
    args=parser.parse_args()
    rows=[]
    for video in args.video or [p.stem for p in sorted(VIDEOS.glob('*.mp4'))]:
        rows.append(run(video,args.variant))
        if args.purge_cache:
            scan=(RUNS/'baseline'/video/'01_slides/_scan').resolve()
            if not scan.is_relative_to(RUNS.resolve()):
                raise ValueError('Unexpected scan cleanup path')
            shutil.rmtree(scan,ignore_errors=True)
            (RUNS/'cache'/video/'samples.pkl.gz').unlink(missing_ok=True)
    (RUNS/args.variant).mkdir(parents=True,exist_ok=True)
    (RUNS/args.variant/'measurements.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
