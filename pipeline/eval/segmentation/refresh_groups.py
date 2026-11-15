"""Refresh grouping after an anchor fix, preserving frozen state/keyframe PTS.

Reconstruct JPEG-90 scan frames exactly as the Kaggle replay kernel (whose
baseline parity is checked). No second full scan or threshold change is needed.
Existing measured decode times are retained; refresh cost is reported separately.
"""
import argparse
import io
import shutil
import time
from pathlib import Path

import av
import cv2
from PIL import Image
import numpy as np

from experiment import load_detection, recommended, counts, RUNS, OLD, VIDEOS

def exact_scan_image(video,source,target):
    with av.open(str(video)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        pts=int((target+source['timeline_origin_sec'])/float(stream.time_base))
        container.seek(pts,stream=stream,backward=True,any_frame=False)
        width=min(1280,source['width'])
        height=max(2,round(source['height']*width/source['width']/2)*2)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if float(frame.pts*frame.time_base)-source['timeline_origin_sec']+1e-6>=target:
                image=frame.to_image(width=width,height=height)
                buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=90);buffer.seek(0)
                return Image.open(buffer).convert('RGB')
    raise RuntimeError(f'Missing frame {target}')

def refresh(video):
    m=load_detection();cv2.setNumThreads(1)
    target=(RUNS/'recommended'/video).resolve()
    if not target.is_relative_to(RUNS.resolve()):
        raise ValueError('Unexpected refresh target')
    previous=m.read_json(target/'measurement.json')
    state_root=target/'01_slides' if video=='y1CvB950VlU' else RUNS/'baseline'/video/'01_slides'
    frozen=m.read_json(state_root/'intervals.json')
    states=sorted([s for i in frozen for s in i['reveal_states']],key=lambda s:s['source_state'])
    manifest=m.read_json(state_root/'manifest.json');source=manifest['source']
    options=recommended(video)
    # For the editor, preserve the already evaluated ROI-selected keyframe.
    options.pop('keyframe_options',None)
    folder=(RUNS/'refresh_frames'/video).resolve()
    if not folder.is_relative_to(RUNS.resolve()):
        raise ValueError('Unexpected refresh-frame path')
    folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();samples=[]
    for number,state in enumerate(states):
        path=folder/f'{number:05d}.png'
        samples.append({'time_sec':state['keyframe_time_sec'],
                        'sharpness':state['sharpness_score'],'image_path':str(path.resolve()),
                        'signature':np.zeros((2,135,240),np.uint8)})
    original_pick=m.pick_scanned_keyframe
    last_image=None
    def lazy_pick(records,start,end):
        nonlocal last_image
        if last_image is not None:
            last_image.unlink(missing_ok=True)
        picked=original_pick(records,start,end)
        if picked is not None:
            image=exact_scan_image(VIDEOS/(video+'.mp4'),source,picked['time_sec'])
            last_image=folder/Path(picked['image_path']).name
            image.save(last_image)
        return picked
    def scan(*args,records=None,**kwargs):
        for sample in samples:
            if records is not None:
                records.append(sample)
            yield {**sample,'signature':sample['signature'].astype('float32'),'fine':[]}
    def fixed(samples,*args,**kwargs):
        list(samples)
        return ([s['start_sec'] for s in states]+[states[-1]['end_sec']],
                [{'time_sec':s['start_sec'],'fine':[]} for s in states[1:]],
                manifest['sample_count'])
    m.scan_frames,m.detect_stable_boundaries=scan,fixed
    m.pick_scanned_keyframe=lazy_pick
    m.probe_video=lambda _:source
    shutil.rmtree(target/'01_slides')
    directory,new_manifest,intervals=m.run_slide_detection(VIDEOS/(video+'.mp4'),target,video,**options)
    if video=='y1CvB950VlU':
        qualities={s['source_state']:s for s in states}
        for item in intervals:
            for state in item['reveal_states']:
                for field in ('content_quality','complementary_keyframes'):
                    if field in qualities[state['source_state']]:
                        state[field]=qualities[state['source_state']][field]
        m.write_json(directory/'intervals.json',intervals)
        core={'schema_version','pipeline_version','stage','video_id','source','parameters',
              'parents','files','artifact_id','created_utc'}
        m.finish_stage(directory,1,video,source,{**recommended(video),'frozen_state_replay':True},
                       extra={k:v for k,v in new_manifest.items() if k not in core})
    previous.update(after=counts(intervals),group_refresh_sec=round(time.monotonic()-started,3))
    m.write_json(target/'measurement.json',previous)
    for file in folder.iterdir():
        if file.is_file():
            file.unlink()
    folder.rmdir()
    print(video,previous['after'],'refresh_sec',previous['group_refresh_sec'],flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',nargs='*')
    args=parser.parse_args()
    for video in args.video or [p.stem for p in sorted(VIDEOS.glob('*.mp4')) if p.stem!='hVU5Mw19V_c']:
        refresh(video)

if __name__=='__main__':
    main()
