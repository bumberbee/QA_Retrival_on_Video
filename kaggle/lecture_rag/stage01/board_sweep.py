"""Losslessly cache board content at the production scan's coarse PTS schedule.

Fine timestamps advance exactly as scan_frames; their unused pixel conversion
and JPEG export are omitted. Compare local sensitivity without rescanning.
This is a content-detection ablation, not the labelled slide F1 evaluator.
"""
import argparse
from collections import Counter
from pathlib import Path
import time

import av
import cv2
import numpy as np

from experiment import load_detection, RUNS, OLD, VIDEOS

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',default='hVU5Mw19V_c')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    m=load_detection()
    root=RUNS/'board_sweep'/args.video
    root.mkdir(parents=True,exist_ok=True)
    source=m.read_json(OLD/args.video/'01_slides/manifest.json')['source']
    cache=root/'content.npz'
    started=time.monotonic()
    if not cache.exists():
        times,views,masks=[],[],[]
        next_fine=next_sample=0.
        with av.open(str(VIDEOS/(args.video+'.mp4'))) as container:
            stream=container.streams.video[0]
            stream.thread_type='AUTO'
            stream.codec_context.skip_frame='NONREF'
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                timestamp=float(frame.pts*frame.time_base)-source['timeline_origin_sec']
                if timestamp<0 or timestamp+1e-6<next_fine:
                    continue
                if timestamp>=source['duration_sec']:
                    break
                next_fine=timestamp+1/8
                if timestamp+1e-6<next_sample:
                    continue
                image=frame.to_image(width=480,height=270)
                times.append(timestamp)
                views.append(m.reveal_view(image,(0,.08,1,.96)))
                masks.append(m.board_presenter_mask(image,(0,.08,1,.96)))
                next_sample=timestamp+1
        np.savez_compressed(cache,times=times,views=views,masks=masks)
        m.write_json(root/'decode_timing.json',{'decode_content_sec':time.monotonic()-started})
    data=np.load(cache)
    records=[{'time_sec':float(t),'content_view':v,'presenter_mask':mask}
             for t,v,mask in zip(data['times'],data['views'],data['masks'])]
    items=m.read_json(OLD/args.video/'01_slides/intervals.json')
    states=sorted([s for i in items for s in i['reveal_states']],key=lambda s:s['source_state'])
    primary=[s['start_sec'] for s in states]+[states[-1]['end_sec']]
    rows=[]
    for pixels in (8,12,18,24):
        started=time.monotonic()
        boundaries,audit=m.board_content_splits(records,primary,minimum_pixels=pixels)
        row={'minimum_pixels':pixels,'states':len(boundaries)-1,
             'local_events':sum(a['reason']=='local_board_change' for a in audit),
             'snapshot_events':sum(a['reason']=='periodic_content_snapshot' for a in audit),
             'seconds':time.monotonic()-started,'events':audit,'boundaries':boundaries}
        rows.append(row)
        print({k:v for k,v in row.items() if k not in ('events','boundaries')},flush=True)
    m.write_json(root/'sweep.json',rows)

if __name__=='__main__':
    main()
