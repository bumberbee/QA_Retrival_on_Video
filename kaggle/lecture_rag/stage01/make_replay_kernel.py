"""Private CPU re-evaluation from frozen baseline state PTS, without full rescans.

Reconstruct the exact original JPEG-90 scan images with PyAV resizing, verify
legacy grouping parity, then evaluate changed grouping. Uses the previous
private kernel as an input. Large LPM videos remain attached on Kaggle.
"""
import json
from pathlib import Path
from make_kernel import ROOT

OUT=ROOT/'runs/stage01/replay-kernel'
RUNNER=r'''
import av, cv2, io, time, traceback
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
cv2.setNumThreads(1)
IDS=[f'L25_V{n:03d}' for n in range(1,11)]+['ALRJCeVT0fQ','BsXUWddl-as','DM4MWxnXOzM',
    'QPA4yLCbMw4','_YHDxyO9W8Q','uS8F-qRUeMM','gsDGPyBzrQM','V-4SZ4f7kMc']
files={p.stem:p for p in Path('/kaggle/input').rglob('*.mp4') if p.stem in IDS}
predictions={p.parent.name:p for p in Path('/kaggle/input').rglob('intervals.json')
             if p.parent.name in IDS and p.parent.parent.name=='baseline'}

def exact_scan_image(video,source,target):
    with av.open(str(video)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        pts=int((target+source['timeline_origin_sec'])/float(stream.time_base))
        container.seek(pts,stream=stream,backward=True,any_frame=False)
        width=min(1280,source['width'])
        height=max(2,round(source['height']*width/source['width']/2)*2)
        for frame in container.decode(stream):
            if frame.pts is None:continue
            timestamp=float(frame.pts*frame.time_base)-source['timeline_origin_sec']
            if timestamp+1e-6>=target:
                image=frame.to_image(width=width,height=height)
                buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=90);buffer.seek(0)
                return Image.open(buffer).convert('RGB')
    raise RuntimeError(f'Missing frame {target}')

def one(video):
    started=time.monotonic()
    try:
        baseline=read_json(predictions[video])
        states=sorted([s for i in baseline for s in i['reveal_states']],key=lambda s:s['source_state'])
        source=probe_video(files[video])
        images=[exact_scan_image(files[video],source,s['keyframe_time_sec']) for s in states]
        views=[reveal_view(im) for im in images]
        displays=[normalized_display_view(im) for im in images]
        variants={'baseline':baseline}
        for name,options in [('baseline_replayed',{}),('identity_fixed',{'display_views':displays}),
                              ('safe_fixed',{'replacement_safe':True})]:
            groups,audit=group_reveal_states(views,**options)
            variants[name]=[{'start_sec':states[g[0]]['start_sec'],
                             'end_sec':states[g[-1]]['end_sec']} for g in groups]
            write_json(Path('/kaggle/working/result')/name/video/'reveal_audit.json',audit)
        spans=lambda items:[(i['start_sec'],i['end_sec']) for i in items]
        parity=spans(variants['baseline'])==spans(variants['baseline_replayed'])
        for name,items in variants.items():
            write_json(Path('/kaggle/working/result')/name/video/'intervals.json',items)
        return {'video':video,'legacy_grouping_parity':parity,
                'seconds':round(time.monotonic()-started,3),
                'intervals':{k:len(v) for k,v in variants.items()}}
    except Exception:
        return {'video':video,'error':traceback.format_exc()}

if __name__=='__main__':
    rows=[]
    with ProcessPoolExecutor(4) as pool:
        for row in pool.map(one,IDS):
            rows.append(row);print(row,flush=True);write_json('/kaggle/working/runtime.json',rows)
    shutil.make_archive('/kaggle/working/result','zip','/kaggle/working/result')
'''

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    install="import subprocess,sys\nsubprocess.run([sys.executable,'-m','pip','install','-q','av>=14,<17','opencv-python-headless>=4.10,<4.12','Pillow>=10'],check=True)\n"
    source='\n\n'.join((ROOT/'notebook_code'/f).read_text(encoding='utf-8') for f in ('common.py','detection.py'))
    (OUT/'run.py').write_text(install+source+RUNNER,encoding='utf-8')
    metadata={'id':'thnhtrungnguynmif/stage01-regression-fixed-20261102',
              'title':'Stage01 regression fixed 20261102','code_file':'run.py',
              'language':'python','kernel_type':'script','is_private':True,
              'enable_gpu':False,'enable_tpu':False,'enable_internet':True,
              'dataset_sources':['thnhtrungnguynmif/l25a-lecture-videos-media','thnhtrungnguynmif/lpm-videos'],
              'kernel_sources':['thnhtrungnguynmif/stage01-regression-20261101'],
              'competition_sources':[]}
    (OUT/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(OUT)

if __name__=='__main__':
    main()
