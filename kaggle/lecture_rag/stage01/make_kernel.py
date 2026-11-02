"""Build a private CPU regression kernel; no video downloads to the workstation.

Run from repo root, then: kaggle kernels push -p <printed directory>
The script embeds a snapshot so concurrent notebook work cannot alter the run.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/stage01/kernel"

RUNNER = r'''
import time, traceback
from concurrent.futures import ProcessPoolExecutor
import cv2
cv2.setNumThreads(1)
LPM_IDS = ['ALRJCeVT0fQ','BsXUWddl-as','DM4MWxnXOzM','QPA4yLCbMw4',
           '_YHDxyO9W8Q','uS8F-qRUeMM','gsDGPyBzrQM','V-4SZ4f7kMc']
VIDEOS = [f'L25_V{n:03d}' for n in range(1,11)] + LPM_IDS
files = {p.stem:p for p in Path('/kaggle/input').rglob('*.mp4') if p.stem in VIDEOS}

def one(video):
    started = time.monotonic()
    try:
        folder, manifest, baseline = run_slide_detection(
            files[video], Path('/kaggle/working/work')/video, video)
        members = sorted([s for item in baseline for s in item['reveal_states']],
                         key=lambda s:s['source_state'])
        images = [Image.open(folder/s['image_path']).convert('RGB') for s in members]
        views = [reveal_view(im) for im in images]
        displays = [normalized_display_view(im) for im in images]
        variants = {'baseline':baseline}
        for name, opts in [('identity',{'display_views':displays}),
                           ('safe',{'replacement_safe':True}),
                           ('safe_identity',{'replacement_safe':True,'display_views':displays})]:
            groups, audit = group_reveal_states(views, **opts)
            variants[name] = [{'start_sec':members[g[0]]['start_sec'],
                               'end_sec':members[g[-1]]['end_sec']} for g in groups]
            write_json(Path('/kaggle/working/result')/name/video/'reveal_audit.json',audit)
        for name, intervals in variants.items():
            write_json(Path('/kaggle/working/result')/name/video/'intervals.json',intervals)
        # Only this kernel's own temporary stage directory, verified by construction.
        shutil.rmtree(Path('/kaggle/working/work')/video)
        return {'video':video,'seconds':round(time.monotonic()-started,2),
                'intervals':{k:len(v) for k,v in variants.items()}}
    except Exception:
        return {'video':video,'error':traceback.format_exc()}

if __name__ == '__main__':
    from PIL import Image
    with ProcessPoolExecutor(4) as pool:
        rows = []
        for row in pool.map(one, VIDEOS):
            rows.append(row); print(row,flush=True)
            write_json('/kaggle/working/runtime.json',rows)
    shutil.make_archive('/kaggle/working/result','zip','/kaggle/working/result')
'''

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = "\n\n".join((ROOT / 'notebook_code' / name).read_text(encoding='utf-8')
                          for name in ('common.py', 'detection.py'))
    # Image is imported at module scope so fork and spawn workers both see it.
    install = "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'av>=14,<17', 'opencv-python-headless>=4.10,<4.12', 'Pillow>=10'], check=True)\n"
    (OUT/'run.py').write_text(install+source+'\nfrom PIL import Image\n'+RUNNER,encoding='utf-8')
    metadata = {
        'id':'thnhtrungnguynmif/stage01-regression-20261101',
        'title':'Stage01 regression 20261101', 'code_file':'run.py',
        'language':'python','kernel_type':'script','is_private':True,
        'enable_gpu':False,'enable_tpu':False,'enable_internet':True,
        'dataset_sources':['thnhtrungnguynmif/l25a-lecture-videos-media',
                           'thnhtrungnguynmif/lpm-videos'],
        'kernel_sources':[], 'competition_sources':[]}
    (OUT/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(OUT)

if __name__ == '__main__':
    main()
