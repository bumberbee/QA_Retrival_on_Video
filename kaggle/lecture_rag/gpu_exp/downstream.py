"""Wrap the existing CPU rebuild/evaluator with a signed transcript override.

No production source edits. The same frozen CPU notebook sources are used for every
system, and ASR post-correction is off to isolate recognition differences.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import sys
import types
from unittest.mock import patch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
sys.path.insert(0,str(ROOT/'eval'))
import offline_common as common
import rebuild_offline as rebuild
import eval_retrieval as evaluate


def freeze_sources():
    dest=BASE/'cpu_snapshot'
    dest.mkdir(exist_ok=True)
    for p in (ROOT/'notebook_code').glob('*.py'):
        if not (dest/p.name).exists():shutil.copy2(p,dest/p.name)
    return dest


def frozen_loader(name):
    module=types.ModuleType('gpu_offline_'+name)
    extras=['textclean','adfilter'] if name=='alignment' else []
    for f in ['common',*extras,name]:
        p=BASE/'cpu_snapshot'/(f+'.py')
        exec(compile(p.read_text(encoding='utf8'),str(p),'exec'),module.__dict__)
    return module


def freeze_backend():
    """Load a read-only copy so concurrent repository edits cannot change the comparator."""
    target=BASE/'cpu_snapshot/backend_retrieval.py'
    if not target.exists():
        live=common.REPO/'backend/app/services/retrieval.py'
        baseline=ROOT/'runs/gpu/downstream/controlled-v2-medium-eval/retrieval.json'
        expected=common.read_json(baseline)['backend_sha256']
        if hashlib.sha256(live.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Backend changed before its controlled snapshot was captured')
        shutil.copy2(live,target)
    tree=ast.parse(inspect.getsource(common.load_backend))
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='path' for t in n.targets))
    assignment.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(target))],keywords=[])
    ast.fix_missing_locations(tree)
    namespace=dict(common.__dict__)
    exec(compile(tree,str(target), 'exec'),namespace)
    common.load_backend=namespace['load_backend']
    evaluate.load_backend=common.load_backend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--systems',nargs='+',required=True)
    p.add_argument('--export-root',type=Path,default=Path('D:/thesis-data/run-20261020/export'))
    p.add_argument('--videos',nargs='+',default=['hVU5Mw19V_c','v2YTz8ToEag','y1CvB950VlU','L25_V011','cs_0001'])
    p.add_argument('--prefix',default='e1')
    args=p.parse_args()
    common.EXPERIMENTS=ROOT/'runs/gpu/downstream'
    snapshot=freeze_sources()
    freeze_backend()
    rebuild.load_notebook=frozen_loader
    original_prepare=rebuild.prepare_inputs
    questions=common.read_json(ROOT/'eval/questions_dev.json')
    questions['questions']=[q for q in questions['questions'] if q['video'] in args.videos and q['type'] in {'speech','term','both'}]
    qpath=ROOT/'runs/gpu/downstream/questions_asr.json'
    common.write_json(qpath,questions)
    captured_config=BASE/'cpu_snapshot/controlled_config.json'
    if not captured_config.exists():
        baseline=ROOT/'runs/gpu/downstream/controlled-v1-exported/rebuild_report.json'
        config=common.read_json(baseline)['config'] if baseline.exists() else common.read_json(ROOT/'eval/production_config.json')
        common.write_json(captured_config,config)
    config=common.read_json(captured_config)
    config['asr_correction']='off'
    # The concurrent phase-3 agent may add defaults unsupported by our frozen sources.
    rebuild.production_config=lambda:dict(config)
    config_path=ROOT/'runs/gpu/downstream/asr_controlled_config.json'
    common.write_json(config_path,config)
    for system in args.systems:
        videos=[v for v in args.videos if (args.raw_root/system/v/'transcript.json').exists()]
        if system=='exported':videos=args.videos
        if not videos:continue
        def prepare(export,target,video,mp4_root,ffmpeg):
            inputs,provenance=original_prepare(export,target,video,mp4_root,ffmpeg)
            if system!='exported':
                override=args.raw_root/system/video/'transcript.json'
                audio_dir=inputs[2]
                transcript=common.read_json(override)
                missing_span_ids=0
                for i,span in enumerate(transcript.get('fallback_spans',[])):
                    if 'span_id' not in span:
                        span['span_id']=f'fallback_{i:07d}'
                        missing_span_ids+=1
                common.write_json(audio_dir/'transcript.json',transcript)
                manifest=common.read_json(audio_dir/'manifest.json')
                manifest['files']['transcript.json']=hashlib.sha256((audio_dir/'transcript.json').read_bytes()).hexdigest()
                manifest['word_count']=len(transcript['words'])
                manifest['parameters']['gpu_experiment_override']={
                    'system':system,'raw_transcript_sha256':hashlib.sha256(override.read_bytes()).hexdigest()}
                contract=frozen_loader('alignment')
                manifest['artifact_id']=contract.fingerprint({k:v for k,v in manifest.items() if k not in {'artifact_id','created_utc'}})
                common.write_json(audio_dir/'manifest.json',manifest)
                contract.load_stage(audio_dir,3,video)
                provenance.append({'transcript_override':str(override),'system':system,
                    'missing_fallback_span_ids_assigned':missing_span_ids,
                    'normalization_note':'Assigns schema IDs only; retains every text and timestamp.'})
            return inputs,provenance
        name=args.prefix+'-'+system
        with patch.object(rebuild,'prepare_inputs',prepare),patch.object(sys,'argv',[
            'rebuild_offline.py','--export-root',str(args.export_root),'--videos',*videos,
            '--name',name,'--config',str(config_path)]):rebuild.main()
        report=common.EXPERIMENTS/name/'rebuild_report.json'
        data=common.read_json(report)
        data['frozen_cpu_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.glob('*.py')}
        data['code_hash_note']='The harness code_sha256 records live repository files; frozen_cpu_sha256 identifies the notebook sources actually executed by this wrapper.'
        data['override_system']=system
        common.write_json(report,data)
        with patch.object(sys,'argv',['eval_retrieval.py','--index-root',str(common.EXPERIMENTS/name),
            '--videos',*videos,'--questions',str(qpath),'--name',name+'-eval']):evaluate.main()


if __name__=='__main__':main()
