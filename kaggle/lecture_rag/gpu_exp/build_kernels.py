"""Generate private scripts with frozen historical baseline helpers, no credentials."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

BASE = Path(__file__).resolve().parent
DISC = BASE.parent / 'runs/gpu/discovery'
OWNER = 'thnhtrungnguynmif'
VIDEOS = ['hVU5Mw19V_c','v2YTz8ToEag','y1CvB950VlU','L25_V011','cs_0001']
GLOSSARIES = {
 'hVU5Mw19V_c':['Bayes','xác suất','biến cố','công thức xác suất đầy đủ'],
 'v2YTz8ToEag':['PPF','đường giới hạn khả năng sản xuất','chi phí cơ hội','khan hiếm','kinh tế học'],
 'y1CvB950VlU':['Python','len','for','print','end','sep','chuỗi','string','a1','a2'],
 'L25_V011':['Liên Hiệp Quốc','Hội đồng Bảo an','quyền bình đẳng','chủ quyền'],
 'cs_0001':['FIFO','MSCs','RSC','synchronous','asynchronous','partial order','message passing','decidable','undecidable']}
MODELS = [
 {'label':'medium','model_id':'Systran/faster-whisper-medium','revision':'08e178d48790749d25932bbc082711ddcfdfbc4f'},
 {'label':'large_v3','model_id':'Systran/faster-whisper-large-v3','revision':'edaa852ec7e145841d8ffdb056a99866b5f0a478'},
 {'label':'turbo','model_id':'dropbox-dash/faster-whisper-large-v3-turbo','revision':'0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf'}]
PHO={'label':'phowhisper','model_id':'vinai/PhoWhisper-large','revision':'b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb'}


def build(kind, name=None, model=None, videos=None, soft_limit=None):
    notebook = json.loads((DISC/'fast-pipeline-10-mixed-20261020.ipynb').read_text(encoding='utf8'))
    cells = [''.join(c['source']) for c in notebook['cells']]
    historical = cells[4] + '\n' + cells[8]
    config = {'videos':VIDEOS,'glossaries':GLOSSARIES,'soft_limit_seconds':5100,'models':MODELS}
    if kind=='asr_hints':
        spec = next(x for x in [*MODELS,PHO] if x['label']==model)
        config.update(soft_limit_seconds=1950 if model=='phowhisper' else 1650,
                      models=[dict(spec,label=model+'_hints',hints=True)])
    runner = 'asr_runner.py' if kind.startswith('asr') else 'focus_runner.py' if kind=='ocr_focus' else 'ocr_runner.py' if kind.startswith('ocr') else 'llm_runner.py' if kind.startswith('llm') else kind+'_runner.py'
    if kind=='ocr': config={'soft_limit_seconds':2550}
    if kind=='ocr_native':config={'soft_limit_seconds':1650,'only_native':True}
    if kind=='ocr_remote':config={'soft_limit_seconds':900,'only_native':True,'only_vl':True,
                                  'vl_remote':True,'vl_max_tokens':1536}
    if kind=='ocr_adapter':config={'soft_limit_seconds':800,'only_native':True,'only_vl':True,
                                  'vl_remote':True,'vl_max_tokens':1536,'vl_mask_keyword_adapter':True}
    if kind=='ocr_focus':config={'soft_limit_seconds':1050}
    if kind=='native':config.update(models=[],soft_limit_seconds=2850)
    if kind.startswith('llm'):
        frozen=BASE/'lecture-rag-gpu-llm-20261101/build-provenance.json'
        if frozen.exists():
            # Retry the exact recorded contexts even if another agent moves/rebuilds phase2 files.
            config=dict(json.loads(frozen.read_text(encoding='utf8'))['config'])
        else:
            data=json.loads((BASE.parent/'eval/questions_dev.json').read_text(encoding='utf8'))
            phase_path=BASE.parent/'runs/exp/phase2-final/retrieval.json'
            phase=json.loads(phase_path.read_text(encoding='utf8'))
            contexts={r['id']:r for r in phase['questions']}
            selected=[]
            for video in sorted({q['video'] for q in data['questions']}):
                rows=[q for q in data['questions'] if q['video']==video and q['type']!='unanswerable']
                chosen=[]
                for preferences in [('term','speech'),('numeric','slide'),('code','visual','both','speech')]:
                    q=next((q for t in preferences for q in rows if q['type']==t and q not in chosen),None)
                    if q:chosen.append(q)
                for q in chosen:
                    c=contexts[q['id']]
                    selected.append({k:q[k] for k in ['id','video','type','question','answer','key_terms']}|
                        {'context':c['context'],'image_segment':c['slides'][0]['segment_id'].split(':')[-1]})
            assert len(selected)==30
            config={'questions_payload':base64.b64encode(zlib.compress(json.dumps(selected,ensure_ascii=False).encode())).decode(),
                    'phase2_report_sha256':hashlib.sha256(phase_path.read_bytes()).hexdigest(),
                    'soft_limit_seconds':1650}
        if kind in ('llm_nonthinking','llm_nonthinking_raw'):
            config.update(ollama_models=['qwen3:4b'], explicit_no_think=True, run_vintern=False)
            if kind=='llm_nonthinking_raw':config['raw_closed_thinking']=True
    if videos:
        assert kind.startswith('asr') or kind=='native'
        config['videos']=videos
    if soft_limit is not None:config['soft_limit_seconds']=soft_limit
    slug = name or ('lecture-rag-gpu-'+kind.replace('_','-')+'-20261101')
    folder = BASE/slug
    folder.mkdir(parents=True,exist_ok=True)
    code = (BASE/'runtime.py').read_text(encoding='utf8')+'\n'
    if kind.startswith('asr') or kind=='native': code+=historical+'\n'
    if kind.startswith('ocr'):
        tree=ast.parse(cells[7]); node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='QWEN_SLIDE_PROMPTS' for t in n.targets))
        code+='\nQWEN_SLIDE_PROMPTS = '+repr(ast.literal_eval(node.value))+'\n'
    code+='\nCONFIG = '+repr(config)+'\n'
    if kind=='native':code+=(BASE/'asr_runner.py').read_text(encoding='utf8')+'\n'
    code+=(BASE/runner).read_text(encoding='utf8')
    (folder/'run.py').write_text(code,encoding='utf8')
    meta={'id':OWNER+'/'+slug,'title':slug,'code_file':'run.py','language':'python','kernel_type':'script',
          'is_private':True,'enable_gpu':True,'enable_tpu':False,'machine_shape':'NvidiaTeslaT4','enable_internet':True,
          'docker_image':'gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435',
          'dataset_sources':[OWNER+'/vn-lecture-test12',OWNER+'/l25a-lecture-videos-media',OWNER+'/slidespeech-train100'],
          'kernel_sources':[OWNER+'/fast-pipeline-10-mixed-20261020']}
    if kind.startswith('ocr'):meta['dataset_sources']=[OWNER+'/ocr-bench-vi-slides']
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    (folder/'build-provenance.json').write_text(json.dumps({'historical_notebook_sha256':hashlib.sha256((DISC/'fast-pipeline-10-mixed-20261020.ipynb').read_bytes()).hexdigest(), 'config':config},indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(folder)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['asr','asr_hints','ocr','ocr_native','ocr_remote','ocr_adapter','ocr_focus','native','llm','llm_nonthinking','llm_nonthinking_raw']);p.add_argument('--name');p.add_argument('--model',choices=[m['label'] for m in [*MODELS,PHO]]);p.add_argument('--videos',nargs='+',choices=VIDEOS);p.add_argument('--soft-limit',type=int);a=p.parse_args();build(a.kind,a.name,a.model,a.videos,a.soft_limit)
