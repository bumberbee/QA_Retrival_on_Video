"""Archive latest submitted sources; record unavailable historical CLI retrieval."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/'runs/gpu'

def archive(row):
    ref=row['ref'];version=row['version']
    if not row['latest']:
        return {'ref':ref,'version':version,'source_retrieval':'unavailable',
                'reason':'Kaggle CLI 2.2.4 version-suffixed pull returned HTTP 403. Logs/pins retained; exact old source not recovered.'}
    target=BASE/ref.split('/')[-1]/'submitted_versions'/f'v{version}'
    target.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(['kaggle','kernels','pull',ref,'-p',str(target),'-m'],
                          capture_output=True,text=True,encoding='utf8',errors='replace',timeout=180)
    if result.returncode:raise RuntimeError(result.stderr[-2000:]+result.stdout[-2000:])
    meta=json.loads((target/'kernel-metadata.json').read_text(encoding='utf8'))
    assert meta['is_private'] is True and meta['enable_gpu'] is True and meta['enable_internet'] is True
    code=target/meta['code_file']
    return {'ref':ref,'version':version,'private':True,'path':str(code),'source_retrieval':'latest source verified',
            'source_sha256':hashlib.sha256(code.read_bytes()).hexdigest()}

if __name__=='__main__':
    rows=json.loads((ROOT/'ledger.json').read_text(encoding='utf8'))
    for row in rows:row['latest']=row['version']==max(x['version'] for x in rows if x['ref']==row['ref'])
    with ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(archive,rows))
    (ROOT/'analysis/submitted_versions.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('Archived latest sources:',sum(r['source_retrieval']=='latest source verified' for r in result),
          'historical versions unavailable:',sum(r['source_retrieval']=='unavailable' for r in result))
