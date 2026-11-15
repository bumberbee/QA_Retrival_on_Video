"""Fetch returned metadata/status through CLI without accessing authentication files."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent.parent/'runs/gpu'

def verify(ref):
    target=ROOT/'metadata'/ref.split('/')[-1]
    target.mkdir(parents=True,exist_ok=True)
    def cli(*args):
        result=subprocess.run(['kaggle',*args],capture_output=True,text=True,encoding='utf8',errors='replace',timeout=180)
        if result.returncode:raise RuntimeError(result.stderr[-2000:])
        return result.stdout
    cli('kernels','pull',ref,'-p',str(target),'-m')
    meta=json.loads((target/'kernel-metadata.json').read_text(encoding='utf8'))
    assert meta['is_private'] is True and meta['enable_gpu'] is True and meta['enable_internet'] is True
    status=cli('kernels','status',ref)
    return {'ref':ref,'is_private':meta['is_private'],'enable_gpu':meta['enable_gpu'],
            'enable_internet':meta['enable_internet'],'machine_shape':meta.get('machine_shape'),
            'docker_image':meta.get('docker_image'),'status':status.strip()}

if __name__=='__main__':
    ledger=json.loads((ROOT/'ledger.json').read_text(encoding='utf8'))
    refs=list(dict.fromkeys(row['ref'] for row in ledger))
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(verify,refs))
    (ROOT/'analysis/kernel_verification.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
    for row in rows:print(row['ref'],row['is_private'],row['status'].splitlines()[-1])
