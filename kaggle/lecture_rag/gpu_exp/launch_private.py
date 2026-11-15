"""Launch an authorized private CLI experiment within the two-kernel/eight-hour limits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

RUNS = Path(__file__).resolve().parent.parent / 'runs/gpu'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--timeout',type=int,required=True)
    args=p.parse_args()
    meta=json.loads((args.directory/'kernel-metadata.json').read_text(encoding='utf8'))
    assert meta['is_private'] is True and meta['enable_gpu'] is True
    assert meta['enable_internet'] is True and meta['machine_shape']=='NvidiaTeslaT4'
    assert meta['id'].startswith('thnhtrungnguynmif/')
    ledger_path=RUNS/'ledger.json'
    rows=json.loads(ledger_path.read_text(encoding='utf8'))
    active=[r for r in rows if r['status']=='RUNNING']
    for row in active:
        status=subprocess.run(['kaggle','kernels','status',row['ref']],capture_output=True,text=True,check=True).stdout
        if not any(x in status for x in ['RUNNING','QUEUED']):
            raise RuntimeError('Refresh terminal ledger entry before launching: '+row['ref'])
    if len(active)>=2:raise RuntimeError('Two experiment kernels are already active')
    reserved=sum(r.get('script_gpu_hours',r['timeout_seconds']*r['max_assumed_devices']/3600) for r in rows)
    bound=reserved+args.timeout*2/3600
    if bound>8:raise RuntimeError(f'Conservative allocated GPU-hour bound would be {bound:.3f}')
    start=datetime.now(timezone.utc).isoformat()
    result=subprocess.run(['kaggle','kernels','push','-p',str(args.directory),'-t',str(args.timeout)],
                          capture_output=True,text=True,check=True)
    print(result.stdout,flush=True)
    # A watcher may have refreshed a completed kernel while the CLI request was in flight.
    rows=json.loads(ledger_path.read_text(encoding='utf8'))
    versions=[r['version'] for r in rows if r['ref']==meta['id']]
    rows.append({'ref':meta['id'],'version':max(versions,default=0)+1,'status':'RUNNING',
                 'timeout_seconds':args.timeout,'max_assumed_devices':2,'local_code_utc':start,
                 'privacy_verified_local':True,'conservative_hours_bound_at_launch':bound})
    ledger_path.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
