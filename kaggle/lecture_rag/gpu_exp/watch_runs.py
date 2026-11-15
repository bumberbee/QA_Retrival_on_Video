"""Poll private CLI kernels, download terminal outputs and update the local budget ledger.

Does not launch kernels. The CLI manages authentication; this code never reads it.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
RUNS = BASE.parent / 'runs/gpu'


def cli(*args):
    p = subprocess.run(['kaggle', *args], capture_output=True, text=True,
                       encoding='utf8', errors='replace', timeout=300)
    if p.returncode:
        raise RuntimeError(p.stderr[-3000:] + p.stdout[-3000:])
    return p.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobs', nargs='+', help='slug=output-folder')
    args = parser.parse_args()
    jobs = dict(x.split('=', 1) for x in args.jobs)
    while jobs:
        for slug, folder in list(jobs.items()):
            ref = 'thnhtrungnguynmif/' + slug
            status = cli('kernels', 'status', ref)
            if not any(x in status for x in ['COMPLETE', 'ERROR', 'CANCELLED']):
                continue
            dest = RUNS / folder
            cli('kernels', 'output', ref, '-p', str(dest), '--page-size', '200',
                '--file-pattern', r'results/|\.log$')
            ledger_path = RUNS / 'ledger.json'
            ledger = json.loads(ledger_path.read_text(encoding='utf8'))
            row = next(x for x in reversed(ledger) if x['ref'] == ref and x['status'] == 'RUNNING')
            row.update(status='COMPLETE' if 'COMPLETE' in status else 'ERROR', output_folder=folder)
            session_path = dest / 'results/session.json'
            if session_path.exists():
                session = json.loads(session_path.read_text(encoding='utf8'))
                row.update(runtime_seconds=session['elapsed_seconds'], devices=2,
                           script_gpu_hours=session['elapsed_seconds'] * 2 / 3600,
                           finished_utc=session['finished_utc'])
            else:
                row['budget_note'] = 'No session record: reserve hard timeout until logs determine runtime.'
            ledger_path.write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf8')
            print(json.dumps({'ref': ref, 'status': row['status'], 'output': str(dest),
                              'gpu_hours': row.get('script_gpu_hours')}, ensure_ascii=False), flush=True)
            del jobs[slug]
        if jobs:
            time.sleep(45)


if __name__ == '__main__':
    main()
