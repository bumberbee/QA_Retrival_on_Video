"""Generate measurement tables directly from raw-derived JSON; no guessed values."""
import argparse
import json
from pathlib import Path


def fraction(v,term):
    t=v.get('terms',{}).get(term)
    return f"{t['correct']}/{t['tracked_total']}" if t and t['tracked_total'] else '—'


def main():
    p=argparse.ArgumentParser();p.add_argument('--asr',type=Path);p.add_argument('--ocr',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    lines=[]
    if a.asr:
        data=json.loads(a.asr.read_text(encoding='utf8'))
        lines+=['## ASR measurements','','Fractions count recognized canonical forms versus the predeclared known aliases; they do not count every omission.','',
            '| System | Bayes | xác suất | PPF | len | Long loops | ≤1ms timings | >2s timings | RTF | VRAM MiB |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for system,vs in data['systems'].items():
            rt=[v['runtime'] for v in vs.values() if 'runtime' in v]
            rtf=sum(x['decode_seconds'] for x in rt)/sum(x['duration_seconds'] for x in rt) if rt else None
            mem=[x.get('peak_sampled_device_vram_mib',x.get('peak_torch_vram_mib')) for x in rt]
            vals=[fraction(vs.get('hVU5Mw19V_c',{}),'Bayes'),fraction(vs.get('hVU5Mw19V_c',{}),'xác suất'),
                  fraction(vs.get('v2YTz8ToEag',{}),'PPF'),fraction(vs.get('y1CvB950VlU',{}),'len')]
            lines.append('| '+system+' | '+' | '.join(vals)+f" | {sum(len(v['loops']) for v in vs.values())} | {sum(v['timing_le_1ms'] for v in vs.values())} | {sum(v['timing_gt_2s'] for v in vs.values())} | "+
                         (f'{rtf:.3f}' if rtf is not None else '—')+' | '+(f'{max(mem):.0f}' if mem else '—')+' |')
        lines+=['','Partial systems must be compared only on shared videos. CT2 memory is sampled device usage; torch memory is peak allocated tensor memory.','',
                '| Rebuild/evaluation | N | Hit@1 | Hit@3 | MRR | Context recall |','|---|---:|---:|---:|---:|---:|']
        for p in sorted((a.output.parents[1]/'downstream').glob('*-eval/retrieval.json')):
            d=json.loads(p.read_text(encoding='utf8'));r=d['overall']
            lines.append(f"| {p.parent.name} | {r['answerable']} | {r['hit1']:.3f} | {r['hit3']:.3f} | {r['mrr']:.3f} | {r['context_recall']:.3f} |")
    if a.ocr:
        data=json.loads(a.ocr.read_text(encoding='utf8'))
        lines+=['','## OCR measurements','','Reference status: '+data['reference_note'],'',
                '| System | Reference images | CER | Word accuracy (1−WER) | Diacritic accuracy | Seconds/image incl. retries |',
                '|---|---:|---:|---:|---:|---:|']
        for s,r in data['aggregate'].items():
            scores=['—' if r[k] is None else f'{r[k]:.3f}' for k in ['cer','word_accuracy','diacritic_accuracy']]
            lines.append(f"| {s} | {r['reference_images']} | "+' | '.join(scores)+f" | {r['seconds_per_image']:.2f} |")
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':main()
