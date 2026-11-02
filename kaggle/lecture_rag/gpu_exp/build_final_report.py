"""Build final report tables from measured JSON and explicit review notes."""
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/'runs/gpu'
ORDER=['medium','large_v3','turbo','phowhisper','chunkformer','phowhisper_hints','medium_hints']
LABELS={'medium':'Medium','large_v3':'Large-v3','turbo':'Large-v3-turbo','phowhisper':'PhoWhisper-large',
        'chunkformer':'ChunkFormer large-vie','phowhisper_hints':'Pho + glossary/fallback','medium_hints':'Medium + glossary/fallback'}

def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def number(value,places=5):return '—' if value is None else f'{value:.{places}f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])

def main():
    asr=read('analysis/asr.json');summary=read('analysis/summary.json');ocr=read('analysis/ocr.json')
    rows=[]
    for s in ORDER:
        vs=asr['systems'][s];a=summary['asr'][s]
        terms=[vs[v]['terms'][t] for v,t in [('hVU5Mw19V_c','Bayes'),('hVU5Mw19V_c','xác suất'),('v2YTz8ToEag','PPF'),('y1CvB950VlU','len')]]
        mem=max(v['runtime'].get('peak_sampled_device_vram_mib',v['runtime'].get('peak_torch_vram_mib',0)) for v in vs.values())
        rows.append([LABELS[s],*[f"{t['correct']}/{t['tracked_total']}" for t in terms],a['long_loops'],a['timing_le_1ms'],a['timing_gt_2s'],f"{a['rtf']:.5f}",f"{mem:.0f}"])
    asr_table=table(['ASR (five complete videos)','Bayes','xác suất','PPF','len','Long loops','≤1ms words','>2s words','RTF','VRAM MiB'],rows)
    labels=['controlled-v2-medium-eval','controlled-v2-large_v3-eval','controlled-v2-turbo-eval',
            'controlled-v4-phowhisper-eval','controlled-v4-chunkformer-eval','controlled-v5-phowhisper_hints-eval','controlled-v6-medium_hints-eval']
    rows=[]
    for s,label in zip(ORDER,labels):
        d=summary['downstream'][label]['overall']
        rows.append([LABELS[s],d['answerable'],*[f"{d[k]:.5f}" for k in ['hit1','hit3','mrr','context_recall']]])
    downstream_table=table(['Controlled stage04/05 + retrieval','Questions','Hit@1','Hit@3','MRR','Context recall'],rows)
    pairs={frozenset(r['systems']):r['disagreement'] for r in summary['pairwise_weighted']}
    disagreement=table(['System',*ORDER],[[s,*['—' if s==t else f'{pairs[frozenset([s,t])]:.4f}' for t in ORDER]] for s in ORDER])
    rows=[]
    for s,d in ocr['aggregate'].items():
        rows.append([s,d['reference_images'],d['all_images'],number(d['cer']),number(d['word_accuracy']),number(d['diacritic_accuracy']),number(d['seconds_per_image'],3),d['loop_images'],d['nonvalidated_images'] if s.startswith('qwen_') else 'not recorded'])
    ocr_table=table(['OCR','Draft GT images','All predictions','CER','1−WER','Diacritics','Seconds/image','Loop images','Invalid EOS/loop outputs'],rows)
    ledger=read('ledger.json')
    ledger_table=table(['Private kernel slug (owner prefix below)','Version','Status','Script seconds','Allocated T4-hours','Output folder'],
        [[r['ref'].split('/')[-1],r['version'],r['status'],f"{r['runtime_seconds']:.2f}" if 'runtime_seconds' in r else 'unavailable',
          f"{r.get('script_gpu_hours',r['timeout_seconds']*2/3600):.5f}",r.get('output_folder','pending')] for r in ledger])
    hours=sum(r.get('script_gpu_hours',r['timeout_seconds']*2/3600) for r in ledger)
    assert all(r['status']!='RUNNING' for r in ledger),'Report requires terminal outputs'
    llm=read('analysis/llm.json')
    rows=[]
    for s,d in llm['systems'].items():
        rows.append([s,d['completed'],d['fully_correct'],d['partial'],d['incorrect_or_abstained'],f"{d['median_request_seconds']:.3f}",f"{d['mean_request_seconds']:.3f}"])
    llm_table=table(['Original E3 run','Responses','Fully correct','Partial','Wrong/abstained','Median request s','Mean request s'],rows)
    retry_path=ROOT/'analysis/llm_qwen3.json'
    retry_table='The explicit `/no_think` retry did not produce scored outputs; see its retained failure logs.'
    if retry_path.exists():
        d=read('analysis/llm_qwen3.json')['systems']['qwen3_4b']
        retry_table=table(['Explicit /no_think retry (still thinking)','Responses','Fully correct','Partial','Wrong/abstained','Median request s','Mean request s'],
            [['Qwen3-4B',d['completed'],d['fully_correct'],d['partial'],d['incorrect_or_abstained'],
              f"{d['median_request_seconds']:.3f}" if d['completed'] else '—',f"{d['mean_request_seconds']:.3f}" if d['completed'] else '—']])
    raw_table='No scored result from the raw ChatML closed-thinking retry; retained failures explain missing coverage.'
    raw_path=ROOT/'analysis/llm_qwen3_raw.json'
    if raw_path.exists():
        d=read('analysis/llm_qwen3_raw.json')['systems']['qwen3_4b']
        raw_table=table(['Raw ChatML, closed empty thinking block','Responses','Fully correct','Partial','Wrong/abstained','Median request s','Mean request s'],
            [['Qwen3-4B',d['completed'],d['fully_correct'],d['partial'],d['incorrect_or_abstained'],
              number(d['median_request_seconds'],3),number(d['mean_request_seconds'],3)]])
    text=(BASE/'final_report_template.md').read_text(encoding='utf8')
    notes=json.loads((BASE/'final_review_notes.json').read_text(encoding='utf8'))
    for key,value in {'ASR_TABLE':asr_table,'DOWNSTREAM_TABLE':downstream_table,'DISAGREEMENT':disagreement,
                      'OCR_TABLE':ocr_table,'LEDGER_TABLE':ledger_table,'GPU_HOURS':f'{hours:.5f}',
                      'LLM_TABLE':llm_table,'LLM_RETRY_TABLE':retry_table,'LLM_RAW_TABLE':raw_table,**notes}.items():text=text.replace('{{'+key+'}}',value)
    assert '{{' not in text,'Unfilled report marker'
    (BASE/'GPU_EXPERIMENTS.md').write_text(text,encoding='utf8')
    print('Allocated T4-hours including failed attempts:',hours)

if __name__=='__main__':main()
