"""Term proxies, timing/loops, paired disagreement and draft OCR reference metrics."""
import argparse
import difflib
import json
from pathlib import Path
import re
import unicodedata

BASE=Path(__file__).resolve().parent
EXPORT=Path('D:/thesis-data/run-20261020/export')
TERMS={
 'hVU5Mw19V_c':{'Bayes':['bs','bias','bis'],'xác suất':['sát xuất','sắc xuất','xác xuất','sát suất'],
                  'biến cố':[],'độc lập':[]},
 'v2YTz8ToEag':{'PPF':['pps'],'chi phí cơ hội':[],'chi phí chìm':[],'khan hiếm':[]},
 'y1CvB950VlU':{'len':['lan'],'for':['4 4'],'print':[],'end':[],'Python':[]},
 'L25_V011':{'Liên Hiệp Quốc':[],'Liên Hợp Quốc':[],'Đại hội đồng':[],'Hội đồng Bảo an':[],'quyền bình đẳng':[]},
 'cs_0001':{'decidable':[],'undecidable':[],'FIFO':[],'MSC':[],'RSC':[],
            'synchronous':[],'asynchronous':[],'partial order':[],'message passing':[]}}


def normalized(s,fold=False):
    s=unicodedata.normalize('NFC',s).lower()
    if fold:s=''.join(c for c in unicodedata.normalize('NFD',s) if not unicodedata.combining(c)).replace('đ','d')
    return ' '.join(re.findall(r'\w+',s))


def edit_alignment(a,b):
    # Levenshtein with alignment, memory linear except compact backtrace; OCR slides are small.
    import numpy as np
    trace=np.zeros((len(a)+1,len(b)+1),dtype=np.uint8)
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):
            vals=[prev[j-1]+(x!=y),prev[j]+1,row[j-1]+1]
            choice=min(range(3),key=lambda k:vals[k]);row.append(vals[choice]);trace[i,j]=choice
        prev=row
    distance=prev[-1]; pairs=[];i=len(a);j=len(b)
    while i or j:
        c=int(trace[i,j]) if i and j else 1 if i else 2
        if c==0:pairs.append((a[i-1],b[j-1]));i-=1;j-=1
        elif c==1:pairs.append((a[i-1],None));i-=1
        else:pairs.append((None,b[j-1]));j-=1
    return distance,list(reversed(pairs))


def count(text,term):return len(re.findall(r'(?<!\w)'+re.escape(normalized(term))+r'(?!\w)',normalized(text)))


def loops(words,threshold=8,max_ngram=8):
    tokens=[normalized(w['text']) for w in words];out=[];i=0
    while i<len(tokens):
        best=None
        for n in range(1,max_ngram+1):
            unit=tokens[i:i+n]
            if len(unit)!=n:break
            j=i+n
            while tokens[j:j+n]==unit and len(tokens[j:j+n])==n:j+=n
            repeats=(j-i)//n
            if repeats>=threshold and (not best or j>best[0]):best=(j,n,repeats)
        if best:
            j,n,repeats=best;out.append({'start':words[i]['start_sec'],'end':words[j-1]['end_sec'],
                'phrase':' '.join(tokens[i:i+n]),'repeats':repeats,'words':j-i});i=j
        else:i+=1
    return out


def asr_metrics(transcript):
    words=transcript['words'];video=transcript['video_id'];text=' '.join(w['text'] for w in words)
    termcounts={t:{'correct':count(text,t),'known_mishearings':{x:count(text,x) for x in aliases}}
                for t,aliases in TERMS[video].items()}
    for d in termcounts.values():
        bad=sum(d['known_mishearings'].values());d['tracked_total']=d['correct']+bad
        d['tracked_correct_fraction']=d['correct']/d['tracked_total'] if d['tracked_total'] else None
    return {'words':len(words),'terms':termcounts,'loops':loops(words,max_ngram=32),
            'timing_le_1ms':sum(w['end_sec']-w['start_sec']<=.001 for w in words),
            'timing_gt_2s':sum(w['end_sec']-w['start_sec']>2 for w in words),
            'invalid_timing':sum(w['end_sec']<w['start_sec'] for w in words)}


def asr(root):
    outputs={};transcripts={}
    for video in TERMS:
        t=json.loads((EXPORT/video/'03_audio/transcript.json').read_text(encoding='utf8'))
        outputs.setdefault('exported',{})[video]=asr_metrics(t);transcripts[('exported',video)]=t
    for p in root.rglob('transcript.json'):
        # Raw model folders only, not rebuilt stage files.
        if p.parent.name not in TERMS or p.parent.parent.name.startswith('_'):continue
        t=json.loads(p.read_text(encoding='utf8'));system=p.parent.parent.name
        outputs.setdefault(system,{})[t['video_id']]=asr_metrics(t);transcripts[(system,t['video_id'])]=t
        m=p.with_name('metrics.json')
        if m.exists():outputs[system][t['video_id']]['runtime']=json.loads(m.read_text(encoding='utf8'))
    disagreements=[]
    for video in TERMS:
        systems=[s for s,v in transcripts if v==video]
        for i,s in enumerate(systems):
            a=transcripts[(s,video)]
            for other in systems[i+1:]:
                b=transcripts[(other,video)]
                paired=[]
                for w in a['windows']:
                    x=normalized(' '.join(z['text'] for z in a['words'] if z['window_id']==w['window_id'])).split()
                    y=normalized(' '.join(z['text'] for z in b['words'] if z['window_id']==w['window_id'])).split()
                    # Symmetric token edit distance: a proxy, never WER against truth.
                    d,_=edit_alignment(x,y);paired.append({'id':w['window_id'],'distance':d,'tokens_a':len(x),'tokens_b':len(y)})
                disagreements.append({'video':video,'a':s,'b':other,'distance':sum(p['distance'] for p in paired),
                    'symmetric_normalized_disagreement':sum(p['distance'] for p in paired)/max(1,sum(max(p['tokens_a'],p['tokens_b']) for p in paired)),
                    'windows':paired})
    return {'terms':TERMS,'systems':outputs,'pairwise':disagreements}


def ocr(root):
    gtpath=next(root.rglob('ground_truth.json'),BASE.parent/'runs/gpu/discovery/gt.json')
    gt=json.loads(gtpath.read_text(encoding='utf8'))
    outputs={}
    for p in root.rglob('*.json'):
        if p.parent.name not in {'qwen_whole','qwen_crop','paddlevl','ppocr'}:continue
        d=json.loads(p.read_text(encoding='utf8'));key=d['id'];text=d['text']
        row={'seconds':d['seconds'],'text':text,'validated':d.get('validated'),'attempts':len(d.get('attempts',[])),
             'max_consecutive_repeats':d.get('max_consecutive_repeats',d.get('attempts',[{}])[-1].get('max_consecutive_repeats')),
             'eos':d.get('eos',d.get('attempts',[{}])[-1].get('eos'))}
        text_tokens=normalized(text).split()
        # Positions below are sequence indices only; no audio timestamps are created.
        text_runs=loops([{'text':t,'start_sec':i,'end_sec':i+1} for i,t in enumerate(text_tokens)],max_ngram=32)
        row['max_consecutive_repeats']=max((r['repeats'] for r in text_runs),default=1)
        row['text_repeat_rule']='at least 8 consecutive repeats, 1–32 tokens; offline analysis'
        if key in gt['items']:
            gold=normalized(gt['items'][key]['text']);pred=normalized(re.sub(r'^\s*\[(Hình|Figure)\].*$', '', text,flags=re.M))
            cd,aligned=edit_alignment(gold,pred);wd,waligned=edit_alignment(gold.split(),pred.split())
            accented=[(a,b) for a,b in aligned if a and normalized(a,True)!=a]
            row.update(kind=gt['items'][key]['kind'],chars=len(gold),char_edits=cd,cer=cd/max(1,len(gold)),
                words=len(gold.split()),word_edits=wd,word_accuracy=1-wd/max(1,len(gold.split())),
                accented_chars=len(accented),correct_accented_chars=sum(a==b for a,b in accented),
                diacritic_accuracy=sum(a==b for a,b in accented)/max(1,len(accented)))
        outputs.setdefault(p.parent.name,{})[key]=row
    aggregate={}
    for system,items in outputs.items():
        scored=[x for x in items.values() if 'cer' in x];chars=sum(x['chars'] for x in scored);words=sum(x['words'] for x in scored)
        accents=sum(x['accented_chars'] for x in scored)
        aggregate[system]={'reference_images':len(scored),'all_images':len(items),'cer':sum(x['char_edits'] for x in scored)/chars if chars else None,
            'word_accuracy':1-sum(x['word_edits'] for x in scored)/words if words else None,
            'diacritic_accuracy':sum(x['correct_accented_chars'] for x in scored)/accents if accents else None,
            'seconds_per_image':sum(x['seconds'] for x in items.values())/max(1,len(items)),
            'loop_images':sum((x['max_consecutive_repeats'] or 0)>=8 for x in items.values()),
            'nonvalidated_images':sum(x['validated'] is False for x in items.values()),
            'per_kind':{}}
        for kind in ['print','hand']:
            group=[x for x in scored if x['kind']==kind]
            aggregate[system]['per_kind'][kind]={'images':len(group),
                'cer':sum(x['char_edits'] for x in group)/sum(x['chars'] for x in group) if group else None,
                'word_accuracy':1-sum(x['word_edits'] for x in group)/sum(x['words'] for x in group) if group else None}
    return {'reference_note':gt['note'],'systems':outputs,'aggregate':aggregate}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['asr','ocr']);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={'asr':asr,'ocr':ocr}[a.kind](a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
