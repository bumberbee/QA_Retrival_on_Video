"""Reproduce thesis protocols: L25 S/SC ±1s, LPM greedy ±1s and ±3s.

python eval/segmentation/evaluate.py --result runs/stage01/kernel-output/result
    --lpm-slides runs/stage01/labels-lpm-videos/slides.csv --output <scores.json>
All predictions and labels are read-only; only the requested report is written.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LPM_IDS = ['ALRJCeVT0fQ','BsXUWddl-as','DM4MWxnXOzM','QPA4yLCbMw4',
           '_YHDxyO9W8Q','uS8F-qRUeMM','gsDGPyBzrQM','V-4SZ4f7kMc']

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def prf(counts):
    tp,fp,fn=counts
    p,r=tp/max(1,tp+fp),tp/max(1,tp+fn)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':p,'recall':r,
            'f1':2*p*r/max(1e-9,p+r)}

def score_l25(boundaries, events, ad_edges):
    # Faithful to gold/eval_ad_detection.py: neutral layout/ad cuts; duplicate
    # detections at the same S/SC event do not add a second TP or an FP.
    tp=fp=0
    matched=set()
    for t in boundaries:
        near=[(i,e) for i,e in enumerate(events) if abs(e['time_sec']-t)<=1.0]
        index,event=min(near,key=lambda pair:abs(pair[1]['time_sec']-t)) if near else (None,None)
        if event is not None and event['label'] in {'S','SC'}:
            tp+=index not in matched
            matched.add(index)
        elif (event is not None and event['label'] in {'C','L'}) or any(abs(t-x)<=2 for x in ad_edges):
            continue
        else:
            fp+=1
    fn=sum(e['label'] in {'S','SC'} and i not in matched for i,e in enumerate(events))
    return tp,fp,fn

def score_lpm(gold,pred,tolerance):
    pairs=sorted((abs(g-p),i,j) for i,g in enumerate(gold)
                 for j,p in enumerate(pred) if abs(g-p)<=tolerance)
    used_g,used_p=set(),set()
    for _,i,j in pairs:
        if i not in used_g and j not in used_p:
            used_g.add(i);used_p.add(j)
    return len(used_g),len(pred)-len(used_p),len(gold)-len(used_g)

def interval_file(config,video):
    for file in (config/video/'intervals.json',config/video/'01_slides/intervals.json'):
        if file.is_file():
            return file
    raise FileNotFoundError(f'Missing {video} in {config}; never score a partial split')

def evaluate(result,lpm_slides=None):
    labels=read(HERE/'labels/l25_protocol.json')['videos']
    spans=defaultdict(list)
    if lpm_slides:
        with Path(lpm_slides).open(encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                spans[row['video_id']].append((float(row['start_s']),float(row['end_s'])))
    report={}
    for config in sorted(p for p in Path(result).iterdir() if p.is_dir()):
        totals=defaultdict(lambda:[0,0,0]);rows=[]
        for v in labels:
            items=read(interval_file(config,v['video']))
            score=score_l25([i['start_sec'] for i in items[1:]],v['events'],v['ad_edges'])
            key='l25_'+v['split']
            totals[key]=[a+b for a,b in zip(totals[key],score)]
            rows.append({'video':v['video'],'split':key,**prf(score)})
        if spans:
            for video in LPM_IDS:
                labelled=sorted(spans[video])
                if not labelled:
                    raise ValueError(f'Missing LPM labels: {video}')
                lo,hi=labelled[0][0],labelled[-1][1]
                gold=[s for s,_ in labelled[1:]]
                items=read(interval_file(config,video))
                for tolerance in (1.,3.):
                    pred=[i['start_sec'] for i in items[1:] if lo-tolerance<=i['start_sec']<=hi+tolerance]
                    score=score_lpm(gold,pred,tolerance)
                    key=f'lpm_{tolerance:g}s'
                    totals[key]=[a+b for a,b in zip(totals[key],score)]
                    rows.append({'video':video,'split':key,**prf(score)})
        report[config.name]={'totals':{k:prf(v) for k,v in totals.items()},'rows':rows}
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--lpm-slides',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=evaluate(args.result,args.lpm_slides)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v['totals'] for k,v in report.items()},indent=2))

if __name__=='__main__':
    main()
