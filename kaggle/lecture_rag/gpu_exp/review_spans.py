"""Export 20 fixed review spans per ASR system, with timestamps and slide support."""
import argparse
import json
from pathlib import Path

SPANS={
 'hVU5Mw19V_c':[(0,20),(940,970),(1548,1572),(1750,1770)],
 'v2YTz8ToEag':[(30,55),(1638,1675),(1700,1725),(1900,1925)],
 'y1CvB950VlU':[(188,214),(300,318),(382,403),(407,429)],
 'L25_V011':[(75,95),(220,240),(430,450),(578,600)],
 'cs_0001':[(60,80),(200,220),(420,440),(700,720)]}
EXPORT=Path('D:/thesis-data/run-20261020/export')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    systems={'exported':{v:EXPORT/v/'03_audio/transcript.json' for v in SPANS}}
    for path in args.root.rglob('transcript.json'):
        if path.parent.name in SPANS:systems.setdefault(path.parent.parent.name,{})[path.parent.name]=path
    lines=['# Fixed textual review spans','','20 spans per complete ASR system. Acoustic listening was not performed. These are text/slide-supported checks, not a human WER reference.','']
    rows=[]
    for system,paths in systems.items():
        lines+=['## '+system,'']
        for video,path in paths.items():
            transcript=json.loads(path.read_text(encoding='utf8'))
            for a,b in SPANS[video]:
                words=[w for w in transcript['words'] if a<=(w['start_sec']+w['end_sec'])/2<b]
                text=' '.join(w['text'] for w in words)
                row={'system':system,'video':video,'start':a,'end':b,'text':text}
                rows.append(row);lines+=[f'### {video} {a}–{b}s','',text or '(No timed words in span.)','']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines),encoding='utf8')
    args.output.with_suffix('.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
