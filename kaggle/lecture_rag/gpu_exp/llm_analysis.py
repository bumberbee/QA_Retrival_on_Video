"""Aggregate explicitly supplied manual reference judgments and measured latency."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--ratings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    ratings=json.loads(a.ratings.read_text(encoding='utf8'))
    questions=json.loads((a.root/'questions.json').read_text(encoding='utf8'))
    summary={'rubric':ratings['rubric'],'reference_status':'Draft, auto-written references; manual Codex comparison, not human-certified accuracy.',
             'selected_questions':len(questions),'systems':{}}
    for system,scores in ratings['scores'].items():
        assert len(scores)==len(questions)
        rows=[]
        for q,score in zip(questions,scores):
            path=a.root/system/(q['id']+'.json')
            if not path.exists():
                assert score is None
                continue
            assert score in [0,1,2]
            d=json.loads(path.read_text(encoding='utf8'))
            rows.append({'id':q['id'],'score':score,'seconds':d['seconds'],
                         'first_load_seconds':d.get('raw',{}).get('load_duration',0)/1e9})
        times=[r['seconds'] for r in rows]
        summary['systems'][system]={'completed':len(rows),'fully_correct':sum(r['score']==2 for r in rows),
            'partial':sum(r['score']==1 for r in rows),'incorrect_or_abstained':sum(r['score']==0 for r in rows),
            'fully_correct_fraction':sum(r['score']==2 for r in rows)/len(rows) if rows else None,
            'mean_request_seconds':statistics.mean(times) if times else None,
            'median_request_seconds':statistics.median(times) if times else None,
            'first_request_seconds':times[0] if times else None,'questions':rows}
    a.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for label,row in summary['systems'].items():print(label,{k:v for k,v in row.items() if k!='questions'})


if __name__=='__main__':main()
