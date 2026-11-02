"""Post-hoc numeral-format sensitivity; keep original retrieval and timings unchanged.

This is supplementary analysis, not a replacement benchmark or ASR correction.
The five spellings were identified during review; the same rules apply to every system.
"""
import argparse
import json
from pathlib import Path
import re
import types
from unittest.mock import patch

import downstream

SPELLINGS={'không phẩy ba sáu':'0,36','một trăm':'100','bảy mươi':'70','ba mươi':'30','mười hai':'12'}


def numeric(text):
    for source, target in SPELLINGS.items():
        text=re.sub(r'(?<!\w)'+re.escape(source)+r'(?!\w)',target,text,flags=re.I)
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    downstream.freeze_backend()
    runs=downstream.ROOT/'runs/gpu/downstream'
    defs={q['id']:q for q in json.loads((runs/'questions_asr.json').read_text(encoding='utf8'))['questions']}
    original=downstream.evaluate.term_match
    def match(term,text,threshold=.88):return original(numeric(term),numeric(text),threshold)
    result={'status':'post-hoc, shared-format sensitivity; rankings/evidence/times fixed',
            'spellings':SPELLINGS,'runs':{}}
    for path in sorted(runs.glob('*-eval/retrieval.json')):
        if not path.parent.name.startswith(('controlled-v2-','controlled-v4-','controlled-v5-','controlled-v6-')):continue
        d=json.loads(path.read_text(encoding='utf8'));rows=[]
        for q in d['questions']:
            if not q.get('context_evidence'):continue
            index=types.SimpleNamespace(folder=path.parent.parent/path.parent.name.removesuffix('-eval')/q['video']/'05_index')
            with patch.object(downstream.evaluate,'term_match',match):
                trace=downstream.evaluate.evidence_context_trace(q,index,defs[q['id']],q['context'])
            rows.append({'id':q['id'],'original':q['context_recall'],'numeric_format':int(trace['recalled'])})
        if rows:
            result['runs'][path.parent.name]={'n':len(rows),'original':sum(r['original'] for r in rows)/len(rows),
                'numeric_format':sum(r['numeric_format'] for r in rows)/len(rows),'questions':rows}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for label,row in result['runs'].items():print(label,row['original'],row['numeric_format'])


if __name__=='__main__':main()
