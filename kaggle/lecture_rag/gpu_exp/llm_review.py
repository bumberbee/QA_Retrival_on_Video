"""Print fixed-context answers beside draft references for manual comparison."""
import argparse
import json
from pathlib import Path

SYSTEMS=['qwen2.5_7b','qwen3_4b','vintern']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--count',type=int,default=10)
    a=p.parse_args()
    qs=json.loads((a.root/'questions.json').read_text(encoding='utf8'))
    for i,q in enumerate(qs[a.start:a.start+a.count],a.start):
        print(f"\n[{i}] {q['id']} | {q['question']}\nREFERENCE: {q['answer']}")
        for system in SYSTEMS:
            path=a.root/system/(q['id']+'.json')
            d=json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
            print(system+': '+str(d.get('answer','(no completed answer)')))


if __name__=='__main__':main()
