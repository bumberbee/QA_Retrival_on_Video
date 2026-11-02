"""Compare OCR on shared draft-GT IDs, including partial native predictions."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent/'runs/gpu'

if __name__=='__main__':
    data=json.loads((ROOT/'analysis/ocr.json').read_text(encoding='utf8'))
    vs=data['systems']
    rows={}
    for candidate in vs:
        if candidate=='qwen_crop':continue
        keys=sorted(k for k in vs[candidate] if 'cer' in vs[candidate][k] and k in vs['qwen_crop'])
        pair={}
        for label in ['qwen_crop',candidate]:
            selected=[vs[label][k] for k in keys]
            if not selected:continue
            pair[label]={'reference_images':len(keys),'ids':keys,
                'cer':sum(x['char_edits'] for x in selected)/sum(x['chars'] for x in selected),
                'word_accuracy':1-sum(x['word_edits'] for x in selected)/sum(x['words'] for x in selected),
                'diacritic_accuracy':sum(x['correct_accented_chars'] for x in selected)/sum(x['accented_chars'] for x in selected),
                'seconds_per_reference_image':sum(x['seconds'] for x in selected)/len(keys)}
        rows[candidate]=pair
    (ROOT/'analysis/ocr_shared.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for label,pair in rows.items():print(label,{k:{x:y for x,y in v.items() if x!='ids'} for k,v in pair.items()})
