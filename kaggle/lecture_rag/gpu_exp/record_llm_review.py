"""Persist manually read E3 judgments; no automatic judge or reference leakage."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent/'runs/gpu'
RATINGS={
    'rubric':'2=correct target answer without material contradiction; 1=partial, ambiguous, or correct fact embedded in unfinished reasoning; 0=wrong/abstained/no target answer. Draft references; manual Codex textual comparison, not human-certified accuracy.',
    'scores':{
        'qwen2.5_7b':[2,2,2,2,2,2,2,1,2,1,2,0,1,2,2,2,2,0,2,2,2,2,2,1,1,0,2,2,2,1],
        'qwen3_4b':[0,0,1,1,1,0,1,1,0,0,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,1],
        'vintern':[2,0,2,2,2,1,2,2,0,0,2,0,0,2,1,2,2,2,1,0,0,0,0,1,0,0,2,2,0,0]
    },
    'notes':{
        'L25_V011_q02':'Qwen2.5 says 50 delegates from countries, whereas the target is 50 countries: partial.',
        'L25_V037_q03':'Qwen2.5 gives Xô Việt instead of Xô viết: terminology spelling error, partial.',
        '_eDGTHhyQ7M_q01':'Qwen2.5 gives one outer electron without ns1 configuration: partial.',
        '_eDGTHhyQ7M_q04':'Qwen2.5 gets 0.9% but switches partly to Chinese; factual score retained; language compliance noted separately.',
        'tHE1mC_n0-E_q08':'Qwen2.5 states the target perpendicular relation but contradicts it in its directional explanation. Vintern copies evidence containing the answer and truncates.',
        'v2YTz8ToEag_q05':'Qwen2.5 supplies the PPF slope relation instead of the requested best-forgone-alternative definition: partial.',
        'y1CvB950VlU_q01':'Qwen2.5 includes triple double quotes but also permits ordinary double quotes; Vintern only ordinary double quotes.',
        'hVU5Mw19V_c_q04':'Vintern names Bayes correctly then invents a materially wrong formula: partial.',
        'qwen3_4b':'Every original response hits length with visible reasoning despite think:false. Correct target facts in unfinished reasoning earn at most partial, so this is not a valid non-thinking accuracy comparison.'
    }
}

if __name__=='__main__':
    dest=ROOT/'analysis/llm_ratings.json'
    dest.write_text(json.dumps(RATINGS,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(dest)
    retry={'rubric':RATINGS['rubric'],
           'scores':{'qwen3_4b':[0,0,1,1,1,1,1,1,1,0,1,0,1,1,1,2,1,0,1,1,1,1,1,0,1,0,1,1,1,1]},
           'notes':{'configuration':'Explicit /no_think still emits reasoning because the Ollama template opens <think>. FIFO mailbox is the one reviewed completed target answer; remaining target facts earn at most partial.'}}
    (ROOT/'analysis/llm_qwen3_ratings.json').write_text(json.dumps(retry,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    raw_retry={'rubric':RATINGS['rubric'],
               'scores':{'qwen3_4b':[0,0,2,2,1,1,2,2,1,0,2,0,1,2,2,2,2,0,2,2,1,2,2,1,1,0,2,2,1,1]},
               'notes':{'configuration':'Raw ChatML with closed empty think prefix still exposes reasoning. 22 normal stops and 8 length stops; not a validated non-thinking comparison.',
                        'L25_V011_q06':'Lists the target branches then abstains: partial.',
                        'y1CvB950VlU_q01':'Mentions triple quotes then abstains: partial.',
                        'tHE1mC_n0-E_q08':'Mentions perpendicular but concludes north and emits spurious source 0220 before truncating: partial.',
                        'y1CvB950VlU_q03':'Correct length answer with a spurious Số:1 metadata field; target factual score retained.'}}
    (ROOT/'analysis/llm_qwen3_raw_ratings.json').write_text(json.dumps(raw_retry,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
