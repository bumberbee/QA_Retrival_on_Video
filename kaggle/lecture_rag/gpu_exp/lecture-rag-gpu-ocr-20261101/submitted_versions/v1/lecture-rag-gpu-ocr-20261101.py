"""Shared, credential-free helpers embedded in private Kaggle scripts."""
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

START = time.monotonic()
OUT = Path('/kaggle/working/results')
OUT.mkdir(parents=True, exist_ok=True)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf8')


def install(*packages):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], check=True, timeout=600)


def environment():
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                          '--format=csv,noheader'], capture_output=True, text=True)
    return {'python': sys.version, 'gpu': gpu.stdout.strip(),
            'packages': {d.metadata['Name']: d.version for d in metadata.distributions()}}


def vram():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
    return [int(x) for x in r.stdout.splitlines() if x.strip().isdigit()]


def export_root():
    # Prefer mounted individual exports, otherwise extract only stages needed by the experiments.
    paths = list(Path('/kaggle/input').rglob('transcript.json'))
    for p in paths:
        if p.parent.name == '03_audio' and p.parent.parent.name == 'hVU5Mw19V_c':
            return p.parents[2]
    for z in Path('/kaggle/input').rglob('*.zip'):
        with zipfile.ZipFile(z) as f:
            names = f.namelist()
            if not any('hVU5Mw19V_c/03_audio/transcript.json' in n for n in names):
                continue
            dest = Path('/kaggle/working/export_inputs')
            for n in names:
                if any('/' + s + '/' in '/' + n for s in ('01_slides', '02_visual', '03_audio')):
                    target = (dest / n).resolve()
                    if not target.is_relative_to(dest.resolve()):
                        raise ValueError('Unsafe archive path')
                    f.extract(n, dest)
            return next(dest.rglob('hVU5Mw19V_c/03_audio/transcript.json')).parents[2]
    raise FileNotFoundError('Attached fast-pipeline output has no stage-03 export')


def finish(extra=None):
    save(OUT / 'session.json', {'elapsed_seconds': time.monotonic() - START,
                              'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                              **(extra or {})})


def failure(label, error):
    detail = {'error': str(error), 'traceback': traceback.format_exc()}
    save(OUT / ('failure_' + label + '.json'), detail)
    print(label, 'FAILED', str(error), flush=True)


QWEN_SLIDE_PROMPTS = {'vi': 'Đọc toàn bộ chữ trên slide bài giảng này theo thứ tự đọc. Giữ đúng dấu tiếng Việt, chép đúng như trên slide, không tự sửa. Công thức viết LaTeX, bảng viết Markdown. Bỏ logo kênh, số điện thoại quảng cáo, phụ đề video. Nếu có biểu đồ, sơ đồ hoặc hình minh họa, thêm cuối cùng một dòng "[Hình] " mô tả ngắn nội dung hình. Chỉ trả về nội dung.', 'en': 'Read all the text on this lecture slide in reading order, exactly as written, without correcting it. Write formulas in LaTeX and tables in Markdown. Skip channel logos and video subtitles. If there is a chart, diagram or picture, add one last line "[Figure] " with a short description. Return only the content.'}

CONFIG = {'soft_limit_seconds': 2550}
"""Matched whole-frame/crop OCR, with bounded retries and exact completion metadata."""
AUDIT = [('hVU5Mw19V_c','seg_00000'),('hVU5Mw19V_c','seg_00012'),('L25_V011','seg_00006'),
         ('tHE1mC_n0-E','seg_00097'),('tHE1mC_n0-E','seg_00201'),('v2YTz8ToEag','seg_00040'),
         ('_eDGTHhyQ7M','seg_00006'),('y1CvB950VlU','seg_00005'),('y1CvB950VlU','seg_00020')]
CROPS = {'L25_V011':(0.195,0.06,0.995,0.88), 'tHE1mC_n0-E:seg_00097':(0.43,0.13,0.97,0.70),
         'tHE1mC_n0-E:seg_00201':(0.33,0.29,0.66,0.66), 'v2YTz8ToEag':(0.22,0.08,0.86,0.99),
         '_eDGTHhyQ7M':(0.25,0.01,1.0,0.73), 'y1CvB950VlU':(0.25,0.05,0.95,0.76)}
QWEN_ID='Qwen/Qwen3-VL-4B-Instruct'
QWEN_REV='ebb281ec70b05090aa6165b016eac8ec08e71b17'
PADDLE_VL_ID='PaddlePaddle/PaddleOCR-VL-1.6'
PADDLE_VL_REV='c5630abae1d940eafe0697512a0325494b02ab42'


def dataset_items():
    from PIL import Image
    gt_path=next(Path('/kaggle/input').rglob('gt.json'))
    gt=json.loads(gt_path.read_text())
    root=export_root()
    items=[]
    for name,gold in gt['items'].items():
        p=gt_path.parent/name
        items.append({'id':name,'path':str(p),'gold':gold,'crop':None})
    for v,s in AUDIT:
        items.append({'id':v+':'+s,'path':str(root/v/'01_slides/keyframes'/(s+'.jpg')),
                      'crop':CROPS.get(v+':'+s,CROPS.get(v)),'video':v})
    save(OUT/'items.json',items)
    save(OUT/'ground_truth.json',gt)
    return items


def loop_info(text):
    import re
    words=re.findall(r'\w+',text.lower())
    longest=1
    for n in range(1,9):
        for i in range(max(0,len(words)-n*4)):
            j=i+n
            while words[j:j+n]==words[i:i+n] and len(words[j:j+n])==n:
                j+=n
            longest=max(longest,(j-i)//n)
    return longest


def image_for(item,crop):
    from PIL import Image
    im=Image.open(item['path']).convert('RGB')
    if crop and item.get('crop'):
        w,h=im.size;b=item['crop'];im=im.crop(tuple(round(x*d) for x,d in zip(b,[w,h,w,h])))
    return im


def qwen():
    install('transformers==4.57.1','huggingface-hub==0.36.0','accelerate==1.11.0','Pillow==11.3.0')
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    items=dataset_items()
    save(OUT/'environment_qwen.json',environment())
    model=Qwen3VLForConditionalGeneration.from_pretrained(QWEN_ID,revision=QWEN_REV,
        torch_dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
    for label in ('qwen_whole','qwen_crop'):
        enhanced=label=='qwen_crop'
        processor=AutoProcessor.from_pretrained(QWEN_ID,revision=QWEN_REV,
            **({'min_pixels':256*32*32,'max_pixels':1280*32*32} if enhanced else {}))
        for item in items:
            if time.monotonic()-START>CONFIG['soft_limit_seconds']:
                raise TimeoutError('OCR soft budget reached')
            try:
                im=image_for(item,enhanced)
                prompt=QWEN_SLIDE_PROMPTS['vi']
                if enhanced:
                    prompt+=' Giữ nguyên mã nguồn, dấu ngoặc, dấu nháy và thụt lề. Không giải thích hoặc lặp nội dung.'
                attempts=[]
                for attempt in range(2 if enhanced else 1):
                    messages=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':prompt}]}]
                    inp=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                        return_dict=True,return_tensors='pt').to(model.device)
                    torch.cuda.reset_peak_memory_stats()
                    t=time.monotonic()
                    with torch.inference_mode():
                        out=model.generate(**inp,max_new_tokens=2048 if attempt else 1536,do_sample=False,
                            repetition_penalty=1.10 if attempt else 1.05)
                    torch.cuda.synchronize()
                    tokens=out[0,inp['input_ids'].shape[1]:]
                    text=processor.decode(tokens,skip_special_tokens=True).strip()
                    eos=model.generation_config.eos_token_id
                    eos_ids=eos if isinstance(eos,list) else [eos]
                    ended=int(tokens[-1]) in eos_ids
                    loops=loop_info(text)
                    attempts.append({'text':text,'seconds':time.monotonic()-t,'generated_tokens':len(tokens),
                        'eos':ended,'max_consecutive_repeats':loops,'image_size':im.size,
                        'image_grid_thw':inp.get('image_grid_thw').tolist(),
                        'peak_torch_vram_mib':torch.cuda.max_memory_allocated()/1048576})
                    if ended and loops<8:break
                    prompt='Chép nguyên văn duy nhất một lần mọi chữ trong ảnh; giữ dấu, số và mã nguồn. Không mô tả, không suy diễn.'
                valid=[a for a in attempts if a['eos'] and a['max_consecutive_repeats']<8]
                selected=valid[-1] if valid else attempts[-1]
                save(OUT/label/(item['id'].replace(':','__')+'.json'),
                    {'id':item['id'],'model_id':QWEN_ID,'revision':QWEN_REV,'attempts':attempts,
                     'text':selected['text'],'validated':bool(valid),'seconds':sum(a['seconds'] for a in attempts)})
                print(label,item['id'],round(sum(a['seconds'] for a in attempts),1),flush=True)
            except Exception as e:failure(label+'_'+item['id'].replace(':','__'),e)
    del model,processor
    gc.collect();torch.cuda.empty_cache()


def paddlevl():
    install('transformers==5.22.1','huggingface-hub==1.4.1','Pillow==11.3.0')
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    save(OUT/'environment_paddlevl.json',environment())
    items=dataset_items()
    model=AutoModelForImageTextToText.from_pretrained(PADDLE_VL_ID,revision=PADDLE_VL_REV,
        trust_remote_code=True,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda:0').eval()
    processor=AutoProcessor.from_pretrained(PADDLE_VL_ID,revision=PADDLE_VL_REV,trust_remote_code=True)
    for item in items:
        try:
            im=image_for(item,True)
            messages=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':'OCR:'}]}]
            inp=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                return_dict=True,return_tensors='pt',images_kwargs={'size':{'shortest_edge':processor.image_processor.min_pixels,
                'longest_edge':1280*28*28}}).to(model.device)
            torch.cuda.reset_peak_memory_stats();t=time.monotonic()
            with torch.inference_mode():out=model.generate(**inp,max_new_tokens=2048,do_sample=False)
            torch.cuda.synchronize();tok=out[0,inp['input_ids'].shape[-1]:]
            text=processor.decode(tok,skip_special_tokens=True)
            save(OUT/'paddlevl'/(item['id'].replace(':','__')+'.json'),{'id':item['id'],
                'model_id':PADDLE_VL_ID,'revision':PADDLE_VL_REV,'text':text,'seconds':time.monotonic()-t,
                'generated_tokens':len(tok),'max_consecutive_repeats':loop_info(text),
                'peak_torch_vram_mib':torch.cuda.max_memory_allocated()/1048576})
            print('paddlevl',item['id'],flush=True)
        except Exception as e:failure('paddlevl_'+item['id'].replace(':','__'),e)


def ppocr():
    install('paddlepaddle-gpu==3.2.1','--index-url','https://www.paddlepaddle.org.cn/packages/stable/cu126/')
    install('paddleocr==3.7.0','paddlex==3.7.2','huggingface-hub==0.36.0')
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
    from huggingface_hub import snapshot_download,HfApi
    from paddleocr import PaddleOCR
    pins={}
    for model_id in ['PaddlePaddle/PP-OCRv5_server_det','PaddlePaddle/latin_PP-OCRv5_mobile_rec']:
        rev=HfApi().model_info(model_id).sha
        pins[model_id]={'revision':rev,'path':snapshot_download(model_id,revision=rev)}
    save(OUT/'ppocr_pins.json',pins)
    save(OUT/'environment_ppocr.json',environment())
    engine=PaddleOCR(text_detection_model_dir=pins['PaddlePaddle/PP-OCRv5_server_det']['path'],
        text_recognition_model_dir=pins['PaddlePaddle/latin_PP-OCRv5_mobile_rec']['path'],
        use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,device='gpu:0')
    import numpy as np
    for item in dataset_items():
        try:
            t=time.monotonic();res=list(engine.predict(np.array(image_for(item,True))))
            raw=res[0].json
            if isinstance(raw,str):raw=json.loads(raw)
            raw=raw.get('res',raw)
            text='\n'.join(raw.get('rec_texts',[]))
            save(OUT/'ppocr'/(item['id'].replace(':','__')+'.json'),{'id':item['id'],
                'model_pins':pins,'text':text,'raw':raw,'seconds':time.monotonic()-t,'vram_mib':vram()})
            print('ppocr',item['id'],flush=True)
        except Exception as e:failure('ppocr_'+item['id'].replace(':','__'),e)


if len(sys.argv)>1:
    try:
        {'--paddlevl':paddlevl,'--ppocr':ppocr}[sys.argv[1]]()
    except Exception as e:failure(sys.argv[1].strip('-'),e)
else:
    try:qwen()
    except Exception as e:failure('qwen',e)
    # Fresh interpreters isolate incompatible Transformers/Paddle runtimes.
    for mode in ['--paddlevl','--ppocr']:
        try:
            remaining=max(30,CONFIG['soft_limit_seconds']-(time.monotonic()-START))
            subprocess.run([sys.executable,__file__,mode],check=True,timeout=remaining)
        except Exception as e:failure(mode.strip('-')+'_process',e)
    finish()
