"""Shared, credential-free helpers embedded in private Kaggle scripts."""
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
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
    # Keep outputs limited to measurements; intermediate model copies are reproducible from pins.
    working=Path('/kaggle/working').resolve()
    for name in ('audio','export_inputs','phowhisper_ct2','ollama.tgz','ollama','ollama_models'):
        path=(working/name).resolve()
        if not path.is_relative_to(working) or path==working:raise ValueError('Unsafe scratch cleanup')
        if path.is_dir():shutil.rmtree(path)
        elif path.is_file():path.unlink()


def failure(label, error):
    detail = {'error': str(error), 'traceback': traceback.format_exc()}
    save(OUT / ('failure_' + label + '.json'), detail)
    print(label, 'FAILED', str(error), flush=True)


QWEN_SLIDE_PROMPTS = {'vi': 'Đọc toàn bộ chữ trên slide bài giảng này theo thứ tự đọc. Giữ đúng dấu tiếng Việt, chép đúng như trên slide, không tự sửa. Công thức viết LaTeX, bảng viết Markdown. Bỏ logo kênh, số điện thoại quảng cáo, phụ đề video. Nếu có biểu đồ, sơ đồ hoặc hình minh họa, thêm cuối cùng một dòng "[Hình] " mô tả ngắn nội dung hình. Chỉ trả về nội dung.', 'en': 'Read all the text on this lecture slide in reading order, exactly as written, without correcting it. Write formulas in LaTeX and tables in Markdown. Skip channel logos and video subtitles. If there is a chart, diagram or picture, add one last line "[Figure] " with a short description. Return only the content.'}

CONFIG = {'soft_limit_seconds': 1050}
"""Five fixed audit-region rereads; compare 4B FP16 with 8B NF4 on identical crops."""
install('transformers==4.57.1','huggingface-hub==0.36.0','accelerate==1.11.0',
        'bitsandbytes==0.48.2','Pillow==11.3.0')
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration,BitsAndBytesConfig
CASES=[
 {'id':'bayes_denominators','video':'hVU5Mw19V_c','segment':'seg_00000','box':(.38,.32,.575,.58)},
 {'id':'seed_weights','video':'hVU5Mw19V_c','segment':'seg_00012','box':(.24,.13,.56,.33)},
 {'id':'seed_percentage','video':'hVU5Mw19V_c','segment':'seg_00012','box':(.825,.765,.922,.90)},
 {'id':'python_triple_quotes','video':'y1CvB950VlU','segment':'seg_00005','box':(.285,.20,.55,.37)},
 {'id':'magnet_statements','video':'tHE1mC_n0-E','segment':'seg_00097','box':(.445,.355,.925,.505)}]
save(OUT/'cases.json',CASES);save(OUT/'environment.json',environment())
(OUT/'crops').mkdir(parents=True,exist_ok=True)
root=export_root()
MODELS=[('qwen_focus4','Qwen/Qwen3-VL-4B-Instruct','ebb281ec70b05090aa6165b016eac8ec08e71b17',False),
        ('qwen_focus8','Qwen/Qwen3-VL-8B-Instruct','0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',True)]
for label,model_id,revision,quantized in MODELS:
    try:
        kw={'quantization_config':BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)} if quantized else {}
        t=time.monotonic()
        model=Qwen3VLForConditionalGeneration.from_pretrained(model_id,revision=revision,torch_dtype=torch.float16,
                device_map='cuda:0',attn_implementation='sdpa',**kw).eval()
        loading=time.monotonic()-t
        processor=AutoProcessor.from_pretrained(model_id,revision=revision,min_pixels=512*32*32,max_pixels=1536*32*32)
        for case in CASES:
            if time.monotonic()-START>CONFIG['soft_limit_seconds']:raise TimeoutError('Focused OCR budget reached')
            im=Image.open(root/case['video']/'01_slides/keyframes'/(case['segment']+'.jpg')).convert('RGB')
            w,h=im.size;box=tuple(round(x*d) for x,d in zip(case['box'],[w,h,w,h]));im=im.crop(box)
            im.save(OUT/'crops'/(case['id']+'.png'))
            prompt='Chép nguyên văn chữ, số và ký hiệu nhìn thấy trong ảnh. Giữ đúng dấu nháy và thụt lề của mã nguồn. Không giải thích, không tính toán hoặc sửa giá trị. Không đoán phần bị che.'
            messages=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':prompt}]}]
            inp=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
            torch.cuda.reset_peak_memory_stats();t=time.monotonic()
            with torch.inference_mode():out=model.generate(**inp,max_new_tokens=768,do_sample=False,repetition_penalty=1.05)
            torch.cuda.synchronize();tokens=out[0,inp['input_ids'].shape[1]:]
            text=processor.decode(tokens,skip_special_tokens=True)
            save(OUT/label/(case['id']+'.json'),{'case':case,'model_id':model_id,'revision':revision,
                'quantization':'NF4 double-quant FP16 compute' if quantized else 'FP16','text':text,
                'seconds':time.monotonic()-t,'loading_seconds':loading,'generated_tokens':len(tokens),
                'image_grid_thw':inp['image_grid_thw'].tolist(),'crop_pixels':box,
                'peak_torch_vram_mib':torch.cuda.max_memory_allocated()/1048576})
            print(label,case['id'],text,flush=True)
    except Exception as e:failure(label,e)
    finally:
        if 'model' in globals():del model
        if 'processor' in globals():del processor
        if 'inp' in globals():del inp
        if 'out' in globals():del out
        gc.collect();torch.cuda.empty_cache()
finish()
