"""Optional fixed-context answer comparison; no retrieval tuning or automated truth claims."""
import base64
import requests
import zlib
QUESTIONS=json.loads(zlib.decompress(base64.b64decode(CONFIG['questions_payload'])))
save(OUT/'questions.json',QUESTIONS)
SYSTEM='Trả lời câu hỏi chỉ dựa vào bằng chứng bài giảng được cung cấp. Trả lời ngắn gọn, chính xác, giữ đúng số và mã nguồn. Nếu bằng chứng thiếu, nói rõ không đủ thông tin. Không tự thêm kiến thức ngoài bài giảng.'


def prompt(q):
    value=SYSTEM+'\n\n--- BẰNG CHỨNG ---\n'+q['context']+'\n--- HẾT ---\nCâu hỏi: '+q['question']
    return value+'\n/no_think' if CONFIG.get('explicit_no_think') else value


def ollama():
    version='0.12.6';sha='de82adce2ab79235115d511ff22fcb099ac53b67127870f12b80198c033ec0a1'
    path=Path('/kaggle/working/ollama.tgz')
    r=requests.get('https://github.com/ollama/ollama/releases/download/v'+version+'/ollama-linux-amd64.tgz',stream=True,timeout=60)
    r.raise_for_status()
    with path.open('wb') as f:
        for block in r.iter_content(1048576):f.write(block)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Ollama binary archive checksum mismatch')
    install_dir=Path('/kaggle/working/ollama');install_dir.mkdir(exist_ok=True)
    subprocess.run(['tar','-xzf',str(path),'-C',str(install_dir)],check=True,timeout=180)
    env=dict(os.environ,OLLAMA_HOST='127.0.0.1:11434',OLLAMA_MODELS='/kaggle/working/ollama_models',CUDA_VISIBLE_DEVICES='0')
    log=(OUT/'ollama_server.log').open('w')
    server=subprocess.Popen([str(install_dir/'bin/ollama'),'serve'],stdout=log,stderr=log,env=env)
    try:
        for _ in range(60):
            try:
                requests.get('http://127.0.0.1:11434/api/version',timeout=2).raise_for_status();break
            except requests.RequestException:time.sleep(1)
        pins={'qwen2.5:7b':'845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e',
              'qwen3:4b':'359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7'}
        for model,digest in pins.items():
            if model not in CONFIG.get('ollama_models',pins):continue
            try:
                name,tag=model.split(':')
                manifest=requests.get('https://registry.ollama.ai/v2/library/'+name+'/manifests/'+tag,timeout=30)
                manifest.raise_for_status()
                if hashlib.sha256(manifest.content).hexdigest()!=digest:raise ValueError('Model alias changed since pinning')
                pull=requests.post('http://127.0.0.1:11434/api/pull',json={'model':model,'stream':False},timeout=600)
                pull.raise_for_status()
                info=requests.post('http://127.0.0.1:11434/api/show',json={'model':model},timeout=30).json()
                save(OUT/(model.replace(':','_')+'_model.json'),{'ollama_version':version,'archive_sha256':sha,
                    'manifest_sha256':digest,'model':model,'info':info})
                for q in QUESTIONS:
                    if time.monotonic()-START>CONFIG['soft_limit_seconds']:raise TimeoutError('Answer experiment budget reached')
                    t=time.monotonic()
                    content=prompt(q)
                    raw_mode=CONFIG.get('raw_closed_thinking',False)
                    if raw_mode:
                        content='<|im_start|>user\n'+content+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
                    result=requests.post('http://127.0.0.1:11434/api/generate',json={'model':model,'prompt':content,
                        'raw':raw_mode,'think':False,'stream':False,'keep_alive':'5m','options':{'temperature':0,'num_ctx':8192,'num_predict':384}},timeout=120)
                    result.raise_for_status();data=result.json()
                    save(OUT/model.replace(':','_')/(q['id']+'.json'),{'id':q['id'],'answer':data.get('response'),
                        'seconds':time.monotonic()-t,'raw':data,'vram_mib':vram(),
                        'prompt_format':'ChatML with closed empty think block' if raw_mode else 'Ollama model template',
                        'prompt_sha256':hashlib.sha256(content.encode()).hexdigest()})
                    print(model,q['id'],flush=True)
                requests.post('http://127.0.0.1:11434/api/generate',json={'model':model,'keep_alive':0},timeout=60)
            except Exception as e:failure(model.replace(':','_'),e)
    finally:
        server.terminate();server.wait(timeout=30);log.close()


def vintern():
    install('transformers==4.57.1','huggingface-hub==0.36.0','sentencepiece==0.2.1','einops==0.8.1','timm==1.0.22')
    import torch
    import torchvision.transforms as T
    from transformers import AutoModel,AutoTokenizer
    from PIL import Image
    model_id='5CD-AI/Vintern-1B-v3_5';revision='b98f263eab246eb5269ade64edbdca8a887dc44d'
    model=AutoModel.from_pretrained(model_id,revision=revision,torch_dtype=torch.float16,
        trust_remote_code=True,low_cpu_mem_usage=True,use_flash_attn=False).eval().to('cuda:0')
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=True,use_fast=False)
    transform=T.Compose([T.Resize((448,448),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    root=export_root()
    for q in QUESTIONS:
        if time.monotonic()-START>CONFIG['soft_limit_seconds']:raise TimeoutError('Answer experiment budget reached')
        try:
            im=Image.open(root/q['video']/'01_slides/keyframes'/(q['image_segment']+'.jpg')).convert('RGB')
            # 2x2 tiles plus a full-frame thumbnail; explicit, deterministic image budget.
            w,h=im.size;tiles=[im.crop((x*w//2,y*h//2,(x+1)*w//2,(y+1)*h//2)) for y in range(2) for x in range(2)]+[im]
            pixels=torch.stack([transform(x) for x in tiles]).to(device='cuda:0',dtype=torch.float16)
            torch.cuda.reset_peak_memory_stats();t=time.monotonic()
            answer=model.chat(tokenizer,pixels,'<image>\n'+prompt(q),
                dict(max_new_tokens=384,do_sample=False,num_beams=1,repetition_penalty=1.05),history=None)
            torch.cuda.synchronize()
            save(OUT/'vintern'/(q['id']+'.json'),{'id':q['id'],'model_id':model_id,'revision':revision,
                'answer':answer,'seconds':time.monotonic()-t,'image_segment':q['image_segment'],
                'peak_torch_vram_mib':torch.cuda.max_memory_allocated()/1048576})
            print('vintern',q['id'],flush=True)
        except Exception as e:failure('vintern_'+q['id'],e)


for label,fn in [('ollama',ollama)]+([('vintern',vintern)] if CONFIG.get('run_vintern',True) else []):
    try:fn()
    except Exception as e:failure(label,e)
save(OUT/'environment.json',environment())
finish()
