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
