"""PhoWhisper conversion plus a Vietnamese CTC comparison, never invented timings."""
PHO_ID='vinai/PhoWhisper-large'
PHO_REV='b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb'
CHUNK_ID='khanhld/chunkformer-ctc-large-vie'
CHUNK_REV='311fc03558a895dc2b32957f2fb4236c7fb1455b'


def phowhisper():
    install('transformers==4.57.1','accelerate==1.11.0','sentencepiece==0.2.1')
    from ctranslate2.converters import TransformersConverter
    path=snapshot_download(PHO_ID,revision=PHO_REV)
    converted='/kaggle/working/phowhisper_ct2'
    t=time.monotonic()
    converter=TransformersConverter(path,copy_files=['tokenizer.json','preprocessor_config.json'])
    converter.convert(converted,quantization='float16',force=True)
    save(OUT/'phowhisper_conversion.json',{'model_id':PHO_ID,'revision':PHO_REV,
        'quantization':'float16','seconds':time.monotonic()-t})
    run_model('phowhisper',PHO_ID,PHO_REV,local_path=converted)


def chunkformer():
    # Keep the image's matched torch/torchaudio versions, recorded in environment.
    install('chunkformer==1.2.2')
    import torch
    from chunkformer import ChunkFormerModel
    import chunkformer.chunkformer_model as cm
    path=snapshot_download(CHUNK_ID,revision=CHUNK_REV)
    model=ChunkFormerModel.from_pretrained(path).to('cuda:0').eval()
    save(OUT/'environment_chunkformer.json',environment())
    captured=[]
    original_output=cm.get_output_with_timestamps
    def capture(hyps,char_dict,model_type,max_silence_duration):
        captured.append((hyps.detach().cpu().numpy(),char_dict))
        return original_output(hyps,char_dict,model_type,max_silence_duration)
    cm.get_output_with_timestamps=capture
    for video in CONFIG['videos']:
        a,source,wav=prepare_audio(video)
        original=json.loads((root/video/'03_audio/transcript.json').read_text())
        captured.clear();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        result=model.endless_decode(str(wav),chunk_size=64,left_context_size=128,right_context_size=128,
                                   total_batch_duration=300,return_timestamps=True)
        torch.cuda.synchronize();elapsed=time.monotonic()-t
        save(OUT/'chunkformer'/video/'native_segments.json',result)
        hyps,vocab=captured[0];tokens=hyps.reshape(-1);pieces=[];i=0
        np.savez_compressed(OUT/'chunkformer'/video/'ctc_frames.npz',ids=hyps.astype(np.int32))
        save(OUT/'chunkformer'/video/'ctc_vocabulary.json',{'frame_seconds':.08,'blank_id':0,'vocabulary':vocab})
        # Actual CTC emission runs on the documented 80-ms encoder clock.
        # These are not forced word alignments and must not be interpreted as acoustic boundaries.
        while i<len(tokens):
            token=int(tokens[i]);j=i+1
            while j<len(tokens) and int(tokens[j])==token:j+=1
            if token!=0:
                text=vocab[token]
                if text:
                    start=i*.08;end=min(j*.08,source['duration_sec'])
                    if start<end:
                        pieces.append({'start_sec':start,'end_sec':end,'text':text})
            i=j
        groups=[]
        for piece in pieces:
            text=piece['text']
            if text.startswith('▁') or not groups:
                groups.append(dict(piece,text=text.replace('▁',' ').strip()))
            else:
                groups[-1]['text']+=text
                groups[-1]['end_sec']=piece['end_sec']
        words=[]
        for group in groups:
            if not group['text']:continue
            w=next((w for w in original['windows'] if w['core_start_sec']<=group['start_sec']<w['core_end_sec']),original['windows'][-1])
            words.append(dict(group,probability=None,window_id=w['window_id'],word_id=f'word_{len(words):07d}',
                              timing_source='ctc_emission_frame_80ms'))
        transcript=dict(original,words=words,fallback_spans=[],status='completed',invalid_word_count=0,
                        overlap_duplicates_removed=0)
        save(OUT/'chunkformer'/video/'transcript.json',transcript)
        save(OUT/'chunkformer'/video/'metrics.json',{'model_id':CHUNK_ID,'revision':CHUNK_REV,
            'settings':{'chunk_size':64,'left_context_size':128,'right_context_size':128,'total_batch_duration':300},
            'decode_seconds':elapsed,'duration_seconds':source['duration_sec'],'rtf':elapsed/source['duration_sec'],
            'peak_torch_vram_mib':torch.cuda.max_memory_allocated()/1048576,
            'timing_note':'CTC emission runs on an 80ms frame clock; not forced acoustic word alignment'})
        print('chunkformer',video,elapsed,flush=True)


for label,fn in [('phowhisper',phowhisper),('chunkformer',chunkformer)]:
    try:fn()
    except Exception as e:failure(label,e)
finish()
