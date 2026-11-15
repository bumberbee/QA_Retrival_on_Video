"""Export the twenty fixed source-audio spans for a later human listening audit."""
import hashlib
import json
from pathlib import Path
import wave

from review_spans import SPANS


def main():
    runs=Path(__file__).resolve().parent.parent/'runs/gpu'
    dest=runs/'analysis/span_audio';dest.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for video,spans in SPANS.items():
        source=runs/'asr-v1/audio'/(video+'.wav')
        with wave.open(str(source),'rb') as wav:
            for start,end in spans:
                wav.setpos(round(start*wav.getframerate()))
                frames=wav.readframes(round((end-start)*wav.getframerate()))
                target=dest/f'{video}_{start}_{end}.wav'
                with wave.open(str(target),'wb') as out:
                    out.setparams(wav.getparams());out.writeframes(frames)
                manifest.append({'video':video,'start':start,'end':end,'file':target.name,
                    'source':str(source),'sample_rate':wav.getframerate(),
                    'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(f'Exported {len(manifest)} source clips. No acoustic judgments are claimed.')


if __name__=='__main__':main()
