"""Validate measured outputs and build compact before/after evidence sheets."""
from pathlib import Path
from PIL import Image, ImageDraw
from experiment import load_detection, RUNS, OLD, VIDEOS

def sheet(choices,path,columns=3,width=426):
    height=265
    canvas=Image.new('RGB',(columns*width,height*((len(choices)+columns-1)//columns)),'white')
    draw=ImageDraw.Draw(canvas)
    for n,(root,item,label) in enumerate(choices):
        with Image.open(root/item['image_path']) as opened:
            image=opened.convert('RGB');image.thumbnail((width,235))
        x,y=n%columns*width,n//columns*height
        canvas.paste(image,(x,y))
        draw.text((x,y+235),label,fill='black')
        draw.text((x,y+249),f"[{item['start_sec']:.1f}, {item['end_sec']:.1f}) @ {item['keyframe_time_sec']:.1f}s",fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=88)

def main():
    m=load_detection();out=RUNS/'evidence';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for video in [p.stem for p in sorted(VIDEOS.glob('*.mp4'))]:
        root=RUNS/'recommended'/video/'01_slides'
        items=m.read_json(root/'intervals.json')
        source=m.read_json(root/'manifest.json')['source']
        m.verify_intervals(items,source['duration_sec'])
        for item in items:
            for state in item['reveal_states']:
                assert item['start_sec']<=state['start_sec']<state['end_sec']<=item['end_sec']
                assert state['start_sec']<=state['keyframe_time_sec']<state['end_sec']
                for extra in state.get('complementary_keyframes',[]):
                    assert state['start_sec']<=extra['keyframe_time_sec']<state['end_sec']
        baseline=m.read_json(RUNS/'baseline'/video/'01_slides/intervals.json')
        original=m.read_json(OLD/video/'01_slides/intervals.json')
        assert baseline==original, f'Legacy defaults changed: {video}'
        row=m.read_json(RUNS/'recommended'/video/'measurement.json')
        row['default_json_parity']=True;rows.append(row)
        sheet([(root,i,str(i['sequence'])) for i in items],out/(video+'_full.jpg'),columns=4,width=320)
    m.write_json(RUNS/'final-measurements.json',rows)
    m.write_json(RUNS/'recommended/measurements.json',rows)
    v='hVU5Mw19V_c';oldroot=OLD/v/'01_slides';newroot=RUNS/'recommended'/v/'01_slides'
    old=m.read_json(oldroot/'intervals.json');new=m.read_json(newroot/'intervals.json')
    choices=[(oldroot,old[0],'Before: one frame for first 579s'),
             (oldroot,old[6],'Before: occluded; later example absent')]
    for t in (180,300,480,940,1040,1190,1230,1300,1400):
        item=next(i for i in new if i['start_sec']<=t<i['end_sec'])
        choices.append((newroot,item,f'After: state near {t}s'))
    sheet(choices,out/'board_before_after.jpg')
    for v in ('L25_V008','D_kuulxKuxc'):
        oldroot=OLD/v/'01_slides';newroot=RUNS/'recommended'/v/'01_slides'
        old=m.read_json(oldroot/'intervals.json');new=m.read_json(newroot/'intervals.json')
        old_starts={i['start_sec'] for i in old};new_starts={i['start_sec'] for i in new}
        choices=[]
        for t in sorted(old_starts-new_starts):
            pos=next(n for n,i in enumerate(old) if i['start_sec']==t)
            choices.extend([(oldroot,old[pos-1],f'Before {pos-1}'),(oldroot,old[pos],f'Before {pos}')])
            after=next(i for i in new if i['start_sec']<=t<i['end_sec'])
            choices.append((newroot,after,'After same-content group'))
        sheet(choices,out/(v+'_view_joins.jpg'))
    v='y1CvB950VlU';oldroot=OLD/v/'01_slides';newroot=RUNS/'recommended'/v/'01_slides'
    old=m.read_json(oldroot/'intervals.json');new=m.read_json(newroot/'intervals.json')
    choices=[(oldroot,old[37],'Before: replacement merged'),(oldroot,old[38],'Before: popup selected')]
    choices += [(newroot,i,'After source '+','.join(str(s['source_state']) for s in i['reveal_states']))
                for i in new if 620<=i['start_sec']<650]
    sheet(choices,out/'code_before_after.jpg')
    print('Validated',len(rows),'videos; exact legacy JSON parity; interval/state/complement timestamps valid.')
    print('Evidence:',out)

if __name__=='__main__':
    main()
