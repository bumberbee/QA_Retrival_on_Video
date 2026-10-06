"""Writes the 5 Kaggle notebooks that assemble the thesis datasets (runs/<date>-ds-*/):

  1 lpm-videos            244 LPM videos (from thivntrung/dataset-video), checked, train/held-out split
  2 slidespeech-devtest   SlideSpeech dev + test videos and the related files (OCR, transcripts)
  3 slidespeech-train100  100 SlideSpeech train videos, streamed out of the 232 GB archive
  4 vn-test12             12 Vietnamese lecture videos from 5 channel types + SponsorBlock labels
  5 m3av-maths            M3AV Oxford Mathematics: slide OCR/LaTeX labels + the 51 videos

When YouTube blocks a session, *-r2/-r3 continue from the previous output (added as Input);
vn-fill downloads only the replacement of a removed video, merged into vn-lecture-test12 locally.

Each notebook writes its dataset to /kaggle/working/<name>/ with a README.md; in the Kaggle UI,
Output → New Dataset turns it into a dataset of the account. No API key is used in code.

    python kaggle/lecture_rag/build_dataset_kernels.py
"""

import json
from pathlib import Path

from build_notebooks import cell

ROOT = Path(__file__).resolve().parent
OWNER = "thnhtrungnguynmif"
DATE = "20261003"

COMMON = r'''
import csv, glob, json, os, shutil, subprocess, sys, time
from pathlib import Path

def sh(cmd, check=False):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(r.stderr[-800:])
    return r

def probe(path):
    r = sh(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', str(path)])
    try:
        j = json.loads(r.stdout)
    except Exception:
        return None
    v = [s for s in j.get('streams', []) if s.get('codec_type') == 'video']
    a = [s for s in j.get('streams', []) if s.get('codec_type') == 'audio']
    return {'vcodec': v[0].get('codec_name', '') if v else '', 'width': int(v[0].get('width', 0)) if v else 0,
            'height': int(v[0].get('height', 0)) if v else 0, 'has_audio': bool(a),
            'acodec': a[0].get('codec_name', '') if a else '',
            'duration': round(float(j.get('format', {}).get('duration', 0) or 0), 2)}

def pyav_decodes(path, at=30.0):
    """True if PyAV (what stage 01 uses) can decode a frame of this file."""
    try:
        import av
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            c.seek(int(at / float(s.time_base)), stream=s, backward=True, any_frame=False)
            for frame in c.decode(s):
                frame.to_ndarray(format='rgb24')
                return True
    except Exception as e:
        print('  PyAV lỗi', path.name, str(e)[:120])
    return False

def write_csv(path, rows):
    rows = list(rows)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def du_gb(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file()) / 2**30

def install_ytdlp():
    sh([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'yt-dlp[default]'])
    if not shutil.which('deno'):
        os.system('curl -fsSL https://deno.land/install.sh | sh -s -- -y > /dev/null 2>&1')
        os.environ['PATH'] = os.path.expanduser('~/.deno/bin') + ':' + os.environ['PATH']
    print(sh([sys.executable, '-m', 'yt_dlp', '--version']).stdout.strip(), shutil.which('deno'))

BLOCKED_IN_A_ROW = [0]


def youtube(video_id, out, max_height=720, max_sec=None, tries=4, cookies=None, pause=(4, 10)):
    """mp4 with H.264 video (PyAV/OpenCV read it everywhere) and AAC audio; cut to max_sec.
    Uses deno explicitly for YouTube's JS challenge; pauses between videos; after 3 videos in a
    row are refused ("Sign in to confirm you're not a bot") it returns 'blocked' at once."""
    import random
    url = f'https://www.youtube.com/watch?v={video_id}'
    if BLOCKED_IN_A_ROW[0] >= 3:
        return {'status': 'blocked', 'reason': 'dừng: YouTube chặn 3 video liên tiếp'}
    time.sleep(random.uniform(*pause))
    deno = shutil.which('deno') or os.path.expanduser('~/.deno/bin/deno')
    js = ['--js-runtimes', f'deno:{deno}'] if os.path.exists(deno) else []
    formats = [f'bv*[height<={max_height}][vcodec^=avc1]+ba[ext=m4a]',
               f'bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]', f'bv*[height<={max_height}]+ba/b']
    target = Path(out) / f'{video_id}.mp4'
    last = ''
    for attempt in range(tries):
        for f in Path(out).glob(f'{video_id}.*'):
            f.unlink()
        r = sh([sys.executable, '-m', 'yt_dlp', '-q', '--no-warnings', '--ignore-config', *js, '--no-playlist',
                '--socket-timeout', '30', '-f', formats[min(attempt, 2)],
                '--merge-output-format', 'mp4', '--retries', '10', '--fragment-retries', '10',
                *(([('--cookies-from-browser' if cookies[0] == 'browser' else '--cookies'), cookies[1]]) if cookies else []),
                '-o', str(target), url])
        if target.exists():
            info = probe(target)
            if info and info['height'] and info['has_audio']:
                if max_sec and info['duration'] > max_sec + 5:
                    cut = target.with_suffix('.cut.mp4')
                    sh(['ffmpeg', '-v', 'error', '-y', '-i', str(target), '-t', str(max_sec), '-c', 'copy', str(cut)], check=True)
                    original = info['duration']
                    cut.replace(target); info = probe(target); info['original_duration'] = original
                BLOCKED_IN_A_ROW[0] = 0
                return {'status': 'ok', **info}
            last = f'thiếu hình/tiếng: {info}'
        else:
            last = (r.stderr or '').strip().splitlines()[-1:] or ['không có file']
            last = last[0][:200]
            if 'Sign in' in last or 'not a bot' in last:
                BLOCKED_IN_A_ROW[0] += 1
                return {'status': 'blocked', 'reason': last}
        time.sleep(10 * (attempt + 1))
    return {'status': 'failed', 'reason': last}
'''

LPM = r'''
SRC = Path(glob.glob('/kaggle/input/**/lecture_dataset/download_log.csv', recursive=True)[0]).parent
SEG = Path(glob.glob('/kaggle/input/**/slides.csv', recursive=True)[0]).parent      # lpm-lecture-slide-segments
OUT = Path('/kaggle/working/lpm-videos'); (OUT / 'video').mkdir(parents=True, exist_ok=True)
print('nguồn video:', SRC, '| mốc slide:', SEG)
sh([sys.executable, '-m', 'pip', 'install', '-q', 'av>=14,<17'])
meta = {r['video_id']: r for r in csv.DictReader(open(SEG / 'videos.csv', encoding='utf-8'))}
# 4 giảng viên để train, 2 giảng viên giữ riêng để test (không dùng để train hay chỉnh ngưỡng)
SPLIT = {'anat-1': 'train', 'bio-1': 'train', 'bio-4': 'train', 'dental': 'train', 'bio-3': 'test', 'anat-2': 'test'}
rows, skipped = [], []
for mp4 in sorted((SRC / 'video').glob('*.mp4')):
    vid, m = mp4.stem, meta.get(mp4.stem)
    info = probe(mp4) or {}
    expected = float(m['duration_s']) if m else 0
    ok = bool(m) and info.get('height') and info.get('has_audio') and abs(info.get('duration', 0) - expected) <= max(10, 0.03 * expected)
    decodes = pyav_decodes(mp4) if ok else False
    if ok and decodes:
        shutil.copy2(mp4, OUT / 'video' / mp4.name)
        rows.append({'video_id': vid, 'speaker': m['speaker'], 'course': m['course'], 'lecture': m['lecture'],
                     'split': SPLIT.get(m['speaker'], 'other'), 'duration_s': info['duration'], 'n_slides': m['n_slides'],
                     'width': info['width'], 'height': info['height'], 'vcodec': info['vcodec'], 'acodec': info['acodec'],
                     'pyav_decodes': decodes, 'file': f'video/{mp4.name}', 'youtube_url': m['youtube_url']})
    else:
        skipped.append({'video_id': vid, 'reason': 'không khớp metadata/thiếu hình tiếng' if not ok else 'PyAV không đọc được'})
write_csv(OUT / 'videos.csv', rows)
have = {r['video_id'] for r in rows}
slides = [r for r in csv.DictReader(open(SEG / 'slides.csv', encoding='utf-8')) if r['video_id'] in have]
for r in slides:
    r['split'] = SPLIT.get(meta[r['video_id']]['speaker'], 'other')
write_csv(OUT / 'slides.csv', slides)
missing = [{**meta[k], 'reason': 'chưa tải được (YouTube yêu cầu xác minh / video bị gỡ)'} for k in meta if k not in have]
write_csv(OUT / 'missing_videos.csv', missing)
if skipped:
    write_csv(OUT / 'skipped.csv', skipped)
from collections import Counter
by = Counter((r['speaker'], r['split']) for r in rows)
summary = {'videos': len(rows), 'hours': round(sum(r['duration_s'] for r in rows) / 3600, 1), 'slides': len(slides),
           'missing': len(missing), 'skipped': len(skipped), 'size_gb': round(du_gb(OUT), 2),
           'vcodec': dict(Counter(r['vcodec'] for r in rows)), 'by_speaker': {f'{a} ({b})': n for (a, b), n in sorted(by.items())},
           'split': dict(Counter(r['split'] for r in rows))}
json.dump(summary, open(OUT / 'summary.json', 'w'), ensure_ascii=False, indent=1)
(OUT / 'README.md').write_text(f"""# LPM lecture videos (cleaned)

Videos of the LPM dataset (Lee et al., ICCV 2023, https://github.com/dondongwon/LPMDataset),
downloaded from YouTube. Each file was checked: video + audio streams, duration matches LPM
(±3 %), and PyAV decodes it. Licence of the annotations: CC BY-NC-SA 4.0; videos stay under
their YouTube licence (research use).

- `video/<id>.mp4` — {summary['videos']} videos, {summary['hours']} h
- `videos.csv` — speaker, course, lecture, **split** (train: anat-1, bio-1, bio-4, dental;
  test: bio-3, anat-2 — held out, never used for training or tuning), duration, codec
- `slides.csv` — {summary['slides']} LPM slide segments (start/end) of these videos, with split
- `missing_videos.csv` — LPM videos not available ({summary['missing']})

Summary: `summary.json`.
""", encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=1))
'''

SS_OUT = """OUT = Path('/kaggle/working/{name}'); OUT.mkdir(parents=True, exist_ok=True)
WITH_RELATED = {related}"""

SS_META = r'''
install_ytdlp()
BASE = 'https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/SlideSpeech/'
TMP = Path('/kaggle/tmp/ss'); TMP.mkdir(parents=True, exist_ok=True)
# Kho Aliyun đã gỡ các file video nén; còn file nhãn và bảng ánh xạ video -> YouTube (cách tải chính thức).
for name in ('wavid2channel.tar.gz',) + (('related_files.tar.gz',) if WITH_RELATED else ()):
    sh(['curl', '-sS', '-L', '--retry', '5', '-o', str(TMP / name), BASE + name], check=True)
    target = OUT / 'related_files' if name.startswith('related') else TMP / 'map'
    target.mkdir(parents=True, exist_ok=True)
    sh(['tar', '-xzf', str(TMP / name), '-C', str(target)], check=True)
    (TMP / name).unlink()
    print(name, 'xong')
MAP = {}
for f in (TMP / 'map').rglob('wavid2channel.json'):
    MAP[f.parent.name] = json.load(open(f, encoding='utf-8'))
print({k: len(v) for k, v in MAP.items()})
'''

SS_DEVTEST = r'''
rows = []
for split in ('dev', 'test'):
    (OUT / 'video' / split).mkdir(parents=True, exist_ok=True)
    for wavid, yt in sorted(MAP.get(split, {}).items()):
        result = youtube(yt, OUT / 'video' / split)
        if result['status'] == 'ok':
            (OUT / 'video' / split / f'{yt}.mp4').rename(OUT / 'video' / split / f'{wavid}.mp4')
        print(split, wavid, result['status'], result.get('duration'), result.get('reason', ''), flush=True)
        rows.append({'wavid': wavid, 'youtube_id': yt, 'category': wavid.split('_')[0], 'split': split,
                     'file': f'video/{split}/{wavid}.mp4' if result['status'] == 'ok' else '',
                     **{k: result.get(k, '') for k in ('status', 'duration', 'height', 'vcodec', 'reason')}})
write_csv(OUT / 'videos.csv', rows)
from collections import Counter
summary = {'videos_ok': sum(r['status'] == 'ok' for r in rows), 'videos': len(rows),
           'hours': round(sum(float(r['duration'] or 0) for r in rows) / 3600, 2),
           'status': dict(Counter(r['status'] for r in rows)), 'size_gb': round(du_gb(OUT), 2)}
(OUT / 'README.md').write_text(f"""# SlideSpeech dev + test

SlideSpeech (Wang et al., ICASSP 2024, https://slidespeech.github.io/), licence CC BY-SA 4.0:
YouTube conference/lecture videos with slides. The video archives were removed from the
authors' storage, so the videos were downloaded from YouTube with the official id mapping
(wavid2channel). Here: dev and test ({summary['videos_ok']}/{summary['videos']} videos, {summary['hours']} h),
H.264 <= 720p + AAC, and `related_files/` (segments, transcriptions, OCR). `videos.csv`: wavid,
YouTube id, category, split, status. Role in the thesis: English test set from unseen channels
(together with 2 held-out LPM speakers).
""", encoding='utf-8')
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1)
print(summary)
'''

SS_TRAIN = r'''
N, LIMIT_GB = 100, 18.0
(OUT / 'video').mkdir(parents=True, exist_ok=True)
from collections import defaultdict, Counter
by_category = defaultdict(list)
for wavid, yt in sorted(MAP.get('train', {}).items()):
    by_category[wavid.split('_')[0]].append((wavid, yt))
# Chọn rải đều theo lĩnh vực: lần lượt mỗi lĩnh vực một video cho tới khi đủ 100 video tải được.
queue, k = [], 0
while any(k < len(v) for v in by_category.values()):
    for cat in sorted(by_category):
        if k < len(by_category[cat]):
            queue.append(by_category[cat][k])
    k += 1
print(len(by_category), 'lĩnh vực ·', len(queue), 'video train')
rows, tried = [], set()
# Chạy tiếp lần trước (output của notebook trước được Add Input): chép video đã có, bỏ qua video đã thử.
for previous in glob.glob('/kaggle/input/**/slidespeech-train100/videos.csv', recursive=True):
    base = Path(previous).parent
    for r in csv.DictReader(open(previous, encoding='utf-8')):
        if r['wavid'] in tried:
            continue
        if r['status'] == 'ok' and (base / r['file']).exists():
            shutil.copy2(base / r['file'], OUT / r['file'])
            rows.append(r); tried.add(r['wavid'])
        elif r['status'] == 'failed':          # bị gỡ / riêng tư: không thử lại
            rows.append(r); tried.add(r['wavid'])
print('đã có từ lần trước:', sum(r['status'] == 'ok' for r in rows), 'video')
for wavid, yt in queue:
    if wavid in tried:
        continue
    if sum(r['status'] == 'ok' for r in rows) >= N or du_gb(OUT) >= LIMIT_GB:
        break
    result = youtube(yt, OUT / 'video')
    if result['status'] == 'ok':
        (OUT / 'video' / f'{yt}.mp4').rename(OUT / 'video' / f'{wavid}.mp4')
    print(len(rows) + 1, wavid, result['status'], result.get('duration'), f'{du_gb(OUT):.1f} GB', flush=True)
    rows.append({'wavid': wavid, 'youtube_id': yt, 'category': wavid.split('_')[0],
                 'file': f'video/{wavid}.mp4' if result['status'] == 'ok' else '',
                 **{k: result.get(k, '') for k in ('status', 'duration', 'height', 'vcodec', 'reason')}})
    if result['status'] == 'blocked':
        print('YouTube chặn tải; dừng.'); break
write_csv(OUT / 'videos.csv', rows)
ok = [r for r in rows if r['status'] == 'ok']
summary = {'videos_ok': len(ok), 'tried': len(rows), 'hours': round(sum(float(r['duration'] or 0) for r in ok) / 3600, 1),
           'categories': dict(Counter(r['category'] for r in ok)), 'size_gb': round(du_gb(OUT), 2)}
(OUT / 'README.md').write_text(f"""# SlideSpeech train · 100 videos

{summary['videos_ok']} training videos of SlideSpeech (https://slidespeech.github.io/, CC BY-SA 4.0),
{summary['hours']} h, picked round-robin over the {len(summary['categories'])} categories of the training set
and downloaded from YouTube with the official id mapping (the authors' video archive was removed).
H.264 <= 720p + AAC. Role in the thesis: varied slide styles, lead-ins and outros from many channels
to train the "lecture content" detector (ad filter, method 2). Transcripts and OCR: `related_files/`
of the slidespeech-devtest dataset.
""", encoding='utf-8')
json.dump(summary, open(OUT / 'summary.json', 'w'), ensure_ascii=False, indent=1)
print(summary)
'''

VN = r'''
install_ytdlp()
OUT = Path('/kaggle/working/vn-lecture-test12'); (OUT / 'video').mkdir(parents=True, exist_ok=True)
MAX_SEC = 45 * 60
# 12 video, 5 kiểu: truyền hình (quảng cáo, nghỉ), báo ôn thi năm khác L25, livestream (nói ngoài lề,
# nghỉ, viết tay), bài giảng đại học có slide, lập trình. Video dài hơn 45 phút được cắt lấy 45 phút đầu.
VIDEOS = [
    ('tHE1mC_n0-E', 'truyen_hinh', 'HTV - Đài Hà Nội', 'Vật lí', 'Giải đề thi môn Vật lí · tốt nghiệp THPT 2025'),
    ('D_kuulxKuxc', 'truyen_hinh', 'VTV7', 'Toán', 'Toán 2 · Bài 12: Phép trừ (có nhớ) trong phạm vi 20'),  # thay K2qJUcj51zQ (bị gỡ)
    ('bstCZZUMJYw', 'bao_on_thi', 'Báo Thanh Niên', 'Toán', 'Bí quyết ôn thi THPT 2025 · Chuyên đề 9: Xác suất có điều kiện'),
    ('_eDGTHhyQ7M', 'bao_on_thi', 'Báo Thanh Niên', 'Hóa học', 'Bí quyết ôn thi THPT 2025 · Chuyên đề 1: Kim loại nhóm IA, IIA'),
    ('8WnDooAH2a4', 'livestream', 'Thầy Vũ Ngọc Anh', 'Vật lí', 'Chữa đề chính thức tốt nghiệp THPT 2025'),
    ('szRBlpJ0kkY', 'livestream', 'Thầy Nguyễn Quốc Chí', 'Toán', 'Xác suất có điều kiện (phần 1) · lớp 12'),
    ('laT0CKatosY', 'livestream', 'Cpen Team', 'Toán', 'ĐGNL HCM · Toán · Buổi 1'),
    ('xysJI15jpAw', 'dai_hoc', 'Quang Trung TV', 'Xác suất thống kê', 'Chương 2 phần 1: Biến ngẫu nhiên rời rạc'),
    ('v2YTz8ToEag', 'dai_hoc', 'Success and Victory (TS. Phan Thế Công)', 'Kinh tế', 'Bài 1. Tổng quan về kinh tế vi mô'),
    ('hVU5Mw19V_c', 'dai_hoc', 'Nguyễn Thành Nam', 'Xác suất thống kê', 'Công thức xác suất đầy đủ và công thức Bayes'),
    ('xLGGWMDJRkY', 'lap_trinh', 'Tri Thức Mở by The Compiler', 'Lập trình', 'CS50P bài giảng 2: Vòng lặp (tiếng Việt)'),
    ('y1CvB950VlU', 'lap_trinh', 'Gà Lại Lập Trình', 'Lập trình', 'Bài 25: String Python part 1'),
]
rows = []
for vid, kind, channel, subject, title in VIDEOS:
    result = youtube(vid, OUT / 'video', max_sec=MAX_SEC)
    print(vid, kind, result.get('status'), result.get('duration'), result.get('height'), result.get('reason', ''), flush=True)
    rows.append({'video_id': vid, 'kind': kind, 'channel': channel, 'subject': subject, 'title': title,
                 'youtube_url': f'https://www.youtube.com/watch?v={vid}', 'file': f'video/{vid}.mp4' if result['status'] == 'ok' else '',
                 **{k: result.get(k, '') for k in ('status', 'duration', 'original_duration', 'width', 'height', 'vcodec', 'acodec', 'reason')}})
    if result['status'] == 'blocked':
        print('YouTube chặn tải từ Kaggle; các video còn lại tải ở máy cá nhân.');
write_csv(OUT / 'videos.csv', rows)
# Ảnh xem nhanh: 8 khung rải đều mỗi video
import cv2, numpy as np
(OUT / 'contact').mkdir(exist_ok=True)
for r in rows:
    if r['status'] != 'ok':
        continue
    cap = cv2.VideoCapture(str(OUT / r['file'])); n = cap.get(cv2.CAP_PROP_FRAME_COUNT); tiles = []
    for k in range(8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / 8)); ok, f = cap.read()
        if ok:
            tiles.append(cv2.resize(f, (320, 180)))
    if tiles:
        while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(OUT / 'contact' / f"{r['video_id']}.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
from collections import Counter
summary = {'videos_ok': sum(r['status'] == 'ok' for r in rows), 'videos': len(rows),
           'hours': round(sum(float(r['duration'] or 0) for r in rows) / 3600, 2),
           'by_kind': dict(Counter(r['kind'] for r in rows if r['status'] == 'ok')), 'size_gb': round(du_gb(OUT), 2)}
json.dump(summary, open(OUT / 'summary.json', 'w'), ensure_ascii=False, indent=1)
print(summary)
'''

SPONSOR = r'''
# Nhãn SponsorBlock (cộng đồng gán, CC BY-NC-SA 4.0) cho mọi video YouTube của bộ dữ liệu luận văn
ids = {r['video_id']: 'vn-lecture-test12' for r in rows}
lpm_csv = glob.glob('/kaggle/input/**/lpm-lecture-slide-segments/**/videos.csv', recursive=True) or \
          glob.glob('/kaggle/input/**/videos.csv', recursive=True)
for f in lpm_csv:
    for r in csv.DictReader(open(f, encoding='utf-8')):
        if r.get('speaker') and r.get('video_id'):
            ids.setdefault(r['video_id'], 'lpm')
print(len(ids), 'video ID cần tra')
import urllib.request, io
SOURCES = ['https://sponsor.ajay.app/database/sponsorTimes.csv', 'https://sb.ltn.fi/database/sponsorTimes.csv']
found, used = [], None
for url in SOURCES:
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            reader = csv.DictReader(io.TextIOWrapper(response, encoding='utf-8', errors='replace'))
            for n, r in enumerate(reader):
                if r.get('videoID') in ids and r.get('hidden') == '0' and r.get('shadowHidden') == '0' and int(r.get('votes') or 0) >= 0:
                    found.append({'video_id': r['videoID'], 'dataset': ids[r['videoID']], 'start_sec': r['startTime'],
                                  'end_sec': r['endTime'], 'category': r['category'], 'action': r.get('actionType', ''),
                                  'votes': r['votes'], 'uuid': r['UUID']})
                if n % 5_000_000 == 0:
                    print(f'  đã đọc {n:,} dòng, khớp {len(found)}', flush=True)
        used = url; break
    except Exception as e:
        print('Không đọc được', url, str(e)[:160])
SB = OUT / 'sponsorblock'; SB.mkdir(exist_ok=True)
if found:
    write_csv(SB / 'segments.csv', found)
cover = Counter(r['dataset'] for r in {r['video_id']: r for r in found}.values())
summary_sb = {'source': used, 'videos_checked': len(ids), 'segments': len(found),
              'videos_with_labels': dict(cover), 'categories': dict(Counter(r['category'] for r in found))}
json.dump(summary_sb, open(SB / 'summary.json', 'w'), ensure_ascii=False, indent=1)
print(summary_sb)
(OUT / 'README.md').write_text(f"""# Vietnamese lecture test set · 12 videos (new channels)

Main test set of the thesis: Vietnamese lecture videos from channels never used for training or
tuning, 5 kinds — TV broadcast (ads, breaks), exam-prep series of another year, livestream
(off-topic talk, breaks, handwriting), university lectures with slides, programming.
Videos longer than 45 min are cut to their first 45 min (`original_duration` keeps the full length).
Research use only; videos stay under their YouTube licence.

- `video/<id>.mp4` — H.264 ≤ 720p + AAC; `videos.csv` — kind, channel, subject, title, status
- `contact/<id>.jpg` — 8 frames per video for a quick look
- `sponsorblock/segments.csv` — SponsorBlock segments (sponsor, selfpromo, intro, outro, interaction,
  filler, preview) for these videos and the LPM videos; `sponsorblock/summary.json` — coverage

Labels to add (gold set 2): slide boundaries, per-second ad/break/off-topic, OCR text of ~20 slides
and 3 min of hand-checked transcript per video.
""", encoding='utf-8')
'''

M3AV = r'''
install_ytdlp()
sh([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'])
from huggingface_hub import hf_hub_download
OUT = Path('/kaggle/working/m3av-maths'); OUT.mkdir(parents=True, exist_ok=True)
TMP = Path('/kaggle/tmp/m3av'); TMP.mkdir(parents=True, exist_ok=True)
archive = hf_hub_download('CHHHH/M3AV_v1.0', 'dataset_v1.0_noaudio.tar.gz', repo_type='dataset', local_dir=str(TMP))
print('tải', round(Path(archive).stat().st_size / 2**30, 2), 'GB')
import tarfile
with tarfile.open(archive, 'r:gz') as t:
    names = [m.name for m in t if m.isfile()]
def is_mls(name):
    return any(part.upper().startswith('MLS') for part in Path(name).parts)
mls = [n for n in names if is_mls(n)]
print(len(names), 'file trong file nén ·', len(mls), 'file của Oxford Mathematics (MLS)')
print('ví dụ:', names[:5], mls[:8])
print('thư mục gốc:', sorted({n.split('/')[0] for n in names})[:10])
wanted = set(mls) | {n for n in names if n.endswith('.json') and n.count('/') <= 1}
with tarfile.open(archive, 'r:gz') as t:
    for member in t:
        if member.name in wanted:
            t.extract(member, OUT)
Path(archive).unlink()
for f in ('set_split.json', 'paper_link.json'):
    try:
        shutil.copy(hf_hub_download('CHHHH/M3AV_v1.0', f, repo_type='dataset', local_dir=str(TMP)), OUT / f)
    except Exception as e:
        print('không có', f, str(e)[:80])
# Tìm YouTube ID của từng video MLS trong file metadata
import re
video_dirs = sorted({next(p for p in Path(n).parts if p.upper().startswith('MLS')) for n in mls})
ids = {}
for f in OUT.rglob('*.ytbUrl'):
    m = re.search(r'(?:v=|youtu\.be/)([\w-]{11})', f.read_text(encoding='utf-8', errors='ignore'))
    if m:
        ids[m.group(1)] = f.name.split('.')[0]          # YouTube ID -> thư mục M3AV (MLS-xxxxx)
print(len(video_dirs), 'thư mục video MLS ·', len(ids), 'YouTube ID tìm thấy trong metadata')
(OUT / 'video').mkdir(exist_ok=True)
rows = []
for vid, where in sorted(ids.items()):
    result = youtube(vid, OUT / 'video')
    if result['status'] == 'ok':
        (OUT / 'video' / f'{vid}.mp4').rename(OUT / 'video' / f'{where}.mp4')
    print(where, vid, result.get('status'), result.get('duration'), result.get('reason', ''), flush=True)
    rows.append({'m3av_id': where, 'youtube_id': vid, 'file': f'video/{where}.mp4' if result['status'] == 'ok' else '', **{k: result.get(k, '') for k in ('status', 'duration', 'height', 'vcodec', 'reason')}})
    if result['status'] == 'blocked':
        break
if rows:
    write_csv(OUT / 'videos.csv', rows)
summary = {'mls_entries': len(mls), 'video_dirs': len(video_dirs), 'youtube_ids': len(ids),
           'videos_ok': sum(r['status'] == 'ok' for r in rows), 'size_gb': round(du_gb(OUT), 2)}
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1)
(OUT / 'README.md').write_text(f"""# M3AV · Oxford Mathematics (MLS)

The mathematics part of M3AV (Chen et al., ACL 2024, https://github.com/Jack-ZC8/M3AV-dataset),
licence CC BY-NC-SA 4.0: slide images with human OCR and LaTeX labels (printed and handwritten,
complex formulas) and word-level transcripts, plus the lecture videos from YouTube (`video/`).
Role in the thesis: scoring OCR on mathematical formulas (character error rate).
Summary: `summary.json`.
""", encoding='utf-8')
print(summary)
'''

KERNELS = {
    "lpm-videos": dict(code=[LPM], datasets=["thivntrung/dataset-video", f"{OWNER}/lpm-lecture-slide-segments"], kernels=[]),
    "slidespeech-devtest": dict(code=[SS_OUT.format(name="slidespeech-devtest", related=True), SS_META, SS_DEVTEST],
                                datasets=[], kernels=[]),
    # 100 video sau 4 lần chạy (YouTube chặn giữa chừng); lần cuối nhận output lần 3 làm Input, bỏ qua
    # video bị chặn, chỉ dừng khi 3 video liền bị chặn. Các lần 1–3 đã xóa khỏi repo.
    "slidespeech-train100-r4": dict(code=[SS_OUT.format(name="slidespeech-train100", related=False), SS_META,
                                          SS_TRAIN.replace("N, LIMIT_GB = 100, 18.0", "N, LIMIT_GB = 100, 19.2").replace(
                                              "    if result['status'] == 'blocked':\n        print('YouTube chặn tải; dừng.'); break",
                                              "    if result['status'] == 'blocked' and BLOCKED_IN_A_ROW[0] >= 3:\n"
                                              "        print('YouTube chặn 3 video liên tiếp; dừng.'); break")],
                                    datasets=[], kernels=[f"{OWNER}/thesis-ds-slidespeech-train100-r3"]),
    "vn-test12": dict(code=[VN, SPONSOR], datasets=[f"{OWNER}/lpm-lecture-slide-segments"], kernels=[]),
    # chỉ tải video thay cho K2qJUcj51zQ (bị gỡ); máy cá nhân gộp vào bộ vn-lecture-test12 rồi lên phiên bản mới
    "vn-fill": dict(code=[VN.replace("vn-lecture-test12'", "vn-fill'").replace(
                        "rows = []\nfor vid", "VIDEOS = [v for v in VIDEOS if v[0] == 'D_kuulxKuxc']\nrows = []\nfor vid")],
                    datasets=[], kernels=[]),
    "m3av-maths": dict(code=[M3AV], datasets=[], kernels=[]),
}


def build():
    for name, spec in KERNELS.items():
        cells = [cell("markdown", f"# Bộ dữ liệu luận văn · {name}\n\n" + __doc__.split("\n\n")[1]),
                 cell("code", COMMON)] + [cell("code", code) for code in spec["code"]]
        notebook = {"cells": cells,
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python"}},
                    "nbformat": 4, "nbformat_minor": 5}
        nb_name = f"ds_{name.replace('-', '_')}.ipynb"
        (ROOT / "notebooks" / nb_name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        run = ROOT / "runs" / f"{DATE}-ds-{name}"
        run.mkdir(parents=True, exist_ok=True)
        (run / nb_name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        json.dump({"id": f"{OWNER}/thesis-ds-{name}", "title": f"thesis-ds-{name}", "code_file": nb_name,
                   "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
                   "enable_tpu": False, "enable_internet": True, "dataset_sources": spec["datasets"],
                   "competition_sources": [], "kernel_sources": spec["kernels"]},
                  open(run / "kernel-metadata.json", "w"), indent=2)
        print(run.name)


if __name__ == "__main__":
    build()
