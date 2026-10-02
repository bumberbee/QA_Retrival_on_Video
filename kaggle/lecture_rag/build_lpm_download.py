"""Writes lpm_video_download.ipynb: a Kaggle notebook that reads the LPM Lecture Slide
Segments dataset, downloads every YouTube video (video + audio), checks each file and
saves them as Kaggle datasets (shards lpm-videos-01, -02, ...). Re-running the notebook
skips videos already saved and re-downloads anything missing or broken.

    python kaggle/lecture_rag/build_lpm_download.py
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Tải video bộ LPM về Kaggle

Đọc `videos.csv` của dataset **LPM Lecture Slide Segments**, tải toàn bộ 332 video
(có cả hình và tiếng), kiểm tra từng file rồi lưu thành các dataset Kaggle riêng tư
`lpm-videos-01`, `lpm-videos-02`, ... (mỗi phần tối đa `MAX_SHARD_GB`).

**Trước khi chạy**
1. *Add Input* → dataset `thnhtrungnguynmif/lpm-lecture-slide-segments`.
2. *Settings* → **Internet on**.
3. *Add-ons → Secrets*: thêm `KAGGLE_USERNAME` và `KAGGLE_KEY` (để notebook tạo dataset).
4. (Nếu YouTube đòi đăng nhập / báo bot) xuất `cookies.txt` từ trình duyệt, đưa lên một
   dataset riêng tư và *Add Input*; notebook tự tìm file `cookies.txt` trong `/kaggle/input`.

**Kiểm tra mỗi video**: có luồng hình, có luồng tiếng, độ cao khung ≥ `MIN_HEIGHT`,
thời lượng lệch so với `videos.csv` không quá `max(10 s, 3 %)`. File không đạt bị xóa và
tải lại (tối đa `RETRIES` lần, mỗi lần một chuỗi định dạng khác).

**Chạy lại để bổ sung**: notebook đọc `manifest.csv` của các shard đã có, bỏ qua video đã
đạt, chỉ tải phần còn thiếu và lưu thành shard mới. Cuối notebook in bảng video còn thiếu.
""")

code("""
# CONFIG
SRC_SLUG      = 'lpm-lecture-slide-segments'
SHARD_PREFIX  = 'lpm-videos'
MAX_HEIGHT    = 720      # chất lượng tối đa tải về (720p); đổi 1080 nếu cần, dung lượng tăng ~2x
MIN_HEIGHT    = 360      # dưới mức này coi là không đạt
MAX_SHARD_GB  = 15       # mỗi dataset shard; ổ tạm của Kaggle khoảng 70 GB, output 20 GB
RETRIES       = 4        # số lần thử mỗi video trong một vòng
FINAL_ROUNDS  = 2        # số vòng tải bù cuối notebook cho video còn thiếu
DUR_TOL_S, DUR_TOL_FRAC = 10, 0.03
SLEEP_S       = 3        # nghỉ giữa các video để tránh bị YouTube chặn
WORK          = '/kaggle/tmp/lpm'
""")

code("""
!pip install -q -U yt-dlp kaggle
# yt-dlp cần một JavaScript runtime để giải mã link YouTube
!curl -fsSL https://deno.land/install.sh | sh -s -- -y > /dev/null 2>&1 || true
import os
os.environ['PATH'] = os.path.expanduser('~/.deno/bin') + ':' + os.environ['PATH']
!yt-dlp --version; ffmpeg -version | head -1; (deno --version | head -1) || echo 'no deno'
""")

code("""
import csv, glob, json, os, random, re, shutil, subprocess, time
from pathlib import Path

try:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient()
    os.environ['KAGGLE_USERNAME'] = sec.get_secret('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = sec.get_secret('KAGGLE_KEY')
except Exception as e:
    print('Không đọc được Kaggle secrets:', e)
OWNER = os.environ.get('KAGGLE_USERNAME', '')
assert OWNER, 'Thiếu secret KAGGLE_USERNAME / KAGGLE_KEY'

src = glob.glob(f'/kaggle/input/**/{SRC_SLUG}/**/videos.csv', recursive=True) \\
      or glob.glob('/kaggle/input/**/videos.csv', recursive=True)
assert src, 'Chưa Add Input dataset lpm-lecture-slide-segments'
VIDEOS = list(csv.DictReader(open(src[0], encoding='utf-8')))
COOKIES = next(iter(glob.glob('/kaggle/input/**/cookies.txt', recursive=True)), None)
print(len(VIDEOS), 'video trong', src[0]); print('cookies:', COOKIES)
Path(WORK).mkdir(parents=True, exist_ok=True)
""")

md("## 1. Đọc các shard đã có (để bỏ qua video đã tải)")

code("""
def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def existing_shards():
    r = sh(['kaggle', 'datasets', 'list', '--mine', '--search', SHARD_PREFIX, '--csv', '--page-size', '100'])
    slugs = []
    for row in csv.DictReader(r.stdout.splitlines()):
        ref = row.get('ref', '')
        m = re.fullmatch(rf'{re.escape(OWNER)}/{SHARD_PREFIX}-(\\d+)', ref)
        if m:
            slugs.append((int(m.group(1)), ref))
    return sorted(slugs)

def read_manifest(ref):
    d = Path(WORK) / 'manifests' / ref.split('/')[1]
    d.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        sh(['kaggle', 'datasets', 'download', '-d', ref, '-f', 'manifest.csv', '-p', str(d), '--force'])
        f = next(iter(d.glob('manifest.csv*')), None)
        if f:
            if f.suffix == '.zip':
                shutil.unpack_archive(f, d); f = d / 'manifest.csv'
            return list(csv.DictReader(open(f, encoding='utf-8')))
        time.sleep(10)
    return None

SHARDS = existing_shards()
DONE = {}
for idx, ref in SHARDS:
    rows = read_manifest(ref)
    if rows is None:
        print('!! không đọc được manifest của', ref, '(dataset có thể còn đang xử lý) — video của shard này sẽ được tải lại')
        continue
    for r in rows:
        if r['status'] == 'ok':
            DONE[r['video_id']] = ref
    print(ref, len(rows), 'video')
# cũng nhận shard được Add Input trực tiếp
for f in glob.glob(f'/kaggle/input/**/{SHARD_PREFIX}-*/manifest.csv', recursive=True):
    for r in csv.DictReader(open(f, encoding='utf-8')):
        if r['status'] == 'ok':
            DONE.setdefault(r['video_id'], f)
NEXT_SHARD = (SHARDS[-1][0] + 1) if SHARDS else 1
print(f'Đã có {len(DONE)}/{len(VIDEOS)} video · shard tiếp theo: {SHARD_PREFIX}-{NEXT_SHARD:02d}')
""")

md("## 2. Tải và kiểm tra từng video")

code("""
# ưu tiên H.264 (avc1) vì OpenCV/ffmpeg cũ không đọc được AV1
FORMATS = [
    f'bv*[height<={MAX_HEIGHT}][vcodec^=avc1]+ba[ext=m4a]/bv*[height<={MAX_HEIGHT}][ext=mp4]+ba[ext=m4a]',
    f'bv*[height<={MAX_HEIGHT}]+ba/b[height<={MAX_HEIGHT}]',
    f'b[height<={MAX_HEIGHT}][acodec!=none]/bv*+ba/b',
    'bv*+ba/b',
]
PERMANENT = re.compile(r'Private video|unavailable|removed|terminated|copyright|not available|members-only', re.I)

def probe(path):
    r = sh(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', str(path)])
    try:
        j = json.loads(r.stdout)
    except Exception:
        return None
    v = [s for s in j.get('streams', []) if s.get('codec_type') == 'video']
    a = [s for s in j.get('streams', []) if s.get('codec_type') == 'audio']
    return dict(height=int(v[0].get('height', 0)) if v else 0, has_video=bool(v), has_audio=bool(a),
                duration=float(j.get('format', {}).get('duration', 0) or 0),
                vcodec=v[0].get('codec_name', '') if v else '', acodec=a[0].get('codec_name', '') if a else '')

def check(info, expected):
    if not info:              return 'ffprobe lỗi'
    if not info['has_video']: return 'thiếu hình'
    if not info['has_audio']: return 'thiếu tiếng'
    if info['height'] < MIN_HEIGHT: return f"độ phân giải thấp {info['height']}p"
    if expected:
        tol = max(DUR_TOL_S, DUR_TOL_FRAC * expected)
        if abs(info['duration'] - expected) > tol:
            return f"thời lượng {info['duration']:.0f}s ≠ {expected:.0f}s"
    return ''

def download(v, out_dir):
    vid, url = v['video_id'], v['youtube_url']
    expected = float(v['duration_s']) if v.get('duration_s') else None
    out = Path(out_dir) / f'{vid}.mp4'
    last = ''
    for attempt in range(RETRIES):
        for f in Path(out_dir).glob(f'{vid}.*'):
            f.unlink()
        cmd = ['yt-dlp', '-q', '--no-warnings', '--no-playlist', '-f', FORMATS[attempt % len(FORMATS)],
               '--merge-output-format', 'mp4', '--retries', '10', '--fragment-retries', '10',
               '--concurrent-fragments', '4', '-o', str(out), url]
        if COOKIES:
            cmd[1:1] = ['--cookies', COOKIES]
        r = sh(cmd)
        err = (r.stderr or '').strip().splitlines()[-1:] or ['']
        if out.exists():
            info = probe(out)
            last = check(info, expected)
            if not last:
                return dict(status='ok', reason='', attempts=attempt + 1, size_mb=round(out.stat().st_size / 2**20, 1), **info)
            out.unlink()
        else:
            last = err[0][:200] or 'không có file'
            if PERMANENT.search(last):
                return dict(status='unavailable', reason=last, attempts=attempt + 1)
        time.sleep(5 * (attempt + 1) + random.random() * 5)
    return dict(status='failed', reason=last, attempts=RETRIES)
""")

md("## 3. Lưu thành dataset Kaggle theo từng shard")

code("""
MANIFEST_COLS = ['video_id', 'speaker', 'course', 'lecture', 'youtube_url', 'expected_s', 'duration',
                 'height', 'vcodec', 'acodec', 'size_mb', 'attempts', 'status', 'reason', 'file']
SHARD_DIR = Path(WORK) / 'shard'
shard_rows = []
LOG = []   # mọi lần tải trong phiên này

def shard_gb():
    return sum(f.stat().st_size for f in SHARD_DIR.glob('*.mp4')) / 2**30 if SHARD_DIR.exists() else 0

def flush_shard():
    global NEXT_SHARD, shard_rows
    ok = [r for r in shard_rows if r['status'] == 'ok']
    if not ok:
        return
    slug = f'{SHARD_PREFIX}-{NEXT_SHARD:02d}'
    with open(SHARD_DIR / 'manifest.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS); w.writeheader(); w.writerows(ok)
    (SHARD_DIR / 'README.md').write_text(
        f'# LPM lecture videos · part {NEXT_SHARD:02d}\\n\\nVideos of the LPM dataset (Lee et al., ICCV 2023, '
        'https://github.com/dondongwon/LPMDataset) downloaded from YouTube with audio. '
        'See manifest.csv (video_id matches videos.csv / slides.csv of lpm-lecture-slide-segments). '
        'Research use only; videos remain under their original YouTube licence.\\n', encoding='utf-8')
    json.dump({'title': f'LPM lecture videos part {NEXT_SHARD:02d}', 'id': f'{OWNER}/{slug}',
               'licenses': [{'name': 'CC-BY-NC-SA-4.0'}]}, open(SHARD_DIR / 'dataset-metadata.json', 'w'))
    for attempt in range(3):
        r = sh(['kaggle', 'datasets', 'create', '-p', str(SHARD_DIR), '-q'])
        print(r.stdout[-300:], r.stderr[-300:])
        if 'error' not in (r.stdout + r.stderr).lower():
            break
        time.sleep(30)
    else:
        raise RuntimeError(f'Không tạo được dataset {slug}; file vẫn nằm trong {SHARD_DIR}')
    for r in ok:
        DONE[r['video_id']] = f'{OWNER}/{slug}'
    shutil.rmtree(SHARD_DIR); NEXT_SHARD += 1; shard_rows = []

def run_round(todo, label):
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(todo, 1):
        res = download(v, SHARD_DIR)
        row = dict(video_id=v['video_id'], speaker=v['speaker'], course=v['course'], lecture=v['lecture'],
                   youtube_url=v['youtube_url'], expected_s=v.get('duration_s', ''),
                   file=f"{v['video_id']}.mp4" if res['status'] == 'ok' else '',
                   **{k: res.get(k, '') for k in ['duration', 'height', 'vcodec', 'acodec', 'size_mb', 'attempts', 'status', 'reason']})
        shard_rows.append(row); LOG.append(dict(row, round=label))
        print(f"[{label}] {i}/{len(todo)} {v['video_id']} {res['status']} {res.get('height', '')}p "
              f"{res.get('size_mb', '')}MB {res.get('reason', '')}", flush=True)
        if shard_gb() >= MAX_SHARD_GB:
            flush_shard(); SHARD_DIR.mkdir(parents=True, exist_ok=True)
        time.sleep(SLEEP_S)
    flush_shard()
""")

code("""
todo = [v for v in VIDEOS if v['video_id'] not in DONE]
print('Cần tải', len(todo), 'video')
run_round(todo, 'vòng 1')
""")

md("## 4. Tải bù video còn thiếu")

code("""
for k in range(FINAL_ROUNDS):
    last = {r['video_id']: r for r in LOG}
    missing = [v for v in VIDEOS if v['video_id'] not in DONE and last.get(v['video_id'], {}).get('status') != 'unavailable']
    if not missing:
        break
    print(f'Tải bù {len(missing)} video (lần {k + 1}), nghỉ 60 s trước khi chạy')
    time.sleep(60)
    run_round(missing, f'bù {k + 1}')
""")

md("## 5. Báo cáo cuối")

code("""
last = {r['video_id']: r for r in LOG}
rows = []
for v in VIDEOS:
    vid = v['video_id']
    st = 'ok' if vid in DONE else last.get(vid, {}).get('status', 'chưa thử')
    rows.append(dict(video_id=vid, speaker=v['speaker'], status=st, dataset=DONE.get(vid, ''),
                     reason='' if st == 'ok' else last.get(vid, {}).get('reason', '')))
with open('/kaggle/working/lpm_download_report.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

from collections import Counter
print('Tổng kết:', dict(Counter(r['status'] for r in rows)), f'· {len(DONE)}/{len(VIDEOS)} video đã lưu')
ok_log = [r for r in LOG if r['status'] == 'ok']
if ok_log:
    hs = Counter(r['height'] for r in ok_log)
    print('Độ phân giải (phiên này):', dict(sorted(hs.items())),
          f"· dung lượng {sum(float(r['size_mb']) for r in ok_log) / 1024:.1f} GB")
bad = [r for r in rows if r['status'] != 'ok']
if bad:
    print(f'\\n{len(bad)} video chưa có — chạy lại notebook để tải bù; video \"unavailable\" đã bị gỡ/khóa trên YouTube:')
    for r in bad:
        print(f"  {r['video_id']:<14} {r['speaker']:<10} {r['status']:<12} {r['reason']}")
else:
    print('Đủ toàn bộ video.')
print('\\nCác dataset:', sorted(set(DONE.values())))
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
    'kaggle': {'accelerator': 'none', 'isInternetEnabled': True}})
out = HERE / 'notebooks' / 'lpm_video_download.ipynb'
nbf.write(nb, out)
print('wrote', out)
