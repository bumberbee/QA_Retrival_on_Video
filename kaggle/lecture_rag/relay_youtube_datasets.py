"""Builds the YouTube-based thesis datasets on this machine and uploads them to Kaggle.

YouTube blocks downloads from Kaggle servers ("Sign in to confirm you're not a bot"), so the
videos are fetched here, checked and uploaded with the Kaggle CLI to the account. The local copy
is kept (D:/thesis-datasets) for labelling. Re-running skips datasets that are already uploaded
and videos already downloaded.

    python kaggle/lecture_rag/relay_youtube_datasets.py D:/thesis-datasets [dataset ...]

Datasets: vn-lecture-test12, slidespeech-devtest, slidespeech-train100, m3av-maths. Each is flat
(videos next to videos.csv) so no zip copy is needed. Video: H.264 <= 720p + AAC (yt-dlp), longer Vietnamese videos cut to 45 min.
"""

import csv
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset_kernels as kernels  # noqa: E402  (reuses its youtube()/probe() helpers)

exec(kernels.COMMON)  # defines sh, probe, youtube, write_csv, du_gb, pyav_decodes

OWNER = "thnhtrungnguynmif"
WORK = Path(sys.argv[1])
ONLY = {arg for arg in sys.argv[2:] if not arg.startswith("--")}
# --cookies-from-browser=<chrome|edge|firefox> (yt-dlp đọc cookie YouTube của trình duyệt trên máy này)
# hoặc --cookies=<cookies.txt> (file xuất từ trình duyệt). Cookie không rời khỏi máy.
COOKIES = next((("browser", arg.split("=", 1)[1]) for arg in sys.argv[2:] if arg.startswith("--cookies-from-browser=")), None)     or next((("file", arg.split("=", 1)[1]) for arg in sys.argv[2:] if arg.startswith("--cookies=")), None)
SS_BASE = "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/SlideSpeech/"


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def vn_videos():
    source = kernels.VN
    block = source[source.index("VIDEOS = ["):source.index("]\nrows = []") + 1]
    scope = {}
    exec(block, scope)
    return scope["VIDEOS"]


def upload(folder, slug, title, subtitle, licence):
    meta = {"title": title, "id": f"{OWNER}/{slug}", "subtitle": subtitle, "licenses": [{"name": licence}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    for attempt in range(3):
        r = subprocess.run(["kaggle", "datasets", "create", "-p", str(folder), "--dir-mode", "skip", "-q"],
                           capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        log("upload", slug, out[-300:])
        if r.returncode == 0 and "error" not in out.lower():
            return True
        if "already" in out.lower():
            return True
        time.sleep(60)
    return False


def contact_sheet(video, target):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    tiles = []
    for k in range(8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / 8))
        ok, frame = cap.read()
        if ok:
            tiles.append(cv2.resize(frame, (320, 180)))
    cap.release()
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(target), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


class YouTubeBlocked(Exception):
    pass


PAUSE_SEC = (30, 60)    # nghỉ giữa 2 lượt tải: tải dồn dập khiến YouTube chặn cả IP nhà


def fetch(video_id, folder, name, max_sec=None):
    """Download (or reuse) <name>.mp4 in folder; returns the youtube() result dict.
    Stops the whole run on "Sign in to confirm you're not a bot": retrying only extends the block."""
    import random
    target = folder / f"{name}.mp4"
    if target.exists():
        info = probe(target)
        if info and info["height"] and info["has_audio"]:
            return {"status": "ok", **info}
    time.sleep(random.uniform(*PAUSE_SEC))
    result = youtube(video_id, folder, max_sec=max_sec, cookies=COOKIES)
    if result["status"] == "blocked":
        raise YouTubeBlocked(result.get("reason", ""))
    if result["status"] == "ok" and name != video_id:
        (folder / f"{video_id}.mp4").replace(target)
    return result


def ss_map():
    cache = WORK / "_slidespeech_map"
    if not cache.exists():
        cache.mkdir(parents=True)
        archive = cache / "wavid2channel.tar.gz"
        urllib.request.urlretrieve(SS_BASE + "wavid2channel.tar.gz", archive)
        with tarfile.open(archive) as t:
            t.extractall(cache)
    return {f.parent.name: json.load(open(f, encoding="utf-8")) for f in cache.rglob("wavid2channel.json")}


def build_vn(folder):
    rows = []
    for vid, kind, channel, subject, title in vn_videos():
        result = fetch(vid, folder, vid, max_sec=45 * 60)
        log("vn", vid, result["status"], result.get("duration"), result.get("reason", ""))
        if result["status"] == "ok":
            contact_sheet(folder / f"{vid}.mp4", folder / f"{vid}_contact.jpg")
        rows.append({"video_id": vid, "kind": kind, "channel": channel, "subject": subject, "title": title,
                     "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                     "file": f"{vid}.mp4" if result["status"] == "ok" else "",
                     **{k: result.get(k, "") for k in ("status", "duration", "original_duration", "width",
                                                       "height", "vcodec", "acodec", "reason")}})
    write_csv(folder / "videos.csv", rows)
    sponsor = Path(__file__).parent / "runs/20261003-ds-vn-test12/outputs/vn-lecture-test12/sponsorblock/segments.csv"
    if sponsor.exists():
        shutil.copy(sponsor, folder / "sponsorblock_segments.csv")
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {"videos_ok": len(ok), "videos": len(rows), "hours": round(sum(float(r["duration"]) for r in ok) / 3600, 2),
               "by_kind": dict(Counter(r["kind"] for r in ok))}
    (folder / "README.md").write_text(f"""# Vietnamese lecture test set · 12 videos (new channels)

Main test set of the thesis: Vietnamese lecture videos from channels never used for training or
tuning, 5 kinds — TV broadcast (ads, breaks), exam-prep series of another year, livestream
(off-topic talk, breaks, handwriting), university lectures with slides, programming.
{summary['videos_ok']}/{summary['videos']} videos, {summary['hours']} h. Videos longer than 45 min are cut to
their first 45 min (`original_duration` keeps the full length). H.264 <= 720p + AAC.
Research use only; videos stay under their YouTube licence.

- `<id>.mp4` — video; `<id>_contact.jpg` — 8 frames for a quick look
- `videos.csv` — kind, channel, subject, title, status, duration
- `sponsorblock_segments.csv` — SponsorBlock segments found for the thesis videos (few: these
  channels are rarely labelled there)

Labels to add (gold set 2): slide boundaries, per-second ad/break/off-topic, OCR text of ~20
slides and 3 min of hand-checked transcript per video.
""", encoding="utf-8")
    return summary


def build_ss_devtest(folder):
    mapping = ss_map()
    rows = []
    for split in ("dev", "test"):
        for wavid, yt in sorted(mapping.get(split, {}).items()):
            result = fetch(yt, folder, wavid)
            log("ss", split, wavid, result["status"], result.get("duration"), result.get("reason", ""))
            rows.append({"wavid": wavid, "youtube_id": yt, "category": wavid.split("_")[0], "split": split,
                         "file": f"{wavid}.mp4" if result["status"] == "ok" else "",
                         **{k: result.get(k, "") for k in ("status", "duration", "height", "vcodec", "reason")}})
    write_csv(folder / "videos.csv", rows)
    related = folder / "related_files.tar.gz"
    if not related.exists():
        urllib.request.urlretrieve(SS_BASE + "related_files.tar.gz", related)
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {"videos_ok": len(ok), "videos": len(rows), "hours": round(sum(float(r["duration"]) for r in ok) / 3600, 2),
               "status": dict(Counter(r["status"] for r in rows))}
    (folder / "README.md").write_text(f"""# SlideSpeech dev + test

SlideSpeech (Wang et al., ICASSP 2024, https://slidespeech.github.io/), licence CC BY-SA 4.0:
YouTube conference/lecture videos with slides. The authors' video archives were removed, so the
videos were downloaded from YouTube with the official id mapping (wavid2channel).
{summary['videos_ok']}/{summary['videos']} videos, {summary['hours']} h; H.264 <= 720p + AAC.

- `<wavid>.mp4` — video; `videos.csv` — wavid, YouTube id, category, split (dev/test), status
- `related_files.tar.gz` — the authors' segments, transcriptions and slide OCR (all splits)

Role in the thesis: English test set from unseen channels (with 2 held-out LPM speakers).
""", encoding="utf-8")
    return summary


def ss_train_queue():
    by_category = defaultdict(list)
    for wavid, yt in sorted(ss_map().get("train", {}).items()):
        by_category[wavid.split("_")[0]].append((wavid, yt))
    queue, k = [], 0
    while any(k < len(v) for v in by_category.values()):
        for category in sorted(by_category):
            if k < len(by_category[category]):
                queue.append(by_category[category][k])
        k += 1
    return queue


def build_ss_train(folder, n=100, limit_gb=18.0):
    """Round-robin over the training categories until n videos have downloaded."""
    rows = []
    for wavid, yt in ss_train_queue():
        if sum(r["status"] == "ok" for r in rows) >= n or du_gb(folder) >= limit_gb:
            break
        result = fetch(yt, folder, wavid)
        log("ss-train", wavid, result["status"], result.get("duration"), f"{du_gb(folder):.1f} GB")
        rows.append({"wavid": wavid, "youtube_id": yt, "category": wavid.split("_")[0],
                     "file": f"{wavid}.mp4" if result["status"] == "ok" else "",
                     **{k: result.get(k, "") for k in ("status", "duration", "height", "vcodec", "reason")}})
    write_csv(folder / "videos.csv", rows)
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {"videos_ok": len(ok), "tried": len(rows), "hours": round(sum(float(r["duration"]) for r in ok) / 3600, 1),
               "categories": dict(Counter(r["category"] for r in ok))}
    (folder / "README.md").write_text(f"""# SlideSpeech train · 100 videos

{summary['videos_ok']} training videos of SlideSpeech (https://slidespeech.github.io/, CC BY-SA 4.0),
{summary['hours']} h, picked round-robin over the {len(summary['categories'])} training categories and downloaded
from YouTube with the official id mapping (the authors' video archive was removed).
H.264 <= 720p + AAC. `videos.csv`: wavid, YouTube id, category, status. Role in the thesis: varied
slide styles, lead-ins and outros from many channels to train the "lecture content" detector (ad
filter, method 2). Transcripts/OCR: slidespeech-devtest/related_files.tar.gz.
""", encoding="utf-8")
    return summary


def build_m3av(folder):
    labels = folder / "labels"
    if not labels.exists():
        from huggingface_hub import hf_hub_download
        archive = hf_hub_download("CHHHH/M3AV_v1.0", "dataset_v1.0_noaudio.tar.gz", repo_type="dataset",
                                  local_dir=str(WORK / "_m3av"))
        with tarfile.open(archive, "r:gz") as t:
            for member in t:
                if member.isfile() and any(p.upper().startswith("MLS") for p in Path(member.name).parts):
                    t.extract(member, labels)
        Path(archive).unlink()
    ids = {}
    for f in labels.rglob("*.ytbUrl"):
        m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", f.read_text(encoding="utf-8", errors="ignore"))
        if m:
            ids[m.group(1)] = f.name.split(".")[0]
    rows = []
    for yt, m3av_id in sorted(ids.items(), key=lambda item: item[1]):
        result = fetch(yt, folder, m3av_id)
        log("m3av", m3av_id, yt, result["status"], result.get("duration"), result.get("reason", ""))
        rows.append({"m3av_id": m3av_id, "youtube_id": yt, "file": f"{m3av_id}.mp4" if result["status"] == "ok" else "",
                     **{k: result.get(k, "") for k in ("status", "duration", "height", "vcodec", "reason")}})
    write_csv(folder / "videos.csv", rows)
    # labels/ is a folder: pack it so the dataset stays flat
    packed = folder / "m3av_mls_labels.tar.gz"
    with tarfile.open(packed, "w:gz") as t:
        t.add(labels, arcname="labels")
    shutil.rmtree(labels)
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {"videos_ok": len(ok), "videos": len(rows), "hours": round(sum(float(r["duration"]) for r in ok) / 3600, 2)}
    (folder / "README.md").write_text(f"""# M3AV · Oxford Mathematics (MLS)

The mathematics part of M3AV (Chen et al., ACL 2024, https://github.com/Jack-ZC8/M3AV-dataset),
licence CC BY-NC-SA 4.0. `m3av_mls_labels.tar.gz`: for each of the 51 lectures, slide images with
human OCR and LaTeX labels (printed and handwritten, complex formulas) and word-level transcripts.
`<MLS-id>.mp4`: the lecture videos from YouTube ({summary['videos_ok']}/{summary['videos']}, {summary['hours']} h),
H.264 <= 720p + AAC; `videos.csv` maps M3AV id -> YouTube id. Role in the thesis: scoring OCR on
mathematical formulas (character error rate).
""", encoding="utf-8")
    return summary


PLAN = [
    ("vn-lecture-test12", build_vn, "Vietnamese Lecture Test 12", "12 Vietnamese lecture videos from 5 kinds of new channels", "other"),
    ("slidespeech-devtest", build_ss_devtest, "SlideSpeech Dev Test Videos", "SlideSpeech dev and test videos with the authors' labels", "CC-BY-SA-4.0"),
    ("slidespeech-train100", build_ss_train, "SlideSpeech Train 100", "100 SlideSpeech training videos picked across categories", "CC-BY-SA-4.0"),
    ("m3av-maths", build_m3av, "M3AV Oxford Mathematics", "Oxford Mathematics lectures with slide OCR and LaTeX labels", "CC-BY-NC-SA-4.0"),
]

for slug, builder, title, subtitle, licence in PLAN:
    if ONLY and slug not in ONLY:
        continue
    marker = WORK / f"{slug}.uploaded.json"
    if marker.exists():
        log("đã có", slug)
        continue
    folder = WORK / slug
    folder.mkdir(parents=True, exist_ok=True)
    log("bắt đầu", slug)
    try:
        summary = builder(folder)
    except YouTubeBlocked as error:
        log("YouTube chặn tải, DỪNG (chạy lại sau vài giờ, script làm tiếp từ chỗ dừng):", str(error)[:160])
        break
    json.dump(summary, open(folder / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("xong tải", slug, json.dumps(summary, ensure_ascii=False)[:300], f"{du_gb(folder):.1f} GB")
    if upload(folder, slug, title, subtitle, licence):
        marker.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        (folder / "dataset-metadata.json").unlink(missing_ok=True)
        log("đã đưa lên Kaggle (giữ bản trên máy)", slug)
    else:
        log("upload lỗi, giữ thư mục để chạy lại", slug)
        break
log("HẾT")
