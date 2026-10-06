"""Generate signal_dump.ipynb: per-second picture signals of 30 videos for content filter A.

For each video of the ad_flag_check list (L25 with gold ads, ad-free English lectures, the 12
Vietnamese test videos of new channels) and every 1 Hz frame: the SigLIP-base image embedding
(float16), the student ad score, the OCR-gate score and the picture change to the previous second
(detection.signature_distance). Saved as signals/<video>.npz so the "outlier vs the video's own
majority" scorer (adfilter.outlier_scores) is tuned locally without the GPU.

    python kaggle/lecture_rag/build_signal_dump.py
"""

from pathlib import Path
import json

import build_ad_flag_check as flag
from build_notebooks import CODE, cell

ROOT = Path(__file__).resolve().parent
NAME = "signal_dump"
DATE = "20261005"
OWNER = "thnhtrungnguynmif"

INTRO = """# Lọc nội dung · A · tín hiệu hình từng giây của 30 video

Mỗi giây: vector SigLIP base, điểm quảng cáo của model trò, điểm bộ gác OCR, độ đổi hình so với
giây trước. Lưu `signals/<video>.npz` để chỉnh cách **tự so với số đông** trên máy, không cần GPU."""

RUN = """import json, time, numpy as np, av, traceback
from pathlib import Path
from PIL import Image

OUT = Path('/kaggle/working/signals'); OUT.mkdir(parents=True, exist_ok=True)
videos = {p.stem: p for p in Path('/kaggle/input').rglob('*.mp4')}
student_path = next(iter(sorted(Path('/kaggle/input').rglob('ad_student.npz'))))
student = dict(np.load(student_path))
embed_images, embed_texts, scale, bias = load_siglip(str(student['backbone']), 'cuda:0')
gate = zero_shot(embed_images, embed_texts, scale, bias, CONTENT_PROMPTS, NO_CONTENT_PROMPTS)
gate_text = embed_texts(list(CONTENT_PROMPTS) + list(NO_CONTENT_PROMPTS))

def frames_1hz(path):
    next_time = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]; stream.thread_type = 'AUTO'
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * frame.time_base)
            if t + 1e-6 < next_time:
                continue
            yield t, frame.to_image()
            next_time = t + 1.0

rows = []
for group, names in VIDEOS.items():
    for name in names:
        if name not in videos:
            print('THIẾU', name); continue
        started = time.monotonic()
        try:
            times, vectors, changes, batch, previous = [], [], [], [], None
            def flush():
                if batch:
                    vectors.append(embed_images([image for _, image in batch]))
                    batch.clear()
            for t, image in frames_1hz(videos[name]):
                signature = frame_signature(image)
                changes.append(0.0 if previous is None else float(signature_distance(previous, signature)))
                previous = signature
                times.append(t); batch.append((t, image.convert('RGB')))
                if len(batch) == 64:
                    flush()
            flush()
            emb = np.vstack(vectors).astype('float32')
            ad = 1 / (1 + np.exp(-(emb @ student['coef'] + float(student['intercept'][0]))))
            logits = emb @ gate_text.T * scale + bias
            logits -= logits.max(1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
            content = probs[:, :len(CONTENT_PROMPTS)].sum(1)
            np.savez_compressed(OUT / f'{name}.npz', times=np.array(times, 'float32'), emb=emb.astype('float16'),
                                student=ad.astype('float32'), gate=content.astype('float32'),
                                change=np.array(changes, 'float32'))
            rows.append({'group': group, 'video': name, 'seconds': len(times), 'took': round(time.monotonic() - started)})
        except Exception:
            traceback.print_exc(); rows.append({'group': group, 'video': name, 'error': True})
        print(rows[-1], flush=True)
json.dump({'videos': VIDEOS, 'rows': rows}, open(OUT / 'index.json', 'w'), indent=1)
import shutil; shutil.make_archive('/kaggle/working/signals', 'zip', OUT)
"""


def build():
    cells = [cell("markdown", INTRO), cell("code", "%pip install -q " + flag.DEPS)]
    cells += [cell("code", (CODE / name).read_text(encoding="utf-8"), hidden=True) for name in flag.MODULES]
    config = flag.CONFIG.split("OUT = ")[0]
    cells += [cell("markdown", "## Video"), cell("code", config),
              cell("markdown", "## Trích tín hiệu"), cell("code", RUN)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    (ROOT / "notebooks" / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    run = ROOT / "runs" / f"{DATE}-signal-dump"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{NAME}.ipynb").write_text(text, encoding="utf-8")
    metadata = json.loads((ROOT / "runs" / f"{DATE}-ad-flag-check" / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata.update(id=f"{OWNER}/l25-signal-dump-{DATE}", title=f"l25-signal-dump-{DATE}", code_file=f"{NAME}.ipynb")
    json.dump(metadata, open(run / "kernel-metadata.json", "w"), indent=2)
    print(run / f"{NAME}.ipynb")


if __name__ == "__main__":
    build()
