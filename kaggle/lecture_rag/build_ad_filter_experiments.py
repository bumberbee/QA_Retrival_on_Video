"""Writes notebooks/ad_filter_experiments.ipynb: experiments E1 and E4 of the ad filter.

E4 (teacher): SigLIP so400m zero-shot, "lecture content" prompts against "ad / promo" prompts.
E1 (student): logistic regression on the image embeddings of a small SigLIP, trained on the
hand labels of the 4 tuning videos only. E1+motion adds the motion signal (secondary cue).
Everything is scored on the 6 held-out videos, per second and per ad boundary, before and
after the same post-processing the pipeline will use (smoothing, ads >= 5 s, gaps <= 2 s).

    python kaggle/lecture_rag/build_ad_filter_experiments.py
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
# Bộ lọc quảng cáo · thí nghiệm E1 và E4

Input: dataset **l25-ad-filter-frames** (khung 1 Hz của 10 video L25, nhãn 1 = quảng cáo).
Thiết bị: GPU T4.

| # | Cấu hình | Học từ |
|---|---|---|
| E4 | **Thầy**: SigLIP so400m zero-shot, so câu mô tả "bài giảng" với "quảng cáo" | không học |
| E1 | **Trò**: hồi quy logistic trên embedding ảnh của SigLIP base | nhãn tay 4 video tuning |
| E1+m | E1 + mức chuyển động | như E1 |

Tất cả đo trên **6 video test**, theo từng giây và theo ranh giới đầu/cuối quảng cáo (±2 s),
trước và sau hậu xử lý (làm mượt, đoạn ≥ 5 s, lấp lỗ ≤ 2 s).
""")

code("""
TEACHER = 'google/siglip-so400m-patch14-384'
STUDENT = 'google/siglip-base-patch16-224'
MIN_AD_SEC, MAX_GAP_SEC, SMOOTH_SEC, TOL_SEC = 5, 2, 5, 2
BATCH = 64
LESSON_PROMPTS = [
    'a slide of a lecture presentation with text',
    'a lecture slide with a diagram, chart or formula',
    'a teacher presenting a lesson next to a slide',
    'the title slide of a lesson',
    'a page of a document or an exam with questions',
]
AD_PROMPTS = [
    'a video advertisement',
    'a promotional video of a school or university',
    'a logo animation',
    'students walking on a school campus',
    'people in a commercial',
    'a building exterior',
]
""")

code("""
import csv, glob, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from sklearn.linear_model import LogisticRegression

base = Path(glob.glob('/kaggle/input/**/frames.csv', recursive=True)[0]).parent
rows = list(csv.DictReader(open(base / 'frames.csv', encoding='utf-8')))
for r in rows:
    r['second'], r['label'], r['motion'] = int(r['second']), int(r['label']), float(r['motion'])
videos = sorted({r['video'] for r in rows})
assert sum(r['label'] for r in rows) == 2068, 'dataset chưa phải bản nhãn mới (version 3)'
split = {r['video']: r['split'] for r in rows}
print(len(rows), 'khung ·', {s: sum(1 for v in videos if split[v] == s) for s in ('tuning', 'held-out')}, 'video')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import transformers; print('transformers', transformers.__version__)

def as_tensor(output):
    # transformers >= 5 trả về BaseModelOutputWithPooling thay vì tensor
    return output if torch.is_tensor(output) else output.pooler_output
""")

md("## 1. Embedding ảnh (thầy và trò)")

code("""
def embed_images(model_id):
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    out, started = [], time.time()
    for i in range(0, len(rows), BATCH):
        images = [Image.open(base / r['image']).convert('RGB') for r in rows[i:i + BATCH]]
        inputs = processor(images=images, return_tensors='pt').to(device)
        with torch.no_grad():
            e = as_tensor(model.get_image_features(pixel_values=inputs['pixel_values'].half()))
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    seconds = time.time() - started
    print(f'{model_id}: {seconds:.0f} s cho {len(rows)} khung · {1000 * seconds / len(rows):.1f} ms/khung')
    return np.concatenate(out), model, processor, seconds

teacher_emb, teacher, teacher_proc, teacher_sec = embed_images(TEACHER)
""")

code("""
with torch.no_grad():
    text = teacher_proc(text=LESSON_PROMPTS + AD_PROMPTS, padding='max_length', return_tensors='pt').to(device)
    t = torch.nn.functional.normalize(as_tensor(teacher.get_text_features(**text)).float(), dim=-1).cpu().numpy()
    scale, bias = teacher.logit_scale.exp().item(), teacher.logit_bias.item()
logits = teacher_emb @ t.T * scale + bias
# xác suất "quảng cáo": softmax trên mọi câu mô tả, cộng phần của các câu quảng cáo
p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
teacher_score = p[:, len(LESSON_PROMPTS):].sum(1)
del teacher; torch.cuda.empty_cache()
student_emb, student, _, student_sec = embed_images(STUDENT)
del student; torch.cuda.empty_cache()
""")

md("## 2. Hậu xử lý và cách chấm")

code("""
def by_video(values):
    out, i = {}, 0
    for v in videos:
        n = sum(1 for r in rows if r['video'] == v)
        out[v] = np.asarray(values[i:i + n]); i += n
    return out

LABEL = by_video([r['label'] for r in rows])
MOTION = by_video([r['motion'] for r in rows])

def runs(mask):
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            j = k
            while j < len(mask) and mask[j]: j += 1
            out.append([k, j]); k = j
        else:
            k += 1
    return out

def postprocess(score, threshold):
    pad = (SMOOTH_SEC // 2, SMOOTH_SEC - 1 - SMOOTH_SEC // 2)
    s = np.convolve(np.pad(score, pad, mode='edge'), np.ones(SMOOTH_SEC) / SMOOTH_SEC, mode='valid')
    mask = (s >= threshold).astype(int)
    for a, b in runs(1 - mask):                    # lấp lỗ ngắn giữa hai đoạn quảng cáo
        if 0 < a and b < len(mask) and b - a <= MAX_GAP_SEC: mask[a:b] = 1
    for a, b in runs(mask):                        # bỏ đoạn quá ngắn, trừ logo ngắn ở cuối video
        if b - a < MIN_AD_SEC and not (b == len(mask) and b - a >= 2): mask[a:b] = 0
    return mask

def score_video(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & (1 - truth)).sum()); fn = int(((1 - pred) & truth).sum())
    tb = [x for a, b in runs(truth) for x in (a, b)]
    pb = [x for a, b in runs(pred) for x in (a, b)]
    hit = sum(any(abs(x - y) <= TOL_SEC for y in pb) for x in tb)
    good = sum(any(abs(x - y) <= TOL_SEC for y in tb) for x in pb)
    return dict(tp=tp, fp=fp, fn=fn, true_bounds=len(tb), hit=hit, pred_bounds=len(pb), good=good)

def summary(per_video):
    s = {k: sum(m[k] for m in per_video.values()) for k in next(iter(per_video.values()))}
    p = s['tp'] / max(s['tp'] + s['fp'], 1); r = s['tp'] / max(s['tp'] + s['fn'], 1)
    return dict(P=round(p, 3), R=round(r, 3), F1=round(2 * p * r / max(p + r, 1e-9), 3),
                bound_R=round(s['hit'] / max(s['true_bounds'], 1), 3),
                bound_P=round(s['good'] / max(s['pred_bounds'], 1), 3))

def evaluate(name, score_of, threshold):
    out = {'threshold': round(float(threshold), 2)}
    for post in (False, True):
        pv = {}
        for v in videos:
            if split[v] != 'held-out': continue
            sc = score_of[v]
            pred = postprocess(sc, threshold) if post else (sc >= threshold).astype(int)
            pv[v] = score_video(pred, LABEL[v])
        out['post' if post else 'raw'] = summary(pv)
        if post: out['per_video'] = {v: summary({v: m}) for v, m in pv.items()}
    print(f"{name:18} ngưỡng {threshold:.2f} | thô F1 {out['raw']['F1']:.3f} | "
          f"hậu xử lý P {out['post']['P']:.3f} R {out['post']['R']:.3f} F1 {out['post']['F1']:.3f} · "
          f"ranh giới ±{TOL_SEC}s R {out['post']['bound_R']:.3f} P {out['post']['bound_P']:.3f}")
    return out

def tune_threshold(score_of):
    # chọn ngưỡng chỉ trên 4 video tuning
    best = max(np.arange(0.05, 0.96, 0.05), key=lambda th: summary(
        {v: score_video(postprocess(score_of[v], th), LABEL[v]) for v in videos if split[v] == 'tuning'})['F1'])
    return float(best)
""")

md("## 3. E4 · thầy zero-shot")

code("""
TEACHER_SCORE = by_video(teacher_score)
results = {}
results['E4 thầy (0,5)'] = evaluate('E4 thầy (0,5)', TEACHER_SCORE, 0.5)
th = tune_threshold(TEACHER_SCORE)
results['E4 thầy (chỉnh)'] = evaluate('E4 thầy (chỉnh)', TEACHER_SCORE, th)
""")

md("## 4. E1 · trò học nhãn tay của 4 video tuning")

code("""
def train_student(features, name):
    train = np.array([split[r['video']] == 'tuning' for r in rows])
    y = np.array([r['label'] for r in rows])
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(features[train], y[train])
    score = by_video(clf.predict_proba(features)[:, 1])
    results[name] = evaluate(name, score, tune_threshold(score))
    return clf, score

def motion_features():
    m = np.array([r['motion'] for r in rows])
    smooth = np.concatenate([np.convolve(MOTION[v], np.ones(5) / 5, mode='same') for v in videos])
    return np.stack([m, smooth], 1) * 10

clf_e1, STUDENT_SCORE = train_student(student_emb, 'E1 trò')
clf_e1m, _ = train_student(np.hstack([student_emb, motion_features()]), 'E1+m trò + chuyển động')
clf_t, _ = train_student(teacher_emb, 'E1-t trò trên đặc trưng thầy')
""")

md("## 5. Tổng kết, tốc độ và lưu model trò")

code("""
print(f'{"Cấu hình":32} {"P":>6} {"R":>6} {"F1":>6} {"RG-R":>6} {"RG-P":>6}')
for name, r in results.items():
    q = r['post']
    print(f'{name:32} {q["P"]:6.3f} {q["R"]:6.3f} {q["F1"]:6.3f} {q["bound_R"]:6.3f} {q["bound_P"]:6.3f}')
print('\\nRG = ranh giới đầu/cuối quảng cáo khớp trong ±2 s. Đo trên 6 video test, sau hậu xử lý.')
speed = {'teacher_ms_per_frame': 1000 * teacher_sec / len(rows), 'student_ms_per_frame': 1000 * student_sec / len(rows)}
print({k: round(v, 1) for k, v in speed.items()})
# ad_student.npz (dùng trong pipeline): E1, chỉ ảnh. ad_student_motion.npz: E1+m, để so sánh.
best = 'E1 trò'
for path, name, clf in (('/kaggle/working/ad_student.npz', 'E1 trò', clf_e1),
                        ('/kaggle/working/ad_student_motion.npz', 'E1+m trò + chuyển động', clf_e1m)):
    np.savez(path, coef=clf.coef_[0], intercept=clf.intercept_, backbone=STUDENT,
             with_motion=name != 'E1 trò', threshold=results[name]['threshold'])
json.dump({'results': results, 'speed': speed, 'best_student': best, 'teacher': TEACHER, 'student': STUDENT,
           'prompts': {'lesson': LESSON_PROMPTS, 'ad': AD_PROMPTS},
           'postprocess': dict(min_ad_sec=MIN_AD_SEC, max_gap_sec=MAX_GAP_SEC, smooth_sec=SMOOTH_SEC, tol_sec=TOL_SEC)},
          open('/kaggle/working/ad_filter_results.json', 'w'), ensure_ascii=False, indent=1)
np.save('/kaggle/working/teacher_score.npy', teacher_score)
print('Model trò dùng trong pipeline:', best)
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
out = HERE / 'notebooks' / 'ad_filter_experiments.ipynb'
nbf.write(nb, out)
print('wrote', out)
