"""Ad filter: score each 1 Hz sample "lesson or ad", turn the scores into ad intervals and
cut them out of the slide intervals as their own segments (content_kind "ad").

Stage 01 calls apply_ad_filter() with the 1 Hz samples it already decoded. Later stages
skip "ad" segments: no OCR (02), their words stay with the ad (04, midpoint rule as for
slides), nothing is indexed (05).
"""

AD_LESSON_PROMPTS = [
    "a slide of a lecture presentation with text",
    "a lecture slide with a diagram, chart or formula",
    "a teacher presenting a lesson next to a slide",
    "the title slide of a lesson",
    "a page of a document or an exam with questions",
]
AD_PROMPTS = [
    "a video advertisement",
    "a promotional video of a school or university",
    "a logo animation",
    "students walking on a school campus",
    "people in a commercial",
    "a building exterior",
]


def ad_runs(mask):
    """[start, end) index runs of a 0/1 sequence where it is 1."""
    runs, index = [], 0
    while index < len(mask):
        if mask[index]:
            end = index
            while end < len(mask) and mask[end]:
                end += 1
            runs.append([index, end])
            index = end
        else:
            index += 1
    return runs


def ad_mask(scores, threshold=0.5, smooth=5, min_ad=5, max_gap=2):
    """Per-sample ad mask: moving average, threshold, fill short gaps between two ads,
    drop ads shorter than min_ad samples except a short closing logo at the very end."""
    import numpy as np

    scores = np.asarray(scores, dtype="float64")
    if smooth > 1 and len(scores):
        # edge padding: a short closing logo is not diluted by zeros past the end of the video
        padded = np.pad(scores, (smooth // 2, smooth - 1 - smooth // 2), mode="edge")
        scores = np.convolve(padded, np.ones(smooth) / smooth, mode="valid")
    mask = (scores >= threshold).astype(int)
    for start, end in ad_runs(1 - mask):
        if 0 < start and end < len(mask) and end - start <= max_gap:
            mask[start:end] = 1
    for start, end in ad_runs(mask):
        if end - start < min_ad and not (end == len(mask) and end - start >= 2):
            mask[start:end] = 0
    return mask


def model_features(output):
    """Embedding tensor from get_image_features/get_text_features (transformers 4 or 5)."""
    import torch

    return output if torch.is_tensor(output) else output.pooler_output


def load_siglip(model_id, device="cuda"):
    """SigLIP loaded once per session (shared by the ad filter and the OCR gate).
    Returns (embed_images(images), embed_texts(texts), logit_scale, logit_bias)."""
    def load():
        import torch
        from transformers import AutoModel, AutoProcessor

        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_id)

        def embed_images(images):
            with torch.no_grad():
                pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device, dtype)
                out = model_features(model.get_image_features(pixel_values=pixels)).float()
            return torch.nn.functional.normalize(out, dim=-1).cpu().numpy()

        def embed_texts(texts):
            with torch.no_grad():
                tokens = processor(text=texts, padding="max_length", return_tensors="pt").to(device)
                out = model_features(model.get_text_features(**tokens)).float()
            return torch.nn.functional.normalize(out, dim=-1).cpu().numpy()

        return embed_images, embed_texts, model.logit_scale.exp().item(), model.logit_bias.item()

    return shared_model(("siglip", model_id, str(device)), load)


def zero_shot(embed_images, embed_texts, scale, bias, positive, negative):
    """P(positive prompts) for each image: softmax over all prompts, summed over the positive ones."""
    import numpy as np

    text = embed_texts(list(positive) + list(negative))

    def score(images):
        logits = embed_images(images) @ text.T * scale + bias
        logits = logits - logits.max(1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        return probs[:, :len(positive)].sum(1).tolist()

    return score


def load_ad_scorer(mode="student", device="cuda", student_path=None,
                   teacher_model="google/siglip-so400m-patch14-384"):
    """Returns (score(images) -> list of P(ad), description). mode "teacher": SigLIP
    zero-shot with the prompts above; "student": logistic regression (student_path, an
    .npz from the ad filter experiments) on the image embeddings of a small SigLIP."""
    import numpy as np

    if mode not in {"student", "teacher"}:
        raise ValueError("AD_FILTER phải là 'student', 'teacher' hoặc None.")
    student = dict(np.load(student_path, allow_pickle=False)) if mode == "student" else None
    if student and bool(student.get("with_motion", False)):
        raise ValueError("Pipeline dùng model trò chỉ có ảnh (ad_student.npz), không phải bản có chuyển động.")
    model_id = str(student["backbone"]) if student else teacher_model
    embed_images, embed_texts, scale, bias = load_siglip(model_id, device)
    if mode == "teacher":
        score = zero_shot(embed_images, embed_texts, scale, bias, AD_PROMPTS, AD_LESSON_PROMPTS)
    else:
        def score(images):
            logits = embed_images(images) @ student["coef"] + float(student["intercept"][0])
            return (1 / (1 + np.exp(-logits))).tolist()
    threshold = float(student["threshold"]) if student else 0.5
    return score, {"mode": mode, "model": model_id, "threshold": threshold,
                   "student_sha256": file_sha256(student_path) if student else None}


# OCR gate: a keyframe goes to OCR only if it shows readable lesson content. A lecturer
# talking to the camera, an intro logo or a blank screen is kept as a segment (its speech
# stays in the lesson) but is not read by the slow OCR models.
CONTENT_PROMPTS = [
    "a slide of a lecture presentation with text",
    "a lecture slide with a diagram, chart, table or formula",
    "a page of a document or an exam with questions",
    "a whiteboard or blackboard with handwriting",
]
NO_CONTENT_PROMPTS = [
    "a person talking to the camera",
    "a teacher speaking in a studio",
    "a logo or a title animation",
    "a blank screen",
]


def load_ocr_gate(device="cuda", model_id="google/siglip-base-patch16-224", threshold=0.10):
    """Returns (check(images) -> list of P(readable content), description)."""
    embed_images, embed_texts, scale, bias = load_siglip(model_id, device)
    check = zero_shot(embed_images, embed_texts, scale, bias, CONTENT_PROMPTS, NO_CONTENT_PROMPTS)
    return check, {"model": model_id, "threshold": threshold}


def snap(time_sec, boundaries, tolerance=1.5):
    """Move an ad edge (from 1 Hz samples) to the nearest exact state boundary of stage 01."""
    nearest = min(boundaries, key=lambda item: abs(item - time_sec))
    return nearest if abs(nearest - time_sec) <= tolerance else time_sec


def ad_time_intervals(sample_times, mask, duration, boundaries):
    intervals = []
    for start, end in ad_runs(mask):
        a = snap(sample_times[start], boundaries)
        b = duration if end >= len(sample_times) else snap(sample_times[end], boundaries)
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], b)
        elif b > a:
            intervals.append([a, b])
    return intervals


def split_ad_intervals(intervals, ads, min_interval_sec=2.0):
    """Cut ad time ranges out of slide intervals. Returns [start, end, source] pieces in time
    order; source is the index of the slide interval, or None for an ad. A lesson remnant
    shorter than min_interval_sec next to an ad is absorbed by that ad."""
    cuts = sorted({item["start_sec"] for item in intervals} | {intervals[-1]["end_sec"]}
                  | {t for a, b in ads for t in (a, b) if intervals[0]["start_sec"] < t < intervals[-1]["end_sec"]})
    pieces = []
    for start, end in zip(cuts, cuts[1:]):
        middle = (start + end) / 2
        if any(a <= middle < b for a, b in ads):
            source = None
        else:
            source = next(i for i, item in enumerate(intervals) if item["start_sec"] <= middle < item["end_sec"])
        if pieces and pieces[-1][2] == source:
            pieces[-1][1] = end
        else:
            pieces.append([start, end, source])
    changed = True
    while changed:
        changed = False
        for i, (start, end, source) in enumerate(pieces):
            next_to_ad = (i > 0 and pieces[i - 1][2] is None) or (i + 1 < len(pieces) and pieces[i + 1][2] is None)
            if source is not None and end - start < min_interval_sec and next_to_ad:
                pieces[i][2] = None
                changed = True
        merged = []
        for piece in pieces:
            if merged and merged[-1][2] is None and piece[2] is None:
                merged[-1][1] = piece[1]
            else:
                merged.append(piece)
        pieces = merged
    return pieces


def apply_ad_filter(intervals, records, directory, video_id, duration, state_boundaries,
                    scorer, *, smooth=5, min_ad=5, max_gap=2, min_interval_sec=2.0,
                    batch_size=64):
    """intervals: stage-01 intervals (before writing). records: the 1 Hz samples of the scan
    (time_sec, image_path in directory). Returns (new intervals, ad report)."""
    from PIL import Image

    score, info = scorer
    times, scores = [item["time_sec"] for item in records], []
    for start in range(0, len(records), batch_size):
        images = []
        for item in records[start:start + batch_size]:
            with Image.open(Path(directory) / item["image_path"]) as opened:
                images.append(opened.convert("RGB"))
        scores += score(images)
    mask = ad_mask(scores, info["threshold"], smooth, min_ad, max_gap)
    ads = ad_time_intervals(times, mask, duration, sorted(set(state_boundaries)))
    pieces = split_ad_intervals(intervals, ads, min_interval_sec)

    result = []
    for start, end, source in pieces:
        number = len(result)
        if source is None:
            # representative frame: the 1 Hz sample nearest the middle of the ad
            middle = (start + end) / 2
            sample = min(records, key=lambda item: abs(item["time_sec"] - middle))
            name = f"keyframes/ad_{number:05d}.jpg"
            shutil.copyfile(Path(directory) / sample["image_path"], Path(directory) / name)
            result.append({
                "start_sec": round(start, 6), "end_sec": round(end, 6),
                "keyframe_time_sec": round(sample["time_sec"], 6), "image_path": name,
                "content_kind": "ad", "sharpness_score": round(sample["sharpness"], 3),
                "boundary_method": "ad_filter", "reveal_states": [],
            })
            continue
        item = dict(intervals[source])
        item["start_sec"], item["end_sec"] = round(start, 6), round(end, 6)
        if not start <= item["keyframe_time_sec"] < end:
            # the slide's own keyframe lies in the ad: use its latest state outside the ad
            inside = [state for state in item["reveal_states"]
                      if start <= state["keyframe_time_sec"] < end]
            if inside:
                state = inside[-1]
                item["keyframe_time_sec"], item["image_path"] = state["keyframe_time_sec"], state["image_path"]
            else:
                sample = min((r for r in records if start <= r["time_sec"] < end),
                             key=lambda r: -r["sharpness"], default=None)
                if sample is not None:
                    name = f"keyframes/trim_{number:05d}.jpg"
                    shutil.copyfile(Path(directory) / sample["image_path"], Path(directory) / name)
                    item["keyframe_time_sec"], item["image_path"] = sample["time_sec"], name
            item["reveal_states"] = [state for state in item["reveal_states"]
                                     if state["end_sec"] > start and state["start_sec"] < end]
        if (start, end) != (intervals[source]["start_sec"], intervals[source]["end_sec"]):
            item["boundary_method"] = item["boundary_method"] + "+ad_trim"
        result.append(item)
    for number, item in enumerate(result):
        item["segment_id"], item["sequence"] = f"{video_id}:seg_{number:05d}", number
        item["image_sha256"] = file_sha256(Path(directory) / item["image_path"])
    report = {**info, "smooth": smooth, "min_ad_sec": min_ad, "max_gap_sec": max_gap,
              "ad_intervals": [[round(a, 3), round(b, 3)] for a, b in ads],
              "ad_seconds": round(sum(b - a for a, b in ads), 1),
              "sample_times": [round(value, 3) for value in times],
              "sample_scores": [round(value, 4) for value in scores]}
    return result, report
