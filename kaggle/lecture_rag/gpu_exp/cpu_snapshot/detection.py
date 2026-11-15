"""PTS-based visual intervals, stable-change detection, and representative frames."""


def frame_signature(image, roi=(0, 0, 1, 1), ignored_regions=()):
    import numpy as np

    return frame_signature_array(np.asarray(image.convert("RGB")), roi, ignored_regions)


def frame_signature_array(pixels, roi=(0, 0, 1, 1), ignored_regions=()):
    import cv2
    import numpy as np

    pixels = pixels.copy()
    height, width = pixels.shape[:2]
    for left, top, right, bottom in ignored_regions:
        pixels[int(top * height):int(bottom * height),
               int(left * width):int(right * width)] = 127
    left, top, right, bottom = roi
    pixels = pixels[int(top * height):max(int(bottom * height), 1),
                    int(left * width):max(int(right * width), 1)]
    if pixels.size == 0:
        raise ValueError("SLIDE_ROI không hợp lệ.")
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (240, 135), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.GaussianBlur(cv2.Canny(gray, 50, 150), (3, 3), 0)
    # Text/layout edges catch two white slides whose average colors barely differ.
    return np.stack([blurred, edges]).astype("float32") / 255.0


def signature_distance(left, right):
    import numpy as np
    # Ignore the most changed 3% of pixels to reduce isolated cursor/overlay noise.
    differences = np.abs(left[0] - right[0]).reshape(-1)
    cutoff = max(1, int(len(differences) * 0.97))
    color_change = float(np.partition(differences, cutoff - 1)[:cutoff].mean())
    edge_change = float(np.abs(left[1] - right[1]).mean())
    edge_density = max(0.08, float(left[1].mean() + right[1].mean()))
    # Either a palette change or a text/layout change is sufficient. Averaging
    # them dilutes text-only changes on otherwise identical white backgrounds.
    return min(1.0, max(color_change, edge_change / edge_density))


def sampled_frames(video_path, source, sample_hz=1.0, roi=(0, 0, 1, 1),
                   ignored_regions=()):
    import av

    next_time = 0.0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base) - source["timeline_origin_sec"]
            if timestamp < 0 or timestamp + 1e-6 < next_time:
                continue
            if timestamp >= source["duration_sec"]:
                break
            yield {
                "time_sec": timestamp,
                "signature": frame_signature(frame.to_image(), roi, ignored_regions),
            }
            next_time = timestamp + 1.0 / sample_hz


def scan_frames(video_path, source, directory, sample_hz=1.0, fine_hz=8.0,
                roi=(0, 0, 1, 1), ignored_regions=(), keep_width=1280,
                skip_nonref=True, records=None):
    """Decode once. Yields 1/sample_hz samples; each carries the fine-rate
    signatures since the previous sample (for boundary refinement without
    seeking) and a saved frame for keyframe selection."""
    import av
    import cv2

    width, height = source["width"], source["height"]
    small_width = min(480, width)
    small_height = max(2, round(height * small_width / width / 2) * 2)
    keep_width = min(keep_width, width)
    keep_height = max(2, round(height * keep_width / width / 2) * 2)
    step, fine_step = 1.0 / sample_hz, 1.0 / max(fine_hz, sample_hz)
    (Path(directory) / "_scan").mkdir(parents=True, exist_ok=True)
    next_sample = next_fine = 0.0
    fine, index = [], 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if skip_nonref:
            # Non-reference frames are never needed at 1-4 Hz; skipping them
            # roughly halves H.264 decoding with B-frames. PTS stay exact.
            stream.codec_context.skip_frame = "NONREF"
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base) - source["timeline_origin_sec"]
            if timestamp < 0 or timestamp + 1e-6 < next_fine:
                continue
            if timestamp >= source["duration_sec"]:
                break
            pixels = frame.to_ndarray(format="rgb24", width=small_width, height=small_height)
            signature = frame_signature_array(pixels, roi, ignored_regions)
            next_fine = timestamp + fine_step
            if timestamp + 1e-6 < next_sample:
                fine.append((timestamp, signature))
                continue
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            name = f"_scan/sample_{index:06d}.jpg"
            frame.to_image(width=keep_width, height=keep_height).save(
                Path(directory) / name, quality=90)
            sample = {"time_sec": timestamp, "signature": signature, "fine": fine,
                      "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                      "image_path": name}
            if records is not None:
                record = {key: sample[key] for key in ("time_sec", "sharpness", "image_path")}
                # uint8 keeps a 30-minute video's signatures near 120 MB.
                record["signature"] = (signature * 255).round().astype("uint8")
                records.append(record)
            yield sample
            fine, index = [], index + 1
            next_sample = timestamp + step


def motion_mask(records, pixel_change=0.10, busy_fraction=0.30):
    """Pixels that change in most seconds of the video: an animated background,
    the lecturer's own picture-in-picture. Slides change far less often."""
    import numpy as np

    signatures = [item["signature"] for item in records if "signature" in item]
    if len(signatures) < 10:
        return None
    changed = np.zeros(signatures[0].shape[1:], dtype="float32")
    previous = signatures[0].astype("float32") / 255.0
    for signature in signatures[1:]:
        current = signature.astype("float32") / 255.0
        # Grey level or edges: flickering particles mostly show up as edges.
        changed += ((np.abs(current[0] - previous[0]) > pixel_change)
                    | (np.abs(current[1] - previous[1]) > 2 * pixel_change))
        previous = current
    busy = changed / (len(signatures) - 1) > busy_fraction
    return busy if 0 < busy.mean() < 0.85 else None


def masked_boundaries(records, duration, mask, change_threshold=0.055,
                      stability_threshold=0.035, stable_samples=2, min_interval_sec=2.0):
    """Stable-change detection on the 1 Hz samples with the busy pixels ignored."""
    def samples():
        for item in records:
            signature = item["signature"].astype("float32") / 255.0
            signature[:, mask] = 0.0
            yield {"time_sec": item["time_sec"], "signature": signature}

    boundaries, _, _ = detect_stable_boundaries(
        samples(), duration, change_threshold, stability_threshold,
        stable_samples, min_interval_sec,
    )
    return boundaries


def union_boundaries(primary, extra, min_interval_sec=2.0):
    """Keep every primary boundary (it has sub-second refinement) and add the
    extra ones that are not within min_interval_sec of any kept boundary."""
    kept = sorted(primary)
    for time in sorted(extra):
        if all(abs(time - other) >= min_interval_sec for other in kept):
            kept.append(time)
            kept.sort()
    return kept


def hidden_change_splits(records, start, end, change_threshold=0.055,
                         stability_threshold=0.035, min_interval_sec=2.0,
                         static_tolerance=0.10, motion_threshold=0.15,
                         settled_threshold=0.25):
    """Split points the stable-change detector cannot confirm because one side
    of the change keeps moving (an advert clip, a camera shot of the lecturer):

    1. still -> moving: a still slide followed by footage that never holds still;
    2. moving -> settled: footage followed by a shot that only moves slightly.

    Gradual changes of a still slide (writing, reveals) are not split here."""
    import statistics

    samples = [item for item in records
               if start <= item["time_sec"] < end and "signature" in item]
    if len(samples) < 4:
        return []
    signatures = [item["signature"].astype("float32") / 255.0 for item in samples]
    times = [item["time_sec"] for item in samples]
    steps = [0.0] + [signature_distance(a, b) for a, b in zip(signatures, signatures[1:])]

    def long_enough(index, first, last):
        return (times[index] - times[first] >= min_interval_sec
                and last - times[index] >= min_interval_sec)

    def still_to_moving(first, stop, last):
        reference = signatures[first]
        to_reference = [signature_distance(reference, signatures[index])
                        for index in range(first, stop)]
        for split in range(first + 1, stop - 1):
            before = to_reference[:split - first]
            after = to_reference[split - first:]
            if (max(before) <= static_tolerance and min(after) >= change_threshold
                    and statistics.median(steps[split + 1:stop]) >= motion_threshold
                    and long_enough(split, first, last)):
                return split
        return None

    def moving_to_settled(first, stop, last):
        # The piece's last sample may already show the next state (boundary lag).
        for split in range(first + 2, stop - 2):
            if (statistics.median(steps[first + 1:split + 1]) >= motion_threshold
                    and max(steps[split + 1:stop - 1]) <= settled_threshold
                    and long_enough(split, first, last)):
                return split
        return None

    splits = []
    pieces = [(0, len(samples), end)]
    split = still_to_moving(0, len(samples), end)
    if split is not None:
        splits.append(times[split])
        pieces = [(0, split, times[split]), (split, len(samples), end)]
    for first, stop, last in pieces:
        settled = moving_to_settled(first, stop, last)
        if settled is not None:
            splits.append(times[settled])
    return sorted(splits)


def refine_from_buffer(candidate, change_threshold, stability_threshold):
    """Earliest fine-rate frame that already shows the new, stable state."""
    for timestamp, signature in candidate.get("fine", ()):
        if (signature_distance(candidate["old_signature"], signature) >= change_threshold
                and signature_distance(candidate["signature"], signature)
                <= stability_threshold):
            return timestamp
    return candidate["time_sec"]


def pick_scanned_keyframe(records, start, end):
    """Sharpest scanned sample inside [start, end), away from the edges if possible."""
    inside = [item for item in records if start <= item["time_sec"] < end]
    margin = min(0.75, (end - start) / 4)
    central = [item for item in inside
               if start + margin <= item["time_sec"] < end - margin]
    choices = central or inside
    return max(choices, key=lambda item: item["sharpness"]) if choices else None


def detect_stable_boundaries(samples, duration, change_threshold=0.055,
                             stability_threshold=0.035, stable_samples=2,
                             min_interval_sec=2.0):
    boundaries = [0.0]
    candidates = []
    reference = None
    candidate = None
    previous_time = 0.0
    sample_count = 0
    for sample in samples:
        sample_count += 1
        signature, timestamp = sample["signature"], sample["time_sec"]
        if reference is None:
            reference = signature
        changed = signature_distance(reference, signature) >= change_threshold
        eligible = timestamp - boundaries[-1] >= min_interval_sec
        if changed and eligible:
            if (candidate is None or
                    signature_distance(candidate["signature"], signature)
                    > stability_threshold):
                candidate = {
                    "time_sec": timestamp, "left_sec": previous_time,
                    "signature": signature, "old_signature": reference, "count": 1,
                    "fine": sample.get("fine", ()),
                }
            else:
                candidate["count"] += 1
            if (candidate["count"] >= stable_samples
                    and duration - candidate["time_sec"] >= min_interval_sec):
                boundaries.append(candidate["time_sec"])
                candidates.append(candidate)
                reference = candidate["signature"]
                candidate = None
        else:
            candidate = None
        previous_time = timestamp
    if sample_count == 0:
        raise ValueError("Không giải mã được frame có timestamp.")
    return boundaries + [duration], candidates, sample_count


def frame_at_time(video_path, source, time_sec):
    import av

    target = max(0.0, min(float(time_sec), source["duration_sec"] - 0.001))
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        origin = source["timeline_origin_sec"]
        seek_pts = int((target + origin) / float(stream.time_base))
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        last = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base) - origin
            if timestamp < 0:
                continue
            last = (frame.to_image(), timestamp)
            if timestamp + 1e-6 >= target:
                return last
        if last is not None:
            return last
    raise ValueError(f"Không lấy được frame tại {time_sec:.3f}s")


def refine_boundary(video_path, source, candidate, roi, ignored_regions,
                    change_threshold, stability_threshold, samples=7):
    import numpy as np

    for target in np.linspace(candidate["left_sec"], candidate["time_sec"], samples):
        image, actual_time = frame_at_time(video_path, source, target)
        signature = frame_signature(image, roi, ignored_regions)
        if (signature_distance(candidate["old_signature"], signature) >= change_threshold
                and signature_distance(candidate["signature"], signature)
                <= stability_threshold):
            return actual_time
    return candidate["time_sec"]


def select_keyframe(video_path, source, start, end, maximum_width=1600):
    import cv2
    import numpy as np

    margin = min(0.75, (end - start) / 4)
    choices = []
    for target in (start + margin, (start + end) / 2, end - margin):
        image, actual = frame_at_time(video_path, source, target)
        if not start <= actual < end:
            continue
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        choices.append((sharpness, image, actual))
    if not choices:
        raise ValueError(f"Không có keyframe nằm trong [{start}, {end}).")
    sharpness, image, actual = max(choices, key=lambda item: item[0])
    image.thumbnail((maximum_width, maximum_width))
    return image, actual, sharpness


def create_contact_sheet(directory, intervals, limit=24):
    from PIL import Image, ImageDraw

    selected = intervals[:limit]
    columns, cell_width, cell_height = 4, 320, 210
    sheet = Image.new("RGB", (columns * cell_width,
                             max(1, math.ceil(len(selected) / columns)) * cell_height),
                      "#f5f1e9")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(selected):
        with Image.open(relative_asset(directory, item["image_path"])) as image:
            image = image.convert("RGB")
            image.thumbnail((cell_width - 12, 174))
            x, y = (index % columns) * cell_width + 6, (index // columns) * cell_height
            sheet.paste(image, (x, y + 6))
            label = (f"{item['sequence']:03d} | {timestamp_label(item['start_sec'])}"
                     f" - {timestamp_label(item['end_sec'])}")
            draw.text((x, y + 185), label, fill="#241f18")
    sheet.save(Path(directory) / "contact_sheet.jpg", quality=90)


def reveal_view(image, roi=(0, 0, 1, 1), ignored_regions=()):
    """Grayscale ROI at a fixed size, for comparing content between visual states."""
    import cv2
    import numpy as np

    pixels = np.asarray(image.convert("RGB")).copy()
    height, width = pixels.shape[:2]
    for left, top, right, bottom in ignored_regions:
        pixels[int(top * height):int(bottom * height),
               int(left * width):int(right * width)] = 127
    left, top, right, bottom = roi
    pixels = pixels[int(top * height):max(int(bottom * height), 1),
                    int(left * width):max(int(right * width), 1)]
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (384, 216), interpolation=cv2.INTER_AREA)


def content_mask(view):
    import cv2

    # Text and drawings differ from their local background; a wide median keeps
    # the slide background and removes thin strokes.
    return cv2.absdiff(view, cv2.medianBlur(view, 41)) > 28


def persistent_template(views, fraction=0.55):
    """Content present in most states (logo, title bar, frame) is not evidence."""
    import numpy as np

    if len(views) < 6:
        return np.zeros(views[0].shape, dtype=bool) if views else None
    indices = np.unique(np.linspace(0, len(views) - 1, min(24, len(views))).astype(int))
    return np.mean([content_mask(views[index]) for index in indices], axis=0) >= fraction


def reveal_evidence(previous, current, template):
    """How much old content survives at the same place, and how much was added."""
    import cv2
    import numpy as np

    old = content_mask(previous) & ~template
    new = content_mask(current) & ~template
    kernel = np.ones((3, 3), np.uint8)
    # Allow one-pixel shifts and compression noise around each old stroke.
    low = cv2.erode(current, kernel).astype(np.int16) - 22
    high = cv2.dilate(current, kernel).astype(np.int16) + 22
    values = previous.astype(np.int16)
    preserved = (values >= low) & (values <= high)
    old_pixels = int(np.count_nonzero(old))
    return {
        "old_content_pixels": old_pixels,
        "content_retained": float(np.count_nonzero(old & preserved) / max(1, old_pixels)),
        "content_growth": float(np.count_nonzero(new) / max(1, old_pixels)),
        "image_change": float(np.mean(cv2.absdiff(previous, current) > 32)),
    }


def ocr_words(text):
    return set(re.findall(r"[^\W_]{2,}", str(text).casefold(), flags=re.UNICODE))


def common_ocr_words(texts):
    """Words on most states of a video (channel name, author, footer) are not content."""
    from collections import Counter

    counts = Counter(word for text in texts for word in ocr_words(text))
    return {word for word, count in counts.items() if count >= max(12, 0.55 * len(texts))}


def text_evidence(previous_text, current_text, common):
    old, new = ocr_words(previous_text) - common, ocr_words(current_text) - common
    shared = len(old & new)
    return {"old_words": len(old), "new_words": len(new), "shared_words": shared,
            "text_containment": shared / max(1, min(len(old), len(new)))}


def reveal_decision(previous, current, anchor, state, *, minimum_pixels=250,
                    retained_threshold=0.85, change_limit=0.25, growth_threshold=0.9,
                    anchor_threshold=0.6, non_slide_threshold=0.2):
    """Merge only when the new state keeps the old content and adds to it.

    previous/current/anchor index the per-video state arrays in ``state``: views,
    the persistent-content mask, and optionally OCR texts and slide probabilities.
    """
    views, mask = state["views"], state["mask"]
    texts, p_slides, common = state.get("texts"), state.get("p_slides"), state.get("common", ())
    evidence = reveal_evidence(views[previous], views[current], mask)
    if p_slides is not None:
        evidence["p_slide_previous"] = p_slides[previous]
        evidence["p_slide_current"] = p_slides[current]
        # A camera shot of the lecturer is never part of a slide build.
        if min(p_slides[previous], p_slides[current]) < non_slide_threshold:
            return "split_non_slide", evidence
    text = None
    if texts is not None:
        text = text_evidence(texts[previous], texts[current], common)
        evidence.update(text)
        visual_build = (evidence["content_retained"] >= 0.88
                        and evidence["image_change"] <= change_limit)
        # Same layout, different words (e.g. remember -> regret): a new slide.
        if (text["old_words"] >= 4 and text["new_words"] >= 4
                and text["text_containment"] < 0.55 and not visual_build):
            return "split_text_replaced", evidence
    if (text is not None and text["shared_words"] >= 4 and text["text_containment"] >= 0.78
            and evidence["image_change"] <= 0.30):
        # OCR keeps the old words even when a moving presenter lowers pixel retention.
        decision = "merge_text_retained"
    elif evidence["old_content_pixels"] < minimum_pixels:
        return "split_little_content", evidence
    elif (evidence["content_retained"] >= retained_threshold
            and evidence["image_change"] <= change_limit
            and evidence["content_growth"] >= growth_threshold):
        decision = "merge_content_kept"
    else:
        return "split_content_changed", evidence
    if anchor != previous:
        # Prevent chained merges: A~B and B~C while C no longer shows A's core.
        first = reveal_evidence(views[anchor], views[current], mask)
        evidence["anchor_retained"] = first["content_retained"]
        text_lost = False
        if texts is not None:
            core = text_evidence(texts[anchor], texts[current], common)
            text_lost = (core["old_words"] >= 4 and core["new_words"] >= 4
                         and core["shared_words"] / core["old_words"] < 0.40)
        if first["content_retained"] < anchor_threshold or (
                text_lost and first["content_retained"] < 0.75):
            return "split_anchor_lost", evidence
    return decision, evidence


def group_reveal_states(views, texts=None, p_slides=None, **thresholds):
    """Return [[state indices]] where each group is one slide revealed over time."""
    if not views:
        return [], []
    state = {"views": views, "mask": persistent_template(views),
             "texts": texts, "p_slides": p_slides,
             "common": common_ocr_words(texts) if texts is not None else set()}
    groups, audit = [[0]], []
    for index in range(1, len(views)):
        group = groups[-1]
        decision, evidence = reveal_decision(group[-1], index, group[0], state, **thresholds)
        audit.append({"previous_state": group[-1], "current_state": index,
                      "decision": decision,
                      **{key: round(value, 4) for key, value in evidence.items()}})
        if decision.startswith("merge_"):
            group.append(index)
        else:
            groups.append([index])
    return groups, audit


SLIDE_CLASSIFIER_URL = (
    "https://huggingface.co/HHousen/lecture2notes/resolve/main/"
    "Slide%20Classifier%20Median%20Weights/three-category/epoch%3D8.ckpt"
)


def load_slide_classifier(checkpoint=None, cache_dir="/kaggle/working/_models"):
    """Lecture2Notes ResNet-34 (other / presenter_slide / slide); returns p_slide(images)."""
    import urllib.request
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms

    if checkpoint is None:
        found = sorted(Path("/kaggle/input").rglob("epoch=8.ckpt"))
        checkpoint = found[0] if found else Path(cache_dir) / "lecture2notes_epoch8.ckpt"
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SLIDE_CLASSIFIER_URL, checkpoint)
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = data.get("hyper_parameters", data.get("hparams", {}))
    hp = hp if isinstance(hp, dict) else vars(hp)
    classes = list(hp.get("classes", []))
    if set(classes) != {"other", "presenter_slide", "slide"} or hp.get("arch") != "resnet34":
        raise RuntimeError(f"Checkpoint không phải ResNet-34 ba lớp: {classes}, {hp.get('arch')}")

    class ConcatPool(nn.Module):
        def forward(self, x):
            return torch.cat([nn.functional.adaptive_max_pool2d(x, 1),
                              nn.functional.adaptive_avg_pool2d(x, 1)], dim=1)

    base = models.resnet34(weights=None)
    if hp.get("feature_extract") == "advanced":
        model = nn.Sequential(
            nn.Sequential(*list(base.children())[:-2]), ConcatPool(), nn.Flatten(),
            nn.BatchNorm1d(1024), nn.Dropout(0.25), nn.Linear(1024, 512),
            nn.ReLU(inplace=True), nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear(512, len(classes)),
        )
    else:
        base.fc = nn.Linear(base.fc.in_features, len(classes))
        model = base
    if hp.get("relu_to_mish"):
        def replace(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, nn.Mish())
                else:
                    replace(child)
        replace(model)
    prefix = "classification_model."
    model.load_state_dict({key[len(prefix):]: value for key, value in data["state_dict"].items()
                           if key.startswith(prefix)}, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    size = int(hp.get("input_size", 224))
    # The authors' inference transform: resize the short side, then center crop.
    preprocess = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    positive = [classes.index("slide"), classes.index("presenter_slide")]

    def p_slide(images, batch_size=32):
        values = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([preprocess(image.convert("RGB"))
                                 for image in images[start:start + batch_size]]).to(device)
            with torch.inference_mode():
                probabilities = torch.softmax(model(batch), dim=1)[:, positive].sum(dim=1)
            values += [float(value) for value in probabilities.cpu()]
        return values

    return p_slide, file_sha256(checkpoint)


def tesseract_languages():
    if not shutil.which("tesseract"):
        raise RuntimeError("Cần tesseract (apt-get install tesseract-ocr tesseract-ocr-vie).")
    listed = run_command(["tesseract", "--list-langs"]).lower().split()
    return "+".join(language for language in ("vie", "eng") if language in listed) or "eng"


def tesseract_text(image, languages, roi=(0, 0, 1, 1)):
    """Tesseract sparse-text OCR; keeps words with confidence >= 35."""
    import csv
    import io
    import cv2
    import numpy as np

    pixels = np.asarray(image.convert("L"))
    height, width = pixels.shape
    left, top, right, bottom = roi
    pixels = pixels[int(top * height):int(bottom * height), int(left * width):int(right * width)]
    if pixels.shape[1] < 720:
        pixels = cv2.resize(pixels, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    result = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", languages, "--psm", "11", "tsv"],
        input=cv2.imencode(".png", pixels)[1].tobytes(), capture_output=True, timeout=60,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-500:])
    rows = csv.DictReader(io.StringIO(result.stdout.decode("utf-8", "replace")),
                          delimiter="\t", quoting=csv.QUOTE_NONE)
    words = []
    for row in rows:
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            confidence = -1
        if confidence >= 35 and len(text) >= 2:
            words.append(text)
    return " ".join(words)


def create_reveal_sheet(directory, intervals, limit=12):
    """One row per merged slide: every reveal state from first to final."""
    from PIL import Image, ImageDraw

    merged = [item for item in intervals if len(item["reveal_states"]) > 1][:limit]
    if not merged:
        return None
    columns = min(6, max(len(item["reveal_states"]) for item in merged))
    cell_width, cell_height = 260, 175
    sheet = Image.new("RGB", (columns * cell_width, len(merged) * cell_height), "#f5f1e9")
    draw = ImageDraw.Draw(sheet)
    for row, item in enumerate(merged):
        states = item["reveal_states"]
        # Keep the first and the final states when a group has too many to show.
        shown = states if len(states) <= columns else states[:columns - 1] + states[-1:]
        for column, state in enumerate(shown):
            with Image.open(relative_asset(directory, state["image_path"])) as image:
                image = image.convert("RGB")
                image.thumbnail((cell_width - 10, 140))
                x, y = column * cell_width + 5, row * cell_height
                sheet.paste(image, (x, y + 5))
            draw.text((x, y + 150), f"{item['sequence']:03d}.{state['state']} "
                      f"{timestamp_label(state['start_sec'])}", fill="#241f18")
    sheet.save(Path(directory) / "reveal_sheet.jpg", quality=90)
    return "reveal_sheet.jpg"


def score_samples(records, directory, scorer, batch_size=64):
    """{"times", "scores", info}: P(lecture content) of every 1 Hz sample of the scan."""
    from PIL import Image

    check, info = scorer
    times, scores = [item["time_sec"] for item in records], []
    for start in range(0, len(records), batch_size):
        images = []
        for item in records[start:start + batch_size]:
            with Image.open(Path(directory) / item["image_path"]) as opened:
                images.append(opened.convert("RGB"))
        scores += [float(value) for value in check(images)]
    return {**{k: v for k, v in info.items() if k != "threshold"},
            "times": [round(t, 3) for t in times], "scores": [round(v, 4) for v in scores]}


def run_slide_detection(video_path, output_root, video_id="L25_V001", *,
                        sample_hz=1.0, roi=(0, 0, 1, 1), ignored_regions=(),
                        change_threshold=0.055, stability_threshold=0.035,
                        stable_samples=2, min_interval_sec=2.0,
                        manual_boundaries=None, merge_reveals=True,
                        reveal_thresholds=None, slide_classifier=None, merge_ocr=False,
                        scan="single_pass", fine_hz=8.0, skip_nonref=True,
                        split_hidden_changes=True, mask_motion=True,
                        ad_scorer=None, ad_options=None, content_scorer=None):
    """slide_classifier: None, or (p_slide function, checkpoint sha256) from
    load_slide_classifier(). merge_ocr: compare Tesseract words between states.
    ad_scorer: None, or load_ad_scorer() of adfilter.py: ad segments are cut out of
    the slide intervals as their own segments (content_kind "ad"; single_pass only).
    ad_options: extra apply_ad_filter() arguments, e.g. {"mode": "flag"} to only flag them.
    content_scorer: None, or (check(images) -> P(lecture content), info), e.g. load_ocr_gate():
    every 1 Hz sample is scored and saved as content_scores.json for the slide+speech filter
    of stage 04 (alignment.slide_speech_flags).
    scan: "single_pass" decodes once with no seeks; "seek" is the original method."""
    from PIL import Image

    if sample_hz <= 0 or stable_samples < 2 or min_interval_sec <= 0:
        raise ValueError("sample_hz/min_interval_sec phải > 0; stable_samples >= 2.")
    for box in (roi, *ignored_regions):
        if not (len(box) == 4 and 0 <= box[0] < box[2] <= 1
                and 0 <= box[1] < box[3] <= 1):
            raise ValueError(f"Vùng chuẩn hóa [left, top, right, bottom] sai: {box}")
    if scan not in {"single_pass", "seek"}:
        raise ValueError("SCAN_MODE phải là 'single_pass' hoặc 'seek'.")
    source = probe_video(video_path)
    directory = begin_stage(output_root, 1)
    records = []
    hidden_split_count = masked_split_count = 0
    motion_mask_fraction = None
    if manual_boundaries is None and scan == "single_pass":
        def run_scan(skip):
            records.clear()
            return detect_stable_boundaries(
                scan_frames(video_path, source, directory, sample_hz, fine_hz,
                            roi, ignored_regions, skip_nonref=skip, records=records),
                source["duration_sec"], change_threshold, stability_threshold,
                stable_samples, min_interval_sec,
            )
        try:
            boundaries, candidates, sample_count = run_scan(skip_nonref)
        except ValueError:
            if not skip_nonref:
                raise
            # Some streams have no reference frames left after skipping.
            boundaries, candidates, sample_count = run_scan(False)
        refined = [0.0]
        for candidate in candidates:
            boundary = refine_from_buffer(candidate, change_threshold, stability_threshold)
            refined.append(max(refined[-1] + min_interval_sec,
                               min(boundary, candidate["time_sec"])))
        boundaries = refined + [source["duration_sec"]]
        if mask_motion:
            mask = motion_mask(records)
            if mask is not None:
                motion_mask_fraction = round(float(mask.mean()), 4)
                before = len(boundaries)
                boundaries = union_boundaries(boundaries, masked_boundaries(
                    records, source["duration_sec"], mask, change_threshold,
                    stability_threshold, stable_samples, min_interval_sec,
                ), min_interval_sec)
                masked_split_count = len(boundaries) - before
        if split_hidden_changes:
            checked = [boundaries[0]]
            for start, end in zip(boundaries, boundaries[1:]):
                checked += hidden_change_splits(
                    records, start, end, change_threshold, stability_threshold,
                    min_interval_sec,
                ) + [end]
            hidden_split_count = len(checked) - len(boundaries)
            boundaries = checked
    elif manual_boundaries is None:
        boundaries, candidates, sample_count = detect_stable_boundaries(
            sampled_frames(video_path, source, sample_hz, roi, ignored_regions),
            source["duration_sec"], change_threshold, stability_threshold,
            stable_samples, min_interval_sec,
        )
        refined = [0.0]
        for candidate in candidates:
            boundary = refine_boundary(
                video_path, source, candidate, roi, ignored_regions,
                change_threshold, stability_threshold,
            )
            # Do not create tiny intervals while refining a transition.
            refined.append(max(refined[-1] + min_interval_sec,
                               min(boundary, candidate["time_sec"])))
        boundaries = refined + [source["duration_sec"]]
    else:
        boundaries = [float(time) for time in manual_boundaries]
        if boundaries != sorted(set(boundaries)):
            raise ValueError("MANUAL_BOUNDARIES_SEC phải tăng dần, không trùng.")
        if (not boundaries or abs(boundaries[0]) > 1e-6
                or abs(boundaries[-1] - source["duration_sec"]) > 0.03):
            raise ValueError("Ranh giới thủ công phải bắt đầu 0 và kết thúc ở duration.")
        sample_count = 0

    # Each visual state gets its own sharp keyframe before states are grouped.
    states, views = [], []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        scanned = pick_scanned_keyframe(records, start, end)
        if scanned is not None:
            with Image.open(directory / scanned["image_path"]) as opened:
                image = opened.convert("RGB")
            keyframe_time, sharpness = scanned["time_sec"], scanned["sharpness"]
        else:
            image, keyframe_time, sharpness = select_keyframe(video_path, source, start, end)
        name = f"keyframes/states/state_{index:05d}.jpg"
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, quality=95)
        states.append({"start_sec": round(start, 6), "end_sec": round(end, 6),
                       "keyframe_time_sec": round(keyframe_time, 6),
                       "image_path": name, "sharpness_score": round(sharpness, 3)})
        views.append(reveal_view(image, roi, ignored_regions))
        if index % 20 == 0:
            print(f"Keyframe {index + 1}/{len(boundaries) - 1}: {start:.2f}s")
    p_slides = texts = None
    if slide_classifier is not None or merge_ocr:
        languages = tesseract_languages() if merge_ocr else None
        if merge_ocr:
            texts = []
        images = []
        for index, item in enumerate(states):
            with Image.open(directory / item["image_path"]) as opened:
                image = opened.convert("RGB")
            if merge_ocr:
                texts.append(tesseract_text(image, languages, roi))
                item["ocr_text"] = texts[-1]
            images.append(image)
            if len(images) == 64 or index == len(states) - 1:
                if slide_classifier is not None:
                    p_slides = (p_slides or []) + slide_classifier[0](images)
                images = []
        if p_slides is not None:
            for item, value in zip(states, p_slides):
                item["p_slide"] = round(value, 4)
    if merge_reveals and manual_boundaries is None:
        groups, reveal_audit = group_reveal_states(
            views, texts, p_slides, **(reveal_thresholds or {}))
    else:
        groups, reveal_audit = [[index] for index in range(len(states))], []
    decisions = {item["current_state"]: item["decision"] for item in reveal_audit}

    method = ("manual" if manual_boundaries is not None
              else "single_pass_stable_visual_change" if scan == "single_pass"
              else "stable_visual_change_with_pts_refinement")
    intervals = []
    for index, group in enumerate(groups):
        members = [states[position] for position in group]
        # The final state of a progressive reveal shows the complete slide.
        final = members[-1]
        name = f"keyframes/seg_{index:05d}.jpg"
        shutil.copyfile(directory / final["image_path"], directory / name)
        intervals.append({
            "segment_id": f"{video_id}:seg_{index:05d}", "sequence": index,
            "start_sec": members[0]["start_sec"], "end_sec": final["end_sec"],
            "keyframe_time_sec": final["keyframe_time_sec"],
            "image_path": name, "image_sha256": file_sha256(directory / name),
            "content_kind": "unclassified",
            "sharpness_score": final["sharpness_score"],
            **({"p_slide": final["p_slide"]} if "p_slide" in final else {}),
            "boundary_method": method + ("+progressive_reveal_merge" if len(group) > 1 else ""),
            "reveal_states": [
                {"state": number, "source_state": position, **states[position],
                 "decision": decisions.get(position, "first_state")}
                for number, position in enumerate(group)
            ],
        })
    ad_report = None
    if ad_scorer is not None and records:
        intervals, ad_report = apply_ad_filter(
            intervals, records, directory, video_id, source["duration_sec"], boundaries,
            ad_scorer, min_interval_sec=min_interval_sec, **(ad_options or {}))
        write_json(directory / "ad_filter.json", ad_report)
    if content_scorer is not None and records:
        write_json(directory / "content_scores.json", score_samples(records, directory, content_scorer))
    shutil.rmtree(directory / "_scan", ignore_errors=True)
    verify_intervals(intervals, source["duration_sec"])
    write_json(directory / "intervals.json", intervals)
    write_json(directory / "reveal_audit.json", reveal_audit)
    create_contact_sheet(directory, intervals)
    reveal_sheet = create_reveal_sheet(directory, intervals)
    parameters = {
        "sample_hz": sample_hz, "roi": roi, "ignored_regions": ignored_regions,
        "change_threshold": change_threshold, "stability_threshold": stability_threshold,
        "stable_samples": stable_samples, "min_interval_sec": min_interval_sec,
        "manual_boundaries_sec": manual_boundaries,
        "merge_reveals": bool(merge_reveals and manual_boundaries is None),
        "reveal_thresholds": reveal_thresholds or {},
        "slide_classifier_sha256": slide_classifier[1] if slide_classifier else None,
        "merge_ocr": bool(merge_ocr),
        "scan": scan, "fine_hz": fine_hz, "skip_nonref": skip_nonref,
        "split_hidden_changes": bool(split_hidden_changes and scan == "single_pass"),
        "mask_motion": bool(mask_motion and scan == "single_pass"),
        "packages": package_versions(["av", "numpy", "opencv-python-headless", "Pillow"]),
        "ad_filter": {key: value for key, value in ad_report.items()
                      if key not in {"sample_scores", "sample_times"}}
                     if ad_report else None,
    }
    manifest = finish_stage(directory, 1, video_id, source, parameters, extra={
        "sample_count": sample_count, "interval_count": len(intervals),
        "visual_state_count": len(states), "hidden_split_count": hidden_split_count,
        "masked_split_count": masked_split_count, "motion_mask_fraction": motion_mask_fraction,
        "ad_segment_count": sum(item["content_kind"] == "ad" for item in intervals),
        "ad_flagged_count": sum(bool(item.get("ad_flagged")) for item in intervals),
        "merged_reveal_count":
            sum(len(group) > 1 for group in groups),
        "reveal_sheet": reveal_sheet,
        "semantic_classification": "pending_stage_02",
    })
    return directory, manifest, intervals
