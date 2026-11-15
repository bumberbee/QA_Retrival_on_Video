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
                skip_nonref=True, records=None, content_options=None):
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
            if content_options is not None:
                small_image = frame.to_image(width=small_width, height=small_height)
                sample["content_view"] = reveal_view(
                    small_image,
                    content_options.get("roi", roi), ignored_regions)
                if content_options.get("mode") == "board":
                    sample["presenter_mask"] = board_presenter_mask(
                        small_image,
                        content_options.get("roi", roi))
            if records is not None:
                record = {key: sample[key] for key in ("time_sec", "sharpness", "image_path")}
                # uint8 keeps a 30-minute video's signatures near 120 MB.
                record["signature"] = (signature * 255).round().astype("uint8")
                for key in ("content_view", "presenter_mask"):
                    if key in sample:
                        record[key] = sample[key]
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


def board_presenter_mask(image, roi=(0, 0, 1, 1)):
    """CPU mask for *green chalkboards*, not a general person segmentation model.

    Large non-green components cover skin/clothes; small chalk strokes survive.
    Use a content ROI excluding the frame/banner. The mask is deliberately local
    to each sample: a video-wide busy-pixel mask would also remove new writing.
    """
    import cv2
    import numpy as np

    pixels = np.asarray(image.convert("RGB"))
    h, w = pixels.shape[:2]
    l, t, r, b = roi
    pixels = pixels[int(t*h):int(b*h), int(l*w):int(r*w)]
    pixels = cv2.resize(pixels, (384, 216), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
    green = ((hsv[..., 0] >= 32) & (hsv[..., 0] <= 100)
             & (hsv[..., 1] >= 35))
    # Fail open on another board colour; the caller can supply ignored regions.
    if green.mean() < 0.35:
        return np.zeros(green.shape, dtype=bool)
    blobs = cv2.morphologyEx((~green).astype("uint8"), cv2.MORPH_OPEN,
                             np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blobs)
    mask = np.zeros(green.shape, dtype="uint8")
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= green.size * 0.008:
            mask[labels == index] = 1
    # A head/hand is much smaller than the shirt and can be disconnected from
    # it. White chalk has low saturation; skin on these green-board videos has
    # a warm hue. Include it before dilation rather than lowering the blob
    # cutoff enough to mask chalk characters.
    skin = ((hsv[..., 0] <= 25) & (hsv[..., 1] >= 35) & (hsv[..., 2] >= 60))
    mask[skin] = 1
    return cv2.dilate(mask, np.ones((7, 7), np.uint8)).astype(bool)


def local_content_evidence(previous, current, valid=None, grid=(6, 4),
                           minimum_pixels=18, deleted_fraction=0.20, pixel_slack=1):
    """Directional stroke overlap, including local losses hidden by global counts.

    Both stroke polarity and position must survive (one pixel compression slack).
    Returned regions are normalized boxes in the content ROI, not OCR lines.
    """
    import cv2
    import numpy as np

    valid = np.ones(previous.shape, dtype=bool) if valid is None else valid
    old_residual = previous.astype("int16") - cv2.medianBlur(previous, 41)
    new_residual = current.astype("int16") - cv2.medianBlur(current, 41)
    old = (np.abs(old_residual) > 28) & valid
    new = (np.abs(new_residual) > 28) & valid
    kernel = np.ones((2*pixel_slack+1, 2*pixel_slack+1), np.uint8)
    kept = np.zeros(old.shape, dtype=bool)
    covered = np.zeros(old.shape, dtype=bool)
    for sign in (-1, 1):
        a = old & (old_residual * sign > 0)
        b = new & (new_residual * sign > 0)
        kept |= a & cv2.dilate(b.astype("uint8"), kernel).astype(bool)
        covered |= b & cv2.dilate(a.astype("uint8"), kernel).astype(bool)
    lost, added = old & ~kept, new & ~covered
    columns, rows = grid
    h, w = old.shape
    deleted, changed = [], []
    for row in range(rows):
        for column in range(columns):
            y0, y1 = row*h//rows, (row+1)*h//rows
            x0, x1 = column*w//columns, (column+1)*w//columns
            box = (slice(y0, y1), slice(x0, x1))
            if valid[box].mean() < 0.65:
                continue
            n_old, n_lost, n_added = (int(x[box].sum()) for x in (old, lost, added))
            region = {"box": [x0/w, y0/h, x1/w, y1/h],
                      "old_pixels": n_old, "deleted_pixels": n_lost,
                      "added_pixels": n_added}
            if n_lost + n_added >= minimum_pixels:
                changed.append(region)
            if n_lost >= minimum_pixels and n_lost / max(1, n_old) >= deleted_fraction:
                deleted.append(region)
    return {"directional_retained": float(kept.sum()/max(1, old.sum())),
            "deleted_pixels": int(lost.sum()), "added_pixels": int(added.sum()),
            "deleted_regions": deleted, "changed_regions": changed}


def board_content_splits(records, boundaries, *, window_samples=5, check_every_sec=5.0,
                         min_state_sec=15.0, snapshot_sec=120.0,
                         minimum_pixels=24, grid=(6, 4)):
    """Discover local writing/erasure states without requiring a still presenter.

    A masked temporal median rejects short motion. Two checks must show changed
    content; periodic snapshots are explicit fallback evidence, not slide changes.
    Primary boundaries are preserved. Manual boundaries never call this function.
    """
    import numpy as np

    if not records:
        return boundaries, []
    extra, audit = [], []
    for start, end in zip(boundaries, boundaries[1:]):
        samples = [r for r in records if start <= r["time_sec"] < end
                   and "content_view" in r]
        reference = reference_valid = None
        pending = None
        last_boundary, last_check = start, -float("inf")
        for index in range(window_samples-1, len(samples)):
            timestamp = samples[index]["time_sec"]
            if timestamp-last_check < check_every_sec:
                continue
            last_check = timestamp
            window = samples[index-window_samples+1:index+1]
            views = np.stack([r["content_view"] for r in window]).astype("float32")
            masks = np.stack([r.get("presenter_mask", np.zeros(views.shape[1:], bool))
                              for r in window])
            valid = (~masks).sum(axis=0) >= max(2, window_samples//2+1)
            # Avoid all-NaN warnings; only 'valid' pixels participate downstream.
            background = float(np.median(views[~masks])) if (~masks).any() else 0.0
            views[masks] = np.nan
            # Zero-filled person holes create artificial high-contrast strokes
            # in the median foreground filter just outside the validity mask.
            views[:, ~valid] = background
            current = np.nanmedian(views, axis=0).astype("uint8")
            if reference is None:
                reference, reference_valid = current, valid
                continue
            shared = valid & reference_valid
            evidence = local_content_evidence(reference, current, shared, grid,
                                               minimum_pixels)
            changed = bool(evidence["changed_regions"])
            eligible = timestamp-last_boundary >= min_state_sec and end-timestamp >= min_state_sec
            periodic = snapshot_sec > 0 and timestamp-last_boundary >= snapshot_sec
            if eligible and (periodic or (changed and pending is not None)):
                onset = (last_boundary + snapshot_sec if periodic
                         else max(pending, last_boundary + min_state_sec))
                if onset-last_boundary >= min_state_sec:
                    extra.append(onset)
                    audit.append({"time_sec": round(onset, 6),
                                  "reason": "periodic_content_snapshot" if periodic else "local_board_change",
                                  "onset_uncertainty_sec": check_every_sec + window_samples,
                                  **evidence})
                    last_boundary = onset
                    # Preserve the last visible board behind the presenter.
                    # Dropping that history at each snapshot would silently
                    # accept an erasure performed while the region is hidden.
                    reference[valid] = current[valid]
                    reference_valid |= valid
                    pending = None
                    continue
            pending = timestamp if changed and pending is None else pending if changed else None
            # Learn content first exposed after an occlusion, without treating
            # the lecturer stepping aside as writing. Never erase hidden history.
            exposed = valid & ~reference_valid
            reference[exposed] = current[exposed]
            reference_valid |= valid
    kept = union_boundaries(boundaries, extra, min_state_sec)
    # Balance a final >snapshot tail when a regular snapshot would leave a
    # tiny state at the next primary cut. These are fallback observations.
    if snapshot_sec > 0:
        additions = []
        for start, end in zip(kept, kept[1:]):
            if end-start > snapshot_sec:
                count = math.ceil((end-start)/snapshot_sec)
                for number in range(1, count):
                    onset = start+(end-start)*number/count
                    if min(onset-start, end-onset) >= min_state_sec:
                        additions.append(onset)
                        audit.append({"time_sec": round(onset, 6),
                                      "reason": "periodic_content_snapshot",
                                      "onset_uncertainty_sec": check_every_sec+window_samples,
                                      "changed_regions": []})
        kept = sorted(kept+additions)
    return kept, sorted(audit, key=lambda item: item["time_sec"])


def normalized_display_view(image):
    """Conservative bright rectangular screen ROI, normalized across camera views.

    No candidate means no identity merge. Cropping is for identity only; saved
    keyframes retain the original frame and the ROI is exported for inspection.
    """
    import cv2
    import numpy as np

    pixels = np.asarray(image.convert("RGB"))
    h, w = pixels.shape[:2]
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
    bright = ((hsv[..., 2] > 155) & (hsv[..., 1] < 95)).astype("uint8")
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8),
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area, ratio = bw*bh/(w*h), bw/max(1, bh)
        fill = float(bright[y:y+bh, x:x+bw].mean())
        if area >= 0.10 and 1.20 <= ratio <= 2.05 and fill >= 0.80:
            candidates.append((area, x, y, bw, bh))
    if not candidates:
        return None
    _, x, y, bw, bh = max(candidates)
    # Remove bezel / compression at screen edges.
    inset = max(1, round(min(bw, bh)*0.012))
    box = [(x+inset)/w, (y+inset)/h, (x+bw-inset)/w, (y+bh-inset)/h]
    return {"view": reveal_view(image, box), "roi": box}


def same_display_evidence(previous, current, retained_threshold=0.96,
                          minimum_pixels=300, change_limit=0.08):
    """Bidirectional content retention rejects a new page with the same template."""
    import numpy as np

    if previous is None or current is None:
        return False, {}
    a, b = previous["view"], current["view"]
    forward = local_content_evidence(a, b, grid=(8, 6), minimum_pixels=12, pixel_slack=2)
    reverse = local_content_evidence(b, a, grid=(8, 6), minimum_pixels=12, pixel_slack=2)
    change = float(np.mean(np.abs(a.astype("int16")-b.astype("int16")) > 32))
    enough = min(content_mask(a).sum(), content_mask(b).sum()) >= minimum_pixels
    same = (enough and min(forward["directional_retained"], reverse["directional_retained"])
            >= retained_threshold and not forward["deleted_regions"]
            and not reverse["deleted_regions"] and change <= change_limit)
    return bool(same), {"identity_retained_forward": forward["directional_retained"],
                        "identity_retained_reverse": reverse["directional_retained"],
                        "identity_image_change": change}


def content_keyframes(records, start, end, *, mode="generic", maximum_candidates=64,
                      occlusion_weight=8.0, transient_weight=6.0, complementary=False):
    """ROI sharpness with sample-local occlusion and temporal-overlay penalties.

    Complementary frames are observations, never stitched images. All candidates
    and the temporal median are confined to this state's half-open interval.
    """
    import cv2
    import numpy as np

    inside = [r for r in records if start <= r["time_sec"] < end and "content_view" in r]
    margin = min(0.75, (end-start)/4)
    choices = [r for r in inside if start+margin <= r["time_sec"] < end-margin] or inside
    if not choices:
        return []
    indices = np.unique(np.linspace(0, len(choices)-1, min(maximum_candidates, len(choices))).astype(int))
    choices = [choices[i] for i in indices]
    views = np.stack([r["content_view"] for r in choices])
    masks = np.stack([r.get("presenter_mask", np.zeros(views.shape[1:], bool)) for r in choices])
    visible = (~masks).any(axis=0)
    values = views.astype("float32")
    background = float(np.median(values[~masks])) if (~masks).any() else 0.0
    values[masks] = np.nan
    values[:, ~visible] = background
    median = np.nanmedian(values, axis=0).astype("uint8")
    support = content_mask(median) & visible
    scored = []
    for item, view, mask in zip(choices, views, masks):
        valid = ~mask
        lap = cv2.Laplacian(view, cv2.CV_64F)
        sharpness = float(lap[valid].var()) if valid.any() else 0.0
        occluded = float(mask.mean())
        difference = cv2.absdiff(view, median) > 32
        # Short-lived large UI rectangles have a strong median residual. A
        # genuinely persistent tooltip remains an unresolved limitation.
        transient = float((difference & valid).sum()/max(1, valid.sum()))
        coverage = float((support & valid).sum()/max(1, support.sum()))
        score = sharpness*(0.25+0.75*coverage)/(1+occlusion_weight*occluded+transient_weight*transient)
        if mode == "board":
            # On boards the most exposed writing is more valuable than a sharp shirt.
            score = coverage/(1+occlusion_weight*occluded+transient_weight*transient) + min(sharpness, 2000)/200000
        scored.append({**item, "content_score": score,
                       "content_quality": {"roi_sharpness": round(sharpness, 3),
                                           "occlusion_fraction": round(occluded, 4),
                                           "transient_fraction": round(transient, 4),
                                           "visible_content_fraction": round(coverage, 4)}})
    best = max(scored, key=lambda r: r["content_score"])
    result = [best]
    if complementary and len(scored) > 1 and support.any():
        blocked = best.get("presenter_mask", np.zeros(support.shape, bool)) & support
        candidates = [r for r in scored if r["time_sec"] != best["time_sec"]]
        other = max(candidates, key=lambda r: int((blocked & ~r.get("presenter_mask", np.zeros(support.shape, bool))).sum()))
        gain = int((blocked & ~other.get("presenter_mask", np.zeros(support.shape, bool))).sum())/int(support.sum())
        if gain >= 0.02:
            result.append({**other, "complementary_gain": round(gain, 4)})
    return result


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
                    anchor_threshold=0.6, non_slide_threshold=0.2,
                    replacement_safe=False, directional_threshold=0.94,
                    deletion_minimum_pixels=12, deletion_fraction=0.20):
    """Merge only when the new state keeps the old content and adds to it.

    previous/current/anchor index the per-video state arrays in ``state``: views,
    the persistent-content mask, and optionally OCR texts and slide probabilities.
    """
    views, mask = state["views"], state["mask"]
    texts, p_slides, common = state.get("texts"), state.get("p_slides"), state.get("common", ())
    evidence = reveal_evidence(views[previous], views[current], mask)
    if replacement_safe:
        # Do not subtract the persistent template from the deletion veto: a
        # repeated editor/sidebar must not hide a changed code line.
        local = local_content_evidence(views[previous], views[current], grid=(16, 9),
                                       minimum_pixels=deletion_minimum_pixels,
                                       deleted_fraction=deletion_fraction)
        evidence.update(local)
        if (local["directional_retained"] < directional_threshold
                or local["deleted_regions"]):
            return "split_deleted_or_replaced_content", evidence
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
        if replacement_safe:
            old_words = ocr_words(texts[previous]) - set(common)
            new_words = ocr_words(texts[current]) - set(common)
            evidence["text_directional_retained"] = len(old_words & new_words)/max(1, len(old_words))
            if len(old_words) >= 4 and evidence["text_directional_retained"] < directional_threshold:
                return "split_text_deleted", evidence
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
        if replacement_safe:
            core = local_content_evidence(views[anchor], views[current], grid=(16, 9),
                                          minimum_pixels=deletion_minimum_pixels,
                                          deleted_fraction=deletion_fraction)
            if core["directional_retained"] < directional_threshold or core["deleted_regions"]:
                return "split_anchor_content_deleted", evidence
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


def group_reveal_states(views, texts=None, p_slides=None, display_views=None, **thresholds):
    """Return [[state indices]] where each group is one slide revealed over time."""
    if not views:
        return [], []
    state = {"views": views, "mask": persistent_template(views),
             "texts": texts, "p_slides": p_slides,
             "common": common_ocr_words(texts) if texts is not None else set()}
    groups, anchors, audit = [[0]], [0], []
    for index in range(1, len(views)):
        group = groups[-1]
        reset_anchor = False
        decision, evidence = reveal_decision(group[-1], index, anchors[-1], state, **thresholds)
        if display_views is not None:
            same, identity = same_display_evidence(display_views[group[-1]], display_views[index])
            if same and anchors[-1] != group[-1]:
                same, _ = same_display_evidence(display_views[anchors[-1]], display_views[index])
            if same:
                decision = "merge_same_display"
                evidence.update(identity)
                # The equivalent content in the new camera view becomes the
                # spatial anchor. Otherwise the next ordinary reveal would
                # fail retention against the old PiP layout and create an FP.
                reset_anchor = evidence["image_change"] > 0.25
                if reset_anchor:
                    evidence["identity_anchor_state"] = index
        audit.append({"previous_state": group[-1], "current_state": index,
                      "decision": decision,
                      **{key: round(value, 4) if isinstance(value, (int, float)) else value
                         for key, value in evidence.items()}})
        if decision.startswith("merge_"):
            group.append(index)
            if reset_anchor:
                anchors[-1] = index
        else:
            groups.append([index])
            anchors.append(index)
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
                        ad_scorer=None, ad_options=None, content_scorer=None,
                        board_options=None, keyframe_options=None,
                        merge_same_content=False, export_state_onsets=False):
    """slide_classifier: None, or (p_slide function, checkpoint sha256) from
    load_slide_classifier(). merge_ocr: compare Tesseract words between states.
    ad_scorer: None, or load_ad_scorer() of adfilter.py: ad segments are cut out of
    the slide intervals as their own segments (content_kind "ad"; single_pass only).
    ad_options: extra apply_ad_filter() arguments, e.g. {"mode": "flag"} to only flag them.
    content_scorer: None, or (check(images) -> P(lecture content), info), e.g. load_ocr_gate():
    every 1 Hz sample is scored and saved as content_scores.json for the slide+speech filter
    of stage 04 (alignment.slide_speech_flags).
    scan: "single_pass" decodes once with no seeks; "seek" is the original method.
    Opt-in stage-01 improvements (defaults preserve legacy artifacts):
      board_options: local green-board comparison/snapshot arguments plus roi.
      keyframe_options: content_keyframes arguments plus roi; mode='board' can
        retain complementary observations. Requires single_pass.
      merge_same_content: compare normalized bright screen regions.
      export_state_onsets: add state evidence windows/deltas for downstream OCR.
    Reveal replacement safety is reveal_thresholds={'replacement_safe': True}."""
    from PIL import Image

    if sample_hz <= 0 or stable_samples < 2 or min_interval_sec <= 0:
        raise ValueError("sample_hz/min_interval_sec phải > 0; stable_samples >= 2.")
    option_rois = [options["roi"] for options in (board_options, keyframe_options)
                   if options is not None and "roi" in options]
    for box in (roi, *ignored_regions, *option_rois):
        if not (len(box) == 4 and 0 <= box[0] < box[2] <= 1
                and 0 <= box[1] < box[3] <= 1):
            raise ValueError(f"Vùng chuẩn hóa [left, top, right, bottom] sai: {box}")
    if scan not in {"single_pass", "seek"}:
        raise ValueError("SCAN_MODE phải là 'single_pass' hoặc 'seek'.")
    if (board_options is not None or keyframe_options is not None) and (
            scan != "single_pass" or manual_boundaries is not None):
        raise ValueError("Content scan options require single_pass and automatic boundaries.")
    if board_options is not None and keyframe_options is not None and (
            board_options.get("roi", roi) != keyframe_options.get("roi", roi)):
        raise ValueError("Board detection and keyframe scoring must use the same content ROI.")
    for options in (board_options, keyframe_options):
        if options is None:
            continue
        for key in ("window_samples", "check_every_sec", "min_state_sec", "minimum_pixels",
                    "maximum_candidates"):
            if key in options and options[key] <= 0:
                raise ValueError(f"{key} must be positive")
    if board_options is not None and (board_options.get("window_samples", 5) < 3
                                     or board_options.get("snapshot_sec", 120) < 0):
        raise ValueError("window_samples >= 3; snapshot_sec >= 0")
    content_options = None
    if board_options is not None or keyframe_options is not None:
        content_options = {"roi": (board_options if board_options is not None else keyframe_options).get("roi", roi),
                           "mode": "board" if board_options is not None else keyframe_options.get("mode", "generic")}
    source = probe_video(video_path)
    directory = begin_stage(output_root, 1)
    records = []
    hidden_split_count = masked_split_count = 0
    board_audit = []
    motion_mask_fraction = None
    if manual_boundaries is None and scan == "single_pass":
        def run_scan(skip):
            records.clear()
            return detect_stable_boundaries(
                scan_frames(video_path, source, directory, sample_hz, fine_hz,
                            roi, ignored_regions, skip_nonref=skip, records=records,
                            **({"content_options": content_options} if content_options is not None else {})),
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

    if board_options is not None:
        boundaries, board_audit = board_content_splits(
            records, boundaries, **{k: v for k, v in board_options.items() if k != "roi"})
        write_json(directory / "board_audit.json", board_audit)

    # Each visual state gets its own sharp keyframe before states are grouped.
    states, views, display_views = [], [], []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        selected = (content_keyframes(records, start, end,
                    **{k: v for k, v in keyframe_options.items() if k != "roi"})
                    if keyframe_options is not None else [])
        scanned = selected[0] if selected else pick_scanned_keyframe(records, start, end)
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
        if selected:
            states[-1]["content_quality"] = selected[0]["content_quality"]
            states[-1]["complementary_keyframes"] = []
            for number, candidate in enumerate(selected[1:], 1):
                extra_name = f"keyframes/states/state_{index:05d}_complement_{number}.jpg"
                shutil.copyfile(directory / candidate["image_path"], directory / extra_name)
                states[-1]["complementary_keyframes"].append({
                    "image_path": extra_name, "keyframe_time_sec": round(candidate["time_sec"], 6),
                    "content_quality": candidate["content_quality"],
                    "visible_content_gain": candidate["complementary_gain"]})
        views.append(reveal_view(image, roi, ignored_regions))
        if merge_same_content:
            display_views.append(normalized_display_view(image))
            if display_views[-1] is not None:
                states[-1]["display_roi"] = display_views[-1]["roi"]
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
            views, texts, p_slides,
            **({"display_views": display_views} if merge_same_content else {}),
            **(reveal_thresholds or {}))
    else:
        groups, reveal_audit = [[index] for index in range(len(states))], []
    decisions = {item["current_state"]: item["decision"] for item in reveal_audit}
    if export_state_onsets:
        for index, item in enumerate(states):
            item["content_onset_sec"] = item["start_sec"]
            item["content_observed_sec"] = item["keyframe_time_sec"]
            item["content_onset_bounds_sec"] = [item["start_sec"], item["keyframe_time_sec"]]
            item["onset_method"] = "detected_visual_state_start_not_per_line_ocr"
            if index:
                item["content_delta"] = local_content_evidence(views[index-1], views[index])
            else:
                item["content_delta"] = {"added_pixels": int(content_mask(views[index]).sum()),
                                         "deleted_pixels": 0, "changed_regions": []}

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
        if export_state_onsets:
            intervals[-1].update({
                "representative_content_onset_sec": final["start_sec"],
                "long_dynamic_review": final["end_sec"]-members[0]["start_sec"] > 120,
                "visual_evidence_windows": [
                    {"source_state": position, "start_sec": states[position]["start_sec"],
                     "end_sec": states[position]["end_sec"], "image_path": states[position]["image_path"],
                     "observed_sec": states[position]["keyframe_time_sec"],
                     "onset_sec": states[position]["start_sec"]} for position in group]})
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
    for key, value in (("board_options", board_options), ("keyframe_options", keyframe_options),
                       ("merge_same_content", merge_same_content), ("export_state_onsets", export_state_onsets)):
        if value is not None and value is not False:
            parameters[key] = value
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
