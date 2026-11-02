"""Stage-04 text policies, preserving raw OCR/ASR and word timing for review.

Conservative OCR keeps repeated lesson lines and literal layout, dropping only explicit
boilerplate. Failure guards exclude decoder loops/placeholders from evidence. Reviewed
ASR aliases require local printed support. Legacy frequency cleaning and unrestricted
correction remain available only for reproducible ablations.
"""

UNREADABLE = re.compile(r"\[(?:unreadable|non-text)\]", re.I)
# Unlimited-OCR box markup that leaks into the text, e.g. a cut-off "<|det|>header" at the end
LEAKED_TAG = re.compile(r"<\|det\|>\s*[A-Za-z_]*\s*(?:\[[\d,\s]*\]?)?|<\|/?\w{1,20}\|>")
NOISE_REGION_WORDS = ("giáo viên", "giảng viên", "người", "logo", "trang trí", "teacher", "presenter",
                      "decorative", "khuôn mặt", "chân dung")
# Onsets Whisper confuses in Vietnamese (s/x, ch/tr); a final c/t or n/ng may differ in the same
# syllable. Tone-only differences are never corrected: slide OCR drops tones more often than ASR.
ONSETS = [("tr", "ch"), ("s", "x")]
FINALS = [("ng", "n"), ("c", "t")]
OCR_FIXES = {"ǎ": "ă", "Ǎ": "Ă"}   # Unlimited-OCR writes a caron for the breve


def strip_tones(text):
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def line_key(line):
    """Comparable form of an OCR line: no accents or punctuation (digits kept: "Câu 1" != "Câu 2")."""
    return re.sub(r"[^a-z0-9]+", "", strip_tones(line))


def repeated_lines(texts, min_share=0.5, min_slides=3):
    """Line keys that appear on at least min_share of the slides (and >= min_slides)."""
    counts, slides = {}, 0
    for text in texts:
        keys = {line_key(line) for line in (text or "").splitlines() if line_key(line)}
        if not keys:
            continue
        slides += 1
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
    need = max(min_slides, math.ceil(min_share * slides))
    return {key for key, count in counts.items() if count >= need}


# Unlimited-OCR sometimes loops: L25_V005 seg 9 (lecturer + a small inset slide) came back as one
# formula line 269 times. Runs of LOOP_RUN or more are kept once; two identical answer options stay.
LOOP_RUN = 3
LOOP_PHRASE = re.compile(r"(\S.{5,}?)(?:\s*\1){%d,}" % (LOOP_RUN - 1))


def collapse_loops(lines, min_run=LOOP_RUN):
    """Keep one line of each run of >= min_run consecutive lines with the same line_key."""
    kept, index = [], 0
    while index < len(lines):
        key, end = line_key(lines[index]), index + 1
        while key and end < len(lines) and line_key(lines[end]) == key:
            end += 1
        kept.extend(lines[index:end] if end - index < min_run else lines[index:index + 1])
        index = end
    return kept


def collapse_phrase_loop(line):
    """A phrase with letters repeated back to back >= LOOP_RUN times in one line is kept once
    ("0 0 0 0" in a table row has no letters and stays)."""
    return LOOP_PHRASE.sub(lambda match: match.group(1) if re.search(r"[^\W\d_]", match.group(1))
                           else match.group(0), line)


def clean_ocr_text(text, repeated=frozenset(), conservative=False, canonical_subscripts=False):
    if conservative:
        kept = []
        for line in (text or "").splitlines():
            line = UNREADABLE.sub("", LEAKED_TAG.sub("", line)).rstrip()
            if not line.strip() or boilerplate_line(line):
                continue
            for bad, good in OCR_FIXES.items():
                line = line.replace(bad, good)
            if canonical_subscripts and not re.search(r"['\"]", line) and not re.match(
                    r"^\s*(?:\w+\s*=|(?:print|input)\s*\(|for\b|def\b|if\b|elif\b|else\b|"
                    r"while\b|return\b|import\b|from\b|#|```)", line):
                # Chemical subscripts and mathematical indices have an equivalent plain
                # digit spelling. Preserve superscript powers and all code whitespace.
                line = line.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
            kept.append(line)
        return "\n".join(kept)
    kept = []
    for line in collapse_loops((text or "").splitlines()):
        line = collapse_phrase_loop(UNREADABLE.sub("", LEAKED_TAG.sub("", line))).strip()
        key = line_key(line)
        if not line or key in repeated:
            continue
        if len(line) <= 3 and not re.search(r"[A-Za-zÀ-ỹ]{2}", line):   # page numbers, bullets
            continue
        for bad, good in OCR_FIXES.items():
            line = line.replace(bad, good)
        kept.append(line)
    return "\n".join(kept)


def code_or_formula(line):
    """Protect literal syntax, including incomplete code and short symbolic lines."""
    return bool(re.search(r"[=<>+*/\\$∑Σ]|^\s*(?:print\s*\(|for\b|def\b|if\b|elif\b|else\s*:|"
                          r"while\b|return\b|import\b|from\b|#|```)", line))


def boilerplate_line(line):
    """Explicit attribution/footer rules; repetition alone is never evidence of noise.
    Bare numbers are ambiguous without trustworthy positions and therefore stay."""
    if code_or_formula(line):
        return False
    text = strip_tones(line).strip()
    return bool(re.fullmatch(r"(?:trang|page)\s+\d+(?:\s*(?:/|of)\s*\d+)?[.]?", text)
                or re.match(r"^(?:giao vien|giang vien|lecturer|copyright|©|hotline)\s*[:：]", text)
                or re.fullmatch(r"(?:logo\s+)?(?:truong (?:dai hoc quoc te sai gon|quoc te a chau)|"
                                r"saigon international university|the asian school|siu|siu[.]edu[.]vn|"
                                r"(?:bao )?thanh[- ]nien)", text))


def qwen_ocr_text(visual):
    """Recover line layout from the exported answer, without modifying stage 02."""
    raw = visual.get("raw_ocr_response")
    blocks = visual.get("ocr_blocks") or []
    if isinstance(raw, str) and raw and blocks and all(b.get("source") == "qwen3-vl" for b in blocks):
        return "\n".join(line.rstrip() for line in raw.splitlines()
                         if line.strip() and not re.match(r"^\s*\[(?:Hình|Figure)\]", line, re.I))
    return visual.get("ocr_text", "")


def heading_title(text):
    """Prefer a visible heading in the first block; body sentences and editor UI stay untitled."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for index, line in enumerate(lines[:6]):
        if code_or_formula(line) or boilerplate_line(line):
            continue
        value = re.sub(r"^#{1,6}\s+", "", line)
        words = re.findall(r"[^\W\d_]+", value)
        if not words or len("".join(words)) < 2 or len(value) > 110 or len(words) > 16:
            continue
        if strip_tones(value).lower() in {"clear update", "run", "time", "next tip", "true", "false"}:
            continue
        letters = "".join(words)
        if len(letters) >= 3 and letters.isupper():
            return value
        if line.startswith("#") or re.match(r"^(?:Câu|Bài|Chương|Chapter|Example|Định lý|Ví dụ)\b", value):
            if len(words) <= 10 and not re.search(r"[.;!?]", value):
                return value
        if index == 0 and len(words) <= 8 and not re.search(r"[,:;.!?()|]", value):
            if sum(w[0].isupper() for w in words) >= max(1, math.ceil(len(words) * 0.6)):
                return value
    return ""


EMPTY_OCR = re.compile(r"^(?:khong co (?:chu|van ban|noi dung|text)(?: tren (?:slide|hinh|anh)(?: bai giang)?)?|"
                       r"no (?:text|readable text)(?: on (?:the )?slide)?)[.!]?$", re.I)


def text_repetition_runs(text, minimum=8, maximum_width=8):
    """Offsets of long consecutive 1-8 token phrases, excluding numeric table rows."""
    matches = list(re.finditer(r"\w+", text or ""))
    tokens = [strip_tones(m.group()) for m in matches]
    runs, index = [], 0
    while index < len(tokens):
        best = None
        for width in range(1, maximum_width + 1):
            phrase = tokens[index:index + width]
            if len(phrase) != width or not any(any(c.isalpha() for c in w) for w in phrase):
                continue
            end = index + width
            while tokens[end:end + width] == phrase:
                end += width
            repeats = (end - index) // width
            if repeats >= minimum and (best is None or end > best["end_token"]):
                best = {"phrase": " ".join(phrase), "repeats": repeats,
                        "start_token": index, "end_token": end,
                        "start_char": matches[index].start(), "end_char": matches[end - 1].end()}
        if best:
            runs.append(best)
            index = best["end_token"]
        else:
            index += 1
    return runs


def guard_ocr_text(text):
    """Remove failed output as evidence; keep raw output and reasons for a GPU retry."""
    lines = (text or "").splitlines()
    placeholders = [line for line in lines if EMPTY_OCR.fullmatch(strip_tones(line).strip())]
    text = "\n".join(line for line in lines if line not in placeholders)
    runs = text_repetition_runs(text, maximum_width=64)
    # Code/formulas can legitimately repeat; do not infer decoder failure from their tokens.
    runs = [run for run in runs if run["repeats"] >= 32
            or not code_or_formula(text[run["start_char"]:run["end_char"]])]
    removed = sum(run["end_token"] - run["start_token"] for run in runs)
    failed = bool(runs and removed / max(1, len(re.findall(r"\w+", text))) >= 0.5)
    for run in reversed(runs):
        text = text[:run["start_char"]] + text[run["end_char"]:]
    reasons = (["empty_ocr_placeholder"] if placeholders else []) + (["ocr_repetition_loop"] if runs else [])
    return ("" if failed else text.strip("\n")), {"status": "failed" if failed or (placeholders and not text.strip())
                                                    else "partial" if reasons else "available",
                                               "reasons": reasons, "loops": runs,
                                               "evidence_excluded": failed or bool(placeholders and not text.strip())}


def flag_asr_loops(words):
    """Keep every raw word/timestamp; only loop words are excluded from evidence spans."""
    tokens, owners = [], []
    for index, word in enumerate(words):
        for token in syllables(word["text"]):
            tokens.append(token)
            owners.append(index)
    runs = text_repetition_runs(" ".join(tokens))
    for run in runs:
        indices = sorted(set(owners[run["start_token"]:run["end_token"]]))
        run["word_ids"] = [words[i]["word_id"] for i in indices]
        run["start_sec"] = min(words[i]["start_sec"] for i in indices)
        run["end_sec"] = max(words[i]["end_sec"] for i in indices)
        for index in indices:
            words[index]["evidence_excluded"] = True
            words[index].setdefault("failure_reasons", []).append("asr_repetition_loop")
    return runs


# Reviewed aliases from audit 3. Canonical terms require local or repeated lecture OCR
# support. This is one global glossary, not arbitrary OCR n-grams or per-video tuning.
ASR_TERM_ALIASES = {heard: "xác suất" for heard in ("xác xuất", "sát xuất", "sắc xuất", "sát suất")}
ASR_TERM_ALIASES.update({"áp xuất": "áp suất", "nhiệt độ xôi": "nhiệt độ sôi",
                         "trí phí": "chi phí", "cao chào": "cao trào",
                         "mặt chặn": "mặt trận", "xuất điện động": "suất điện động"})


def correct_glossary_words(words, segments, lecture_support=False, code_feedback=False):
    """Atomic reviewed replacements with printed support and raw-word provenance."""
    words = [dict(word, text_asr=word.get("text_asr", word["text"])) for word in words]
    aliases = dict(ASR_TERM_ALIASES)
    if code_feedback:
        # A reviewed x/s repair for literal program feedback. Unlike technical terms,
        # it requires the local print message; lecture-wide support is insufficient.
        aliases["xai rồi"] = "sai rồi"
    support, printed_sources, feedback_words = {}, {}, set()
    for segment in segments:
        text = segment["visual"]["ocr_text_clean"].lower()
        terms = {term for term in ASR_TERM_ALIASES.values()
                 if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)}
        if code_feedback and re.search(r"\bprint\s*\([^\n]*[\"']sai roi\b", strip_tones(text)):
            terms.add("sai rồi")
            feedback_words.update(segment["speech"]["word_ids"])
        if terms:
            for term in terms:
                printed_sources.setdefault(term, []).append(segment["segment_id"])
            for word_id in segment["speech"]["word_ids"]:
                support[word_id] = (segment["segment_id"], terms)
    if lecture_support:
        # The aliases are reviewed terms, never arbitrary OCR n-grams. Require two
        # printed occurrences in the lecture before allowing non-local support.
        global_terms = {term for term, ids in printed_sources.items() if len(ids) >= 2}
        for word in words:
            segment_id, local_terms = support.get(word["word_id"], (None, set()))
            support[word["word_id"]] = (segment_id, local_terms | global_terms)
    corrections, index = [], 0
    ordered_aliases = sorted(aliases, key=lambda value: len(value.split()), reverse=True)
    while index < len(words):
        matched = False
        for alias in ordered_aliases:
            width = len(alias.split())
            window = words[index:index + width]
            if (len(window) != width or any(w.get("evidence_excluded") for w in window)
                    or not all(w["word_id"] in support for w in window)
                    or (not lecture_support and len({support[w["word_id"]][0] for w in window}) != 1)
                    or any(re.search(r"[.!?,;:]$", w["text"].strip()) for w in window[:-1])
                    or any(len(syllables(w["text"])) != 1 for w in window)):
                continue
            heard = " ".join(syllables(w["text"])[0] for w in window)
            if heard != alias:
                continue
            fixed = aliases[alias]
            if fixed == "sai rồi" and not all(w["word_id"] in feedback_words for w in window):
                continue
            if not all(fixed in support[w["word_id"]][1] for w in window):
                continue
            for word, printed in zip(window, fixed.split()):
                word["text"] = re.sub(r"\w+", lambda m: printed.capitalize() if m.group()[0].isupper() else printed,
                                      word["text"], count=1)
            corrections.append({"at_sec": window[0]["start_sec"], "heard": heard, "fixed": fixed,
                                "policy": "printed_code_feedback" if fixed == "sai rồi" else
                                          "reviewed_lecture_glossary" if lecture_support else "reviewed_glossary",
                                "support_segment_id": support[window[0]["word_id"]][0],
                                **({"printed_support_segment_ids": printed_sources[fixed]} if lecture_support else {})})
            index += width
            matched = True
            break
        if not matched:
            index += 1
    return words, corrections


def lines_new_since(previous_text, text, min_key=12):
    """Lines of text not already on the previous slide (same line_key), and the share of long
    lines that were. A slide shown again after an ad, a reveal split in two or a document
    page moved down a little repeats ~32 % of its lines on L25; the first line (title) stays."""
    seen = {line_key(line) for line in (previous_text or "").splitlines()}
    lines = (text or "").splitlines()
    long_keys = [line_key(line) for line in lines if len(line_key(line)) >= min_key]
    share = sum(key in seen for key in long_keys) / len(long_keys) if long_keys else 0.0
    kept = [line for index, line in enumerate(lines)
            if index == 0 or len(line_key(line)) < min_key or line_key(line) not in seen]
    return "\n".join(kept), round(share, 3)


def noise_region(region):
    """A figure region whose model description is only about the lecturer, a logo or decoration."""
    if region.get("kind") not in {"diagram", "figure", "image"}:
        return False
    text = f"{region.get('text') or ''} {region.get('description') or ''}".lower()
    has_numbers = bool(re.search(r"\d", region.get("text") or ""))
    return any(word in text for word in NOISE_REGION_WORDS) and not has_numbers


def clean_regions(regions):
    """Marks noise and duplicate regions (same text on the same slide) with noise=True."""
    seen = set()
    out = []
    for region in regions or []:
        region = dict(region)
        key = line_key(f"{region.get('text') or ''}{region.get('description') or ''}")
        region["noise"] = noise_region(region) or (key in seen and bool(key))
        seen.add(key)
        out.append(region)
    return out


def syllables(text):
    return re.findall(r"\w+", (text or "").lower())


def split_syllable(s):
    """(onset, rest) of a toneless syllable for the confusable onsets."""
    for onset in ("tr", "ch", "s", "x"):
        if s.startswith(onset):
            return onset, s[len(onset):]
    return "", s


def confusable(heard, printed):
    """True if a heard syllable can be a mishearing of the printed one: identical, or the onset
    is s/x or ch/tr swapped (rest equal up to tone and a c/t or n/ng final)."""
    if heard == printed:
        return True
    a, b = strip_tones(heard), strip_tones(printed)
    (oa, ra), (ob, rb) = split_syllable(a), split_syllable(b)
    if not oa or not ob or oa == ob or {oa, ob} not in ({"s", "x"}, {"ch", "tr"}):
        return False
    if ra == rb:
        return True
    for x, y in FINALS:
        for p, q in ((x, y), (y, x)):
            if ra.endswith(p) and ra[:-len(p)] + q == rb:
                return True
    return False


def ocr_vocabulary(texts, sizes=(2, 3)):
    """Counts of the syllable n-grams printed on the slides (accented, lower case)."""
    vocab = {}
    for text in texts:
        for line in clean_ocr_text(text).splitlines():
            syl = syllables(line)
            for n in sizes:
                for i in range(len(syl) - n + 1):
                    gram = tuple(syl[i:i + n])
                    if all(s.isalpha() for s in gram):
                        vocab[gram] = vocab.get(gram, 0) + 1
    return vocab


def correct_words(words, vocab, min_count=2, atomic=True):
    """Replace misheard n-grams of ASR words by the near-identical n-gram printed on the slides.
    words: [{'text', ...}] in time order. Returns (new words, corrections)."""
    words = [dict(w) for w in words]
    by_size = {}
    for gram, count in vocab.items():
        if count >= min_count:
            by_size.setdefault(len(gram), []).append(gram)
    corrections = []
    i = 0
    while i < len(words):
        done = False
        for n in ((3, 2) if atomic else (2, 3)):
            window = words[i:i + n]
            if len(window) < n:
                continue
            heard = tuple(syllables(w["text"])[0] if syllables(w["text"]) else "" for w in window)
            if "" in heard or heard in vocab:
                continue
            matches = [gram for gram in by_size.get(n, ())
                       if gram != heard and all(confusable(h, g) for h, g in zip(heard, gram))]
            if len({m for m in matches}) != 1:
                continue
            gram = matches[0]
            for word, printed in zip(window, gram):
                if syllables(word["text"])[0] != printed:
                    word.setdefault("text_asr", word["text"])
                    word["text"] = re.sub(r"\w+", lambda m: printed.capitalize() if m.group(0)[0].isupper() else printed,
                                          word["text"], count=1)
            corrections.append({"at_sec": window[0]["start_sec"], "heard": " ".join(heard), "fixed": " ".join(gram)})
            # Legacy ablations retain the greedy skip. Otherwise a compatible repair may
            # overlap this window ("thức sát xuất" -> "thức xác" -> "xác suất").
            i += 1 if atomic else n
            done = True
            break
        if not done:
            i += 1
    return words, corrections
