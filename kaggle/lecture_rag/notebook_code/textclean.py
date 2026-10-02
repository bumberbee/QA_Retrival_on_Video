"""Clean slide OCR before indexing and correct lecture terms the ASR misheard, using the
words printed on the slides of the same video (stage 04; raw outputs stay untouched).

OCR: lines repeated on most slides (channel logo, lecturer name, school, page numbers),
"[Unreadable]" markers and figure regions that only describe the lecturer or a logo are
noise for search. ASR: Whisper often hears a domain term with a confusable consonant or
final ("xác xuất" for "xác suất", "số chẳng" for "số chẵn"); when the slides print a
near-identical term and never the heard one, the heard words are replaced. Corrections
keep the word timing and the original text (text_asr).
"""

UNREADABLE = re.compile(r"\[(?:unreadable|non-text)\]", re.I)
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


def clean_ocr_text(text, repeated=frozenset()):
    kept = []
    for line in (text or "").splitlines():
        line = UNREADABLE.sub("", line).strip()
        key = line_key(line)
        if not line or key in repeated:
            continue
        if len(line) <= 3 and not re.search(r"[A-Za-zÀ-ỹ]{2}", line):   # page numbers, bullets
            continue
        for bad, good in OCR_FIXES.items():
            line = line.replace(bad, good)
        kept.append(line)
    return "\n".join(kept)


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


def correct_words(words, vocab, min_count=2):
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
        for n in (2, 3):
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
            i += n
            done = True
            break
        if not done:
            i += 1
    return words, corrections
