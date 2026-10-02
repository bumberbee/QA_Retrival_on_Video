"""Scores the pipeline's OCR and OCR gate against gold set 2 (labels exported from ocr_review.html).

    python kaggle/lecture_rag/gold/eval_ocr.py <ocr_gold.json> <export_dir> [--out metrics.json]

<export_dir> is the fast-pipeline export the review page was built from. Text metrics use only
keyframes whose text a human typed and ticked as checked:
  CER        character edit distance / gold length, after normalize() (micro: summed over slides)
  word recall share of gold words (multiset) the OCR found; "lost" = recall < 0.5
A keyframe the OCR gate skipped counts with empty OCR text: the text it held is lost.
Gate metrics use every keyframe with a "readable content" label.
"""

import argparse
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LOST_RECALL = 0.5
VERBATIM_REGIONS = {"table", "formula"}  # region text read verbatim; figure/graph text is not


def prediction_text(visual):
    """What the pipeline read off a keyframe: OCR lines plus verbatim table and formula regions."""
    parts = [visual.get("ocr_text") or ""]
    parts += [region.get("text") or "" for region in visual.get("regions") or []
              if region.get("kind") in VERBATIM_REGIONS]
    return "\n".join(part for part in parts if part.strip())


def ocr_status(visual):
    """"read", "skipped_no_content" (OCR gate) or "skipped_ad"; older exports lack the field."""
    if visual.get("ocr_status"):
        return visual["ocr_status"]
    if visual.get("description_status") in {"skipped_no_content", "skipped_ad"}:
        return visual["description_status"]
    return "read"


def normalize(text):
    """NFC, no HTML tags, markdown table rules or LaTeX math delimiters, single spaces."""
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", " ", text, flags=re.M)
    text = text.replace("|", " ")
    text = re.sub(r"\\[()\[\]]|\$+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text):
    return re.findall(r"\w+", normalize(text).lower())


def levenshtein(a, b):
    """Edit distance (insert, delete, substitute = 1), bit-parallel (Myers 1999, Hyyrö 2001):
    O(len(b)) big-integer steps, fast enough for slides with thousands of characters."""
    if not a or not b:
        return len(a) + len(b)
    m = len(a)
    peq = {}
    for i, char in enumerate(a):
        peq[char] = peq.get(char, 0) | (1 << i)
    mask, high = (1 << m) - 1, 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for char in b:
        eq = peq.get(char, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return score


def word_recall(prediction, gold):
    gold_words = Counter(words(gold))
    if not gold_words:
        return None
    found = Counter(words(prediction))
    return sum(min(count, found[word]) for word, count in gold_words.items()) / sum(gold_words.values())


def load_visuals(export_dir):
    """segment_id -> visual record, from <video>/04_bundle/segments.json."""
    visuals = {}
    for path in sorted(Path(export_dir).glob("*/04_bundle/segments.json")):
        for segment in json.loads(path.read_text(encoding="utf-8")):
            visual = dict(segment.get("visual") or {})
            visual.setdefault("content_kind", segment.get("content_kind"))
            visuals[segment["segment_id"]] = visual
    return visuals


def summarize_text(rows):
    if not rows:
        return {"slides": 0}
    gold_chars = sum(row["gold_chars"] for row in rows)
    return {
        "slides": len(rows),
        "cer_micro": round(sum(row["edits"] for row in rows) / max(1, gold_chars), 4),
        "cer_median": round(statistics.median(row["cer"] for row in rows), 4),
        "word_recall_mean": round(statistics.mean(row["word_recall"] for row in rows), 4),
        "lost": sum(row["lost"] for row in rows),
        "lost_share": round(sum(row["lost"] for row in rows) / len(rows), 4),
    }


def evaluate(gold, visuals):
    labels = gold["labels"]
    missing = sorted(item for item in labels if item not in visuals)
    if missing:
        raise ValueError(f"{len(missing)} nhãn không có trong export (sai export?): {missing[:3]}")

    text_rows = []
    for item, label in labels.items():
        text = normalize(label.get("text"))
        if not (label.get("transcribe") and label.get("checked") and label.get("has_content") and text):
            continue
        visual = visuals[item]
        status = ocr_status(visual)
        prediction = normalize(prediction_text(visual)) if status == "read" else ""
        edits = levenshtein(prediction, text)
        recall = word_recall(prediction, text) or 0.0
        text_rows.append({"segment_id": item, "video": item.split(":")[0], "blind": bool(label.get("blind")),
                          "ocr_status": status, "gold_chars": len(text), "edits": edits,
                          "cer": round(edits / len(text), 4), "word_recall": round(recall, 4),
                          "lost": recall < LOST_RECALL})

    gate = Counter()
    false_skips = []
    for item, label in labels.items():
        if label.get("has_content") is None:
            continue
        skipped = ocr_status(visuals[item]) == "skipped_no_content"
        key = ("content" if label["has_content"] else "empty") + ("_skipped" if skipped else "_read")
        gate[key] += 1
        if key == "content_skipped":
            false_skips.append(item)

    by_video = defaultdict(list)
    for row in text_rows:
        by_video[row["video"]].append(row)
    return {
        "text": summarize_text(text_rows),
        "text_prefilled": summarize_text([row for row in text_rows if not row["blind"]]),
        "text_blind": summarize_text([row for row in text_rows if row["blind"]]),
        "text_by_video": {video: summarize_text(rows) for video, rows in sorted(by_video.items())},
        "gate": {
            "labelled": sum(gate.values()),
            "content_read": gate["content_read"],
            "content_skipped": gate["content_skipped"],  # the gate threw away a slide with text
            "empty_skipped": gate["empty_skipped"],
            "empty_read": gate["empty_read"],
            "false_skips": sorted(false_skips),
        },
        "slides": sorted(text_rows, key=lambda row: row["segment_id"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gold", type=Path)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    if gold.get("format") != "ocr-gold-v1":
        sys.exit("File nhãn không đúng định dạng ocr-gold-v1 (xuất từ ocr_review.html).")
    result = evaluate(gold, load_visuals(args.export_dir))

    def line(name, s):
        if not s["slides"]:
            return f"  {name:12} chưa có slide đã kiểm"
        return (f"  {name:12} {s['slides']:4} slide · CER {s['cer_micro']:.3f} (trung vị {s['cer_median']:.3f})"
                f" · recall từ {s['word_recall_mean']:.3f} · mất chữ {s['lost']}/{s['slides']}")

    print("Chữ trên slide")
    print(line("tất cả", result["text"]))
    print(line("điền sẵn", result["text_prefilled"]))
    print(line("gán từ đầu", result["text_blind"]))
    for video, summary in result["text_by_video"].items():
        print(line(video, summary))
    g = result["gate"]
    print(f"Bộ gác OCR ({g['labelled']} keyframe đã gán)")
    print(f"  bỏ nhầm slide có chữ {g['content_skipped']} · bỏ đúng {g['empty_skipped']}"
          f" · để lọt khung rỗng {g['empty_read']} · đọc slide có chữ {g['content_read']}")
    for item in g["false_skips"]:
        print("   bỏ nhầm:", item)
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
