"""Gold set 2 tools: the OCR review page builder and the OCR / OCR gate scoring (no GPU, no network)."""

from pathlib import Path
import json
import random
import sys
import tempfile
import unicodedata
import unittest

from PIL import Image

GOLD = Path(__file__).resolve().parents[1] / "gold"
sys.path.insert(0, str(GOLD))
import build_ocr_review  # noqa: E402
import eval_ocr  # noqa: E402


def naive_levenshtein(a, b):
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        row = [i]
        for j, char_b in enumerate(b, 1):
            row.append(min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = row
    return previous[-1]


def visual(text="", status="read", regions=()):
    return {"image_path": None, "ocr_text": text, "regions": list(regions), "ocr_status": status,
            "ocr_gate_score": 0.05 if status == "skipped_no_content" else 0.9}


def write_export(root, videos):
    """videos: {video: [(content_kind, visual), ...]} -> a fast-pipeline export layout."""
    for video, segments in videos.items():
        bundle = root / video / "04_bundle"
        (bundle / "keyframes").mkdir(parents=True)
        records = []
        for index, (kind, record) in enumerate(segments):
            name = f"keyframes/{'ad' if kind == 'ad' else 'seg'}_{index:05d}.jpg"
            Image.new("RGB", (64, 36), (index * 7 % 255, 90, 160)).save(bundle / name)
            records.append({"segment_id": f"{video}:seg_{index:05d}", "sequence": index,
                            "start_sec": index * 10.0, "end_sec": index * 10.0 + 10,
                            "content_kind": kind, "visual": {**record, "image_path": name}})
        (bundle / "segments.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


class TextMetricsTest(unittest.TestCase):
    def test_levenshtein_matches_dynamic_programming(self):
        rng = random.Random(7)
        alphabet = "abđăâêôơưáàạ x"
        for _ in range(500):
            a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
            b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
            self.assertEqual(eval_ocr.levenshtein(a, b), naive_levenshtein(a, b), (a, b))

    def test_normalize_unifies_unicode_and_drops_markup(self):
        self.assertEqual(eval_ocr.normalize(unicodedata.normalize("NFD", "Tiếng Việt")), "Tiếng Việt")
        self.assertEqual(eval_ocr.normalize("<td>Đạo hàm</td>\n|---|---|\n| $x^2$ | \\(y\\) |"), "Đạo hàm x^2 y")

    def test_word_recall_counts_repeated_words(self):
        self.assertAlmostEqual(eval_ocr.word_recall("hàm số", "Hàm số hàm số"), 0.5)
        self.assertIsNone(eval_ocr.word_recall("abc", "  "))

    def test_prediction_text_keeps_only_verbatim_regions(self):
        record = visual("Bảng biến thiên", regions=[{"kind": "table", "text": "x 0 1"},
                                                    {"kind": "diagram", "text": "trục Ox"}])
        self.assertEqual(eval_ocr.prediction_text(record), "Bảng biến thiên\nx 0 1")


class ReviewPageTest(unittest.TestCase):
    def test_sampling_is_spread_and_reproducible(self):
        items = list(range(100))
        first = build_ocr_review.pick_spread(items, 10, build_ocr_review.video_rng(0, "V1"))
        again = build_ocr_review.pick_spread(items, 10, build_ocr_review.video_rng(0, "V1"))
        self.assertEqual(first, again)
        self.assertEqual([value // 10 for value in first], list(range(10)))
        self.assertEqual(build_ocr_review.pick_spread(items[:3], 10, random.Random(0)), [0, 1, 2])

    def test_build_page_then_score_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            export, out = Path(tmp) / "export", Path(tmp) / "review"
            write_export(export, {
                "V1": [("speaker", visual(status="skipped_no_content")),
                       ("slide", visual("Đạo hàm cấp hai")),
                       ("ad", {"ocr_text": "", "regions": [], "ocr_status": "skipped_ad"}),
                       ("slide", visual("Bảng", regions=[{"kind": "table", "text": "x 0 1"}]))],
                "V2": [("slide", visual(f"Slide {i}")) for i in range(30)],
            })
            (export / "V3" / "01_slides").mkdir(parents=True)  # a video that failed after stage 01
            meta = build_ocr_review.build(export, out, per_video=4, blind=2, seed=0)
            self.assertEqual(meta["missing_videos"], ["V3"])

            page = (out / "ocr_review.html").read_text(encoding="utf-8")
            self.assertNotIn("/*DATA*/null", page)
            data = json.loads((out / "ocr_review_items.json").read_text(encoding="utf-8"))
            by_video = {v["video"]: v["items"] for v in data["videos"]}
            self.assertEqual(len(by_video["V1"]), 3, "ad keyframes are not labelled")
            self.assertEqual(sum(it["transcribe"] for it in by_video["V2"]), 4)
            sampled = [it for items in by_video.values() for it in items if it["transcribe"]]
            self.assertEqual(meta["transcribe"], len(sampled))
            blind = [it for it in sampled if it.get("blind")]
            self.assertEqual(len(blind), 2)
            self.assertTrue(all(it["prefill"] == "" for it in blind))
            for it in sampled:
                self.assertTrue((out / it["image"]).is_file())
                self.assertTrue((out / it["thumb"]).is_file())

            gold = {"format": "ocr-gold-v1", "labels": {
                # the gate skipped a keyframe that does hold text: a false skip, and its text is lost
                "V1:seg_00000": {"has_content": True, "transcribe": True, "checked": True, "text": "Bài 1"},
                "V1:seg_00001": {"has_content": True, "transcribe": True, "checked": True, "text": "Đạo hàm cấp hai"},
                "V1:seg_00003": {"has_content": True, "transcribe": True, "checked": False, "text": "Bảng x 0 1"},
                "V2:seg_00000": {"has_content": False, "transcribe": False, "checked": False},
            }}
            result = eval_ocr.evaluate(gold, eval_ocr.load_visuals(export))
            self.assertEqual(result["text"]["slides"], 2, "unchecked slides are not scored")
            self.assertEqual(result["text"]["lost"], 1)
            exact = next(row for row in result["slides"] if row["segment_id"] == "V1:seg_00001")
            self.assertEqual(exact["cer"], 0.0)
            self.assertEqual(result["gate"]["content_skipped"], 1)
            self.assertEqual(result["gate"]["false_skips"], ["V1:seg_00000"])
            self.assertEqual(result["gate"]["empty_read"], 1)

            gold["labels"]["V9:seg_00000"] = {"has_content": True}
            with self.assertRaises(ValueError):
                eval_ocr.evaluate(gold, eval_ocr.load_visuals(export))

            only = build_ocr_review.build(export, Path(tmp) / "dev", per_video=4, blind=0, only=["V2"])
            self.assertEqual((only["videos"], only["keyframes"], only["missing_videos"]), (["V2"], 30, []))
            with self.assertRaises(SystemExit):
                build_ocr_review.build(export, Path(tmp) / "bad", only=["V7"])


if __name__ == "__main__":
    unittest.main()
