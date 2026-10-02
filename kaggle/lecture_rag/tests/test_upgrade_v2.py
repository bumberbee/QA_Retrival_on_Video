"""Upgrade v2 switches (baseline behaviour stays the default): OCR loop cleaning, the OCR gate's
extra views, the hybrid re-read area and keeping the speech of ad segments that sound like the lesson."""

from pathlib import Path
import json
import types
import unittest

from PIL import Image, ImageStat

CODE =Path(__file__).resolve().parents[1] / "notebook_code"


def load_code(*names):
    module = types.ModuleType("test_upgrade_" + names[-1])
    for filename in ("common.py", *(name + ".py" for name in names)):
        source = CODE / filename
        exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), module.__dict__)
    return module


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False, **kwargs):
        values = [ord(character) for character in text]
        return [0] + values + [0] if add_special_tokens else values

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(value) for value in tokens if value)


class OcrLoopCleaningTests(unittest.TestCase):
    def setUp(self):
        self.clean = load_code("textclean").clean_ocr_text

    def test_a_looping_line_is_kept_once(self):
        text = "Bài 3\n" + "\\( \\therefore m = \\frac{3}{11} \\)\n" * 269 + "Câu 1"
        self.assertEqual(self.clean(text), "Bài 3\n\\( \\therefore m = \\frac{3}{11} \\)\nCâu 1")

    def test_two_identical_lines_and_digit_rows_stay(self):
        self.assertEqual(self.clean("A. 12 cm\nA. 12 cm\nB. 13 cm"), "A. 12 cm\nA. 12 cm\nB. 13 cm")
        self.assertEqual(self.clean("x 0 0 0 0 0 0 0"), "x 0 0 0 0 0 0 0")

    def test_a_phrase_looping_inside_one_line_is_kept_once(self):
        self.assertEqual(self.clean("2. cộc bộ cộc bộ cộc bộ cộc bộ cộc bộ"), "2. cộc bộ")


class OcrGateViewTests(unittest.TestCase):
    def setUp(self):
        self.visual = load_code("visual")
        self.image = Image.new("RGB", (400, 200))
        self.slide = Image.new("RGB", (100, 100), (255, 255, 255))
        self.image.paste(self.slide, (300, 100))   # a small slide in the bottom right quarter

    @staticmethod
    def gate(views, threshold=0.10):
        # P(readable content): high only for a mostly white view (the inset slide)
        def check(images):
            return [0.9 if ImageStat.Stat(image.convert("L")).mean[0] > 120 else 0.02 for image in images]
        return check, {"threshold": threshold, "views": views}

    def test_frame_view_is_the_baseline(self):
        skip, scores = self.visual.gate_keyframe(self.gate("frame"), self.image)
        self.assertTrue(skip)
        self.assertEqual(set(scores), {"frame"})

    def test_crops_keep_a_keyframe_with_a_small_slide(self):
        skip, scores = self.visual.gate_keyframe(self.gate("frame+crops"), self.image)
        self.assertFalse(skip)
        self.assertEqual(len(scores["crops"]), 5)

    def test_crops_still_skip_an_empty_keyframe_and_score_nothing_above_the_threshold(self):
        skip, scores = self.visual.gate_keyframe(self.gate("frame+crops"), Image.new("RGB", (400, 200)))
        self.assertTrue(skip)
        white = Image.new("RGB", (400, 200), (255, 255, 255))
        skip, scores = self.visual.gate_keyframe(self.gate("frame+crops"), white)
        self.assertFalse(skip)
        self.assertNotIn("crops", scores, "crops are only scored for keyframes the frame view would skip")

    def test_roi_crop_comes_first(self):
        crops = self.visual.gate_crops(self.image, (0.75, 0.5, 1.0, 1.0))
        self.assertEqual(len(crops), 6)
        self.assertEqual(crops[0].size, (100, 100))

    def test_no_gate_never_skips(self):
        self.assertEqual(self.visual.gate_keyframe(None, self.image), (False, {}))

    def test_unknown_view_is_rejected_before_loading_a_model(self):
        adfilter = load_code("adfilter")
        with self.assertRaises(ValueError):
            adfilter.load_ocr_gate("cpu", views="roi-only")


class HybridRescueAreaTests(unittest.TestCase):
    def read(self, big_area):
        visual = load_code("visual")
        passes, crops = [], []

        def unlimited(image):
            passes.append(image.size)
            if len(passes) == 1:   # a text page taken for one picture covering 30 % of the slide
                return "<|det|>image [0, 0, 600, 500]<|/det|>\n<|det|>text [600, 0, 999, 100]<|/det|>Tiêu đề"
            return "<|det|>text [0, 0, 999, 999]<|/det|>" + "Xác suất có điều kiện của biến cố A khi B. " * 3

        def read_crop(image, box):
            crops.append(box)
            return "raw", {"kind": "decorative"}
        blocks, _, raw = visual.hybrid_read_slide(Image.new("RGB", (1000, 600)), unlimited, read_crop,
                                                  big_area=big_area)
        return blocks, json.loads(raw), passes, crops

    def test_default_area_sends_the_block_to_the_vlm(self):
        blocks, raw, passes, crops = self.read(0.4)
        self.assertEqual((len(passes), len(crops)), (1, 1))
        self.assertFalse(raw["second_pass"])

    def test_lower_area_reads_the_block_again_and_rescues_its_text(self):
        blocks, raw, passes, crops = self.read(0.2)
        self.assertEqual((len(passes), len(crops)), (2, 0))
        self.assertTrue(any("Xác suất có điều kiện" in block["text"] for block in blocks))


def segment(index, kind, text, start):
    return {"segment_id": f"V:seg_{index:05d}", "sequence": index, "start_sec": start, "end_sec": start + 60,
            "content_kind": kind,
            "speech": {"text": text, "fallback_span_ids": [],
                       "spans": [{"span_id": f"V:seg_{index:05d}:speech_0000", "start_sec": start,
                                  "end_sec": start + 60, "text": text}]},
            "visual": {"occurrence_id": f"V:seg_{index:05d}:visual", "image_path": f"keyframes/seg_{index:05d}.jpg",
                       "keyframe_time_sec": start, "ocr_text": "" if kind == "ad" else "Xác suất có điều kiện",
                       "regions": [], "title": ""}}


LESSON = ("hôm nay chúng ta học xác suất có điều kiện của biến cố a khi biết biến cố b đã xảy ra "
          "công thức xác suất có điều kiện bằng xác suất của a giao b chia cho xác suất của b ")
PROMO = ("để có được thành công như hôm nay tôi đã có những ước mơ và tôi đã hiện thực hóa ước mơ "
         "của mình bắt đầu từ trường đại học với chương trình đào tạo chuẩn quốc tế đăng ký ngay ")


class AdSpeechReviewTests(unittest.TestCase):
    def setUp(self):
        self.alignment = load_code("textclean", "alignment")
        self.segments = [segment(0, "slide", LESSON, 0), segment(1, "ad", PROMO, 60),
                         segment(2, "ad", LESSON + "ví dụ một hộp có bốn mươi tấm thẻ", 120),
                         segment(3, "ad", "xin chào", 180), segment(4, "slide", LESSON, 240)]

    def test_lesson_speech_inside_an_ad_is_flagged_and_promo_is_not(self):
        reviewed = self.alignment.review_ad_speech(self.segments, 0.25)
        self.assertFalse(reviewed[1]["ad_suspect"])
        self.assertTrue(reviewed[2]["ad_suspect"])
        self.assertFalse(reviewed[3]["ad_suspect"], "too few word pairs to judge")
        self.assertIsNone(reviewed[3]["ad_review"]["lesson_overlap"])
        self.assertNotIn("ad_suspect", reviewed[0])

    def test_index_keeps_suspect_speech_and_drops_plain_ads(self):
        indexing = load_code("indexing")
        reviewed = self.alignment.review_ad_speech(self.segments, 0.25)
        chunks = indexing.make_evidence_chunks(reviewed, [], CharacterTokenizer(), video_id="V")
        indexed = {chunk["segment_id"] for chunk in chunks}
        self.assertIn("V:seg_00002", indexed)
        self.assertNotIn("V:seg_00001", indexed)
        self.assertNotIn("V:seg_00003", indexed)
        baseline = indexing.make_evidence_chunks(self.segments[:1] + self.segments[4:], [], CharacterTokenizer(),
                                                 video_id="V")
        self.assertEqual({chunk["segment_id"] for chunk in baseline}, {"V:seg_00000", "V:seg_00004"})


if __name__ == "__main__":
    unittest.main()
