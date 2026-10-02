"""Data/media tests. Model stubs exercise wiring, not OCR/ASR/embedding accuracy."""

from pathlib import Path
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "notebook_code"


EXTRA_MODULES = {"alignment": ["textclean.py"]}


def load_code(name):
    module = types.ModuleType("test_" + name)
    for filename in ("common.py", *EXTRA_MODULES.get(name, []), name + ".py"):
        source = CODE / filename
        exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), module.__dict__)
    return module


class CharacterTokenizer:
    """Deliberately pessimistic token counts for testing budget/overlap behavior."""

    def encode(self, text, add_special_tokens=False, **kwargs):
        values = [ord(character) for character in text]
        return [0] + values + [0] if add_special_tokens else values

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(value) for value in tokens if value)


class FixtureEmbeddingModel:
    """Small deterministic features; never presented as a real semantic encoder."""

    max_seq_length = 512

    def __init__(self, *args, **kwargs):
        self.tokenizer = CharacterTokenizer()

    def get_sentence_embedding_dimension(self):
        return 4

    def _first_module(self):
        return types.SimpleNamespace(auto_model=types.SimpleNamespace(
            config=types.SimpleNamespace(_commit_hash="fixture-encoder-revision"),
        ))

    def encode(self, texts, **kwargs):
        import numpy as np
        common = load_code("retrieval")

        def encode_one(text):
            normalized = common.normalize_search(text)
            vector = np.array([
                0.1 + normalized.count("hang chai"),
                0.1 + normalized.count("bao luc"),
                0.1 + normalized.count("mo dau"),
                0.1 + normalized.count("ket luan"),
            ], dtype=np.float32)
            return vector / np.linalg.norm(vector)
        return encode_one(texts) if isinstance(texts, str) else np.stack(
            [encode_one(text) for text in texts],
        )


def slide_image(lines, start=0, width=640, height=360):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 20, 610, 50), fill="navy")  # repeated title bar
    for line in range(start, start + lines):
        y = 80 + line * 34
        # Word-like blocks with a per-line pattern so different lines differ.
        for word in range(5):
            x = 50 + word * 105 + (line * 17) % 40
            draw.rectangle((x, y, x + 60 + (line * 11 + word * 7) % 30, y + 10), fill="black")
    return image


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ["numpy", "cv2", "PIL"]),
                     "Cần numpy, OpenCV và Pillow")
class ProgressiveRevealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detection = load_code("detection")

    def views(self, images):
        return [self.detection.reveal_view(image) for image in images]

    def test_lines_added_one_by_one_are_one_slide(self):
        groups, audit = self.detection.group_reveal_states(
            self.views([slide_image(count) for count in (1, 2, 3, 4)]))
        self.assertEqual(groups, [[0, 1, 2, 3]])
        self.assertTrue(all(item["decision"] == "merge_content_kept" for item in audit))

    def test_replaced_text_with_same_title_bar_starts_new_slide(self):
        groups, _ = self.detection.group_reveal_states(
            self.views([slide_image(4), slide_image(4, start=5)]))
        self.assertEqual(groups, [[0], [1]])

    def test_speaker_cut_and_return_split_reveals(self):
        from PIL import Image, ImageDraw
        speaker = Image.new("RGB", (640, 360), (120, 90, 70))
        ImageDraw.Draw(speaker).ellipse((250, 60, 390, 250), fill=(220, 180, 150))
        groups, _ = self.detection.group_reveal_states(
            self.views([slide_image(2), slide_image(3), speaker, slide_image(3)]))
        self.assertEqual(groups, [[0, 1], [2], [3]])

    def test_ocr_noise_on_identical_image_does_not_split_but_kept_words_merge(self):
        views = self.views([slide_image(4), slide_image(4)])
        noisy, _ = self.detection.group_reveal_states(
            views, ["tam giác vuông cạnh huyền định lý", "phương trình bậc hai nghiệm số delta"])
        self.assertEqual(noisy, [[0, 1]])
        kept, audit = self.detection.group_reveal_states(
            self.views([slide_image(2), slide_image(3)]),
            ["định lý pytago tam giác vuông", "định lý pytago tam giác vuông cạnh huyền"])
        self.assertEqual(kept, [[0, 1]])
        self.assertEqual(audit[0]["decision"], "merge_text_retained")

    def test_low_slide_probability_blocks_merge(self):
        groups, audit = self.detection.group_reveal_states(
            self.views([slide_image(2), slide_image(3)]), p_slides=[0.9, 0.1])
        self.assertEqual(groups, [[0], [1]])
        self.assertEqual(audit[0]["decision"], "split_non_slide")

    def test_unstable_animation_after_a_slide_is_split_off(self):
        import numpy as np
        from PIL import Image, ImageDraw
        detection = self.detection

        def record(time, image):
            signature = detection.frame_signature(image)
            return {"time_sec": float(time),
                    "signature": (signature * 255).round().astype("uint8")}

        slide = slide_image(4)
        spinning = []
        random = np.random.default_rng(7)
        for step in range(8):  # advert footage: every frame looks different
            frame = Image.new("RGB", (640, 360), tuple(int(v) for v in random.integers(0, 255, 3)))
            draw = ImageDraw.Draw(frame)
            for _ in range(12):
                x, y = random.integers(0, 600), random.integers(0, 320)
                draw.rectangle((x, y, x + 120, y + 80),
                               fill=tuple(int(v) for v in random.integers(0, 255, 3)))
            spinning.append(frame)
        records = ([record(t, slide) for t in range(10)]
                   + [record(10 + t, image) for t, image in enumerate(spinning)])
        self.assertEqual(detection.hidden_change_splits(records, 0, 18), [10.0])
        # The animation alone and an unchanged slide are left as they are.
        self.assertEqual(detection.hidden_change_splits(records[10:], 10, 18), [])
        self.assertEqual(detection.hidden_change_splits(records[:10], 0, 10), [])

    def test_removed_content_is_not_a_reveal(self):
        groups, _ = self.detection.group_reveal_states(
            self.views([slide_image(5), slide_image(2)]))
        self.assertEqual(groups, [[0], [1]])


class TextCleanTests(unittest.TestCase):
    def setUp(self):
        self.t = load_code("textclean")

    def test_repeated_lines_and_noise_are_removed_but_question_labels_kept(self):
        slides = [f"Câu {n}.\nGiáo viên: Thầy LÊ MINH HUY\n25\nNội dung {n} [Unreadable]" for n in range(1, 6)]
        repeated = self.t.repeated_lines(slides)
        cleaned = self.t.clean_ocr_text(slides[0], repeated)
        self.assertEqual(cleaned.splitlines(), ["Câu 1.", "Nội dung 1"])
        regions = self.t.clean_regions([
            {"kind": "diagram", "text": "Giáo viên, Logo Thanh Niên", "description": "Một giáo viên đang giảng"},
            {"kind": "graph", "text": '{"data": [54.8]}', "description": ""},
            {"kind": "graph", "text": '{"data": [54.8]}', "description": ""}])
        self.assertEqual([r["noise"] for r in regions], [True, False, True])

    def test_asr_terms_fixed_from_slides_without_trusting_slide_tones(self):
        vocab = self.t.ocr_vocabulary(["Xác suất của biến cố A", "tính xác suất của biến cố", "số lê và số lê"])
        heard = "xác xuất của biến cố là số lẻ tính sát xuất"
        words = [{"text": w, "start_sec": float(i), "end_sec": i + 0.5} for i, w in enumerate(heard.split())]
        fixed, corrections = self.t.correct_words(words, vocab)
        self.assertEqual(" ".join(w["text"] for w in fixed), "xác suất của biến cố là số lẻ tính xác suất")
        self.assertEqual(fixed[1]["text_asr"], "xuất")
        self.assertEqual(len(corrections), 2)


class HybridSlideReadTests(unittest.TestCase):
    def test_large_image_block_is_reread_and_duplicates_skip_the_vlm(self):
        from PIL import Image
        visual = load_code("visual")
        page = "<|det|>text [0, 0, 999, 200]<|/det|>" + "Câu 3. Trong hộp có 40 tấm thẻ, chọn ngẫu nhiên 3 thẻ. " * 3

        passes = []

        def unlimited(image):
            passes.append(image.size)
            if len(passes) == 1:       # whole frame: the text page is taken for one picture
                return ("<|det|>image [0, 0, 999, 999]<|/det|>\n<|det|>image [0, 0, 999, 999]<|/det|>\n"
                        "<|det|>image [500, 500, 999, 999]<|/det|>")
            return page                # the big block read on its own: text
        calls = []

        def read_crop(image, box):
            calls.append(box)
            return "raw", {"kind": "chart", "data": [1, 2]}
        blocks, regions, raw = visual.hybrid_read_slide(Image.new("RGB", (1000, 600)), unlimited, read_crop)
        self.assertIn("40 tấm thẻ", blocks[0]["text"])
        self.assertTrue(json.loads(raw)["second_pass"])
        # the duplicate big block was read once; only the small figure (25 % of the slide) went to the VLM
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(passes), 2)
        self.assertEqual(regions[0]["kind"], "graph")


class NotebookAndContractTests(unittest.TestCase):
    def test_six_notebooks_validate_and_every_code_cell_compiles(self):
        import nbformat

        notebooks = sorted((ROOT / "notebooks").glob("0*.ipynb"))
        self.assertEqual(len(notebooks), 6)
        for path in notebooks:
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type == "code":
                        self.assertEqual(cell.outputs, [])
                        self.assertIsNone(cell.execution_count)
                        if not cell.source.startswith("%pip"):
                            compile(cell.source, f"{path.name}:cell{index}", "exec")

    def test_embedded_sources_match_maintainable_python_sources(self):
        spec = importlib.util.spec_from_file_location("notebook_builder", ROOT / "build_notebooks.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        for item in builder.SPECS:
            notebook = json.loads((ROOT / "notebooks" / (item["name"] + ".ipynb")).read_text(encoding="utf-8"))
            sources = ["".join(cell["source"]).strip() for cell in notebook["cells"]]
            self.assertIn((CODE / item["module"]).read_text(encoding="utf-8").strip(), sources)
            self.assertIn((CODE / "common.py").read_text(encoding="utf-8").strip(), sources)

    def test_artifact_checks_reject_corruption_and_unfinished_runs(self):
        pipeline = load_code("alignment")
        with tempfile.TemporaryDirectory(prefix="lecture-contract-") as temporary:
            directory = pipeline.begin_stage(temporary, 1)
            pipeline.write_json(directory / "intervals.json", [])
            manifest = pipeline.finish_stage(
                directory, 1, "L25_V001", {"sha256": "test-video", "duration_sec": 12}, {},
            )
            self.assertEqual(pipeline.load_stage(directory, 1)["artifact_id"], manifest["artifact_id"])
            pipeline.write_json(directory / "intervals.json", ["changed"])
            with self.assertRaisesRegex(ValueError, "checksum"):
                pipeline.load_stage(directory, 1)
            pipeline.begin_stage(temporary, 1)
            with self.assertRaisesRegex(ValueError, "chưa hoàn tất"):
                pipeline.load_stage(directory, 1)

    def test_branch_lineage_must_match_even_for_the_same_video(self):
        pipeline = load_code("alignment")
        parent = {"video_id": "L25_V001", "source": {"sha256": "same"},
                  "pipeline_version": "l25-v1", "stage": 1, "artifact_id": "new-detection"}
        child = {**parent, "stage": 2, "parents": {"1": "old-detection"}}
        with self.assertRaisesRegex(ValueError, "phiên bản"):
            pipeline.require_parent(child, parent)
        child["parents"]["1"] = parent["artifact_id"]
        child["source"] = {"sha256": "different-video"}
        with self.assertRaisesRegex(ValueError, "cùng video"):
            pipeline.require_parent(child, parent)

    def test_half_open_ownership_and_gap_validation(self):
        pipeline = load_code("alignment")
        intervals = [
            {"segment_id": "a", "start_sec": 0, "end_sec": 3},
            {"segment_id": "b", "start_sec": 3, "end_sec": 8},
        ]
        self.assertEqual(pipeline.owner_interval(intervals, 3)["segment_id"], "b")
        self.assertEqual(pipeline.owner_interval(intervals, 8)["segment_id"], "b")
        pipeline.verify_intervals(intervals, 8)
        intervals[1]["start_sec"] = 3.2
        with self.assertRaisesRegex(ValueError, "trống"):
            pipeline.verify_intervals(intervals, 8)


class TimingAndRetrievalTests(unittest.TestCase):
    def test_malformed_vision_output_cannot_enter_search_evidence(self):
        visual, indexing = load_code("visual"), load_code("indexing")
        for malformed in ('{"description": "unfinished', '!' * 900):
            parsed = visual.parse_visual_response(malformed)
            self.assertEqual(parsed["parse_status"], "unstructured")
            self.assertEqual(parsed["description"], "")
            segment = {
                "segment_id": "s", "sequence": 0, "start_sec": 0, "end_sec": 10,
                "speech": {"spans": [{"span_id": "spoken", "start_sec": 1,
                                      "end_sec": 3, "text": "Lời giảng được giữ lại."}],
                           "fallback_span_ids": []},
                "visual": {"title": "", "occurrence_id": "s:visual",
                           "image_path": "s.jpg", "keyframe_time_sec": 2,
                           "ocr_text": "Chữ trên slide được giữ lại.",
                           # Reproduce an already-exported legacy bundle.
                           "visual_description": malformed,
                           "description_status": "unstructured", "regions": []},
            }
            chunks = indexing.make_evidence_chunks([segment], [], CharacterTokenizer())
            self.assertEqual({chunk["source_type"] for chunk in chunks}, {"ocr", "speech"})
            self.assertTrue(all(malformed not in chunk["embedding_text"] for chunk in chunks))
            segment["visual"].update(description_status="json", visual_description="Ảnh bài giảng.")
            chunks = indexing.make_evidence_chunks([segment], [], CharacterTokenizer())
            self.assertNotIn("visual", {chunk["source_type"] for chunk in chunks})
            chunks = indexing.make_evidence_chunks(
                [segment], [], CharacterTokenizer(), include_generated_descriptions=True,
            )
            self.assertIn("visual", {chunk["source_type"] for chunk in chunks})

    def test_ocr_crop_coordinates_and_reading_lines(self):
        from PIL import Image
        visual = load_code("visual")

        class Reader:
            def readtext(self, pixels, **kwargs):
                self.shape = pixels.shape
                return [([[20, 20], [120, 20], [120, 60], [20, 60]], "Nội dung", 0.9)]

        reader = Reader()
        blocks = visual.extract_ocr_blocks(reader, Image.new("RGB", (200, 100)),
                                           roi=(0.25, 0.1, 0.95, 0.9), scale=2)
        self.assertEqual(reader.shape[:2], (160, 280))
        for actual, expected in zip(blocks[0]["bbox"], (0.3, 0.2, 0.55, 0.4)):
            self.assertAlmostEqual(actual, expected)
        unordered = [
            {"text": "chài", "bbox": [0.4, 0.109, 0.5, 0.149]},
            {"text": "hàng", "bbox": [0.2, 0.1, 0.3, 0.14]},
            {"text": "Dòng sau", "bbox": [0.2, 0.25, 0.5, 0.29]},
        ]
        self.assertEqual(visual.ocr_lines_text(visual.order_ocr_lines(unordered)), "hàng chài\nDòng sau")

    def test_grouped_speech_preserves_every_source_span(self):
        indexing = load_code("indexing")
        spans = [{"span_id": f"s{i}", "start_sec": i * 2, "end_sec": i * 2 + 1,
                  "text": "Một câu ngắn."} for i in range(8)]
        groups = indexing.group_speech_spans(spans, CharacterTokenizer(), 45, 50)
        self.assertLess(len(groups), len(spans))
        self.assertEqual([key for group in groups for key in group["source_ids"]],
                         [span["span_id"] for span in spans])
        self.assertEqual(groups[0]["start_sec"], 0)
        self.assertEqual(groups[-1]["end_sec"], 15)

    def test_current_slide_anchors_do_not_include_unrelated_global_hits(self):
        retrieval = load_code("retrieval")
        segments = [{"segment_id": "current", "start_sec": 0, "end_sec": 10},
                    {"segment_id": "other", "start_sec": 10, "end_sec": 20}]
        chunks = [{"chunk_id": "a", "segment_id": "current", "start_sec": 1,
                   "end_sec": 9, "source_type": "speech"},
                  {"chunk_id": "b", "segment_id": "other", "start_sec": 11,
                   "end_sec": 19, "source_type": "speech"}]
        anchors = retrieval.choose_anchors(chunks, [{"chunk_id": "b"}, {"chunk_id": "a"}],
                                           segments, current_time=5, prefer_current=True, count=4)
        self.assertEqual(anchors, ["a"])

    def test_boundary_word_and_coarse_span_keep_all_evidence(self):
        alignment = load_code("alignment")
        intervals = [
            {"segment_id": "a", "start_sec": 0, "end_sec": 10,
             "image_path": "a.jpg", "keyframe_time_sec": 1},
            {"segment_id": "b", "start_sec": 10, "end_sec": 20,
             "image_path": "b.jpg", "keyframe_time_sec": 11},
        ]
        visuals = [{"segment_id": "a", "content_kind": "speaker"},
                   {"segment_id": "b", "content_kind": "slide"}]
        transcript = {
            "status": "completed",
            "words": [{"word_id": "w", "start_sec": 9.8, "end_sec": 10.2, "text": "qua"}],
            "fallback_spans": [{"span_id": "f", "start_sec": 9, "end_sec": 11,
                               "text": "Một câu không có timestamp từng từ."}],
            "slide_audio_clips": [{"segment_id": item["segment_id"]} for item in intervals],
        }
        segments = alignment.align_records(intervals, visuals, transcript, 20)
        self.assertEqual(segments[0]["speech"]["word_ids"], [])
        self.assertEqual(segments[1]["speech"]["word_ids"], ["w"])
        self.assertEqual([segment["speech"]["fallback_span_ids"] for segment in segments],
                         [["f"], ["f"]])
        self.assertEqual(segments[0]["next_segment_id"], "b")
        self.assertEqual(segments[1]["previous_segment_id"], "a")

    def test_context_cannot_drop_all_anchors_and_return_only_neighbors(self):
        retrieval = load_code("retrieval")
        template = {
            "segment_id": "s", "sequence": 0, "source_type": "speech",
            "start_sec": 0, "end_sec": 10, "title": "", "image_path": "a.jpg",
            "keyframe_time_sec": 1, "timing_precision": "word_span",
        }
        chunks = [
            {**template, "chunk_id": "s:u0000:p000", "text": "a" * 1000},
            {**template, "chunk_id": "s:u0001:p000", "text": "short optional neighbor"},
        ]
        with self.assertRaisesRegex(ValueError, "anchor"):
            retrieval.budget_context(
                chunks, [(chunks[0]["chunk_id"], "anchor"), (chunks[1]["chunk_id"], "next")],
                CharacterTokenizer(), maximum_tokens=300,
            )

    def test_asr_windows_group_short_slides_and_prefer_pauses(self):
        asr = load_code("transcription")
        intervals = [{"segment_id": str(index), "start_sec": start, "end_sec": end}
                     for index, (start, end) in enumerate(zip(
                         [0, 10, 12, 80], [10, 12, 80, 137],
                     ))]
        windows = asr.make_asr_windows(137, intervals, [43.5, 87.2])
        self.assertEqual([window["core_end_sec"] for window in windows], [43.5, 87.2, 137])
        self.assertEqual(windows[0]["segment_ids"], ["0", "1", "2"])
        self.assertTrue(all(window["core_end_sec"] - window["core_start_sec"] <= 60
                            for window in windows))
        self.assertEqual(windows[0]["clip_start_sample"], 0)
        self.assertEqual(windows[-1]["clip_end_sample"], 137 * 16000)
        for left, right in zip(windows, windows[1:]):
            self.assertEqual(left["core_end_sec"], right["core_start_sec"])
            self.assertGreater(left["clip_end_sec"], right["clip_start_sec"])

    def test_word_at_core_boundary_is_kept_once_with_global_timing(self):
        asr = load_code("transcription")
        windows = asr.make_asr_windows(90, [{"segment_id": "s", "start_sec": 0, "end_sec": 90}])
        first, second = windows
        word = {"start_sec": 44.8, "end_sec": 45.2, "text": "ranh giới", "probability": 0.9}
        left, _ = asr.globalize_window_words([word], first, 90)
        relative = {**word, "start_sec": 44.8 - second["clip_start_sec"],
                    "end_sec": 45.2 - second["clip_start_sec"]}
        right, _ = asr.globalize_window_words([relative], second, 90)
        self.assertEqual(len(left), 0)
        self.assertEqual(len(right), 1)
        self.assertEqual((right[0]["start_sec"], right[0]["end_sec"]), (44.8, 45.2))

    def test_overlap_dedup_preserves_real_repeated_words(self):
        asr = load_code("transcription")
        words = [
            {"text": "rất", "start_sec": 1.0, "end_sec": 1.3, "window_id": "a"},
            {"text": "rất", "start_sec": 1.04, "end_sec": 1.31, "window_id": "b"},
            {"text": "rất", "start_sec": 1.4, "end_sec": 1.7, "window_id": "a"},
        ]
        kept, removed = asr.deduplicate_overlap_words(words)
        self.assertEqual(removed, 1)
        self.assertEqual(len(kept), 2)

    def test_fts_vietnamese_accent_insensitive_and_literal_query(self):
        indexing, retrieval = load_code("indexing"), load_code("retrieval")
        chunks = [
            {"chunk_id": "one", "title": "Văn học", "text": "Người đàn bà hàng chài chịu bạo lực."},
            {"chunk_id": "two", "title": "Toán", "text": "Đạo hàm của hàm số."},
        ]
        with tempfile.TemporaryDirectory(prefix="lecture-fts-") as temporary:
            database = Path(temporary) / "lexical.sqlite3"
            indexing.build_lexical_index(database, chunks)
            hits = retrieval.lexical_search(database, "nguoi dan ba hang chai")
            self.assertEqual(hits[0][0], "one")
            self.assertEqual(retrieval.lexical_search(database, '""" OR (NEAR) --'), [])

    def test_token_splitting_preserves_all_text_and_respects_overlap(self):
        indexing, retrieval = load_code("indexing"), load_code("retrieval")
        text = " ".join(f"từ{index}" for index in range(150))
        tokenizer = CharacterTokenizer()
        parts = indexing.split_text_tokens(text, tokenizer, limit=100, overlap=20)
        self.assertGreater(len(parts), 3)
        self.assertTrue(all(len(tokenizer.encode(part)) <= 100 for part in parts))
        # Token splitting need not align with whitespace: verify substring coverage.
        self.assertEqual(parts[0], text[:100].strip())
        self.assertEqual(parts[-1], text[(len(parts) - 1) * 80:].strip())
        self.assertEqual(retrieval.merge_overlap_text("A B C D", "C D E F"), "A B C D E F")

    def test_temporal_questions_require_time_instead_of_guessing(self):
        retrieval = load_code("retrieval")
        self.assertTrue(retrieval.temporal_question("Giải thích đoạn này giúp tôi."))
        self.assertFalse(retrieval.temporal_question("Nhân vật được phân tích như thế nào?"))


MEDIA_AVAILABLE = (
    all(importlib.util.find_spec(name) for name in ["numpy", "cv2", "PIL", "av"])
    and shutil.which("ffmpeg") and shutil.which("ffprobe")
)


@unittest.skipUnless(MEDIA_AVAILABLE, "Cần dependencies media trong requirements-test.txt và FFmpeg")
class MediaAndPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="lecture-media-tests-")
        cls.root = Path(cls.temporary.name)
        cls.detection = load_code("detection")
        cls.video = cls.root / "L25_V001.mkv"
        cls.detection.run_command([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=10:d=3",
            "-f", "lavfi", "-i", "color=c=red:s=320x180:r=10:d=4",
            "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=10:d=5",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v];[3:a]asetpts=PTS+1.5/TB[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le", "-t", "12", str(cls.video),
        ])
        cls.stage01, cls.manifest01, cls.intervals = cls.detection.run_slide_detection(
            cls.video, cls.root / "artifacts", sample_hz=2.0, min_interval_sec=1.0,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_real_frame_pts_find_nonuniform_boundaries_and_repeated_occurrences(self):
        self.assertEqual(len(self.intervals), 3)
        self.assertAlmostEqual(self.intervals[1]["start_sec"], 3, delta=0.15)
        self.assertAlmostEqual(self.intervals[2]["start_sec"], 7, delta=0.15)
        self.assertNotEqual(self.intervals[0]["segment_id"], self.intervals[2]["segment_id"])
        for item in self.intervals:
            self.assertLessEqual(item["start_sec"], item["keyframe_time_sec"])
            self.assertLess(item["keyframe_time_sec"], item["end_sec"])

    def test_single_pass_scan_matches_seek_method_and_cleans_up(self):
        self.assertFalse((self.stage01 / "_scan").exists())
        self.assertTrue(self.intervals[1]["boundary_method"].startswith("single_pass"))
        _, _, seek_intervals = self.detection.run_slide_detection(
            self.video, self.root / "seek", sample_hz=2.0, min_interval_sec=1.0, scan="seek",
        )
        self.assertEqual(len(seek_intervals), len(self.intervals))
        for fast, slow in zip(self.intervals, seek_intervals):
            self.assertAlmostEqual(fast["start_sec"], slow["start_sec"], delta=0.15)

    def test_delayed_audio_and_silent_tail_keep_video_clock(self):
        import numpy as np
        asr = load_code("transcription")
        target = self.root / "normalized.wav"
        asr.normalize_audio(self.video, self.manifest01["source"], target)
        audio = asr.read_pcm_audio(target).astype("float32") / 32768
        self.assertAlmostEqual(len(audio) / 16000, 12, places=2)
        self.assertLess(np.sqrt(np.mean(audio[:16000] ** 2)), 0.0001)
        self.assertGreater(np.sqrt(np.mean(audio[2 * 16000:3 * 16000] ** 2)), 0.05)
        self.assertLess(np.sqrt(np.mean(audio[6 * 16000:7 * 16000] ** 2)), 0.0001)

    def test_cursor_and_ignored_presenter_do_not_replace_white_slide_detection(self):
        from PIL import Image, ImageDraw
        detection = self.detection
        first = Image.new("RGB", (640, 360), "white")
        second = first.copy()
        for image, shift in [(first, 0), (second, 50)]:
            draw = ImageDraw.Draw(image)
            for line in range(7):
                draw.rectangle((40 + shift, 40 + line * 35, 350 + line * 20,
                                48 + line * 35), fill="black")
        cursor = first.copy()
        ImageDraw.Draw(cursor).rectangle((610, 300, 614, 304), fill="red")
        signature = detection.frame_signature(first)
        self.assertGreater(detection.signature_distance(signature, detection.frame_signature(second)), 0.055)
        self.assertLess(detection.signature_distance(signature, detection.frame_signature(cursor)), 0.055)
        presenter = first.copy()
        ImageDraw.Draw(presenter).rectangle((512, 252, 639, 359), fill="green")
        ignored = [(0.8, 0.7, 1.0, 1.0)]
        self.assertEqual(detection.signature_distance(
            detection.frame_signature(first, ignored_regions=ignored),
            detection.frame_signature(presenter, ignored_regions=ignored),
        ), 0)

    def test_transient_frame_is_rejected_and_returning_slide_is_retained(self):
        from PIL import Image
        detection = self.detection
        a = detection.frame_signature(Image.new("RGB", (320, 180), "blue"))
        b = detection.frame_signature(Image.new("RGB", (320, 180), "red"))
        values = [a, a, a, b, a, b, b, b, a, a, a, a]
        samples = [{"time_sec": index, "signature": value} for index, value in enumerate(values)]
        boundaries, _, _ = detection.detect_stable_boundaries(samples, 12)
        self.assertEqual(boundaries, [0, 5, 8, 12])

    def test_hybrid_ocr_output_parsing_and_thread_capture(self):
        visual = load_code("visual")
        raw = ("<|det|>title [100, 50, 900, 120]<|/det|>Độ hụt khối\n"
               "<|det|>equation [100, 200, 600, 260]<|/det|>\\Delta m = Zm_p + (A-Z)m_n - m_X\n"
               "<|det|>chart [500, 400, 999, 999]<|/det|>\n")
        blocks = visual.parse_unlimited_blocks(raw)
        self.assertEqual([b["type"] for b in blocks], ["title", "equation", "chart"])
        self.assertAlmostEqual(blocks[2]["bbox"][2], 1.0)
        self.assertEqual(blocks[0]["text"], "Độ hụt khối")
        # Box on the ROI crop -> full keyframe.
        self.assertEqual(visual.roi_to_image_box([0, 0, 1, 1], (0.1, 0.2, 0.5, 0.6)), [0.1, 0.2, 0.5, 0.6])
        import io
        import threading
        real = io.StringIO()
        capture = visual.ThreadStdoutCapture(real)
        mine = io.StringIO()
        capture.buffers[threading.get_ident()] = mine
        other = threading.Thread(target=lambda: capture.write("ASR 1/3\n"))
        other.start(); other.join()
        capture.write("<|det|>text [0, 0, 1, 1]<|/det|>x")
        self.assertEqual(real.getvalue(), "ASR 1/3\n")
        self.assertIn("<|det|>", mine.getvalue())

    def test_ad_filter_cuts_ad_segment_and_later_stages_skip_it(self):
        # The red middle part (3-7 s) plays the ad; a fixture scorer stands in for SigLIP.
        detection = load_code("detection")
        source = CODE / "adfilter.py"
        exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), detection.__dict__)
        scorer = (lambda images: [1.0 if image.getpixel((5, 5))[0] > 150 else 0.0 for image in images],
                  {"mode": "fixture", "model": "fixture", "threshold": 0.5, "student_sha256": None})
        output = self.root / "ads"
        stage01, manifest01, intervals = detection.run_slide_detection(
            self.video, output, sample_hz=2.0, min_interval_sec=1.0, ad_scorer=scorer,
            ad_options={"smooth": 1, "min_ad": 2, "max_gap": 1})
        ads = [item for item in intervals if item["content_kind"] == "ad"]
        self.assertEqual(len(ads), 1)
        self.assertEqual(manifest01["ad_segment_count"], 1)
        self.assertAlmostEqual(ads[0]["start_sec"], 3, delta=0.2)
        self.assertAlmostEqual(ads[0]["end_sec"], 7, delta=0.2)
        self.assertTrue((stage01 / ads[0]["image_path"]).exists())
        self.assertTrue((stage01 / "ad_filter.json").exists())

        visual, asr = load_code("visual"), load_code("transcription")
        alignment, indexing = load_code("alignment"), load_code("indexing")

        class Reader:
            def __init__(self, *args, **kwargs):
                pass

            def readtext(self, pixels, **kwargs):
                return [([[10, 10], [290, 10], [290, 40], [10, 40]], "Mở đầu và kết luận.", 0.95)]

        with mock.patch.dict(sys.modules, {"easyocr": types.SimpleNamespace(Reader=Reader)}):
            stage02, _, records = visual.run_visual_extraction(stage01, output, ocr_device="cpu", ocr_scale=1.0)
        ad_record = next(record for record in records if record["segment_id"] == ads[0]["segment_id"])
        self.assertEqual((ad_record["content_kind"], ad_record["ocr_text"]), ("ad", ""))
        words = [{"start_sec": 3.5 + index * 0.4, "end_sec": 3.8 + index * 0.4, "text": text, "probability": 0.9}
                 for index, text in enumerate("Đăng ký tuyển sinh ngay hôm nay.".split())]
        with mock.patch.object(asr, "load_asr_model", return_value=(object(), "fixture", "cpu", "int8")), \
                mock.patch.object(asr, "transcribe_window",
                                  return_value={"words": words, "fallback_spans": [], "language": "vi"}):
            stage03, _, transcript = asr.run_transcription(None, self.video, output)
        stage04, manifest04, segments = alignment.run_alignment(stage01, stage02, stage03, output)
        ad_segment = next(segment for segment in segments if segment["content_kind"] == "ad")
        self.assertEqual(len(ad_segment["speech"]["word_ids"]), len(transcript["words"]))
        self.assertEqual(manifest04["ad_word_count"], len(transcript["words"]))
        app = json.loads((stage04 / "transcript_app.json").read_text(encoding="utf-8"))
        self.assertEqual(app["segments"], [])
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": types.SimpleNamespace(SentenceTransformer=FixtureEmbeddingModel),
        }):
            stage05, _, chunks = indexing.run_indexing(stage04, output)
            engine = load_code("retrieval").load_retrieval_engine(stage05)
        self.assertTrue(chunks)
        self.assertNotIn(ad_segment["segment_id"], {chunk["segment_id"] for chunk in chunks})
        # "this slide" asked while the ad plays: answered from the slide shown before the ad
        moment = load_code("retrieval").retrieve_question(engine, "Giải thích slide này.", current_time=5)
        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.assertEqual({by_id[c]["segment_id"] for c in moment["anchor_chunk_ids"]}, {segments[0]["segment_id"]})

    def test_ocr_gate_skips_frames_without_content_but_keeps_segments(self):
        visual = load_code("visual")
        calls = []

        class Reader:
            def __init__(self, *args, **kwargs):
                pass

            def readtext(self, pixels, **kwargs):
                calls.append(1)
                return [([[10, 10], [290, 10], [290, 40], [10, 40]], "Định luật bảo toàn năng lượng.", 0.95)]

        # fixture gate: the red middle part is "content", the blue parts "no content"
        gate = (lambda images: [0.9 if image.getpixel((5, 5))[0] > 150 else 0.1 for image in images],
                {"model": "fixture", "threshold": 0.5})
        with mock.patch.dict(sys.modules, {"easyocr": types.SimpleNamespace(Reader=Reader)}):
            stage02, manifest, records = visual.run_visual_extraction(
                self.stage01, self.root / "gate", ocr_device="cpu", ocr_scale=1.0, ocr_gate=gate)
        self.assertEqual(len(records), len(self.intervals))
        self.assertEqual([r["ocr_status"] for r in records], ["skipped_no_content", "read", "skipped_no_content"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(records[0]["ocr_text"], "")
        self.assertIn(records[0]["content_kind"], {"speaker", "other"})
        self.assertEqual((manifest["ocr_read_count"], manifest["ocr_skipped_no_content"]), (1, 2))

    def test_shared_model_loads_once_across_threads(self):
        import threading
        common = load_code("adfilter")
        loads = []

        def loader():
            loads.append(1)
            return object()

        results = []
        threads = [threading.Thread(target=lambda: results.append(common.shared_model(("x",), loader)))
                   for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(loads), 1)
        self.assertEqual(len({id(item) for item in results}), 1)

    def test_ad_intervals_split_slides_and_absorb_short_remnants(self):
        ad = load_code("adfilter")
        intervals = [{"start_sec": 0, "end_sec": 50}, {"start_sec": 50, "end_sec": 120},
                     {"start_sec": 120, "end_sec": 200}]
        self.assertEqual(ad.split_ad_intervals(intervals, [[64, 110]]),
                         [[0, 50, 0], [50, 64, 1], [64, 110, None], [110, 120, 1], [120, 200, 2]])
        self.assertEqual(ad.split_ad_intervals(intervals, [[49, 119]]),
                         [[0, 49, 0], [49, 120, None], [120, 200, 2]])
        mask = ad.ad_mask([0] * 10 + [1] * 7 + [0] + [1] * 3 + [0] * 10 + [1] * 3 + [0] * 5 + [1] * 2,
                          0.5, smooth=1, min_ad=5, max_gap=2)
        self.assertEqual(ad.ad_runs(mask), [[10, 21], [39, 41]])   # gap filled; short ad dropped; closing logo kept
        self.assertEqual(ad.ad_time_intervals([k + 0.5 for k in range(20)], [0] * 5 + [1] * 6 + [0] * 9,
                                              20, [0, 5.2, 11.1, 20]), [[5.2, 11.1]])

    def test_all_artifact_stages_with_explicit_model_stubs(self):
        visual = load_code("visual")
        asr = load_code("transcription")
        alignment = load_code("alignment")
        indexing = load_code("indexing")
        retrieval = load_code("retrieval")
        output = self.root / "artifacts"

        class Reader:
            def __init__(self, *args, **kwargs):
                pass

            def readtext(self, pixels, **kwargs):
                text = "Người đàn bà hàng chài chịu bạo lực." if pixels[0, 0, 0] > 150 else "Mở đầu và kết luận."
                return [([[10, 10], [290, 10], [290, 40], [10, 40]], text, 0.95)]

        def describe(image):
            red = image.getpixel((0, 0))[0] > 150
            return json.dumps({
                "content_kind": "slide", "title": "Người đàn bà hàng chài" if red else "Mở đầu",
                "description": "Bạo lực trong gia đình." if red else "Mở đầu bài học.",
                "regions": [], "uncertainties": ["Fixture model, không phải OCR/vision thật."],
            })

        with mock.patch.dict(sys.modules, {"easyocr": types.SimpleNamespace(Reader=Reader)}), \
                mock.patch.object(visual, "load_local_describer",
                                  return_value=(describe, "fixture-vision-revision")):
            stage02, _, _ = visual.run_visual_extraction(
                self.stage01, output, vision_mode="all", ocr_device="cpu", ocr_scale=1.0,
            )
        def hybrid_read(image):
            return ([{"text": "Δm = Zm_p", "bbox": [0.1, 0.1, 0.5, 0.2], "confidence": None,
                      "source": "unlimited-ocr", "block_type": "equation", "line_index": 0, "reading_order": 0}],
                    [{"kind": "graph", "bbox": [0.5, 0.4, 1.0, 1.0], "text": '{"data": [54.8]}',
                      "description": "", "extraction_status": "hybrid_ocr_unverified"}], "raw")
        with mock.patch.object(visual, "load_hybrid_ocr", return_value=hybrid_read):
            hybrid02, _, hybrid_records = visual.run_visual_extraction(
                self.stage01, output / "hybrid", ocr_engine="hybrid")
        self.assertEqual(hybrid_records[0]["ocr_text"], "Δm = Zm_p")
        self.assertEqual(hybrid_records[0]["regions"][0]["kind"], "graph")
        self.assertTrue((hybrid02 / hybrid_records[0]["regions"][0]["image_path"]).exists())
        fixture_words = [
            {"start_sec": 3.2 + index * 0.3, "end_sec": 3.4 + index * 0.3,
             "text": text, "probability": 0.9}
            for index, text in enumerate("Người đàn bà hàng chài chịu bạo lực.".split())
        ]
        with mock.patch.object(asr, "load_asr_model",
                               return_value=(object(), "fixture-asr-revision", "cpu", "int8")), \
                mock.patch.object(asr, "transcribe_window",
                                  return_value={"words": fixture_words, "fallback_spans": [],
                                                "language": "vi"}):
            stage03, _, transcript = asr.run_transcription(self.stage01, self.video, output)
            # Stage 03 without stage 01 (parallel run): stage 04 cuts the slide clips.
            parallel = output / "parallel"
            stage03_parallel, manifest03, _ = asr.run_transcription(None, self.video, parallel)
        self.assertEqual(manifest03["parents"], {})
        stage04_parallel, _, parallel_segments = alignment.run_alignment(
            self.stage01, stage02, stage03_parallel, parallel)
        self.assertTrue(all((stage04_parallel / segment["audio"]["path"]).exists()
                            for segment in parallel_segments))
        stage04, _, segments = alignment.run_alignment(self.stage01, stage02, stage03, output)
        self.assertEqual([s["speech"]["word_ids"] for s in segments],
                         [s["speech"]["word_ids"] for s in parallel_segments])
        word_ids = [word_id for segment in segments for word_id in segment["speech"]["word_ids"]]
        self.assertEqual(len(word_ids), len(transcript["words"]))
        self.assertEqual(len(word_ids), len(set(word_ids)))
        self.assertTrue(all(segment["speech"]["word_ids"] == [] for segment in
                            [segments[0], segments[2]]))
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": types.SimpleNamespace(SentenceTransformer=FixtureEmbeddingModel),
        }):
            stage05, manifest05, chunks = indexing.run_indexing(stage04, output)
            engine = retrieval.load_retrieval_engine(stage05)
        self.assertTrue(all(chunk["encoded_tokens"] <= 512 for chunk in chunks))
        result = retrieval.retrieve_question(
            engine, "Người đàn bà hàng chài chịu bạo lực như thế nào?", maximum_tokens=4000,
        )
        first_hit = {chunk["chunk_id"]: chunk for chunk in chunks}[result["hits"][0]["chunk_id"]]
        self.assertEqual(first_hit["segment_id"], self.intervals[1]["segment_id"])
        self.assertLessEqual(result["context_tokens"], 4000)
        times = [block["start_sec"] for block in result["evidence"]]
        self.assertEqual(times, sorted(times))
        with self.assertRaisesRegex(ValueError, "CURRENT_TIME_SEC"):
            retrieval.retrieve_question(engine, "Giải thích slide này.")
        moment = retrieval.retrieve_question(engine, "Giải thích slide này.", current_time=4)
        self.assertTrue(moment["parameters"]["prefer_current"])
        result["question"] += " <script>alert(1)</script>"
        stage06, _, exported = retrieval.export_retrieval(engine, result, output, maximum_images=2)
        retrieval.load_stage(stage06, 6)
        self.assertTrue(exported["images"])
        self.assertEqual(exported["source_sha256"], self.manifest01["source"]["sha256"])
        preview = (stage06 / "preview.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>alert(1)</script>", preview)
        self.assertIn("&lt;script&gt;", preview)
        self.assertEqual(manifest05["index_status"], "ready")


if __name__ == "__main__":
    unittest.main()
