"""Production port contracts, with no GPU, model download or external services."""

import ast
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from test_pipeline import CharacterTokenizer, load_code

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_fast_pipeline as fast
import build_full_walkthrough as walk
import build_notebooks as stages
import build_run_pipeline as single


def config_values(source):
    values = {}
    exec(compile(source, "production-config", "exec"), values)
    return values


def run_function(name, values):
    node = next(n for n in ast.parse(fast.RUN).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "production-run", "exec"), values)
    return values[name]


class OCRValidationTests(unittest.TestCase):
    def setUp(self):
        self.m = load_code("visual")

    def output(self, text="Nội dung slide", eos=True):
        return {"text": text, "token_ids": [10, 2 if eos else 99], "eos_token_id": [2, 3]}

    def test_requires_generated_eos_including_multiple_eos_ids(self):
        self.assertTrue(self.m.qwen_ocr_validation("Đã xong", [10, 3], [2, 3])["validated"])
        for tokens, eos in (([], 2), ([99], [2, 3]), ([2], None)):
            with self.subTest(tokens=tokens, eos=eos):
                result = self.m.qwen_ocr_validation("Đã xong <|im_end|>", tokens, eos)
                self.assertFalse(result["validated"])
                self.assertIn("ocr_missing_eos", result["reasons"])

    def test_eos_with_empty_output_is_invalid(self):
        self.assertIn("ocr_empty_output", self.m.qwen_ocr_validation("  ", [2], 2)["reasons"])

    def test_eight_repeats_at_end_and_32_word_phrases_are_rejected(self):
        for phrase in ("Câu hỏi", "123", " ".join(f"word{i}" for i in range(32))):
            with self.subTest(phrase=phrase):
                text = "nội dung mở đầu " + (phrase + " ") * 8
                result = self.m.qwen_ocr_validation(text, [2], 2)
                self.assertIn("ocr_repetition_loop", result["reasons"])
                self.assertGreaterEqual(result["max_consecutive_repeats"], 8)
                self.assertTrue(self.m.qwen_ocr_validation((phrase + " ") * 7, [2], 2)["validated"])
                self.assertEqual(self.m.qwen_ocr_validation((phrase + " ") * 7, [2], 2)["max_consecutive_repeats"], 7)

    def test_valid_first_attempt_does_not_retry(self):
        generate = Mock(return_value=self.output())
        result = self.m.qwen_ocr_with_retry(generate)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(generate.call_args.args[1:], (1536, 1.05))
        self.assertTrue(result["validated"])
        self.assertFalse(result["ocr_quality"]["evidence_excluded"])

    def test_invalid_first_attempt_gets_exact_experiment_retry(self):
        generate = Mock(side_effect=[self.output("Câu hỏi " * 8), self.output("    print('sai roi')")])
        result = self.m.qwen_ocr_with_retry(generate)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(generate.call_args.args, (self.m.QWEN_RETRY_PROMPTS["vi"], 2048, 1.10))
        self.assertTrue(result["validated"])
        self.assertEqual(result["text"], "    print('sai roi')")
        self.assertEqual(result["attempts"][0]["text"], "Câu hỏi " * 8)

    def test_second_failure_is_retained_and_never_gets_a_third_attempt(self):
        generate = Mock(return_value=self.output("partial", eos=False))
        result = self.m.qwen_ocr_with_retry(generate, "en")
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["text"], "partial")
        self.assertEqual(result["ocr_quality"]["status"], "failed")
        self.assertTrue(result["ocr_quality"]["evidence_excluded"])
        self.assertIn("ocr_missing_eos", result["ocr_quality"]["reasons"])
        self.assertEqual(generate.call_args.args[0], self.m.QWEN_RETRY_PROMPTS["en"])

    def test_generation_exception_retains_provenance_and_can_retry(self):
        generate = Mock(side_effect=[RuntimeError("fixture OOM"), self.output()])
        result = self.m.qwen_ocr_with_retry(generate)
        self.assertTrue(result["validated"])
        self.assertEqual(result["attempts"][0]["error"], "fixture OOM")

    def test_retry_and_validation_switches_are_explicit(self):
        generate = Mock(return_value=self.output(eos=False))
        result = self.m.qwen_ocr_with_retry(generate, retry_invalid=False)
        self.assertEqual(generate.call_count, 1)
        self.assertTrue(result["ocr_quality"]["evidence_excluded"])
        generate.reset_mock()
        result = self.m.qwen_ocr_with_retry(generate, validate=False)
        self.assertEqual(generate.call_count, 1)
        self.assertFalse(result["validation_enabled"])
        self.assertEqual(result["ocr_quality"]["status"], "unchecked")

    def test_content_crop_priority_padding_and_safe_fallback(self):
        image = Image.new("RGB", (1000, 500))
        interval = {"reveal_states": [{"display_roi": (.2, .2, .8, .8)}]}
        roi, method = self.m.qwen_content_roi(image, interval)
        self.assertLess(roi[0], .2)
        self.assertGreater(roi[2], .8)
        self.assertEqual(method, "expanded_state_display_roi")
        configured = (.1, .1, .9, .9)
        self.assertEqual(self.m.qwen_content_roi(image, interval, configured_roi=configured)[0], configured)
        self.assertEqual(self.m.qwen_content_roi(image, {}), (None, "full_frame_fallback"))
        with self.assertRaises(ValueError):
            self.m.qwen_content_roi(image, {}, configured_roi=(.8, .1, .2, .9))

    def test_model_adapter_forwards_pins_pixels_and_real_eos_to_retry(self):
        class Input(dict):
            def to(self, device):
                return self

        class Output:
            def __getitem__(self, key):
                return types.SimpleNamespace(tolist=lambda: [10, 2])

        model = types.SimpleNamespace(device="cuda:0", config=types.SimpleNamespace(_commit_hash="fixture"),
                                      generation_config=types.SimpleNamespace(eos_token_id=[2, 3]),
                                      generate=Mock(return_value=Output()))
        model.eval = lambda: model
        processor = Mock()
        processor.apply_chat_template.return_value = Input(
            input_ids=types.SimpleNamespace(shape=(1, 5)),
            image_grid_thw=types.SimpleNamespace(tolist=lambda: [[1, 16, 16]]))
        processor.decode.side_effect = ["repeat " * 8, "valid literal"]
        processor_loader = Mock(return_value=processor)
        model_loader = Mock(return_value=model)
        torch = types.SimpleNamespace(float16="float16", inference_mode=nullcontext)
        transformers = types.SimpleNamespace(
            AutoProcessor=types.SimpleNamespace(from_pretrained=processor_loader),
            Qwen3VLForConditionalGeneration=types.SimpleNamespace(from_pretrained=model_loader))
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            read = self.m.load_qwen_ocr(("cuda:0",))
            result = read([Image.new("RGB", (100, 100))])[0]
        self.assertEqual(processor_loader.call_args.kwargs,
                         {"revision": self.m.QWEN_OCR_REVISION, "min_pixels": 262144, "max_pixels": 1310720})
        self.assertEqual(model_loader.call_args.args, (self.m.QWEN_OCR_MODEL,))
        self.assertEqual(model_loader.call_args.kwargs["revision"], self.m.QWEN_OCR_REVISION)
        self.assertEqual(model_loader.call_args.kwargs["torch_dtype"], "float16")
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 2048)
        self.assertEqual(model.generate.call_args.kwargs["repetition_penalty"], 1.10)
        self.assertTrue(result["validated"])
        self.assertEqual(result["text"], "valid literal")
        processor.decode.side_effect = None
        processor.decode.return_value = 'stress literal'
        callback = Mock()
        stress = read([Image.new('RGB', (100, 100))], force_tokens=2048, on_answer=callback)[0]
        self.assertEqual(model.generate.call_args.kwargs['min_new_tokens'], 2048)
        self.assertEqual(model.generate.call_args.kwargs['max_new_tokens'], 2048)
        self.assertFalse(stress['validation_enabled'])
        callback.assert_called_once_with(0, stress)


class ProductionConfigTests(unittest.TestCase):
    def test_all_pipeline_builders_share_pinned_models_and_enabled_guards(self):
        for source in (fast.CONFIG, walk.CONFIG, single.config()):
            with self.subTest(source=source[:80]):
                cfg = config_values(source)
                self.assertEqual(cfg["ASR_MODEL"], "dropbox-dash/faster-whisper-large-v3-turbo")
                self.assertEqual(cfg["ASR_REVISION"], "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf")
                self.assertEqual(cfg["QWEN_OCR_REVISION"], "ebb281ec70b05090aa6165b016eac8ec08e71b17")
                self.assertEqual(cfg["QWEN_OCR_MODEL"], "Qwen/Qwen3-VL-4B-Instruct")
                self.assertEqual((cfg["QWEN_MIN_PIXELS"], cfg["QWEN_MAX_PIXELS"]), (262144, 1310720))
                for key in ("QWEN_CONTENT_CROP", "QWEN_VALIDATE", "QWEN_RETRY_INVALID",
                            "EXPORT_STATE_ONSETS", "MERGE_SAME_CONTENT", "FAILURE_GUARD"):
                    self.assertTrue(cfg[key], key)
                self.assertEqual(cfg["ASR_CORRECTION"], "glossary_code")
                self.assertEqual((cfg["ASR_MODE"], cfg["ASR_BATCH_SIZE"]), ("windowed", 1))
                self.assertEqual((cfg["ASR_DEVICE_INDEX"], cfg["EMBEDDING_DEVICE"]), (1, "cuda:1"))
        cfg = config_values(stages.SPECS[2]["config"])
        self.assertEqual((cfg["ASR_MODEL"], cfg["ASR_REVISION"]),
                         (config_values(fast.CONFIG)["ASR_MODEL"], config_values(fast.CONFIG)["ASR_REVISION"]))
        stage01 = config_values(stages.SPECS[0]["config"])
        self.assertTrue(stage01["EXPORT_STATE_ONSETS"])
        self.assertTrue(stage01["MERGE_SAME_CONTENT"])
        self.assertFalse(stage01["REVEAL_THRESHOLDS"])

    def test_detector_wiring_scopes_board_and_editor_to_the_audited_videos(self):
        cfg = config_values(fast.CONFIG)
        cfg.update(time=time, find_video_file=lambda video: Path(video + ".mp4"),
                   ad_scorer=None, content_scorer=None,
                   run_slide_detection=Mock(return_value=("stage01", {}, [])))
        detect = run_function("detect", cfg)
        for video in cfg["VIDEO_IDS"]:
            detect(video)
            options = cfg["run_slide_detection"].call_args.kwargs
            self.assertTrue(options["export_state_onsets"])
            self.assertTrue(options["merge_same_content"])
            self.assertEqual(options["scan"], "single_pass")
            if video == "hVU5Mw19V_c":
                self.assertFalse(options["merge_reveals"])
                self.assertEqual(options["board_options"]["minimum_pixels"], 24)
                self.assertEqual(options["board_options"]["roi"], options["keyframe_options"]["roi"])
            elif video == "y1CvB950VlU":
                self.assertTrue(options["reveal_thresholds"]["replacement_safe"])
                self.assertNotIn("board_options", options)
            else:
                for key in ("board_options", "keyframe_options", "reveal_thresholds"):
                    self.assertNotIn(key, options)
        self.assertEqual(set(cfg["STAGE01_VIDEO_OPTIONS"]), {"hVU5Mw19V_c", "y1CvB950VlU"})


class VisualProvenanceTests(unittest.TestCase):
    def test_failed_ocr_survives_cache_alignment_and_index_exclusion(self):
        visual = load_code("visual")
        alignment = load_code("alignment")
        indexing = load_code("indexing")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage01 = visual.begin_stage(root, 1)
            Image.new("RGB", (64, 64), "white").save(stage01 / "slide.jpg")
            interval = {"segment_id": "V:seg_00000", "sequence": 0, "start_sec": 0, "end_sec": 20,
                        "image_path": "slide.jpg", "image_sha256": visual.file_sha256(stage01 / "slide.jpg"),
                        "keyframe_time_sec": 12, "representative_content_onset_sec": 10,
                        "visual_evidence_windows": [{"start_sec": 10, "end_sec": 20, "onset_sec": 10,
                                                     "observed_sec": 12, "image_path": "slide.jpg"}]}
            source = {"sha256": "fixture", "duration_sec": 20, "has_audio": False}
            visual.write_json(stage01 / "intervals.json", [interval])
            visual.finish_stage(stage01, 1, "V", source, {})
            failure = visual.qwen_ocr_with_retry(lambda *args: {
                "text": "Looks fluent but is truncated", "token_ids": [10], "eos_token_id": 2})
            def answer(images, language, *, on_answer=None):
                if on_answer:
                    on_answer(0, failure)
                return [failure]
            reader = Mock(side_effect=answer)
            with patch.object(visual, "load_qwen_ocr", return_value=reader):
                stage02, manifest, records = visual.run_visual_extraction(
                    stage01, root, "V", ocr_engine="qwen", ocr_device="cpu")
                visual.run_visual_extraction(stage01, root, "V", ocr_engine="qwen", ocr_device="cpu")
            self.assertEqual(reader.call_count, 1, "same recipe reuses the diagnostic cache")
            self.assertEqual(manifest["ocr_failed_count"], 1)
            self.assertEqual(records[0]["ocr_status"], "failed")
            self.assertEqual(len(records[0]["ocr_generation"]["attempts"]), 2)
            self.assertIn("truncated", records[0]["raw_ocr_response"])
            stage03 = visual.begin_stage(root, 3)
            transcript = {"status": "no_audio", "words": [], "fallback_spans": [],
                          "windows": [], "slide_audio_clips": [], "language": "vi"}
            visual.write_json(stage03 / "transcript.json", transcript)
            visual.finish_stage(stage03, 3, "V", source, {},
                                parents=[visual.load_stage(stage01, 1, "V")])
            stage04, _, segments = alignment.run_alignment(
                stage01, stage02, stage03, root, "V", conservative_ocr=True,
                asr_correction="glossary_code", failure_guard=True)
            self.assertEqual(segments[0]["visual"]["ocr_text_clean"], "")
            self.assertIn("ocr_missing_eos", segments[0]["visual"]["ocr_quality"]["reasons"])
            retries = visual.read_json(stage04 / "gpu_retry_segments.json")
            self.assertEqual(retries[0]["status"], "failed")
            self.assertEqual(retries[0]['raw_response_path'], '02_visual/visual.json')
            import shutil
            for stage in ('01_slides', '02_visual', '03_audio'):
                shutil.copytree(root / stage, root / '_inputs' / stage)
            offline04, _, _ = alignment.run_alignment(
                root / '_inputs/01_slides', root / '_inputs/02_visual', root / '_inputs/03_audio',
                root, 'V', conservative_ocr=True, failure_guard=True)
            offline_retry = visual.read_json(offline04 / 'gpu_retry_segments.json')[0]
            self.assertEqual(offline_retry['raw_response_path'], '_inputs/02_visual/visual.json')
            self.assertTrue((root / offline_retry['raw_response_path']).is_file())
            with self.assertRaisesRegex(ValueError, "Không có bằng chứng"):
                indexing.make_evidence_chunks(segments, [], CharacterTokenizer(), video_id="V")

    def test_final_reveal_onset_changes_visual_citations_but_preserves_speech_ownership(self):
        alignment = load_code("alignment")
        indexing = load_code("indexing")
        intervals = [{"segment_id": "V:s", "start_sec": 34.7, "end_sec": 130,
                      "sequence": 0, "image_path": "slide.jpg", "keyframe_time_sec": 120,
                      "representative_content_onset_sec": 117.8, "visual_evidence_windows": []}]
        intervals.insert(0, {"segment_id": "V:opening", "start_sec": 0, "end_sec": 34.7,
                             "image_path": "opening.jpg", "keyframe_time_sec": 1})
        visuals = [{"segment_id": "V:opening", "content_kind": "speaker", "occurrence_id": "V:opening:visual",
                    "ocr_text": "", "title": "", "regions": []},
                   {"segment_id": "V:s", "content_kind": "slide", "occurrence_id": "V:s:visual",
                    "ocr_text": "newly revealed lines", "title": "", "regions": []}]
        transcript = {"status": "no_audio", "slide_audio_clips": [], "words": [{
            "word_id": "w", "text": "Earlier speech", "start_sec": 40, "end_sec": 41}]}
        segments = alignment.align_records(intervals, visuals, transcript, 130)
        chunks = indexing.make_evidence_chunks(segments, [], CharacterTokenizer(), video_id="V")
        ocr = next(c for c in chunks if c["source_type"] == "ocr")
        speech = next(c for c in chunks if c["source_type"] == "speech")
        self.assertEqual((ocr["start_sec"], ocr["end_sec"], ocr["timing_precision"]), (117.8, 130, "visual_state"))
        self.assertEqual((speech["start_sec"], speech["end_sec"]), (40, 41))
        self.assertEqual(segments[1]["start_sec"], 34.7)
