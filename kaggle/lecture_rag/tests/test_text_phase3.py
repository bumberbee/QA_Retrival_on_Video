"""CPU text-quality regressions; no model downloads or external services."""

import json
from contextlib import closing
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from offline_common import production_config


def load_text_code():
    module = types.ModuleType("phase3_text")
    for name in ("common", "textclean", "adfilter", "alignment", "indexing"):
        path = ROOT / "notebook_code" / (name + ".py")
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


class TextPhase3Tests(unittest.TestCase):
    def setUp(self):
        self.m = load_text_code()

    def words(self, text):
        return [{"text": word, "word_id": f"w{i}", "start_sec": i * 0.2, "end_sec": i * 0.2 + 0.1}
                for i, word in enumerate(text.split())]

    def test_repeated_headings_code_formula_and_numbers_survive(self):
        raw = 'MỤC TIÊU\na2="Python"\nfor i in a2:\n    print(i)\ni=1\n2\n+\nTrang 25\nHotline: 123456'
        repeated = self.m.repeated_lines([raw] * 5)
        clean = self.m.clean_ocr_text(raw, repeated, conservative=True)
        self.assertEqual(clean, 'MỤC TIÊU\na2="Python"\nfor i in a2:\n    print(i)\ni=1\n2\n+')

    def test_qwen_export_layout_and_figure_separation(self):
        raw = 'for i in a2:\n    print(i)\n[Figure] A diagram.'
        visual = {"ocr_text": "for i in a2:\nprint(i)", "raw_ocr_response": raw,
                  "ocr_blocks": [{"source": "qwen3-vl"}]}
        self.assertEqual(self.m.qwen_ocr_text(visual), 'for i in a2:\n    print(i)')
        module = types.ModuleType("visual_phase3")
        for name in ("common", "visual"):
            path = ROOT / "notebook_code" / (name + ".py")
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
        blocks, regions = module.parse_qwen_slide(raw)
        self.assertEqual(blocks[1]["text"], "    print(i)")
        self.assertEqual(regions[0]["description"], "A diagram.")

    def test_subscript_normalization_preserves_powers_code_and_raw_input(self):
        raw = 'Cl₂\nNaHCO₃ + CO₂\nx² + i₁\n    print("H₂O")\na2="H₂O"'
        self.assertEqual(self.m.clean_ocr_text(raw, conservative=True), raw)
        self.assertEqual(self.m.clean_ocr_text(raw, conservative=True, canonical_subscripts=True),
                         'Cl2\nNaHCO3 + CO2\nx² + i1\n    print("H₂O")\na2="H₂O"')
        self.assertIn('Cl₂', raw)

    def test_short_titles_heading_priority_and_body_rejection(self):
        for text, expected in [('Bayes', 'Bayes'), ('MỤC TIÊU\nbody', 'MỤC TIÊU'),
                               ('Bình đẳng chủ quyền giữa các quốc gia...\nLIÊN HIỆP QUỐC', 'LIÊN HIỆP QUỐC'),
                               ('Clear Update', ''), ('Q', ''), ('L10', ''), ('a2="Python"\nprint(a2)', ''),
                               ('Đây là một lời giải dài cho câu hỏi.', '')]:
            with self.subTest(text=text):
                self.assertEqual(self.m.heading_title(text), expected)

    def test_atomic_overlap_repairs_complete_probability_term(self):
        words = self.words('chúng ta tính sắc xuất để A diễn ra')
        vocab = self.m.ocr_vocabulary(['tính xác suất\nxác suất'] * 2)
        fixed, _ = self.m.correct_words(words, vocab)
        self.assertEqual(self.m.words_to_text(fixed), 'chúng ta tính xác suất để A diễn ra')
        self.assertEqual(self.m.words_to_text(words), 'chúng ta tính sắc xuất để A diễn ra')

    def test_glossary_keeps_generic_phrases_and_raw_words(self):
        words = self.words('trong kinh tế tính sắc xuất, chân không')
        segments = [{"segment_id": "v:s", "visual": {"ocr_text_clean": "tính xác suất"},
                     "speech": {"word_ids": [w["word_id"] for w in words]}}]
        fixed, events = self.m.correct_glossary_words(words, segments)
        self.assertEqual(self.m.words_to_text(fixed), 'trong kinh tế tính xác suất, chân không')
        self.assertEqual([w['text_asr'] for w in fixed], [w['text'] for w in words])
        self.assertEqual(events[0]['support_segment_id'], 'v:s')
        self.assertEqual(len(events), 1)

    def test_glossary_requires_local_support_and_punctuation_boundary(self):
        words = self.words('xác xuất sắc, xuất')
        segments = [{"segment_id": "v:s", "visual": {"ocr_text_clean": "chọn kinh tế"},
                     "speech": {"word_ids": [w["word_id"] for w in words]}}]
        fixed, events = self.m.correct_glossary_words(words, segments)
        self.assertFalse(events)
        self.assertEqual(self.m.words_to_text(fixed), self.m.words_to_text(words))
        segments[0]['visual']['ocr_text_clean'] = 'xác suất'
        _, events = self.m.correct_glossary_words(words, segments)
        self.assertEqual(len(events), 1)  # "sắc, xuất" cannot cross the comma.

    def test_reviewed_technical_aliases_do_not_enable_generic_corrections(self):
        words = self.words('trí phí cơ hội, nhiệt độ xôi, xuất điện động, trong kinh tế, học chương trình')
        segments = [{"segment_id": "v:s", "visual": {"ocr_text_clean": 'chi phí\nnhiệt độ sôi\nsuất điện động\nchọn kinh\nhọc trường'},
                     "speech": {"word_ids": [w['word_id'] for w in words]}}]
        fixed, events = self.m.correct_glossary_words(words, segments)
        self.assertEqual(self.m.words_to_text(fixed),
                         'chi phí cơ hội, nhiệt độ sôi, suất điện động, trong kinh tế, học chương trình')
        self.assertEqual(len(events), 3)

    def test_lecture_glossary_requires_two_printed_support_segments(self):
        words = self.words('tính sắc xuất trong kinh tế')
        segments = [{'segment_id': 'v:a', 'visual': {'ocr_text_clean': 'xác suất'}, 'speech': {'word_ids': []}},
                    {'segment_id': 'v:b', 'visual': {'ocr_text_clean': 'xác suất'}, 'speech': {'word_ids': []}}]
        fixed, events = self.m.correct_glossary_words(words, segments, lecture_support=True)
        self.assertEqual(self.m.words_to_text(fixed), 'tính xác suất trong kinh tế')
        self.assertEqual(events[0]['printed_support_segment_ids'], ['v:a', 'v:b'])
        _, events = self.m.correct_glossary_words(words, segments[:1], lecture_support=True)
        self.assertFalse(events)

    def test_code_feedback_requires_literal_local_print_message(self):
        words = self.words('xai rồi trong kinh tế')
        segment = {'segment_id': 'v:a', 'visual': {'ocr_text_clean': 'print("sai roi, bye bye")'},
                   'speech': {'word_ids': [w['word_id'] for w in words]}}
        fixed, events = self.m.correct_glossary_words(words, [segment], lecture_support=True, code_feedback=True)
        self.assertEqual(self.m.words_to_text(fixed), 'sai rồi trong kinh tế')
        self.assertEqual(events[0]['policy'], 'printed_code_feedback')
        for text in ('sai rồi', 'Đây là một ví dụ sai rồi', 'print("đúng rồi")'):
            segment['visual']['ocr_text_clean'] = text
            _, events = self.m.correct_glossary_words(words, [segment], lecture_support=True, code_feedback=True)
            self.assertFalse(events)
        segment['visual']['ocr_text_clean'] = 'print("sai roi")'
        segment['speech']['word_ids'] = []
        _, events = self.m.correct_glossary_words(words, [segment, {**segment, 'segment_id': 'v:b'}],
                                                  lecture_support=True, code_feedback=True)
        self.assertFalse(events)

    def test_ocr_loops_and_placeholders_cannot_be_evidence(self):
        for raw in ['Câu hỏi\n' * 324, 'Không có chữ trên slide bài giảng.',
                    'Sau đó, cho dòng điện chạy trong dây dẫn, cân chỉ một giá trị xác định. ' * 40]:
            text, quality = self.m.guard_ocr_text(raw)
            self.assertEqual(text, '')
            self.assertTrue(quality['evidence_excluded'])
            self.assertEqual(quality['status'], 'failed')
        numeric = '0 0 0 0 0 0 0 0 0'
        self.assertEqual(self.m.guard_ocr_text(numeric)[0], numeric)
        self.assertEqual(self.m.guard_ocr_text('print(i)\n' * 8)[0], ('print(i)\n' * 8).strip('\n'))

    def test_partial_ocr_loop_keeps_other_evidence_with_retry_reason(self):
        raw = 'Một nội dung bài giảng đầy đủ ' * 1 + 'Câu hỏi ' * 8 + '\n' + 'Nội dung khác ' * 20
        # The repeated tail is itself a failed loop; a unique long prefix provides healthy evidence.
        raw = ' '.join(f'nội_dung_{i}' for i in range(100)) + '\n' + 'Câu hỏi ' * 8
        text, quality = self.m.guard_ocr_text(raw)
        self.assertIn('nội_dung_99', text)
        self.assertNotIn('Câu hỏi', text)
        self.assertEqual(quality['status'], 'partial')

    def test_asr_loop_flags_preserve_raw_ownership_and_teaching_words(self):
        words = self.words('Bài học. ' + 'tổ chức ' * 49 + 'Liên Hiệp Quốc.')
        original = [dict(w) for w in words]
        loops = self.m.flag_asr_loops(words)
        spans = self.m.evidence_speech_spans(words, 'v:s')
        self.assertEqual(loops[0]['repeats'], 49)
        self.assertEqual(len(loops[0]['word_ids']), 98)
        self.assertEqual([w['text'] for w in words], [w['text'] for w in original])
        self.assertEqual([w['start_sec'] for w in words], [w['start_sec'] for w in original])
        self.assertEqual(' '.join(s['text'] for s in spans), 'Bài học. Liên Hiệp Quốc.')
        self.assertEqual(len({s['span_id'] for s in spans}), len(spans))

    def test_mixed_ad_interval_keeps_teaching_on_both_sides(self):
        words = self.words('Cao trào kháng Nhật. Trường Đại học Quốc tế Sài Gòn SIU tuyển sinh '
                           'học bổng học phí. SIU Tâm hồn Việt Nam giáo dục thế giới. '
                           'So sánh ba cao trào cách mạng.')
        spans = self.m.promotional_word_spans(words)
        self.assertEqual(len(spans), 1)
        self.assertTrue(spans[0]['text'].startswith('Trường'))
        excluded = set(spans[0]['word_ids'])
        kept = self.m.words_to_text([w for w in words if w['word_id'] not in excluded])
        self.assertEqual(kept, 'Cao trào kháng Nhật. So sánh ba cao trào cách mạng.')

    def test_ad_filter_does_not_exclude_school_or_qr_teaching_examples(self):
        for text in ['Trường quốc tế Á Châu có 40 học sinh. Tính xác suất chọn một học sinh.',
                     'SIU là ví dụ về một trường. Tính chi phí cơ hội và học phí.',
                     'Quét mã QR xem video thí nghiệm phản ứng với nước.',
                     'Tuyển sinh là ví dụ trong bài học về kinh tế.']:
            self.assertFalse(self.m.promotional_word_spans(self.words(text)))
            self.assertFalse(self.m.promotional_word_spans(self.words(text), asian_spots=True))

    def test_asian_school_spot_requires_bounded_slogan_without_recruitment_words(self):
        words = self.words('Bài học trước. Trường Quốc tế Á Châu mang đến nền giáo dục hiện đại. '
                           'Đã đến lúc vươn xa hơn những gì đã học. Bài học sau.')
        self.assertFalse(self.m.promotional_word_spans(words))
        spans = self.m.promotional_word_spans(words, asian_spots=True)
        ids = {i for span in spans for i in span['word_ids']}
        self.assertEqual(self.m.words_to_text([w for w in words if w['word_id'] not in ids]),
                         'Bài học trước. Bài học sau.')
        distant = [dict(w) for w in words]
        distant[-4]['end_sec'] += 300
        self.assertFalse(self.m.promotional_word_spans(distant, asian_spots=True))

    def test_subscription_call_keeps_adjacent_code_explanation(self):
        words = self.words('Dùng len tính độ dài. Hãy đăng ký kênh. Dùng for duyệt chuỗi.')
        spans = self.m.promotional_word_spans(words)
        ids = {i for s in spans for i in s['word_ids']}
        self.assertEqual(self.m.words_to_text([w for w in words if w['word_id'] not in ids]),
                         'Dùng len tính độ dài. Dùng for duyệt chuỗi.')

    def test_asian_signature_uses_observed_opening_and_closing(self):
        words = self.words('Bài học trước. Trường Quốc tế Á Châu tự hào là ngôi trường mang đến '
                           'nền giáo dục tiên tiến hiện đại. Viết ngân tôn chuyện tương lai của chính mình. '
                           'Bài học sau.')
        spans = self.m.promotional_word_spans(words, asian_spots='signature')
        ids = {i for span in spans for i in span['word_ids']}
        self.assertEqual(self.m.words_to_text([w for w in words if w['word_id'] not in ids]),
                         'Bài học trước. Bài học sau.')
        self.assertFalse(self.m.promotional_word_spans(
            self.words('Trường Quốc tế Á Châu là ví dụ về một trường. Tương lai của chính mình.'),
            asian_spots='signature'))

    def test_failed_visual_is_not_indexed_or_prepended_as_title(self):
        tokenizer = types.SimpleNamespace(encode=lambda t, **kw: list(t.encode()),
                                          decode=lambda t, **kw: bytes(t).decode())
        segment = {"segment_id": "v:s", "sequence": 0, "start_sec": 0, "end_sec": 10,
                   "visual": {"title_clean": "Câu hỏi", "ocr_text_clean": "Câu hỏi",
                              "ocr_quality": {"evidence_excluded": True}, "occurrence_id": "o",
                              "image_path": "frame.jpg", "keyframe_time_sec": 1,
                              "regions": [{"text": "invented figure"}]},
                   "speech": {"spans": [{"span_id": "s", "text": "Healthy speech.", "start_sec": 2,
                                           "end_sec": 3}], "fallback_span_ids": []}}
        chunks = self.m.make_evidence_chunks([segment], [], tokenizer, include_generated_descriptions=True)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['source_type'], 'speech')
        self.assertEqual(chunks[0]['title'], '')

    def test_layout_chunks_keep_indentation_and_budget(self):
        tokenizer = types.SimpleNamespace(encode=lambda t, **kw: list(t),
                                          decode=lambda t, **kw: ''.join(t))
        raw = '\n'.join(f'    print({i})' for i in range(12))
        chunks = self.m.split_layout_tokens(raw, tokenizer, 45, 15)
        self.assertTrue(all(len(c) <= 45 for c in chunks))
        for line in raw.splitlines():
            self.assertTrue(any(line in c.splitlines() for c in chunks))

    def test_generated_figure_noise_and_semantic_only_lexical_policy(self):
        import sqlite3
        for text in ('Không có hình minh họa.', 'No figure on this slide.', '...', 'figure ' * 30):
            self.assertFalse(self.m.usable_figure_description(text))
        self.assertTrue(self.m.usable_figure_description('A Venn diagram with two intersecting circles.'))
        chunks = [{'chunk_id': 'speech', 'title': '', 'text': 'Venn diagram'},
                  {'chunk_id': 'figure', 'title': '', 'text': 'Venn diagram',
                   'verification': 'model_generated_unverified', 'retrieval_policy': 'semantic_only'}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'lexical.sqlite3'
            self.m.build_lexical_index(path, chunks)
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute('SELECT chunk_id FROM evidence').fetchall(), [('speech',)])

    def test_harness_config_matches_generator_defaults(self):
        captured = json.loads((ROOT / 'eval/production_config.json').read_text(encoding='utf-8'))
        self.assertEqual(production_config(), captured)

    def test_asset_sharing_preserves_bytes_and_differing_experiment_text(self):
        import hashlib
        import os
        import dedupe_audio
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, target = [root / 'runs/exp' / name / 'v' for name in ('ref', 'trial')]
            for directory, text in ((reference, b'raw ASR'), (target, b'new ASR')):
                stage = directory / '_inputs/01_slides'
                stage.mkdir(parents=True)
                (stage / 'frame.jpg').write_bytes(b'identical image bytes')
                (stage / 'text.json').write_bytes(text)
                (stage / 'manifest.json').write_text(json.dumps({'files': {
                    name: hashlib.sha256((stage / name).read_bytes()).hexdigest()
                    for name in ('frame.jpg', 'text.json')}}))
            with patch.object(dedupe_audio, 'ROOT', root):
                linked = dedupe_audio.dedupe_video_assets(target, reference)
                self.assertEqual(len(linked), 1)
                self.assertTrue(os.path.samefile(target / '_inputs/01_slides/frame.jpg',
                                                 reference / '_inputs/01_slides/frame.jpg'))
                self.assertEqual((target / '_inputs/01_slides/text.json').read_bytes(), b'new ASR')
                with self.assertRaises(ValueError):
                    dedupe_audio.dedupe_video_assets(root / 'outside', reference)

    def test_asset_sharing_rejects_corrupt_same_size_target(self):
        import hashlib
        import dedupe_audio
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, target = [root / 'runs/exp' / name / 'v' for name in ('ref', 'trial')]
            for directory, data in ((reference, b'good'), (target, b'evil')):
                stage = directory / '_inputs/01_slides'
                stage.mkdir(parents=True)
                (stage / 'frame.jpg').write_bytes(data)
                (stage / 'manifest.json').write_text(json.dumps({'files': {
                    'frame.jpg': hashlib.sha256(b'good').hexdigest()}}))
            with patch.object(dedupe_audio, 'ROOT', root), self.assertRaises(ValueError):
                dedupe_audio.dedupe_video_assets(target, reference)
            self.assertEqual((target / '_inputs/01_slides/frame.jpg').read_bytes(), b'evil')


if __name__ == '__main__':
    unittest.main()
