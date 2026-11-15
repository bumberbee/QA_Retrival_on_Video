"""Fast measurement-contract tests; no model loading or external services."""

import os

for _name in set(os.environ) | {"SUPABASE_DB_URL", "SUPABASE_URL", "SUPABASE_ANON_KEY",
                               "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"}:
    if _name.startswith(("SUPABASE_", "R2_")):
        os.environ[_name] = ""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from offline_common import (ModelAdapter, TokenizerAdapter, experiment_dir, load_backend,
                            production_config, write_json)
from eval_retrieval import (aggregate, context_trace, overlap, quality_lints, repetition_runs,
                            term_match, validate_questions)
from rebuild_offline import compare_indexes


class HarnessTests(unittest.TestCase):
    def test_half_open_gold_overlap(self):
        self.assertFalse(overlap(0, 10, [[10, 20]]))
        self.assertTrue(overlap(9.99, 10.01, [[10, 20]]))

    def test_fuzzy_terms_preserve_numbers_and_short_identifiers(self):
        self.assertTrue(term_match("xác suất", "xác suấtt"))
        self.assertTrue(term_match("chi phí cơ hội", "chi phi co hoi"))
        self.assertFalse(term_match("1,5", "4,5"))
        self.assertFalse(term_match("len", "LAN"))
        self.assertFalse(term_match("3", "13"))

    def test_long_loop_and_legitimate_repeated_values(self):
        self.assertEqual(repetition_runs("tổ chức " * 49)[0]["repeats"], 49)
        self.assertFalse(repetition_runs("0 " * 20))
        self.assertFalse(repetition_runs("Câu hỏi Câu hỏi"))

    def test_production_decoder_cleanup(self):
        tokenizer = TokenizerAdapter(types.SimpleNamespace(decode=lambda *a, **kw: "a , b . isn't !"))
        self.assertEqual(tokenizer.decode([1]), "a, b. isn't!")

    def test_adapter_does_not_double_prefix_passages(self):
        import numpy as np
        calls = []
        model = object.__new__(ModelAdapter)
        model.encoder = types.SimpleNamespace(encode_many=lambda texts, prefix: (
            calls.append((texts, prefix)) or np.ones((len(texts), 2))))
        model.encode(["passage: title\nbody", "passage: second"], batch_size=1)
        self.assertEqual(calls, [(["passage: title\nbody"], ""), (["passage: second"], "")])

    def test_isolated_backend_and_config(self):
        backend = load_backend()
        self.assertFalse(backend.storage.cloud())
        self.assertEqual(backend.storage.__name__, "_lecture_eval_backend.shared.storage")
        self.assertEqual(production_config()["revision"], backend.E5_REVISION)
        self.assertFalse(production_config()["dedupe_previous"])
        self.assertIsNone(production_config()["slide_speech"])

    def test_outputs_cannot_escape_experiments(self):
        with self.assertRaises(ValueError):
            experiment_dir("../../../../outside")

    def fixture(self, folder, tail=False):
        backend = load_backend()
        words = [{"word_id": f"w{i}", "text": "padding", "start_sec": i, "end_sec": i + 0.9}
                 for i in range(400 if tail else 2)]
        words.append({"word_id": "answer", "text": "Bayes", "start_sec": 500, "end_sec": 501})
        speech = " ".join(w["text"] for w in words)
        write_json(folder / "lecture/words.json", words)
        segment = {"segment_id": "v:s", "start_sec": 0, "end_sec": 600,
                   "speech": {"word_ids": [w["word_id"] for w in words]}, "visual": {"title_clean": ""}}
        chunk = {"chunk_id": "c", "segment_id": "v:s", "source_type": "speech", "text": "Bayes",
                 "start_sec": 500, "end_sec": 501, "encoded_tokens": 12}
        index = backend.LectureIndex(folder, [chunk], None, [segment], {"v:s": segment})
        found = {"slides": [{"segment_id": "v:s", "start_sec": 0, "end_sec": 600,
                             "title": "Bayes", "speech_text": speech, "ocr_text": "", "figures": []}]}
        question = {"gold_spans": [[500, 501]], "key_terms": ["Bayes"], "evidence_source": "speech"}
        return backend, index, found, question

    def test_context_excludes_gold_words_after_cutoff_even_if_title_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend, index, found, question = self.fixture(Path(tmp), tail=True)
            result = context_trace(found, index, question, backend.build_context(found))
            self.assertFalse(result["recalled"])
            self.assertFalse(result["slides"][0]["gold_overlap_word_ids"])

    def test_context_requires_both_retained_text_and_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend, index, found, question = self.fixture(Path(tmp))
            self.assertTrue(context_trace(found, index, question, backend.build_context(found))["recalled"])
            question["gold_spans"] = [[100, 101]]
            self.assertFalse(context_trace(found, index, question, backend.build_context(found))["recalled"])

    def test_ranked_trace_does_not_credit_omitted_words_with_shared_unit_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend, index, found, question = self.fixture(Path(tmp))
            found["context_evidence"] = [{"chunk_id": "partial", "segment_id": "v:s", "source_type": "speech",
                                           "text": "padding padding", "start_sec": 0, "end_sec": 600,
                                           "selection_reason": "matched"}]
            trace = context_trace(found, index, question, "Bayes in the title; padding padding")
            self.assertFalse(trace["recalled"])
            self.assertNotIn("answer", trace["slides"][0]["included_word_ids"])

    def test_neighbor_trace_keeps_its_own_time_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend, index, found, question = self.fixture(Path(tmp), tail=True)
            found["context_evidence"] = [{"chunk_id": "neighbor", "segment_id": "v:next", "source_type": "speech",
                                           "text": "Bayes", "start_sec": 500, "end_sec": 501,
                                           "selection_reason": "next"}]
            trace = context_trace(found, index, question, "[08:20] Bayes")
            self.assertTrue(trace["recalled"])
            self.assertEqual(trace["slides"][0]["segment_id"], "v:next")
            question["gold_spans"] = [[100, 101]]
            self.assertFalse(context_trace(found, index, question, "[08:20] Bayes")["recalled"])

    def test_lints_count_excess_copies_and_actual_word_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend, index, found, question = self.fixture(Path(tmp))
            a = {**index.chunks[0], "source_type": "ocr", "text": "tuyển sinh " * 10}
            index.chunks = [a, {**a, "chunk_id": "d"}]
            write_json(Path(tmp) / "lecture/words.json", [
                {"text_asr": "Trong", "text": "Chọn", "start_sec": 40.92},
                {"text_asr": "kinh", "text": "kinh", "start_sec": 41}])
            write_json(Path(tmp) / "lecture/asr_corrections.json", [
                {"heard": "trong kinh", "fixed": "chọn kinh", "at_sec": 40.92}])
            lint = quality_lints(index)
            self.assertEqual(lint["exact_duplicate_ocr_excess"], 1)
            self.assertEqual(lint["promotional_chunks"], 2)
            self.assertEqual(lint["changed_asr_words"], 1)
            self.assertEqual(lint["known_bad_asr_corrections"], 1)

    def test_parity_detects_text_change_and_unmatched_ids(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            folders = [Path(tmp) / name for name in ("a", "b")]
            for folder in folders:
                folder.mkdir()
                (folder / "chunks.jsonl").write_text(json.dumps({"chunk_id": "c", "text": "one",
                    "embedding_text": "passage: one", "encoded_tokens": 4}) + "\n", encoding="utf-8")
                np.save(folder / "embeddings.npy", [[1.0, 0.0]])
            self.assertTrue(compare_indexes(*folders)["pass"])
            text = (folders[0] / "chunks.jsonl").read_text(encoding="utf-8").replace("one", "two")
            (folders[0] / "chunks.jsonl").write_text(text, encoding="utf-8")
            self.assertFalse(compare_indexes(*folders)["pass"])
            text = text.replace('"c"', '"d"')
            (folders[0] / "chunks.jsonl").write_text(text, encoding="utf-8")
            self.assertEqual(compare_indexes(*folders)["missing_ids"], ["c"])

    def test_unanswerable_excluded_from_metric_denominator(self):
        result = aggregate([{"id": "a", "type": "speech", "hit1": 1, "hit3": 1, "hit5": 1,
                             "mrr": 1, "context_recall": 0},
                            {"id": "b", "type": "unanswerable", "best_semantic": 0.9, "top_rrf": 0.02}])
        self.assertEqual(result["answerable"], 1)
        self.assertEqual(result["hit1"], 1)
        self.assertEqual(result["unanswerable_top_scores"][0]["semantic"], 0.9)

    def test_dev_question_contract(self):
        path = Path(__file__).resolve().parents[1] / "eval/questions_dev.json"
        validate_questions(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
