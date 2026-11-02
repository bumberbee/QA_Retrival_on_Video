"""Offline controls: strict OCR frame matching and timing without speech retiming."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from extraction_ablation import legacy_frame_plan, match_frame
from rebuild_offline import copy_input_file
from offline_common import ModelAdapter
from test_pipeline import CharacterTokenizer, load_code


def test_ocr_matching_requires_same_frame_not_group_overlap():
    target = {"image_sha256": "a", "keyframe_time_sec": 12.0}
    wrong = {"image_sha256": "b", "keyframe_time_sec": 11.0}
    assert match_frame(target, [wrong]) == (None, "unmatched")
    same_time = {"image_sha256": "b", "keyframe_time_sec": 12.0001}
    assert match_frame(target, [same_time]) == (same_time, "time")
    same_hash = {"image_sha256": "a", "keyframe_time_sec": 10.0}
    assert match_frame(target, [same_time, same_hash]) == (same_hash, "hash")


def test_legacy_frames_do_not_cross_replacement_or_final_reveal_boundaries():
    old = {"segment_id": "V:old", "start_sec": 0, "end_sec": 20, "keyframe_time_sec": 5}
    same = {"segment_id": "V:same", "start_sec": 0, "end_sec": 20,
            "reveal_states": [{"start_sec": 0, "end_sec": 20}]}
    replacement = {"segment_id": "V:replacement", "start_sec": 0, "end_sec": 10}
    late = {"segment_id": "V:late", "start_sec": 0, "end_sec": 20,
            "reveal_states": [{"start_sec": 10, "end_sec": 20}]}
    assert legacy_frame_plan([same, replacement, late], [old]) == {"V:same": old}


def test_input_metadata_is_copied_before_manifest_resigning():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, target = root / "source.json", root / "target.json"
        source.write_text('{"artifact_id":"original"}', encoding="utf-8")
        copy_input_file(source, target)
        target.write_text('{"artifact_id":"new"}', encoding="utf-8")
        assert "original" in source.read_text(encoding="utf-8")


def test_timing_control_changes_visuals_without_changing_speech():
    indexing = load_code("indexing")
    segment = {"segment_id": "V:s", "sequence": 0, "start_sec": 34.7, "end_sec": 130,
               "visual": {"image_path": "frame.jpg", "keyframe_time_sec": 120,
                          "occurrence_id": "V:s:visual", "representative_content_onset_sec": 117.8,
                          "ocr_text": "revealed formula", "title": "", "regions": []},
               "speech": {"spans": [{"span_id": "w", "word_ids": ["w"], "text": "earlier speech",
                                      "start_sec": 40, "end_sec": 41}], "fallback_span_ids": []}}
    state = indexing.make_evidence_chunks([segment], [], CharacterTokenizer(), video_id="V")
    interval = indexing.make_evidence_chunks([segment], [], CharacterTokenizer(), video_id="V",
                                             visual_state_timing=False)
    assert next(c for c in state if c["source_type"] == "speech") == next(
        c for c in interval if c["source_type"] == "speech")
    a = next(c for c in state if c["source_type"] == "ocr")
    b = next(c for c in interval if c["source_type"] == "ocr")
    assert (a["start_sec"], a["timing_precision"]) == (117.8, "visual_state")
    assert (b["start_sec"], b["timing_precision"]) == (34.7, "visual_interval")
    assert a["embedding_text"] == b["embedding_text"]


def test_vector_cache_preserves_distinct_occurrences_of_identical_text():
    import hashlib
    import json
    import numpy as np
    from tokenizers import Tokenizer, models
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    class Encoder:
        def __init__(self):
            self.tokenizer = tokenizer
        def encode_many(self, *args, **kwargs):
            raise AssertionError("Unchanged occurrences must reuse their own vectors")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        chunks = [{"chunk_id": "V:a", "embedding_text": "passage: duplicate"},
                  {"chunk_id": "V:b", "embedding_text": "passage: duplicate"}]
        (root / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
        vectors = np.array([[1, 0], [1, 1e-7]], dtype=np.float32)
        np.save(root / "embeddings.npy", vectors)
        manifest = {"files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ("chunks.jsonl", "embeddings.npy")}}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        actual = ModelAdapter(Encoder(), [root]).encode_chunks(list(reversed(chunks)))
        assert np.array_equal(actual, vectors[::-1])
