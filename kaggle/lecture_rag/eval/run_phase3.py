"""Run frozen CPU text ablations through the real rebuild/evaluation entry points."""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

from offline_common import ROOT, load_backend, read_json, write_json
import rebuild_offline
import eval_retrieval


class CachedEncoder:
    """Execution-cost reuse of exact (prefix, text) vectors from the pinned CPU encoder."""

    def __init__(self, encoder, warm_indexes=()):
        import json
        import numpy as np
        self.encoder = encoder
        self.tokenizer = encoder.tokenizer
        self.cache = {}
        self.warm_indexes = list(warm_indexes)
        backend = load_backend()
        for name in warm_indexes:
            # Cache reads may follow a workspace junction to a read-only archive. Writes
            # still use experiment_dir/dedupe_video_audio's resolved workspace checks.
            root = (ROOT / "runs/exp" / name).absolute()
            if not root.is_relative_to((ROOT / "runs/exp").absolute()):
                raise ValueError("Warm indexes must stay inside runs/exp.")
            try:
                report = read_json(root / "rebuild_report.json")
            except FileNotFoundError:
                print("WARM_SKIP", name, "rebuild report unavailable", flush=True)
                continue
            if (not report["execution"].startswith("CPU ONNX")
                    or (report["config"]["embedding_model"], report["config"]["revision"])
                    != (backend.E5_REPO, backend.E5_REVISION)):
                raise ValueError("Warm vectors must come from the same pinned CPU encoder.")
            for video in report["videos"]:
                directory = root / video / "05_index"
                chunks = [json.loads(line) for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
                ids = read_json(directory / "chunk_ids.json")
                vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
                if (ids != [c["chunk_id"] for c in chunks] or len(vectors) != len(chunks)
                        or not np.isfinite(vectors).all()
                        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)):
                    raise ValueError("Invalid warm index vectors/order.")
                self.cache.update({("", chunk["embedding_text"]): vector.copy()
                                   for chunk, vector in zip(chunks, vectors)})

    def encode_many(self, texts, prefix=None):
        import numpy as np
        missing = list(dict.fromkeys(text for text in texts if (prefix, text) not in self.cache))
        for start in range(0, len(missing), 32):
            batch = missing[start:start + 32]
            vectors = self.encoder.encode_many(batch, prefix=prefix)
            self.cache.update({(prefix, text): vector.copy() for text, vector in zip(batch, vectors)})
        return np.stack([self.cache[(prefix, text)] for text in texts])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Fresh experiment prefix; rebuild refuses overwrites")
    parser.add_argument("--variants", nargs="+", default=["reference", "ocr", "titles", "asr-off", "asr-glossary", "failures", "ads", "combined"])
    parser.add_argument("--export-root", type=Path, default=Path("D:/thesis-data/run-20261020/export"))
    parser.add_argument("--warm-index", nargs="*", default=[], help="Completed pinned CPU rebuilds for exact vector reuse")
    parser.add_argument("--audio-reference", type=Path, help="Share checksum-identical generated WAVs with a completed experiment")
    args = parser.parse_args()
    backend = load_backend()
    encoder = CachedEncoder(backend.OnnxEncoder(), args.warm_index)
    for variant in args.variants:
        config = ROOT / "eval/phase3_configs" / (variant + ".json")
        if not config.is_file():
            parser.error(f"Unknown variant: {variant}")
        name = args.prefix + "-" + variant
        print("REBUILD", name, flush=True)
        rebuild_args = ["rebuild_offline.py", "--export-root", str(args.export_root), "--name", name, "--config", str(config)]
        if args.audio_reference:
            rebuild_args += ["--audio-reference", str(args.audio_reference)]
        with patch.object(backend, "OnnxEncoder", return_value=encoder), patch.object(sys, "argv", rebuild_args):
            rebuild_offline.main()
        with patch.object(backend, "OnnxEncoder", return_value=encoder), patch.object(sys, "argv", [
                "eval_retrieval.py", "--index-root", str(ROOT / "runs/exp" / name), "--name", name + "-eval"]):
            eval_retrieval.main()
        report = read_json(ROOT / "runs/exp" / (name + "-eval") / "retrieval.json")
        write_json(ROOT / "runs/exp" / name / "encoder_reuse.json", {
            "policy": "exact_prefix_and_text_CPU_ONNX_cache", "unique_vectors_so_far": len(encoder.cache),
            "warm_cpu_rebuilds": encoder.warm_indexes})
        print("RESULT", name, {k: report["overall"][k] for k in ("hit1", "hit3", "mrr", "context_recall")}, flush=True)


if __name__ == "__main__":
    main()
