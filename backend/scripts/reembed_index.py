"""Re-embed the app's stage-05 indexes with another encoder (default bge-m3), on CPU with ONNX.

    cd backend
    .venv/Scripts/python scripts/reembed_index.py [--model BAAI/bge-m3] [--videos L25_V011 L25_V037 ...]
    .venv/Scripts/python scripts/reembed_index.py --restore      # back to the E5 vectors

Chunks and the BM25 index stay as they are; only embeddings.npy and the manifest's embedding
parameters change. The first run keeps the E5 vectors as embeddings_e5.npy (and the manifest
as manifest_e5.json) so --restore can put them back. Restart the backend afterwards.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.shared import storage  # noqa: E402
from app.features.search import retrieval  # noqa: E402


def passages(folder: Path, prefix: str) -> list[str]:
    texts = []
    for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            text = json.loads(line)["embedding_text"]
            texts.append(prefix + text.removeprefix("passage: "))
    return texts


def reembed(folder: Path, encoder, model: str, batch_size: int) -> float:
    import numpy as np

    if not (folder / "embeddings_e5.npy").exists():
        shutil.copy(folder / "embeddings.npy", folder / "embeddings_e5.npy")
        shutil.copy(folder / "manifest.json", folder / "manifest_e5.json")
    texts = passages(folder, "")
    started = time.perf_counter()
    order = sorted(range(len(texts)), key=lambda n: len(texts[n]))   # similar lengths per batch
    vectors = np.zeros((len(texts), 0), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        batch = encoder.encode_many([texts[n] for n in rows], prefix="").astype(np.float32)
        if vectors.shape[1] == 0:
            vectors = np.zeros((len(texts), batch.shape[1]), dtype=np.float32)
        vectors[rows] = batch
    np.save(folder / "embeddings.npy", vectors, allow_pickle=False)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["parameters"].update({
        "embedding_model": model, "revision": retrieval.ENCODERS[model]["revision"],
        "resolved_revision": retrieval.ENCODERS[model]["revision"], "query_prefix": "",
        "passage_prefix": "", "dimensions": int(vectors.shape[1]),
        "reembedded_by": "backend/scripts/reembed_index.py (ONNX, CPU)",
    })
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-m3", choices=sorted(retrieval.ENCODERS))
    parser.add_argument("--videos", nargs="*", help="original file names without .mp4 (default: all)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--restore", action="store_true", help="put the kept E5 vectors back")
    args = parser.parse_args()
    root = storage.DATA_DIR / retrieval.INDEX_DIR_NAME
    if args.videos:
        ids = [storage.find_video_id_by_original_filename(f"{name}.mp4") for name in args.videos]
        folders = [root / video_id for video_id in ids if video_id]
    else:
        folders = sorted(p for p in root.iterdir() if (p / "chunks.jsonl").exists())
    if args.restore:
        for folder in folders:
            if (folder / "embeddings_e5.npy").exists():
                shutil.copy(folder / "embeddings_e5.npy", folder / "embeddings.npy")
                shutil.copy(folder / "manifest_e5.json", folder / "manifest.json")
        print("Đã trả lại vector E5 cho", len(folders), "chỉ mục")
        return
    encoder = retrieval.OnnxEncoder(args.model)
    for k, folder in enumerate(folders, 1):
        seconds = reembed(folder, encoder, args.model, args.batch_size)
        print(f"{k}/{len(folders)} {folder.name}: {seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
