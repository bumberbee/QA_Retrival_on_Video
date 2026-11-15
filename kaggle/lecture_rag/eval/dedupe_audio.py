"""Lossless hard-link sharing of checksum-identical CPU experiment assets."""

import argparse
import hashlib
import os
import uuid
from pathlib import Path

from offline_common import ROOT, read_json, write_json


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def dedupe_video_audio(target, reference):
    root = (ROOT / "runs/exp").resolve()
    target, reference = Path(target).resolve(), Path(reference).resolve()
    if not target.is_relative_to(root) or not reference.is_relative_to(root):
        raise ValueError("Audio sharing must stay inside runs/exp.")
    linked = []
    for stage in ("_inputs/03_audio", "04_bundle", "05_index"):
        directory = target / stage
        manifest = read_json(directory / "manifest.json")
        for name, expected in manifest["files"].items():
            if not name.endswith(".wav"):
                continue
            source = reference / ("_inputs/03_audio" if stage == "_inputs/03_audio" else "04_bundle") / name.removeprefix("lecture/")
            dest = directory / name
            source, dest = source.resolve(), dest.resolve()
            if not source.is_relative_to(reference) or not dest.is_relative_to(target):
                raise ValueError("Audio asset escaped its video directory.")
            if source == dest or not source.is_file() or os.path.samefile(source, dest):
                continue
            if sha256(source) != expected or sha256(dest) != expected:
                raise ValueError(f"Refusing to replace non-identical WAV: {dest}")
            temporary = dest.with_name(dest.name + ".link-" + uuid.uuid4().hex)
            os.link(source, temporary)
            os.replace(temporary, dest)
            linked.append({"asset": str(dest.relative_to(target)), "reference": str(source),
                           "sha256": expected, "bytes": source.stat().st_size})
    write_json(target / "audio_sharing.json", {"policy": "identical_SHA256_hardlinks", "assets": linked})
    return linked


def dedupe_video_assets(target, reference):
    """Share verified identical manifest files; preserve differing text and vectors."""
    root = (ROOT / "runs/exp").resolve()
    target, reference = Path(target).resolve(), Path(reference).resolve()
    if not target.is_relative_to(root) or not reference.is_relative_to(root):
        raise ValueError("Asset sharing must stay inside runs/exp.")
    linked = []
    for stage in ("_inputs/01_slides", "_inputs/02_visual", "_inputs/03_audio", "04_bundle", "05_index"):
        directory = target / stage
        if not (directory / "manifest.json").is_file():
            continue  # Interrupted stages stay incomplete; their bytes are never repaired.
        manifest = read_json(directory / "manifest.json")
        for name, expected in manifest["files"].items():
            source_stage = "04_bundle" if stage == "05_index" and name.startswith("lecture/") else stage
            source = (reference / source_stage / name.removeprefix("lecture/")).resolve()
            dest = (directory / name).resolve()
            if not source.is_relative_to(reference) or not dest.is_relative_to(target):
                raise ValueError("Asset escaped its video directory.")
            if source == dest or not source.is_file() or not dest.is_file() or os.path.samefile(source, dest):
                continue
            if source.stat().st_size != dest.stat().st_size or sha256(source) != expected:
                continue  # Differing experiment content stays independent.
            if sha256(dest) != expected:
                raise ValueError(f"Corrupt experiment asset: {dest}")
            temporary = dest.with_name(dest.name + ".link-" + uuid.uuid4().hex)
            os.link(source, temporary)
            os.replace(temporary, dest)
            linked.append({"asset": str(dest.relative_to(target)), "reference": str(source),
                           "sha256": expected, "bytes": source.stat().st_size})
    write_json(target / "asset_sharing.json", {"policy": "identical_SHA256_hardlinks", "assets": linked})
    return linked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--all-assets", action="store_true", help="Also share identical images, crops and JSON")
    args = parser.parse_args()
    root = ROOT / "runs/exp"
    assets = []
    for name in args.experiments:
        report = read_json(root / name / "rebuild_report.json")
        videos = (sorted(p.name for p in (root / name).iterdir() if (p / "_inputs").is_dir())
                  if args.all_assets else report["videos"])
        for video in videos:
            share = dedupe_video_assets if args.all_assets else dedupe_video_audio
            assets.extend(share(root / name / video, root / args.reference / video))
    print({"linked_files": len(assets), "deduplicated_bytes": sum(a["bytes"] for a in assets)})


if __name__ == "__main__":
    main()
