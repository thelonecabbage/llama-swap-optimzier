#!/usr/bin/env python3
"""Fetch a small set of pinned, permissively-licensed sample assets for
multimodal benchmark cases (real document/image/audio content that synthetic
generators cannot produce, e.g. legible document text or real speech).

This is a manual, one-time step. Benchmark runs never reach out to the
network on their own; cases that need a real asset simply read the local
file this script writes under `assets/` (gitignored) and skip gracefully if
it is missing.

Usage:
  python3 fetch_assets.py            # fetch everything in the manifest
  python3 fetch_assets.py --only cat_image
  python3 fetch_assets.py --force    # re-download even if already verified
  python3 fetch_assets.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ASSETS_DIR = Path(__file__).with_name("assets")

# Hugging Face Hub paths are pinned to an exact commit (not a moving branch)
# so a fetch today reproduces the same bytes later. Each entry's sha256 was
# recorded from that exact pinned download and is re-verified on every fetch.


@dataclass(frozen=True)
class Asset:
    name: str
    repo: str
    revision: str
    repo_path: str
    local_filename: str
    sha256: str
    license: str
    source_url: str


MANIFEST: tuple[Asset, ...] = (
    Asset(
        name="cat_image",
        repo="huggingface/documentation-images",
        revision="8273e70b0430839fe3c217613034d8d679e6cee4",
        repo_path="transformers/tasks/cat.jpg",
        local_filename="cat.jpg",
        sha256="f311fbd22e8a0562a237f0985abe185b2dba82435acebd9b452f1dbdd4387dda",
        license="CC-BY-NC-SA-4.0",
        source_url="https://huggingface.co/datasets/huggingface/documentation-images",
    ),
    Asset(
        name="docvqa_document_image",
        repo="huggingface/documentation-images",
        revision="8273e70b0430839fe3c217613034d8d679e6cee4",
        repo_path="transformers/tasks/docvqa_example.jpg",
        local_filename="docvqa_example.jpg",
        sha256="d3a02144eeca3af991d14ec3bcae4d5feb1d252708fdec5e6d5c0e96a09bf244",
        license="CC-BY-NC-SA-4.0",
        source_url="https://huggingface.co/datasets/huggingface/documentation-images",
    ),
    Asset(
        name="tts_speech_audio",
        repo="huggingface/documentation-images",
        revision="8273e70b0430839fe3c217613034d8d679e6cee4",
        repo_path="transformers/tts_example.wav",
        local_filename="tts_example.wav",
        sha256="533aed77530835e9a1f3036aeb257cfc0ac449bd2ae122a9c7ab9351ae90b561",
        license="CC-BY-NC-SA-4.0",
        source_url="https://huggingface.co/datasets/huggingface/documentation-images",
    ),
)


def asset_url(asset: Asset) -> str:
    return f"https://huggingface.co/datasets/{asset.repo}/resolve/{asset.revision}/{asset.repo_path}"


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(asset: Asset, timeout: int = 60) -> bytes:
    url = asset_url(asset)
    req = urllib.request.Request(url, headers={"User-Agent": "llama-swap-optimizer-fetch-assets/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_one(asset: Asset, *, force: bool, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / asset.local_filename

    if dest.exists() and not force:
        existing = sha256_of(dest.read_bytes())
        if existing == asset.sha256:
            print(f"OK (cached)   {asset.name} -> {dest}")
            return True
        print(f"STALE         {asset.name}: local file does not match pinned checksum, re-fetching", file=sys.stderr)

    try:
        data = download(asset)
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"FAILED        {asset.name}: {error}", file=sys.stderr)
        return False

    actual = sha256_of(data)
    if actual != asset.sha256:
        print(
            f"CHECKSUM MISMATCH {asset.name}: expected {asset.sha256}, got {actual}. Not saving.",
            file=sys.stderr,
        )
        return False

    dest.write_bytes(data)
    print(f"OK (fetched)  {asset.name} -> {dest}  [{asset.license}, pinned {asset.revision[:12]}]")
    return True


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", action="append", dest="names", help="Asset name to fetch. Repeat for multiple.")
    p.add_argument("--force", action="store_true", help="Re-download even if a verified local copy exists.")
    p.add_argument("--out-dir", type=Path, default=ASSETS_DIR)
    p.add_argument("--list", action="store_true", help="List manifest entries and exit.")
    return p


def main() -> int:
    args = make_parser().parse_args()

    if args.list:
        for asset in MANIFEST:
            print(f"{asset.name}: {asset.repo_path} ({asset.license}, source: {asset.source_url})")
        return 0

    wanted = set(args.names) if args.names else None
    if wanted:
        unknown = wanted - {a.name for a in MANIFEST}
        if unknown:
            print(f"Unknown asset name(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2

    ok = True
    for asset in MANIFEST:
        if wanted and asset.name not in wanted:
            continue
        ok = fetch_one(asset, force=args.force, out_dir=args.out_dir) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
