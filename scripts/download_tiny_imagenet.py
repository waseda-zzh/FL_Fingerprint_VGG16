#!/usr/bin/env python3
"""Download and extract Tiny ImageNet-200 (Stanford CS231N mirror).

The default extract layout matches configs/fl_imagenet_vgg16.yaml:

  data.root: ./data/tiny-imagenet-200
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=str, default="data", help="Directory to store zip + extracted tree")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Tiny ImageNet zip URL")
    parser.add_argument("--force", action="store_true", help="Re-download even if zip exists")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = dest / "tiny-imagenet-200.zip"
    extract_root = dest / "tiny-imagenet-200"

    if extract_root.is_dir() and any(extract_root.iterdir()) and not args.force:
        print(f"Already present: {extract_root} (skip download/extract; use --force to redo)")
        return

    if args.force and zip_path.exists():
        zip_path.unlink()

    if not zip_path.exists():
        print(f"Downloading: {args.url}")
        print(f"Saving to: {zip_path}")
        urllib.request.urlretrieve(args.url, zip_path)

    if extract_root.exists() and args.force:
        shutil.rmtree(extract_root)

    print(f"Extracting: {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    if not extract_root.is_dir():
        raise RuntimeError(f"Expected extracted folder missing: {extract_root}")

    print(f"Done. Point configs/fl_imagenet_vgg16.yaml data.root to:\n  {extract_root}")


if __name__ == "__main__":
    main()
