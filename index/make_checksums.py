#!/usr/bin/env python3
"""
make_checksums.py — emit SHA-256 checksums for index artefacts so that users who
download the prebuilt FAISS indices from Zenodo can verify integrity.

Usage:
    python index/make_checksums.py index/ > index/CHECKSUMS.sha256

Verify later with:
    sha256sum -c index/CHECKSUMS.sha256
"""
import hashlib
import os
import sys

EXTS = (".faiss", ".npy", ".meta")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "index"
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if name.endswith(EXTS):
                path = os.path.join(dirpath, name)
                # sha256sum-compatible format: "<hash>  <path>"
                print(f"{sha256(path)}  {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
