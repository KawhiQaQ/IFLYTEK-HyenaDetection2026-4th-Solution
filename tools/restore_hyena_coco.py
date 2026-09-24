#!/usr/bin/env python3
"""Restore the public LILA/Wild Me Hyena ID 2022 COCO archive."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


EXPECTED_ARCHIVE_SHA256 = (
    "ed590d95d6061df133ced6028d4178e226cf2190be20c7220da08795760b550e"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--competition-root", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    output_root = args.competition_root.resolve() / "hyena"
    actual_sha = sha256_file(archive)
    if actual_sha != EXPECTED_ARCHIVE_SHA256:
        raise AssertionError(f"Unexpected Hyena ID archive SHA-256: {actual_sha}")
    output_root.mkdir(parents=True, exist_ok=True)
    written = 0
    with ZipFile(archive) as source:
        if source.testzip() is not None:
            raise AssertionError("Hyena ID archive CRC validation failed")
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive member: {info.filename}")
            if not path.parts or path.parts[0] != "hyena.coco":
                continue
            if any(part.startswith("._") for part in path.parts):
                continue
            target = output_root.joinpath(*path.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as input_handle, target.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle)
            written += 1
    image_count = sum(
        1 for _ in (output_root / "hyena.coco/images/train2022").glob("*.jpg")
    )
    annotation = output_root / "hyena.coco/annotations/instances_train2022.json"
    if image_count != 3104 or not annotation.is_file():
        raise AssertionError(
            f"Unexpected restored Hyena ID geometry: images={image_count}, "
            f"annotation={annotation.is_file()}"
        )
    print(
        f"restored_files={written} source_images={image_count} root={output_root}"
    )


if __name__ == "__main__":
    main()
